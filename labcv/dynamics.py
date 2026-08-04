"""T(s, a) - the state transition, which is the one loop stage lab-cv never named.

Every other stage of the perception-action loop in this repo is already a named,
swappable module: `labcv.synth` is the observation model o = g(s), `detect()` /
`classify()` / `verify_well()` are perception, `decide()` is the policy, and
`EventLog` is the trajectory log. The transition was the exception. It lived as
two bare constants in a demo's `Config` -

    rewash_leave = 0.12    # a re-aspirate leaves ~12% of the residual
    drydown      = 0.35    # an extend-dry evaporates down to ~35%

- applied inline, three lines further down, inside the demo's own loop. That is
a world model. Naming it is the whole content of this module, and once it is
named three things become possible that were not: a learned model can be swapped
in behind the same call, the coefficients can be *drawn* per episode instead of
published as two magic numbers anyone can regress in twenty lines, and a rollout
can be generated on purpose rather than as a side effect of a demo run.

    apply_residual        the drawdown arithmetic, in one place
    apply_dispense        the volume-correction arithmetic, in one place
    Transition            the Protocol: step(latent, action) -> latent
    AnalyticTransition    the two constants above, finally named and documented
    DispenseTransition    the volume-correction transition, with the residual
                          error a real dispense leaves behind
    PlantedTransition     per-episode coefficients drawn from a distribution,
                          plus the invisible carryover that makes an episode
                          unknowable by construction
    InjectedTransition    the seam a learned transition model plugs into

**Why `AnalyticTransition`'s constants are public and `PlantedTransition`'s are
not.** Publishing `0.12` and `0.35` turns any benchmark built on them into a
twenty-line regression contest: the closed form is the answer. So a dev pack
publishes its constants (they are the documentation of the demo), and a scoring
pack draws them per episode from `PlantedTransition` and does not.

**Why `InjectedTransition` refuses instead of falling back.** An adapter with no
model injected is not a transition that happens to be bad; it is the absence of
a transition. Returning the input unchanged would make a missing model look like
a model that predicts perfect persistence, which scores respectably on smooth
dynamics and is exactly how a benchmark ends up ranking something that never ran.
So it raises, and names the signature it wants.
"""
from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Optional, Protocol, runtime_checkable

import numpy as np

from labcv.bridge import Action

#: Any latent this module transports is a liquid volume in microlitres, and no
#: physical action produces a negative one. A transition that computes one has a
#: sign error, not a very dry well, so it is clamped at the floor; the
#: plausibility ceiling below catches the other direction.
MIN_uL = 0.0

#: A well holds a few hundred microlitres at the very most. A predicted volume
#: past this is a diverged model, not a full well, and it is caught here rather
#: than silently rendered as a saturated frame that looks fine.
PLAUSIBLE_uL = 1.0e4


class TransitionAdapterError(ValueError):
    """Raised when an injected transition model breaks the adapter contract."""


@runtime_checkable
class Transition(Protocol):
    """Maps a latent state and the action commanded on it to the next latent."""

    name: str

    def step(self, latent: float, action: Action) -> float:
        """Apply one commanded action. Actions with no physical effect on this
        latent (PROCEED, HOLD) return it unchanged, which is a claim about the
        world and not a fallback for an unrecognised action."""
        ...


def _finite(value: float, name: str) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise FloatingPointError(f"{name} produced a non-finite latent: {out!r}")
    if abs(out) > PLAUSIBLE_uL:
        raise FloatingPointError(
            f"{name} left the plausible range for a well: {out:.4g} uL "
            f"(|latent| > {PLAUSIBLE_uL:g}). This is divergence, not a full well.")
    return out


def apply_residual(latent: Any, action_name: str, coefficients: Dict[str, float]) -> Any:
    """Residual drawdown after one commanded action, keyed by `Action.value`.

    This arithmetic is written once and called from three places - the analytic
    transition, the per-episode planted transition, and the identifiability
    oracle - because an oracle that models the generator slightly differently
    from the generator is not an oracle, it is a second model, and every episode
    it fails to invert would be silently credited as unknowable.

    `latent` may be a scalar or a whole array of candidate latents; the oracle
    sweeps thousands of them at a time and a scalar-only version would have made
    it minutes per episode, which is how a brute-force check quietly turns into
    a check nobody runs.
    """
    if action_name == Action.REWASH.value:
        return np.maximum(MIN_uL, latent * float(coefficients["rewash_leave"]))
    if action_name == Action.EXTEND_DRY.value:
        return np.maximum(MIN_uL, latent * float(coefficients["drydown"]))
    return np.maximum(MIN_uL, latent)


def apply_dispense(latent: Any, action_name: str, coefficients: Dict[str, float]) -> Any:
    """Volume after one corrective move commanded against the *true* volume.

    Deliberately not the same model as :class:`DispenseTransition`, which
    commands against what the camera measured and adds delivery noise. This one
    is the clean per-channel gain a fixture generator and its oracle share, so
    the two never drift; the demo's version is the one that carries the readout
    error, because that is the error a demo is there to expose. Scalars and
    arrays both, for the same reason as above.
    """
    if action_name in (Action.TOP_UP.value, Action.REWASH.value):
        gain = float(coefficients["gain"])
        target = float(coefficients["target_uL"])
        return np.maximum(MIN_uL, latent + gain * (target - latent))
    return np.maximum(MIN_uL, latent)


class AnalyticTransition:
    """Residual drawdown after a wash: the two constants from the pipette-cam demo.

    A re-aspirate cannot pull a well perfectly dry - surface tension holds a film
    on the plastic - so it leaves a fraction `rewash_leave` of what was there. An
    extended air-dry evaporates rather than removes, so it multiplies by `drydown`
    instead. Both are proportional, which is why repeated re-aspirates converge
    geometrically and why the demo needs at most three attempts on a wet well.

    This is the near-oracle for any pack generated from it. It exists as a
    control - the thing a submitted model has to beat, and the thing whose
    failure means the harness is broken rather than the model. It is not a
    scoreable entry.
    """

    #: A re-aspirate leaves ~12% of the residual behind (film on the plastic).
    REWASH_LEAVE = 0.12
    #: An extended air-dry evaporates down to ~35% of the residual.
    DRYDOWN = 0.35

    def __init__(self, rewash_leave: float = REWASH_LEAVE, drydown: float = DRYDOWN,
                 name: str = "analytic"):
        self.rewash_leave = float(rewash_leave)
        self.drydown = float(drydown)
        self.name = name

    @property
    def coefficients(self) -> Dict[str, float]:
        return {"rewash_leave": self.rewash_leave, "drydown": self.drydown}

    def step(self, latent: float, action: Action) -> float:
        return _finite(apply_residual(float(latent), action.value, self.coefficients), self.name)


class DispenseTransition:
    """Volume correction: what a top-up or a re-aspirate actually lands on.

    The transition this replaces asserted that a corrective dispense lands
    exactly on target with zero error - `vol_final = target`. That is not a
    measurement, it is the strongest possible claim about a pipetting channel,
    made silently in one line, and it makes the closed loop look perfect for a
    reason that has nothing to do with the camera: a well is declared in spec
    because the code assigned it the target value.

    What actually happens is three-layered and every layer is here. The robot
    commands the shortfall it *believes* in - `target - measured` - so the
    camera's own readout error is inherited by the correction. The channel
    delivers a `gain` fraction of what it was commanded. The delivery itself is
    noisy. The result lands near target, usually inside tolerance, which is the
    honest version of the same PASS.
    """

    #: Fraction of the commanded volume a channel actually delivers.
    GAIN = 0.94
    #: 1-sigma dispense noise, in uL, on a single corrective move.
    NOISE_uL = 0.08

    def __init__(self, target_uL: float, gain: float = GAIN, noise_uL: float = NOISE_uL,
                 rng: Optional[np.random.Generator] = None, name: str = "dispense"):
        self.target_uL = float(target_uL)
        self.gain = float(gain)
        self.noise_uL = float(noise_uL)
        self.rng = np.random.default_rng(0) if rng is None else rng
        self.name = name

    @property
    def coefficients(self) -> Dict[str, float]:
        return {"gain": self.gain, "noise_uL": self.noise_uL, "target_uL": self.target_uL}

    def step(self, latent: float, action: Action, measured: Optional[float] = None) -> float:
        """`measured` is what the camera read, which is what the robot commands
        against. Passing None means the robot is commanding against the truth,
        which no camera-in-the-loop ever does; it is offered only so the
        transition is usable without a readout, never as the default story."""
        if action not in (Action.TOP_UP, Action.REWASH):
            return _finite(latent, self.name)
        belief = float(latent if measured is None else measured)
        commanded = self.target_uL - belief
        delivered = self.gain * commanded + float(self.rng.normal(0.0, self.noise_uL))
        return _finite(max(MIN_uL, latent + delivered), self.name)


class PlantedTransition:
    """One episode's coefficients, drawn rather than published.

    Construct one per episode with that episode's seed. The draw happens once, at
    construction, so `.coefficients` is the planted ground truth for exactly this
    episode: it can be written into a dev manifest and withheld from a scoring
    one.

    `hidden_carryover_uL` is the piece that matters, and it is not a coefficient.
    It is liquid a tip that was never fully blown out deposits back into the well
    on one commanded move. It is drawn by the caller from a support that is
    independent of everything the renderer reads, so a frame rendered from the
    residual volume of the well *before* the move carries no information about
    it. The post-action latent is then not identifiable from the frames at any
    model scale. That is the difference between an episode that is hard and an
    episode that is unknowable, and it is a property of the construction rather
    than a label somebody applied.

    It is additive rather than multiplicative on purpose. A hidden gain scales
    with a residual that is already collapsing geometrically, so on the second or
    third drawdown every branch of the hidden draw lands inside tolerance of
    every other and of the nominal - the episode reads as unknowable while being,
    numerically, perfectly predictable. An additive carryover keeps the branches
    a fixed distance apart no matter how dry the well got.
    """

    #: Range the per-episode re-aspirate leave fraction is drawn from.
    LEAVE_RANGE = (0.05, 0.25)
    #: Range the per-episode air-dry drawdown fraction is drawn from.
    DRYDOWN_RANGE = (0.20, 0.55)

    def __init__(self, seed: int, leave_range=LEAVE_RANGE, drydown_range=DRYDOWN_RANGE,
                 hidden_carryover_uL: float = 0.0, name: str = "planted"):
        rng = np.random.default_rng(seed)
        self.seed = int(seed)
        self.rewash_leave = float(rng.uniform(*leave_range))
        self.drydown = float(rng.uniform(*drydown_range))
        self.hidden_carryover_uL = float(hidden_carryover_uL)
        self.name = name

    @property
    def coefficients(self) -> Dict[str, float]:
        """The publishable-for-dev half. `hidden_carryover_uL` is deliberately
        not in here; it is recorded separately as an invisible latent, and it is
        never scoreable, because scoring it would be scoring a coin flip."""
        return {"rewash_leave": self.rewash_leave, "drydown": self.drydown}

    def step(self, latent: float, action: Action, apply_hidden: bool = False) -> float:
        out = float(apply_residual(float(latent), action.value, self.coefficients))
        if apply_hidden:
            out = max(MIN_uL, out + self.hidden_carryover_uL)
        return _finite(out, self.name)


class InjectedTransition:
    """The seam a learned transition model plugs into, and nothing more.

    A checkpoint is metadata, not executable integration. The caller owns loading
    whatever model it trusts and supplies a synchronous callable that takes
    `(latent_uL, action_name)` and returns the next latent in the same units.
    This class refuses everything it cannot verify:

    * no callable injected -> `NotImplementedError` naming the signature, because
      an installed package is not an adapter;
    * an awaitable return -> the coroutine is closed and `TransitionAdapterError`
      is raised, because a silently un-awaited coroutine evaluates truthy and
      would sail through every downstream check as a plausible number;
    * a non-numeric, non-finite, or implausible return -> refused, so a diverged
      model produces a refusal rather than a very confident number.
    """

    def __init__(self, predictor: Optional[Callable[[float, str], Any]] = None,
                 checkpoint: Optional[Any] = None, name: str = "injected"):
        if predictor is not None and not callable(predictor):
            raise TypeError("predictor must be callable or None")
        self.predictor = predictor
        self.checkpoint = checkpoint
        self.name = name

    def step(self, latent: float, action: Action) -> float:
        if self.predictor is None:
            note = " Checkpoint metadata was recorded." if self.checkpoint is not None else ""
            raise NotImplementedError(
                f"the {self.name} transition adapter has no predictor callable.{note} "
                "A checkpoint or an installed package alone is not an adapter: inject a "
                "synchronous predictor(latent_uL, action_name) -> next_latent_uL callable, "
                "after mapping the checkpoint's state, action and units.")
        raw = self.predictor(float(latent), action.value)
        if inspect.isawaitable(raw):
            close = getattr(raw, "close", None)
            if callable(close):
                close()
            raise TransitionAdapterError(
                f"{self.name} predictor returned an awaitable; transitions must be synchronous")
        try:
            out = float(raw)
        except (TypeError, ValueError) as exc:
            raise TransitionAdapterError(
                f"{self.name} predictor output must be a single numeric latent in uL") from exc
        return _finite(out, self.name)
