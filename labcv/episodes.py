"""Action-conditioned rollouts: sequences where the scene changed BECAUSE of an action.

Everything lab-cv generated until now was a passive observation. `synth.tip_view`
renders one well at one residual volume; `synth.microplate` renders one plate in
one state. Nothing in the repo produced `o_0, a_0, o_1, a_1, ... , s_terminal`,
which is the only thing a world model can be scored on, because a model that is
never shown an action can never be caught ignoring one.

This module builds those sequences on top of the renderers that already exist,
and refuses to emit them when the construction cannot be verified.

    scalar_episodes         residual drawdown / dispense volume rollouts
    occluded_plate_episodes plate rollouts with a target well fully covered
    export                  frames/ + manifest.json at SCENES_FORMAT_VERSION

## Unknowability is constructed, never asserted

The most fragile thing a benchmark can do is *label* an episode unknowable. If
the label is wrong, a good model is punished for predicting and a lazy one is
rewarded for abstaining, and the abstention metric measures the labeller. So no
episode here is labelled unknowable by judgement. There are exactly two
constructions, and each is checked:

* **By independence.** The deciding quantity is drawn from a support that is
  independent of everything the renderer reads. A tip that was never fully blown
  out deposits `hidden_carryover_uL` back into the well on the commanded move.
  The frames render the well *before* that move, from its volume alone, so they
  are mathematically independent of the draw. A non-abstaining model's ceiling is
  chance over the support, and that is a proof rather than an observation.
* **By full occlusion.** An occluder covers 100% of the target well's ROI in
  every frame, checked pixel-wise against the boxes `synth.microplate` returns,
  plus a direct check that the ROI pixels are byte-identical to those of a render
  in which the target well holds the other level.

Both constructions additionally require the branches to be **separated by more
than the tolerance**, from each other and from the nominal no-hidden-effect
prediction. Without that check the "ceiling is chance" claim is false: branches
that collapse inside tolerance mean any guess hits, and the episode is knowable
while wearing an unknowable label.

## The oracle is a falsifier and only a falsifier

`identifiability_oracle` brute-forces the latent grid with frames-only access.
If it recovers the terminal latent within tolerance on an episode labelled
UNKNOWABLE, the fixture build **fails** with `UnknowableLabelViolation` and no
manifest is written. It never promotes an episode to UNKNOWABLE, and it is never
run to "confirm" a label. The asymmetry is the whole point and it is encoded in
the code rather than described in a comment:

    oracle succeeds  ->  the episode is knowable          (conclusive)
    oracle fails     ->  the oracle could not invert it   (says nothing)

A brute-force search that fails has demonstrated something about the search. An
unknowability claim needs a construction, and the construction is above.

## The file format is the entire interface

`export()` writes `frames/` plus `manifest.json` at `SCENES_FORMAT_VERSION`. That
versioned directory is how a consumer reads these fixtures. There is no package
import in either direction and there must not be one: a benchmark harness that
imports its own data generator can never be run against data it did not generate,
and `frames_sha256` per frame is what makes "did you actually run on our data" a
mechanical check instead of a promise.

The shape the manifest is written in is the consumer's, not this module's, and it
is restated field by field under "the reader's half of the file format" below.
That block exists because of a defect worth remembering: this exporter and the
harness reader both declared format version 1 while writing and reading different
shapes, so the one check whose entire job is to catch a shape mismatch passed on
every file ever written. A version number neither side ties to a structure is a
decoration. The tests here rebuild the reader's refusals from the restated block
and run them over a real export, so the two copies cannot drift in silence.

## The task vocabulary is the consumer's, and so is the transition

A pack written here used to name its own tasks - `S2-residual-drawdown`,
`S3-dispense-volume`, `S4-plate-fill` - and the consumer scores the five tasks it
declares, `S1` through `S5`. No mapping existed anywhere, so the file format that is
supposedly "the entire interface" carried, end to end, exactly zero episodes: the
consumer refused every pack this module ever wrote, with an unhandled `ValueError`
naming a task nobody had mentioned.

**The consumer's vocabulary is authoritative**, and not out of deference. A task id
is the key that selects a tolerance, a state-field layout, a horizon list and a
hidden-branch count. All four are declarations that live on the scoring side with no
counterpart here, every metric over there divides by that tolerance, and a tolerance
guessed at the scoring step is not the task's tolerance. So this module declares, per
family, which consumer task it produces, and refuses to export a family that
produces none.

That has teeth beyond the name. Producing task `S2` means producing `S2`'s *declared
transition*, because the consumer's floor row, its shuffled-action control and its
separation gate all reproduce truth by advancing that transition. `S2` with its three
continuous columns at zero is exactly `apply_residual`: REWASH multiplies the
residual by `leave`, EXTEND_DRY by `drydown`, and the retained film of a wet tip is
added to the reading. So `RESIDUAL_FAMILY` draws its per-episode constants around the
consumer's published `S2` nominal rather than around the pipette-cam demo's own
measured 0.12 and 0.35, which stay where they are and describe the demo.

`DISPENSE_FAMILY` and the plate family declare no consumer task and `export` refuses
them by name. Each refusal is a stated fact rather than an omission:

* `S3`'s declared transition adds a commanded volume delta in `p0` plus a constant
  per-verb correction of 0.15 uL against a tolerance of 0.50 uL, so a verb-only
  rollout - which is all the `Transition` protocol carries - cannot move the truth by
  even one tolerance in three steps, and a do-nothing null would pass on it.
  Producing `S3` needs a continuous commanded magnitude per step, which is a change
  to `labcv.dynamics` and not to this file.
* `S4` is a four-field plate-seating task in millimetres and degrees; the plate
  family here carries a one-field fill fraction. They share a renderer and nothing
  else, and writing a fill fraction under `S4` would score it against a seating
  tolerance.

Both generators keep running and keep their gates. What they cannot do is claim a
consumer task they do not produce.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from labcv import synth
from labcv.bridge import Action
from labcv.dynamics import PlantedTransition, apply_dispense, apply_residual

#: Bumped whenever the on-disk layout changes in a way a reader must notice.
#: A reader that does not recognise this number must refuse the pack rather than
#: guess, which is why it is the first field of the manifest.
SCENES_FORMAT_VERSION = 1

#: Per-task fraction of unknowable episodes that keeps the abstention metrics
#: about the model rather than about the base rate. Outside this band the
#: fixture build refuses; a pack of 2% unknowable episodes scores a model that
#: never abstains almost as well as a perfect one.
UNKNOWABLE_BAND = (0.05, 0.40)

#: Seed used for the noise-free reference renders that build the oracle's
#: statistic lookup. Fixed so the table is identical on every machine.
REFERENCE_SEED = 20260803

# --- the consumer's task declarations, restated -----------------------------
#
# Restated rather than imported, for the same reason as everything else in the
# reader's-half block further down: the interface is a file, and a generator that
# imports its consumer cannot be built before the consumer exists. Restating means
# the two copies can drift, so the drift is what the tests hold -
# `test_the_exported_transition_is_the_consumers_declared_one` rebuilds the
# consumer's S2 arithmetic out of this block and checks it against a real episode.

#: The one consumer task this module can produce, and every fact about it a
#: producer has to satisfy. `S2` is `residual-drawdown`: a one-field residual volume
#: in microlitres at a tolerance of 0.30 uL, scored at three horizons, whose
#: unknowable episodes are built by an independently drawn latent over three
#: branches.
CONSUMER_S2 = "S2"
CONSUMER_STATE_FIELDS: Dict[str, Tuple[str, ...]] = {CONSUMER_S2: ("residual_uL",)}
CONSUMER_UNITS: Dict[str, Tuple[str, ...]] = {CONSUMER_S2: ("uL",)}
CONSUMER_TOLERANCE: Dict[str, Tuple[float, ...]] = {CONSUMER_S2: (0.30,)}
CONSUMER_HORIZONS: Dict[str, Tuple[int, ...]] = {CONSUMER_S2: (1, 2, 3)}
CONSUMER_HIDDEN_KEYS: Dict[str, Tuple[str, ...]] = {CONSUMER_S2: ("wet_uL",)}
CONSUMER_HIDDEN_BRANCHES: Dict[str, int] = {CONSUMER_S2: 3}
#: The physically reachable range the consumer asserts S2's planted truth stays
#: inside. It checks rather than clamps, because a clamped truth stops responding to
#: the action once it saturates and a saturated truth is one an interventional gate
#: cannot tell apart from a model that ignores its actions.
CONSUMER_RESIDUAL_RANGE = (0.0, 5.0)

#: The consumer's published nominal coefficients for S2. `leave` and `drydown` are
#: the two this module draws per episode; the four `c_` constants are the
#: continuous-column half of the same transition and are held at the published value,
#: so they are restated but never varied.
CONSUMER_S2_NOMINAL: Dict[str, float] = {
    "leave": 0.72,
    "drydown": 0.80,
    "wet_uL": 0.0,
    "c_wash": 0.08,
    "c_blot": 0.02,
    "c_dry": 0.04,
    "c_quad": 0.80,
}

#: The per-column ranges a commanded S2 step draws from: wash volume in uL, blot
#: displacement in mm, dry height in mm. The consumer's own generator draws from
#: these, and matching it is not cosmetic. A pack that writes zero in all three
#: columns is very nearly action-invariant at one step, because the only thing left
#: moving the truth is the choice between a 0.72 and a 0.80 verb factor, which at a
#: typical state0 is 0.22 uL against a 0.30 uL tolerance. Measured on exactly that
#: pack: the consumer's shuffled-action control scored TolAcc@1 0.8450 against the
#: submission's 0.8050, the paired interval was [-0.0950, +0.0650], and all three
#: rows printed NOT ACTION-CONDITIONED with their headlines withheld. These are also
#: the three columns the consumer's interventional gate perturbs.
S2_WASH_uL = (0.0, 5.0)
S2_BLOT_MM = (-1.0, 1.0)
S2_DRY_MM = (0.0, 3.0)
S2_IDLE_DRY_MM = (0.0, 1.0)

#: How far a per-episode draw moves off the published nominal, as a fraction. The
#: same spread the consumer's own demonstration generator uses. A pack whose
#: coefficients were all exactly the nominal would make the consumer's analytic floor
#: an exact oracle on it, and a floor that is exact on the development pack teaches a
#: submitter only that the pack is not the task.
CONSUMER_COEFFICIENT_SPREAD = 0.12

#: The consumer's declared hidden support for S2's `wet_uL`: the retained film, in
#: microlitres, of a tip that was never fully blown out. Three branches, none of them
#: the nominal of zero retention, so the ceiling a non-abstaining model faces on
#: these episodes is one in three.
CONSUMER_S2_HIDDEN_SUPPORT: Tuple[float, ...] = (0.45, 1.10, 1.75)

#: The consumer refuses a branch pair closer than this many tolerances, and a branch
#: closer than this many tolerances to the nominal. Restated here so the writer holds
#: the gate the reader holds: a threshold only the reader has is a refusal the author
#: discovers after publishing. This module's own `_branch_check` keeps the weaker
#: one-tolerance rule for families with no consumer task, because these two numbers
#: are the consumer's and they bind exactly the episodes the consumer will score.
CONSUMER_MIN_BRANCH_SEPARATION = 2.0
CONSUMER_MIN_NOMINAL_SEPARATION = 1.0


class Knowability(str, Enum):
    KNOWABLE = "knowable"
    UNKNOWABLE = "unknowable"


class UnknowableReason(str, Enum):
    """Closed enum. An unknowable episode with no construction behind it is the
    exact failure this vocabulary exists to make impossible to express."""

    NONE = "none"
    INDEPENDENT_LATENT = "independent-latent"
    FULL_OCCLUSION = "full-occlusion"


class UnknowableLabelViolation(RuntimeError):
    """An episode labelled UNKNOWABLE that the build could not stand behind.

    Raised before any manifest is written, because a bad label that reaches a
    scoreboard is unrecoverable: every later number is about the labeller.
    """


class OcclusionIncomplete(UnknowableLabelViolation):
    """The occluder did not cover 100% of the target ROI in every frame.

    99% coverage is not a weaker version of this construction, it is a different
    one: the visible 1% is exactly where a model with enough capacity reads the
    answer, and the episode would then be scored as unknowable while being
    knowable.
    """


class FixtureDuplicationError(RuntimeError):
    """Two episodes rendered to identical frames.

    A duplicated fixture inflates n without adding evidence, and it silently
    correlates the paired bootstrap that is supposed to resample independent
    episodes.
    """


class UnknowableFractionRefused(RuntimeError):
    """A task's unknowable fraction landed outside `UNKNOWABLE_BAND`."""


class ContiguousUnknowableBlock(RuntimeError):
    """A task whose unknowable episodes can be picked out from the index alone.

    Same family of exploit as a telltale identifier. A submission that noticed
    the block would score a perfect refusal recall while predicting nothing.
    """


class TelltaleEpisodeId(RuntimeError):
    """An episode identifier that names its own knowability.

    The same exploit as a dedicated unknowable task, one layer down: a
    submission that noticed the token would collect refusal credit by string
    match, with no model involved.
    """


@dataclass
class RolloutEpisode:
    """One action-conditioned rollout.

    `prefix_frames` is what a model sees. `prefix_actions` are the actions
    between those frames, so the observed segment is self-describing. The
    `future_actions` are commanded but never observed, and `future_latents` is
    the ground truth they produced; the terminal latent is the last of them and
    is the only thing scored.
    """

    episode_id: str
    task_id: str
    latent_field: str
    units: str
    tolerance: float
    prefix_frames: List[np.ndarray]
    prefix_latents: List[float]
    prefix_actions: List[str]
    future_actions: List[str]
    #: The three continuous magnitudes commanded alongside each verb, in the
    #: consumer's column order. Carried separately from the verb because the verb is
    #: what selects a coefficient and these are what the consumer's interventional
    #: gate perturbs.
    prefix_magnitudes: List[Tuple[float, float, float]]
    future_magnitudes: List[Tuple[float, float, float]]
    future_latents: List[float]
    future_frames: List[np.ndarray] = field(default_factory=list)
    knowability: str = Knowability.KNOWABLE.value
    reason: str = UnknowableReason.NONE.value
    coefficients: Dict[str, float] = field(default_factory=dict)
    hidden: Dict[str, float] = field(default_factory=dict)
    #: Support the deciding quantity was drawn from, and the terminal each
    #: element of that support would have produced. Kept so the separation check
    #: can be re-run at export time instead of trusted from generation time.
    branch_terminals: List[float] = field(default_factory=list)
    nominal_terminal: Optional[float] = None
    #: The terminal at every future horizon under no hidden effect at all, so the
    #: separation check can be re-run at every horizon the consumer scores rather
    #: than only at the last one. An episode separated at horizon 3 and collapsed at
    #: horizon 1 is the exact defect the consumer's gate was written for, and
    #: checking one horizon would not see it.
    nominal_latents: List[float] = field(default_factory=list)
    #: The value the deciding quantity took on this episode, under the name the
    #: consumer's task declares for it. Empty on a knowable episode.
    hidden_key: str = ""
    #: Full-occlusion bookkeeping. `target_box` is the box `synth.microplate`
    #: returned for the deciding well; `alt_rois` are that box's pixels taken
    #: from a render in which the well held the other level.
    target_box: Optional[Tuple[float, float, float, float]] = None
    occluder_masks: List[np.ndarray] = field(default_factory=list)
    alt_rois: List[np.ndarray] = field(default_factory=list)

    @property
    def terminal(self) -> float:
        return float(self.future_latents[-1])

    @property
    def horizon(self) -> int:
        return len(self.future_actions)

    @property
    def n_observed(self) -> int:
        return len(self.prefix_frames)


# ---------------------------------------------------------------------------
# observation statistics - the oracle's only channel into a frame
# ---------------------------------------------------------------------------
_MASK_CACHE: Dict[int, np.ndarray] = {}


def disk_statistic(img: np.ndarray) -> float:
    """Mean intensity inside the tip-cam's well-bottom disk.

    `tip_view` jitters the residual pool's position within that disk, so a
    statistic keyed on position would move for a reason that has nothing to do
    with volume. The disk mean does not: it integrates the pool wherever it sits,
    and averages the render noise away over several thousand pixels.
    """
    px = img.shape[0]
    mask = _MASK_CACHE.get(px)
    if mask is None:
        c = px // 2
        radius = px * synth.WELL_FRAC
        yy, xx = np.mgrid[0:px, 0:px]
        mask = ((yy - c) ** 2 + (xx - c) ** 2) <= radius * radius
        _MASK_CACHE[px] = mask
    return float(img[mask].mean())


def column_statistic(img: np.ndarray) -> float:
    """Mean intensity inside the side-view well interior, which rises with fill."""
    px = img.shape[0]
    yt, yb = int(synth.COL_TOP * px), int(synth.COL_BOT * px)
    xl, xr = int(synth.COL_L * px), int(synth.COL_R * px)
    return float(img[yt:yb, xl:xr].mean())


# ---------------------------------------------------------------------------
# families
# ---------------------------------------------------------------------------
RESIDUAL_MAX_uL = 5.0
DISPENSE_MAX_uL = 20.0
DISPENSE_TARGET_uL = 10.0
#: Per-channel delivery gain a dispense episode draws from. Declared here
#: because the oracle has to sweep exactly the range the generator drew from;
#: a sweep narrower than the draw invents unknowability out of arithmetic.
DISPENSE_GAIN_RANGE = (0.55, 0.85)


def _render_tip(latent: float, rng: np.random.Generator, noise: float = 0.02) -> np.ndarray:
    img, _ = synth.tip_view(float(latent), rng=rng, max_uL=RESIDUAL_MAX_uL, noise=noise)
    return img


def _render_column(latent: float, rng: np.random.Generator, noise: float = 0.02) -> np.ndarray:
    img, _ = synth.column_view(float(latent), rng=rng, max_uL=DISPENSE_MAX_uL, noise=noise)
    return img


def _draw_residual_coefficients(seed: int) -> Dict[str, float]:
    """This episode's drawdown constants, drawn around the consumer's S2 nominal.

    Goes through `PlantedTransition` so there is one definition of what a
    per-episode residual transition is, but with the ranges the consumer publishes
    for task `S2` rather than the pipette-cam demo's own. That is the price of
    producing a declared task: the consumer's floor row, its shuffled-action control
    and its separation gate all reproduce truth by advancing `S2`'s declared
    transition, so a pack whose constants sit an order of magnitude away is a pack
    on which the declared floor misses every episode, the shuffled control is
    indistinguishable from the true one, and the null suite refuses before a single
    row is written. The demo's measured 0.12 and 0.35 are untouched and still
    describe the demo.

    The keys are the consumer's spelling. `apply_residual` wants this module's, and
    `_apply_s2` is the one place that translates.
    """
    spread = CONSUMER_COEFFICIENT_SPREAD
    planted = PlantedTransition(
        int(seed),
        leave_range=_jittered_range(CONSUMER_S2_NOMINAL["leave"], spread),
        drydown_range=_jittered_range(CONSUMER_S2_NOMINAL["drydown"], spread),
    )
    return {"leave": float(planted.rewash_leave), "drydown": float(planted.drydown)}


def _jittered_range(nominal: float, spread: float) -> Tuple[float, float]:
    """The interval a per-episode draw comes from, and the interval the oracle sweeps.

    One definition for both. A sweep narrower than the draw invents unknowability out
    of arithmetic; a sweep wider than the draw makes the falsifier weaker than it has
    to be and every episode it then fails to invert is credited as unknowable for
    free.
    """
    return (float(nominal) * (1.0 - float(spread)), float(nominal) * (1.0 + float(spread)))


#: A commanded step's three continuous magnitudes, in the consumer's column order.
#: `(0.0, 0.0, 0.0)` is a step that commands nothing continuous, which is what a
#: family with no consumer task writes.
NO_MAGNITUDE: Tuple[float, float, float] = (0.0, 0.0, 0.0)


def _apply_s2(latent: Any, action_name: str, coefficients: Dict[str, float],
              magnitude: Tuple[float, float, float] = NO_MAGNITUDE) -> Any:
    """Task S2's declared transition for one commanded step, restated.

    The consumer's `S2` multiplies a residual by `exp(-(c_wash*p0 + c_blot*|p1| +
    c_dry*|p2| + c_quad*(p0^2+p1^2+p2^2)/100))` and then, on the verb, by `leave` for
    REWASH or `drydown` for EXTEND_DRY. The verb half is exactly `apply_residual`
    under the consumer's coefficient names, so it stays in `labcv.dynamics` and this
    function only renames it. The continuous half has no counterpart in this package
    at all - nothing here ever modelled a wash volume or a blot displacement - so it
    is restated, like everything else in the reader's-half block, and
    `test_the_exported_transition_is_the_consumers_declared_one` is what holds the two
    copies together.

    The retained film is deliberately not applied here. The consumer adds `wet_uL`
    after the whole action prefix rather than inside it, so the film is present in the
    reading at every horizon and undiminished by the drawdowns that follow. A film
    carried through the loop instead is shrunk by every later step, and by the third
    horizon two branches that started a tolerance apart are not.
    """
    p0, p1, p2 = (float(v) for v in magnitude)
    exponent = (float(coefficients.get("c_wash", CONSUMER_S2_NOMINAL["c_wash"])) * p0
                + float(coefficients.get("c_blot", CONSUMER_S2_NOMINAL["c_blot"])) * abs(p1)
                + float(coefficients.get("c_dry", CONSUMER_S2_NOMINAL["c_dry"])) * abs(p2)
                + float(coefficients.get("c_quad", CONSUMER_S2_NOMINAL["c_quad"]))
                * (p0 * p0 + p1 * p1 + p2 * p2) / 100.0)
    moved = np.asarray(latent, float) * float(np.exp(-exponent))
    renamed: Dict[str, float] = {}
    # Read lazily, one key per verb. The oracle sweeps only the axes an episode's own
    # actions exercise, so an all-EXTEND_DRY episode arrives with no `leave` at all,
    # and an eager rename would turn "this coefficient is not constrained by this
    # episode" into a KeyError inside the falsifier.
    if action_name == Action.REWASH.value:
        renamed["rewash_leave"] = float(coefficients["leave"])
    elif action_name == Action.EXTEND_DRY.value:
        renamed["drydown"] = float(coefficients["drydown"])
    return apply_residual(moved, action_name, renamed)


def _apply_verb_only(latent: Any, action_name: str, coefficients: Dict[str, float],
                     magnitude: Tuple[float, float, float] = NO_MAGNITUDE) -> Any:
    """A family with no consumer task keeps its verb-only transition unchanged."""
    del magnitude
    return apply_dispense(latent, action_name, coefficients)


def _draw_s2_command(rng: np.random.Generator, verb: str) -> Tuple[float, float, float]:
    """The three continuous magnitudes a commanded S2 step carries, drawn per step.

    Tied to the verb the way the consumer's own generator ties them: a wash volume
    only on a re-aspirate, a taller dry height on an air-dry, and a blot displacement
    on every step. A magnitude drawn independently of the verb would put a five
    microlitre wash on a step that commanded nothing, which is not a command any
    protocol issues.
    """
    wash = float(rng.uniform(*S2_WASH_uL)) if verb == Action.REWASH.value else 0.0
    dry = float(rng.uniform(*(S2_DRY_MM if verb == Action.EXTEND_DRY.value
                              else S2_IDLE_DRY_MM)))
    return (wash, float(rng.uniform(*S2_BLOT_MM)), dry)


def _draw_no_command(rng: np.random.Generator, verb: str) -> Tuple[float, float, float]:
    del rng, verb
    return NO_MAGNITUDE


def _draw_dispense_coefficients(seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(int(seed))
    return {"gain": float(rng.uniform(*DISPENSE_GAIN_RANGE)),
            "target_uL": float(DISPENSE_TARGET_uL)}


@dataclass(frozen=True)
class ScalarFamily:
    """Everything the generator and the oracle must agree on for a 1-D latent.

    The generator and the oracle share `apply` and `coeff_ranges` on purpose. An
    oracle that models the generator even slightly differently is not an oracle,
    it is a second model, and every episode it fails to invert would be credited
    as unknowable for free.
    """

    task_id: str
    latent_field: str
    units: str
    tolerance: float
    max_latent: float
    commanded_verbs: Tuple[str, ...]
    start_range: Tuple[float, float]
    coeff_ranges: Tuple[Tuple[str, float, float], ...]
    fixed_coefficients: Tuple[Tuple[str, float], ...]
    axis_for_action: Tuple[Tuple[str, str], ...]
    hidden_support: Tuple[float, ...]
    render: Callable[[float, np.random.Generator], np.ndarray]
    statistic: Callable[[np.ndarray], float]
    apply: Callable[..., Any]
    draw_coefficients: Callable[[int], Dict[str, float]]
    draw_magnitude: Callable[[np.random.Generator, str], Tuple[float, float, float]]
    stat_tol: float
    grid_latent: int
    grid_coeff: int
    n_observed: int = 3
    horizon: int = 1
    #: Which consumer task a pack of these episodes may be written under, or `None`
    #: when this family produces none. `None` is not a gap to be filled in later by
    #: whoever needs an export: it is the statement that no declared task has this
    #: family's state layout, tolerance and transition, and `export` refuses on it.
    consumer_task_id: Optional[str] = None
    #: The consumer's name for the invisible latent, when there is a consumer task.
    consumer_hidden_key: str = ""

    def fixed(self) -> Dict[str, float]:
        return {k: v for k, v in self.fixed_coefficients}

    def axes_for(self, actions: Sequence[str]) -> Tuple[str, ...]:
        """Only the coefficients the episode's own actions actually reference.

        An episode of pure air-dries says nothing about the re-aspirate leave
        fraction, so sweeping it would multiply the oracle's work by the grid
        size while adding no constraint.
        """
        table = dict(self.axis_for_action)
        seen: List[str] = []
        for a in actions:
            name = table.get(a)
            if name is not None and name not in seen:
                seen.append(name)
        return tuple(seen)

    def axis_verbs(self) -> Tuple[str, ...]:
        """The verbs that move a coefficient, in declared order.

        Everything else is a declared no-op: `S2`'s PROCEED with the continuous
        columns at zero leaves the residual exactly where it was. A no-op is a real
        commanded action and belongs in a rollout - it is most of what a policy does
        - but two things have to hold around it, and both are checked rather than
        hoped for. The observed prefix has to be built from these, because a no-op
        constrains no coefficient. And the first commanded future step has to be one
        of these, because a terminal that is identical to `state0` does not respond
        to its action at all, and an interventional gate cannot tell that apart from
        a model ignoring its action argument.
        """
        table = dict(self.axis_for_action)
        return tuple(v for v in self.commanded_verbs if table.get(v))


#: The verbs a commanded step may carry on task S2, in the consumer's declared order.
#: PROCEED is an exact no-op here, because with the three continuous columns at zero
#: S2's declared transition leaves the residual where it was, and it is in the draw
#: rather than out of it: the two drawdown verbs differ by a factor of 0.72 against
#: 0.80, which at a typical state0 is 0.22 uL against a 0.30 uL tolerance, so a pack
#: built from those two alone is very nearly action-invariant at one step and the
#: consumer withholds every row on it. `_scalar_episode` keeps the no-op out of the
#: observed prefix and out of the first commanded step, which is where it would do
#: harm.
S2_COMMANDED_VERBS: Tuple[str, ...] = (Action.PROCEED.value, Action.REWASH.value,
                                       Action.EXTEND_DRY.value)

RESIDUAL_FAMILY = ScalarFamily(
    task_id=CONSUMER_S2,
    latent_field="residual_uL",
    units="uL",
    tolerance=CONSUMER_TOLERANCE[CONSUMER_S2][0],
    max_latent=RESIDUAL_MAX_uL,
    commanded_verbs=S2_COMMANDED_VERBS,
    # High in the renderable range on purpose, and bounded above by it. The
    # tip-cam's disk statistic resolves a latent to about +/-0.07 uL wherever it
    # sits, so the *relative* precision on the two coefficients is set by how much
    # liquid is in the well while they are being observed; drawing low made the
    # oracle's survivors spread past the tolerance on episodes that were perfectly
    # knowable. Bounded above because the largest retained film in the consumer's
    # declared support must not push a terminal outside the range the consumer
    # asserts for S2: the drawdown never increases the residual, so the worst case
    # is 4.9 * 0.896^5 + 1.75 = 4.6 uL, inside [0, 5]. The transition deliberately
    # does not clamp.
    start_range=(4.0, 4.9),
    coeff_ranges=(
        ("drydown",) + _jittered_range(CONSUMER_S2_NOMINAL["drydown"],
                                       CONSUMER_COEFFICIENT_SPREAD),
        ("leave",) + _jittered_range(CONSUMER_S2_NOMINAL["leave"],
                                     CONSUMER_COEFFICIENT_SPREAD),
    ),
    fixed_coefficients=(),
    axis_for_action=((Action.EXTEND_DRY.value, "drydown"),
                     (Action.REWASH.value, "leave")),
    hidden_support=CONSUMER_S2_HIDDEN_SUPPORT,
    render=_render_tip,
    statistic=disk_statistic,
    apply=_apply_s2,
    draw_coefficients=_draw_residual_coefficients,
    draw_magnitude=_draw_s2_command,
    stat_tol=0.004,
    grid_latent=401,
    grid_coeff=401,
    n_observed=3,
    horizon=len(CONSUMER_HORIZONS[CONSUMER_S2]),
    consumer_task_id=CONSUMER_S2,
    consumer_hidden_key=CONSUMER_HIDDEN_KEYS[CONSUMER_S2][0],
)

#: No consumer task. The consumer's `S3` advances a volume by a commanded delta in
#: `p0` plus a per-verb constant of 0.15 uL against a 0.50 uL tolerance, and a
#: verb-only rollout cannot move that truth by one tolerance in three steps, so a
#: do-nothing null would land inside tolerance on every episode and the consumer
#: refuses the run. This family keeps its generator, its oracle and its gates; what
#: it may not do is claim `S3`.
DISPENSE_FAMILY = ScalarFamily(
    task_id="dispense-volume",
    latent_field="volume_uL",
    units="uL",
    tolerance=0.50,
    max_latent=DISPENSE_MAX_uL,
    commanded_verbs=(Action.TOP_UP.value, Action.REWASH.value),
    start_range=(4.0, 8.0),
    coeff_ranges=(("gain",) + DISPENSE_GAIN_RANGE,),
    fixed_coefficients=(("target_uL", DISPENSE_TARGET_uL),),
    axis_for_action=((Action.TOP_UP.value, "gain"), (Action.REWASH.value, "gain")),
    hidden_support=(-1.6, 1.2, 3.0),
    render=_render_column,
    statistic=column_statistic,
    apply=_apply_verb_only,
    draw_coefficients=_draw_dispense_coefficients,
    draw_magnitude=_draw_no_command,
    stat_tol=0.004,
    grid_latent=401,
    grid_coeff=401,
)

FAMILIES: Dict[str, ScalarFamily] = {
    RESIDUAL_FAMILY.task_id: RESIDUAL_FAMILY,
    DISPENSE_FAMILY.task_id: DISPENSE_FAMILY,
}

# --- the occluded-plate family --------------------------------------------
#: Names the latent, never the construction. Every episode id in this task is
#: this string plus an index, and an id containing "occlusion" or "occluded"
#: would let the unknowable half of the task be selected by string match; the
#: consumer refuses a whole task over one such id and is right to.
#: No consumer task, and the name says so by not borrowing one. The consumer's `S4`
#: is a four-field plate-seating task in millimetres and degrees at a half-millimetre
#: tolerance; this family carries a one-field fill fraction. They share a renderer and
#: nothing else, and an episode written under `S4` would have its fill fraction
#: scored against a seating tolerance.
PLATE_TASK_ID = "plate-fill"
PLATE_ROWS, PLATE_COLS, PLATE_PX = 4, 6, 320
#: Fill levels `synth.microplate` can actually render: empty, under-filled
#: (`partial_idx`), full. Three is the whole vocabulary the renderer has, and
#: inventing a fourth would mean rendering something the repo does not draw.
PLATE_LEVELS = (0.0, 0.5, 1.0)
PLATE_TOLERANCE = 0.25
#: Two-level support for the occluded well. Three levels collapse under TOP_UP
#: and under REWASH - `{0, .5, 1}` becomes `{.5, 1, 1}` - so two branches would
#: land inside tolerance of each other and the chance ceiling would be a lie.
PLATE_HIDDEN_SUPPORT = (0.0, 1.0)
PLATE_OCCLUDER_PAD = 0.45


_STAT_TABLES: Dict[Tuple[str, float, int], Tuple[np.ndarray, np.ndarray]] = {}


def statistic_table(family: ScalarFamily, n: int = 601) -> Tuple[np.ndarray, np.ndarray]:
    """Noise-free reference renders, latent -> statistic, computed once.

    The oracle is a brute-force sweep, and re-rendering inside the sweep would
    make it minutes per episode for no extra rigour: the renderer is
    deterministic given a fixed seed, so the sweep interpolates this table
    instead. Every entry is a real render, not a model of one.

    The cache key carries the latent span and the resolution as well as the task
    id, so a family that was edited in place cannot be served a table built for
    the version before the edit - a stale table would move the oracle's answers
    without moving anything visible.
    """
    key = (family.task_id, float(family.max_latent), int(n))
    cached = _STAT_TABLES.get(key)
    if cached is not None:
        return cached
    xs = np.linspace(0.0, family.max_latent, n)
    ys = np.empty(n, float)
    for i, x in enumerate(xs):
        img = family.render(float(x), np.random.default_rng(REFERENCE_SEED), 0.0)
        ys[i] = family.statistic(img)
    _STAT_TABLES[key] = (xs, ys)
    return xs, ys


@dataclass
class OracleOutcome:
    """What a brute-force inversion attempt found. `recovered=False` is not a
    result about the episode; it is a result about the search."""

    recovered: bool
    terminal_hat: Optional[float]
    n_survivors: int
    spread: Optional[float]
    note: str = ""


def identifiability_oracle(episode: RolloutEpisode, family: ScalarFamily) -> OracleOutcome:
    """Brute-force the latent grid with frames-only access. FALSIFIER ONLY.

    Sweeps every starting latent on the grid against every coefficient the
    episode's actions reference, keeps the pairs whose rendered statistics match
    the observed prefix, propagates the survivors through the commanded future
    actions, and reports the terminal only if the survivors agree within
    tolerance.

    A `recovered=True` outcome is conclusive evidence that the episode is
    knowable. A `recovered=False` outcome is evidence about this search and
    nothing else, and callers must not read it as support for an UNKNOWABLE
    label. `refuse_mislabelled` is the only caller and it uses exactly one
    direction of this.
    """
    stats = [float(family.statistic(f)) for f in episode.prefix_frames]
    table_x, table_y = statistic_table(family)
    axes = family.axes_for(list(episode.prefix_actions) + list(episode.future_actions))
    if not axes:
        return OracleOutcome(False, None, 0, None, "no coefficient axis is exercised")

    ranges = {name: (lo, hi) for name, lo, hi in family.coeff_ranges}
    per_axis = max(2, int(round(family.grid_coeff ** (1.0 / len(axes)))))
    sweeps = [np.linspace(*ranges[a], num=per_axis) for a in axes if a in ranges]
    if len(sweeps) != len(axes):
        return OracleOutcome(False, None, 0, None, "an exercised axis has no declared range")

    latents = np.linspace(0.0, family.max_latent, family.grid_latent)
    survivors: List[np.ndarray] = []
    for combo in itertools.product(*sweeps):
        coeffs = dict(family.fixed())
        coeffs.update({a: float(v) for a, v in zip(axes, combo)})
        cur = latents.copy()
        alive = np.ones(cur.shape, bool)
        for t, observed in enumerate(stats):
            predicted = np.interp(cur, table_x, table_y)
            alive &= np.abs(predicted - observed) <= family.stat_tol
            if not alive.any():
                break
            if t < len(episode.prefix_actions):
                cur = np.asarray(family.apply(cur, episode.prefix_actions[t], coeffs,
                                              episode.prefix_magnitudes[t]), float)
        if not alive.any():
            continue
        cur = cur[alive]
        for k, action_name in enumerate(episode.future_actions):
            cur = np.asarray(family.apply(cur, action_name, coeffs,
                                          episode.future_magnitudes[k]), float)
        survivors.append(np.atleast_1d(cur))

    if not survivors:
        return OracleOutcome(False, None, 0, None, "no (latent, coefficient) pair matched")
    pool = np.concatenate(survivors)
    spread = float(pool.max() - pool.min())
    if spread > family.tolerance:
        return OracleOutcome(False, None, int(pool.size), spread,
                             "survivors disagree by more than the tolerance")
    return OracleOutcome(True, float(0.5 * (pool.max() + pool.min())), int(pool.size), spread)


# ---------------------------------------------------------------------------
# occlusion checks - pixel-wise, against the boxes synth.microplate returns
# ---------------------------------------------------------------------------
def roi_pixels(box: Sequence[float], shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """The integer pixel span a float well box covers, rounded outward.

    Rounding outward rather than to nearest is deliberate: a half-covered edge
    pixel still shows part of the well, so it belongs to the region the occluder
    has to hide.
    """
    h, w = shape
    x1 = max(0, int(np.floor(box[0])))
    y1 = max(0, int(np.floor(box[1])))
    x2 = min(w, int(np.ceil(box[2])) + 1)
    y2 = min(h, int(np.ceil(box[3])) + 1)
    return x1, y1, x2, y2


def roi_coverage(mask: np.ndarray, box: Sequence[float]) -> float:
    """Fraction of the well ROI the occluder mask covers. Anything below 1.0 is
    a different construction, not a weaker one."""
    x1, y1, x2, y2 = roi_pixels(box, mask.shape)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    patch = mask[y1:y2, x1:x2]
    return float(patch.sum()) / float(patch.size)


def occlude(img: np.ndarray, box: Sequence[float],
            pad: float = PLATE_OCCLUDER_PAD) -> Tuple[np.ndarray, np.ndarray]:
    """Lay a pipettor-shaped bar over a well, and return the frame and its mask.

    The bar's shading is a function of column index alone - never of the pixels
    underneath - so the covered region carries exactly zero bits about the well.
    A bar that blended with what it covers would leak the state it is there to
    hide, which is the subtle version of the same mistake as leaving 1% visible.
    """
    h, w = img.shape
    bw = box[2] - box[0]
    bh = box[3] - box[1]
    x1 = max(0, int(np.floor(box[0] - pad * bw)))
    y1 = max(0, int(np.floor(box[1] - pad * bh)))
    x2 = min(w, int(np.ceil(box[2] + pad * bw)) + 1)
    y2 = min(h, int(np.ceil(box[3] + pad * bh)) + 1)
    mask = np.zeros(img.shape, bool)
    mask[y1:y2, x1:x2] = True
    span = max(1, x2 - x1 - 1)
    shade = 0.045 + 0.030 * (np.arange(x2 - x1, dtype=np.float32) / span)
    out = img.copy()
    out[y1:y2, x1:x2] = shade[None, :]
    return out, mask


def roi_invariant(frame: np.ndarray, alt_roi: np.ndarray, box: Sequence[float]) -> bool:
    """True when the frame's ROI is byte-identical to the same ROI rendered with
    the target well holding the other level. This is the independence claim
    itself, checked rather than argued."""
    x1, y1, x2, y2 = roi_pixels(box, frame.shape)
    patch = frame[y1:y2, x1:x2]
    return patch.shape == alt_roi.shape and bool(np.array_equal(patch, alt_roi))


# ---------------------------------------------------------------------------
# generators
# ---------------------------------------------------------------------------
def branches_separated(terminals: Sequence[float], tolerance: float,
                       nominal: Optional[float] = None) -> bool:
    """True when every branch is more than `tolerance` from every other and from
    the nominal. The generator calls this to redraw; `_branch_check` calls the
    raising version so a construction that cannot be made separable fails the
    build instead of being quietly softened."""
    terms = list(terminals)
    if len(terms) < 2:
        return False
    for i, j in itertools.combinations(range(len(terms)), 2):
        if abs(terms[i] - terms[j]) <= tolerance:
            return False
    if nominal is not None:
        for t in terms:
            if abs(t - nominal) <= tolerance:
                return False
    return True


def _branch_check(episode: RolloutEpisode) -> None:
    """Every hidden branch must sit more than a tolerance away from every other
    branch and from the nominal no-hidden-effect prediction.

    Without this the "a non-abstaining model's ceiling is chance" sentence is
    false: if two branches land inside tolerance of each other, guessing either
    one scores on both, and the episode is knowable while labelled otherwise.
    """
    terms = list(episode.branch_terminals)
    if len(terms) < 2:
        raise UnknowableLabelViolation(
            f"{episode.episode_id}: an unknowable episode needs at least two branches; "
            f"got {len(terms)}. One branch is a prediction, not an ambiguity.")
    tol = episode.tolerance
    for i, j in itertools.combinations(range(len(terms)), 2):
        if abs(terms[i] - terms[j]) <= tol:
            raise UnknowableLabelViolation(
                f"{episode.episode_id}: hidden branches {terms[i]:.4g} and {terms[j]:.4g} are "
                f"within tolerance {tol:.4g}. Both are hit by one guess, so the chance ceiling "
                "this episode claims does not exist.")
    if episode.nominal_terminal is not None:
        for t in terms:
            if abs(t - episode.nominal_terminal) <= tol:
                raise UnknowableLabelViolation(
                    f"{episode.episode_id}: branch {t:.4g} is within tolerance {tol:.4g} of the "
                    f"nominal {episode.nominal_terminal:.4g}. A model that ignores the hidden "
                    "effect entirely would score on this episode.")


def _is_contiguous(flags: Sequence[bool]) -> bool:
    """True when the set flags form one unbroken run, a run of length one
    included. A single set flag has no arrangement that is not a run, which is
    why two is the floor rather than one."""
    idx = [i for i, f in enumerate(flags) if f]
    return bool(idx) and idx == list(range(idx[0], idx[0] + len(idx)))


def _split_flags(n: int, unknowable_frac: float, rng: np.random.Generator) -> np.ndarray:
    """Which episodes are unknowable, shuffled until the answer is not the index.

    The shuffle alone is not enough. At the batch sizes used in tests a fair
    shuffle lands the unknowable episodes in one contiguous block often enough to
    matter (two out of ten is a run about one draw in five), and a block is
    recoverable from the episode index with no model involved. Redrawing is
    rejection sampling on a structural leak rather than on an inconvenient
    result: the rejected arrangements are exactly the ones a consumer refuses.
    """
    lo, hi = UNKNOWABLE_BAND
    if not lo <= unknowable_frac <= hi:
        raise UnknowableFractionRefused(
            f"unknowable fraction {unknowable_frac:.3f} is outside {UNKNOWABLE_BAND}; "
            "outside that band the abstention metrics are about the base rate, not the model.")
    k = int(round(n * unknowable_frac))
    if k < 2 or k > n - 2:
        raise UnknowableFractionRefused(
            f"n={n} at fraction {unknowable_frac:.3f} gives {k} unknowable episode(s) of "
            f"{n}. Fewer than two on either side is one contiguous block whichever way it "
            "falls, and a block is recoverable from the episode index without a model.")
    flags = np.zeros(n, bool)
    flags[:k] = True
    for _attempt in range(64):
        rng.shuffle(flags)
        if not _is_contiguous(flags):
            return flags
    raise ContiguousUnknowableBlock(
        f"no interleaving of {k} unknowable episodes among {n} was reached in 64 draws.")


def scalar_episodes(family: ScalarFamily, n: int, seed: int,
                    unknowable_frac: float = 0.20) -> List[RolloutEpisode]:
    """Rollouts for a 1-D latent, with the unknowable ones interleaved.

    Interleaved, and with sequential ids assigned after the shuffle, so that no
    part of an episode's identity encodes its knowability. A dedicated
    "ambiguous" task or an id prefix is won by one line of `if`, and then the
    abstention column measures string matching.
    """
    rng = np.random.default_rng(seed)
    flags = _split_flags(n, unknowable_frac, rng)
    episodes: List[RolloutEpisode] = []
    for i in range(n):
        episodes.append(_draw_scalar_episode(family, i, bool(flags[i]), rng))
    return episodes


#: How many redraws a knowable episode gets before the family is declared unable to
#: produce identifiable episodes at this horizon. An absolute constant: a retry count
#: tuned against the batch being generated would be a fitted parameter deciding which
#: episodes exist.
MAX_IDENTIFIABILITY_REDRAWS = 24


def _draw_scalar_episode(family: ScalarFamily, index: int, unknowable: bool,
                         rng: np.random.Generator) -> RolloutEpisode:
    """One episode, redrawn until a KNOWABLE one is demonstrably recoverable.

    This is rejection sampling on an identifiability property, and it is the same
    argument as the rejection sampling in `_split_flags`: the rejected draws are
    exactly the ones whose label this module cannot stand behind. At the one-step
    horizon the earlier version scored at, every knowable episode was invertible and
    the question never came up. At the three horizons the consumer declares for `S2`
    the coefficient uncertainty is compounded three more times, and the observation
    model has a floor - `tip_view` resolves a residual to about +/-0.07 uL because the
    wet region grows by whole pixels - so a few percent of otherwise ordinary draws
    produce a terminal the frames do not determine. Measured before this gate: 3 of
    128 knowable episodes had oracle survivor spreads of 0.30 to 0.39 uL against a
    0.30 uL tolerance, and no amount of grid refinement or tightening of `stat_tol`
    moved them, because the limit is the render and not the search.

    Those episodes are unknowable by accident. Shipping them means a model is scored
    wrong on episodes nothing could have got right, in the accuracy column, where no
    abstention metric is looking. So `KNOWABLE` here means "the brute-force oracle
    recovered this terminal from the frames alone", which is a construction rather
    than a leftover, and it is the same standard the UNKNOWABLE half is held to.
    """
    hidden_key = family.consumer_hidden_key or "carryover_uL"
    last: Optional[OracleOutcome] = None
    for _attempt in range(MAX_IDENTIFIABILITY_REDRAWS):
        episode = _scalar_episode(family, index, unknowable, rng, hidden_key)
        if unknowable:
            return episode
        outcome = identifiability_oracle(episode, family)
        if outcome.recovered and outcome.terminal_hat is not None \
                and abs(outcome.terminal_hat - episode.terminal) <= episode.tolerance:
            return episode
        last = outcome
    raise UnconstrainedCoefficient(
        f"{family.task_id}-{index:04d}: {MAX_IDENTIFIABILITY_REDRAWS} draws in a row "
        "produced a knowable episode the brute-force oracle could not invert from the "
        f"frames (last attempt: {'no survivors' if last is None else last.note}). At that "
        "rate the family is not producing identifiable episodes at this horizon, and the "
        "repair is the observation model or the horizon rather than another draw.")


def _scalar_episode(family: ScalarFamily, i: int, unknowable: bool,
                    rng: np.random.Generator, hidden_key: str) -> RolloutEpisode:
    """One draw. `_draw_scalar_episode` decides whether to keep it."""
    ep_seed = int(rng.integers(0, 2 ** 31 - 1))
    carry = float(rng.choice(family.hidden_support)) if unknowable else 0.0
    coeffs = dict(family.fixed())
    coeffs.update(family.draw_coefficients(ep_seed))
    latent = float(rng.uniform(*family.start_range))

    # Three rules on how the verbs are drawn, each a measured failure rather than a
    # preference.
    #
    # The future is drawn per step rather than repeated. A rollout that commands one
    # verb over and over has the same action row on every episode, so permuting
    # actions across the batch permutes identical rows, the consumer's
    # shuffled-action control measures nothing, its paired interval straddles zero,
    # and the whole task prints NOT ACTION-CONDITIONED with its headline withheld.
    #
    # The future draws from every commanded verb, declared no-ops included.
    # Restricted to the two drawdown verbs the truth barely moves in one step: their
    # factors are 0.72 and 0.80, so at a state0 of 2.7 uL the difference between the
    # two commands is 0.22 uL against a 0.30 uL tolerance. Measured on exactly that
    # pack: the shuffled-action control scored TolAcc@1 0.8200 against the
    # submission's 0.8050, the paired interval was [-0.0950, +0.0650], and all three
    # rows were withheld. With the no-op back in, a command is worth 1.00 against
    # 0.72, which is 0.76 uL and well outside the tolerance.
    #
    # The prefix is built from the axis-bearing verbs only, and so is the first
    # future step. A coefficient the observed segment never exercises is one the
    # frames carry no evidence about: episodes observed through two re-aspirates and
    # then air-dried three times had oracle survivor spreads of 0.77 to 0.91 uL
    # against a 0.30 uL tolerance, which is an episode unknowable by accident while
    # wearing a knowable label. And a horizon-1 terminal reached by a no-op is a
    # truth identical to `state0`, which does not respond to its action at all.
    axis_verbs = list(family.axis_verbs())
    prefix_actions = [str(v) for v in rng.permutation(axis_verbs)][: family.n_observed - 1]
    future_actions = [str(rng.choice(axis_verbs))]
    future_actions += [str(v) for v in rng.choice(
        list(family.commanded_verbs), size=max(0, family.horizon - 1))]
    prefix_magnitudes = [family.draw_magnitude(rng, v) for v in prefix_actions]
    future_magnitudes = [family.draw_magnitude(rng, v) for v in future_actions]

    prefix_latents = [latent]
    frames = [family.render(latent, np.random.default_rng(ep_seed + 100), 0.02)]
    for t, a in enumerate(prefix_actions):
        latent = float(family.apply(latent, a, coeffs, prefix_magnitudes[t]))
        prefix_latents.append(latent)
        frames.append(family.render(latent, np.random.default_rng(ep_seed + 101 + t), 0.02))

    def roll(start: float) -> List[float]:
        out, cur = [], float(start)
        for k, a in enumerate(future_actions):
            cur = float(family.apply(cur, a, coeffs, future_magnitudes[k]))
            out.append(cur)
        return out

    # The retained film is added to the *reading*, at every horizon, rather than
    # deposited on one step and then carried through the drawdowns after it. The
    # branches are then exactly the support apart at every horizon the consumer
    # scores. Under the deposited-on-step-zero construction they are not: a
    # multiplicative drawdown shrinks the gap once per later step, so an episode
    # separated at horizon 1 collapses inside tolerance by horizon 3 and is
    # labelled unknowable while having one right answer.
    nominal_latents = roll(latent)
    future_latents = [float(v + carry) for v in nominal_latents]
    branches = [float(nominal_latents[-1] + c) for c in family.hidden_support]
    # Rendered from the nominal roll, not from the truth. The film is invisible
    # by construction, so a frame that showed it would put the deciding quantity
    # into the pixels and make the episode knowable.
    future_frames = [family.render(v, np.random.default_rng(ep_seed + 200 + k), 0.02)
                     for k, v in enumerate(nominal_latents)]

    ep = RolloutEpisode(
        episode_id=f"{family.task_id}-{i:04d}",
        task_id=family.task_id,
        latent_field=family.latent_field,
        units=family.units,
        tolerance=family.tolerance,
        prefix_frames=frames,
        prefix_latents=prefix_latents,
        prefix_actions=prefix_actions,
        future_actions=future_actions,
        prefix_magnitudes=prefix_magnitudes,
        future_magnitudes=future_magnitudes,
        future_latents=future_latents,
        future_frames=future_frames,
        knowability=(Knowability.UNKNOWABLE if unknowable else Knowability.KNOWABLE).value,
        reason=(UnknowableReason.INDEPENDENT_LATENT if unknowable
                else UnknowableReason.NONE).value,
        coefficients={k: float(v) for k, v in coeffs.items()},
        hidden={hidden_key: carry} if unknowable else {},
        branch_terminals=[float(b) for b in branches] if unknowable else [],
        nominal_terminal=float(nominal_latents[-1]) if unknowable else None,
        nominal_latents=[float(v) for v in nominal_latents],
        hidden_key=hidden_key if unknowable else "",
    )
    if unknowable:
        _branch_check(ep)
    return ep


def _plate_frame(levels: Sequence[float], rng: np.random.Generator) -> np.ndarray:
    states = [1 if lv > 0.0 else 0 for lv in levels]
    partial = [i for i, lv in enumerate(levels) if lv == 0.5]
    img, _boxes, _st = synth.microplate(PLATE_ROWS, PLATE_COLS, PLATE_PX, rng=rng,
                                        states=states, partial_idx=partial)
    return img


def _plate_boxes() -> np.ndarray:
    _img, boxes, _st = synth.microplate(PLATE_ROWS, PLATE_COLS, PLATE_PX,
                                        rng=np.random.default_rng(REFERENCE_SEED),
                                        states=[0] * (PLATE_ROWS * PLATE_COLS))
    return boxes


def _plate_apply(level: float, action_name: str) -> float:
    if action_name == Action.TOP_UP.value:
        return min(1.0, level + 0.5)
    if action_name == Action.REWASH.value:
        return max(0.0, level - 0.5)
    return level


def occluded_plate_episodes(n: int, seed: int,
                            unknowable_frac: float = 0.25) -> List[RolloutEpisode]:
    """Plate rollouts whose deciding well is either visible or fully covered.

    The occluder is drawn here rather than by `synth.microplate` because the
    check has to be against the boxes that function returns: a renderer that both
    hides a well and certifies that it hid it is checking its own homework.
    """
    rng = np.random.default_rng(seed)
    flags = _split_flags(n, unknowable_frac, rng)
    boxes = _plate_boxes()
    n_wells = PLATE_ROWS * PLATE_COLS
    interior = [i for i in range(n_wells)
                if 0 < (i // PLATE_COLS) < PLATE_ROWS - 1 and 0 < (i % PLATE_COLS) < PLATE_COLS - 1]

    episodes: List[RolloutEpisode] = []
    for i in range(n):
        unknowable = bool(flags[i])
        ep_seed = int(rng.integers(0, 2 ** 31 - 1))
        target = int(rng.choice(interior))
        box = tuple(float(v) for v in boxes[target])
        others = [float(rng.choice(PLATE_LEVELS)) for _ in range(n_wells)]
        level = float(rng.choice(PLATE_HIDDEN_SUPPORT if unknowable else PLATE_LEVELS))
        alt_level = float([v for v in PLATE_HIDDEN_SUPPORT if v != level][0]) if unknowable \
            else level

        # An action pair whose branches collapse is not generated at all. TOP_UP
        # after TOP_UP drives both hidden levels to a full well, so the two
        # branches land on the same terminal and the chance ceiling would be a
        # fiction; the same check that would refuse it at build time picks the
        # pair here instead of leaving a landmine in the pack.
        prefix_action = Action.PROCEED.value
        future_action = Action.TOP_UP.value
        branches: List[float] = []
        for _attempt in range(16):
            prefix_action = str(rng.choice([Action.PROCEED.value, Action.TOP_UP.value]))
            future_action = str(rng.choice([Action.TOP_UP.value, Action.REWASH.value]))
            branches = [_plate_apply(_plate_apply(float(c), prefix_action), future_action)
                        for c in PLATE_HIDDEN_SUPPORT]
            if not unknowable or branches_separated(branches, PLATE_TOLERANCE):
                break
        else:
            raise UnknowableLabelViolation(
                f"{PLATE_TASK_ID}-{i:04d}: no action pair keeps the hidden levels more than "
                f"{PLATE_TOLERANCE} apart at the terminal step.")

        # The terminal frame is rendered too, and is occluded on exactly the same
        # terms as the observed ones. A dev pack ships it so a pixel-replay
        # control has something to replay; shipping an un-occluded terminal frame
        # would hand the answer to anyone who read the directory.
        levels_t = [level, _plate_apply(level, prefix_action)]
        levels_t.append(_plate_apply(levels_t[-1], future_action))
        alt_t = [alt_level, _plate_apply(alt_level, prefix_action)]
        alt_t.append(_plate_apply(alt_t[-1], future_action))
        frames, masks, alt_rois = [], [], []
        for t, lv in enumerate(levels_t):
            scene = list(others)
            scene[target] = lv
            img = _plate_frame(scene, np.random.default_rng(ep_seed + 300 + t))
            if unknowable:
                alt_scene = list(others)
                alt_scene[target] = alt_t[t]
                alt_img = _plate_frame(alt_scene, np.random.default_rng(ep_seed + 300 + t))
                img, mask = occlude(img, box)
                alt_img, _ = occlude(alt_img, box)
                x1, y1, x2, y2 = roi_pixels(box, alt_img.shape)
                alt_rois.append(alt_img[y1:y2, x1:x2].copy())
                masks.append(mask)
            frames.append(img)

        terminal = levels_t[-1]
        ep = RolloutEpisode(
            episode_id=f"{PLATE_TASK_ID}-{i:04d}",
            task_id=PLATE_TASK_ID,
            latent_field="fill_level",
            units="fraction",
            tolerance=PLATE_TOLERANCE,
            prefix_frames=frames[:-1],
            prefix_latents=[float(v) for v in levels_t[:-1]],
            prefix_actions=[prefix_action],
            future_actions=[future_action],
            prefix_magnitudes=[NO_MAGNITUDE],
            future_magnitudes=[NO_MAGNITUDE],
            future_latents=[float(terminal)],
            future_frames=frames[-1:],
            knowability=(Knowability.UNKNOWABLE if unknowable else Knowability.KNOWABLE).value,
            reason=(UnknowableReason.FULL_OCCLUSION if unknowable
                    else UnknowableReason.NONE).value,
            coefficients={},
            hidden={"target_well": float(target), "hidden_level": level} if unknowable else {},
            branch_terminals=[float(b) for b in branches] if unknowable else [],
            nominal_terminal=None,
            target_box=box if unknowable else None,
            occluder_masks=masks,
            alt_rois=alt_rois,
        )
        if unknowable:
            _branch_check(ep)
        episodes.append(ep)
    return episodes


# ---------------------------------------------------------------------------
# build gates
# ---------------------------------------------------------------------------
def refuse_unknowable_fraction(episodes: Sequence[RolloutEpisode]) -> Dict[str, float]:
    """Per task, the unknowable fraction must land inside `UNKNOWABLE_BAND`."""
    lo, hi = UNKNOWABLE_BAND
    per_task: Dict[str, List[int]] = {}
    for ep in episodes:
        hit = int(ep.knowability == Knowability.UNKNOWABLE.value)
        per_task.setdefault(ep.task_id, []).append(hit)
    fractions = {}
    for task_id, hits in per_task.items():
        frac = float(sum(hits)) / float(len(hits))
        if not lo <= frac <= hi:
            raise UnknowableFractionRefused(
                f"{task_id}: unknowable fraction {frac:.3f} outside {UNKNOWABLE_BAND} over "
                f"n={len(hits)}. Outside that band the abstention metrics measure the base "
                "rate rather than the model, so no pack is written.")
        fractions[task_id] = frac
    return fractions


def refuse_mislabelled(episodes: Sequence[RolloutEpisode],
                       families: Optional[Dict[str, ScalarFamily]] = None) -> None:
    """Fail the build on any UNKNOWABLE label the construction does not support.

    Three gates, in the order they can fire:

    1. a label with `reason == NONE` is refused outright - unknowability is a
       construction, and an episode that names none of them was labelled by
       judgement;
    2. `FULL_OCCLUSION` must survive the pixel-wise coverage check on every
       frame plus the byte-identity check against the alternate render;
    3. `INDEPENDENT_LATENT` must survive the branch-separation check and must
       then defeat the brute-force oracle.

    Only step 3 consults the oracle, and only in the direction that is
    conclusive. Nothing here can promote an episode to UNKNOWABLE.
    """
    families = FAMILIES if families is None else families
    for ep in episodes:
        if ep.knowability != Knowability.UNKNOWABLE.value:
            continue
        if ep.reason == UnknowableReason.NONE.value:
            raise UnknowableLabelViolation(
                f"{ep.episode_id}: labelled unknowable with no construction. Unknowability is "
                "built, not asserted; pick a construction or drop the label.")

        if ep.reason == UnknowableReason.FULL_OCCLUSION.value:
            # "every frame" means every frame the pack can ship, terminal
            # included: an un-occluded terminal frame in a dev pack is the answer
            # sitting in the directory next to the question.
            all_frames = list(ep.prefix_frames) + list(ep.future_frames)
            if ep.target_box is None or len(ep.occluder_masks) != len(all_frames):
                raise OcclusionIncomplete(
                    f"{ep.episode_id}: full-occlusion label with no target box or no mask per "
                    "frame, so the coverage claim cannot be checked at all.")
            if len(ep.alt_rois) != len(all_frames):
                raise OcclusionIncomplete(
                    f"{ep.episode_id}: no alternate-level ROI recorded for every frame, so the "
                    "independence claim cannot be checked at all.")
            for t, mask in enumerate(ep.occluder_masks):
                cov = roi_coverage(mask, ep.target_box)
                if cov < 1.0:
                    raise OcclusionIncomplete(
                        f"{ep.episode_id}: frame {t} covers {cov * 100:.4f}% of the target ROI. "
                        "The visible remainder is exactly where the answer is read.")
            for t, (frame, alt) in enumerate(zip(all_frames, ep.alt_rois)):
                if not roi_invariant(frame, alt, ep.target_box):
                    raise OcclusionIncomplete(
                        f"{ep.episode_id}: frame {t} ROI differs from the same ROI rendered with "
                        "the other level, so the frame does carry information about it.")
            _branch_check(ep)
            continue

        if ep.reason == UnknowableReason.INDEPENDENT_LATENT.value:
            _branch_check(ep)
            family = families.get(ep.task_id)
            if family is None:
                raise UnknowableLabelViolation(
                    f"{ep.episode_id}: no family declared for task {ep.task_id!r}, so the "
                    "oracle cannot be run and the label cannot be falsified. An unfalsifiable "
                    "label is refused rather than trusted.")
            outcome = identifiability_oracle(ep, family)
            if outcome.recovered and outcome.terminal_hat is not None:
                if abs(outcome.terminal_hat - ep.terminal) <= ep.tolerance:
                    raise UnknowableLabelViolation(
                        f"{ep.episode_id}: the brute-force oracle recovered the terminal "
                        f"{outcome.terminal_hat:.4g} against truth {ep.terminal:.4g} within "
                        f"tolerance {ep.tolerance:.4g} from the frames alone. The episode is "
                        "knowable and the label is wrong; no manifest written.")
            continue

        raise UnknowableLabelViolation(
            f"{ep.episode_id}: unrecognised unknowability reason {ep.reason!r}.")


def _encode(frame: np.ndarray) -> bytes:
    """Frames go to disk as 8-bit PNG, which is what the hash is over.

    Hashing the encoded file rather than the float array is deliberate: the file
    is what a consumer downloads, so the hash they can recompute is the hash the
    manifest has to carry.
    """
    u8 = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
    ok, buf = cv2.imencode(".png", u8)
    if not ok:
        raise RuntimeError("PNG encoding failed for a rendered frame")
    return bytes(buf.tobytes())


def episode_digest(frame_hashes: Sequence[str]) -> str:
    """One hash per episode, over its per-frame hashes in order."""
    h = hashlib.sha256()
    for fh in frame_hashes:
        h.update(fh.encode("ascii"))
    return h.hexdigest()


def refuse_duplicate_fixtures(digests: Sequence[str], n_episodes: int) -> None:
    if len(set(digests)) != n_episodes:
        raise FixtureDuplicationError(
            f"{n_episodes} episodes produced {len(set(digests))} distinct frame digests. "
            "Duplicated fixtures inflate n without adding evidence and correlate every "
            "paired resample that is supposed to be independent.")


# ---------------------------------------------------------------------------
# the reader's half of the file format, restated
# ---------------------------------------------------------------------------
#
# Every name in this block is a fact about the consumer, not about this package.
# It is restated rather than imported because the interface is a file: a harness
# that imports its own generator can never be run against data it did not
# generate, and an exporter that imports the harness cannot be built without it.
# Restating means the two copies can drift, so the drift is what the tests hold -
# `test_manifest_satisfies_the_documented_reader_schema` rebuilds the consumer's
# refusals out of this block and runs them over a real export.

#: What the manifest says produced it. Non-empty on the reader's side, where it
#: is the provenance field that survives the pack being copied somewhere else.
SOURCE = "labcv.episodes"

#: Splits the consumer accepts, and what it takes them to mean: `dev` publishes
#: the per-episode transition coefficients so a submitter can develop against
#: them, `scoring` withholds them. See `export` for what is and is not withheld.
PACK_SPLITS = ("dev", "scoring")

#: Action row layout. Every row is exactly these five columns, in this order.
ACTION_FIELDS = ("p0", "p1", "p2", "p3", "gripper")

#: Gripper-column sentinel for "this action leaves the gripper alone". A real
#: width is never negative, so no valid command collides with it.
GRIPPER_UNCHANGED = -1.0

#: Which column carries the discrete verb code. Never p0, p1 or p2: those three
#: are what an interventional gate perturbs, and perturbing a discrete code asks
#: a model to extrapolate a verb that does not exist rather than asking whether
#: it reads its actions at all.
VERB_COLUMN = 3

#: The closed knowability enum the consumer accepts, spelled its way. This
#: module carries the same information in two fields and with hyphens; the
#: consumer carries it in one field and with underscores. Both spellings are
#: written out here because that difference is exactly the kind a shared
#: constant would have hidden and a version number would not have caught.
READER_KNOWABLE = "knowable"
READER_INDEPENDENT_LATENT = "independent_latent"
READER_FULL_OCCLUSION = "full_occlusion"
READER_KNOWABILITY = (READER_KNOWABLE, READER_INDEPENDENT_LATENT, READER_FULL_OCCLUSION)

_READER_KNOWABILITY: Dict[Tuple[str, str], str] = {
    (Knowability.KNOWABLE.value, UnknowableReason.NONE.value): READER_KNOWABLE,
    (Knowability.UNKNOWABLE.value,
     UnknowableReason.INDEPENDENT_LATENT.value): READER_INDEPENDENT_LATENT,
    (Knowability.UNKNOWABLE.value,
     UnknowableReason.FULL_OCCLUSION.value): READER_FULL_OCCLUSION,
}

#: Where a frame's timestamp came from. Saying `rendered` is the point: an
#: unnamed clock reads like a device clock in a table and usually is not one.
CLOCK_RENDERED = "rendered"

#: Seconds between rendered frames. These frames have no capture clock at all,
#: so the stamp is the frame index in seconds and `clock_source` says which.
FRAME_PERIOD_S = 1.0

#: Substrings an episode identifier may not contain.
FORBIDDEN_ID_TOKENS = ("unknow", "knowable", "ambiguous", "occlud", "occlusion",
                       "independent", "abstain", "ceiling")

#: The verbs each task's actions may carry, in code order. The code written into
#: `VERB_COLUMN` is the index into this tuple, and the table is published in the
#: manifest so a row can be decoded without this file.
#:
#: For a task with a consumer, the order is the consumer's declared order and not a
#: choice made here: the code is an index, so a table in a different order writes
#: every verb as some other verb's number, and the consumer's own transition reads
#: code 1 as REWASH and code 2 as EXTEND_DRY.
VERB_CODES: Dict[str, Tuple[str, ...]] = {
    CONSUMER_S2: (Action.PROCEED.value, Action.REWASH.value, Action.EXTEND_DRY.value),
    DISPENSE_FAMILY.task_id: (Action.PROCEED.value, Action.TOP_UP.value,
                              Action.REWASH.value),
    PLATE_TASK_ID: (Action.PROCEED.value, Action.TOP_UP.value, Action.REWASH.value),
}

#: The same table in the consumer's spelling, which is what the manifest publishes.
#: This module names its verbs "re-aspirate" and "extend-dry" and the consumer names
#: them "REWASH" and "EXTEND_DRY"; the two are the same verbs in the same order, and
#: writing both out is the only way that claim is checkable rather than assumed.
CONSUMER_VERBS: Dict[str, Tuple[str, ...]] = {
    CONSUMER_S2: ("PROCEED", "REWASH", "EXTEND_DRY"),
}


def published_verbs(task_id: str) -> List[str]:
    """The verb table written into the manifest, in the consumer's words where there
    is a consumer and in this module's otherwise."""
    consumer = CONSUMER_VERBS.get(task_id)
    if consumer is not None:
        return list(consumer)
    return list(VERB_CODES[task_id])


def action_row(task_id: str, verb: str,
               magnitude: Tuple[float, float, float] = NO_MAGNITUDE) -> List[float]:
    """One commanded action as the five numbers the consumer's layout expects.

    `p0`, `p1` and `p2` carry the step's continuous magnitudes and the verb code
    rides in `VERB_COLUMN`. The three continuous columns used to be hard zeros, on
    the reasoning that `labcv.dynamics` transitions are verb-driven and there was
    nothing honest to put in them. That reasoning was right about this package and
    wrong about the task: the consumer's `S2` is driven by those three columns as
    much as by the verb, so a pack of zeros is a pack of a different task. It also
    cannot answer the interventional gate, and it is very nearly action-invariant at
    one step, which the consumer detects and prints as `NOT ACTION-CONDITIONED` while
    withholding every row.
    """
    verbs = VERB_CODES.get(task_id)
    if verbs is None:
        raise ValueError(
            f"no verb table for task {task_id!r}. A code no reader can decode is a "
            f"column of noise; declare the task's verbs in VERB_CODES first.")
    if verb not in verbs:
        raise ValueError(
            f"task {task_id} commanded {verb!r}, which is not one of its declared verbs "
            f"{list(verbs)}. The code is an index into that tuple, so an undeclared verb "
            "would be written as some other verb's number.")
    row = [0.0] * len(ACTION_FIELDS)
    for column, value in enumerate(magnitude):
        row[column] = float(value)
    row[VERB_COLUMN] = float(verbs.index(verb))
    row[len(ACTION_FIELDS) - 1] = GRIPPER_UNCHANGED
    return row


def reader_knowability(episode: RolloutEpisode) -> str:
    """This module's (knowability, reason) pair as the consumer's single label.

    The consumer's enum is closed, so an unmapped pair is refused here rather
    than written into a manifest that will be rejected after publication. An
    episode labelled unknowable with reason NONE never reaches this function:
    `refuse_mislabelled` has already failed the build on it.
    """
    key = (episode.knowability, episode.reason)
    try:
        return _READER_KNOWABILITY[key]
    except KeyError:
        raise UnknowableLabelViolation(
            f"{episode.episode_id}: knowability {episode.knowability!r} with reason "
            f"{episode.reason!r} has no label in the closed enum "
            f"{list(READER_KNOWABILITY)} the consumer accepts.") from None


def refuse_recoverable_unknowable(episodes: Sequence[RolloutEpisode]) -> None:
    """Per task, the unknowable episodes must not be separable without a model.

    Two ways they can be, both refused on load by the consumer and so both
    refused here, before a pack exists to be published:

    * they sit in one contiguous block, so the episode index is the answer;
    * an identifier prefix appears on unknowable episodes and on no knowable
      one, so the prefix is the answer.

    The band check in `refuse_unknowable_fraction` is the third of the same
    family and already runs. Together they are the consumer's whole refusal set
    for this property, which is the point: a gate the writer does not have is a
    refusal the author discovers after publishing.
    """
    per_task: Dict[str, List[RolloutEpisode]] = {}
    for ep in episodes:
        per_task.setdefault(ep.task_id, []).append(ep)
    for task_id, group in per_task.items():
        flags = [ep.knowability == Knowability.UNKNOWABLE.value for ep in group]
        if _is_contiguous(flags):
            idx = [i for i, f in enumerate(flags) if f]
            raise ContiguousUnknowableBlock(
                f"{task_id}: the {len(idx)} unknowable episodes occupy positions "
                f"{idx[0]}..{idx[-1]} of {len(group)} with nothing between them. A "
                "submission that noticed would score a perfect refusal recall while "
                "predicting nothing.")
        unknowable_prefixes = {group[i].episode_id.rsplit("-", 1)[0]
                               for i, f in enumerate(flags) if f}
        knowable_prefixes = {group[i].episode_id.rsplit("-", 1)[0]
                             for i, f in enumerate(flags) if not f}
        only_unknowable = sorted(unknowable_prefixes - knowable_prefixes)
        if only_unknowable:
            raise TelltaleEpisodeId(
                f"{task_id}: identifier prefixes {only_unknowable} appear on unknowable "
                "episodes and on no knowable one, so the prefix is the answer.")


class UnconstrainedCoefficient(RuntimeError):
    """A commanded future that depends on a coefficient the observed prefix never
    exercised.

    Not a mislabelling and not a shape error: an episode like this is genuinely not
    recoverable from its frames at any model scale, and it is carrying a KNOWABLE
    label. That is the mirror image of the defect the unknowability constructions
    exist to prevent - unknowable by accident rather than by design - and it is worse
    in one respect, because nothing in the abstention columns is looking for it. It
    is refused at build time, where the fix is a redraw.
    """


class UnmappedConsumerTask(RuntimeError):
    """An episode whose task the consumer does not define.

    Raised before a byte is written, because the alternative is what shipped: a
    manifest naming a task id nothing on the scoring side has ever heard of, which
    the consumer cannot read at all. A pack that no consumer can score is not a
    fixture, and discovering that at scoring time rather than at export time is how
    the seam carried zero episodes while both halves had green suites.
    """


class ConsumerContractViolation(RuntimeError):
    """A pack that names a consumer task but does not satisfy its declaration.

    The declaration is a tolerance, a state width, a horizon list and a hidden-branch
    count, and every one of them is something the consumer will divide by, index
    into, or print a ceiling from. A producer claiming the task and missing any of
    them is worse than one that claims nothing: the pack loads, scores, and produces
    numbers about a task it is not.
    """


def refuse_unconstrained_coefficients(
        episodes: Sequence[RolloutEpisode],
        families: Optional[Dict[str, ScalarFamily]] = None) -> None:
    """Every coefficient the commanded future depends on has to be exercised by the
    observed prefix.

    A one-line check for a property that is otherwise invisible. Measured on the
    version without it: episodes observed through two re-aspirates and then air-dried
    three times had oracle survivor spreads of 0.77 to 0.91 uL against a 0.30 uL
    tolerance, because nothing in the observed segment says anything about the
    drydown fraction. Those episodes are not hard, they are unidentifiable, and they
    were labelled knowable.
    """
    families = FAMILIES if families is None else families
    for ep in episodes:
        family = families.get(ep.task_id)
        if family is None:
            continue
        observed = set(family.axes_for(ep.prefix_actions))
        needed = set(family.axes_for(ep.future_actions))
        missing = sorted(needed - observed)
        if missing:
            raise UnconstrainedCoefficient(
                f"{ep.episode_id}: the commanded future depends on {missing} and the "
                f"observed prefix {list(ep.prefix_actions)} exercises {sorted(observed)}. "
                "The frames carry no evidence about the difference, so this episode's "
                "terminal is not recoverable at any model scale while wearing a knowable "
                "label.")


def refuse_unmapped_tasks(episodes: Sequence[RolloutEpisode]) -> Dict[str, str]:
    """Every episode's task has to map to a consumer task. Returns the mapping."""
    mapping: Dict[str, str] = {}
    for ep in episodes:
        family = FAMILIES.get(ep.task_id)
        consumer = None if family is None else family.consumer_task_id
        if ep.task_id == PLATE_TASK_ID:
            consumer = None
        if consumer is None:
            raise UnmappedConsumerTask(
                f"{ep.task_id}: this family declares no consumer task, so a pack carrying "
                f"it names an id the consumer does not define and refuses on load. The "
                f"consumer tasks this module can produce are {sorted(CONSUMER_TOLERANCE)}. "
                "The generator and its gates still run; what it may not do is claim a "
                "task it does not produce.")
        mapping[ep.task_id] = consumer
    return mapping


def refuse_consumer_contract(episodes: Sequence[RolloutEpisode]) -> None:
    """The consumer's declaration for the claimed task, checked against the episodes.

    Six facts, each of which the consumer either divides by or indexes with:
    the state width, the tolerance, the horizon list, the verb table, the hidden
    latent's name, and the number of branches its support offers. A producer that
    claims a task and misses one of them writes a pack that loads and scores and is
    about a different task.
    """
    for ep in episodes:
        family = FAMILIES.get(ep.task_id)
        task = None if family is None else family.consumer_task_id
        if task is None:
            continue
        width = len(CONSUMER_STATE_FIELDS[task])
        if width != 1:
            raise ConsumerContractViolation(
                f"{ep.episode_id}: consumer task {task} declares {width} state fields and "
                "this module writes a one-field scalar latent.")
        if abs(ep.tolerance - CONSUMER_TOLERANCE[task][0]) > 1e-12:
            raise ConsumerContractViolation(
                f"{ep.episode_id}: writes tolerance {ep.tolerance:.6g} under consumer task "
                f"{task}, which declares {CONSUMER_TOLERANCE[task][0]:.6g}. Every metric "
                "over there divides by the consumer's number, so the one written here is "
                "never read and the disagreement is invisible on the scorecard.")
        horizons = tuple(range(1, ep.horizon + 1))
        if horizons != CONSUMER_HORIZONS[task]:
            raise ConsumerContractViolation(
                f"{ep.episode_id}: carries truth at horizons {list(horizons)} and consumer "
                f"task {task} is scored at {list(CONSUMER_HORIZONS[task])}. The consumer "
                "refuses to interpolate a truth it was not given, so a missing horizon is "
                "a row that never prints and an extra one is a truth nobody reads.")
        table = published_verbs(task)
        if table != list(CONSUMER_VERBS[task]):
            raise ConsumerContractViolation(
                f"{ep.episode_id}: publishes verb table {table} under consumer task "
                f"{task}, which declares {list(CONSUMER_VERBS[task])}. The code is an "
                "index into that tuple.")
        if ep.knowability != Knowability.UNKNOWABLE.value:
            continue
        if (ep.hidden_key,) != CONSUMER_HIDDEN_KEYS[task]:
            raise ConsumerContractViolation(
                f"{ep.episode_id}: names its invisible latent {ep.hidden_key!r} and consumer "
                f"task {task} declares {list(CONSUMER_HIDDEN_KEYS[task])}. A hidden latent "
                "nobody declared has no grid to be separated over, so the consumer refuses "
                "the label rather than checking it.")
        branches = len(family.hidden_support)
        if branches != CONSUMER_HIDDEN_BRANCHES[task]:
            raise ConsumerContractViolation(
                f"{ep.episode_id}: draws its latent from {branches} branches and consumer "
                f"task {task} prints a chance ceiling of one over "
                f"{CONSUMER_HIDDEN_BRANCHES[task]}. The ceiling is a published number and "
                "this would make it an overstatement.")


def refuse_consumer_separation(episodes: Sequence[RolloutEpisode]) -> None:
    """The consumer's separation thresholds, applied at every horizon it scores.

    The consumer refuses a branch pair closer than `CONSUMER_MIN_BRANCH_SEPARATION`
    tolerances and a branch closer than `CONSUMER_MIN_NOMINAL_SEPARATION` tolerances
    to the nominal, and it refuses the whole pack when any unknowable episode fails
    at any horizon. Running the same two numbers here is not duplication for its own
    sake: a threshold only the reader holds is a refusal the author discovers after
    publishing, and this module's own `_branch_check` is deliberately weaker because
    it also guards families the consumer never scores.

    Checked at every horizon rather than at the last one. An episode separated at the
    terminal step and collapsed at horizon 1 is labelled unknowable while having one
    right answer at the horizon most rows are printed at.
    """
    for ep in episodes:
        if ep.knowability != Knowability.UNKNOWABLE.value:
            continue
        family = FAMILIES.get(ep.task_id)
        task = None if family is None else family.consumer_task_id
        if task is None:
            continue
        support = family.hidden_support
        tol = CONSUMER_TOLERANCE[task][0]
        for horizon in CONSUMER_HORIZONS[task]:
            nominal = float(ep.nominal_latents[horizon - 1])
            terminals = [nominal + float(c) for c in support]
            for i, j in itertools.combinations(range(len(terminals)), 2):
                gap = abs(terminals[i] - terminals[j]) / tol
                if gap <= CONSUMER_MIN_BRANCH_SEPARATION:
                    raise UnknowableLabelViolation(
                        f"{ep.episode_id}@{horizon}: branches {terminals[i]:.4g} and "
                        f"{terminals[j]:.4g} land {gap:.4f} tolerances apart, at or under the "
                        f"{CONSUMER_MIN_BRANCH_SEPARATION:g} the consumer requires. One "
                        "prediction placed between them is within tolerance of both, so part "
                        "of the answer is available without the latent and the chance ceiling "
                        "this task publishes is an overstatement.")
            for terminal in terminals:
                gap = abs(terminal - nominal) / tol
                if gap <= CONSUMER_MIN_NOMINAL_SEPARATION:
                    raise UnknowableLabelViolation(
                        f"{ep.episode_id}@{horizon}: branch {terminal:.4g} lands {gap:.4f} "
                        f"tolerances from the nominal {nominal:.4g}, at or under the "
                        f"{CONSUMER_MIN_NOMINAL_SEPARATION:g} the consumer requires. A model "
                        "holding the published nominal and ignoring the hidden latent "
                        "entirely would score on this episode.")


def refuse_out_of_range_truth(episodes: Sequence[RolloutEpisode]) -> None:
    """S2's planted truth has to stay inside the range the consumer asserts for it.

    Checked rather than clamped, and checked here because the consumer checks it
    there. Clamping the truth would make the terminal stop responding to the action
    once it saturated, and a saturated truth is one the interventional gate cannot
    tell apart from a model that ignores its actions.
    """
    low, high = CONSUMER_RESIDUAL_RANGE
    for ep in episodes:
        family = FAMILIES.get(ep.task_id)
        if family is None or family.consumer_task_id != CONSUMER_S2:
            continue
        for value in ep.future_latents:
            if not low <= float(value) <= high:
                raise ConsumerContractViolation(
                    f"{ep.episode_id}: a planted terminal of {value:.4g} uL is outside the "
                    f"[{low:g}, {high:g}] uL range consumer task {CONSUMER_S2} asserts. The "
                    "initial draw and the hidden support have to keep the trajectory inside "
                    "it; clamping the truth is the alternative and it is worse.")


def refuse_telltale_ids(episodes: Sequence[RolloutEpisode]) -> None:
    """No episode identifier may name its own knowability.

    Checked before a pack is written, because the consumer refuses an entire
    task over one such id and the cheapest place to learn that is here.
    """
    for ep in episodes:
        lowered = str(ep.episode_id).lower()
        for token in FORBIDDEN_ID_TOKENS:
            if token in lowered:
                raise TelltaleEpisodeId(
                    f"{ep.episode_id}: the identifier contains {token!r}, so this task's "
                    "unknowable episodes can be picked out by string match with no model "
                    "involved, and the whole task is refused on load.")


def export(episodes: Sequence[RolloutEpisode], out_dir: str, pack_id: str,
           pack: str = "dev", families: Optional[Dict[str, ScalarFamily]] = None,
           seed_epoch: str = "", created_utc: str = "") -> Dict[str, Any]:
    """Write `frames/` and `manifest.json`, or write nothing at all.

    Every gate runs before the first byte is written, because a half-written pack
    with a refusal in the log is a pack somebody will score anyway.

    The manifest is written in the consumer's shape, restated above. It is the
    harness-side fixture, so it carries the planted terminal and the knowability
    label in **both** splits: the consumer requires both fields and refuses a
    manifest missing either, and a file nothing can read is not an interface.
    What a submitter is handed is the frames plus a request built from the pack,
    and the request has no field the truth could travel in. So the redaction
    boundary is the request, not this file, and moving it here was the mistake
    that made the two halves of the format unreadable to each other.

    `pack="dev"` publishes the per-episode transition coefficients and the
    invisible latent, and ships the future frames so a pixel-replay control has
    something to replay. `pack="scoring"` withholds the coefficients and the
    hidden values and ships observed frames only. Both facts are recorded under
    `withheld`, so a consumer can tell a withheld field from a missing one.

    `seed_epoch` has no usable default and an empty one is refused rather than
    filled in: it prints on every scored row and is what makes a cross-epoch
    table refusable, and an invented epoch would make two unrelated packs look
    comparable.
    """
    if pack not in PACK_SPLITS:
        raise ValueError(f"pack must be one of {list(PACK_SPLITS)}, got {pack!r}")
    if not seed_epoch:
        raise ValueError(
            "refusing to export a pack with no seed_epoch. The epoch prints on every "
            "scored row and is what makes a cross-epoch table refusable; a pack that "
            "will not name its own is unrankable against any other.")
    episodes = list(episodes)
    if not episodes:
        raise ValueError("refusing to export an empty pack; nothing was checked")

    refuse_unmapped_tasks(episodes)
    refuse_consumer_contract(episodes)
    refuse_unconstrained_coefficients(episodes, families)
    fractions = refuse_unknowable_fraction(episodes)
    refuse_mislabelled(episodes, families)
    refuse_consumer_separation(episodes)
    refuse_out_of_range_truth(episodes)
    refuse_telltale_ids(episodes)
    refuse_recoverable_unknowable(episodes)

    payloads: List[List[Tuple[str, bytes, str]]] = []
    future_payloads: List[List[Tuple[str, bytes, str]]] = []
    digests: List[str] = []
    for ep in episodes:
        rows = []
        for t, fr in enumerate(ep.prefix_frames):
            blob = _encode(fr)
            rows.append((os.path.join("frames", ep.episode_id, "t%03d.png" % t), blob,
                         hashlib.sha256(blob).hexdigest()))
        payloads.append(rows)
        digests.append(episode_digest([r[2] for r in rows]))
        # Written to disk on a dev pack and kept out of the `frames` list. Everything
        # under that key is what the consumer hands a predictor as its observed
        # prefix, so a future frame listed there is the answer arriving as an
        # observation. A dev pack still ships the files, under `future_frames`, so a
        # pixel-replay control has something to replay.
        future: List[Tuple[str, bytes, str]] = []
        if pack == "dev":
            for k, fr in enumerate(ep.future_frames):
                blob = _encode(fr)
                future.append((
                    os.path.join("future_frames", ep.episode_id, "k%03d.png" % (k + 1)),
                    blob, hashlib.sha256(blob).hexdigest()))
        future_payloads.append(future)
    refuse_duplicate_fixtures(digests, len(episodes))

    #: `hidden` is not on this list any more, and that is a correction rather than a
    #: relaxation. The consumer's separation gate reads the invisible latent to decide
    #: which grid an unknowable label is checked over, so a scoring pack that withheld
    #: it produced a pack whose labels can never be checked at all. The knowability
    #: column is in the file on both splits regardless, because the consumer requires
    #: it; the redaction boundary is the request the consumer builds from the pack,
    #: and that object has no field either the label or the latent could travel in.
    withheld = [] if pack == "dev" else ["coefficients", "future_frames"]
    #: What a scoring pack still carries, named so the file declares its own gaming
    #: channel instead of leaving a reader to discover it. These fields are here for
    #: the harness, which needs `hidden` to check the labels and `knowability` to
    #: score abstention at all. A submitter who reads the manifest directly rather
    #: than through the harness can abstain on exactly the unknowable episodes and
    #: report a perfect RefusalRecall, so a scoring split is NOT an adversarial
    #: holdout on its own. It is one only where the harness mediates access, and the
    #: honest statement of that belongs in the file rather than in a README nobody
    #: reads next to the data.
    disclosed = [] if pack == "dev" else ["knowability", "hidden", "terminal_by_horizon"]
    manifest: Dict[str, Any] = {
        # -- the fields the consumer reads, in the order it declares them.
        "format_version": SCENES_FORMAT_VERSION,
        "pack_id": pack_id,
        "seed_epoch": seed_epoch,
        "split": pack,
        "source": SOURCE,
        "created_utc": created_utc,
        #: No device serial hash, so a consumer reads this pack's basis as
        #: simulated. Every frame here is rendered, and there is no argument and
        #: no flag that promotes it: the label follows the absence of a hash.
        "device_serial_sha256": None,
        "episodes": [],
        # -- fields this package adds. A consumer that does not know them ignores
        # them; they are here so the file describes itself rather than needing
        # this module to be read alongside it.
        "format": "labcv-rollout-scenes",
        "generator": SOURCE,
        "n_episodes": len(episodes),
        "action_fields": list(ACTION_FIELDS),
        "verb_column": VERB_COLUMN,
        "action_verbs": {
            task_id: published_verbs(task_id)
            for task_id in sorted({ep.task_id for ep in episodes})
        },
        #: True where a task's declared transition reads p0, p1 and p2, which are
        #: the three columns an interventional gate perturbs. False means the rows
        #: carry zeros there and a gate run against the pack would report a model
        #: reading its actions when nothing in the actions moved.
        "continuous_action_columns": {
            task_id: bool(FAMILIES[task_id].consumer_task_id)
            for task_id in sorted({ep.task_id for ep in episodes})
            if task_id in FAMILIES
        },
        "unknowable_band": list(UNKNOWABLE_BAND),
        "unknowable_fraction": {k: round(v, 6) for k, v in sorted(fractions.items())}
        if pack == "dev" else {},
        "withheld": withheld,
        "disclosed_to_harness": disclosed,
    }
    for ep, rows, future, digest in zip(episodes, payloads, future_payloads, digests):
        #: The commanded actions only, counted from the state the manifest declares.
        #: The consumer's horizon is the number of actions from `state0`, so `state0`
        #: is the state after the observed prefix and the prefix actions are not in
        #: the scored list. Writing the whole action list from the first observed
        #: frame instead put horizons 1 and 2 of every episode on states whose frames
        #: the model had already been shown, which is scoring a prediction of
        #: something in view, and it left the consumer's declared horizons with no
        #: truth at all.
        terminals = {str(k + 1): [float(v)] for k, v in enumerate(ep.future_latents)}
        published = dict(ep.coefficients)
        if ep.hidden_key:
            published[ep.hidden_key] = float(ep.hidden.get(ep.hidden_key, 0.0))
        elif FAMILIES.get(ep.task_id) is not None \
                and FAMILIES[ep.task_id].consumer_hidden_key:
            #: A knowable episode holds the hidden key at the consumer's published
            #: nominal, and holds it explicitly. Leaving it out would make "a model
            #: holding the nominal is wrong on the unknowable episodes and right here"
            #: a sentence with nothing behind it in the file.
            published[FAMILIES[ep.task_id].consumer_hidden_key] = float(
                CONSUMER_S2_NOMINAL[FAMILIES[ep.task_id].consumer_hidden_key])
        row: Dict[str, Any] = {
            # -- what the consumer reads
            "episode_id": ep.episode_id,
            "task_id": ep.task_id,
            "state0": [float(ep.prefix_latents[-1])],
            "actions": [action_row(ep.task_id, a, m)
                        for a, m in zip(ep.future_actions, ep.future_magnitudes)],
            "terminal_by_horizon": terminals,
            "knowability": reader_knowability(ep),
            #: Observed frames only, on both splits. Everything under this key is
            #: handed to a predictor as its observed prefix, so a future frame listed
            #: here is the answer arriving as an observation.
            "frames": [{"path": rel,
                        "sha256": sha,
                        "t_capture_s": round(t * FRAME_PERIOD_S, 6),
                        "clock_source": CLOCK_RENDERED}
                       for t, (rel, _blob, sha) in enumerate(rows)],
            # -- what this package adds
            "latent_field": ep.latent_field,
            "units": ep.units,
            "tolerance": ep.tolerance,
            "n_observed": ep.n_observed,
            "horizon": ep.horizon,
            "prefix_actions": list(ep.prefix_actions),
            "future_actions": list(ep.future_actions),
            "frames_sha256": [r[2] for r in rows],
            "episode_sha256": digest,
        }
        if pack == "dev":
            row.update({
                #: This module's own two-field spelling, kept beside the
                #: consumer's one-field label so a mislabelling is visible as a
                #: disagreement between two fields rather than only as a wrong
                #: string. Withheld from a scoring pack along with the rest of
                #: the construction bookkeeping.
                "reason": ep.reason,
                "prefix_latents": [float(v) for v in ep.prefix_latents],
                "future_latents": [float(v) for v in ep.future_latents],
                "terminal": ep.terminal,
                #: In the consumer's coefficient spelling, because the consumer's
                #: separation gate reads this dict and advances its own declared
                #: transition with it. A dict in this module's spelling loads fine
                #: and silently falls back to the consumer's nominal for every key
                #: it does not recognise, which is a pack certified against
                #: coefficients it does not have.
                "coefficients": {k: float(v) for k, v in sorted(published.items())},
                "hidden": {k: float(v) for k, v in sorted(ep.hidden.items())},
                "future_frames": [{"path": rel, "sha256": sha}
                                  for rel, _blob, sha in future],
            })
        if ep.hidden_key:
            #: Written on both splits, because the consumer's separation gate reads
            #: it to decide which grid an unknowable label is checked over, and a
            #: scoring pack withholding it is a pack whose labels can never be
            #: checked at all. It is not a leak the request could carry: the
            #: redaction boundary is the request the consumer builds, which has no
            #: field either the label or the latent could travel in.
            row["hidden_keys"] = [ep.hidden_key]
            row["hidden"] = {k: float(v) for k, v in sorted(ep.hidden.items())}
        manifest["episodes"].append(row)

    os.makedirs(out_dir, exist_ok=True)
    for rows in list(payloads) + list(future_payloads):
        for rel, blob, _h in rows:
            path = os.path.join(out_dir, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as fh:
                fh.write(blob)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="ascii") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=True)
        fh.write("\n")
    return manifest
