"""The CV -> robot message: one verdict, one action, one audit row.

These types used to live in `demos/pipette_cam/plr_bridge.py`. They are here now
for one structural reason: `labcv/dynamics.py` needs `Action` to say what a
commanded action does to the world, and **a library must never import from a
demo**. A demo is allowed to be a script - to insert paths, to parse argv, to
print, to exit non-zero. The moment a library imports one, every one of those
becomes a load-bearing side effect of `import labcv.dynamics`. So the shared
vocabulary moves up and `plr_bridge.py` stays behind as a re-export shim; the
async PyLabRobot call sites, which really are demo documentation, stay with it.

Nothing here imports `pylabrobot`. That is deliberate and unchanged: the whole
point of a small message type is that either side of the seam can be swapped
without the other side being installed.

    Verdict          one well's residual-liquid readout, the message that crosses
    Action           what the protocol does next about it
    Policy           every threshold that turns the first into the second
    decide           residual   -> Action
    decide_volume    volume     -> Action
    EventLog         the per-well audit trail a real run persists

The two exceptions are refusals, not errors. `ResidualLiquidError` stops a run
before elution because ethanol carryover is invisible in the run log and only
shows up as lost yield days later. `HoldForReviewError` stops a batch because a
wrong concentration cannot be fixed by pipetting at all - a protocol that
"handled" it by dispensing more would be inventing a recovery that does not
exist.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional


class Action(Enum):
    PROCEED = "proceed"        # dry enough / on-spec -> continue
    REWASH = "re-aspirate"     # visible residual or over-volume -> pull the excess
    EXTEND_DRY = "extend-dry"  # borderline / low-confidence -> a little more air-dry
    TOP_UP = "top-up"          # under-volume -> dispense the shortfall, re-check
    HALT = "halt"              # gross residual -> stop the run for a human
    HOLD = "hold-for-review"   # off-spec chemistry -> can't be fixed by pipetting


@dataclass
class Verdict:
    """One well's residual-liquid readout - the message that crosses to PLR."""
    well: str
    residual_uL: float     # estimated leftover volume
    wet_frac: float        # raw fraction of wet-looking pixels in the well disk
    confidence: float      # certainty of the dry/residual call, [0, 1]
    state: str             # "dry" | "residual"

    def ok(self) -> bool:
        return self.state == "dry"


class ResidualLiquidError(RuntimeError):
    """A well still holds ethanol past tolerance. A PLR protocol catches this to
    re-aspirate / extend dry / halt before elution, instead of eluting wet."""

    def __init__(self, verdict: Verdict):
        self.verdict = verdict
        super().__init__(
            f"{verdict.well}: {verdict.residual_uL:.2f} uL residual "
            f"(conf {verdict.confidence:.2f}) - not safe to elute")


class HoldForReviewError(RuntimeError):
    """Off-spec chemistry a plate-reader dye QC caught. Unlike residual liquid,
    a wrong concentration cannot be fixed by pipetting, so the protocol holds the
    batch for a human instead of eluting it."""

    def __init__(self, wells: Iterable[str], target: float, tol: float):
        self.wells = list(wells)
        super().__init__(
            f"{len(self.wells)} well(s) off-spec vs {target:.2f}+/-{tol:.2f} "
            f"(dye QC) - batch held for review, not eluted")


@dataclass
class Policy:
    """Thresholds that turn a verdict into an action. Every number lives here."""
    dry_uL: float = 0.30       # <= this counts as 'dry enough' to elute
    flag_conf: float = 0.60    # below this -> ambiguous, hold for orthogonal QC
    halt_uL: float = 5.0       # gross residual -> stop the run for a human
    conf_scale: float = 0.10   # sigmoid width for the dry/residual confidence


def decide(v: Verdict, pol: Policy) -> Action:
    """Map a residual verdict to the action a PLR protocol should take next."""
    if v.residual_uL >= pol.halt_uL:
        return Action.HALT
    if v.residual_uL > pol.dry_uL:
        return Action.REWASH
    if v.confidence < pol.flag_conf:          # dry by volume, but not certain
        return Action.EXTEND_DRY
    return Action.PROCEED


def decide_volume(vol_est: float, target: float, tol: float) -> Action:
    """Map a camera height/volume readout to a corrective action.
    Under-volume -> top up the shortfall; over-volume -> re-aspirate the excess."""
    if vol_est < target - tol:
        return Action.TOP_UP
    if vol_est > target + tol:
        return Action.REWASH
    return Action.PROCEED


@dataclass
class EventLog:
    """The per-well audit trail a real run persists - the 'tracking' half."""
    rows: List[Dict[str, object]] = field(default_factory=list)

    def record(self, well: str, attempt: int, v: Verdict, action: Action) -> None:
        self.rows.append({
            "well": well, "attempt": attempt, "state": v.state,
            "residual_uL": round(v.residual_uL, 3),
            "conf": round(v.confidence, 3), "action": action.value,
        })

    def table(self, wells: Optional[Iterable[str]] = None) -> str:
        """Compact text table of the recorded events (optionally a subset)."""
        keep = None if wells is None else set(wells)
        rows = [r for r in self.rows if keep is None or r["well"] in keep]
        head = f"  {'well':<6}{'try':>4}{'state':>10}{'residual_uL':>13}{'conf':>7}{'action':>14}"
        lines = [head, "  " + "-" * (len(head) - 2)]
        for r in rows:
            lines.append(f"  {r['well']:<6}{r['attempt']:>4}{r['state']:>10}"
                         f"{r['residual_uL']:>13.3f}{r['conf']:>7.2f}{r['action']:>14}")
        return "\n".join(lines)
