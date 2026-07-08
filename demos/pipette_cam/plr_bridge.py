"""PLR bridge - turn a residual-liquid verdict into a robot action, and log it.

This is the seam where lab-cv meets PyLabRobot. The tip-cam verifier returns a
`Verdict` per well; `decide()` maps it to an `Action` the protocol can take, and
a well that is still wet raises `ResidualLiquidError` - a structured exception a
PLR sequence catches to re-aspirate, extend the air-dry, or halt, rather than
silently carrying ethanol into the elution (a classic yield killer). Every
verdict is appended to an `EventLog`: the per-well audit trail a real run
persists (well, attempt, state, residual, confidence, action).

There is no `pylabrobot` import here - the real call sites are the documented
seam below, guarded and optional, exactly like the RF-DETR / SAM2 seams
elsewhere in this repo. The verdict object is the whole point: one small message
crosses the CV -> robot boundary, so either side can be swapped independently.

    >>> SWAP SEAM: inside an async PyLabRobot protocol <<<
    # ... after removing the 80% ethanol wash over the magnet:
    #     await lh.aspirate(wells, vols=[cfg.super_uL], ...)   # pull supernatant
    #     frame = await tip_cam.grab(channel=ch)               # borescope on channel
    #     v   = verify_well(frame, well.name, cal, pol)        # demos/pipette_cam/verify
    #     log.record(well.name, attempt, v, decide(v, pol))
    #     act = decide(v, pol)
    #     if act is Action.REWASH:
    #         await lh.aspirate([well], vols=[pol.rewash_uL], ...)  # pull the dregs
    #         continue                                         # re-image, re-check
    #     elif act is Action.EXTEND_DRY:
    #         await asyncio.sleep(pol.extra_dry_s)             # let it evaporate
    #         continue
    #     elif act is Action.HALT:
    #         raise ResidualLiquidError(v)                     # stop before elution
    # PLR's own volume tracker stays authoritative for what was moved; the cam
    # is the independent check that the move actually left the well dry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


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

    def __init__(self, wells, target: float, tol: float):
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
    rows: list = field(default_factory=list)

    def record(self, well: str, attempt: int, v: Verdict, action: Action) -> None:
        self.rows.append({
            "well": well, "attempt": attempt, "state": v.state,
            "residual_uL": round(v.residual_uL, 3),
            "conf": round(v.confidence, 3), "action": action.value,
        })

    def table(self, wells=None) -> str:
        """Compact text table of the recorded events (optionally a subset)."""
        rows = [r for r in self.rows if wells is None or r["well"] in wells]
        head = f"  {'well':<6}{'try':>4}{'state':>10}{'residual_uL':>13}{'conf':>7}{'action':>14}"
        lines = [head, "  " + "-" * (len(head) - 2)]
        for r in rows:
            lines.append(f"  {r['well']:<6}{r['attempt']:>4}{r['state']:>10}"
                         f"{r['residual_uL']:>13.3f}{r['conf']:>7.2f}{r['action']:>14}")
        return "\n".join(lines)
