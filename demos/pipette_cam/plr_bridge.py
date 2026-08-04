"""PLR bridge - turn a residual-liquid verdict into a robot action, and log it.

This is the seam where lab-cv meets PyLabRobot. The tip-cam verifier returns a
`Verdict` per well; `decide()` maps it to an `Action` the protocol can take, and
a well that is still wet raises `ResidualLiquidError` - a structured exception a
PLR sequence catches to re-aspirate, extend the air-dry, or halt, rather than
silently carrying ethanol into the elution (a classic yield killer). Every
verdict is appended to an `EventLog`: the per-well audit trail a real run
persists (well, attempt, state, residual, confidence, action).

The types themselves now live in `labcv/bridge.py` and this file re-exports
them unchanged. They moved because `labcv/dynamics.py` needs `Action` in order
to say what a commanded action does to the world, and a library must not import
from a demo - importing a script makes its argv parsing, its path inserts and
its `sys.exit` load-bearing side effects of an `import`. What stays here is what
is genuinely demo-local: the async PyLabRobot call sites below. Downstream code
that already does `from plr_bridge import ...` keeps working; the objects are
the same objects, not copies.

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

from labcv.bridge import (
    Action,
    EventLog,
    HoldForReviewError,
    Policy,
    ResidualLiquidError,
    Verdict,
    decide,
    decide_volume,
)

__all__ = [
    "Action",
    "EventLog",
    "HoldForReviewError",
    "Policy",
    "ResidualLiquidError",
    "Verdict",
    "decide",
    "decide_volume",
]
