#!/usr/bin/env python3
"""Pipette-cam - "was the ethanol fully removed?" - the closed-loop half of
video-verified execution: a tip-mounted camera checks each well after supernatant
removal, and a residual verdict is fed back to PyLabRobot to RE-ACT before the
plate is eluted.

    python3 demos/pipette_cam/run.py              # camera catches + clears every wet well
    python3 demos/pipette_cam/run.py --glare      # rim scratches glint -> baseline false-alarms
    python3 demos/pipette_cam/run.py --residual 8 # more wells left wet

Residual ethanol after a bead wash is invisible to the run log and silently kills
downstream yield - the canonical "the motions look right, the chemistry's off."
We plant a known leftover volume in each well, image it blind through the tip cam,
verify dry/residual with a confidence, and - this is the new axis versus the other
demos - close the loop: each verdict becomes a PLR `Action` (re-aspirate / extend
dry / halt), the well is re-imaged, and we score both the catch AND the recovery.
Every number below is printed by this run on synthetic frames (classical baseline,
no models installed). See plr_bridge.py for the async PyLabRobot call sites.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from labcv import synth, viz                              # noqa: E402
import verify                                             # noqa: E402
from plr_bridge import Action, EventLog, Policy, decide   # noqa: E402


@dataclass
class Config:
    n_wells: int = 24
    n_residual: int = 5
    seed: int = 7
    max_uL: float = 5.0          # volume that fills the readable well bottom
    cal_uL: float = 3.0          # known reference the tip cam is calibrated on
    dry_uL: float = 0.30         # <= this is 'dry enough' to elute
    max_attempts: int = 4        # camera-guided re-tries before giving up
    rewash_leave: float = 0.12   # a re-aspirate leaves ~12% of the residual
    drydown: float = 0.35        # an extend-dry evaporates down to ~35%
    glare: bool = False
    out: str = "output/pipette_cam_qc.png"


def run(cfg: Config) -> bool:
    rng = np.random.default_rng(cfg.seed)
    pol = Policy(dry_uL=cfg.dry_uL)
    n = cfg.n_wells

    # one-point cam calibration on a known reference droplet (real rigs do this too)
    cal = verify.calibrate(cfg.cal_uL, synth, max_uL=cfg.max_uL,
                           rng=np.random.default_rng(cfg.seed + 99))

    # plant: most wells dry, a hidden subset still holding ethanol = ground truth
    residual0 = np.zeros(n)
    wet_idx = np.sort(rng.choice(n, cfg.n_residual, replace=False))
    residual0[wet_idx] = rng.uniform(0.6, 3.4, cfg.n_residual)
    wells = [f"W{i + 1:02d}" for i in range(n)]
    truth_wet = residual0 > pol.dry_uL

    log = EventLog()
    first_v = [None] * n
    first_frame = [None] * n
    final_res = residual0.copy()
    attempts = np.zeros(n, int)
    proceeded = np.zeros(n, bool)

    # ---- closed loop: verify -> PLR action -> re-image, per well ---------------
    for i in range(n):
        res = float(residual0[i])
        for attempt in range(1, cfg.max_attempts + 1):
            frame, _ = synth.tip_view(res, rng=rng, max_uL=cfg.max_uL, glare=cfg.glare)
            v = verify.verify_well(frame, wells[i], cal, pol)
            act = decide(v, pol)
            log.record(wells[i], attempt, v, act)
            attempts[i] = attempt
            if attempt == 1:
                first_v[i], first_frame[i] = v, frame
            if act is Action.PROCEED:
                proceeded[i] = True
                break
            if act is Action.REWASH:
                res *= cfg.rewash_leave        # pull the dregs
            elif act is Action.EXTEND_DRY:
                res *= cfg.drydown             # let it evaporate
            elif act is Action.HALT:
                break                          # gross residual -> stop for a human
        final_res[i] = res

    # ---- score: first-pass catch, then closed-loop recovery -------------------
    detected = np.array([v.state == "residual" for v in first_v])
    caught = int((detected & truth_wet).sum())
    n_wet = int(truth_wet.sum())
    false_alarm = int((detected & ~truth_wet).sum())
    recall = caught / max(n_wet, 1)
    vol_mae = float(np.mean([abs(first_v[i].residual_uL - residual0[i])
                             for i in np.where(truth_wet)[0]])) if n_wet else 0.0
    rewashes = sum(1 for r in log.rows if r["action"] == Action.REWASH.value)
    cleared = int((final_res <= pol.dry_uL).sum())
    max_tries = int(attempts[truth_wet].max()) if n_wet else 0
    converged = bool(proceeded.all())

    print("\nPIPETTE-CAM - was the ethanol fully removed?  (tip-cam verify -> PLR closed loop)")
    print(f"  wells={n}   tip-cam calibrated on {cfg.cal_uL:.1f} uL reference "
          f"(cal={cal:.4f} wet_frac/uL)")
    print(f"  planted wet (residual > {pol.dry_uL:.2f} uL) = {n_wet}"
          f"   glare confounder = {'ON' if cfg.glare else 'off'}\n")
    print("  first pass - camera vs plant:")
    print(f"    residual wells caught      {caught}/{n_wet}   (recall {recall:.2f})")
    print(f"    false alarms on dry wells  {false_alarm}/{n - n_wet}")
    print(f"    residual-volume MAE        {vol_mae:.2f} uL\n")
    print("  closed loop - verdict -> PLR action -> re-image:")
    print(f"    re-aspirates issued        {rewashes}   (<= {max_tries} attempts on a wet well)")
    print(f"    wells cleared to dry       {cleared}/{n}")
    print(f"    WITHOUT the camera, {n_wet} wells elute wet -> silent ethanol carryover.\n")
    print("  audit trail - the per-well events a PLR run persists (wet wells):")
    print(log.table(wells=[wells[i] for i in np.where(truth_wet)[0]]) + "\n")

    _qc_panel(cfg, wells, residual0, final_res, truth_wet, wet_idx, first_v, first_frame, pol)

    ok = converged and recall >= 1.0 and cleared == n
    if ok:
        print(f"  PASS: every wet well caught (recall {recall:.2f}) and driven dry "
              f"({cleared}/{n}); 0 carried into elution.\n")
    else:
        print(f"  FAIL: recall {recall:.2f}, cleared {cleared}/{n}, "
              f"converged={converged} - the baseline could not close the loop.")
        if cfg.glare:
            print("  ^ rim scratches glint like residual, so the wet-pixel reader keeps")
            print("    re-aspirating already-dry wells. THIS is where SAM2 film")
            print("    segmentation (no pool -> don't fire) or a VLM 'is it dry?' earns")
            print("    its keep - same verify_well seam, same PLR bridge, unchanged.\n")
    return ok


def _qc_panel(cfg, wells, residual0, final_res, truth_wet, wet_idx, first_v, first_frame, pol):
    """One figure: a strip of tip-cam frames with verdicts + the before/after plate."""
    Rect = viz.plt.Rectangle
    show_idx = list(wet_idx[:5])
    for i in range(cfg.n_wells):                       # pad to 6 with a dry well
        if len(show_idx) >= 6:
            break
        if i not in show_idx:
            show_idx.append(i)
    show_idx = show_idx[:6]

    mosaic = [["m0", "m1", "m2", "bar", "bar"],
              ["m3", "m4", "m5", "bar", "bar"]]
    fig, ax = viz.plt.subplot_mosaic(mosaic, figsize=(11.2, 4.7))

    for k, i in enumerate(show_idx):
        a = ax[f"m{k}"]
        v = first_v[i]
        viz.show(a, first_frame[i], title=f"{wells[i]}  {v.state} {v.residual_uL:.1f}uL")
        col = viz.S.OUTLINE["pink"] if v.state == "residual" else viz.S.OUTLINE["green"]
        px = first_frame[i].shape[0]
        a.add_patch(Rect((1, 1), px - 3, px - 3, fill=False, edgecolor=col, linewidth=3))

    b = ax["bar"]
    grid = np.vstack([residual0, final_res])
    im = b.imshow(grid, cmap=viz.S.cmap("blue"), vmin=0, vmax=cfg.max_uL, aspect="auto")
    b.set_title("residual per well - planted (top) vs after the closed loop (bottom)")
    b.set_yticks([0, 1], ["planted", "after loop"])
    b.set_xticks(range(0, cfg.n_wells, 4), [wells[j] for j in range(0, cfg.n_wells, 4)],
                 fontsize=7)
    b.grid(False)
    for i in np.where(truth_wet)[0]:                   # outline the truly-wet wells
        b.add_patch(Rect((i - 0.5, -0.5), 1, 1, fill=False,
                         edgecolor=viz.S.OUTLINE["pink"], linewidth=2.2))
    fig.colorbar(im, ax=b, fraction=0.046, pad=0.03, label="residual (uL)")
    viz.save(fig, os.path.join(ROOT, cfg.out))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--residual", type=int, default=5, help="wells left holding ethanol")
    p.add_argument("--glare", action="store_true",
                   help="rim scratches / condensation glint -> baseline false alarms")
    a = p.parse_args()
    ok = run(Config(seed=a.seed, n_residual=a.residual, glare=a.glare))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
