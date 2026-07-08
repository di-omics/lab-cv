#!/usr/bin/env python3
"""Integrated QC checkpoints - tip-cam liquid-height CV + sampled plate-reader dye
QC, composed into one GO/HOLD gate and fed back to PyLabRobot.

    python3 demos/pipette_cam/run_qc.py                 # both checkpoints, escalation
    python3 demos/pipette_cam/run_qc.py --sample-stride 2   # denser dye sampling (50%)

Two orthogonal failure modes hide in a dispensed plate, and it takes two
orthogonal checkpoints to clear them:

  1. WRONG VOLUME (mis-pipetted). The tip cam reads each well's meniscus HEIGHT ->
     volume on 100% of wells, cheaply, and the robot tops up / re-aspirates to fix
     it. This is what a camera is good at: a physical quantity it can see.

  2. WRONG CONCENTRATION (right volume, off-spec reagent). Invisible to any camera
     - same height, same look. Only a chemical readout catches it, so a small
     aliquot from a SAMPLED fraction goes to a plate-reader dye QC. Coverage is the
     cost, so a hit escalates: read the whole plate and HOLD the batch, because a
     wrong concentration can't be fixed by pipetting.

We plant both error types, run both checkpoints blind, and score what each catches
- camera-only, dye-at-sample, and the composed+escalated gate. Every number is
printed from this run (classical baseline, synthetic frames, modelled reader).
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

from labcv import synth, viz                                          # noqa: E402
import height                                                         # noqa: E402
import dye_qc                                                         # noqa: E402
from plr_bridge import Action, HoldForReviewError, decide_volume      # noqa: E402


@dataclass
class Config:
    rows: int = 8
    cols: int = 12
    seed: int = 5
    target_vol: float = 10.0     # uL target dispense
    max_uL: float = 20.0         # camera-readable well fill span
    tol_vol: float = 1.5         # volume spec tolerance (uL)
    target_conc: float = 1.0     # normalized target concentration
    tol_conc: float = 0.15       # concentration spec tolerance
    aliquot_uL: float = 2.0      # sample pulled per dye-QC well
    sample_stride: int = 4       # dye QC samples every Nth well (4 -> 25%)
    alarm_frac: float = 0.05     # sampled off-spec rate that triggers escalation
    out: str = "output/pipette_qc_checkpoints.png"


def _plate_labels(rows, cols):
    return [f"{chr(65 + i // cols)}{i % cols + 1}" for i in range(rows * cols)]


def run(cfg: Config) -> bool:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.rows * cfg.cols
    wells = _plate_labels(cfg.rows, cfg.cols)

    # ---- plant two independent failure modes = ground truth -------------------
    vol = np.full(n, cfg.target_vol) + rng.normal(0, 0.05, n)
    under_idx, over_idx = [2, 6, 10], [50, 54, 58]
    vol[under_idx] = [7.0, 7.5, 6.8]
    vol[over_idx] = [13.2, 12.8, 13.5]
    conc = np.full(n, cfg.target_conc) + rng.normal(0, 0.02, n)
    off_idx = [8, 40, 5, 45]                     # 8,40 fall in the sample; 5,45 do not
    conc[off_idx] = [0.65, 1.35, 0.70, 1.32]
    true_vol_err = np.abs(vol - cfg.target_vol) > cfg.tol_vol
    true_off = np.abs(conc - cfg.target_conc) > cfg.tol_conc

    # ---- checkpoint 1: tip-cam height -> volume, 100% of wells ----------------
    vol_est = np.zeros(n)
    vol_final = vol.copy()
    vol_flag = np.zeros(n, bool)
    for i in range(n):
        frame, _ = synth.column_view(vol[i], rng=rng, max_uL=cfg.max_uL)
        v, _conf = height.recover_volume(frame, cfg.max_uL)
        vol_est[i] = v
        act = decide_volume(v, cfg.target_vol, cfg.tol_vol)
        if act is not Action.PROCEED:
            vol_flag[i] = True
            vol_final[i] = cfg.target_vol        # top-up / re-aspirate corrects it
    vol_caught = int((vol_flag & true_vol_err).sum())
    n_vol = int(true_vol_err.sum())
    vol_fp = int((vol_flag & ~true_vol_err).sum())
    vol_mae = float(np.mean(np.abs(vol_est - vol)))
    vol_cleared = int((np.abs(vol_final - cfg.target_vol) <= cfg.tol_vol).sum())

    # ---- checkpoint 2: sampled plate-reader dye QC (the chemistry) ------------
    sampled = dye_qc.sample_indices(n, cfg.sample_stride)
    est = dye_qc.read_dye(conc, sampled, target=cfg.target_conc, rng=rng)
    flagged = dye_qc.flag_offspec(est, cfg.target_conc, cfg.tol_conc)
    n_off = int(true_off.sum())
    dye_caught = len(flagged)
    hit_frac = dye_caught / max(len(sampled), 1)

    escalate = hit_frac > cfg.alarm_frac
    held = set()
    if escalate:                                 # read the whole plate, hold the batch
        est_all = dye_qc.read_dye(conc, range(n), target=cfg.target_conc, rng=rng)
        held = dye_qc.flag_offspec(est_all, cfg.target_conc, cfg.tol_conc)
    dye_caught_final = len(held) if escalate else dye_caught
    sample_used = cfg.aliquot_uL * (n if escalate else len(sampled))

    # ---- report ---------------------------------------------------------------
    print("\nINTEGRATED QC CHECKPOINTS - tip-cam height + sampled plate-reader dye QC")
    print(f"  plate={cfg.rows}x{cfg.cols} ({n} wells)   target={cfg.target_vol:.0f} uL @ "
          f"conc {cfg.target_conc:.2f}   dye sampling=1/{cfg.sample_stride} "
          f"({len(sampled)} wells)")
    print(f"  planted: {n_vol} volume errors, {n_off} off-spec (wrong concentration), "
          f"independent\n")

    print(f"  checkpoint 1 - tip-cam liquid height (100% of wells):")
    print(f"    volume errors caught       {vol_caught}/{n_vol}   false alarms {vol_fp}")
    print(f"    volume MAE                 {vol_mae:.2f} uL")
    print(f"    closed loop -> re-dispensed to spec: {vol_cleared}/{n} wells within tol\n")

    print(f"  checkpoint 2 - plate-reader dye QC (sampled, orthogonal):")
    print(f"    off-spec caught @1/{cfg.sample_stride} sampling   {dye_caught}/{n_off}"
          f"   (sample hit rate {hit_frac*100:.0f}%)")
    if escalate:
        print(f"    hit rate > {cfg.alarm_frac*100:.0f}% alarm -> ESCALATE: read all {n} wells, HOLD batch")
        print(f"    off-spec caught after escalation  {dye_caught_final}/{n_off}")
    print(f"    sample consumed            {sample_used:.0f} uL "
          f"({cfg.aliquot_uL:.0f} uL/well)\n")

    print("  composed gate - a well passes only if BOTH checkpoints clear it:")
    print(f"    {'failure mode':<26}{'camera':>10}{'dye@1/'+str(cfg.sample_stride):>10}{'escalated':>11}")
    print("    " + "-" * 57)
    print(f"    {'wrong volume  (n=%d)' % n_vol:<26}{f'{vol_caught}/{n_vol} fix':>10}{'--':>10}{'--':>11}")
    print(f"    {'wrong conc.   (n=%d)' % n_off:<26}{'0/%d' % n_off:>10}"
          f"{f'{dye_caught}/{n_off}':>10}{f'{dye_caught_final}/{n_off}':>11}")

    _qc_panel(cfg, conc, sampled, true_off)

    # PASS: camera catches+fixes every volume error, and the escalated dye QC
    # catches every off-spec well the camera is blind to.
    ok = (vol_caught == n_vol and vol_cleared == n and dye_caught_final == n_off)
    if ok:
        print(f"\n  PASS: camera cleared {n_vol}/{n_vol} volume errors; dye QC held "
              f"{dye_caught_final}/{n_off} off-spec wells the camera could not see.\n")
    else:
        print(f"\n  FAIL: vol {vol_caught}/{n_vol} cleared {vol_cleared}/{n}, "
              f"off-spec {dye_caught_final}/{n_off}.\n")

    if not escalate and n_off:
        print("  ^ at this sampling the dye QC missed off-spec wells outside the sample.")
        print("    Denser sampling (--sample-stride 2) or the escalation rule closes it.\n")
    return ok


def _qc_panel(cfg, conc, sampled, true_off):
    """One figure: representative height frames (camera) + the dye-QC plate map."""
    Rect = viz.plt.Rectangle
    demo_vols = [cfg.target_vol, 7.0, 13.3, 0.4]         # on-spec, under, over, ethanol-residual
    tags = ["on-spec", "under", "over", "residual"]
    mosaic = [["h0", "h1", "plate", "plate"],
              ["h2", "h3", "plate", "plate"]]
    fig, ax = viz.plt.subplot_mosaic(mosaic, figsize=(10.6, 5.0))

    rng = np.random.default_rng(0)
    for k, (vv, tag) in enumerate(zip(demo_vols, tags)):
        a = ax[f"h{k}"]
        frame, _ = synth.column_view(vv, rng=rng, max_uL=cfg.max_uL)
        est, _c = height.recover_volume(frame, cfg.max_uL)
        viz.show(a, frame, title=f"{tag}\n{est:.1f} uL")

    grid = conc.reshape(cfg.rows, cfg.cols)
    p = ax["plate"]
    im = p.imshow(grid, cmap=viz.S.cmap("teal"), vmin=cfg.target_conc - 0.5,
                  vmax=cfg.target_conc + 0.5)
    p.set_title("plate-reader dye QC - concentration (circles = sampled, boxes = off-spec, held)")
    p.set_xticks(range(cfg.cols), [str(c + 1) for c in range(cfg.cols)], fontsize=8)
    p.set_yticks(range(cfg.rows), [chr(65 + r) for r in range(cfg.rows)], fontsize=8)
    p.grid(False)
    for i in sampled:                                    # sampled wells -> open circle
        r, c = divmod(i, cfg.cols)
        p.add_patch(viz.plt.Circle((c, r), 0.30, fill=False,
                    edgecolor=viz.S.INK, linewidth=1.3))
    for i in np.where(true_off)[0]:                      # off-spec (held) -> pink box
        r, c = divmod(i, cfg.cols)
        p.add_patch(Rect((c - 0.45, r - 0.45), 0.9, 0.9, fill=False,
                    edgecolor=viz.S.OUTLINE["pink"], linewidth=2.4))
    fig.colorbar(im, ax=p, fraction=0.046, pad=0.03, label="concentration")
    viz.save(fig, os.path.join(ROOT, cfg.out))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--sample-stride", type=int, default=4,
                   help="dye QC samples every Nth well (2 -> 50%%, 4 -> 25%%)")
    a = p.parse_args()
    ok = run(Config(seed=a.seed, sample_stride=a.sample_stride))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
