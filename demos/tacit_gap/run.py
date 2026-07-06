#!/usr/bin/env python3
"""The tacit gap - where spatial verification says GO and the chemistry says NO.

    python3 demos/tacit_gap/run.py                # 6 off-spec wells hidden in plain sight
    python3 demos/tacit_gap/run.py --offspec 10

Premise: a plate where every well is correctly *filled*, so a spatial verifier
(per-well filled/empty + confidence - the visible half of video-verified
execution) passes all of them: "video says GO." But a subset were dispensed at
the wrong concentration. That failure is invisible to a camera - same volume,
same look - so spatial verification misses 100% of it. A paired ground-truth
readout (an orthogonal plate-reader signal, i.e. the validation ladder) catches
it: "chemistry says NO."

We plant the mismatch, run both layers blind, and score how many tacit failures
each catches. The point is composition: spatial proves the motion, the
ground-truth layer proves the chemistry, and only together are they a real trust
layer. Same clean-room style; every number is printed from this run.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "demos", "well_detection"))
sys.path.insert(0, os.path.join(ROOT, "demos", "well_state"))

from eval import metrics as M          # noqa: E402
from labcv import synth, viz           # noqa: E402
from detect import detect              # noqa: E402
from classify import classify_states   # noqa: E402

LABELS = ["in_spec", "off_spec"]


@dataclass
class Config:
    rows: int = 6
    cols: int = 8
    px: int = 384
    seed: int = 4
    offspec: int = 6
    target: float = 1.0          # target concentration (normalized)
    tol: float = 0.15            # spec tolerance
    reader_noise: float = 0.035  # plate-reader measurement noise (1 sigma)
    out: str = "output/tacit_gap_qc.png"


def run(cfg: Config) -> bool:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.rows * cfg.cols

    # every well is filled (present) -> the plate looks perfect to a camera
    states = np.ones(n, int)
    # plant true concentrations: most on-target, a hidden subset off-spec
    conc = cfg.target + rng.normal(0, 0.03, n)
    off_idx = rng.choice(n, size=cfg.offspec, replace=False)
    conc[off_idx] = cfg.target + rng.choice([-1, 1], cfg.offspec) * rng.uniform(0.25, 0.5, cfg.offspec)
    true_offspec = (np.abs(conc - cfg.target) > cfg.tol).astype(int)

    img, gt_boxes, _ = synth.microplate(cfg.rows, cfg.cols, cfg.px, rng=rng, states=states)

    # ---- layer 1: spatial verification (the visible half) --------------------
    pred_boxes, _ = detect(img, model="classical")
    pred_states, confs = classify_states(img, pred_boxes)
    # spatial passes any well it sees as filled; it has no concentration signal,
    # so its verdict on the tacit question is "GO" for every filled well.
    spatial_flag = np.zeros(n, int)   # 0 = GO (nothing flagged as off-spec)

    # ---- layer 2: ground-truth / validation readout (the invisible half) -----
    reader = conc + rng.normal(0, cfg.reader_noise, n)
    gt_flag = (np.abs(reader - cfg.target) > cfg.tol).astype(int)

    # ---- score both layers against the planted tacit failures ----------------
    def recall(flag):
        return int(((flag == 1) & (true_offspec == 1)).sum()), int(true_offspec.sum())

    s_hit, k = recall(spatial_flag)
    g_hit, _ = recall(gt_flag)
    g_fp = int(((gt_flag == 1) & (true_offspec == 0)).sum())
    composed = ((states == 1) & (gt_flag == 0)).astype(int)  # GO iff filled AND in-spec

    print("\nTHE TACIT GAP - spatial says GO, chemistry says NO")
    print(f"  wells={n}  all filled  planted off-spec (wrong concentration)={k}\n")
    print(f"  {'layer':<34}{'off-spec caught':>16}")
    print("  " + "-" * 50)
    print(f"  {'spatial verification (camera only)':<34}{f'{s_hit}/{k}':>16}")
    print(f"  {'+ ground-truth readout (validation)':<34}{f'{g_hit}/{k}':>16}"
          f"   (false alarms: {g_fp})")
    print(f"\n  spatial-only MISSES {k - s_hit}/{k} tacit failures - invisible to a camera.")
    print(f"  composed trust layer clears {int(composed.sum())}/{n} wells "
          f"(filled AND in-spec); holds {n - int(composed.sum())} for review.")

    # ---- side-by-side QC: 'video says GO' vs 'chemistry says NO' --------------
    fig, ax = viz.plt.subplots(1, 2, figsize=(10.4, 5.2))
    viz.show(ax[0], img, title="spatial verification - every well: GO")
    for box, cf in zip(pred_boxes, confs):
        viz.boxes(ax[0], [box], viz.S.OUTLINE["green"], lw=1.4)

    grid = reader.reshape(cfg.rows, cfg.cols)
    im = ax[1].imshow(grid, cmap=viz.S.cmap("blue"), vmin=cfg.target - 0.6, vmax=cfg.target + 0.6)
    ax[1].set_title("ground-truth readout - off-spec flagged")
    ax[1].set_xticks([]); ax[1].set_yticks([]); ax[1].grid(False)
    for i in np.where(true_offspec == 1)[0]:
        r, c = divmod(i, cfg.cols)
        ax[1].add_patch(viz.plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                        edgecolor=viz.S.OUTLINE["pink"], linewidth=2.4))
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04, label="concentration")
    viz.save(fig, os.path.join(ROOT, cfg.out))

    ok = (g_hit == k) and (s_hit < k)
    print(f"\n  {'PASS' if ok else 'FAIL'}: ground-truth layer catches the tacit failures "
          f"spatial-only cannot ({g_hit}/{k} vs {s_hit}/{k}).\n")
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=4)
    p.add_argument("--offspec", type=int, default=6)
    a = p.parse_args()
    ok = run(Config(seed=a.seed, offspec=a.offspec))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
