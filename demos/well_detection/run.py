#!/usr/bin/env python3
"""Well detection demo - plant well boxes, detect them blind, score with COCO AP.

    python3 demos/well_detection/run.py            # classical baseline
    python3 demos/well_detection/run.py --occluder # a 'hand' hides ~2 wells

Ground truth is the boxes we drew; the detector never sees them. We report
AP@0.5 and AP@[.5:.95] (COCO) plus precision/recall at a 0.5 score threshold,
using the shared eval.metrics. Every number below is printed from this run.
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

from eval import metrics as M          # noqa: E402
from labcv import synth, viz           # noqa: E402
from detect import detect              # noqa: E402


@dataclass
class Config:
    rows: int = 6
    cols: int = 8
    px: int = 384
    seed: int = 0
    occluder: bool = False
    distractors: int = 4
    score_thr: float = 0.5
    pass_ap50: float = 0.80
    out: str = "output/well_detection_qc.png"


def run(cfg: Config) -> bool:
    rng = np.random.default_rng(cfg.seed)
    img, gt, _ = synth.microplate(cfg.rows, cfg.cols, cfg.px, rng=rng,
                                  occluder=cfg.occluder, distractors=cfg.distractors)
    pred, scores = detect(img, model="classical")

    ap50 = M.average_precision(gt, pred, scores, 0.5)
    mAP, per = M.ap_range(gt, pred, scores)
    op = M.precision_recall(gt, pred, scores, 0.5, cfg.score_thr)

    print("\nWELL DETECTION - clean-room plant-and-recover (classical baseline)")
    print(f"  plate={cfg.rows}x{cfg.cols}  wells={len(gt)}  distractors={cfg.distractors}"
          f"  occluder={'ON' if cfg.occluder else 'off'}\n")
    print(f"  {'metric':<22}{'value':>8}")
    print("  " + "-" * 30)
    print(f"  {'AP@0.50':<22}{ap50:>8.3f}")
    print(f"  {'AP@[.5:.95] (COCO)':<22}{mAP:>8.3f}")
    print(f"  {'precision @0.5':<22}{op['precision']:>8.3f}")
    print(f"  {'recall @0.5':<22}{op['recall']:>8.3f}")
    det_str = f"{op['tp'] + op['fp']} / {len(gt)}"
    print(f"  {'detections / wells':<22}{det_str:>8}")

    labels = [f"{s:.2f}" for s in scores]
    fig, ax = viz.plt.subplots(1, 1, figsize=(5.2, 5.2))
    viz.show(ax, img, title=f"well detection - AP50={ap50:.2f}  AP[.5:.95]={mAP:.2f}")
    viz.boxes(ax, gt, viz.S.OUTLINE["green"], lw=1.8)               # ground truth
    viz.boxes(ax, pred, viz.S.OUTLINE["blue"], lw=1.2, labels=labels)  # detected
    ax.plot([], [], color=viz.S.OUTLINE["green"], label="planted well")
    ax.plot([], [], color=viz.S.OUTLINE["blue"], label="detected")
    ax.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08))
    viz.save(fig, os.path.join(ROOT, cfg.out))

    ok = ap50 >= cfg.pass_ap50
    print(f"\n  {'PASS' if ok else 'FAIL'}: AP@0.5 {ap50:.3f} "
          f"{'>=' if ok else '<'} {cfg.pass_ap50:.2f}\n")
    if cfg.occluder and op["fn"] > 0:
        print(f"  ^ the classical detector misses {op['fn']} occluded well(s). That is the")
        print("    seam where a learned detector (RF-DETR / RT-DETRv4) recovers hidden")
        print("    instances behind the same detect() interface.\n")
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--occluder", action="store_true")
    p.add_argument("--distractors", type=int, default=4)
    a = p.parse_args()
    ok = run(Config(seed=a.seed, occluder=a.occluder, distractors=a.distractors))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
