#!/usr/bin/env python3
"""Well-state verification demo - the spatial-verification half of video-verified
execution: detect each well, then classify it filled/empty with a confidence,
and flag the ambiguous ones.

    python3 demos/well_state/run.py                # random fill pattern
    python3 demos/well_state/run.py --partial 4    # 4 under-filled -> low-conf flags

We plant the true fill state of every well, detect instances blind, classify
each, and score accuracy + per-class precision/recall + a confusion matrix.
Reports instances and per-frame latency ("N instances, X ms"), like a live
verifier. The low-confidence flags are the QC checkpoint: appearance looked
plausible, but the state is not certain -> hold for orthogonal QC. That is the
"motions look right, chemistry off" seam, in CV terms.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "demos", "well_detection"))

from eval import metrics as M          # noqa: E402
from labcv import synth, viz           # noqa: E402
from detect import detect              # noqa: E402
from classify import classify_states   # noqa: E402

LABELS = ["empty", "filled"]


@dataclass
class Config:
    rows: int = 6
    cols: int = 8
    px: int = 384
    seed: int = 1
    filled_frac: float = 0.5
    partial: int = 0
    flag_conf: float = 0.60
    out: str = "output/well_state_qc.png"


def _true_state_for(box, gt_boxes, gt_states):
    """Which planted well does this detection cover? -> its true fill state."""
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    for i, (x1, y1, x2, y2) in enumerate(gt_boxes):
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return int(gt_states[i])
    return None


def run(cfg: Config) -> bool:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.rows * cfg.cols
    states = (rng.random(n) < cfg.filled_frac).astype(int)
    partial_idx = list(np.where(states == 1)[0][:cfg.partial])   # under-fill some filled wells
    img, gt_boxes, gt_states = synth.microplate(
        cfg.rows, cfg.cols, cfg.px, rng=rng, states=states, partial_idx=partial_idx)

    t0 = time.perf_counter()
    pred_boxes, _ = detect(img, model="classical")
    pred_states, confs = classify_states(img, pred_boxes)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    y_true, y_pred, keep_conf = [], [], []
    for box, ps, c in zip(pred_boxes, pred_states, confs):
        ts = _true_state_for(box, gt_boxes, gt_states)
        if ts is None:
            continue
        y_true.append(LABELS[ts])
        y_pred.append(LABELS[ps])
        keep_conf.append(c)
    rep = M.classification_report(y_true, y_pred, LABELS)
    flagged = int(np.sum(np.array(keep_conf) < cfg.flag_conf))

    print("\nWELL-STATE VERIFICATION - detect -> classify -> score (classical baseline)")
    print(f"  instances={len(pred_boxes)}   latency={latency_ms:.1f} ms/frame"
          f"   planted filled={int(gt_states.sum())}/{n}"
          f"   under-filled={cfg.partial}\n")
    print(f"  accuracy            {rep['accuracy']:.3f}")
    for lab in LABELS:
        pc = rep["per_class"][lab]
        print(f"  {lab:<8} P={pc['precision']:.3f}  R={pc['recall']:.3f}  n={pc['support']}")
    cm = rep["confusion"]
    print("\n  confusion (rows=true, cols=pred)      pred_empty  pred_filled")
    print(f"    true_empty                          {cm[0,0]:>10}  {cm[0,1]:>11}")
    print(f"    true_filled                         {cm[1,0]:>10}  {cm[1,1]:>11}")
    print(f"\n  flagged (conf < {cfg.flag_conf:.2f}): {flagged}"
          f"  -> QC checkpoint, hold for orthogonal verification")

    # QC panel: each well boxed + labelled state@confidence, low-conf in red
    labels = [f"{LABELS[s]} {c:.2f}" for s, c in zip(pred_states, confs)]
    cols = [viz.S.OUTLINE["blue"] if s else viz.S.OUTLINE["peach"] for s in pred_states]
    fig, ax = viz.plt.subplots(1, 1, figsize=(5.4, 5.6))
    viz.show(ax, img, title=f"well-state - acc {rep['accuracy']:.2f}, "
                            f"{len(pred_boxes)} instances, {latency_ms:.0f} ms")
    for box, lab, c, cf in zip(pred_boxes, labels, cols, confs):
        col = viz.S.OUTLINE["pink"] if cf < cfg.flag_conf else c
        viz.boxes(ax, [box], col, lw=1.6, labels=[lab])
    viz.save(fig, os.path.join(ROOT, cfg.out))

    ok = rep["accuracy"] >= 0.95
    print(f"\n  {'PASS' if ok else 'FAIL'}: state accuracy {rep['accuracy']:.3f} "
          f"{'>=' if ok else '<'} 0.95\n")
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--filled-frac", type=float, default=0.5)
    p.add_argument("--partial", type=int, default=0, help="under-fill N wells -> low-conf flags")
    a = p.parse_args()
    ok = run(Config(seed=a.seed, filled_frac=a.filled_frac, partial=a.partial))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
