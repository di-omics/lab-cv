#!/usr/bin/env python3
"""ROI tracking demo - keep labware identities stable across a deck video.

    python3 demos/roi_tracking/run.py            # objects cross -> baseline swaps ids
    python3 demos/roi_tracking/run.py --no-cross # parallel paths -> 0 switches

We plant per-frame boxes with stable ids, hand the tracker per-frame detections
(jittered, id-free - as a detector would), and score identity stability
(ID switches, from eval.metrics) plus mean localization IoU. The headline number
is ID switches: the classical baseline swaps ids where paths cross; SAM2's memory
does not - same interface (see tracker.SAM2Tracker).
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
from tracker import IoUTracker         # noqa: E402


@dataclass
class Config:
    n_objs: int = 3
    n_frames: int = 24
    px: int = 320
    seed: int = 2
    cross: bool = True
    jitter: float = 2.0
    out: str = "output/roi_tracking_qc.png"


def run(cfg: Config) -> bool:
    rng = np.random.default_rng(cfg.seed)
    frames, gt = synth.moving_labware(cfg.n_objs, cfg.n_frames, cfg.px,
                                      rng=rng, cross=cfg.cross)

    # per-frame detections: ground-truth boxes + small jitter, identities stripped
    dets = {}
    for f, objs in gt.items():
        dets[f] = [np.array(b) + rng.normal(0, cfg.jitter, 4) for _, b in objs]

    tk = IoUTracker()
    tk.init_state(frames, dets)
    for oid, b in gt[0]:                     # seed identities on frame 0
        tk.add_new_box(0, oid, b)

    pred = {}
    for f, ids, boxes in tk.propagate_in_video():
        pred[f] = list(zip(ids, boxes))

    switches = M.count_id_switches(gt, pred, iou_thr=0.3)
    ious = [M.iou(gb, pb) for f in gt
            for (_, gb), (_, pb) in zip(sorted(gt[f]), sorted(pred[f]))]
    mean_iou = float(np.mean(ious))

    print("\nROI TRACKING - SAM2-shaped interface, classical IoU baseline")
    print(f"  objects={cfg.n_objs}  frames={cfg.n_frames}  paths="
          f"{'CROSSING' if cfg.cross else 'parallel'}\n")
    print(f"  mean localization IoU   {mean_iou:.3f}")
    print(f"  ID switches             {switches}")

    fig, ax = viz.plt.subplots(1, 2, figsize=(9, 4.4))
    viz.show(ax[0], frames[cfg.n_frames // 2], title="mid-clip frame")
    for oid, b in gt[cfg.n_frames // 2]:
        viz.boxes(ax[0], [b], list(viz.S.OUTLINE.values())[oid % 7], lw=1.8,
                  labels=[f"id{oid}"])
    ax[1].set_title("recovered id tracks (centroid paths)")
    for oid in range(cfg.n_objs):
        xs = [np.mean([b[0], b[2]]) for f in sorted(pred)
              for i, b in pred[f] if i == oid]
        ys = [np.mean([b[1], b[3]]) for f in sorted(pred)
              for i, b in pred[f] if i == oid]
        ax[1].plot(xs, ys, "-o", ms=2.5, color=list(viz.S.OUTLINE.values())[oid % 7],
                   label=f"id{oid}")
    ax[1].invert_yaxis(); ax[1].set_aspect("equal"); ax[1].legend()
    ax[1].set_xlabel("x (px)"); ax[1].set_ylabel("y (px)")
    viz.save(fig, os.path.join(ROOT, cfg.out))

    ok = mean_iou >= 0.70
    print(f"\n  {'PASS' if ok else 'FAIL'}: localization IoU {mean_iou:.3f} "
          f"{'>=' if ok else '<'} 0.70")
    if cfg.cross and switches > 0:
        print(f"\n  ^ {switches} id switch(es) at the crossing. The greedy IoU tracker has")
        print("    no appearance memory; SAM2's per-object memory holds identity through")
        print("    the crossing behind the same interface (tracker.SAM2Tracker).\n")
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--no-cross", dest="cross", action="store_false")
    p.add_argument("--objs", type=int, default=3)
    a = p.parse_args()
    ok = run(Config(seed=a.seed, cross=a.cross, n_objs=a.objs))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
