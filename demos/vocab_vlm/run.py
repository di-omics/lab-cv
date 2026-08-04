#!/usr/bin/env python3
"""Detector -> VLM layering demo - cheap detection everywhere, VLM only on doubt.

    python3 demos/vocab_vlm/run.py            # offline deterministic mock VLM

A classical detector proposes boxes on every frame (fast, no network). Instead
of trusting every box, we ESCALATE only the low-confidence ones to an
open-vocabulary VLM (mock backend here; Qwen3-VL / Gemini 3 behind the same
interface) to name what they actually are - separating real wells from bubbles
and reading fill state open-vocab. We score labeling accuracy against the plant.

Every number below is printed from this run. The point is the layering, not the
mock: run the $0 detector at frame rate, spend VLM calls only where it matters.
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
sys.path.insert(0, os.path.join(ROOT, "demos", "well_detection"))

from labcv import synth, viz           # noqa: E402
from detect import detect              # noqa: E402
from adapter import label_regions      # noqa: E402

VOCAB = ["empty well", "filled well", "bubble"]


@dataclass
class Config:
    rows: int = 4
    cols: int = 6
    px: int = 320
    seed: int = 3
    distractors: int = 3
    escalate_below: float = 0.85
    out: str = "output/vocab_vlm_qc.png"


def _cheap_label(image, box):
    """Name a box without spending a VLM call.

    This is the $0 path, and its vocabulary is deliberately narrower than
    ``VOCAB``: brightness alone separates a filled well from an empty one, and
    nothing about a single crop's brightness says "bubble". A speck that the
    detector scored confidently therefore comes back named as a well, which is
    the cost of not making the call. Reporting that cost is the point; a gate
    whose savings are printed without them is a gate that looks free.
    """
    img = np.asarray(image, dtype=float)
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if crop.size == 0:
        return "empty well", 0.0
    inten = float(crop.mean())
    label = "filled well" if inten >= 0.45 else "empty well"
    conf = float(min(1.0, abs(inten - 0.45) / 0.45))
    return label, conf


def _truth_for(box, gt_boxes, gt_states):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    for i, (x1, y1, x2, y2) in enumerate(gt_boxes):
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return "filled well" if gt_states[i] else "empty well"
    return "bubble"                      # not over any planted well -> a distractor


def run(cfg: Config) -> bool:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.rows * cfg.cols
    states = (rng.random(n) < 0.5).astype(int)
    img, gt_boxes, gt_states = synth.microplate(
        cfg.rows, cfg.cols, cfg.px, rng=rng, states=states, distractors=cfg.distractors)

    boxes, scores = detect(img, model="classical")
    escalate = scores < cfg.escalate_below

    # The gate is enforced here, not described here. Only the escalated boxes
    # cross the adapter boundary; the rest are named by the cheap path. Before
    # this, `escalate` was computed and every box was sent anyway, so the
    # "calls saved" line counted a branch nothing took.
    esc_idx = [i for i, e in enumerate(escalate) if e]
    escalated_boxes = [boxes[i] for i in esc_idx]
    vlm_labels = (label_regions(img, escalated_boxes, VOCAB, backend="mock")
                  if escalated_boxes else [])

    labels = [None] * len(boxes)
    for slot, i in enumerate(esc_idx):
        labels[i] = vlm_labels[slot]
    for i in range(len(boxes)):
        if labels[i] is None:
            labels[i] = _cheap_label(img, boxes[i])

    correct, total = 0, 0
    cheap_wrong, cheap_total = 0, 0
    cheap_wrong_on_specks = 0
    for i, (box, (lab, _)) in enumerate(zip(boxes, labels)):
        truth = _truth_for(box, gt_boxes, gt_states)
        total += 1
        hit = int(lab == truth)
        correct += hit
        if not escalate[i]:
            cheap_total += 1
            if not hit:
                cheap_wrong += 1
                if truth == "bubble":
                    cheap_wrong_on_specks += 1

    acc = correct / max(total, 1)
    n_esc = int(escalate.sum())

    print("\nVOCAB-VLM LAYERING - detector proposes, VLM adjudicates (mock backend)")
    print(f"  vocab={VOCAB}")
    print(f"  detections={total}   escalated to VLM (conf<{cfg.escalate_below})={n_esc}"
          f"   ({100*n_esc/max(total,1):.0f}% of frames' boxes)\n")
    print(f"  open-vocab labeling accuracy vs plant   {acc:.3f}")
    print(f"  VLM calls actually made                 {n_esc}/{total}")
    print(f"  VLM calls saved by layering             {total - n_esc}/{total}"
          f"  ({100*(total-n_esc)/max(total,1):.0f}%)")
    print(f"  cost of the saved calls                 {cheap_wrong}/{cheap_total}"
          f"  cheap-path labels wrong")
    if cheap_wrong_on_specks:
        print(f"  {cheap_wrong_on_specks} of the boxes the gate kept from the VLM are specks, and the")
        print("  cheap path has no word for a bubble, so it called them wells. Raising")
        print("  the threshold would hide this by fitting it to the data it judges.")

    lab_txt = [f"{name} {c:.2f}" for (name, c) in labels]
    col_of = {"filled well": viz.S.OUTLINE["blue"],
              "empty well": viz.S.OUTLINE["peach"],
              "bubble": viz.S.OUTLINE["pink"]}
    fig, ax = viz.plt.subplots(1, 1, figsize=(5.8, 4.6))
    viz.show(ax, img, title=f"open-vocab labels (mock VLM) - acc {acc:.2f}")
    viz.plate_labels(ax, gt_boxes, cfg.rows, cfg.cols)
    for box, (lab, _), t in zip(boxes, labels, lab_txt):
        viz.boxes(ax, [box], col_of[lab], lw=1.6, labels=[t])
    viz.save(fig, os.path.join(ROOT, cfg.out))

    ok = acc >= 0.95
    print(f"\n  {'PASS' if ok else 'FAIL'}: labeling accuracy {acc:.3f} "
          f"{'>=' if ok else '<'} 0.95")
    print("  layering: classical detector runs on every frame; a VLM (Qwen3-VL for")
    print("  vocabulary, Gemini 3 for reasoning) is spent only on low-confidence boxes.\n")
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--distractors", type=int, default=3)
    a = p.parse_args()
    ok = run(Config(seed=a.seed, distractors=a.distractors))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
