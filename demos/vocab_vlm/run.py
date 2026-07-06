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
    labels = label_regions(img, boxes, VOCAB, backend="mock")

    correct, total = 0, 0
    for box, (lab, _), esc in zip(boxes, labels, escalate):
        truth = _truth_for(box, gt_boxes, gt_states)
        total += 1
        correct += int(lab == truth)

    acc = correct / max(total, 1)
    n_esc = int(escalate.sum())

    print("\nVOCAB-VLM LAYERING - detector proposes, VLM adjudicates (mock backend)")
    print(f"  vocab={VOCAB}")
    print(f"  detections={total}   escalated to VLM (conf<{cfg.escalate_below})={n_esc}"
          f"   ({100*n_esc/max(total,1):.0f}% of frames' boxes)\n")
    print(f"  open-vocab labeling accuracy vs plant   {acc:.3f}")
    print(f"  VLM calls saved by layering             {total - n_esc}/{total}"
          f"  ({100*(total-n_esc)/max(total,1):.0f}%)")

    lab_txt = [f"{l} {c:.2f}" for (l, c) in labels]
    col_of = {"filled well": viz.S.OUTLINE["blue"],
              "empty well": viz.S.OUTLINE["peach"],
              "bubble": viz.S.OUTLINE["pink"]}
    fig, ax = viz.plt.subplots(1, 1, figsize=(5.6, 4.6))
    viz.show(ax, img, title=f"open-vocab labels (mock VLM) - acc {acc:.2f}")
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
