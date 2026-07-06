"""Per-instance state classification - is each detected well filled or empty?

This is the spatial-verification half of video-verified execution: detection
says *where* the instances are, classification says *what state* each is in,
with a confidence. The classical baseline reads central intensity; the seam is
identical for a learned classifier or a VLM (see demos/vocab_vlm).

    classify_well(crop)          -> (state{0,1}, confidence[0..1])
    classify_states(img, boxes)  -> (states[N], confidence[N])

Confidence collapses toward 0 as a well's appearance approaches the decision
boundary (an under-filled / air-gapped well) - that low-confidence flag is the
QC checkpoint: the motion looked right, but the state is ambiguous.
"""
from __future__ import annotations

import numpy as np

FILLED, EMPTY = 1, 0


def classify_well(crop, thr=0.42, scale=0.09):
    """Read the central patch of a well ROI; brighter center -> filled."""
    if crop.size == 0:
        return EMPTY, 0.0
    h, w = crop.shape
    ry, rx = int(h * 0.30), int(w * 0.30)
    cy, cx = h // 2, w // 2
    center = crop[max(0, cy - ry):cy + ry + 1, max(0, cx - rx):cx + rx + 1]
    val = float(center.mean())
    state = FILLED if val >= thr else EMPTY
    # confidence = distance from the boundary, squashed to [0,1]
    conf = 2.0 * abs(1.0 / (1.0 + np.exp(-(val - thr) / scale)) - 0.5)
    return state, float(round(conf, 4))


def classify_states(image, boxes, **kw):
    states, confs = [], []
    H, W = image.shape
    for x1, y1, x2, y2 in np.asarray(boxes, float).reshape(-1, 4):
        xa, ya = max(0, int(round(x1))), max(0, int(round(y1)))
        xb, yb = min(W, int(round(x2))), min(H, int(round(y2)))
        s, c = classify_well(image[ya:yb, xa:xb], **kw)
        states.append(s)
        confs.append(c)
    return np.array(states, int), np.array(confs, float)
