"""Open-vocabulary labeling via a VLM adapter - one interface, swappable backend.

    label_regions(image, boxes, vocab, backend="mock") -> [(label, score), ...]

backend="mock" is a deterministic, offline stand-in: it scores each region
against a tiny prototype table (intensity / size / fill) so the demo runs with
no API keys and no network, yet exercises the exact call site. The live paths
are guarded and optional:

    backend="qwen"    Qwen3-VL   - open-vocabulary naming
    backend="gemini"  Gemini 3   - reasoning / adjudication

Layering (see run.py): a cheap detector proposes boxes on every frame; only
low-confidence or novel instances are escalated to a VLM. Same interface, so the
backend is a config flag, not a rewrite.
"""
from __future__ import annotations

import numpy as np

# prototype appearance for each concept: (box mean-intensity, size relative to
# the scene's median box). Wells are full-size; a filled well is bright, an
# empty well is dark; a bubble is a small bright speck. A real VLM keys on the
# same cues (brightness + size in context) - the mock just makes them explicit.
_PROTO = {
    "filled well": (0.66, 1.0),
    "empty well": (0.25, 1.0),
    "bubble": (0.50, 0.28),
}


def _features(image, box, median_area):
    H, W = image.shape
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    crop = image[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
    if crop.size == 0:
        return 0.0, 0.0
    inten = float(crop.mean())
    area = max(1.0, (x2 - x1) * (y2 - y1))
    rel_size = float(area / max(median_area, 1.0))
    return inten, rel_size


def _mock(image, boxes, vocab):
    out = []
    protos = {k: _PROTO[k] for k in vocab if k in _PROTO}
    B = np.asarray(boxes, float).reshape(-1, 4)
    areas = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) if len(B) else np.array([1.0])
    median_area = float(np.median(areas))
    for b in B:
        inten, size = _features(image, b, median_area)
        dists = {k: (inten - pi) ** 2 + (size - pf) ** 2 for k, (pi, pf) in protos.items()}
        best = min(dists, key=dists.get)
        # softmax-ish confidence from the distance gap to the runner-up
        vals = sorted(dists.values())
        gap = (vals[1] - vals[0]) if len(vals) > 1 else 1.0
        conf = float(round(1.0 / (1.0 + np.exp(-12.0 * gap)), 4))
        out.append((best, conf))
    return out


def label_regions(image, boxes, vocab, backend="mock"):
    if backend == "mock":
        return _mock(image, boxes, vocab)
    if backend in ("qwen", "gemini"):  # pragma: no cover - optional live path
        raise RuntimeError(
            f"live VLM backend {backend!r} needs credentials + "
            "`pip install -r requirements-models.txt`; the mock backend is offline "
            "and deterministic so the demo runs anywhere."
        )
    raise ValueError(f"unknown backend {backend!r}")
