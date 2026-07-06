"""Well detection - one clean interface, a classical baseline, a learned seam.

    detect(image, model="classical") -> (boxes[N,4], scores[N])

The baseline is threshold -> morphology -> connected components -> box, scored
by how close each blob's size is to the plate's dominant well size. It is
honest about where it breaks: occluded or touching wells. That is exactly the
seam where a learned detector drops in behind the SAME interface:

    detect(image, model="rfdetr")   # pip install rfdetr  (Apache-2.0), optional

so every downstream demo (state, tracking) is model-agnostic.
"""
from __future__ import annotations

import cv2
import numpy as np


def _classical(image, thr=0.45, min_area=140, max_area_frac=0.02):
    g = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    _, mask = cv2.threshold(g, int(thr * 255), 255, cv2.THRESH_BINARY)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker, iterations=1)

    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    max_area = image.shape[0] * image.shape[1] * max_area_frac
    cand = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area or area > max_area:
            continue
        aspect = w / max(h, 1)
        if not (0.55 < aspect < 1.8):                 # wells are ~round
            continue
        cand.append((x, y, w, h, area))

    if not cand:
        return np.zeros((0, 4), float), np.zeros((0,), float)

    areas = np.array([c[4] for c in cand], float)
    expected = float(np.median(areas))                # wells dominate -> median
    boxes, scores = [], []
    for x, y, w, h, area in cand:
        # confidence = closeness (in sqrt-area) to the dominant well size,
        # so wells outrank specks/streaks -> a usable score ranking for AP.
        z = (np.sqrt(area) - np.sqrt(expected)) / (0.4 * np.sqrt(expected))
        size_fit = float(np.exp(-0.5 * z * z))
        aspect = w / max(h, 1)
        aspect_fit = float(np.exp(-0.5 * ((aspect - 1.0) / 0.35) ** 2))
        boxes.append([x, y, x + w, y + h])
        scores.append(round(0.5 + 0.5 * size_fit * aspect_fit, 4))
    return np.array(boxes, float), np.array(scores, float)


def _rfdetr(image, **kw):
    try:
        from rfdetr import RFDETRBase  # noqa: F401
    except Exception as e:  # pragma: no cover - optional path
        raise RuntimeError(
            "real-model path needs `pip install -r requirements-models.txt` "
            "(rfdetr, Apache-2.0). The classical baseline runs with no extra deps."
        ) from e
    raise NotImplementedError(
        "Seam stub: load RFDETRBase weights, run model.predict(image), and map "
        "the returned instances to (boxes[N,4], scores[N]). Interface is identical."
    )


def detect(image, model="classical", **kw):
    if model == "classical":
        return _classical(image, **kw)
    if model in ("rfdetr", "rt-detr", "rtdetrv4"):
        return _rfdetr(image, **kw)
    raise ValueError(f"unknown model {model!r}")
