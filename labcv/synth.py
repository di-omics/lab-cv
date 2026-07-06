"""Clean-room synthetic lab imagery - generated on the fly from a seed.

No real media, no downloads. Each generator returns the image AND the planted
ground truth (well boxes, fill states, or object tracks) so a demo can score
recovery against exactly what was drawn.

    microplate(...)     -> (img, boxes, states)   still frame of a well plate
    moving_labware(...) -> (frames, tracks)        short deck video, moving labware
"""
from __future__ import annotations

import cv2
import numpy as np

DECK = 0.12      # dark deck background
PLATE = 0.24     # plate body
RIM = 0.60       # bright well rim (always visible -> what detection keys on)
FILLED = 0.78    # liquid in a well
EMPTY = 0.20     # dry well interior


def microplate(rows=6, cols=8, px=384, rng=None, states=None, well_frac=0.40,
               noise=0.015, occluder=False, distractors=0, partial_idx=None):
    """A top-down microplate frame.

    states       per-well 0/1 fill (row-major). None -> all wells rendered
                 visible with neutral interior (detection-only frame).
    occluder     draw a dark 'gloved hand / pipettor' bar over ~2 wells so a
                 classical detector misses them (recall < 1) - the learned-
                 detector seam.
    distractors  bright specks (bubbles/condensation) -> candidate false pos.
    partial_idx  wells drawn under-filled (ambiguous) -> low classifier
                 confidence, i.e. a QC flag ("motions look right, chemistry off").
    """
    rng = np.random.default_rng(0) if rng is None else rng
    img = np.full((px, px), DECK, np.float32)
    m = int(px * 0.07)
    cv2.rectangle(img, (m, m), (px - m, px - m), PLATE, -1)

    xs = np.linspace(m + m, px - m - m, cols)
    ys = np.linspace(m + m, px - m - m, rows)
    r = float(min(xs[1] - xs[0], ys[1] - ys[0]) * well_frac)

    partial_idx = set(partial_idx or [])
    boxes, st = [], []
    k = 0
    for yc in ys:
        for xc in xs:
            filled = 1 if states is None else int(states[k])
            interior = 0.50 if states is None else (FILLED if filled else EMPTY)
            if k in partial_idx:                        # under-filled -> ambiguous
                interior = 0.5 * (FILLED + EMPTY)
            cv2.circle(img, (int(xc), int(yc)), int(r), float(interior), -1, cv2.LINE_AA)
            cv2.circle(img, (int(xc), int(yc)), int(r), RIM, 2, cv2.LINE_AA)
            boxes.append([xc - r, yc - r, xc + r, yc + r])
            st.append(filled)
            k += 1

    centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
    placed, attempts = 0, 0                             # bright specks away from wells -> isolated FP bait
    while placed < distractors and attempts < 400:
        attempts += 1
        sx, sy = rng.uniform(r, px - r), rng.uniform(r, px - r)
        if all((sx - cx) ** 2 + (sy - cy) ** 2 > (r * 1.7) ** 2 for cx, cy in centers):
            cv2.circle(img, (int(sx), int(sy)), int(r * 0.5), RIM, -1, cv2.LINE_AA)
            placed += 1

    if occluder:                                        # dark bar over ~2 wells
        cx = int(xs[cols // 2]); cy = int(ys[0])
        cv2.rectangle(img, (cx - int(r), cy - int(r * 1.4)),
                      (cx + int(r * 3.2), cy + int(r * 1.4)), DECK * 0.6, -1)

    img = np.clip(img + rng.normal(0, noise, img.shape), 0, 1).astype(np.float32)
    return img, np.array(boxes, float), np.array(st, int)


def moving_labware(n_objs=3, n_frames=24, px=320, rng=None, cross=True):
    """Short synthetic deck video: rectangular labware sliding across the deck.

    If cross, two objects' paths intersect mid-clip to stress identity
    association (that's where a greedy IoU tracker swaps ids and SAM2's
    appearance memory would not). Returns (frames, tracks) where
    tracks = {frame_idx: [(obj_id, [x1,y1,x2,y2]), ...]} is the ground truth.
    """
    rng = np.random.default_rng(1) if rng is None else rng
    w, h = int(px * 0.16), int(px * 0.12)
    starts = np.linspace(px * 0.12, px * 0.12, n_objs)
    ys = np.linspace(px * 0.25, px * 0.72, n_objs)
    vx = np.full(n_objs, (px * 0.72) / n_frames)
    vy = np.zeros(n_objs)
    if cross and n_objs >= 2:                            # make obj 0 and 1 cross
        ys[1] = ys[0] + (ys[1] - ys[0])
        vy[0] = (ys[1] - ys[0]) / n_frames
        vy[1] = -(ys[1] - ys[0]) / n_frames
    shades = np.linspace(0.45, 0.85, n_objs)

    frames, tracks = [], {}
    for f in range(n_frames):
        img = np.full((px, px), DECK, np.float32)
        m = int(px * 0.05)
        cv2.rectangle(img, (m, m), (px - m, px - m), PLATE, -1)
        per = []
        for oid in range(n_objs):
            cx = starts[oid] + vx[oid] * f
            cy = ys[oid] + vy[oid] * f
            x1, y1, x2, y2 = cx - w, cy - h, cx + w, cy + h
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                          float(shades[oid]), -1)
            per.append((oid, [float(x1), float(y1), float(x2), float(y2)]))
        img = np.clip(img + rng.normal(0, 0.01, img.shape), 0, 1).astype(np.float32)
        frames.append(img)
        tracks[f] = per
    return frames, tracks
