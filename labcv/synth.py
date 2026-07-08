"""Clean-room synthetic lab imagery - generated on the fly from a seed.

No real media, no downloads. Each generator returns the image AND the planted
ground truth (well boxes, fill states, or object tracks) so a demo can score
recovery against exactly what was drawn.

    microplate(...)     -> (img, boxes, states)   still frame of a well plate
    moving_labware(...) -> (frames, tracks)        short deck video, moving labware
    tip_view(...)       -> (img, residual_uL)      tip-cam close-up of one well (top-down)
    column_view(...)    -> (img, volume_uL)        tip-cam side view of a liquid column
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


# tip-mounted camera geometry - a borescope on the pipette channel is coaxial
# with the tip, so the well bottom lands in a fixed central disk of the frame.
WELL_FRAC = 0.42     # well-bottom radius as a fraction of the frame half-width
DRY = 0.22           # matte, dry well bottom under coaxial LED
WET = 0.56           # pooled residual liquid (brighter, reflective)
GLINT = 0.97         # specular highlight off a meniscus - the strong 'wet' cue


def tip_view(residual_uL=0.0, px=200, rng=None, max_uL=5.0, glare=False, noise=0.02):
    """Tip-mounted downward close-up of ONE well bottom, for residual-liquid QC.

    A borescope / endoscope mounted on the pipette channel looks straight down
    the tip into a well *after* supernatant removal. A dry bottom is matte and
    uniform; residual ethanol pools into a brighter film and throws a specular
    glint off its meniscus (the reliable wet cue). Pool area grows with the
    leftover volume, so a calibrated reader can both DECIDE dry/residual and
    ESTIMATE how much is left.

    residual_uL  planted leftover volume (ground truth). 0 -> dry.
    max_uL       volume that fills the readable well bottom (calibration span).
    glare        add a confounding specular glint on the dry rim (a scratch or
                 condensation bead) so the classical wet-pixel reader false-alarms
                 -> the learned-segmentation seam (SAM2 film mask / VLM 'dry?').
    Returns (img, residual_uL) - the frame and exactly what was planted.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    img = np.full((px, px), 0.06, np.float32)              # tip shadow / vignette
    c = px // 2
    R = int(px * WELL_FRAC)
    cv2.circle(img, (c, c), R, DRY, -1, cv2.LINE_AA)       # dry well bottom
    cv2.circle(img, (c, c), R, 0.30, 2, cv2.LINE_AA)       # faint bottom rim

    frac = float(np.clip(residual_uL / max_uL, 0.0, 1.0))
    if frac > 0:                                           # residual liquid pool
        rp = max(2, int(0.9 * R * np.sqrt(frac)))          # area proportional to volume
        ox = int(rng.uniform(-0.12, 0.12) * R)
        oy = int(rng.uniform(-0.12, 0.12) * R)
        cv2.circle(img, (c + ox, c + oy), rp, WET, -1, cv2.LINE_AA)
        gr = max(1, int(rp * 0.28))                        # meniscus specular glint
        gx, gy = c + ox + int(rp * 0.35), c + oy - int(rp * 0.35)
        cv2.circle(img, (gx, gy), gr, GLINT, -1, cv2.LINE_AA)

    if glare:                                              # confounder: dry-rim glint
        gx, gy = c + int(R * 0.45), c - int(R * 0.45)      # scratch / condensation bead
        cv2.circle(img, (gx, gy), max(3, int(R * 0.26)), 0.95, -1, cv2.LINE_AA)

    img = np.clip(img + rng.normal(0, noise, img.shape), 0, 1).astype(np.float32)
    return img, float(residual_uL)


# side-view well geometry, as fractions of the frame - the cam is fixed on the
# channel, so the well walls land in the same box every frame (height.py keys on
# this). Liquid fills bottom-up; the meniscus row -> height -> volume.
COL_TOP, COL_BOT = 0.12, 0.90     # well interior top / bottom edge
COL_L, COL_R = 0.34, 0.66         # well interior left / right edge


def column_view(volume_uL=10.0, px=200, rng=None, max_uL=20.0, noise=0.02, bead_pellet=False):
    """Tip-cam SIDE view of a well as a liquid column, for height -> volume readout.

    The channel-mounted camera sees the well from the side (or via an angled
    mirror): liquid fills from the bottom to a height proportional to volume, with
    a bright meniscus band at the surface - the feature the height reader keys on.
    Air above is dark. This is the 'capture liquid heights' view; residual ethanol
    is just a very short column, so ONE reader does both ethanol-removal QC (is the
    column ~0?) and volume QC (how many uL?).

    volume_uL    planted liquid volume (ground truth).
    max_uL       volume that fills the readable well height (calibration span).
    bead_pellet  draw a dark SPRI pellet at the bottom (present after a wash) - it
                 sits below the meniscus and must not be read as liquid.
    Returns (img, volume_uL).
    """
    rng = np.random.default_rng(0) if rng is None else rng
    img = np.full((px, px), 0.10, np.float32)
    yt, yb = int(COL_TOP * px), int(COL_BOT * px)
    xl, xr = int(COL_L * px), int(COL_R * px)
    cv2.rectangle(img, (xl, yt), (xr, yb), 0.14, -1)       # well interior (air)

    f = float(np.clip(volume_uL / max_uL, 0.0, 1.0))
    if f > 0:
        top = int(yb - f * (yb - yt))                      # meniscus row
        cv2.rectangle(img, (xl + 2, top), (xr - 2, yb - 1), 0.50, -1)          # liquid
        cv2.rectangle(img, (xl + 2, top), (xr - 2, min(top + 3, yb - 1)), 0.90, -1)  # meniscus
    if bead_pellet:
        cv2.circle(img, ((xl + xr) // 2, yb - 4), max(3, int((xr - xl) * 0.16)),
                   0.06, -1, cv2.LINE_AA)
    cv2.rectangle(img, (xl, yt), (xr, yb), 0.42, 2)        # well walls (bright rims)

    img = np.clip(img + rng.normal(0, noise, img.shape), 0, 1).astype(np.float32)
    return img, float(volume_uL)
