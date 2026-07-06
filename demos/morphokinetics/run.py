#!/usr/bin/env python3
"""
morphokinetics - clean-room synthetic time-lapse embryo morphokinetics: plant a
division schedule, recover it blind, score the recovery. Same discipline as the
omics-demos: known ground truth in, honest error out.

Run:
    python3 demos/morphokinetics/run.py              # baseline recovers the schedule
    python3 demos/morphokinetics/run.py --crowding   # cells pack -> baseline breaks

Deps: numpy only. No hardware, no downloads, no real embryo data.

WHY THIS SHAPE
--------------
The pipeline is split so a real model drops into one seam:

    frames --> detect_cell_count()  <-- SWAP IN RF-DETR / RT-DETRv4 here
           --> (track identities)   <-- SWAP IN SAM2 video memory here
           --> extract_events()     <-- pure function, timing logic
           --> score()              <-- plant-and-recover, gates the claim

Both seams are built out concretely on synthetic data in the sibling demos:
detection with a classical -> RF-DETR swap in demos/well_detection, and
SAM2-shaped identity tracking in demos/roi_tracking. The classical counter here
(blur + threshold + connected-components) is the baseline on purpose. It nails
the schedule when blastomeres are separable; turn on --crowding and it
UNDERCOUNTS once they pack - that failure is the argument for a learned detector
plus SAM2 identity memory. The eval harness never changes, only the model does.

Extension seam: the same plant-and-recover harness accepts tPNf (pronuclei
fading) and tSB (start of blastulation) once frames model those appearances -
add them to `schedule` with detectors and the scoring is unchanged.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

# ----------------------------------------------------------------------------
# Config - every number lives here.
# ----------------------------------------------------------------------------


@dataclass
class Config:
    frame_px: int = 180
    interval_min: float = 15.0        # time-lapse cadence (EmbryoScope ~ 5-20 min)
    total_hours: float = 44.0         # through ~8-cell
    seed: int = 0
    noise: float = 0.05               # gaussian sensor noise
    crowding: bool = False            # pack late cells -> classical counter fails

    # planted morphokinetic schedule (hours post-insemination) = ground truth.
    # key = cell count n, value = time-to-n-cells t_n.
    schedule: dict = field(default_factory=lambda: {
        2: 25.0,   # t2
        3: 27.5,   # t3   cc2 = t3 - t2 = 2.5h  (normal; <5h flags direct cleavage)
        4: 28.5,   # t4   s2  = t4 - t3 = 1.0h  (synchrony)
        5: 34.0,   # t5
        6: 36.0,   # t6
        7: 38.5,   # t7
        8: 40.0,   # t8
    })


# ----------------------------------------------------------------------------
# 1. Synthetic time-lapse - n gaussian "blastomeres" packed in a zona.
# ----------------------------------------------------------------------------


def true_count_at(t_h: float, cfg: Config) -> int:
    n = 1
    for k, t_k in sorted(cfg.schedule.items()):
        if t_h >= t_k:
            n = k
    return n


def make_frame(n_cells: int, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    px = cfg.frame_px
    yy, xx = np.mgrid[0:px, 0:px].astype(np.float32)
    img = np.zeros((px, px), np.float32)

    cx = cy = px / 2.0
    zona_r = px * 0.40
    spacing = zona_r / np.sqrt(n_cells)
    # separable when not crowded (radius << spacing); merged when crowded.
    cell_r = max(4.0, spacing * (0.52 if cfg.crowding else 0.26))
    golden = np.pi * (3.0 - np.sqrt(5.0))       # phyllotaxis -> even packing

    for i in range(n_cells):
        if n_cells == 1:
            rad, ang = 0.0, 0.0
        else:
            rad = zona_r * 0.80 * np.sqrt((i + 0.5) / n_cells)
            ang = i * golden
        px_i = cx + rad * np.cos(ang) + rng.uniform(-1.5, 1.5)
        py_i = cy + rad * np.sin(ang) + rng.uniform(-1.5, 1.5)
        blob = np.exp(-(((xx - px_i) ** 2 + (yy - py_i) ** 2) / (2 * cell_r ** 2)))
        img = np.maximum(img, blob)             # cells occlude, don't sum

    img += rng.normal(0, cfg.noise, img.shape)
    zona_mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (zona_r * 1.15) ** 2
    img *= zona_mask
    return np.clip(img, 0, 1)


# ----------------------------------------------------------------------------
# 2. Perception - the seam. Classical baseline; RF-DETR / SAM2 go right here.
# ----------------------------------------------------------------------------


def box_blur(img: np.ndarray, k: int = 5) -> np.ndarray:
    """k x k moving-average denoise, numpy-only (kills sensor speckle)."""
    pad = k // 2
    p = np.pad(img, pad, mode="edge")
    acc = np.zeros_like(img)
    h, w = img.shape
    for dy in range(k):
        for dx in range(k):
            acc += p[dy:dy + h, dx:dx + w]
    return acc / (k * k)


def _label_components(mask: np.ndarray, min_area: int = 12) -> int:
    """4-connectivity connected-components via BFS. numpy-only, no scipy."""
    seen = np.zeros_like(mask, bool)
    h, w = mask.shape
    count = 0
    stack: list[tuple[int, int]] = []
    for sy in range(h):
        for sx in range(w):
            if mask[sy, sx] and not seen[sy, sx]:
                stack.append((sy, sx))
                seen[sy, sx] = True
                area = 0
                while stack:
                    y, x = stack.pop()
                    area += 1
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                if area >= min_area:
                    count += 1
    return max(count, 1)


def detect_cell_count(frame: np.ndarray, cfg: Config) -> int:
    """
    BASELINE detector: denoise -> threshold -> count connected components.

    >>> SWAP SEAM <<<
    Production: run RF-DETR / RT-DETRv4 instance detection and return the number
    of 'blastomere' boxes. Then SAM2's video predictor propagates each box across
    frames so identities survive touching/occluding cells (fixes the crowding
    undercount this baseline shows):

        # from sam2.build_sam import build_sam2_video_predictor
        # state = predictor.init_state(frames)
        # for box in first_frame_boxes:            # RF-DETR boxes on frame 0
        #     predictor.add_new_points_or_box(state, box=box, frame_idx=0)
        # for fidx, obj_ids, masks in predictor.propagate_in_video(state):
        #     count[fidx] = len(obj_ids)           # identity-stable count
    """
    b = box_blur(frame, k=5)
    mask = b > 0.42 * b.max()
    return _label_components(mask, min_area=14)


# ----------------------------------------------------------------------------
# 3. Event extraction - pure timing logic over the recovered count series.
# ----------------------------------------------------------------------------


def extract_events(times_h: np.ndarray, counts: np.ndarray, cfg: Config) -> dict:
    """First time recovered count reaches n (debounced 1 frame) -> t_n."""
    events: dict[int, float] = {}
    for n in sorted(cfg.schedule):
        hit = np.where(counts >= n)[0]
        for idx in hit:
            if idx + 1 < len(counts) and counts[idx + 1] >= n:
                events[n] = float(times_h[idx])
                break
    return events


def derived_kinetics(events: dict) -> dict:
    d = {}
    if 2 in events and 3 in events:
        d["cc2 (t3-t2)"] = round(events[3] - events[2], 2)
    if 3 in events and 4 in events:
        d["s2 (t4-t3)"] = round(events[4] - events[3], 2)
    if 3 in events and 5 in events:
        d["cc3 (t5-t3)"] = round(events[5] - events[3], 2)
    return d


# ----------------------------------------------------------------------------
# 4. Score - plant vs recover. The part that earns trust.
# ----------------------------------------------------------------------------


def score(events: dict, cfg: Config) -> tuple[float, int]:
    abs_err = []
    print(f"{'event':>7}  {'planted(h)':>11}  {'recovered(h)':>13}  {'err(min)':>9}")
    print("  " + "-" * 46)
    for n, t_true in sorted(cfg.schedule.items()):
        if n in events:
            err_min = abs(events[n] - t_true) * 60.0
            abs_err.append(err_min)
            flag = "" if err_min <= cfg.interval_min else "  <-- off"
            print(f"    t{n:<2}  {t_true:>11.2f}  {events[n]:>13.2f}  {err_min:>9.1f}{flag}")
        else:
            print(f"    t{n:<2}  {t_true:>11.2f}  {'MISSED':>13}  {'--':>9}")
    mae = float(np.mean(abs_err)) if abs_err else float("nan")
    return mae, len(abs_err)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------


def run(cfg: Config) -> None:
    rng = np.random.default_rng(cfg.seed)
    times = np.arange(0, cfg.total_hours, cfg.interval_min / 60.0)

    counts = np.empty(len(times), int)
    for i, t in enumerate(times):
        frame = make_frame(true_count_at(t, cfg), cfg, rng)
        counts[i] = detect_cell_count(frame, cfg)

    events = extract_events(times, counts, cfg)

    print("\nEMBRYO MORPHOKINETICS - clean-room plant-and-recover")
    print(f"  frames={len(times)}  cadence={cfg.interval_min:.0f}min  "
          f"span={cfg.total_hours:.0f}h  crowding={'ON' if cfg.crowding else 'off'}\n")
    mae, hit = score(events, cfg)

    print("\n  derived kinetics:", derived_kinetics(events) or "(insufficient events)")
    print(f"\n  recovered {hit}/{len(cfg.schedule)} events   "
          f"MAE = {mae:.1f} min   (pass if <= {cfg.interval_min:.0f} min)\n")

    if cfg.crowding and hit < len(cfg.schedule):
        print("  ^ the baseline undercounts once cells pack. THIS is where RF-DETR")
        print("    (learned blastomere prior) + SAM2 (identity across frames) earn")
        print("    their keep. The eval harness above doesn't change - the model does.\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise", type=float, default=0.05)
    p.add_argument("--interval-min", type=float, default=15.0)
    p.add_argument("--total-hours", type=float, default=44.0)
    p.add_argument("--crowding", action="store_true",
                   help="pack late-stage cells so the classical counter fails")
    a = p.parse_args()
    run(Config(seed=a.seed, noise=a.noise, interval_min=a.interval_min,
               total_hours=a.total_hours, crowding=a.crowding))


if __name__ == "__main__":
    main()
