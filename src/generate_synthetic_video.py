#!/usr/bin/env python3
"""Generate a short synthetic MP4 simulating a lab deck camera.

Creates a static grey background with four ROI regions that brighten on a
known schedule, simulating tip pickup / plate motion. Writes the activity
schedule to a sidecar JSON so downstream validation can score detection.
"""
import argparse
import json
import os

import cv2
import numpy as np

WIDTH, HEIGHT, FPS, DURATION = 400, 400, 10, 10  # 10 s at 10 fps = 100 frames

# ROI positions matching example_roi_config.json defaults
ROIS = [
    {"name": "tip-rack",  "x": 50,  "y": 50,  "w": 100, "h": 100},
    {"name": "plate-A",   "x": 200, "y": 50,  "w": 100, "h": 100},
    {"name": "plate-B",   "x": 50,  "y": 200, "w": 100, "h": 100},
    {"name": "reservoir", "x": 200, "y": 200, "w": 100, "h": 100},
]

# Activity schedule: each ROI is "active" (brightens) during [start, end) seconds
SCHEDULE = {
    "tip-rack":  (1.0, 3.0),
    "plate-A":   (3.0, 5.0),
    "plate-B":   (5.0, 7.0),
    "reservoir": (7.0, 9.0),
}

BG_LEVEL = 60       # baseline grey
ACTIVE_BOOST = 120  # added brightness when ROI is active


def generate(outpath, schedule_path):
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(outpath, fourcc, FPS, (WIDTH, HEIGHT))

    total_frames = FPS * DURATION
    for i in range(total_frames):
        t = i / FPS
        frame = np.full((HEIGHT, WIDTH, 3), BG_LEVEL, dtype=np.uint8)
        for roi in ROIS:
            start, end = SCHEDULE[roi["name"]]
            if start <= t < end:
                x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
                frame[y:y+h, x:x+w] = np.clip(
                    frame[y:y+h, x:x+w].astype(np.int16) + ACTIVE_BOOST, 0, 255
                ).astype(np.uint8)
        writer.write(frame)

    writer.release()
    print(f"wrote {total_frames} frames -> {outpath}")

    # Save ground-truth schedule for validation
    os.makedirs(os.path.dirname(schedule_path) or ".", exist_ok=True)
    with open(schedule_path, "w") as f:
        json.dump(SCHEDULE, f, indent=2)
    print(f"wrote activity schedule -> {schedule_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="videos/synthetic_deck.mp4")
    p.add_argument("--schedule", default="output/ground_truth_schedule.json")
    args = p.parse_args()
    generate(args.out, args.schedule)


if __name__ == "__main__":
    main()
