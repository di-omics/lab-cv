#!/usr/bin/env python3
"""Compute per-frame brightness and frame-to-frame absdiff for each ROI.

For each ROI defined in the config, outputs per-frame:
  - mean brightness (greyscale mean of the ROI crop)
  - absdiff_mean / absdiff_p95: mean and 95th-percentile of the absolute
    difference between the current and previous frame's ROI crop.
  - First frame per ROI has absdiff = 0 (no previous frame).

Writes results to a CSV.
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
import pandas as pd


def analyze(framedir, roi_config, out_csv):
    with open(roi_config) as f:
        cfg = json.load(f)
    rois = cfg["rois"]

    paths = sorted(glob.glob(os.path.join(framedir, "*.png")))
    if not paths:
        raise SystemExit(f"no PNGs in {framedir}")

    prev_crops = {r["name"]: None for r in rois}
    rows = []

    for frame_idx, path in enumerate(paths):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        for roi in rois:
            x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
            crop = img[y:y+h, x:x+w].astype(np.float32)
            brightness = float(crop.mean())

            prev = prev_crops[roi["name"]]
            if prev is None:
                absdiff_mean, absdiff_p95 = 0.0, 0.0
            else:
                diff = np.abs(crop - prev)
                absdiff_mean = float(diff.mean())
                absdiff_p95 = float(np.percentile(diff, 95))

            prev_crops[roi["name"]] = crop
            rows.append({
                "frame": frame_idx,
                "roi": roi["name"],
                "brightness": round(brightness, 2),
                "absdiff_mean": round(absdiff_mean, 2),
                "absdiff_p95": round(absdiff_p95, 2),
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"wrote {len(df)} rows ({len(paths)} frames x {len(rois)} ROIs) -> {out_csv}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--framedir", default="frames")
    p.add_argument("--roi-config", default="example_roi_config.json")
    p.add_argument("--out", default="output/roi_motion.csv")
    args = p.parse_args()
    analyze(args.framedir, args.roi_config, args.out)


if __name__ == "__main__":
    main()
