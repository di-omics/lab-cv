#!/usr/bin/env python3
"""Write an output directory with latest-frame copy and JSON activity summary."""
import argparse
import glob
import json
import os
import shutil

import pandas as pd


def summarize(framedir, csv_path, outdir):
    os.makedirs(outdir, exist_ok=True)

    # Copy latest frame
    frames = sorted(glob.glob(os.path.join(framedir, "*.png")))
    if frames:
        shutil.copy2(frames[-1], os.path.join(outdir, "latest_frame.png"))

    # Read motion CSV and summarize
    df = pd.read_csv(csv_path)
    summary = {}
    for roi_name, grp in df.groupby("roi"):
        peak_diff = grp["absdiff_mean"].max()
        active_frames = int((grp["absdiff_mean"] > 5.0).sum())
        summary[roi_name] = {
            "peak_absdiff_mean": round(float(peak_diff), 2),
            "active_frames": active_frames,
            "total_frames": len(grp),
            "activity_detected": active_frames > 0,
        }

    out_path = os.path.join(outdir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"dashboard summary -> {out_path}")

    detected = sum(1 for v in summary.values() if v["activity_detected"])
    print(f"activity detected in {detected}/{len(summary)} ROIs")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--framedir", default="frames")
    p.add_argument("--csv", default="output/roi_motion.csv")
    p.add_argument("--outdir", default="output")
    args = p.parse_args()
    summarize(args.framedir, args.csv, args.outdir)


if __name__ == "__main__":
    main()
