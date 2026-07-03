#!/usr/bin/env python3
"""QC figure: per-ROI motion (brightness + absdiff) over time.

Uses the shared omics_style for consistent visuals. Optionally validates
detected motion peaks against a ground-truth activity schedule.
"""
import argparse
import json
import os
import sys

import pandas as pd

# Add repo root to path so omics_style is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import omics_style as S
import matplotlib.pyplot as plt

S.apply()


def load_schedule(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def validate(df, schedule):
    """Check that each planted ROI shows detected motion (absdiff_mean > 5)."""
    detected = 0
    for roi_name in schedule:
        grp = df[df["roi"] == roi_name]
        if grp["absdiff_mean"].max() > 5.0:
            detected += 1
    total = len(schedule)
    print(f"validation: detected activity in {detected}/{total} planted ROIs")
    return detected, total


def plot(csv_path, outpath, schedule_path=None):
    df = pd.read_csv(csv_path)
    roi_names = df["roi"].unique()
    colors_fill = list(S.PALETTE.values())
    colors_line = list(S.OUTLINE.values())

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    for i, roi in enumerate(roi_names):
        grp = df[df["roi"] == roi]
        ci = i % len(colors_fill)
        axes[0].plot(grp["frame"], grp["brightness"],
                     color=colors_line[ci], label=roi)
        axes[1].plot(grp["frame"], grp["absdiff_mean"],
                     color=colors_line[ci], label=roi)

    axes[0].set_ylabel("Mean brightness")
    axes[0].set_title("ROI brightness over time")
    axes[0].legend()
    axes[1].set_ylabel("Absdiff (mean)")
    axes[1].set_xlabel("Frame")
    axes[1].set_title("ROI frame-to-frame motion")
    axes[1].legend()

    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    fig.savefig(outpath)
    plt.close(fig)
    print(f"QC plot -> {outpath}")

    # Validation against ground truth if available
    schedule = load_schedule(schedule_path)
    if schedule:
        validate(df, schedule)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default="output/roi_motion.csv")
    p.add_argument("--out", default="output/roi_motion_qc.png")
    p.add_argument("--schedule", default="output/ground_truth_schedule.json",
                   help="Ground-truth schedule for validation (optional)")
    args = p.parse_args()
    plot(args.csv, args.out, args.schedule)


if __name__ == "__main__":
    main()
