#!/usr/bin/env python3
"""Tile sampled frames into a single review image."""
import argparse
import glob
import os

import cv2
import numpy as np


def make_sheet(framedir, out, cols, thumb_width):
    paths = sorted(glob.glob(os.path.join(framedir, "*.png")))
    if not paths:
        raise SystemExit(f"no PNGs in {framedir}")

    thumbs = []
    for p in paths:
        img = cv2.imread(p)
        h, w = img.shape[:2]
        scale = thumb_width / w
        thumbs.append(cv2.resize(img, (thumb_width, int(h * scale))))

    th = thumbs[0].shape[0]
    rows = (len(thumbs) + cols - 1) // cols
    # pad to fill last row
    while len(thumbs) % cols:
        thumbs.append(np.zeros_like(thumbs[0]))

    grid = np.vstack([
        np.hstack(thumbs[r * cols:(r + 1) * cols])
        for r in range(rows)
    ])

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    cv2.imwrite(out, grid)
    print(f"contact sheet ({rows}x{cols}) -> {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--framedir", default="frames")
    p.add_argument("--out", default="output/contact_sheet.png")
    p.add_argument("--cols", type=int, default=5)
    p.add_argument("--thumb-width", type=int, default=200)
    args = p.parse_args()
    make_sheet(args.framedir, args.out, args.cols, args.thumb_width)


if __name__ == "__main__":
    main()
