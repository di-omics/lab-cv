#!/usr/bin/env python3
"""Sample frames from a video at a fixed interval."""
import argparse
import os

import cv2


def extract(video, every_sec, outdir, prefix):
    os.makedirs(outdir, exist_ok=True)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, int(round(fps * every_sec)))
    idx, saved = 0, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            path = os.path.join(outdir, f"{prefix}{saved:04d}.png")
            cv2.imwrite(path, frame)
            saved += 1
        idx += 1

    cap.release()
    print(f"saved {saved} frames to {outdir}/ (every {every_sec}s, step={step})")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video")
    p.add_argument("--every-sec", type=float, default=1.0)
    p.add_argument("--outdir", default="frames")
    p.add_argument("--prefix", default="frame_")
    args = p.parse_args()
    extract(args.video, args.every_sec, args.outdir, args.prefix)


if __name__ == "__main__":
    main()
