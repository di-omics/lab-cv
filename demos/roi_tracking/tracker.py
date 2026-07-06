"""Multi-object tracking behind a SAM2-shaped interface.

The interface deliberately mirrors SAM2's video predictor so the baseline and
the real model are drop-in swappable:

    tk = IoUTracker()                       # or SAM2Tracker() (optional, guarded)
    tk.init_state(frames, detections)
    tk.add_new_box(frame_idx=0, obj_id=i, box=b)
    for f, obj_ids, boxes in tk.propagate_in_video():
        ...

IoUTracker is a classical greedy IoU-association baseline: it has no appearance
memory, so when two objects cross it can swap their ids. That id-switch is the
seam where SAM2's per-object memory earns its keep - same interface, fewer
switches.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from eval.metrics import iou  # noqa: E402


class IoUTracker:
    def __init__(self, iou_thr=0.15):
        self.iou_thr = iou_thr
        self.dets = {}
        self.tracks = {}

    def init_state(self, frames, detections):
        """detections: {frame_idx: [box, ...]}  (boxes only - no identities)."""
        self.frames = frames
        self.dets = {int(f): [list(map(float, b)) for b in bxs]
                     for f, bxs in detections.items()}
        self.tracks = {}

    def add_new_box(self, frame_idx, obj_id, box):
        # seed a track identity on the prompt frame (expected frame 0)
        self.tracks[int(obj_id)] = list(map(float, box))

    def propagate_in_video(self):
        frames = sorted(self.dets)
        first = frames[0]
        yield first, list(self.tracks.keys()), [self.tracks[i][:] for i in self.tracks]
        next_id = (max(self.tracks) + 1) if self.tracks else 0

        for f in frames[1:]:
            dets = self.dets[f]
            used, assigned = set(), {}
            # greedy: each existing track grabs its best-IoU free detection
            for oid in sorted(self.tracks):
                last = self.tracks[oid]
                best_j, best = -1, self.iou_thr
                for j, d in enumerate(dets):
                    if j in used:
                        continue
                    v = iou(last, d)
                    if v >= best:
                        best, best_j = v, j
                if best_j >= 0:
                    assigned[oid] = dets[best_j]
                    self.tracks[oid] = list(dets[best_j])
                    used.add(best_j)
            for j, d in enumerate(dets):        # unmatched detections -> new ids
                if j not in used:
                    assigned[next_id] = d
                    self.tracks[next_id] = list(d)
                    next_id += 1
            ids = list(assigned.keys())
            yield f, ids, [assigned[i] for i in ids]


class SAM2Tracker:  # pragma: no cover - optional real-model seam
    """Same interface, backed by SAM2's video predictor. Optional install."""

    def __init__(self, checkpoint=None):
        try:
            from sam2.build_sam import build_sam2_video_predictor  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "SAM2 path needs `pip install -r requirements-models.txt` "
                "(sam2 + torch). The IoUTracker baseline runs with no extra deps."
            ) from e
        raise NotImplementedError(
            "Seam stub: init_state -> predictor.init_state(frames); add_new_box -> "
            "predictor.add_new_points_or_box(...); propagate_in_video -> yield "
            "predictor.propagate_in_video() masks reduced to boxes. Identity is "
            "carried by SAM2 memory, so crossings don't switch ids."
        )
