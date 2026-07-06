"""Shared CV metrics - pure numpy, no sklearn, no pycocotools.

Every number a demo reports comes from here, so the scoring is one small,
readable, unit-tested file (see test_metrics.py) rather than a black box.

Boxes are [x1, y1, x2, y2] in pixels. Detections carry a confidence score.

    detection      iou, iou_matrix, match, precision_recall,
                   pr_curve, average_precision (AP@0.5), ap_range (AP@[.5:.95])
    classification confusion_matrix, classification_report
    tracking       count_id_switches (MOTA-style)
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def iou(a, b) -> float:
    """Intersection-over-union of two [x1,y1,x2,y2] boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def iou_matrix(A, B) -> np.ndarray:
    """(len A, len B) matrix of pairwise IoUs."""
    A = np.asarray(A, float).reshape(-1, 4)
    B = np.asarray(B, float).reshape(-1, 4)
    M = np.zeros((len(A), len(B)), float)
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            M[i, j] = iou(a, b)
    return M


# ---------------------------------------------------------------------------
# detection matching
# ---------------------------------------------------------------------------
def match(gt, pred, scores, iou_thr=0.5):
    """Greedy match: predictions high->low score, each claims the best still-
    free GT box with IoU >= iou_thr. Returns (tp, fp, n_gt, order) where tp/fp
    are boolean arrays aligned to the ORIGINAL pred order."""
    gt = np.asarray(gt, float).reshape(-1, 4)
    pred = np.asarray(pred, float).reshape(-1, 4)
    scores = np.asarray(scores, float).reshape(-1)
    order = np.argsort(-scores, kind="stable")
    used = np.zeros(len(gt), bool)
    tp = np.zeros(len(pred), bool)
    for idx in order:
        if len(gt) == 0:
            break
        ious = np.array([iou(pred[idx], g) for g in gt])
        ious[used] = -1.0
        best = int(np.argmax(ious))
        if ious[best] >= iou_thr:
            tp[idx] = True
            used[best] = True
    return tp, ~tp, len(gt), order


def precision_recall(gt, pred, scores, iou_thr=0.5, score_thr=0.0):
    """Precision/recall at one operating point (keep preds with score>=score_thr)."""
    pred = np.asarray(pred, float).reshape(-1, 4)
    scores = np.asarray(scores, float).reshape(-1)
    n_gt = len(np.asarray(gt, float).reshape(-1, 4))
    keep = scores >= score_thr
    pk, sk = pred[keep], scores[keep]
    if len(pk) == 0:
        return {"precision": 0.0, "recall": 0.0, "tp": 0, "fp": 0, "fn": n_gt}
    tp, fp, n_gt, _ = match(gt, pk, sk, iou_thr)
    tp_n, fp_n = int(tp.sum()), int(fp.sum())
    fn = n_gt - tp_n
    prec = tp_n / max(tp_n + fp_n, 1)
    rec = tp_n / max(n_gt, 1)
    return {"precision": prec, "recall": rec, "tp": tp_n, "fp": fp_n, "fn": fn}


def pr_curve(gt, pred, scores, iou_thr=0.5):
    """Recall/precision arrays swept over the score threshold (high->low)."""
    pred = np.asarray(pred, float).reshape(-1, 4)
    scores = np.asarray(scores, float).reshape(-1)
    n_gt = len(np.asarray(gt, float).reshape(-1, 4))
    if len(pred) == 0:
        return np.array([0.0]), np.array([1.0]), np.array([])
    tp, fp, n_gt, order = match(gt, pred, scores, iou_thr)
    tp_c = np.cumsum(tp[order].astype(float))
    fp_c = np.cumsum(fp[order].astype(float))
    recall = tp_c / max(n_gt, 1)
    precision = tp_c / np.maximum(tp_c + fp_c, 1e-9)
    return recall, precision, scores[order]


def average_precision(gt, pred, scores, iou_thr=0.5) -> float:
    """AP at a single IoU threshold, COCO 101-point interpolation."""
    n_gt = len(np.asarray(gt, float).reshape(-1, 4))
    pred = np.asarray(pred, float).reshape(-1, 4)
    if n_gt == 0:
        return 1.0 if len(pred) == 0 else 0.0
    if len(pred) == 0:
        return 0.0
    recall, precision, _ = pr_curve(gt, pred, scores, iou_thr)
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 101):
        mask = recall >= t
        ap += float(precision[mask].max()) if mask.any() else 0.0
    return ap / 101.0


def ap_range(gt, pred, scores, thrs=None):
    """Mean AP over IoU thresholds - COCO AP@[.5:.95]. Returns (mAP, per-thr)."""
    if thrs is None:
        thrs = np.round(np.arange(0.5, 1.0, 0.05), 2)
    per = {float(t): average_precision(gt, pred, scores, float(t)) for t in thrs}
    return float(np.mean(list(per.values()))), per


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------
def confusion_matrix(y_true, y_pred, labels) -> np.ndarray:
    """rows = true label, cols = predicted label, in `labels` order."""
    idx = {l: i for i, l in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)), int)
    for t, p in zip(y_true, y_pred):
        M[idx[t], idx[p]] += 1
    return M


def classification_report(y_true, y_pred, labels) -> dict:
    """Accuracy plus per-class precision/recall/support and the confusion matrix."""
    M = confusion_matrix(y_true, y_pred, labels)
    acc = float(np.trace(M) / max(M.sum(), 1))
    per = {}
    for i, l in enumerate(labels):
        tp = M[i, i]
        fp = M[:, i].sum() - tp
        fn = M[i, :].sum() - tp
        per[l] = {
            "precision": float(tp / max(tp + fp, 1)),
            "recall": float(tp / max(tp + fn, 1)),
            "support": int(M[i, :].sum()),
        }
    return {"accuracy": acc, "per_class": per, "confusion": M}


# ---------------------------------------------------------------------------
# tracking
# ---------------------------------------------------------------------------
def count_id_switches(gt_by_frame, pred_by_frame, iou_thr=0.5) -> int:
    """MOTA-style identity switches.

    gt_by_frame / pred_by_frame: {frame_idx: [(obj_id, box), ...]}.
    For each GT track, follow which predicted id it best-matches (IoU>=thr)
    over time; a switch is counted whenever that predicted id changes between
    consecutive matched frames. Perfect tracking -> 0.
    """
    last_pred_for_gt: dict = {}
    switches = 0
    for f in sorted(gt_by_frame):
        preds = pred_by_frame.get(f, [])
        for gid, gbox in gt_by_frame.get(f, []):
            best_pid, best_iou = None, iou_thr
            for pid, pbox in preds:
                v = iou(gbox, pbox)
                if v >= best_iou:
                    best_iou, best_pid = v, pid
            if best_pid is not None:
                prev = last_pred_for_gt.get(gid)
                if prev is not None and prev != best_pid:
                    switches += 1
                last_pred_for_gt[gid] = best_pid
    return switches
