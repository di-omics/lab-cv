"""Unit tests for eval.metrics - runnable two ways:

    python -m pytest eval/test_metrics.py     # if pytest is installed
    python -m eval.test_metrics               # plain, no test runner needed

Every expected value is hand-computed in the comments so the metrics are
auditable, not just green.
"""
from __future__ import annotations

import numpy as np

from eval import metrics as M


def _approx(a, b, eps=1e-6):
    assert abs(a - b) <= eps, f"expected {b}, got {a}"


def test_iou():
    # identical boxes -> 1
    _approx(M.iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
    # overlap [5,5,10,10]=25, union 100+100-25=175
    _approx(M.iou([0, 0, 10, 10], [5, 5, 15, 15]), 25.0 / 175.0)
    # disjoint -> 0
    _approx(M.iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)


def test_ap_perfect():
    gt = [[0, 0, 10, 10], [20, 20, 30, 30]]
    pred = [[0, 0, 10, 10], [20, 20, 30, 30]]
    _approx(M.average_precision(gt, pred, [0.9, 0.8], 0.5), 1.0)


def test_ap_fp_after_recall_is_still_one():
    # 1 GT, a perfect high-score TP then a low-score FP -> recall hits 1 at
    # precision 1 before the FP, so AP == 1.0
    gt = [[0, 0, 10, 10]]
    pred = [[0, 0, 10, 10], [100, 100, 110, 110]]
    _approx(M.average_precision(gt, pred, [0.9, 0.3], 0.5), 1.0)


def test_ap_fp_ranked_first_drops_ap():
    # same boxes, but the FP outranks the TP -> precision is 0.5 everywhere
    gt = [[0, 0, 10, 10]]
    pred = [[0, 0, 10, 10], [100, 100, 110, 110]]
    ap = M.average_precision(gt, pred, [0.3, 0.9], 0.5)
    _approx(ap, 0.5)


def test_missed_detection_halves_recall():
    gt = [[0, 0, 10, 10], [20, 20, 30, 30]]
    pred = [[0, 0, 10, 10]]                       # second GT missed
    r = M.precision_recall(gt, pred, [0.9], 0.5)
    _approx(r["recall"], 0.5)
    _approx(r["precision"], 1.0)
    assert r["fn"] == 1


def test_classification_report():
    y_true = ["empty", "filled", "filled", "empty"]
    y_pred = ["empty", "filled", "empty", "empty"]
    rep = M.classification_report(y_true, y_pred, ["empty", "filled"])
    _approx(rep["accuracy"], 0.75)                 # 3/4 correct
    _approx(rep["per_class"]["filled"]["recall"], 0.5)   # 1 of 2 filled found


def test_id_switch():
    # one GT track; predicted id 7 for two frames then swaps to 9 -> 1 switch
    gt = {0: [(1, [0, 0, 10, 10])], 1: [(1, [1, 0, 11, 10])], 2: [(1, [2, 0, 12, 10])]}
    pred = {0: [(7, [0, 0, 10, 10])], 1: [(7, [1, 0, 11, 10])], 2: [(9, [2, 0, 12, 10])]}
    assert M.count_id_switches(gt, pred, 0.5) == 1
    # stable ids -> 0 switches
    pred_stable = {f: [(7, b) for _, b in v] for f, v in gt.items()}
    assert M.count_id_switches(gt, pred_stable, 0.5) == 0


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} metric tests passed.")


if __name__ == "__main__":
    _main()
