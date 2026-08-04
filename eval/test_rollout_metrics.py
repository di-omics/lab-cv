"""Unit tests for eval.rollout_metrics - runnable two ways:

    python -m pytest eval/test_rollout_metrics.py
    python -m eval.test_rollout_metrics

Every expected value is hand-computed in the comments so the scoring is
auditable, not just green. Several of these tests exist to pin a refusal rather
than a number: `None` where a sentinel integer would have been averaged, a
denominator that includes the episodes that never arrived, and an error vector
that is returned rather than meaned.
"""
from __future__ import annotations

import numpy as np

from eval import rollout_metrics as R


def _approx(a, b, eps=1e-9):
    assert abs(a - b) <= eps, f"expected {b}, got {a}"


def test_terminal_error_is_max_over_fields_in_native_units():
    # episode 0: |1.0-1.2| = 0.2, |3.0-3.05| = 0.05 -> 0.2
    # episode 1: |5.0-4.0| = 1.0, |2.0-2.0| = 0.0   -> 1.0
    err = R.terminal_error([[1.0, 3.0], [5.0, 2.0]], [[1.2, 3.05], [4.0, 2.0]])
    _approx(float(err[0]), 0.2)
    _approx(float(err[1]), 1.0)


def test_terminal_error_accepts_a_scalar_state():
    err = R.terminal_error([1.0, 2.0, 3.0], [1.5, 2.0, 1.0])
    assert np.allclose(err, [0.5, 0.0, 2.0])


def test_in_tolerance_is_joint_and_per_field_normalised():
    # tolerance (0.5 mm, 0.5 deg). Episode 0 misses on mm only, episode 1 misses
    # on deg only, episode 2 is inside both. A joint criterion fails 0 and 1.
    pred = [[0.6, 0.0], [0.0, 0.6], [0.4, 0.4]]
    truth = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    hits = R.in_tolerance(pred, truth, [0.5, 0.5])
    assert list(hits) == [False, False, True]


def test_in_tolerance_boundary_is_inclusive():
    assert bool(R.in_tolerance([[0.5]], [[0.0]], [0.5])[0]) is True
    assert bool(R.in_tolerance([[0.5000001]], [[0.0]], [0.5])[0]) is False


def test_in_tolerance_cannot_buy_millimetres_with_degrees():
    # tight on one field, sloppy on the other: normalising per field then taking
    # the max is what stops a large mm error being offset by a tiny deg error
    assert bool(R.in_tolerance([[2.0, 0.0]], [[0.0, 0.0]], [1.0, 100.0])[0]) is False


def test_zero_tolerance_is_refused():
    try:
        R.in_tolerance([[1.0]], [[1.0]], [0.0])
    except ValueError as exc:
        assert "strictly positive" in str(exc)
    else:
        raise AssertionError("a zero tolerance makes every prediction a miss")


def test_trajectory_rmse():
    # errors 0, 1, 2 over three steps of one field -> sqrt((0+1+4)/3) = sqrt(5/3)
    rmse = R.trajectory_rmse([[0.0, 1.0, 2.0]], [[0.0, 0.0, 0.0]])
    _approx(float(rmse[0]), float(np.sqrt(5.0 / 3.0)))


def test_first_divergence_step():
    # tolerance 1.0. Episode 0 diverges at step 2; episode 1 never does.
    pred = [[0.0, 0.5, 3.0, 3.0], [0.0, 0.5, 0.9, 1.0]]
    truth = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    assert R.first_divergence_step(pred, truth, 1.0) == [2, None]


def test_never_diverged_is_none_not_a_sentinel():
    # an all-inside episode must not report step 0, which is what argmax over an
    # all-False boolean row would have returned
    assert R.first_divergence_step([[0.0, 0.0]], [[0.0, 0.0]], 1.0) == [None]


def test_time_to_tolerance():
    # tolerance 0.5: episode 0 arrives at step 2, episode 1 never arrives
    pred = [[3.0, 1.0, 0.2, 0.1], [3.0, 3.0, 3.0, 3.0]]
    truth = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    assert R.time_to_tolerance(pred, truth, 0.5) == [2, None]


def test_cdf_denominator_includes_the_episodes_that_never_arrived():
    # 4 episodes, arriving at steps 0, 1, 1 and never. By step 1, 3 of 4 = 0.75,
    # and the curve stays at 0.75 - it does not renormalise to 1.0 by dropping
    # the failure, which would turn "converges rarely but fast" into "fast".
    ks, cdf = R.time_to_tolerance_cdf([0, 1, 1, None], n_steps=3)
    assert list(ks) == [0, 1, 2]
    _approx(float(cdf[0]), 0.25)
    _approx(float(cdf[1]), 0.75)
    _approx(float(cdf[2]), 0.75)


def test_cdf_ignores_arrivals_past_the_window_without_dropping_the_episode():
    # arrival at step 9 is outside a 3-step window: it counts in the denominator
    # and never in the numerator, so the curve tops out at 0.5, not 1.0
    _ks, cdf = R.time_to_tolerance_cdf([0, 9], n_steps=3)
    _approx(float(cdf[-1]), 0.5)


def test_cdf_refuses_an_empty_measurement():
    try:
        R.time_to_tolerance_cdf([], n_steps=3)
    except ValueError as exc:
        assert "nothing was measured" in str(exc)
    else:
        raise AssertionError("zero episodes is not a CDF of zero")


def test_shape_mismatch_is_refused():
    try:
        R.terminal_error([[1.0, 2.0]], [[1.0]])
    except ValueError:
        pass
    else:
        raise AssertionError("expected a shape refusal")


def test_no_risk_coverage_function_lives_here():
    # section 6 of the plan: selective-prediction metrics have exactly one home.
    for banned in ("aurc", "risk_coverage", "coverage_at_risk", "selective_risk", "ada"):
        assert not hasattr(R, banned), f"{banned} belongs in plr_lr.learn.selective"
    assert "plr_lr.learn.selective" in R.__doc__


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} rollout metric tests passed.")


if __name__ == "__main__":
    _main()
