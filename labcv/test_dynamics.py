"""Unit tests for labcv.dynamics - runnable two ways:

    python -m pytest labcv/test_dynamics.py
    python -m labcv.test_dynamics

The load-bearing test here is `test_analytic_reproduces_the_inline_arithmetic`.
The two constants this module names were previously applied inline in
`demos/pipette_cam/run.py`, and the whole point of naming them is that the demo
keeps printing the same numbers afterwards. That test compares against the old
expression itself, in float, with exact equality - not almost-equal, because
almost-equal is how a demo's published table drifts a digit at a time.
"""
from __future__ import annotations

import numpy as np

from labcv.bridge import Action
from labcv.dynamics import (
    AnalyticTransition,
    DispenseTransition,
    InjectedTransition,
    PlantedTransition,
    Transition,
    TransitionAdapterError,
    apply_residual,
)


def test_analytic_reproduces_the_inline_arithmetic():
    # The exact expressions run.py used before the transition was named:
    #     res *= cfg.rewash_leave    (0.12)
    #     res *= cfg.drydown         (0.35)
    t = AnalyticTransition()
    assert t.rewash_leave == 0.12 and t.drydown == 0.35
    for res in (3.1014, 0.6031, 2.8951, 1.8834, 0.0, 5.0):
        assert t.step(res, Action.REWASH) == res * 0.12
        assert t.step(res, Action.EXTEND_DRY) == res * 0.35


def test_analytic_leaves_non_acting_actions_alone():
    t = AnalyticTransition()
    for action in (Action.PROCEED, Action.HALT, Action.TOP_UP, Action.HOLD):
        assert t.step(2.75, action) == 2.75


def test_analytic_satisfies_the_protocol():
    assert isinstance(AnalyticTransition(), Transition)


def test_apply_residual_is_array_safe():
    # the oracle sweeps thousands of candidate latents at once
    out = apply_residual(np.array([1.0, 2.0, 4.0]), Action.EXTEND_DRY.value, {"drydown": 0.5})
    assert np.allclose(out, [0.5, 1.0, 2.0])


def test_dispense_does_not_land_on_target():
    # gain 0.94 on a commanded shortfall of 10.0 - 7.0 = 3.0 -> 2.82 delivered,
    # so a well that was at 7.0 lands at 9.82 plus noise, never at exactly 10.0
    t = DispenseTransition(10.0, noise_uL=0.0)
    assert t.step(7.0, Action.TOP_UP) == 7.0 + 0.94 * 3.0
    assert t.step(7.0, Action.TOP_UP) != 10.0


def test_dispense_commands_against_the_measurement_not_the_truth():
    # the camera read 7.4 while the well truly held 7.0; the robot commands
    # 10.0 - 7.4 = 2.6, delivers 0.94 * 2.6 = 2.444, and lands at 9.444
    t = DispenseTransition(10.0, noise_uL=0.0)
    assert abs(t.step(7.0, Action.TOP_UP, measured=7.4) - (7.0 + 0.94 * 2.6)) < 1e-12


def test_dispense_ignores_actions_that_move_nothing():
    t = DispenseTransition(10.0, noise_uL=0.0)
    assert t.step(7.0, Action.PROCEED) == 7.0
    assert t.step(7.0, Action.HALT) == 7.0


def test_planted_is_seed_stable_and_inside_its_declared_ranges():
    a, b = PlantedTransition(42), PlantedTransition(42)
    assert a.coefficients == b.coefficients
    assert PlantedTransition(43).coefficients != a.coefficients
    lo, hi = PlantedTransition.DRYDOWN_RANGE
    assert lo <= a.drydown <= hi
    lo, hi = PlantedTransition.LEAVE_RANGE
    assert lo <= a.rewash_leave <= hi


def test_planted_hides_the_carryover_from_its_published_coefficients():
    p = PlantedTransition(7, hidden_carryover_uL=1.4)
    assert "carryover" not in " ".join(p.coefficients)      # never published as a coefficient
    nominal = p.step(2.0, Action.EXTEND_DRY)
    hidden = p.step(2.0, Action.EXTEND_DRY, apply_hidden=True)
    assert abs(hidden - (nominal + 1.4)) < 1e-12
    assert nominal == 2.0 * p.drydown


def test_injected_refuses_without_a_predictor():
    try:
        InjectedTransition().step(1.0, Action.REWASH)
    except NotImplementedError as exc:
        assert "predictor(latent_uL, action_name)" in str(exc)
    else:
        raise AssertionError("a transition adapter with no model must refuse, not pass through")


def test_injected_mentions_a_checkpoint_when_one_was_recorded():
    try:
        InjectedTransition(checkpoint="some.ckpt").step(1.0, Action.REWASH)
    except NotImplementedError as exc:
        assert "Checkpoint metadata was recorded" in str(exc)
    else:
        raise AssertionError("expected a refusal")


def test_injected_refuses_an_awaitable():
    async def predictor(latent, action):     # noqa: RUF029 - the point is that it is async
        return 0.0

    try:
        InjectedTransition(predictor).step(1.0, Action.REWASH)
    except TransitionAdapterError as exc:
        assert "synchronous" in str(exc)
    else:
        raise AssertionError("an un-awaited coroutine is truthy and must not be accepted")


def test_injected_refuses_non_numeric_output():
    try:
        InjectedTransition(lambda latent, action: "dry").step(1.0, Action.REWASH)
    except TransitionAdapterError:
        pass
    else:
        raise AssertionError("expected a refusal")


def test_injected_refuses_non_finite_and_implausible_output():
    for bad in (float("nan"), float("inf"), 1e9):
        try:
            InjectedTransition(lambda latent, action, b=bad: b).step(1.0, Action.REWASH)
        except FloatingPointError:
            continue
        raise AssertionError(f"expected a refusal for {bad!r}")


def test_injected_accepts_a_well_formed_predictor():
    t = InjectedTransition(lambda latent, action: latent * 0.5)
    assert t.step(3.0, Action.REWASH) == 1.5


def test_injected_rejects_a_non_callable():
    try:
        InjectedTransition(predictor=3)
    except TypeError:
        pass
    else:
        raise AssertionError("expected a TypeError")


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} dynamics tests passed.")


if __name__ == "__main__":
    _main()
