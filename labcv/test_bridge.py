"""Unit tests for labcv.bridge - runnable two ways:

    python -m pytest labcv/test_bridge.py
    python -m labcv.test_bridge

Every expected value is hand-computed in the comments, so the decision ladder is
auditable rather than merely green. These types moved out of a demo, and the
thing worth pinning is that nothing about their behaviour moved with them.
"""
from __future__ import annotations

from labcv.bridge import (
    Action,
    EventLog,
    HoldForReviewError,
    Policy,
    ResidualLiquidError,
    Verdict,
    decide,
    decide_volume,
)


def _v(residual, conf=1.0, well="W01", state=None):
    state = ("residual" if residual > 0.30 else "dry") if state is None else state
    return Verdict(well=well, residual_uL=residual, wet_frac=0.0, confidence=conf, state=state)


def test_decide_ladder():
    pol = Policy()                                  # dry 0.30, flag 0.60, halt 5.0
    assert decide(_v(5.0), pol) is Action.HALT      # >= halt_uL, boundary is inclusive
    assert decide(_v(4.99), pol) is Action.REWASH   # over dry, under halt
    assert decide(_v(0.31), pol) is Action.REWASH   # just over dry_uL
    assert decide(_v(0.30), pol) is Action.PROCEED  # exactly dry_uL, confident
    assert decide(_v(0.10, conf=0.59), pol) is Action.EXTEND_DRY   # dry but unsure
    assert decide(_v(0.10, conf=0.60), pol) is Action.PROCEED      # conf boundary


def test_decide_volume():
    # target 10.0 +/- 1.5 -> [8.5, 11.5] proceeds, below tops up, above re-aspirates
    assert decide_volume(8.49, 10.0, 1.5) is Action.TOP_UP
    assert decide_volume(8.5, 10.0, 1.5) is Action.PROCEED
    assert decide_volume(11.5, 10.0, 1.5) is Action.PROCEED
    assert decide_volume(11.51, 10.0, 1.5) is Action.REWASH


def test_verdict_ok():
    assert _v(0.0).ok() is True
    assert _v(2.0).ok() is False


def test_event_log_rounds_and_filters():
    log = EventLog()
    log.record("W01", 1, _v(1.23456, conf=0.987654), Action.REWASH)
    log.record("W02", 1, _v(0.0, conf=0.5), Action.PROCEED)
    assert log.rows[0]["residual_uL"] == 1.235      # rounded to 3 dp for the audit trail
    assert log.rows[0]["conf"] == 0.988
    assert log.rows[0]["action"] == "re-aspirate"
    table = log.table(wells=["W01"])
    assert "W01" in table and "W02" not in table
    assert "W02" in log.table()                     # no filter -> every row


def test_residual_error_names_the_well_and_the_number():
    err = ResidualLiquidError(_v(2.5, conf=0.9, well="H12"))
    text = str(err)
    assert "H12" in text and "2.50" in text and "not safe to elute" in text
    assert err.verdict.residual_uL == 2.5


def test_hold_for_review_counts_the_wells():
    err = HoldForReviewError(["A1", "B2", "C3"], 1.0, 0.15)
    assert err.wells == ["A1", "B2", "C3"]
    assert "3 well(s)" in str(err) and "held for review" in str(err)


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} bridge tests passed.")


if __name__ == "__main__":
    _main()
