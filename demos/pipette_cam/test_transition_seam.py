"""Unit tests for the pipette-cam transition seam - runnable two ways:

    python -m pytest demos/pipette_cam/test_transition_seam.py
    python -m demos.pipette_cam.test_transition_seam

These are regression guards, not feature tests. `plr_bridge.py` became a
re-export shim and `run.py` stopped applying its transition inline, and both
changes are only safe if the demo keeps printing exactly the numbers the README
quotes. The shim test checks object identity rather than equality, because two
equal-but-distinct `Action` enums would make `act is Action.REWASH` false
everywhere and break the loop silently.
"""
from __future__ import annotations

from labcv import bridge
from labcv.bridge import Action
from labcv.dynamics import AnalyticTransition, DispenseTransition
from demos.pipette_cam import plr_bridge
from demos.pipette_cam import run as run_mod
from demos.pipette_cam import run_qc as run_qc_mod


def test_shim_re_exports_the_same_objects_not_copies():
    for name in ("Action", "EventLog", "HoldForReviewError", "Policy",
                 "ResidualLiquidError", "Verdict", "decide", "decide_volume"):
        assert getattr(plr_bridge, name) is getattr(bridge, name), name


def test_verify_still_reads_its_types_through_the_shim():
    from demos.pipette_cam import verify
    assert verify.Verdict is bridge.Verdict
    assert verify.Policy is bridge.Policy


def test_demo_config_still_carries_the_published_constants():
    # the README quotes numbers produced with 0.12 and 0.35; the constants moved
    # to labcv.dynamics and the demo must still resolve to the same two values
    cfg = run_mod.Config()
    assert cfg.rewash_leave == 0.12
    assert cfg.drydown == 0.35
    assert cfg.rewash_leave == AnalyticTransition.REWASH_LEAVE
    assert cfg.drydown == AnalyticTransition.DRYDOWN


def test_transition_matches_the_arithmetic_run_py_used_to_do_inline():
    cfg = run_mod.Config()
    t = AnalyticTransition(cfg.rewash_leave, cfg.drydown)
    for res in (3.1014, 0.6031, 0.3212, 2.8951, 1.8834, 0.2264):
        assert t.step(res, Action.REWASH) == res * cfg.rewash_leave
        assert t.step(res, Action.EXTEND_DRY) == res * cfg.drydown


def test_run_qc_no_longer_asserts_a_zero_error_correction():
    # what used to sit in run_qc.py was `vol_final[i] = cfg.target_vol`
    cfg = run_qc_mod.Config()
    t = DispenseTransition(cfg.target_vol, noise_uL=0.0)
    landed = t.step(7.0, Action.TOP_UP, measured=7.2)
    assert landed != cfg.target_vol
    assert abs(landed - cfg.target_vol) <= cfg.tol_vol      # near target, not on it


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} transition seam tests passed.")


if __name__ == "__main__":
    _main()
