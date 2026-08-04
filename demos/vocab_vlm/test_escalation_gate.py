"""Unit tests for the vocab-VLM escalation gate - runnable two ways:

    python -m pytest demos/vocab_vlm/test_escalation_gate.py
    python -m demos.vocab_vlm.test_escalation_gate

The demo's headline claim is that a VLM is spent only on low-confidence boxes.
Before this test existed, `escalate` was computed and then every box was sent to
the VLM anyway, so the claim was bookkeeping about a branch nothing took and no
test could have noticed: the accuracy was identical either way.

So the test counts the regions that actually crossed the adapter boundary. That
is the only observation that can tell an enforced gate from a described one.
"""
from __future__ import annotations

import io
import contextlib

import numpy as np

from demos.vocab_vlm import run as run_mod


class _CountingVLM:
    """Stands in for the adapter and records how many regions it was handed."""

    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.calls = 0
        self.n_regions = 0

    def __call__(self, image, boxes, vocab, backend="mock"):
        self.calls += 1
        self.n_regions += len(np.asarray(boxes, float).reshape(-1, 4))
        return self.wrapped(image, boxes, vocab, backend=backend)


def _run_counted(cfg=None):
    counter = _CountingVLM(run_mod.label_regions)
    original = run_mod.label_regions
    run_mod.label_regions = counter
    try:
        with contextlib.redirect_stdout(io.StringIO()) as out:
            ok = run_mod.run(cfg or run_mod.Config())
    finally:
        run_mod.label_regions = original
    return ok, counter, out.getvalue()


def test_the_gate_actually_gates():
    ok, counter, text = _run_counted()
    assert ok, "the demo must still pass with the gate enforced"
    assert "detections=27" in text
    assert "escalated to VLM (conf<0.85)=13" in text
    # 13 regions crossed the boundary, not 27. This is the whole fix.
    assert counter.n_regions == 13, f"the VLM saw {counter.n_regions} regions, expected 13"
    assert "VLM calls actually made                 13/27" in text


def test_the_saved_count_is_derived_from_calls_that_happened():
    _ok, counter, text = _run_counted()
    saved = 27 - counter.n_regions
    assert f"VLM calls saved by layering             {saved}/27" in text
    assert f"({round(100 * saved / 27)}%)" in text          # 14/27 -> 52%


def test_accuracy_is_unchanged_by_the_gate():
    # the non-escalated boxes are all confidently-sized wells, so the classical
    # classifier names them correctly and the published 1.000 still holds
    _ok, _counter, text = _run_counted()
    assert "open-vocab labeling accuracy vs plant   1.000" in text


def test_the_cost_of_the_saved_calls_is_printed():
    _ok, _counter, text = _run_counted()
    assert "cost of the saved calls                 0/14  cheap-path labels wrong" in text


def test_the_gate_has_a_measurable_cost_at_another_seed():
    # seed 11: three specks score above 0.85, take the cheap path, and are named
    # "empty well" because that path has no word for a bubble. Accuracy 0.889 and
    # a FAIL. The un-gated version printed 1.000 here, which was the gate's
    # absence flattering it, so this failing number is the honest one.
    ok, counter, text = _run_counted(run_mod.Config(seed=11))
    assert ok is False
    assert counter.n_regions == 11
    assert "cost of the saved calls                 3/16  cheap-path labels wrong" in text
    assert "open-vocab labeling accuracy vs plant   0.889" in text
    assert "the gate kept from the VLM are specks" in text


def test_cheap_path_has_the_narrow_vocabulary_and_owns_it():
    # the $0 path cannot say "bubble"; that is the cost of not spending the call
    img = np.full((40, 40), 0.9, np.float32)
    label, conf = run_mod._cheap_label(img, [5, 5, 35, 35])
    assert label in ("filled well", "empty well")
    assert 0.0 <= conf <= 1.0


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} escalation gate tests passed.")


if __name__ == "__main__":
    _main()
