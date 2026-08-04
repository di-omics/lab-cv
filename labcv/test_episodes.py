"""Unit tests for labcv.episodes - runnable two ways:

    python -m pytest labcv/test_episodes.py
    python -m labcv.test_episodes

Two of these tests are the ones the whole module exists for.

`test_oracle_recovers_a_knowable_episode` proves the falsifier can actually
bite. A brute-force search that never succeeds would let every wrong label
through while looking like a rigorous gate, so the falsifier is tested in the
direction where it is conclusive before it is trusted in the other.

`test_seeded_bad_label_fails_the_build` takes a genuinely knowable episode,
mislabels it UNKNOWABLE, and requires the build to refuse. That is the exact
failure mode this construction exists to prevent: a bad label reaching a
scoreboard makes every downstream abstention number a measurement of the
labeller.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile

import numpy as np

from labcv import episodes as E


#: Every pack has to name the epoch its seeds came from, so the tests do too.
SEED_EPOCH = "2026-08-unit"


def _residual(n=10, seed=5, frac=0.20):
    return E.scalar_episodes(E.RESIDUAL_FAMILY, n, seed=seed, unknowable_frac=frac)


def _dispense(n=10, seed=23, frac=0.20):
    return E.scalar_episodes(E.DISPENSE_FAMILY, n, seed=seed, unknowable_frac=frac)


def _plate(n=8, seed=31, frac=0.25):
    return E.occluded_plate_episodes(n, seed=seed, unknowable_frac=frac)


def _export(eps, out, pack_id, pack="dev"):
    return E.export(eps, out, pack_id, pack=pack, seed_epoch=SEED_EPOCH)


def _first(episodes, knowability):
    for ep in episodes:
        if ep.knowability == knowability:
            return copy.deepcopy(ep)
    raise AssertionError(f"no {knowability} episode in the batch")


def test_format_version_is_declared():
    assert E.SCENES_FORMAT_VERSION == 1


def test_episodes_are_action_conditioned():
    ep = _residual()[0]
    # one action per gap between observed frames, then the commanded future ones
    assert len(ep.prefix_actions) == len(ep.prefix_frames) - 1
    assert len(ep.future_latents) == len(ep.future_actions) == ep.horizon
    assert ep.prefix_latents[1] != ep.prefix_latents[0]      # the action moved the state
    assert ep.terminal == ep.future_latents[-1]


def test_ids_do_not_encode_knowability():
    eps = _residual(n=20, seed=9)
    flags = [ep.knowability == E.Knowability.UNKNOWABLE.value for ep in eps]
    assert any(flags) and not all(flags)
    # interleaved: the unknowable ones are neither a prefix nor a suffix block
    idx = [i for i, f in enumerate(flags) if f]
    assert idx != list(range(len(idx)))
    assert all(ep.episode_id.startswith(E.RESIDUAL_FAMILY.task_id) for ep in eps)


def test_unknowable_fraction_lands_in_band():
    fractions = E.refuse_unknowable_fraction(_residual(n=20, seed=9))
    lo, hi = E.UNKNOWABLE_BAND
    for frac in fractions.values():
        assert lo <= frac <= hi


def test_out_of_band_fraction_is_refused():
    for bad in (0.0, 0.9):
        try:
            _residual(n=10, frac=bad)
        except E.UnknowableFractionRefused:
            continue
        raise AssertionError(f"unknowable fraction {bad} should have been refused")


def test_oracle_recovers_a_knowable_episode():
    # the falsifier is only meaningful if it can succeed; check that first
    recovered = 0
    knowable = [ep for ep in _residual(n=10) if ep.knowability == E.Knowability.KNOWABLE.value]
    assert knowable
    for ep in knowable:
        out = E.identifiability_oracle(ep, E.RESIDUAL_FAMILY)
        if out.recovered and abs(out.terminal_hat - ep.terminal) <= ep.tolerance:
            recovered += 1
    assert recovered == len(knowable), f"oracle recovered only {recovered}/{len(knowable)}"


def test_oracle_cannot_invert_an_independence_constructed_episode():
    unknowable = [ep for ep in _residual(n=10)
                  if ep.knowability == E.Knowability.UNKNOWABLE.value]
    assert unknowable
    for ep in unknowable:
        out = E.identifiability_oracle(ep, E.RESIDUAL_FAMILY)
        if out.recovered:
            assert abs(out.terminal_hat - ep.terminal) > ep.tolerance


def test_seeded_bad_label_fails_the_build():
    ep = _first(_residual(n=10), E.Knowability.KNOWABLE.value)
    ep.knowability = E.Knowability.UNKNOWABLE.value
    ep.reason = E.UnknowableReason.INDEPENDENT_LATENT.value
    ep.nominal_terminal = ep.terminal
    ep.branch_terminals = [ep.terminal + c for c in E.RESIDUAL_FAMILY.hidden_support]
    try:
        E.refuse_mislabelled([ep])
    except E.UnknowableLabelViolation as exc:
        assert "brute-force oracle" in str(exc)
    else:
        raise AssertionError("a knowable episode labelled unknowable must fail the build")


def test_label_with_no_construction_is_refused():
    ep = _first(_residual(n=10), E.Knowability.KNOWABLE.value)
    ep.knowability = E.Knowability.UNKNOWABLE.value       # reason stays NONE
    try:
        E.refuse_mislabelled([ep])
    except E.UnknowableLabelViolation as exc:
        assert "no construction" in str(exc)
    else:
        raise AssertionError("unknowability must be constructed, not asserted")


def test_collapsed_branches_are_refused():
    ep = _first(_residual(n=10), E.Knowability.UNKNOWABLE.value)
    # two branches a hair apart: one guess hits both, so the chance ceiling is a lie
    ep.branch_terminals = [1.0, 1.0 + 0.5 * ep.tolerance, 4.0]
    ep.nominal_terminal = None
    try:
        E.refuse_mislabelled([ep])
    except E.UnknowableLabelViolation as exc:
        assert "within tolerance" in str(exc)
    else:
        raise AssertionError("branches inside tolerance of each other must be refused")


def test_branch_equal_to_the_nominal_is_refused():
    ep = _first(_residual(n=10), E.Knowability.UNKNOWABLE.value)
    ep.branch_terminals = [1.0, 3.0, 6.0]
    ep.nominal_terminal = 1.0 + 0.5 * ep.tolerance
    try:
        E.refuse_mislabelled([ep])
    except E.UnknowableLabelViolation as exc:
        assert "nominal" in str(exc)
    else:
        raise AssertionError("a model ignoring the hidden effect must not score on this episode")


def test_occluder_covers_the_whole_roi_pixel_wise():
    eps = _plate()
    unknowable = [ep for ep in eps if ep.knowability == E.Knowability.UNKNOWABLE.value]
    assert unknowable
    for ep in unknowable:
        assert len(ep.occluder_masks) == len(ep.prefix_frames) + len(ep.future_frames)
        for mask in ep.occluder_masks:
            assert E.roi_coverage(mask, ep.target_box) == 1.0
    E.refuse_mislabelled(eps)                     # the same check, at build time


def test_partial_occlusion_is_refused():
    ep = _first(_plate(), E.Knowability.UNKNOWABLE.value)
    x1, y1, _x2, _y2 = E.roi_pixels(ep.target_box, ep.occluder_masks[0].shape)
    ep.occluder_masks[0][y1, x1] = False          # one pixel of the well left visible
    assert E.roi_coverage(ep.occluder_masks[0], ep.target_box) < 1.0
    try:
        E.refuse_mislabelled([ep])
    except E.OcclusionIncomplete as exc:
        assert "target ROI" in str(exc)
    else:
        raise AssertionError("99.9% coverage is a different construction, not a weaker one")


def test_occluded_roi_is_invariant_to_the_hidden_level():
    ep = _first(_plate(), E.Knowability.UNKNOWABLE.value)
    for frame, alt in zip(list(ep.prefix_frames) + list(ep.future_frames), ep.alt_rois):
        assert E.roi_invariant(frame, alt, ep.target_box)
    ep.alt_rois[0] = ep.alt_rois[0].copy()
    ep.alt_rois[0][0, 0] += 0.5                   # the other render would have differed here
    try:
        E.refuse_mislabelled([ep])
    except E.OcclusionIncomplete as exc:
        assert "carry information" in str(exc)
    else:
        raise AssertionError("an ROI that varies with the level is not full occlusion")


def test_duplicate_fixtures_are_refused_and_nothing_is_written():
    eps = _residual(n=10)
    known = [i for i, ep in enumerate(eps) if ep.knowability == E.Knowability.KNOWABLE.value]
    eps[known[1]] = eps[known[0]]                 # same frames twice
    out = os.path.join(tempfile.mkdtemp(), "pack")
    try:
        E.export(eps, out, "dup", seed_epoch=SEED_EPOCH)
    except E.FixtureDuplicationError as exc:
        assert "distinct frame digests" in str(exc)
        assert not os.path.exists(os.path.join(out, "manifest.json"))
    else:
        raise AssertionError("duplicated fixtures inflate n without adding evidence")
    finally:
        shutil.rmtree(os.path.dirname(out), ignore_errors=True)


def test_export_writes_a_versioned_manifest_whose_hashes_match_the_files():
    out = os.path.join(tempfile.mkdtemp(), "pack")
    try:
        manifest = _export(_residual(n=10), out, "unit-dev", pack="dev")
        assert manifest["format_version"] == E.SCENES_FORMAT_VERSION
        assert manifest["format"] == "labcv-rollout-scenes"
        assert manifest["n_episodes"] == 10
        on_disk = json.load(open(os.path.join(out, "manifest.json"), encoding="ascii"))
        assert on_disk == manifest
        for row in manifest["episodes"]:
            assert len(row["frames"]) == len(row["frames_sha256"])
            for frame, digest in zip(row["frames"], row["frames_sha256"]):
                assert frame["sha256"] == digest
                with open(os.path.join(out, frame["path"]), "rb") as fh:
                    assert hashlib.sha256(fh.read()).hexdigest() == digest
            assert row["episode_sha256"] == E.episode_digest(row["frames_sha256"])
    finally:
        shutil.rmtree(os.path.dirname(out), ignore_errors=True)


def test_scoring_pack_withholds_the_coefficients_and_declares_what_it_still_carries():
    """The split the consumer defines, which is not the split this module used to
    write. The consumer requires `knowability` and `terminal_by_horizon` on every
    episode and refuses a manifest missing either, so withholding those produced a
    file nothing could read. `hidden` stays for the same reason: the separation gate
    reads the invisible latent to decide which grid an unknowable label is checked
    over, and a pack without it carries labels that can never be checked.

    That leaves a real gaming channel, and the point of this test is that the file
    NAMES it rather than leaving a reader to find it. A submitter reading the
    manifest directly can abstain on exactly the unknowable episodes, so a scoring
    split is an adversarial holdout only where the harness mediates access."""
    out = os.path.join(tempfile.mkdtemp(), "pack")
    try:
        manifest = _export(_residual(n=10), out, "unit-scoring", pack="scoring")
        row = manifest["episodes"][0]
        for withheld in ("coefficients", "prefix_latents", "future_latents",
                         "terminal", "reason"):
            assert withheld not in row, withheld
        for named in ("coefficients", "future_frames"):
            assert named in manifest["withheld"], named
        # the channel is declared, not hidden, and every field it names is really there
        for named in ("knowability", "hidden", "terminal_by_horizon"):
            assert named in manifest["disclosed_to_harness"], named
        for named in ("knowability", "terminal_by_horizon"):
            assert named in row, named
        # `hidden` rides only on the rows that have something hidden, which is
        # itself the channel: an unknowable row is the one carrying the key.
        unknowable = [r for r in manifest["episodes"] if r["knowability"] != "knowable"]
        assert unknowable, "a pack with no unknowable row cannot exercise abstention"
        assert all("hidden" in r and r["hidden"] for r in unknowable)
        assert all("hidden" not in r for r in manifest["episodes"]
                   if r["knowability"] == "knowable")
        # observed frames only: no future frame is shipped to be read off disk
        assert len(row["frames"]) == E.RESIDUAL_FAMILY.n_observed
        # and the two fields the consumer cannot do without are still there
        assert row["knowability"] in E.READER_KNOWABILITY
        assert row["terminal_by_horizon"]
    finally:
        shutil.rmtree(os.path.dirname(out), ignore_errors=True)


def test_dev_pack_publishes_the_planted_truth():
    out = os.path.join(tempfile.mkdtemp(), "pack")
    try:
        manifest = _export(_residual(n=10), out, "unit-dev", pack="dev")
        row = manifest["episodes"][0]
        assert row["knowability"] in E.READER_KNOWABILITY
        assert row["reason"] in (E.UnknowableReason.NONE.value,
                                 E.UnknowableReason.INDEPENDENT_LATENT.value)
        assert "drydown" in row["coefficients"]
        # The hidden latent is a coefficient with a published nominal, so it appears
        # in both places on a dev pack. Its name is the consumer's, not this module's.
        assert "wet_uL" in row["coefficients"]
        unknowable = _first(E.scalar_episodes(E.RESIDUAL_FAMILY, 10, seed=11,
                                             unknowable_frac=0.20),
                            E.Knowability.UNKNOWABLE)
        assert unknowable.hidden and set(unknowable.hidden) == {"wet_uL"}
        assert manifest["withheld"] == []
        assert manifest["disclosed_to_harness"] == []
        # Observed and future frames live in separate lists, so a reader cannot get
        # the answer by taking the last frame of `frames`. A dev pack publishes both.
        assert len(row["frames"]) == E.RESIDUAL_FAMILY.n_observed
        assert len(row["future_frames"]) == E.RESIDUAL_FAMILY.horizon
    finally:
        shutil.rmtree(os.path.dirname(out), ignore_errors=True)


def test_export_refuses_an_empty_pack():
    try:
        E.export([], tempfile.mkdtemp(), "empty", seed_epoch=SEED_EPOCH)
    except ValueError as exc:
        assert "nothing was checked" in str(exc)
    else:
        raise AssertionError("an empty pack must not be reported as a clean pack")


def test_export_refuses_a_pack_with_no_seed_epoch():
    try:
        E.export(_residual(n=10), tempfile.mkdtemp(), "no-epoch")
    except ValueError as exc:
        assert "seed_epoch" in str(exc)
    else:
        raise AssertionError("an unepoched pack is unrankable against any other")


def test_episode_ids_never_name_their_own_knowability():
    eps = _residual(n=10) + _plate(n=8)
    E.refuse_telltale_ids(eps)                     # the same check, at build time
    for ep in eps:
        lowered = ep.episode_id.lower()
        for token in E.FORBIDDEN_ID_TOKENS:
            assert token not in lowered, f"{ep.episode_id} contains {token}"
    ep = copy.deepcopy(eps[0])
    ep.episode_id = "S9-plate-occlusion-0001"
    try:
        E.refuse_telltale_ids([ep])
    except E.TelltaleEpisodeId as exc:
        assert "string match" in str(exc)
    else:
        raise AssertionError("an id that names the answer is refused before it is written")


def test_unknowable_episodes_are_never_one_contiguous_block():
    """A block is recoverable from the episode index with no model involved, and
    a fair shuffle produces one often enough at these batch sizes to matter."""
    for eps in (_residual(n=10), _dispense(n=10), _plate(n=8), _residual(n=20, seed=9)):
        E.refuse_recoverable_unknowable(eps)
        flags = [ep.knowability == E.Knowability.UNKNOWABLE.value for ep in eps]
        assert not E._is_contiguous(flags)
        assert 2 <= sum(flags) <= len(flags) - 2
    eps = _residual(n=10)
    for i, ep in enumerate(eps):                   # every unknowable one up front
        ep.knowability = (E.Knowability.UNKNOWABLE if i < 2 else E.Knowability.KNOWABLE).value
    try:
        E.refuse_recoverable_unknowable(eps)
    except E.ContiguousUnknowableBlock as exc:
        assert "refusal recall" in str(exc)
    else:
        raise AssertionError("a contiguous block is answered by the index alone")


def test_a_single_unknowable_episode_is_refused():
    # one set flag is a run of length one, so there is no arrangement of it that
    # is not a block; the fraction has to buy at least two on either side
    try:
        E.scalar_episodes(E.RESIDUAL_FAMILY, 10, seed=5, unknowable_frac=0.10)
    except E.UnknowableFractionRefused as exc:
        assert "contiguous block" in str(exc)
    else:
        raise AssertionError("one unknowable episode is a block whichever way it falls")


def test_reader_knowability_covers_every_pair_the_generators_produce():
    seen = set()
    for ep in _residual(n=10) + _plate(n=8):
        seen.add((ep.knowability, ep.reason))
        assert E.reader_knowability(ep) in E.READER_KNOWABILITY
    assert len(seen) >= 3, f"only {sorted(seen)} exercised; the mapping is untested"
    ep = copy.deepcopy(_first(_residual(n=10), E.Knowability.KNOWABLE.value))
    ep.reason = "vibes"
    try:
        E.reader_knowability(ep)
    except E.UnknowableLabelViolation as exc:
        assert "closed enum" in str(exc)
    else:
        raise AssertionError("a pair with no label in the closed enum must be refused")


def test_actions_carry_the_verb_code_out_of_the_perturbed_columns():
    row = E.action_row(E.RESIDUAL_FAMILY.task_id, E.Action.EXTEND_DRY.value)
    assert len(row) == len(E.ACTION_FIELDS) == 5
    assert row[:3] == [0.0, 0.0, 0.0], "p0, p1 and p2 are what a gate perturbs"
    assert row[E.VERB_COLUMN] == float(
        E.VERB_CODES[E.RESIDUAL_FAMILY.task_id].index(E.Action.EXTEND_DRY.value))
    assert row[-1] == E.GRIPPER_UNCHANGED
    for bad in (("no-such-task", E.Action.PROCEED.value),
                (E.RESIDUAL_FAMILY.task_id, E.Action.HALT.value)):
        try:
            E.action_row(*bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} should have been refused rather than encoded")


def test_manifest_satisfies_the_documented_reader_schema():
    """The consumer's refusals, rebuilt from this module's restated copy of them
    and run over a real export.

    This is the test whose absence let the defect through. Both sides declared
    format version 1 while writing and reading different shapes, so the one check
    that exists to catch a shape mismatch passed on every file: a version number
    neither side ties to a structure is a decoration. Everything asserted below is
    a refusal the consumer raises on load, restated field by field.
    """
    out = os.path.join(tempfile.mkdtemp(), "pack")
    try:
        # Only the residual family declares a consumer task. Exporting either of the
        # other two is a refusal, not an oversight, and asserting that here is what
        # stops a later edit from quietly letting a family claim an id whose
        # tolerance would then be applied to a quantity it does not measure.
        for unmapped in (_dispense(n=10), _plate(n=8)):
            try:
                _export(unmapped, os.path.join(out, "refused"), "unit-refused")
            except E.UnmappedConsumerTask as exc:
                assert "declares no consumer task" in str(exc)
            else:
                raise AssertionError("a family with no consumer task must not export")

        for split in E.PACK_SPLITS:
            manifest = _export(_residual(n=10),
                               os.path.join(out, split), f"unit-{split}", pack=split)
            assert int(manifest["format_version"]) == E.SCENES_FORMAT_VERSION
            for name in ("pack_id", "seed_epoch", "source"):
                assert isinstance(manifest[name], str) and manifest[name], name
            assert manifest["split"] in E.PACK_SPLITS
            assert isinstance(manifest["created_utc"], str)
            assert manifest["device_serial_sha256"] is None       # basis: simulated
            rows = manifest["episodes"]
            assert rows, "an empty pack reads on a scorecard exactly like a clean run"
            ids = [r["episode_id"] for r in rows]
            assert len(set(ids)) == len(ids), "the identifier is what every sort keys on"

            digests = set()
            for row in rows:
                assert isinstance(row["episode_id"], str) and row["episode_id"]
                assert isinstance(row["task_id"], str) and row["task_id"]
                assert row["state0"] and all(np.isfinite(v) for v in row["state0"])
                actions = row["actions"]
                assert actions, "an episode with no actions cannot be action-conditioned"
                for action in actions:
                    assert len(action) == len(E.ACTION_FIELDS)
                    assert all(np.isfinite(v) for v in action)
                terminals = row["terminal_by_horizon"]
                assert terminals, "an episode with no terminal is unscoreable"
                for key, value in terminals.items():
                    assert 1 <= int(key) <= len(actions), key
                    assert value and all(np.isfinite(v) for v in value)
                assert row["knowability"] in E.READER_KNOWABILITY
                for token in E.FORBIDDEN_ID_TOKENS:
                    assert token not in row["episode_id"].lower()
                for frame in row["frames"]:
                    assert isinstance(frame["path"], str) and frame["path"]
                    assert len(frame["sha256"]) >= 8
                    assert np.isfinite(frame["t_capture_s"])
                    assert frame["clock_source"] == E.CLOCK_RENDERED
                # the consumer's own duplication key: the frame hashes, in order
                digests.add("|".join(f["sha256"] for f in row["frames"]))
            assert len(digests) == len(rows), "two episodes carry the same frames"
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_dispense_family_generates_and_passes_its_gates():
    eps = E.scalar_episodes(E.DISPENSE_FAMILY, 10, seed=23, unknowable_frac=0.20)
    E.refuse_mislabelled(eps)
    assert all(0.0 <= ep.terminal <= E.DISPENSE_MAX_uL for ep in eps)


def test_render_statistics_are_monotone_in_the_latent():
    xs, ys = E.statistic_table(E.RESIDUAL_FAMILY)
    assert np.all(np.diff(ys) >= -1e-9), "a non-monotone readout makes the oracle meaningless"
    assert xs[0] == 0.0 and xs[-1] == E.RESIDUAL_MAX_uL


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} episode tests passed.")


if __name__ == "__main__":
    _main()
