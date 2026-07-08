"""Sampled plate-reader dye QC - the orthogonal checkpoint the camera can't be.

The tip-cam proves the *volume* (a correct height), but it is blind to *chemistry*:
a well filled to the right level with the wrong-concentration reagent looks
identical. So a small aliquot from a sampled subset of wells is pulled into a QC
plate, mixed with a readiness dye, and read on a plate reader - signal tracks
concentration, so this catches the off-spec wells the camera cannot. The cost is
coverage: you only read the fraction you sample, which is exactly why a hit has to
escalate (read more, hold the batch) rather than being silently trusted.

    sample_indices(n, stride)                 -> systematic sample (every `stride`)
    read_dye(conc, idx, target, ...)          -> {well_idx: concentration_est}
    flag_offspec(est, target, tol)            -> {well_idx off tolerance}

This mirrors the shared Rhodamine-B / dye readiness QC used across the lab-
automation stack: an independent, quantitative readout that turns 'the motions
looked right' into a measured pass/fail. Fully synthetic here - the reader signal
is modelled, not real - but the interface is what a real plate reader returns.
"""
from __future__ import annotations

import numpy as np


def sample_indices(n, stride):
    """Systematic sample: every `stride`-th well (stride=4 -> 25% coverage)."""
    return list(range(0, n, stride))


def read_dye(conc, idx, target=1.0, gain=1.0, noise=0.02, rng=None):
    """Model a plate-reader dye readout for the sampled wells.

    A readiness dye's signal scales with concentration (path length and read
    volume are fixed in the QC plate), so signal = gain*conc + reader noise, and
    the recovered concentration is signal/gain. Returns {well_idx: conc_est}.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    est = {}
    for i in idx:
        signal = gain * float(conc[i]) + rng.normal(0, noise)
        est[i] = signal / gain
    return est


def flag_offspec(est, target=1.0, tol=0.15):
    """Wells whose measured concentration is outside spec -> flagged off-spec."""
    return {i for i, c in est.items() if abs(c - target) > tol}
