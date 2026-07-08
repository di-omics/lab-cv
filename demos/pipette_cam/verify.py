"""Residual-liquid verification - is this well DRY, or is ethanol left behind?

The tip-mounted camera looks straight down one well after supernatant removal.
The classical baseline reads the fraction of 'wet' (bright pooled + specular)
pixels inside the fixed central well disk: a dry bottom is matte and dark,
residual ethanol pools and glints. That fraction both DECIDES dry/residual and,
once calibrated against one known reference volume, ESTIMATES the leftover uL.

    wet_fraction(frame)                  -> wet-pixel fraction in the well disk
    calibrate(reference_uL, ...)         -> wet_frac-per-uL (one-point cam cal)
    residual_of(frame, cal)              -> (residual_uL_est, wet_frac)
    verify_well(frame, well, cal, pol)   -> Verdict            (see plr_bridge)

`verify_well` is the single call site. The learned seam swaps the reader without
touching the bridge or the harness: SAM2 segments the liquid film for an exact
area -> volume, or a VLM answers 'is this well dry?' - same `Verdict` out. The
baseline is honest about its break: a scratch or condensation bead that glints
on a dry rim (`--glare`) reads as wet, a false alarm a segmentation model fixes.
"""
from __future__ import annotations

import numpy as np

from plr_bridge import Policy, Verdict

WET_THR = 0.40       # pixel intensity above which a well pixel looks 'wet'
DISK_FRAC = 0.42     # readable well-bottom radius (matches synth.tip_view geometry)


def _disk_mask(shape):
    """Boolean mask of the central well-bottom disk the tip cam sees."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = DISK_FRAC * min(h, w)
    return (xx - w // 2) ** 2 + (yy - h // 2) ** 2 <= r * r


def wet_fraction(frame) -> float:
    """Fraction of the well disk that reads as wet (pooled liquid or specular)."""
    m = _disk_mask(frame.shape)
    interior = frame[m]
    if interior.size == 0:
        return 0.0
    return float((interior > WET_THR).mean())


def calibrate(reference_uL, synth, max_uL=5.0, rng=None, n=5) -> float:
    """One-point cam calibration: average wet_fraction on a known reference volume,
    so residual estimates are in real uL. Real rigs calibrate the same way - image
    a pipetted reference droplet once, before the run."""
    rng = np.random.default_rng(123) if rng is None else rng
    fr = np.mean([wet_fraction(synth.tip_view(reference_uL, rng=rng, max_uL=max_uL)[0])
                  for _ in range(n)])
    return max(float(fr), 1e-6) / reference_uL


def residual_of(frame, cal_per_uL):
    """Estimate leftover volume (uL) from wet fraction and the cam calibration."""
    wf = wet_fraction(frame)
    return wf / cal_per_uL, wf


def verify_well(frame, well, cal_per_uL, pol: Policy = None) -> Verdict:
    """Read one tip-cam frame into a Verdict the PLR bridge can act on."""
    pol = pol or Policy()
    res_uL, wf = residual_of(frame, cal_per_uL)
    state = "residual" if res_uL > pol.dry_uL else "dry"
    # confidence = distance from the dry/residual boundary, squashed to [0, 1];
    # collapses toward 0 for a well hovering right at the threshold -> a QC flag.
    z = (res_uL - pol.dry_uL) / pol.conf_scale
    conf = 2.0 * abs(1.0 / (1.0 + np.exp(-z)) - 0.5)
    return Verdict(well=well, residual_uL=float(res_uL), wet_frac=float(wf),
                   confidence=float(round(conf, 4)), state=state)
