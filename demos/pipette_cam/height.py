"""Liquid-height readout - find the meniscus, convert to volume.

The tip-cam side view (labcv.synth.column_view) shows liquid filled bottom-up with
a bright meniscus band. Because the camera is fixed on the channel, the well walls
land in a known box, so the reader just scans down the interior column, finds the
topmost liquid row (the meniscus), and turns that height into uL against the
calibrated fill span. Same call for ethanol-removal QC (is the column ~0?) and
volume QC (how many uL?).

    recover_volume(frame, max_uL) -> (volume_uL, confidence)

The learned seam is identical in shape: a small segmentation net for the
air/liquid boundary (robust to foam, bubbles, tilted menisci, a bead pellet) drops
in behind recover_volume, and everything downstream - the checkpoint, the PLR
bridge - is unchanged. The baseline is a row-intensity threshold: honest, fast,
and soft exactly where foam or a clinging droplet blurs the boundary.
"""
from __future__ import annotations

import numpy as np

from labcv import synth

AIR_LIQUID_THR = 0.30     # row-mean intensity between dark air (~0.14) and liquid (~0.50)


def recover_volume(frame, max_uL, thr=AIR_LIQUID_THR, pad=3):
    """Meniscus row -> liquid height -> volume, with a boundary-sharpness confidence.

    The ROI is inset by `pad` from every wall so the bright well rim is never read
    as liquid; the reader then finds the topmost interior row above the air/liquid
    threshold (the meniscus) and scales its height to the calibrated fill span.
    """
    px = frame.shape[0]
    yt, yb = int(synth.COL_TOP * px), int(synth.COL_BOT * px)
    xl, xr = int(synth.COL_L * px), int(synth.COL_R * px)

    r0, r1 = yt + pad, yb - pad                 # inset ROI: exclude top/bottom rims
    roi = frame[r0:r1, xl + pad:xr - pad]
    row_mean = roi.mean(axis=1)
    liquid_rows = np.where(row_mean > thr)[0]
    if liquid_rows.size == 0:                   # dry / empty well
        return 0.0, 1.0

    rel = int(liquid_rows[0])                   # topmost liquid row within the ROI
    meniscus = r0 + rel                         # absolute meniscus row
    height_px = yb - meniscus
    vol = max_uL * height_px / max(yb - yt, 1)

    # confidence = sharpness of the air->liquid transition at the meniscus
    above = row_mean[max(rel - 3, 0):rel].mean() if rel > 0 else 0.0
    below = row_mean[rel:rel + 3].mean()
    conf = float(np.clip(2.0 * (below - above), 0.0, 1.0))
    return float(vol), conf
