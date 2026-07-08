"""QC visualization helpers, styled with the repo-wide omics_style palette.

Kept deliberately small: each demo composes these primitives into its own
one-figure QC panel that shows planted ground truth vs recovered result.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import omics_style as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

S.apply()


def show(ax, img, title=None):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    if title:
        ax.set_title(title)


def plate_labels(ax, bxs, rows, cols, color="#CBD3DB", fs=8):
    """Overlay SBS row (A..) and column (1..) labels using the well geometry,
    so a microplate frame reads like a real plate map. Light text for the dark deck."""
    b = np.asarray(bxs, float).reshape(-1, 4)
    if len(b) < rows * cols:
        return
    cx = (b[:, 0] + b[:, 2]) / 2
    cy = (b[:, 1] + b[:, 3]) / 2
    pitch = (cx[cols - 1] - cx[0]) / max(cols - 1, 1)
    for c in range(cols):
        ax.text(cx[c], cy[0] - pitch * 0.95, str(c + 1), color=color, fontsize=fs,
                ha="center", va="center", weight="bold")
    for r in range(rows):
        ax.text(cx[0] - pitch * 0.95, cy[r * cols], chr(65 + r), color=color, fontsize=fs,
                ha="center", va="center", weight="bold")


def boxes(ax, bxs, color, lw=1.4, labels=None):
    for i, (x1, y1, x2, y2) in enumerate(np.asarray(bxs, float).reshape(-1, 4)):
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                               fill=False, edgecolor=color, linewidth=lw))
        if labels is not None:
            ax.text(x1, y1 - 2, labels[i], color=color, fontsize=6.5,
                    va="bottom", ha="left")


def save(fig, outpath):
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  QC plot -> {outpath}")
