"""Shared plot style for omics-demos.

Baby-pastel palette, fine-line outlines, Manrope typography, and constrained
layout so text never overlaps. Import and call `apply()` at the top of any
plots.py, then pull colors from PALETTE / OUTLINE and colormaps from cmap().
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap

# --- baby-pastel palette: soft fill + a slightly deeper sibling for fine outlines ---
PALETTE = {
    "blue":   "#A9C9E8",
    "pink":   "#F4B8C6",
    "green":  "#B4DBA6",
    "lav":    "#C7BCE8",
    "peach":  "#F7C9A3",
    "teal":   "#A6DBD2",
    "gold":   "#F1DEA0",
}
OUTLINE = {
    "blue":   "#5E8DBF",
    "pink":   "#D07A90",
    "green":  "#74A860",
    "lav":    "#8E7BC4",
    "peach":  "#DE9A66",
    "teal":   "#5FAA9C",
    "gold":   "#C9AE5C",
}
INK, MUTED, GRID, HAIR = "#3B4048", "#7A828C", "#EDF0F3", "#D9DEE4"
ORDER = ["blue", "pink", "green", "lav", "peach", "teal", "gold"]


def cmap(name):
    """White -> pastel colormap for soft, clear heatmaps."""
    return LinearSegmentedColormap.from_list(f"baby_{name}", ["#FFFFFF", PALETTE[name]])


def _load_manrope():
    fdir = Path(__file__).resolve().parent / ".fonts"
    if not fdir.exists():
        return False
    for ttf in sorted(fdir.glob("Manrope-*.ttf")):
        try:
            fm.fontManager.addfont(str(ttf))
        except Exception:
            pass
    return any("Manrope" == f.name for f in fm.fontManager.ttflist)


def apply():
    has_manrope = _load_manrope()
    family = "Manrope" if has_manrope else "DejaVu Sans"
    plt.rcParams.update({
        "font.family": family,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold" if has_manrope else "bold",
        "axes.labelsize": 9.5,
        "axes.labelcolor": INK,
        "text.color": INK,
        "axes.edgecolor": HAIR,
        "axes.linewidth": 0.9,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.9,
        "lines.linewidth": 1.4,
        "patch.linewidth": 1.1,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "figure.constrained_layout.use": True,
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.facecolor": "white",
        "legend.frameon": False,
        "legend.fontsize": 8.5,
    })
    return family
