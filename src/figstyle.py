"""Publication style for Physical Review two-column figures.

Central place for column widths, the colour-blind-safe Okabe-Ito palette,
matplotlib rcParams and a save helper that always emits a vector PDF (for
LaTeX \\includegraphics) and a 600-dpi PNG under figures/ with the same
basename.  Import and call ``apply_style()`` once, then use ``PALETTE`` and
``savefig()``.
"""
from __future__ import annotations

import os
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- geometry
# Physical Review Revtex column widths (inches).
COL_SINGLE = 3.375   # single column
COL_DOUBLE = 7.00    # full page width (double column)

# ------------------------------------------------------------- Okabe-Ito
# Fixed categorical order (colour-blind safe).
OKABE_ITO = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish-green
    "#CC79A7",  # reddish-purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#000000",  # black
]
PALETTE = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "black": "#000000",
    "grey": "#BBBBBB",
}

# Sequential magnitude maps use viridis (perceptually uniform, CVD-safe).
SEQ_CMAP = "viridis"


def apply_style() -> None:
    """Install the journal rcParams (serif mathtext, small legible fonts)."""
    mpl.rcParams.update({
        # fonts -- Computer-Modern-like serif via mathtext
        "font.family": "serif",
        "font.serif": ["cmr10", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        # lines -- thin
        "lines.linewidth": 1.3,
        "lines.markersize": 3.5,
        "axes.linewidth": 0.7,
        # ticks inward, on all we want
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.6,
        "ytick.minor.size": 1.6,
        "xtick.top": True,
        "ytick.right": True,
        # recessive grid
        "grid.color": "#999999",
        "grid.alpha": 0.30,
        "grid.linewidth": 0.5,
        # colour cycle = Okabe-Ito
        "axes.prop_cycle": mpl.cycler(color=OKABE_ITO),
        # legend
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.borderaxespad": 0.4,
        "legend.labelspacing": 0.3,
        # figure/output
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,   # embed TrueType so text stays editable/searchable
        "ps.fonttype": 42,
    })


def _figdir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    d = os.path.normpath(os.path.join(here, os.pardir, "figures"))
    os.makedirs(d, exist_ok=True)
    return d


def savefig(fig, basename: str) -> list[str]:
    """Write ``basename`` as both PDF (vector) and 600-dpi PNG to figures/.

    Returns the list of absolute paths written.
    """
    outdir = _figdir()
    paths = []
    for ext, dpi in (("pdf", None), ("png", 600)):
        p = os.path.join(outdir, f"{basename}.{ext}")
        fig.savefig(p, dpi=dpi)
        paths.append(p)
    return paths
