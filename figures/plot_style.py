"""
Shared matplotlib style for the two checkpoint-driven figure scripts
(``sva_verb_attention.py``, ``tinystories_demo.py``); the other seven use ``style.py``.

Both import ``apply_style()`` (which sets rcParams) and ``PALETTE`` / ``CMAP_SEQUENTIAL``
for color choices.

Design choices:
  * Font:        text.usetex=True when available (matches the LaTeX paper),
                 falls back to DejaVu Sans serif if usetex is broken.
  * Palette:     Okabe-Ito colorblind-safe categorical palette.
  * Sequential:  viridis (used only for the SVA attention heatmap).
  * Gridlines:   light gray (#cfcfcf), alpha 0.30, globally enabled.
  * Font sizes:  axis labels 10, ticks 9, title 10  -- TMLR-appropriate.
"""

import os
import tempfile
import matplotlib
import matplotlib.pyplot as plt


# Okabe-Ito categorical palette (colorblind-safe, 8 colors).
# Order: black, orange, sky blue, bluish green, yellow, blue, vermillion, redd. purple
PALETTE = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
]

# Convenience semantic aliases used across multiple figures.
COLOR_SOFTMAX     = "#888888"           # neutral gray
COLOR_OSCILLATOR  = PALETTE[5]          # blue (#0072B2)
COLOR_ACCENT      = PALETTE[1]          # orange  (anchors, RK45 markers...)
COLOR_HIGHLIGHT   = PALETTE[3]          # bluish green
COLOR_NEG         = PALETTE[6]          # vermillion

CMAP_SEQUENTIAL = "viridis"


def _usetex_works() -> bool:
    """Probe whether a tiny LaTeX-mode render actually succeeds."""
    try:
        matplotlib.use("Agg", force=True)
        # Save and restore so we don't pollute global state during the probe.
        prev = matplotlib.rcParams["text.usetex"]
        matplotlib.rcParams["text.usetex"] = True
        try:
            fig, ax = plt.subplots()
            ax.set_xlabel(r"$\alpha$")
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                fig.savefig(f.name)
            os.unlink(f.name)
            plt.close(fig)
            return True
        finally:
            matplotlib.rcParams["text.usetex"] = prev
    except Exception:
        return False


def apply_style(usetex: bool = None) -> bool:
    """Apply the shared rcParams. Returns the effective usetex flag."""
    if usetex is None:
        usetex = _usetex_works()

    matplotlib.rcParams.update({
        "text.usetex":          usetex,
        "font.family":          "serif" if usetex else "DejaVu Sans",
        "font.size":            10,
        "axes.titlesize":       10,
        "axes.labelsize":       10,
        "xtick.labelsize":      9,
        "ytick.labelsize":      9,
        "legend.fontsize":      8,
        "legend.frameon":       True,
        "legend.framealpha":    0.9,
        "legend.edgecolor":     "#cfcfcf",
        "axes.grid":            True,
        "grid.color":           "#cfcfcf",
        "grid.alpha":           0.30,
        "grid.linewidth":       0.6,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.linewidth":       0.8,
        "lines.linewidth":      1.6,
        "lines.markersize":     5,
        # "tight" (content-dependent crop), NOT style.py's "standard". These two figures are
        # bespoke sizes outside the SINGLE/TWO_PANEL system, so style.py's reason for "standard"
        # -- keeping every two-panel figure physically identical -- does not apply. Switching them
        # would add ~15% whitespace to the SVA figure and shrink its content on the page.
        "savefig.bbox":         "tight",
        "savefig.dpi":          200,
        "pdf.fonttype":         42,   # embed TrueType so PDF is editable
        "ps.fonttype":          42,
    })
    if usetex:
        # Match the paper body: tmlr.sty loads Latin Modern via \usepackage{lmodern}. Without
        # this, LaTeX falls back to Computer Modern and the figures do not match the text.
        matplotlib.rcParams["text.latex.preamble"] = \
            r"\usepackage{lmodern}\usepackage{amsmath,amssymb}"
    return usetex
