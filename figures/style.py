"""Shared matplotlib style for the paper figure set.

Every figure script in this directory calls ``apply_style()``, uses the semantic colors and
figure-size constants defined here, and writes its output through ``save()`` (see below).
``plot_style.py`` is the older style module still used by the two checkpoint-driven scripts
(``sva_verb_attention.py``, ``tinystories_demo.py``); the color values agree.

Design (matches the TMLR paper):
  * usetex when a probe render succeeds, else DejaVu Sans serif fallback.
  * font.size 11, axes.labelsize 11, ticks 10, legend 9.
  * light-gray grid (#cfcfcf, alpha 0.30); top/right spines off.
  * softmax = neutral gray (#888888), oscillator = blue (#0072B2), accent = orange.
"""
import os
import tempfile
import matplotlib
import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe categorical palette (from figures/plot_style.py).
PALETTE = ["#000000", "#E69F00", "#56B4E9", "#009E73",
           "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

# Semantic aliases.
COLOR_SOFTMAX = "#888888"       # neutral gray
COLOR_OSCILLATOR = PALETTE[5]   # blue  (#0072B2)
COLOR_ACCENT = PALETTE[1]       # orange (#E69F00) — bounds / references / held-out obs
COLOR_HIGHLIGHT = PALETTE[3]    # bluish green
COLOR_NEG = PALETTE[6]          # vermillion
CMAP = "viridis"

# Standard figure dimensions (inches) — ONE convention across the whole paper.
# TWO_PANEL is the reference two-panel layout (the scaling figure); SINGLE_PANEL is exactly one
# panel of it (half the width, same height, same margins), so a single-panel figure reads as
# one panel of the standard layout rather than a stretched/shrunken independent design.
# The 2x4 robustness grid keeps its own larger size (FIG_GRID).
TWO_PANEL = (9.5, 3.8)          # standard two-panel (the scaling figure is the reference)
SINGLE_PANEL = (4.75, 3.8)      # one panel of TWO_PANEL: half width, same height
FIG_GRID = (10.0, 4.6)          # 2x4 robustness grid only
# Back-compat aliases so older scripts keep working AND inherit the unified sizes.
FIG_WIDE = TWO_PANEL
FIG_DOUBLE = TWO_PANEL
FIG_SINGLE = SINGLE_PANEL


def _usetex_works() -> bool:
    try:
        matplotlib.use("Agg", force=True)
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
    """Apply shared rcParams; returns the effective usetex flag."""
    if usetex is None:
        usetex = _usetex_works()
    matplotlib.rcParams.update({
        "text.usetex": usetex,
        # One serif family for the whole document, matching the paper body (Latin Modern, loaded
        # via \usepackage{lmodern} under usetex — see preamble below). The non-usetex fallback list
        # is Latin-Modern / Computer-Modern serif (NEVER a sans fallback), so fonts stay uniform
        # whether or not usetex is active.
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "CMU Serif", "Palatino", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 9, "legend.frameon": True, "legend.framealpha": 0.9,
        "legend.edgecolor": "#cfcfcf",
        "axes.grid": True, "grid.color": "#cfcfcf", "grid.alpha": 0.30,
        "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "lines.linewidth": 1.6, "lines.markersize": 5,
        # NOT "tight": tight-cropping makes the saved size content-dependent, so figures of the
        # same figsize would render at different heights. "standard" honours the exact figsize, so
        # every two-panel figure is identical in physical dimensions (hence same height in-document).
        "savefig.bbox": "standard", "savefig.dpi": 200,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    if usetex:
        # Match the paper body exactly: tmlr.sty loads Latin Modern via \usepackage{lmodern}.
        matplotlib.rcParams["text.latex.preamble"] = \
            r"\usepackage{lmodern}\usepackage{amsmath,amssymb}"
    return usetex


def repo_root() -> str:
    """Absolute path to the repo root (this file is figures/)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def results_root() -> str:
    """The results tree the figures read: ``results/``, the run behind the paper, so the
    figures regenerate on a clean clone and can be checked against the published PDFs.

    Set OSCILLATOR_RESULTS to read a different tree -- point it at ``runs`` to redraw the
    figures from your own re-run::

        OSCILLATOR_RESULTS=runs python figures/dimension_scaling.py
    """
    return os.environ.get("OSCILLATOR_RESULTS", os.path.join(repo_root(), "results"))


def output_dir() -> str:
    """Directory every figure script writes into: ``figures/output/`` (gitignored)."""
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(d, exist_ok=True)
    return d


def save(fig, stem: str):
    """Save ``figures/output/<stem>.pdf`` and ``.png``.

    ``stem`` is the name ``main.tex`` includes the figure by (``fig_<name>``), so the emitted
    PDF can be copied into the paper's ``figures/`` directory unrenamed. See figures/README.md
    for the script -> paper-figure table.
    """
    outdir = output_dir()
    pdf = os.path.join(outdir, stem + ".pdf")
    png = os.path.join(outdir, stem + ".png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print(f"wrote {pdf}\nwrote {png}")
