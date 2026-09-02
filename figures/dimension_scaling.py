"""d_osc scaling law -> figures/output/fig_scaling_law.pdf.

Uniform seed design: five seeds per configuration, ten at each held-out test point. Plots, for
TinyStories and WikiText-2: the [2,32] d_osc scaling points (5-seed mean PPL with SEM error bars),
the refit power-law line, a shaded 95% PREDICTION-INTERVAL band at the held-out d_osc=64 (fit
uncertainty (+) observation noise -- the out-of-sample test, NOT the fit-only bootstrap CI), and the
held-out d_osc=64 observations as distinct markers (10 seeds per dataset). Semilog-x (log2 d_osc),
linear-y PPL.

Inputs (every plotted number comes from these analysis JSONs -- nothing typed):
    results/lm_dimension_scaling/ts_uniform.json   (TinyStories)
    results/lm_dimension_scaling/wt2_uniform.json  (WikiText-2)
each carrying: base_mean, C, alpha, pred_interval95, d64_ppls, and scaling_ppls (per-seed PPLs
for d in {2,4,8,16,32}).

Run standalone from the repo root:
  python figures/dimension_scaling.py
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import style as ps



def _load(rel):
    """`rel` is relative to the results root (see style.results_root)."""
    return json.load(open(os.path.join(ps.results_root(), rel)))


def _scaling_mean_sem(analysis):
    """d_osc -> (mean, sem) from the per-seed scaling PPLs stored in the uniform json."""
    out = {}
    for d, ppls in analysis["scaling_ppls"].items():
        a = np.asarray(ppls, float)
        out[int(d)] = (float(a.mean()), float(a.std(ddof=1) / np.sqrt(a.size)))
    return out


def _panel(ax, analysis, title, show_ylabel):
    pts = _scaling_mean_sem(analysis)
    d = sorted(pts)
    means = [pts[k][0] for k in d]
    sems = [pts[k][1] for k in d]
    base = analysis["base_mean"]
    C, alpha = analysis["C"], analysis["alpha"]

    # scaling points (5-seed mean +/- SEM)
    ax.errorbar(d, means, yerr=sems, fmt="o", color=ps.COLOR_OSCILLATOR, markersize=5.5,
                capsize=2.5, elinewidth=1.0, zorder=4,
                label=r"Oscillator (5-seed mean, $d\!\le\!32$)")
    # softmax baseline (5-seed mean)
    ax.axhline(base, color=ps.COLOR_SOFTMAX, ls="--", lw=1.3, zorder=2,
               label="Softmax baseline (5 seeds)")
    # power-law fit line (extended through d=64)
    dfine = np.logspace(np.log10(d[0] * 0.85), np.log10(64 * 1.15), 200)
    ax.plot(dfine, base + C * dfine ** (-alpha), color=ps.COLOR_OSCILLATOR, ls=":",
            lw=1.3, alpha=0.8, zorder=3,
            label=rf"Fit $\Delta={C:.2f}\,d^{{-{alpha:.2f}}}$")

    # Held-out d=64: the 10-seed MEAN as a diamond (distinct color, SAME size as the d<=32 dots),
    # with a single capped whisker spanning the 95% PREDICTION INTERVAL for that mean, drawn in the
    # same style/weight as the SEM whiskers above. Individual seeds are not shown (the interval
    # applies to the mean, not to single seeds).
    obs = np.asarray(analysis["d64_ppls"], float)
    obs_mean = float(obs.mean())
    lo, hi = [base + x for x in analysis["pred_interval95"]]
    pilbl = rf"Held-out $d\!=\!64$ mean ($n={obs.size}$), 95\% PI" \
        if plt.rcParams["text.usetex"] else rf"Held-out $d=64$ mean (n={obs.size}), 95% PI"
    ax.errorbar([64], [obs_mean], yerr=[[obs_mean - lo], [hi - obs_mean]], fmt="D",
                color=ps.COLOR_NEG, markersize=5.5, markeredgecolor="black", markeredgewidth=0.6,
                capsize=2.5, elinewidth=1.0, zorder=6, label=pilbl)

    ax.set_xscale("log", base=2)
    ticks = d + [64]
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter([str(t) for t in ticks]))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlim(d[0] * 0.8, 64 * 1.25)
    ax.set_xlabel(r"$d_{\rm osc}$")
    if show_ylabel:
        ax.set_ylabel("Validation perplexity")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")


def main():
    ps.apply_style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=ps.FIG_WIDE)
    fig.subplots_adjust(wspace=0.34)
    _panel(a1, _load("lm_dimension_scaling/ts_uniform.json"), "TinyStories", True)
    _panel(a2, _load("lm_dimension_scaling/wt2_uniform.json"), "WikiText-2", False)
    ps.save(fig, "fig_scaling_law")


if __name__ == "__main__":
    main()
