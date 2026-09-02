"""Shared data loaders + drawing functions for the KWS / SVA accuracy panels.

Both the main-text composite (bidirectional_accuracy.py -> fig_bidirectional) and the appendix
seed-distribution panel (attention_entropy.py -> fig_entropy_seeds, right panel) import from here so
the two designs never diverge; styling comes from style.py. All values come from the
kws_position_matched / sva_seed_robustness result JSONs; nothing is hardcoded.
"""
import glob
import json
import os

import matplotlib
import numpy as np

import style as ps


def _tex():
    return bool(matplotlib.rcParams.get("text.usetex", False))

# ── shared axis convention (matches fig_bidirectional's 60–100 accuracy range) ──
Y_LO, Y_HI = 60.0, 100.0
DY_LABEL = (Y_HI - Y_LO) * 0.022      # bar top -> value label (as in fig_bidirectional)
FAIL_THRESH = 85.0                    # SVA "training failure" threshold (sva_seed_robustness convention)

PES = ["none", "sinusoidal", "learned_abs"]
PE_LABEL = {"none": "no PE", "sinusoidal": "sinusoidal", "learned_abs": "learned abs."}
# (mech key in kws_position_matched, legend label, color)
MECHS = [("softmax", "Softmax", ps.COLOR_SOFTMAX),
         ("osc", "Oscillator", ps.COLOR_OSCILLATOR)]


def load_e1():
    """kws_position_matched KWS: -> {(mech, pe): [5 per-seed accuracies in %]}."""
    root = ps.results_root()
    data = {}
    for path in glob.glob(os.path.join(root, "kws_position_matched", "*.json")):
        r = json.load(open(path))
        data.setdefault((r["mech"], r["pe"]), []).append(r["val_acc"] * 100.0)
    return data


def load_e2_hard():
    """sva_seed_robustness SVA main runs: -> {'softmax': [50 hard acc %], 'oscillator': [50 hard acc %]}.

    sva_seed_robustness stores the oscillator under attn_type 'kuramoto' (= LoheAttention d_osc=2)."""
    root = ps.results_root()
    out = {}
    for e2key, label in [("softmax", "softmax"), ("kuramoto", "oscillator")]:
        vals = []
        for path in sorted(glob.glob(os.path.join(
                root, "sva_seed_robustness", f"main_{e2key}_s*.json"))):
            vals.append(json.load(open(path))["val_acc"]["hard"])
        out[label] = vals
    return out


def load_e2_all3():
    """sva_seed_robustness SVA main runs -> {'softmax': {'all':[50],'simple':[50],'hard':[50]},
    'oscillator': {...}} (per-seed overall/simple/hard accuracy in %)."""
    root = ps.results_root()
    out = {}
    for e2key, label in [("softmax", "softmax"), ("kuramoto", "oscillator")]:
        buckets = {"all": [], "simple": [], "hard": []}
        for path in sorted(glob.glob(os.path.join(
                root, "sva_seed_robustness", f"main_{e2key}_s*.json"))):
            va = json.load(open(path))["val_acc"]
            for k in buckets:
                buckets[k].append(va[k])
        out[label] = buckets
    return out


def draw_sva_grouped(ax, show_ylabel=True, title="(b) Subject--Verb Agreement"):
    """Grouped bars over {Overall, Simple, Hard} × {softmax, oscillator}, mirroring
    fig_bidirectional panel (b): each bar = mean over the 50 seeds, value printed above,
    ±1 std error bars. The UPPER whisker is truncated at 100 % (asymmetric yerr), and the
    y-limit is raised so value labels never overlap the panel title. No in-plot annotations
    (failure counts / criterion belong in the caption)."""
    data = load_e2_all3()
    metrics = [("all", "Overall"), ("simple", "Simple"), ("hard", "Hard")]
    mechs = [("softmax", "Softmax", ps.COLOR_SOFTMAX),
             ("oscillator", "Oscillator", ps.COLOR_OSCILLATOR)]
    xs = np.arange(len(metrics))
    w = 0.38
    CAP = 100.0
    max_label_top = Y_LO
    for i, (mkey, mlabel, color) in enumerate(mechs):
        means = [float(np.mean(data[mkey][mk])) for mk, _ in metrics]
        stds = [float(np.std(data[mkey][mk], ddof=1)) for mk, _ in metrics]
        lo_err = stds
        hi_err = [min(s, CAP - m) for m, s in zip(means, stds)]  # truncate upper at 100%
        pos = xs + (i - 0.5) * w
        bars = ax.bar(pos, means, width=w, color=color,
                      yerr=[lo_err, hi_err], capsize=3, error_kw={"lw": 1.0},
                      edgecolor="white", linewidth=0.5, label=mlabel)
        for bar, m, s in zip(bars, means, stds):
            ytop = min(m + s, CAP)                 # label above the (truncated) whisker
            ax.text(bar.get_x() + bar.get_width() / 2, ytop + DY_LABEL,
                    f"{m:.2f}", ha="center", va="bottom", fontsize=7.0)
            max_label_top = max(max_label_top, ytop + DY_LABEL)
    ax.set_xticks(xs)
    ax.set_xticklabels([lbl for _, lbl in metrics])
    # Extra headroom above the tallest label so it clears the (loc='left') title.
    ax.set_ylim(Y_LO, max(Y_HI, max_label_top + (Y_HI - Y_LO) * 0.06))
    if show_ylabel:
        ax.set_ylabel(r"Accuracy (\%)" if _tex() else "Accuracy (%)")
    if title:
        ax.set_title(title, loc="left", fontweight="bold", pad=8)
    ax.legend(loc="lower right", frameon=True, fontsize=7.5)


def draw_kws_bars(ax, show_ylabel=True, title="(a) Keyword Spotting"):
    """Grouped bars: PE ∈ {no PE, sinusoidal, learned abs.} × {softmax, oscillator}.

    Mean printed above each bar (fig_bidirectional style), thin ±std error bar
    (std over the 5 seeds). No per-seed dots, no delta annotations."""
    data = load_e1()
    xs = np.arange(len(PES))
    w = 0.38
    for i, (mkey, mlabel, color) in enumerate(MECHS):
        means, stds = [], []
        for pe in PES:
            vals = data[(mkey, pe)]
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals, ddof=1)))
        pos = xs + (i - 0.5) * w
        bars = ax.bar(pos, means, width=w, color=color,
                      yerr=stds, capsize=3, error_kw={"lw": 1.0},
                      edgecolor="white", linewidth=0.5, label=mlabel)
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, m + s + DY_LABEL,
                    f"{m:.2f}", ha="center", va="bottom", fontsize=7.0)
    ax.set_xticks(xs)
    ax.set_xticklabels([PE_LABEL[p] for p in PES])
    ax.set_ylim(Y_LO, Y_HI)
    if show_ylabel:
        ax.set_ylabel(r"Accuracy (\%)" if _tex() else "Accuracy (%)")
    if title:
        ax.set_title(title, loc="left", fontweight="bold", pad=8)
    ax.legend(loc="lower right", frameon=True, fontsize=7.5)


def draw_sva_bars(ax, show_ylabel=True, title="(b) Subject--Verb Agreement"):
    """One bar per mechanism at the 50-seed MEAN hard accuracy, with a ±std error bar.

    Style-identical to draw_kws_bars: same colors, same [60,100] y-range, mean printed
    above each bar in the same label style. The error bars (std over the 50 seeds,
    ≈9.28 softmax / ≈5.37 oscillator) differ visibly — that difference is the point.
    No threshold line, no per-seed dots, no in-plot annotations (failure counts -> caption)."""
    hard = load_e2_hard()
    w = 0.55
    for i, (label, color) in enumerate([("softmax", ps.COLOR_SOFTMAX),
                                        ("oscillator", ps.COLOR_OSCILLATOR)]):
        vals = hard[label]
        m = float(np.mean(vals))
        s = float(np.std(vals, ddof=1))
        ax.bar(i, m, width=w, color=color, yerr=s, capsize=3,
               error_kw={"lw": 1.0}, edgecolor="white", linewidth=0.5)
        ax.text(i, m + s + DY_LABEL, f"{m:.2f}", ha="center", va="bottom", fontsize=7.0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Softmax", "Oscillator"])
    ax.set_ylim(Y_LO, Y_HI)
    if show_ylabel:
        ax.set_ylabel(r"Hard accuracy (\%)" if _tex() else "Hard accuracy (%)")
    if title:
        ax.set_title(title, loc="left", fontweight="bold", pad=8)


def draw_sva_box(ax, show_ylabel=True, title="(b) Subject--Verb Agreement",
                 threshold=FAIL_THRESH):
    """Box-and-whisker of the 50-seed hard accuracies per mechanism.

    Box = IQR + median; standard whiskers; seeds beyond the whiskers appear as
    small outlier markers (the failed seeds). ``threshold`` (default 85) draws a
    dashed failure-threshold line + legend; pass ``threshold=None`` to omit it
    (used by the appendix distribution figure). No in-plot annotations."""
    hard = load_e2_hard()
    for i, (label, color) in enumerate([("softmax", ps.COLOR_SOFTMAX),
                                        ("oscillator", ps.COLOR_OSCILLATOR)]):
        ax.boxplot(
            [hard[label]], positions=[i], widths=0.5, patch_artist=True,
            boxprops=dict(facecolor=color, alpha=0.55, edgecolor=color, lw=1.0),
            medianprops=dict(color="black", lw=1.4),
            whiskerprops=dict(color=color, lw=1.0),
            capprops=dict(color=color, lw=1.0),
            flierprops=dict(marker="o", markersize=3.2, markerfacecolor=color,
                            markeredgecolor=color, alpha=0.75, linestyle="none"),
        )
    if threshold is not None:
        ax.axhline(threshold, ls="--", color=ps.COLOR_ACCENT, lw=1.0,
                   label=(rf"{threshold:.0f}\% failure threshold" if _tex()
                          else f"{threshold:.0f}% failure threshold"))
        ax.legend(loc="lower right", frameon=True, fontsize=7.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Softmax", "Oscillator"])
    ax.set_ylim(Y_LO, Y_HI)
    if show_ylabel:
        ax.set_ylabel(r"Hard accuracy (\%)" if _tex() else "Hard accuracy (%)")
    if title:
        ax.set_title(title, loc="left", fontweight="bold", pad=8)


def failure_counts():
    """(for captions) -> {'softmax': (n_fail, n), 'oscillator': (n_fail, n)}."""
    hard = load_e2_hard()
    return {k: (sum(1 for v in vs if v < FAIL_THRESH), len(vs)) for k, vs in hard.items()}
