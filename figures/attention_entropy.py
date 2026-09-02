"""Attention entropy + SVA seed distribution -> figures/output/fig_entropy_seeds.pdf.

Two-panel appendix figure (TWO_PANEL size, plain left-aligned panel titles, no letter prefixes):

  left  — normalized attention-row entropy per mechanism / d_osc (violin + median/IQR box overlay;
          dashed reference at 1.0 = near-uniform / max-entropy limit).
  right — SVA 50-seed hard-accuracy distribution per mechanism (box-and-whisker; failed seeds as
          outliers; NO 85% threshold line -- that lives on the main-text bidirectional figure).
          Reuses bidirectional_common.draw_sva_box.

Inputs (every plotted number from result JSONs):
  left  — results/lm_attention_entropy/raw_{softmax,osc_d2,osc_d4,osc_d8,osc_d16}.json
          (fields: mechanism, d_osc, samples:[...])
  right — results/sva_seed_robustness/main_{softmax,kuramoto}_s{0..49}.json  (val_acc.hard, %)

Run standalone from the repo root:
  python figures/attention_entropy.py
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import style as ps
import bidirectional_common as c



def _load_entropy():
    items = []
    for p in glob.glob(os.path.join(ps.results_root(), "lm_attention_entropy/raw_*.json")):
        d = json.load(open(p))
        if not d.get("samples"):
            continue
        items.append(d)
    # order: softmax first, then oscillator by ascending d_osc
    def key(d):
        return (0, 0) if d["mechanism"] == "softmax" else (1, d.get("d_osc") or 0)
    return sorted(items, key=key)


def draw_entropy(ax, show_ylabel=True, title="attention entropy"):
    """Left panel — attention-row entropy violins (styling identical to former fig11)."""
    items = _load_entropy()
    data, colors, labels = [], [], []
    for d in items:
        data.append(np.asarray(d["samples"], dtype=float))
        if d["mechanism"] == "softmax":
            colors.append(ps.COLOR_SOFTMAX); labels.append("softmax")
        else:
            colors.append(ps.COLOR_OSCILLATOR); labels.append(rf"osc $d\!=\!{d['d_osc']}$")

    pos = range(1, len(data) + 1)
    vp = ax.violinplot(data, positions=pos, showextrema=False, widths=0.75)
    for body, col in zip(vp["bodies"], colors):
        body.set_facecolor(col); body.set_alpha(0.45); body.set_edgecolor(col)
        body.set_linewidth(0.8)

    box_stats = []
    for v in data:
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        box_stats.append({"med": med, "q1": q1, "q3": q3, "whislo": q1, "whishi": q3,
                          "fliers": []})
    bp = ax.bxp(box_stats, positions=list(pos), showmeans=False, showfliers=False,
                patch_artist=True, widths=0.14, zorder=5)
    for patch in bp["boxes"]:
        patch.set_facecolor("white"); patch.set_alpha(0.85); patch.set_edgecolor("black")
        patch.set_linewidth(0.7)
    for med in bp["medians"]:
        med.set_color("black"); med.set_linewidth(1.3)
    for wk in bp["whiskers"] + bp["caps"]:
        wk.set_color("black"); wk.set_linewidth(0.7)

    ax.axhline(1.0, color=ps.COLOR_ACCENT, ls="--", lw=1.2, label="uniform (max. entropy)")
    ax.set_xticks(list(pos))
    ax.set_xticklabels(labels, rotation=30, ha="right", rotation_mode="anchor", fontsize=7.5)
    if show_ylabel:
        ax.set_ylabel("normalized row entropy")
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title(title, loc="left", fontweight="bold", pad=8)


def main():
    ps.apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=ps.TWO_PANEL)
    fig.subplots_adjust(wspace=0.34, bottom=0.16, top=0.88)
    draw_entropy(ax1, show_ylabel=True, title="attention entropy")
    c.draw_sva_box(ax2, show_ylabel=True, title="SVA hard accuracy (50 seeds)", threshold=None)
    ps.save(fig, "fig_entropy_seeds")


if __name__ == "__main__":
    main()
