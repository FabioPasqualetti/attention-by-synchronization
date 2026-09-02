"""Antipodal-initialization validation -> figures/output/fig_antipodal_validation.pdf.

Proposition 4: empirical vs. theoretical probability that an oscillator initialized within angle
alpha of the antipode still escapes. Single-panel (SINGLE_PANEL size, one panel of the standard
two-panel layout); no in-figure title -- the paper caption carries it. Reads the cached JSON, no
compute:
  results/theory_validation/antipodal_data.json

Run standalone from the repo root:
  python figures/antipodal_escape.py
"""
import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import style as ps

DATA_PATH = os.path.join(ps.results_root(), "theory_validation", "antipodal_data.json")
D_OSC_ALL = [2, 4, 8, 16, 32, 64]


def main():
    if not os.path.exists(DATA_PATH):
        print(f"Data not found: {DATA_PATH}")
        sys.exit(1)
    ps.apply_style()
    with open(DATA_PATH) as f:
        data = json.load(f)

    d_osc_values = [d for d in D_OSC_ALL if str(d) in data]
    cmap = plt.colormaps[ps.CMAP]
    colors = [cmap(t) for t in np.linspace(0.05, 0.92, len(d_osc_values))]

    fig, ax = plt.subplots(figsize=ps.SINGLE_PANEL)
    for i, d in enumerate(d_osc_values):
        inner = data[str(d)]
        alphas = sorted(float(a) for a in inner.keys())
        empir = [inner[str(a)][0] for a in alphas]
        theor = [inner[str(a)][1] for a in alphas]
        ax.plot(alphas, theor, color=colors[i], lw=1.6, label=rf"$d_{{\rm osc}}={d}$")
        ax.plot(alphas, empir, color=colors[i], lw=0, marker=".", markersize=4, alpha=0.65,
                label="_nolegend_")

    ax.set_xlabel(r"$\alpha$ (radians)")
    ax.set_ylabel(r"$P(\angle(z_i(0), -z_i^*) < \alpha)$")
    ax.set_xlim(0, math.pi / 2)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), ncol=2, frameon=True,
              columnspacing=0.9, handlelength=1.4, handletextpad=0.4, borderpad=0.4)
    ps.save(fig, "fig_antipodal_validation")


if __name__ == "__main__":
    main()
