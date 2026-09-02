"""Physical-robustness grid -> figures/output/fig_robustness_grid.pdf.

2 rows (KWS accuracy; TinyStories perplexity) x 4 columns (coupling mismatch, state noise,
frequency disorder, finite settling). Coupling/state/settle from robustness_perturbations; frequency
disorder from robustness_frequency_disorder. For finite settling the x-axis is the settling horizon
T and the analytic fixed-point value is drawn as a horizontal dashed reference (computed on that
panel's own evaluation set).

Inputs:
  results/robustness_perturbations/perturbation_grid.csv  (type in {coupling,state,settle}; cols scale,T,acc,ppl)
  results/robustness_perturbations/{kws,ts}_baseline.json   (analytic fixed-point metric)
  results/robustness_frequency_disorder/{kws,ts}_s{scale}_d{draw}.json + {kws,ts}_baseline.json

Run standalone from the repo root:
  python figures/robustness_grid.py
"""
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import style as ps

TASKS = [("kws", "acc", "KWS accuracy"), ("ts", "ppl", "TinyStories PPL")]


def _load(rel):
    """`rel` is relative to the results root (see style.results_root)."""
    return json.load(open(os.path.join(ps.results_root(), rel)))


def e4(target, ptype, xcol, metric):
    agg = defaultdict(list)
    with open(os.path.join(ps.results_root(), "robustness_perturbations/perturbation_grid.csv")) as f:
        for r in csv.DictReader(f):
            if r["target"] == target and r["type"] == ptype and r[metric] and r[xcol]:
                agg[float(r[xcol])].append(float(r[metric]))
    xs = sorted(agg)
    return xs, [np.mean(agg[x]) for x in xs]


def e10(target, metric):
    agg = defaultdict(list)
    for path in glob.glob(os.path.join(ps.results_root(), f"robustness_frequency_disorder/{target}_s*_d*.json")):
        r = json.load(open(path))
        if metric in r:
            agg[float(r["s"])].append(float(r[metric]))
    xs = sorted(agg)
    return xs, [np.mean(agg[x]) for x in xs]


def main():
    ps.apply_style()
    fig, axes = plt.subplots(2, 4, figsize=(10.0, 4.6))
    cols = ["coupling mismatch", "state noise", "frequency disorder", "finite settling"]
    # Per-panel y-limits (row, col). KWS (row 0): >=2pp windows so robust-flat curves read as
    # flat; the two full-val panels (coupling, state) share a window, frequency-disorder uses its
    # own (512-subset baseline, ~90.2%). TS (row 1): comparable 7.5-8.1 window for the three mild
    # panels; TS state (1,1) left autoscaled (genuine large degradation, ppl -> ~11).
    # (0,3) KWS finite-settling: the "analytic FP" dashed reference is the MATCHED 512-example
    # subset baseline (robustness_frequency_disorder, ~90.2%) — the same eval set as the settle series — so the curve
    # converges onto its own fixed point (corrected from the earlier full-val robustness_perturbations ref, 88.9%, which
    # produced a spurious ~1.4pp gap). Shares the frequency-disorder window (also 512-subset,
    # ~90.2%) by the same >=2pp flatness rule.
    YLIM = {(0, 0): (87.9, 89.9), (0, 1): (87.9, 89.9), (0, 2): (89.2, 91.2), (0, 3): (89.2, 91.2),
            (1, 0): (7.5, 8.1), (1, 2): (7.5, 8.1), (1, 3): (7.5, 8.1)}
    for ri, (target, metric, ylab) in enumerate(TASKS):
        base4 = _load(f"robustness_perturbations/{target}_baseline.json")[metric]
        base10 = _load(f"robustness_frequency_disorder/{target}_baseline.json")[metric]
        scale = 100 if metric == "acc" else 1.0
        panels = [
            (*e4(target, "coupling", "scale", metric), base4, r"scale $s$"),
            (*e4(target, "state", "scale", metric), base4, r"scale $s$"),
            (*e10(target, metric), base10, r"scale $s$"),
            # settle series is on the 512-example subset -> reference is the matched-subset
            # baseline (base10: KWS 90.23%; TS 7.5946, identical to robustness_perturbations), each on its own eval set
            (*e4(target, "settle", "T", metric), base10, r"settling $T$"),
        ]
        for ci, (xs, ys, base, xlab) in enumerate(panels):
            ax = axes[ri, ci]
            is_settle = (ci == 3)
            if is_settle:
                ax.plot(xs, [y * scale for y in ys], "o-", color=ps.COLOR_OSCILLATOR)
                ax.axhline(base * scale, color=ps.COLOR_SOFTMAX, ls="--", lw=1.2,
                           label="analytic FP")
                ax.set_xscale("log")
                ax.legend(fontsize=6.5, loc="best")
            else:
                xx = [0.0] + list(xs)
                ax.plot(xx, [base * scale] + [y * scale for y in ys], "o-",
                        color=ps.COLOR_OSCILLATOR)
            if (ri, ci) in YLIM:
                ax.set_ylim(*YLIM[(ri, ci)])
            if ri == 0:
                ax.set_title(cols[ci], fontsize=9)
            if ri == 1:
                ax.set_xlabel(xlab)
            if ci == 0:
                unit = "(\\%)" if (metric == "acc" and plt.rcParams["text.usetex"]) else \
                       ("(%)" if metric == "acc" else "")
                ax.set_ylabel(f"{ylab} {unit}".strip())
    fig.subplots_adjust(wspace=0.51, hspace=0.32)
    ps.save(fig, "fig_robustness_grid")


if __name__ == "__main__":
    main()
