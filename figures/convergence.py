"""ODE convergence budget -> figures/output/fig_convergence_tmax_dosc.pdf.

Two panels (TWO_PANEL size, plain left-aligned titles): fraction of oscillator states converged vs.
the integration horizon T_max, and the per-d_osc failure rate under the adaptive integrator. Reads
the cached convergence JSONs, no compute:
  results/convergence/L6_ode_verify_extended.json
  results/convergence/TS_verify_adaptive.json

Run standalone from the repo root:
  python figures/convergence.py
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import style as ps

CONV_DIR = os.path.join(ps.results_root(), "convergence")
EXT_JSON = os.path.join(CONV_DIR, "L6_ode_verify_extended.json")
ADP_JSON = os.path.join(CONV_DIR, "TS_verify_adaptive.json")


def main():
    for path in (EXT_JSON, ADP_JSON):
        if not os.path.exists(path):
            print(f"Data not found: {path}")
            sys.exit(1)
    ps.apply_style()
    ext = json.load(open(EXT_JSON))
    adp = json.load(open(ADP_JSON))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=ps.TWO_PANEL)
    fig.subplots_adjust(wspace=0.49, top=0.88)

    # left: fraction converged vs T_max
    fbt = ext["frac_by_tmax"]
    t_vals = sorted(int(k) for k in fbt)
    fracs = [fbt[str(t)] for t in t_vals]
    ax1.semilogx(t_vals, fracs, "o-", color=ps.COLOR_OSCILLATOR, lw=1.8, ms=5)
    ax1.axhline(0.95, color=ps.COLOR_ACCENT, lw=1.2, ls="--", label=r"$95\%$ threshold")
    ax1.set_xlabel(r"$T_{\rm max}$ (log scale)")
    ax1.set_ylabel("Fraction converged\n" + r"(err $< 0.01$)")
    ax1.set_title("convergence vs integration horizon", loc="left", fontweight="bold")
    ax1.set_ylim(0.80, 1.02)
    ax1.legend(loc="lower right")
    for t, f in zip(t_vals, fracs):
        if abs(f - 0.95) < 0.015:
            ax1.annotate(f"{f:.2f}", (t, f), textcoords="offset points", xytext=(8, -10),
                         fontsize=7.5)
        else:
            ax1.annotate(f"{f:.2f}", (t, f), textcoords="offset points", xytext=(4, 5),
                         fontsize=7.5)

    # right: failure rates by d_osc
    cond_map = {"TS2_d2": 2, "TS3_d8": 8, "TS4_d32": 32}
    d_oscs, ap_rates, deg_rates = [], [], []
    for cname, d_osc in sorted(cond_map.items(), key=lambda x: x[1]):
        v = adp[cname].get("uniqueness", adp[cname])
        ap = v["antipodal_rate"]
        frac_nc = 1.0 - v["frac_lt_001"]
        d_oscs.append(d_osc)
        ap_rates.append(ap * 100)
        deg_rates.append(max(0.0, frac_nc - ap) * 100)

    n = len(d_oscs); w = 0.35; xs = np.arange(n)
    ax2.bar(xs - w / 2, ap_rates, width=w, color=ps.COLOR_OSCILLATOR,
            edgecolor="white", linewidth=0.5, label=r"Slow convergence (err $> 0.1$)")
    ax2.bar(xs + w / 2, deg_rates, width=w, color=ps.COLOR_ACCENT,
            edgecolor="white", linewidth=0.5, label=r"Near threshold (err $\in [0.01, 0.1]$)")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([str(d) for d in d_oscs])
    ax2.set_xlabel(r"$d_{\rm osc}$")
    ax2.set_ylabel(r"Failure rate (\%)")
    ax2.set_title("convergence within fixed budget", loc="left", fontweight="bold")
    ax2.legend(loc="upper right", bbox_to_anchor=(1.0, 0.93), ncol=1, frameon=True,
               fontsize=7.5, handlelength=1.2, borderpad=0.4)

    fig.tight_layout()
    ps.save(fig, "fig_convergence_tmax_dosc")


if __name__ == "__main__":
    main()
