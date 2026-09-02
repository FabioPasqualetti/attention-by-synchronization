"""
Experiment 1: Validate Proposition 4 (antipodal initialization probability).

Section 3.2 of the paper gives the closed-form expression:
  P(angle(k(0), -k*) < alpha) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2))
                                 * integral_0^alpha sin^{d-2}(theta) d theta

We validate it empirically by drawing N=1000 uniform samples from S^{d-1}
and comparing the empirical fraction angle < alpha against the formula.

Usage:
    python theory/antipodal_escape.py
    python theory/antipodal_escape.py --N 50000 --seed 42
"""

import argparse
import json
import math
import os
import sys

import numpy as np
from scipy.integrate import quad
from scipy.special import gamma

# Allow running as `python theory/antipodal_escape.py`
# from the repo root
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
from training import harness, paths  # noqa: E402

RESULTS_DIR = os.path.join(paths.runs_root(), "theory_validation")
FIGURES_DIR = os.path.join(ROOT, "figures", "output")

DOSC_VALUES = [2, 4, 8, 16, 32, 64]
N_ALPHA     = 50


# ── Core computations ─────────────────────────────────────────────────────────

def antipodal_theory(alpha: float, d: int) -> float:
    """Proposition 4 closed-form probability."""
    prefactor = gamma(d / 2) / (math.sqrt(math.pi) * gamma((d - 1) / 2))
    val, _ = quad(lambda t: math.sin(t) ** (d - 2), 0, alpha)
    return prefactor * val


def run_simulation(dosc_values, n_alpha, N, seed):
    """
    Returns {dosc: {alpha: [empirical, predicted]}} for all (dosc, alpha) pairs.
    """
    rng = np.random.default_rng(seed)
    alpha_grid = np.linspace(0.01, math.pi / 2, n_alpha)
    results = {}

    for d in dosc_values:
        print(f"  d_osc={d}: drawing {N} samples from S^{d-1}...", flush=True)
        samples = rng.standard_normal(size=(N, d))
        norms = np.linalg.norm(samples, axis=1, keepdims=True)
        samples = samples / np.maximum(norms, 1e-12)

        # pole = e_1; dot product with pole is just the first coordinate
        cos_a = np.clip(samples[:, 0], -1.0, 1.0)
        angles = np.arccos(cos_a)

        results[d] = {}
        for alpha in alpha_grid:
            empirical   = float((angles < alpha).mean())
            theoretical = antipodal_theory(float(alpha), d)
            results[d][float(alpha)] = [empirical, theoretical]

    return results, alpha_grid


# ── Figure generation ─────────────────────────────────────────────────────────

def make_figure(results, alpha_grid, dosc_values, out_paths):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("  matplotlib/seaborn not available — skipping figure generation")
        return

    sns.set_theme(style="whitegrid", font_scale=1.1)
    palette = sns.color_palette("tab10", n_colors=len(dosc_values))

    fig, ax = plt.subplots(figsize=(6, 4))

    for i, d in enumerate(dosc_values):
        empir = [results[d][float(a)][0] for a in alpha_grid]
        theor = [results[d][float(a)][1] for a in alpha_grid]
        color = palette[i]
        ax.plot(alpha_grid, theor,  color=color, lw=1.8, label=f"$d={d}$ (theory)")
        ax.plot(alpha_grid, empir,  color=color, lw=0, marker=".", markersize=3,
                alpha=0.6, label="_nolegend_")

    ax.set_xlabel(r"$\alpha$ (radians)")
    ax.set_ylabel(r"$P(\angle(k(0),\,-k^{\!*}) < \alpha)$")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    fig.tight_layout()

    for path in out_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved: {path}")
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validate Proposition 4 (antipodal init probability)")
    parser.add_argument("--N",      type=int, default=1000,
                        help="Monte Carlo samples per d_osc (default 1000, matching the committed markers)")
    parser.add_argument("--seed",   type=int, default=0,
                        help="Random seed (default 0)")
    parser.add_argument("--no-fig", action="store_true",
                        help="Skip figure generation")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print(f"\nAntipodal simulation  N={args.N}  seed={args.seed}")
    print(f"  d_osc values: {DOSC_VALUES}\n")

    results, alpha_grid = run_simulation(DOSC_VALUES, N_ALPHA, args.N, args.seed)

    # Save JSON
    json_path = os.path.join(RESULTS_DIR, "antipodal_data.json")
    # JSON keys must be strings
    json_out = {str(d): {str(a): v for a, v in inner.items()}
                for d, inner in results.items()}
    harness.guarded_dump(json_path, json_out)
    print(f"\nData saved: {json_path}")

    # Summary table
    print(f"\n{'d_osc':>6}  {'alpha=pi/8':>12}  {'theory':>10}  {'delta':>8}")
    check_alpha = math.pi / 8
    for d in DOSC_VALUES:
        keys = sorted(results[d].keys())
        closest = min(keys, key=lambda a: abs(a - check_alpha))
        emp, th = results[d][closest]
        delta = abs(emp - th)
        print(f"  {d:>4}  {closest:>12.4f}  {emp:>10.6f}  {th:>10.6f}  {delta:>8.6f}")

    # Figure
    if not args.no_fig:
        fig_pdf = os.path.join(RESULTS_DIR, "antipodal_validation.pdf")
        fig_copy = os.path.join(FIGURES_DIR, "fig_antipodal_validation.pdf")
        make_figure(results, alpha_grid, DOSC_VALUES, [fig_pdf, fig_copy])

    print("\nExperiment 1 complete.")


if __name__ == "__main__":
    main()
