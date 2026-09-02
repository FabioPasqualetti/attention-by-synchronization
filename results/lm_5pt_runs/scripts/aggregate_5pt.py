"""
Aggregate the 5-point d_osc scaling-law extension and refit the power law.

Combines the d_osc in {4,16} runs in this directory with the d_osc in {2,8,32}
5-seed baselines elsewhere in results/. The softmax baselines (WT2=99.58,
TS=8.54) are read from stored results, not retrained.

Fit:  log Δ = log C − α·log d_osc,  Δ(d) = mean_PPL(d) − softmax_PPL,
fit on the per-d means by linear least squares in log-log space (the same estimator as
language_modeling/wikitext_extrapolation_stats.py::_powerlaw_logfit, via scipy.curve_fit). Reports
C, α and its standard error (from the fit covariance), R²(log), and per-point
residuals — for both the 3-point and the 5-point fits.

Reads the per-seed JSONs from the results root; writes runs/lm_5pt_runs/aggregate.json.
"""

import glob
import json
import math
import os
import statistics
import sys

import numpy as np
from scipy.optimize import curve_fit

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
RESULTS = os.environ.get("OSCILLATOR_RESULTS", os.path.join(ROOT, "results"))
RUNS    = os.environ.get("OSCILLATOR_RUNS", os.path.join(ROOT, "runs"))
NEW = os.path.join(RESULTS, "lm_5pt_runs")

SOFTMAX = {"WikiText-2": 99.58, "TinyStories": 8.54}
PREV_ALPHA = {"WikiText-2": 0.47, "TinyStories": 0.52}
DS_TOKEN = {"WikiText-2": "wt2", "TinyStories": "ts"}

# Existing 5-seed baseline files for d in {2,8,32}.
EXISTING = {
    "WikiText-2": {
        2:  os.path.join(RESULTS, "wikitext2_d2_5seeds.json"),
        8:  os.path.join(RESULTS, "lm", "wikitext2_d8_5seeds.json"),
        32: os.path.join(RESULTS, "lm", "wikitext2_d32_5seeds.json"),
    },
    "TinyStories": {
        2:  os.path.join(RESULTS, "lm", "tinystories_d2_5seeds.json"),
        8:  os.path.join(RESULTS, "lm", "tinystories_d8_5seeds.json"),
        32: os.path.join(RESULTS, "lm", "tinystories_d32_5seeds.json"),
    },
}


def existing_point(path):
    d = json.load(open(path))
    return {"mean": d["mean"], "std": d.get("std", 0.0),
            "n": d.get("n_seeds"), "source": os.path.relpath(path, ROOT)}


def new_point(dataset, d_osc):
    token = DS_TOKEN[dataset]
    files = sorted(glob.glob(os.path.join(NEW, f"{token}_dosc{d_osc}_seed*.json")))
    ppls, seeds = [], {}
    for f in files:
        r = json.load(open(f))
        if r.get("val_ppl") is not None:
            ppls.append(r["val_ppl"])
            seeds[str(r["seed"])] = r["val_ppl"]
    if not ppls:
        return None
    return {"mean": statistics.mean(ppls),
            "std": statistics.stdev(ppls) if len(ppls) > 1 else 0.0,
            "n": len(ppls), "seeds": seeds,
            "source": f"{len(ppls)} files: {token}_dosc{d_osc}_seed*.json"}


def _lin(log_d, log_C, neg_a):
    return log_C + neg_a * log_d


def fit(d_values, means, softmax_ppl):
    deltas = [m - softmax_ppl for m in means]
    if any(x <= 0 for x in deltas):
        return {"error": "non-positive Δ; cannot fit in log space",
                "deltas": deltas}
    log_d = np.log(np.array(d_values, float))
    log_delta = np.log(np.array(deltas, float))
    popt, pcov = curve_fit(_lin, log_d, log_delta, p0=[1.0, -0.5])
    log_C, neg_a = popt
    se_log_C, se_neg_a = np.sqrt(np.diag(pcov))
    pred = _lin(log_d, *popt)
    ss_res = float(np.sum((log_delta - pred) ** 2))
    ss_tot = float(np.sum((log_delta - log_delta.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    resid = {int(d): {"delta_obs": float(de),
                      "delta_pred": float(math.exp(p)),
                      "resid_log": float(ld - p)}
             for d, de, p, ld in zip(d_values, deltas, pred, log_delta)}
    return {"C": math.exp(log_C), "alpha": -neg_a,
            "alpha_stderr": float(se_neg_a), "C_stderr_log": float(se_log_C),
            "r2_log": r2, "n_points": len(d_values), "residuals": resid}


def main():
    out = {"softmax_baselines": SOFTMAX, "datasets": {}}
    for ds, sm in SOFTMAX.items():
        print(f"\n{'='*70}\n{ds}   (softmax baseline PPL = {sm})\n{'='*70}")
        points = {}
        for d in (2, 8, 32):
            points[d] = existing_point(EXISTING[ds][d]); points[d]["origin"] = "existing"
        for d in (4, 16):
            p = new_point(ds, d)
            if p:
                p["origin"] = "new"
                points[d] = p

        print(f"  {'d_osc':>5}  {'Osc PPL (mean±std)':>22}  {'softmax':>8}  "
              f"{'Δ':>8}  {'n':>2}  origin")
        for d in sorted(points):
            p = points[d]
            print(f"  {d:>5}  {p['mean']:>10.4f} ± {p['std']:<7.4f}  {sm:>8.2f}  "
                  f"{p['mean']-sm:>8.4f}  {p['n']:>2}  {p['origin']}")

        ds_out = {"softmax_ppl": sm, "points": points, "fits": {}}
        for tag, dvals in [("3point_d2_8_32", [2, 8, 32]),
                           ("5point_d2_4_8_16_32", [2, 4, 8, 16, 32])]:
            if all(d in points for d in dvals):
                f = fit(dvals, [points[d]["mean"] for d in dvals], sm)
                ds_out["fits"][tag] = f
                if "error" not in f:
                    print(f"\n  {tag:>22}:  Δ = {f['C']:.2f} · d^(-{f['alpha']:.3f}"
                          f" ± {f['alpha_stderr']:.3f})   R²(log)={f['r2_log']:.4f}")
                    print("      residuals(log): " + ", ".join(
                        f"d{d}:{f['residuals'][d]['resid_log']:+.4f}" for d in dvals))
            else:
                missing = [d for d in dvals if d not in points]
                ds_out["fits"][tag] = {"error": f"missing d_osc {missing}"}
                print(f"\n  {tag}: incomplete (missing d_osc {missing})")

        prev = PREV_ALPHA[ds]
        f5 = ds_out["fits"].get("5point_d2_4_8_16_32", {})
        if "alpha" in f5:
            da = f5["alpha"] - prev
            within = abs(da) <= f5["alpha_stderr"]
            print(f"\n  prev 3-pt α(reported)={prev}; new 5-pt α={f5['alpha']:.3f}"
                  f" ± {f5['alpha_stderr']:.3f}  (Δα={da:+.3f}, "
                  f"{'within' if within else 'OUTSIDE'} 1 stderr)")
        out["datasets"][ds] = ds_out

    out_path = os.path.join(RUNS, "lm_5pt_runs", "aggregate.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved aggregate -> {out_path}")


if __name__ == "__main__":
    main()
