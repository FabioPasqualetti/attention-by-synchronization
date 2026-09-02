"""lm_wikitext_extrapolation — WikiText-2 out-of-sample test + variance-source check + TS symmetry.

Fixes the statistical framing: a bootstrap CI on the fit's PREDICTION is compared against the
observed MEAN with its own standard error (prediction interval = fit uncertainty (+) observation
noise). Extends WT2 d_osc=64 to 10 seeds, adds a 3-seed softmax WT2 baseline, and reruns the
same proper test on TinyStories for symmetry. Resumable; all seeds reported.
"""
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training import harness, paths  # noqa: E402

paths.ensure_paths()
import numpy as np  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402
from scipy.stats import norm  # noqa: E402

EXP = "lm_wikitext_extrapolation"
REPO = paths.REPO_ROOT
CKPT = os.path.join(harness.RUNS_ROOT, EXP, "ckpt")
RNG = np.random.default_rng(0)

WT2_D64_SEEDS = list(range(10))          # 0..9 (0,1,2 reused)
WT2_BASE_SEEDS = [0, 1, 2]


# ---------- data loaders (existing repo results) ----------

def wt2_scaling():
    """d_osc -> list of per-seed WT2 PPLs for d in {2,4,8,16,32}."""
    out = {}
    for d, p in [(2, "results/wikitext2_d2_5seeds.json"),
                 (8, "results/lm/wikitext2_d8_5seeds.json"),
                 (32, "results/lm/wikitext2_d32_5seeds.json")]:
        out[d] = [float(v) for v in json.load(open(os.path.join(REPO, p)))["seeds"].values()]
    for d, patt in [(4, "wt2_dosc4_seed*.json"), (16, "wt2_dosc16_seed*.json")]:
        out[d] = [float(json.load(open(f))["val_ppl"])
                  for f in sorted(glob.glob(os.path.join(paths.results_root(), "lm_5pt_runs", patt)))]
    return out


def ts_scaling():
    out = {}
    for d, p in [(2, "results/tinystories_d2_3seeds.json"),
                 (8, "results/tinystories_d8_3seeds.json"),
                 (32, "results/tinystories_d32_3seeds.json")]:
        out[d] = [float(r["val_ppl"]) for r in json.load(open(os.path.join(REPO, p)))]
    for d, patt in [(4, "ts_dosc4_seed*.json"), (16, "ts_dosc16_seed*.json")]:
        out[d] = [float(json.load(open(f))["val_ppl"])
                  for f in sorted(glob.glob(os.path.join(paths.results_root(), "lm_5pt_runs", patt)))]
    return out


def ts_d64():
    ppls = [harness.load_reference("lm_tinystories_extrapolation", "train_d64_s0")["val_ppl"]]
    for s in (1, 2):
        ppls.append(harness.load_reference("lm_tinystories_extrapolation", f"train_d64_s{s}")["val_ppl"])
    return ppls


# ---------- training ----------

def train_wt2(wt, d_osc, attn_type, seed, device, key):
    if harness.exists(EXP, key):
        return harness.load_result(EXP, key)["val_ppl"]
    t0 = time.time()
    r = wt.run_one(d_osc, attn_type, seed, device)
    harness.save_result(EXP, key, {"d_osc": d_osc, "attn_type": attn_type, "seed": seed,
                                   "val_ppl": r["val_ppl"], "wall_sec": round(time.time() - t0, 1)})
    harness.free_memory(device)
    print(f"[{key}] PPL={r['val_ppl']:.3f}", flush=True)
    return r["val_ppl"]


def collect_wt2_d64(wt, device):
    # seed 0 from wikitext_extrapolation.py, seeds 1-2 from wikitext_extrapolation_seeds.py:
    # other drivers, so published values -> reference.
    ppls = {0: harness.load_reference("lm_wikitext_extrapolation", "train_wt2_d64_s0")["val_ppl"],
            1: harness.load_reference("lm_wikitext_extrapolation", "train_wt2_d64_s1")["val_ppl"],
            2: harness.load_reference("lm_wikitext_extrapolation", "train_wt2_d64_s2")["val_ppl"]}
    for s in WT2_D64_SEEDS:
        if s in ppls:
            continue
        ppls[s] = train_wt2(wt, 64, "kuramoto", s, device, f"train_wt2_d64_s{s}")
    return [ppls[s] for s in WT2_D64_SEEDS]


def collect_wt2_baseline(wt, device):
    return [train_wt2(wt, 0, "softmax", s, device, f"train_wt2_softmax_s{s}")
            for s in WT2_BASE_SEEDS]


# ---------- stats ----------

def _powerlaw_logfit(dvals, deltas):
    ld, lD = np.log(np.asarray(dvals, float)), np.log(np.asarray(deltas, float))
    popt, _ = curve_fit(lambda x, lC, na: lC + na * x, ld, lD, p0=[1.0, -0.5])
    logC, neg_a = popt
    resid = lD - (logC + neg_a * ld)
    ss_res = float(np.sum(resid ** 2)); ss_tot = float(np.sum((lD - lD.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(np.exp(logC)), float(-neg_a), r2  # C, alpha, r2


def proper_test(scaling, d64_ppls, base_ppls, target=64, n_boot=2000, label=""):
    dvals = sorted(scaling)  # {2,4,8,16,32}
    base_mean = float(np.mean(base_ppls)); n_base = len(base_ppls)
    base_sem = float(np.std(base_ppls, ddof=1) / np.sqrt(n_base)) if n_base > 1 else 0.0

    # point fit on multi-seed means
    means = {d: float(np.mean(scaling[d])) for d in dvals}
    deltas = [means[d] - base_mean for d in dvals]
    C, alpha, r2 = _powerlaw_logfit(dvals, deltas)
    pred_delta_point = C * target ** (-alpha)

    # bootstrap: resample per-seed PPLs within each d + baseline seeds
    boot = []
    boot_alphas = []
    for _ in range(n_boot):
        bmean = float(np.mean(RNG.choice(base_ppls, size=n_base, replace=True))) if n_base > 1 else base_mean
        dd = []
        ok = True
        for d in dvals:
            m = float(np.mean(RNG.choice(scaling[d], size=len(scaling[d]), replace=True)))
            val = m - bmean
            if val <= 0:
                ok = False; break
            dd.append(val)
        if not ok:
            continue
        try:
            Cb, ab, _ = _powerlaw_logfit(dvals, dd)
            boot.append(Cb * target ** (-ab)); boot_alphas.append(ab)
        except Exception:
            continue
    boot = np.asarray(boot)
    boot_alphas = np.asarray(boot_alphas)
    pred_median = float(np.median(boot))
    sigma_pred = float(np.std(boot))
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    alpha_ci = [float(np.percentile(boot_alphas, 2.5)), float(np.percentile(boot_alphas, 97.5))]

    # observed
    d64 = np.asarray(d64_ppls, float)
    obs_delta = d64 - base_mean
    n_obs = len(d64)
    obs_mean = float(np.mean(obs_delta))
    obs_std = float(np.std(d64, ddof=1))
    sem_obs = float(obs_std / np.sqrt(n_obs))

    deviation = obs_mean - pred_median
    combined = float(np.sqrt(sem_obs ** 2 + sigma_pred ** 2))
    z = deviation / combined if combined > 0 else float("nan")
    p = float(2 * norm.sf(abs(z)))
    pi = [pred_median - 1.96 * combined, pred_median + 1.96 * combined]
    inside = pi[0] <= obs_mean <= pi[1]

    pct_ppl = 100.0 * deviation / (base_mean + pred_median)
    if p > 0.05 and inside:
        conclusion = (f"CONSISTENT with the extrapolation (z={z:.2f}, two-sided p={p:.3f}; "
                      f"observed mean Δ inside the 95% prediction interval).")
    else:
        conclusion = (f"SYSTEMATIC DEVIATION of {deviation:+.3f} in Δ ({pct_ppl:+.2f}% PPL); "
                      f"z={z:.2f}, p={p:.3f}, mean {'inside' if inside else 'outside'} the 95% PI.")

    return dict(label=label, dvals=dvals, base_mean=base_mean, base_sem=base_sem, n_base=n_base,
                alpha=alpha, C=C, r2=r2, pred_delta_point=pred_delta_point,
                pred_delta_median=pred_median, sigma_pred=sigma_pred, boot_ci95=ci,
                alpha_ci95=alpha_ci,
                alpha_ci95_method=("2.5/97.5 percentile of the fitted exponent alpha over the "
                                   "same per-seed bootstrap (RNG default_rng(0), n_boot=2000) "
                                   "used for the predicted-Delta CI"),
                predicted_ppl64=base_mean + pred_median,
                d64_ppls=list(map(float, d64)), obs_delta_by_seed=list(map(float, obs_delta)),
                obs_mean_delta=obs_mean, obs_std_ppl=obs_std, sem_obs=sem_obs, n_obs=n_obs,
                obs_mean_ppl64=float(np.mean(d64)),
                deviation=deviation, combined_uncertainty=combined, z=z, p_two_sided=p,
                pred_interval95=pi, obs_mean_inside_PI=bool(inside), pct_ppl=pct_ppl,
                conclusion=conclusion)


def variance_row(name, ppls):
    a = np.asarray(ppls, float)
    m = float(np.mean(a)); s = float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
    return dict(name=name, n=len(a), mean=m, std_abs=s, std_rel=(s / m if m else 0.0))


def main():
    device = harness.pick_device("mps")
    print(f"lm_wikitext_extrapolation device={device}", flush=True)
    os.makedirs(CKPT, exist_ok=True)
    wt = paths.load_py(os.path.join(REPO, "training", "train_wikitext.py"), "wt2_train_mod")
    wt.CKPT_DIR = CKPT

    # ---- training ----
    wt2_base = collect_wt2_baseline(wt, device)     # 3 fresh softmax seeds
    wt2_64 = collect_wt2_d64(wt, device)            # 10 seeds (reuse 0-2)

    # ---- WT2 proper test ----
    wt2sc = wt2_scaling()
    wt2 = proper_test(wt2sc, wt2_64, wt2_base, label="WikiText-2")

    # ---- TS symmetry (baseline single-seed 8.544) ----
    tssc = ts_scaling()
    ts = proper_test(tssc, ts_d64(), [8.544], label="TinyStories")

    # ---- variance-source ----
    var = {
        "WT2_softmax_baseline": variance_row("WT2 softmax baseline", wt2_base),
        "WT2_d32": variance_row("WT2 d_osc=32", wt2sc[32]),
        "WT2_d64": variance_row("WT2 d_osc=64 (10 seeds)", wt2_64),
        "TS_softmax_baseline": variance_row("TS softmax baseline", [8.544]),
        "TS_d32": variance_row("TS d_osc=32", tssc[32]),
        "TS_d64": variance_row("TS d_osc=64 (3 seeds)", ts_d64()),
    }
    wt2_grows = var["WT2_d64"]["std_abs"] > 1.5 * var["WT2_d32"]["std_abs"]
    ts_grows = var["TS_d64"]["std_abs"] > 1.5 * var["TS_d32"]["std_abs"]
    if not wt2_grows and not ts_grows and var["WT2_d64"]["std_abs"] > 2 * max(var["TS_d64"]["std_abs"], 1e-9):
        var_verdict = ("DATASET-linked: seed variance is roughly uniform across d_osc within a "
                       "dataset but far larger on WikiText-2 than TinyStories.")
    elif wt2_grows or ts_grows:
        var_verdict = "DIMENSION-linked: seed variance grows with d_osc."
    else:
        var_verdict = ("Mostly DATASET-linked: variance does not grow strongly with d_osc; "
                       "WT2 >> TS in absolute and relative seed std.")

    harness.save_result(EXP, "analysis", wt2)
    harness.save_result(EXP, "ts_analysis", ts)
    harness.save_result(EXP, "variance", {"rows": var, "verdict": var_verdict})

    # ---- print transcribable block ----
    def fmt(d):
        return (f"  fit: alpha={d['alpha']:.4f} C={d['C']:.4f} R2={d['r2']:.4f}\n"
                f"  predicted Δ(64): median={d['pred_delta_median']:.4f} sigma_pred={d['sigma_pred']:.4f} "
                f"boot95CI={[round(x,4) for x in d['boot_ci95']]} -> predicted PPL(64)={d['predicted_ppl64']:.4f}\n"
                f"  observed Δ(64): mean={d['obs_mean_delta']:.4f} ± SEM {d['sem_obs']:.4f} "
                f"(std {d['obs_std_ppl']:.4f}, n={d['n_obs']}) -> PPL(64)={d['obs_mean_ppl64']:.4f}\n"
                f"  TEST: deviation={d['deviation']:+.4f} combined={d['combined_uncertainty']:.4f} "
                f"z={d['z']:.3f} p={d['p_two_sided']:.4f}\n"
                f"  95% prediction interval (Δ)={[round(x,4) for x in d['pred_interval95']]} "
                f"inside={d['obs_mean_inside_PI']}\n  CONCLUSION: {d['conclusion']}")
    print("\n================ lm_wikitext_extrapolation SUMMARY ================")
    print(f"\nWT2 d64 PPLs (10 seeds): {[round(x,3) for x in wt2_64]}")
    print(f"WT2 softmax baseline (3 seeds): {[round(x,3) for x in wt2_base]} -> "
          f"mean {wt2['base_mean']:.4f} ± SEM {wt2['base_sem']:.4f}")
    print("WT2 proper test:\n" + fmt(wt2))
    print(f"\nTS d64 PPLs (3 seeds): {[round(x,3) for x in ts_d64()]}  (baseline single-seed 8.544)")
    print("TS proper test:\n" + fmt(ts))
    print("\nVARIANCE (per-seed std):")
    for k, r in var.items():
        print(f"  {r['name']:<28} n={r['n']} mean={r['mean']:.3f} std_abs={r['std_abs']:.4f} "
              f"std_rel={100*r['std_rel']:.3f}%")
    print(f"VARIANCE VERDICT: {var_verdict}")
    print("\nlm_wikitext_extrapolation stats COMPLETE", flush=True)


if __name__ == "__main__":
    main()
