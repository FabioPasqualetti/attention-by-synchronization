"""lm_tinystories_extrapolation — Out-of-sample power-law check.

Train ONE oscillator TinyStories run at d_osc=64, seed 0 (ref config). Fit the scaling law
Delta = C * d_osc^(-alpha) on d_osc in {2,4,8,16,32} ONLY (Delta = mean PPL(d_osc) - softmax
baseline 8.544), bootstrap over seeds for a 95% CI on the d_osc=64 prediction, and report
predicted vs. observed Delta(64) and whether the observation falls inside the CI.

Resumable. No training beyond the single d_osc=64 run.
"""
import glob
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training import harness, paths  # noqa: E402
paths.ensure_paths()

import numpy as np  # noqa: E402

EXP = "lm_tinystories_extrapolation"
REPO = paths.REPO_ROOT
CKPT = os.path.join(harness.RUNS_ROOT, EXP, "ckpt")
SOFTMAX_PPL = 8.544
FIT_DOSC = [2, 4, 8, 16, 32]
N_BOOT = 2000


def _val_ppl_from(obj):
    if isinstance(obj, dict):
        for k in ("val_ppl", "ppl", "best_ppl"):
            if k in obj:
                return obj[k]
    return None


def load_seed_ppls():
    """d_osc -> list of per-seed val PPLs, from the paper's scaling files."""
    data = {}
    for d, path in [(2, "tinystories_d2_3seeds.json"),
                    (8, "tinystories_d8_3seeds.json"),
                    (32, "tinystories_d32_3seeds.json")]:
        recs = json.load(open(os.path.join(paths.results_root(), path)))
        data[d] = [r["val_ppl"] for r in recs]
    for d, patt in [(4, "ts_dosc4_seed*.json"), (16, "ts_dosc16_seed*.json")]:
        vals = []
        for p in sorted(glob.glob(os.path.join(paths.results_root(), "lm_5pt_runs", patt))):
            j = json.load(open(p))
            v = _val_ppl_from(j)
            if v is None and isinstance(j, list) and j:
                v = _val_ppl_from(j[0])
            if v is not None:
                vals.append(v)
        data[d] = vals
    return data


def fit_power(d_arr, delta_arr):
    """log Delta = log C - alpha log d  -> returns (C, alpha, R^2)."""
    log_d = np.log(np.asarray(d_arr, float))
    log_delta = np.log(np.asarray(delta_arr, float))
    A = np.vstack([np.ones_like(log_d), log_d]).T
    coef, *_ = np.linalg.lstsq(A, log_delta, rcond=None)
    logC, slope = coef
    pred = A @ coef
    ss_res = float(np.sum((log_delta - pred) ** 2))
    ss_tot = float(np.sum((log_delta - log_delta.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return math.exp(logC), -slope, r2


def compute_fit(observed_ppl64=None, seed=0):
    data = load_seed_ppls()
    d_arr = np.array(FIT_DOSC, float)
    delta_arr = np.array([np.mean(data[d]) - SOFTMAX_PPL for d in FIT_DOSC])
    C, alpha, r2 = fit_power(d_arr, delta_arr)
    pred64 = C * 64.0 ** (-alpha)

    rng = np.random.default_rng(seed)
    preds = []
    for _ in range(N_BOOT):
        dl, dd = [], []
        ok = True
        for d in FIT_DOSC:
            s = np.asarray(data[d], float)
            rs = rng.choice(s, size=s.size, replace=True)
            delta = rs.mean() - SOFTMAX_PPL
            if delta <= 0:
                ok = False; break
            dl.append(d); dd.append(delta)
        if not ok:
            continue
        Cb, ab, _ = fit_power(dl, dd)
        preds.append(Cb * 64.0 ** (-ab))
    preds = np.array(preds)
    lo, med, hi = (float(np.percentile(preds, q)) for q in (2.5, 50, 97.5))

    obs = None if observed_ppl64 is None else (observed_ppl64 - SOFTMAX_PPL)
    in_ci = None if obs is None else bool(lo <= obs <= hi)
    return {
        "fit_dosc": FIT_DOSC, "delta_points": {str(d): float(dl) for d, dl in zip(FIT_DOSC, delta_arr)},
        "alpha": alpha, "C": C, "r2": r2, "softmax_ppl": SOFTMAX_PPL,
        "predicted_delta64": {"point": pred64, "boot_median": med, "ci95": [lo, hi], "n_boot": int(preds.size)},
        "observed_ppl64": observed_ppl64, "observed_delta64": obs, "in_ci": in_ci,
    }


def main():
    device = harness.pick_device("mps")
    print(f"lm_tinystories_extrapolation device={device}", flush=True)

    # (1) train d_osc=64 seed 0 (MPS, ~3.8h) if not done
    if not harness.exists(EXP, "train_d64_s0"):
        ts = paths.load_py(os.path.join(REPO, "training",
                                        "train_tinystories.py"), "ts_train_mod")
        os.makedirs(CKPT, exist_ok=True)
        ts.CKPT_DIR = CKPT
        t0 = time.time()
        r = ts.run_one(64, 0, device)
        harness.save_result(EXP, "train_d64_s0", {
            "d_osc": 64, "seed": 0, "val_ppl": r["val_ppl"],
            "wall_sec": round(time.time() - t0, 1)})
        print(f"lm_tinystories_extrapolation d_osc=64 PPL={r['val_ppl']:.3f}", flush=True)
        harness.free_memory(device)

    # (2) fit + bootstrap (CPU); fill observed if the d64 run exists
    obs = None
    if harness.exists(EXP, "train_d64_s0"):
        obs = harness.load_result(EXP, "train_d64_s0")["val_ppl"]
    res = compute_fit(observed_ppl64=obs)
    harness.save_result(EXP, "powerlaw_fit", res)
    p = res["predicted_delta64"]
    print(f"lm_tinystories_extrapolation fit: alpha={res['alpha']:.4f} C={res['C']:.4f} R2={res['r2']:.4f}", flush=True)
    print(f"  predicted Delta(64) median={p['boot_median']:.4f} "
          f"95%CI=[{p['ci95'][0]:.4f},{p['ci95'][1]:.4f}]", flush=True)
    if obs is not None:
        print(f"  observed Delta(64)={res['observed_delta64']:.4f} in_CI={res['in_ci']}", flush=True)
    print("lm_tinystories_extrapolation COMPLETE", flush=True)


if __name__ == "__main__":
    main()
