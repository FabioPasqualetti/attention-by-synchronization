"""lm_wikitext_extrapolation — WikiText-2 out-of-sample power-law check.

Resolve the paper's WT2 d_osc-scaling reference config from the repo (training/
train_wikitext.py: d_model=128, n_heads=4, n_layers=2, d_ff=512, max_seq_len=50, 30 epochs,
LoheLanguageTransformer, softmax baseline PPL=99.58). Fit Delta = C*d_osc^-alpha on the paper's
WT2 points d_osc in {2,4,8,16,32} (Delta = mean PPL - 99.58), bootstrap for a 95% CI on the
d_osc=64 prediction, then train one oscillator WT2 run at d_osc=64 seed 0 and test predicted vs
observed. If WT2 data is unavailable, write skipped.json and exit cleanly.

Resumable.
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

EXP = "lm_wikitext_extrapolation"
REPO = paths.REPO_ROOT
CKPT = os.path.join(harness.RUNS_ROOT, EXP, "ckpt")
SOFTMAX_PPL = 99.58
FIT_DOSC = [2, 4, 8, 16, 32]
N_BOOT = 2000

_e6 = paths.load_py(os.path.join(REPO, "language_modeling", "tinystories_extrapolation.py"), "e6_mod")

# WT2 reference config, resolved from training/train_wikitext.py (Table 5 scaling).
WT2_CONFIG = dict(d_model=128, n_heads=4, n_layers=2, d_ff=512, max_seq_len=50,
                  dropout=0.1, lr=5e-4, weight_decay=1e-4, batch_size=64, grad_clip=1.0,
                  n_epochs=30, optimizer="AdamW", cosine_lr=True, vocab="top-10K",
                  source="training/train_wikitext.py")


def load_wt2_seed_ppls():
    """d_osc -> list of per-seed WT2 val PPLs (paper's scaling files)."""
    data = {}
    for d, path in [(2, "results/wikitext2_d2_5seeds.json"),
                    (8, "results/lm/wikitext2_d8_5seeds.json"),
                    (32, "results/lm/wikitext2_d32_5seeds.json")]:
        j = json.load(open(os.path.join(REPO, path)))
        data[d] = [float(v) for v in j["seeds"].values()]
    for d, patt in [(4, "wt2_dosc4_seed*.json"), (16, "wt2_dosc16_seed*.json")]:
        vals = []
        for p in sorted(glob.glob(os.path.join(paths.results_root(), "lm_5pt_runs", patt))):
            vals.append(float(json.load(open(p))["val_ppl"]))
        data[d] = vals
    return data


def compute_fit(observed_ppl64=None):
    data = load_wt2_seed_ppls()
    d_arr = np.array(FIT_DOSC, float)
    delta_arr = np.array([np.mean(data[d]) - SOFTMAX_PPL for d in FIT_DOSC])
    C, alpha, r2 = _e6.fit_power(d_arr, delta_arr)
    pred64 = C * 64.0 ** (-alpha)
    rng = np.random.default_rng(0)
    preds = []
    for _ in range(N_BOOT):
        dl, dd, ok = [], [], True
        for d in FIT_DOSC:
            s = np.asarray(data[d], float)
            delta = rng.choice(s, size=s.size, replace=True).mean() - SOFTMAX_PPL
            if delta <= 0:
                ok = False; break
            dl.append(d); dd.append(delta)
        if ok:
            Cb, ab, _ = _e6.fit_power(dl, dd)
            preds.append(Cb * 64.0 ** (-ab))
    preds = np.array(preds)
    lo, med, hi = (float(np.percentile(preds, q)) for q in (2.5, 50, 97.5))
    obs = None if observed_ppl64 is None else (observed_ppl64 - SOFTMAX_PPL)
    return {
        "fit_dosc": FIT_DOSC, "softmax_ppl": SOFTMAX_PPL,
        "delta_points": {str(d): float(x) for d, x in zip(FIT_DOSC, delta_arr)},
        "alpha": alpha, "C": C, "r2": r2,
        "predicted_delta64": {"point": pred64, "boot_median": med, "ci95": [lo, hi], "n_boot": int(preds.size)},
        "predicted_ppl64": pred64 + SOFTMAX_PPL,
        "observed_ppl64": observed_ppl64, "observed_delta64": obs,
        "in_ci": (None if obs is None else bool(lo <= obs <= hi)),
    }


def wt2_available():
    try:
        du = paths.load_py(os.path.join(REPO, "training", "data_utils.py"), "wt2_du")
        _ = du.load_wikitext2(max_seq_len=WT2_CONFIG["max_seq_len"])
        return True, du
    except Exception as e:
        return False, str(e)


def train_d64(device):
    if harness.exists(EXP, "train_wt2_d64_s0"):
        return harness.load_result(EXP, "train_wt2_d64_s0")["val_ppl"]
    wt = paths.load_py(os.path.join(REPO, "training", "train_wikitext.py"), "wt2_train_mod")
    os.makedirs(CKPT, exist_ok=True)
    wt.CKPT_DIR = CKPT
    t0 = time.time()
    r = wt.run_one(64, "kuramoto", 0, device)
    harness.save_result(EXP, "train_wt2_d64_s0", {"d_osc": 64, "seed": 0, "val_ppl": r["val_ppl"],
                                                  "wall_sec": round(time.time() - t0, 1)})
    harness.free_memory(device)
    print(f"lm_wikitext_extrapolation WT2 d_osc=64 PPL={r['val_ppl']:.3f}", flush=True)
    return r["val_ppl"]


def main():
    harness.save_result(EXP, "config_used", WT2_CONFIG)
    print(f"lm_wikitext_extrapolation WT2 config: {WT2_CONFIG['source']} d_model=128 n_h=4 n_l=2 max_seq_len=50 30ep", flush=True)

    ok, du = wt2_available()
    if not ok:
        harness.save_result(EXP, "skipped", {"reason": f"WikiText-2 data unavailable: {du}"})
        print(f"lm_wikitext_extrapolation SKIPPED: WT2 data unavailable ({du})", flush=True)
        return

    # CPU fit first (fills observed later)
    device = harness.pick_device("mps")
    print(f"lm_wikitext_extrapolation device={device}", flush=True)
    obs = harness.load_result(EXP, "train_wt2_d64_s0")["val_ppl"] if harness.exists(EXP, "train_wt2_d64_s0") else None
    fit0 = compute_fit(observed_ppl64=obs)
    p = fit0["predicted_delta64"]
    print(f"lm_wikitext_extrapolation WT2 fit: alpha={fit0['alpha']:.4f} R2={fit0['r2']:.4f} predicted PPL(64)="
          f"{fit0['predicted_ppl64']:.3f} (Delta CI=[{p['ci95'][0]:.3f},{p['ci95'][1]:.3f}])", flush=True)

    ppl64 = train_d64(device)
    fit = compute_fit(observed_ppl64=ppl64)
    harness.save_result(EXP, "powerlaw_fit", fit)
    print(f"lm_wikitext_extrapolation: observed PPL(64)={ppl64:.3f} Delta={fit['observed_delta64']:.4f} "
          f"in_CI={fit['in_ci']}", flush=True)
    print("lm_wikitext_extrapolation COMPLETE", flush=True)


if __name__ == "__main__":
    main()
