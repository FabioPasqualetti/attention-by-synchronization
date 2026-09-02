"""lm_tinystories_extrapolation — power-law out-of-sample check with seed variance at d_osc=64.

After tinystories_extrapolation.py (d_osc=64 seed 0), train seeds 1 and 2 at d_osc=64 (TinyStories,
ref config). Report the observed Delta(64) mean +/- std over seeds {0,1,2} against the bootstrap
95% CI of the [2,32] fit (reused from tinystories_extrapolation.py). Resumable.
"""
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
SEEDS = [1, 2]

_e6 = paths.load_py(os.path.join(REPO, "language_modeling", "tinystories_extrapolation.py"), "e6_mod")


def _seed0_ppl():
    # seed 0 is produced by tinystories_extrapolation.py -> reference.
    if harness.reference_exists("lm_tinystories_extrapolation", "train_d64_s0"):
        return harness.load_reference("lm_tinystories_extrapolation", "train_d64_s0")["val_ppl"]
    return None


def train_seed(seed, device):
    key = f"train_d64_s{seed}"
    if harness.exists(EXP, key):
        return harness.load_result(EXP, key)["val_ppl"]
    ts = paths.load_py(os.path.join(REPO, "training", "train_tinystories.py"),
                       "ts_train_mod")
    os.makedirs(CKPT, exist_ok=True)
    ts.CKPT_DIR = CKPT
    t0 = time.time()
    r = ts.run_one(64, seed, device)
    harness.save_result(EXP, key, {"d_osc": 64, "seed": seed, "val_ppl": r["val_ppl"],
                                   "wall_sec": round(time.time() - t0, 1)})
    harness.free_memory(device)
    print(f"lm_tinystories_extrapolation d_osc=64 s{seed} PPL={r['val_ppl']:.3f}", flush=True)
    return r["val_ppl"]


def main():
    device = harness.pick_device("mps")
    print(f"lm_tinystories_extrapolation device={device}", flush=True)

    ppls = {}
    p0 = _seed0_ppl()
    if p0 is not None:
        ppls[0] = p0
    for s in SEEDS:
        ppls[s] = train_seed(s, device)

    # observed Delta(64) mean +/- std over available seeds
    vals = np.array([ppls[s] for s in sorted(ppls)], float)
    deltas = vals - SOFTMAX_PPL
    obs_mean, obs_std = float(deltas.mean()), float(deltas.std(ddof=1) if deltas.size > 1 else 0.0)

    fit = _e6.compute_fit(observed_ppl64=float(vals.mean()))
    ci = fit["predicted_delta64"]["ci95"]
    mean_in = bool(ci[0] <= obs_mean <= ci[1])
    band_overlaps = bool((obs_mean - obs_std) <= ci[1] and (obs_mean + obs_std) >= ci[0])

    payload = {
        "seeds": sorted(ppls), "ppl_by_seed": {str(s): ppls[s] for s in ppls},
        "observed_ppl64_mean": float(vals.mean()), "observed_ppl64_std": float(vals.std(ddof=1) if vals.size > 1 else 0.0),
        "observed_delta64_mean": obs_mean, "observed_delta64_std": obs_std,
        "predicted_delta64": fit["predicted_delta64"], "alpha": fit["alpha"], "C": fit["C"], "r2": fit["r2"],
        "observed_mean_in_ci": mean_in, "observed_pm1std_band_overlaps_ci": band_overlaps,
    }
    harness.save_result(EXP, "powerlaw_seeds", payload)
    print(f"lm_tinystories_extrapolation: observed Delta(64)={obs_mean:.4f}±{obs_std:.4f} (n={vals.size}) | "
          f"predicted median={fit['predicted_delta64']['boot_median']:.4f} "
          f"95%CI=[{ci[0]:.4f},{ci[1]:.4f}] | mean_in_CI={mean_in}", flush=True)
    print("lm_tinystories_extrapolation COMPLETE", flush=True)


if __name__ == "__main__":
    main()
