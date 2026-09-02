"""lm_wikitext_extrapolation-ext — WikiText-2 d_osc=64 seeds 1 and 2 (final out-of-sample confirmation).

Trains two more oscillator WT2 runs at d_osc=64 (seeds 1,2) with the EXACT config lm_wikitext_extrapolation seed 0
used (resolved from training/train_wikitext.py). No re-gating (seed 0 validated the
pipeline). Then recomputes the out-of-sample test over seeds {0,1,2} as one analysis, mirroring
the TinyStories treatment: observed Δ(64) mean±std vs the EXISTING bootstrap 95% CI from the
[2,32] fit in results/lm_wikitext_extrapolation/powerlaw_fit.json (do NOT refit).
"""
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
SOFTMAX_WT2 = 99.58
SEEDS = [1, 2]


def train_seed(wt, seed, device):
    key = f"train_wt2_d64_s{seed}"
    if harness.exists(EXP, key):
        return harness.load_result(EXP, key)["val_ppl"]
    t0 = time.time()
    r = wt.run_one(64, "kuramoto", seed, device)
    harness.save_result(EXP, key, {"d_osc": 64, "seed": seed, "val_ppl": r["val_ppl"],
                                   "wall_sec": round(time.time() - t0, 1)})
    harness.free_memory(device)
    print(f"lm_wikitext_extrapolation WT2 d_osc=64 seed{seed} PPL={r['val_ppl']:.3f}", flush=True)
    return r["val_ppl"]


def main():
    device = harness.pick_device("mps")
    print(f"lm_wikitext_extrapolation device={device}", flush=True)
    wt = paths.load_py(os.path.join(REPO, "training", "train_wikitext.py"),
                       "wt2_train_mod")
    os.makedirs(CKPT, exist_ok=True)
    wt.CKPT_DIR = CKPT  # keep checkpoints under runs/

    ppls = {}
    # seed 0 from lm_wikitext_extrapolation (already trained + validated)
    # seed 0 and the fit are produced by wikitext_extrapolation.py -> reference.
    ppls[0] = harness.load_reference("lm_wikitext_extrapolation", "train_wt2_d64_s0")["val_ppl"]
    for s in SEEDS:
        ppls[s] = train_seed(wt, s, device)

    # existing bootstrap CI from the [2,32] fit — do NOT refit
    fit = harness.load_reference("lm_wikitext_extrapolation", "powerlaw_fit")
    ci = fit["predicted_delta64"]["ci95"]
    pred_median = fit["predicted_delta64"]["boot_median"]
    alpha = fit["alpha"]

    seeds_sorted = [0] + SEEDS
    ppl_list = [ppls[s] for s in seeds_sorted]
    deltas = [p - SOFTMAX_WT2 for p in ppl_list]
    mean_d = float(np.mean(deltas))
    std_d = float(np.std(deltas))  # population std (matches lm_tinystories_extrapolation treatment)
    ppl_mean = float(np.mean(ppl_list))
    ppl_std = float(np.std(ppl_list))

    mean_in_ci = ci[0] <= mean_d <= ci[1]
    all_above = all(d > ci[1] for d in deltas)
    all_below = all(d < ci[0] for d in deltas)
    band_lo, band_hi = mean_d - std_d, mean_d + std_d
    band_overlaps = not (band_hi < ci[0] or band_lo > ci[1])

    if mean_in_ci:
        conclusion = ("(a) 3-seed mean inside CI — WT2 prediction CONFIRMED; the earlier "
                      "single-seed marginal miss was seed noise.")
    elif all_above:
        pct = 100.0 * (mean_d - pred_median) / pred_median
        conclusion = (f"(b) all three seeds above CI — small systematic UNDER-prediction "
                      f"(~{pct:.1f}% in Δ; predicted median {pred_median:.3f}, observed mean "
                      f"{mean_d:.3f}).")
    elif all_below:
        conclusion = "(b') all three seeds below CI — systematic OVER-prediction."
    else:
        conclusion = ("(c) seeds straddle the CI — reported as mean±std vs CI without a "
                      "categorical call.")

    harness.save_result(EXP, "powerlaw_seeds", {
        "seeds": seeds_sorted, "ppl_by_seed": {str(s): ppls[s] for s in seeds_sorted},
        "observed_ppl64_mean": ppl_mean, "observed_ppl64_std": ppl_std,
        "observed_delta64_mean": mean_d, "observed_delta64_std": std_d,
        "deltas_by_seed": {str(s): ppls[s] - SOFTMAX_WT2 for s in seeds_sorted},
        "existing_ci95": ci, "predicted_delta64_boot_median": pred_median, "alpha": alpha,
        "softmax_ppl": SOFTMAX_WT2, "observed_mean_in_ci": mean_in_ci,
        "all_seeds_above_ci": all_above, "pm1std_band_overlaps_ci": band_overlaps,
        "conclusion": conclusion})
    print(f"\nextra seeds: PPL {ppl_list} -> Δ mean {mean_d:.4f} ± {std_d:.4f} "
          f"vs CI {ci} | mean_in_ci={mean_in_ci}\n{conclusion}\n"
          f"lm_wikitext_extrapolation seeds COMPLETE", flush=True)


if __name__ == "__main__":
    main()
