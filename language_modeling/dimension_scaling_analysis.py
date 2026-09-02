"""lm_dimension_scaling — uniform re-analysis of the d_osc scaling out-of-sample test.

CPU-only. Re-runs the corrected statistical test (mirroring lm_wikitext_extrapolation: observed mean +/- SEM vs
prediction +/- uncertainty; z-test + 95% prediction interval = fit uncertainty (+) observation
noise) on the SYMMETRIC 5-seed design produced by the lm_dimension_scaling fill:

  * scaling points d in {2,4,8,16,32}: 5 seeds each (both datasets)
  * softmax baseline: 5 seeds (both datasets)
  * held-out d_osc=64: 10 seeds (both datasets)

The fit/bootstrap routine (proper_test, _powerlaw_logfit) is imported from
wikitext_extrapolation_stats.py rather than reimplemented, so both analyses use one estimator.
Writes results/lm_dimension_scaling/{ts_uniform,wt2_uniform}.json (same schema as
results/lm_wikitext_extrapolation/{ts_analysis,analysis}.json plus a seed inventory) and prints an
OLD-vs-NEW side-by-side table for both datasets.

No training, no GPU — CPU-only re-analysis of committed per-seed JSONs.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training import harness, paths  # noqa: E402
paths.ensure_paths()
import numpy as np  # noqa: E402

REPO = paths.REPO_ROOT
# import the exact statistical routine used in lm_wikitext_extrapolation (do not reimplement)
_e13 = paths.load_py(os.path.join(os.path.dirname(__file__),
                                  "wikitext_extrapolation_stats.py"), "wt2_extrap_stats_mod")
proper_test = _e13.proper_test


def jload(p):
    return json.load(open(os.path.join(REPO, p)))


# ---------- uniform data assembly (5-seed scaling + baseline, 10-seed d64) ----------

def ts_scaling():
    out = {}
    for d, p in [(2, "results/tinystories_d2_3seeds.json"),
                 (8, "results/tinystories_d8_3seeds.json"),
                 (32, "results/tinystories_d32_3seeds.json")]:
        base = [float(r["val_ppl"]) for r in jload(p)]
        ext = [harness.load_reference("lm_dimension_scaling", f"ts_d{d}_s{s}")["val_ppl"] for s in (3, 4)]
        out[d] = base + ext
    for d, patt in [(4, "ts_dosc4_seed*.json"), (16, "ts_dosc16_seed*.json")]:
        out[d] = [float(json.load(open(f))["val_ppl"])
                  for f in sorted(glob.glob(os.path.join(paths.results_root(), "lm_5pt_runs", patt)))]
    return out


def ts_softmax_baseline():
    return [harness.load_reference("lm_dimensional_bottleneck", "softmax_train")["val_ppl"]] + \
           [harness.load_reference("lm_dimension_scaling", f"ts_softmax_s{s}")["val_ppl"] for s in (1, 2, 3, 4)]


def ts_d64():
    return [harness.load_reference("lm_tinystories_extrapolation", "train_d64_s0")["val_ppl"]] + \
           [harness.load_reference("lm_tinystories_extrapolation", f"train_d64_s{s}")["val_ppl"] for s in (1, 2)] + \
           [harness.load_reference("lm_dimension_scaling", f"ts_d64_s{s}")["val_ppl"] for s in range(3, 10)]


def wt2_scaling():
    out = {}
    for d, p in [(2, "results/wikitext2_d2_5seeds.json"),
                 (8, "results/lm/wikitext2_d8_5seeds.json"),
                 (32, "results/lm/wikitext2_d32_5seeds.json")]:
        out[d] = [float(v) for v in jload(p)["seeds"].values()]
    for d, patt in [(4, "wt2_dosc4_seed*.json"), (16, "wt2_dosc16_seed*.json")]:
        out[d] = [float(json.load(open(f))["val_ppl"])
                  for f in sorted(glob.glob(os.path.join(paths.results_root(), "lm_5pt_runs", patt)))]
    return out


def wt2_softmax_baseline():
    return [harness.load_reference("lm_wikitext_extrapolation", f"train_wt2_softmax_s{s}")["val_ppl"] for s in (0, 1, 2)] + \
           [harness.load_reference("lm_dimension_scaling", f"wt2_softmax_s{s}")["val_ppl"] for s in (3, 4)]


def wt2_d64():
    keys = {0: ("lm_wikitext_extrapolation", "train_wt2_d64_s0"), 1: ("lm_wikitext_extrapolation", "train_wt2_d64_s1"),
            2: ("lm_wikitext_extrapolation", "train_wt2_d64_s2")}
    for s in range(3, 10):
        keys[s] = ("lm_wikitext_extrapolation", f"train_wt2_d64_s{s}")
    return [harness.load_reference(e, k)["val_ppl"] for s in range(10) for e, k in [keys[s]]]


def inventory(scaling, base, d64):
    return {"scaling_n": {str(d): len(v) for d, v in sorted(scaling.items())},
            "baseline_n": len(base), "d64_n": len(d64)}


def main():
    results = {}
    for label, scaling, base, d64, old_key in [
        ("TinyStories", ts_scaling(), ts_softmax_baseline(), ts_d64(), "ts_analysis"),
        ("WikiText-2", wt2_scaling(), wt2_softmax_baseline(), wt2_d64(), "analysis"),
    ]:
        inv = inventory(scaling, base, d64)
        assert all(n == 5 for n in inv["scaling_n"].values()), f"{label} scaling not 5/5: {inv}"
        assert inv["baseline_n"] == 5, f"{label} baseline not 5: {inv}"
        assert inv["d64_n"] == 10, f"{label} d64 not 10: {inv}"
        res = proper_test(scaling, d64, base, label=label)
        res["seed_inventory"] = inv
        res["scaling_ppls"] = {str(d): [float(x) for x in v] for d, v in sorted(scaling.items())}
        res["baseline_ppls"] = [float(x) for x in base]
        old = harness.load_reference("lm_wikitext_extrapolation", old_key)
        res["_old"] = old
        results[label] = res
        outkey = "ts_uniform" if label == "TinyStories" else "wt2_uniform"
        # strip the bulky _old before saving the canonical uniform json
        save = {k: v for k, v in res.items() if k != "_old"}
        harness.save_result("lm_dimension_scaling", outkey, save)

    # ---- OLD vs NEW table ----
    print("\n================ lm_dimension_scaling uniform re-analysis: OLD vs NEW ================")
    hdr = f"{'dataset':<12} {'quantity':<22} {'OLD (lm_wikitext_extrapolation)':>16} {'NEW (uniform)':>16}"
    for label in ("TinyStories", "WikiText-2"):
        r = results[label]; o = r["_old"]
        print("\n" + hdr)
        print("-" * len(hdr))
        rows = [
            ("baseline seeds (n)", o["n_base"], r["n_base"]),
            ("d64 seeds (n)", o["n_obs"], r["n_obs"]),
            ("alpha", f"{o['alpha']:.4f}", f"{r['alpha']:.4f}"),
            ("R^2", f"{o['r2']:.4f}", f"{r['r2']:.4f}"),
            ("predicted d(64) median", f"{o['pred_delta_median']:.4f}", f"{r['pred_delta_median']:.4f}"),
            ("observed d(64) mean", f"{o['obs_mean_delta']:.4f}", f"{r['obs_mean_delta']:.4f}"),
            ("predicted PPL(64)", f"{o['predicted_ppl64']:.4f}", f"{r['predicted_ppl64']:.4f}"),
            ("observed PPL(64)", f"{o['obs_mean_ppl64']:.4f}", f"{r['obs_mean_ppl64']:.4f}"),
            ("z", f"{o['z']:.3f}", f"{r['z']:.3f}"),
            ("two-sided p", f"{o['p_two_sided']:.4f}", f"{r['p_two_sided']:.4f}"),
            ("inside 95% PI", o["obs_mean_inside_PI"], r["obs_mean_inside_PI"]),
        ]
        print(f"{label:<12} {'':<22} {'':>16} {'':>16}")
        for name, ov, nv in rows:
            print(f"{'':<12} {name:<22} {str(ov):>16} {str(nv):>16}")
        print(f"{'':<12} PI(Δ) new = [{r['pred_interval95'][0]:.4f}, {r['pred_interval95'][1]:.4f}]")
        print(f"{'':<12} CONCLUSION: {r['conclusion']}")
    print("\nlm_dimension_scaling analysis COMPLETE", flush=True)


if __name__ == "__main__":
    main()
