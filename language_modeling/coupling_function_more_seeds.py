"""lm_coupling_function — TinyStories seeds {3,4}, completing a uniform 5-seed cohort.

Reuses the exact TinyStories recipe of coupling_function.py::run_ts (analytic LoheLMSigma d_osc=8;
d_model=128 n_h=4 n_l=2 d_ff=512; AdamW lr=5e-4, 5 epochs, batch 256; best-PPL). Adds seeds
{3,4} for softplus / relu_eps=relu(x)+1e-3 / elu1=elu(x)+1. With seed 0 (coupling_function.py) and
seeds {1,2} (coupling_function_seeds.py) this yields 5 seeds per variant.

GATE: each variant's first new seed (seed 3) vs that variant's seed-0 value, 3% rel tol (>> ~0.3-0.5%
TS seed std). On GATE_FAIL that variant stops (seed 4 skipped) and is flagged; others continue.

Writes results/lm_coupling_function/. Resumable.
Run: python language_modeling/coupling_function_more_seeds.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training import harness, paths  # noqa: E402
paths.ensure_paths()
import numpy as np  # noqa: E402

EXP = "lm_coupling_function"
SIGMAS = ["softplus", "relu_eps", "elu1"]
NEW_SEEDS = [3, 4]
GATE_TOL = 0.03

_e3 = paths.load_py(os.path.join(os.path.dirname(__file__), "coupling_function.py"), "e3_mod")


def seed0_ppl(sigma):
    # produced by coupling_function.py -> reference, not this driver's own work.
    return harness.load_reference("lm_coupling_function", f"ts_{sigma}_s0")["val_ppl"]


def _prior_ppls(sigma):
    """seed 0 (coupling_function.py) + seeds 1,2 (coupling_function_seeds.py).

    All three come from other drivers, so they are published values -> reference.
    This driver's own seeds 3,4 are read from runs/ by the caller."""
    ppls = [seed0_ppl(sigma)]
    for s in (1, 2):
        if harness.reference_exists("lm_coupling_function", f"ts_{sigma}_s{s}"):
            ppls.append(harness.load_reference("lm_coupling_function", f"ts_{sigma}_s{s}")["val_ppl"])
    return ppls


def main():
    device = harness.pick_device("mps")
    print(f"{EXP} device={device}", flush=True)
    for sigma in SIGMAS:
        ref = seed0_ppl(sigma)
        gate_failed = False
        for seed in NEW_SEEDS:
            key = f"ts_{sigma}_s{seed}"
            if harness.exists(EXP, key):
                print(f"skip {key}", flush=True); continue
            if gate_failed:
                print(f"skip {key} (gate failed for {sigma})", flush=True); continue
            t0 = time.time()
            ppl = _e3.run_ts(sigma, seed, device)
            payload = {"task": "tinystories", "sigma": sigma, "d_osc": 8, "seed": seed,
                       "val_ppl": ppl, "arch": _e3.TS_ARCH, "train": _e3.TS_TRAIN,
                       "wall_sec": round(time.time() - t0, 1)}
            if seed == NEW_SEEDS[0]:  # gate on first new seed (seed 3) vs seed-0
                dev = abs(ppl - ref) / ref
                payload.update(gate_ref_ppl=ref, gate_rel_dev=dev, gate_ok=dev <= GATE_TOL)
                if dev > GATE_TOL:
                    gate_failed = True
                    payload["GATE_FAIL"] = True
                    print(f"GATE FAIL {key}: ppl={ppl:.3f} vs seed0 {ref:.3f} "
                          f"(rel dev {dev*100:.2f}% > {GATE_TOL*100:.0f}%) — stopping {sigma}", flush=True)
                else:
                    print(f"[gate] {key}: ppl={ppl:.3f} vs seed0 {ref:.3f} (rel dev {dev*100:.3f}% OK)", flush=True)
            harness.save_result(EXP, key, payload)
            print(f"DONE {key}: ppl={ppl:.4f} ({(time.time()-t0)/60:.1f}m)", flush=True)
            harness.free_memory(device)

    # ---- 5-seed summary per variant (seeds 0..4) + relu offset persistence ----
    summary = {}
    for sigma in SIGMAS:
        ppls = _prior_ppls(sigma)
        for seed in NEW_SEEDS:
            if harness.exists(EXP, f"ts_{sigma}_s{seed}"):
                ppls.append(harness.load_result(EXP, f"ts_{sigma}_s{seed}")["val_ppl"])
        summary[sigma] = {"n": len(ppls), "ppls": ppls, "mean": float(np.mean(ppls)),
                          "std": float(np.std(ppls, ddof=1)) if len(ppls) > 1 else 0.0}
    if all(summary[s]["n"] >= 2 for s in ("softplus", "relu_eps")):
        sp, rl = summary["softplus"], summary["relu_eps"]
        offset = rl["mean"] - sp["mean"]
        pooled = float(np.hypot(sp["std"], rl["std"]))
        summary["relu_offset_vs_softplus"] = {
            "mean_offset": offset, "pooled_std": pooled,
            "persists_beyond_seed_noise": offset > pooled}
    harness.save_result(EXP, "summary_5seed", summary)
    for sigma in SIGMAS:
        s = summary[sigma]
        print(f"{sigma}: {s['mean']:.3f} ± {s['std']:.3f} (n={s['n']}) ppls={[round(p,3) for p in s['ppls']]}", flush=True)
    if "relu_offset_vs_softplus" in summary:
        o = summary["relu_offset_vs_softplus"]
        print(f"relu offset mean {o['mean_offset']:+.3f} (pooled std {o['pooled_std']:.3f}); "
              f"persists beyond seed noise: {o['persists_beyond_seed_noise']}", flush=True)
    print(f"{EXP} COMPLETE", flush=True)


if __name__ == "__main__":
    main()
