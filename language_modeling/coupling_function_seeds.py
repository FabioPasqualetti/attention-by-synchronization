"""lm_coupling_function — coupling-function ablation, TinyStories seed statistics (seeds 1-2).

The single-seed TS ablation left the relu variant's +0.12 PPL offset (relu_eps 9.868 vs
softplus 9.752 at seed 0) uninterpretable: +0.12 is several times the typical TS seed spread, so
seed statistics are needed to tell a real coupling effect from seed noise. This trains 2 additional
TinyStories seeds (seeds {1,2}) for each coupling variant — softplus, relu_eps = relu(x)+1e-3,
elu1 = elu(x)+1 — at the exact TS config of coupling_function.py (LoheLMSigma d_osc=8; d_model=128, n_heads=4,
n_layers=2, d_ff=512; AdamW lr=5e-4, 5 epochs, batch 256). 6 runs total; seed 0 is reused.

Gate: each variant's FIRST new seed (seed 1) must be consistent with that variant's seed-0 value
within known TS relative variance (GATE_TOL=3% relative, >> the ~0.3-0.5% TS seed std, so it flags
gross pipeline failures, not the effect under study). Actual relative deviation is recorded. On a
gate failure that variant stops (its seed 2 is skipped) and is flagged; other variants continue.

Deliverable (results/lm_coupling_function): mean +/- std per variant over seeds {0,1,2}; whether the
relu offset persists across seeds or sits within seed noise. Resumable per-run JSON.
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
NEW_SEEDS = [1, 2]
GATE_TOL = 0.03  # relative; >> TS seed std (~0.3-0.5%), flags gross failures only

# reuse the exact lm_coupling_function TinyStories training recipe (LoheLMSigma d_osc=8, same ARCH/TRAIN)
_e3 = paths.load_py(os.path.join(os.path.dirname(__file__), "coupling_function.py"), "e3_mod")


def seed0_ppl(sigma):
    # seed 0 is produced by coupling_function.py, not by this driver, so it is a
    # published value rather than our own prior work -> reference.
    return harness.load_reference("lm_coupling_function", f"ts_{sigma}_s0")["val_ppl"]


def main():
    device = harness.pick_device("mps")
    print(f"lm_coupling_function device={device}", flush=True)
    for sigma in SIGMAS:
        ref = seed0_ppl(sigma)
        gate_failed = False
        for seed in NEW_SEEDS:
            key = f"ts_{sigma}_s{seed}"
            if harness.exists(EXP, key):
                print(f"skip {key}", flush=True)
                continue
            if gate_failed:
                print(f"skip {key} (gate failed for {sigma})", flush=True)
                continue
            t0 = time.time()
            ppl = _e3.run_ts(sigma, seed, device)   # d_osc=8, same recipe
            payload = {"task": "tinystories", "sigma": sigma, "d_osc": 8, "seed": seed,
                       "val_ppl": ppl, "arch": _e3.TS_ARCH, "train": _e3.TS_TRAIN,
                       "wall_sec": round(time.time() - t0, 1)}
            first_new = not any(harness.exists(EXP, f"ts_{sigma}_s{s}")
                                for s in NEW_SEEDS if s != seed)
            if first_new:
                dev = abs(ppl - ref) / ref
                payload.update(gate_ref_ppl=ref, gate_rel_dev=dev, gate_ok=dev <= GATE_TOL)
                if dev > GATE_TOL:
                    gate_failed = True
                    payload["GATE_FAIL"] = True
                    print(f"GATE FAIL {key}: ppl={ppl:.3f} vs seed0 {ref:.3f} "
                          f"(rel dev {dev*100:.2f}% > {GATE_TOL*100:.0f}%) — stopping {sigma}",
                          flush=True)
                else:
                    print(f"[gate] {key}: ppl={ppl:.3f} vs seed0 {ref:.3f} "
                          f"(rel dev {dev*100:.3f}% ✓)", flush=True)
            harness.save_result(EXP, key, payload)
            print(f"DONE {key}: ppl={ppl:.4f} ({(time.time()-t0)/60:.1f}m)", flush=True)

    # ── summary: mean±std per variant over seeds {0,1,2}; does the relu offset persist? ──
    summary = {}
    for sigma in SIGMAS:
        ppls = [seed0_ppl(sigma)]
        for seed in NEW_SEEDS:
            if harness.exists(EXP, f"ts_{sigma}_s{seed}"):
                ppls.append(harness.load_result(EXP, f"ts_{sigma}_s{seed}")["val_ppl"])
        summary[sigma] = {"seeds": list(range(len(ppls))), "ppls": ppls,
                          "mean": float(np.mean(ppls)), "std": float(np.std(ppls, ddof=1)),
                          "n": len(ppls)}
    if all(summary[s]["n"] >= 2 for s in SIGMAS):
        sp, rl = summary["softplus"], summary["relu_eps"]
        offset = rl["mean"] - sp["mean"]
        pooled = float(np.hypot(sp["std"], rl["std"]))
        persists = offset > pooled  # offset exceeds combined seed spread
        summary["relu_offset_vs_softplus"] = {
            "seed0_offset": rl["ppls"][0] - sp["ppls"][0], "mean_offset": offset,
            "pooled_std": pooled, "persists_beyond_seed_noise": persists}
    harness.save_result(EXP, "summary", summary)
    for sigma in SIGMAS:
        s = summary[sigma]
        print(f"{sigma}: {s['mean']:.3f} ± {s['std']:.3f} (n={s['n']}) ppls={[round(p,3) for p in s['ppls']]}",
              flush=True)
    if "relu_offset_vs_softplus" in summary:
        o = summary["relu_offset_vs_softplus"]
        print(f"relu offset: seed0 {o['seed0_offset']:+.3f} -> mean {o['mean_offset']:+.3f} "
              f"(pooled std {o['pooled_std']:.3f}); persists beyond seed noise: "
              f"{o['persists_beyond_seed_noise']}", flush=True)
    print("lm_coupling_function COMPLETE", flush=True)


if __name__ == "__main__":
    main()
