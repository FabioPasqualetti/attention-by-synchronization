"""lm_dimension_scaling — seed-uniformity fill for the d_osc scaling study.

Allocation rule: five seeds per configuration; ten at each held-out test point (d_osc=64).
Fills the missing TinyStories/WikiText-2 runs (reusing all existing seeds), TS first (long pole).
Per-run JSON in results/lm_dimension_scaling/, resumable (skip if json exists), checkpoints under
results/lm_dimension_scaling/ckpt/.

Deterministic gate: for each configuration that already has ≥1 seed, the FIRST newly trained seed
must be consistent with the existing seed mean within GATE_TOL relative (3%, which is ≫ the corpus
seed spread of ~0.3–0.5%, so it flags gross pipeline failures, not normal seed variance). The actual
relative deviation is recorded for every first-new-seed so "consistent within known variance" is
evidenced. On a gate failure the affected configuration stops and is flagged; other configs continue.
"""
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training import harness, paths  # noqa: E402

paths.ensure_paths()

EXP = "lm_dimension_scaling"
GATE_TOL = 0.03  # relative; ≫ corpus seed std (~0.3–0.5%), flags gross failures only
CKPT = os.path.join(harness.RUNS_ROOT, EXP, "ckpt")

# (dataset, tag, d_osc, attn_type, new_seeds). TS first (long pole), WT2 last (interleave/quick).
JOBS = [
    ("ts",  "d2",      2,  "kuramoto", [3, 4]),
    ("ts",  "d8",      8,  "kuramoto", [3, 4]),
    ("ts",  "d32",     32, "kuramoto", [3, 4]),
    ("ts",  "softmax", 0,  "softmax",  [1, 2, 3, 4]),
    ("ts",  "d64",     64, "kuramoto", [3, 4, 5, 6, 7, 8, 9]),
    ("wt2", "softmax", 0,  "softmax",  [3, 4]),
]


def existing_ref(dataset, tag):
    """Mean PPL of the existing seeds for a config (for the gate)."""
    vals = []
    if dataset == "ts":
        if tag in ("d2", "d8", "d32"):
            f = f"results/tinystories_{tag}_3seeds.json"
            vals = [r["val_ppl"] for r in json.load(open(os.path.join(paths.REPO_ROOT, f)))]
        elif tag == "softmax":
            vals = [harness.load_reference("lm_dimensional_bottleneck", "softmax_train")["val_ppl"]]
        elif tag == "d64":
            vals = [harness.load_reference("lm_tinystories_extrapolation", "train_d64_s0")["val_ppl"]] + \
                   [harness.load_reference("lm_tinystories_extrapolation", f"train_d64_s{s}")["val_ppl"] for s in (1, 2)]
    elif dataset == "wt2" and tag == "softmax":
        vals = [harness.load_reference("lm_wikitext_extrapolation", f"train_wt2_softmax_s{s}")["val_ppl"]
                for s in (0, 1, 2)]
    return sum(vals) / len(vals) if vals else None


def train(dataset, d_osc, attn_type, seed, device):
    if dataset == "ts":
        tt = paths.load_py(os.path.join(paths.REPO_ROOT, "training",
                                        "train_tinystories.py"), "tt_mod")
        tt.CKPT_DIR = os.path.join(CKPT, "ts")
        os.makedirs(tt.CKPT_DIR, exist_ok=True)
        r = tt.run_one(d_osc, seed, device)   # d_osc=0 => softmax
        return r["val_ppl"]
    else:
        tw = paths.load_py(os.path.join(paths.REPO_ROOT, "training",
                                        "train_wikitext.py"), "tw_mod")
        tw.CKPT_DIR = os.path.join(CKPT, "wt2")
        os.makedirs(tw.CKPT_DIR, exist_ok=True)
        r = tw.run_one(d_osc, attn_type, seed, device)
        return r["val_ppl"]


def main():
    device = harness.pick_device("mps")
    print(f"lm_dimension_scaling device={device}", flush=True)
    for dataset, tag, d_osc, attn_type, new_seeds in JOBS:
        ref = existing_ref(dataset, tag)
        gate_failed = False
        for i, seed in enumerate(new_seeds):
            key = f"{dataset}_{tag}_s{seed}"
            if harness.exists(EXP, key):
                print(f"skip {key}", flush=True)
                continue
            if gate_failed:
                print(f"skip {key} (gate failed for {dataset}_{tag})", flush=True)
                continue
            t0 = time.time()
            ppl = train(dataset, d_osc, attn_type, seed, device)
            payload = {"dataset": dataset, "tag": tag, "d_osc": d_osc,
                       "attn_type": attn_type, "seed": seed, "val_ppl": ppl,
                       "wall_sec": round(time.time() - t0, 1)}
            # Gate on the first newly-trained seed of this config
            first_new = not any(harness.exists(EXP, f"{dataset}_{tag}_s{s}")
                                for s in new_seeds if s != seed)
            if first_new and ref is not None:
                dev = abs(ppl - ref) / ref
                payload["gate_ref_ppl"] = ref
                payload["gate_rel_dev"] = dev
                payload["gate_ok"] = dev <= GATE_TOL
                if dev > GATE_TOL:
                    gate_failed = True
                    payload["GATE_FAIL"] = True
                    print(f"GATE FAIL {key}: ppl={ppl:.3f} vs ref {ref:.3f} "
                          f"(rel dev {dev*100:.2f}% > {GATE_TOL*100:.0f}%) — stopping {dataset}_{tag}",
                          flush=True)
                else:
                    print(f"[gate] {key}: ppl={ppl:.3f} vs ref {ref:.3f} "
                          f"(rel dev {dev*100:.3f}% ✓)", flush=True)
            harness.save_result(EXP, key, payload)
            print(f"DONE {key}: ppl={ppl:.4f} ({(time.time()-t0)/60:.1f}m)", flush=True)
    print("lm_dimension_scaling COMPLETE", flush=True)


if __name__ == "__main__":
    main()
