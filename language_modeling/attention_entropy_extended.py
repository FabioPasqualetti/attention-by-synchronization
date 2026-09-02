"""lm_attention_entropy — attention-entropy measurement with RAW samples (eval-only, CPU).

Definitive version of attention_entropy.py: same per-row normalized attention-entropy measurement
(H = -sum p ln p over each causal row's valid keys, divided by ln(i+1) for query
position i) over the same >=100 TinyStories val sequences, but for the complete set
oscillator d_osc in {2,4,8,16} + softmax, and it SAVES the RAW per-row normalized-entropy samples
(a >=10k random subsample per condition, fixed RNG seed) alongside the summary statistics, so the
entropy figure can render violins from the raw distributions.

Inputs (trained checkpoints):
  softmax  results/lm_dimensional_bottleneck/ckpt/ts_softmax_s0_ep5.pt
  d_osc=2  results/lm_dimensional_bottleneck/ckpt/ts_d2_s0_ep5.pt
  d_osc=4  results/lm_5pt_runs/checkpoints/ts_dosc4_seed0/ts_d4_s0_ep5.pt
  d_osc=8  results/lm_coupling_function/ckpt/ts_softplus_s0.pt
  d_osc=16 results/lm_5pt_runs/checkpoints/ts_dosc16_seed0/ts_d16_s0_ep5.pt

Outputs (results/lm_attention_entropy/):
  summary_<cond>.json  {overall median/iqr/mean/n, per_head}
  raw_<cond>.json      {"samples": [...]}  (<=10000 subsample, rng seed 0)

Resumable.
"""
import math
import os
import pickle
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training import harness, paths  # noqa: E402
from training.data_utils import load_tinystories  # noqa: E402
paths.ensure_paths()

import json  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

EXP = "lm_attention_entropy"
CACHE = os.path.expanduser(os.path.join(paths.cache_root(), "tinystories"))
ARCH = dict(d_model=128, n_heads=4, n_layers=2, d_ff=512, max_seq_len=128, dropout=0.1)
N_VAL = 128
MAXLEN = 128
RAW_MAX = 10000          # subsample size per condition
RAW_SEED = 0
RESDIR = os.path.join(harness.RUNS_ROOT, EXP)

MODELS = {
    "softmax": (None, os.path.join(harness.RUNS_ROOT, "lm_dimensional_bottleneck", "ckpt", "ts_softmax_s0_ep5.pt")),
    "osc_d2": (2, os.path.join(harness.RUNS_ROOT, "lm_dimensional_bottleneck", "ckpt", "ts_d2_s0_ep5.pt")),
    "osc_d4": (4, os.path.join(paths.runs_root(), "lm_5pt_runs", "checkpoints",
                               "ts_dosc4_seed0", "ts_d4_s0_ep5.pt")),
    "osc_d8": (8, os.path.join(harness.RUNS_ROOT, "lm_coupling_function", "ckpt", "ts_softplus_s0.pt")),
    "osc_d16": (16, os.path.join(paths.runs_root(), "lm_5pt_runs", "checkpoints",
                                 "ts_dosc16_seed0", "ts_d16_s0_ep5.pt")),
}


def load_val():
    # via load_tinystories, not the raw cache: it resolves the shipped
    # data/tinystories_eval/ vocabulary, which is what indexes the released
    # checkpoints. Reading <cache>/tinystories directly fails on a clean clone.
    vocab, _, _, val_all = load_tinystories(max_len=MAXLEN)
    val = val_all[:N_VAL]
    toks = torch.zeros(len(val), MAXLEN, dtype=torch.long)
    for i, c in enumerate(val):
        c = c[:MAXLEN]
        toks[i, :len(c)] = torch.tensor(c, dtype=torch.long)
    return toks, (toks == 0), len(vocab)


def build(name, d_osc, vocab_size):
    if name == "softmax":
        ts = paths.load_py(os.path.join(paths.REPO_ROOT, "training",
                                        "train_tinystories.py"), "ts_train_e11ext")
        return ts.make_softmax_lm(vocab_size)
    from oscillator_attention.sigma_models import LoheLMSigma
    return LoheLMSigma(vocab_size=vocab_size, d_osc=d_osc, sigma="softplus", **ARCH)


def _stats(v):
    v = np.asarray(v)
    if v.size == 0:
        return None
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {"median": float(med), "iqr": [float(q1), float(q3)],
            "mean": float(v.mean()), "n": int(v.size)}


def _summary_path(name):
    return os.path.join(RESDIR, f"summary_{name}.json")


def _raw_path(name):
    return os.path.join(RESDIR, f"raw_{name}.json")


@torch.no_grad()
def measure(name, d_osc, ckpt, toks, pad, vocab_size):
    if os.path.exists(_summary_path(name)) and os.path.exists(_raw_path(name)):
        with open(_summary_path(name)) as f:
            return json.load(f)["overall"]
    if not os.path.exists(ckpt):
        print(f"lm_attention_entropy {name}: no ckpt {ckpt}", flush=True); return None
    model = build(name, d_osc, vocab_size)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
    model.eval()
    H = ARCH["n_heads"]
    outs = []
    hs = [layer.attn.register_forward_hook(lambda m, i, o: outs.append(o[1].detach()))
          for layer in model.layers]
    per_head = {h: [] for h in range(H)}
    allvals = []
    B = 32
    for s in range(0, toks.shape[0], B):
        outs.clear()
        tb = toks[s:s + B]; pb = pad[s:s + B]
        model(tb, padding_mask=pb)
        lengths = (~pb).sum(1).tolist()
        for a in outs:
            if a.dim() == 3:
                a = a.view(tb.shape[0], H, MAXLEN, MAXLEN)
            for b in range(a.shape[0]):
                L = int(lengths[b])
                for h in range(H):
                    for i in range(1, L):
                        row = a[b, h, i, :i + 1].cpu().numpy()
                        row = row[row > 0]
                        if row.size < 2:
                            continue
                        ent = -(row * np.log(row)).sum()
                        norm = ent / math.log(i + 1)
                        per_head[h].append(norm)
                        allvals.append(norm)
    for hd in hs:
        hd.remove()

    os.makedirs(RESDIR, exist_ok=True)
    # summary (all values) + raw subsample (fixed RNG)
    summary = {"name": name, "mechanism": "softmax" if name == "softmax" else "oscillator",
               "d_osc": d_osc, "ckpt": ckpt, "normalized_by": "ln(row_length)",
               "overall": _stats(allvals),
               "per_head": {str(h): _stats(per_head[h]) for h in range(H)}}
    with open(_summary_path(name), "w") as f:
        json.dump(summary, f, indent=2)

    arr = np.asarray(allvals, dtype=float)
    rng = np.random.default_rng(RAW_SEED)
    if arr.size > RAW_MAX:
        idx = rng.choice(arr.size, size=RAW_MAX, replace=False)
        sub = arr[np.sort(idx)]
    else:
        sub = arr
    with open(_raw_path(name), "w") as f:
        json.dump({"name": name, "mechanism": summary["mechanism"], "d_osc": d_osc,
                   "n_total": int(arr.size), "n_saved": int(sub.size),
                   "rng_seed": RAW_SEED, "samples": [float(x) for x in sub]}, f)

    o = summary["overall"]
    print(f"DONE lm_attention_entropy {name}: norm-entropy median={o['median']:.4f} "
          f"IQR=[{o['iqr'][0]:.4f},{o['iqr'][1]:.4f}] (n={o['n']}, raw_saved={sub.size})",
          flush=True)
    harness.free_memory(None)
    return summary["overall"]


def main():
    print("lm_attention_entropy device=cpu", flush=True)
    toks, pad, vocab_size = load_val()
    for name, (d_osc, ckpt) in MODELS.items():
        measure(name, d_osc, ckpt, toks, pad, vocab_size)
    # report d_osc=2 explicitly
    p = _summary_path("osc_d2")
    if os.path.exists(p):
        o = json.load(open(p))["overall"]
        print(f"\nE11ext d_osc=2: median={o['median']:.4f} "
              f"IQR=[{o['iqr'][0]:.4f}, {o['iqr'][1]:.4f}]  (n={o['n']})", flush=True)
    print("lm_attention_entropy COMPLETE", flush=True)


if __name__ == "__main__":
    main()
