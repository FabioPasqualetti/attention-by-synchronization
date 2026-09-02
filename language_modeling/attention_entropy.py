"""lm_attention_entropy — Attention-entropy measurement (eval-only, CPU).

Per-row Shannon entropy of attention weights over >=100 TinyStories val sequences, for
softmax and oscillator d_osc in {4,8,16}; d_osc=2 is added by attention_entropy_extended.py.
Entropy H = -sum p ln p over each causal row's valid keys, NORMALIZED by ln(row length)
= ln(i+1) for causal query position i. Report median + IQR per mechanism/d_osc and per head.

Resumable per-run JSON.
"""
import math
import os
import pickle
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training import harness, paths  # noqa: E402
from training.data_utils import load_tinystories  # noqa: E402
paths.ensure_paths()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXP = "lm_attention_entropy"
CACHE = os.path.expanduser(os.path.join(paths.cache_root(), "tinystories"))
ARCH = dict(d_model=128, n_heads=4, n_layers=2, d_ff=512, max_seq_len=128, dropout=0.1)
N_VAL = 128
MAXLEN = 128
FIGDIR = os.path.join(paths.REPO_ROOT, "figures", "diagnostics", EXP)

MODELS = {
    "softmax": (None, os.path.join(harness.RUNS_ROOT, "lm_dimensional_bottleneck", "ckpt", "ts_softmax_s0_ep5.pt")),
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
                                        "train_tinystories.py"), "ts_train_e11")
        return ts.make_softmax_lm(vocab_size)
    from oscillator_attention.sigma_models import LoheLMSigma
    return LoheLMSigma(vocab_size=vocab_size, d_osc=d_osc, sigma="softplus", **ARCH)


@torch.no_grad()
def measure(name, d_osc, ckpt, toks, pad, vocab_size):
    key = name
    if harness.exists(EXP, key):
        return
    if not os.path.exists(ckpt):
        print(f"lm_attention_entropy {name}: no ckpt {ckpt}", flush=True); return
    model = build(name, d_osc, vocab_size)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
    model.eval()
    H = ARCH["n_heads"]
    outs = []
    hs = [layer.attn.register_forward_hook(lambda m, i, o: outs.append(o[1].detach()))
          for layer in model.layers]
    # normalized entropy values per head (across all layers/rows/seqs)
    per_head = {h: [] for h in range(H)}
    allvals = []
    B = 32
    for s in range(0, toks.shape[0], B):
        outs.clear()
        tb = toks[s:s + B]; pb = pad[s:s + B]
        model(tb, padding_mask=pb)
        lengths = (~pb).sum(1).tolist()
        for a in outs:                                   # each layer's attn
            if a.dim() == 3:
                a = a.view(tb.shape[0], H, MAXLEN, MAXLEN)
            for b in range(a.shape[0]):
                L = int(lengths[b])
                for h in range(H):
                    for i in range(1, L):                # position i has i+1 keys; skip i=0 (ln1=0)
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

    def _stats(v):
        v = np.asarray(v)
        if v.size == 0:
            return None
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        return {"median": float(med), "iqr": [float(q1), float(q3)],
                "mean": float(v.mean()), "n": int(v.size)}

    payload = {"name": name, "mechanism": "softmax" if name == "softmax" else "oscillator",
               "d_osc": d_osc, "ckpt": ckpt, "normalized_by": "ln(row_length)",
               "overall": _stats(allvals),
               "per_head": {str(h): _stats(per_head[h]) for h in range(H)}}
    harness.save_result(EXP, key, payload)
    o = payload["overall"]
    print(f"DONE lm_attention_entropy {name}: norm-entropy median={o['median']:.4f} "
          f"IQR=[{o['iqr'][0]:.4f},{o['iqr'][1]:.4f}] (n={o['n']})", flush=True)
    harness.free_memory(None)
    return allvals


def main():
    print("lm_attention_entropy device=cpu", flush=True)
    toks, pad, vocab_size = load_val()
    dists = {}
    for name, (d_osc, ckpt) in MODELS.items():
        vals = measure(name, d_osc, ckpt, toks, pad, vocab_size)
        if vals:
            dists[name] = vals
    # figure: violin of normalized entropy per model
    if dists:
        os.makedirs(FIGDIR, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        names = list(dists.keys())
        data = [np.asarray(dists[n]) for n in names]
        ax.violinplot(data, showmedians=True)
        ax.set_xticks(range(1, len(names) + 1)); ax.set_xticklabels(names, rotation=20)
        ax.set_ylabel("normalized row entropy (H / ln L)")
        ax.set_title("lm_attention_entropy — attention-entropy distribution")
        fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "entropy_violin.pdf"))
        plt.close(fig)
        print(f"lm_attention_entropy figure -> {FIGDIR}/entropy_violin.pdf", flush=True)
    else:
        print("lm_attention_entropy SKIPPED: no checkpoints present (not shipped); "
              "nothing was computed. Retrain with the included configs to regenerate.",
              flush=True)
        sys.exit(1)
    print("lm_attention_entropy COMPLETE", flush=True)


if __name__ == "__main__":
    main()
