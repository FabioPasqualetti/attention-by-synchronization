"""SVA verb-attention violins -> figures/output/fig_sva_attention.pdf.

Loads the 5 config-G checkpoints (sva/checkpoints/{osc,softmax}_s{0..4}_ep20.pt) and scores them on
data/sva/sva_test.jsonl, then draws the per-mechanism attention-mass violins with the
significance brackets. The oscillator panel title reports n_h*T*(d_osc-1) = 1*9*1 = 9 oscillators.

Run standalone from the repo root:
  python figures/sva_verb_attention.py
"""
import json
import os
import sys

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # .../code
FIGROOT = os.path.join(CODE, "figures")
sys.path.insert(0, os.path.join(FIGROOT, "lib"))   # dataset_sva
sys.path.insert(0, FIGROOT)                         # plot_style
sys.path.insert(0, CODE)                            # oscillator_attention

from plot_style import apply_style, PALETTE  # noqa: E402
import dataset_sva as sva_ds  # noqa: E402
from oscillator_attention import SVATransformer  # noqa: E402

CKPT     = os.path.join(CODE, "sva", "checkpoints")
SVA_DATA = os.path.join(CODE, "data", "sva", "sva_test.jsonl")
OUT_PDF  = os.path.join(CODE, "figures", "output", "fig_sva_attention.pdf")
OUT_PNG  = os.path.join(CODE, "figures", "output", "fig_sva_attention.png")

GREEN = PALETTE[3]
RED   = PALETTE[6]


def _make_model(attn_type, seed):
    m = SVATransformer(vocab_size=len(sva_ds.VOCAB), d_model=32, n_heads=1, n_layers=1,
                       d_ff=64, max_seq_len=50, dropout=0.0, attn_type=attn_type, d_osc=2)
    prefix = "softmax" if attn_type == "softmax" else "osc"
    m.load_state_dict(torch.load(os.path.join(CKPT, f"{prefix}_s{seed}_ep20.pt"),
                                 map_location="cpu", weights_only=True))
    m.eval()
    return m


def _get_attentions(model, rec):
    bos = sva_ds.WORD2IDX["<BOS>"]
    eos = sva_ds.WORD2IDX["<EOS>"]
    # Match the training/eval loader, which encodes [BOS] + tokens + [EOS] (T=9).
    # Omitting EOS normalizes attention over the wrong support and shifts the weights.
    toks = [bos] + [sva_ds.WORD2IDX.get(t, 0) for t in rec["tokens"]] + [eos]
    x = torch.tensor([toks]); pad = torch.zeros(1, len(toks), dtype=torch.bool)
    v_idx = torch.tensor([rec["verb_idx"] + 1])
    attns, hooks = [], []

    def make_hook(lst):
        def h(module, inp, out):
            if isinstance(out, tuple) and len(out) >= 2:
                a = out[1].detach().cpu().numpy()
                if a.ndim == 4:
                    a = a[0]
                lst.append(a[:, 1:, 1:])
        return h
    for layer in model.layers:
        hooks.append(layer.attn.register_forward_hook(make_hook(attns)))
    with torch.no_grad():
        model(x, padding_mask=pad, verb_idx=v_idx)
    for h in hooks:
        h.remove()
    return attns


def main():
    apply_style()
    records = [json.loads(l) for l in open(SVA_DATA)]
    hard = [r for r in records if r.get("has_distractor")]
    n_seeds = 5
    print(f"[relabel] {len(hard)} distractor sentences x {n_seeds} config-G seeds", flush=True)

    data = {"softmax": {"vs": [], "vd": []}, "kuramoto": {"vs": [], "vd": []}}
    for seed in range(n_seeds):
        m_sm = _make_model("softmax", seed)
        m_ku = _make_model("kuramoto", seed)
        for rec in hard:
            si, vi, di = rec["subject_idx"], rec["verb_idx"], rec["distractor_idx"]
            for key, model in [("softmax", m_sm), ("kuramoto", m_ku)]:
                a = _get_attentions(model, rec)
                if a:
                    am = a[0].mean(0)
                    data[key]["vs"].append(float(am[vi, si]))
                    data[key]["vd"].append(float(am[vi, di]))
        print(f"  seed {seed} done", flush=True)

    ymax = max(max(data["softmax"]["vs"]), max(data["softmax"]["vd"]),
               max(data["kuramoto"]["vs"]), max(data["kuramoto"]["vd"])) * 1.08
    ymax = max(ymax, 0.30) * 1.10

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)
    fig.subplots_adjust(top=0.86, bottom=0.16, wspace=0.10)
    model_labels = [("softmax", "Softmax"),
                    ("kuramoto", "Oscillator (9 oscillators)")]   # <-- 14 -> 9 (only change)
    for ci, (mkey, mlabel) in enumerate(model_labels):
        ax = axes[ci]
        vs = np.array(data[mkey]["vs"]); vd = np.array(data[mkey]["vd"])
        parts = ax.violinplot([vs, vd], positions=[0, 1], showmedians=False,
                              showextrema=False, widths=0.6)
        for pi, pc in enumerate(parts["bodies"]):
            pc.set_facecolor([GREEN, RED][pi]); pc.set_alpha(0.55); pc.set_edgecolor("none")
        ax.plot([0, 1], [vs.mean(), vd.mean()], "o", color="white", markersize=5, zorder=5,
                markeredgecolor="black", markeredgewidth=0.8)
        delta_mean = vs.mean() - vd.mean()
        violin_top = max(vs.max(), vd.max())
        yb = violin_top + ymax * 0.06; yt = yb - ymax * 0.018
        ax.plot([0, 0, 1, 1], [yt, yb, yb, yt], color="#444444", lw=0.9)
        ax.text(0.5, yb + ymax * 0.012, rf"$\Delta_{{\rm mean}} = {delta_mean:+.2f}$",
                ha="center", va="bottom", fontsize=8, color="#444444")
        ax.set_xticks([0, 1]); ax.set_xticklabels([r"$\rightarrow$ subject", r"$\rightarrow$ distractor"])
        ax.set_ylim(0, ymax); ax.set_ylabel("Attention weight" if ci == 0 else "")
        ax.set_title(mlabel, loc="left", fontweight="bold", pad=8)
    ax0 = axes[0]
    ax0.annotate("mean", xy=(0, np.mean(data["softmax"]["vs"])), xycoords="data",
                 xytext=(15, 14), textcoords="offset points", fontsize=7.5, color="#444444",
                 style="italic", arrowprops=dict(arrowstyle="-", color="#888888", lw=0.6, shrinkA=0, shrinkB=3))

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PDF); fig.savefig(OUT_PNG, dpi=150); plt.close(fig)
    print(f"Saved: {OUT_PDF}", flush=True)
    for mkey, ml in [("softmax", "Softmax"), ("kuramoto", "Oscillator")]:
        vs_m = np.mean(data[mkey]["vs"]); vd_m = np.mean(data[mkey]["vd"])
        _, pv = stats.ttest_rel(data[mkey]["vs"], data[mkey]["vd"])
        print(f"  {ml}: vs={vs_m:.3f} vd={vd_m:.3f} Delta={vs_m-vd_m:+.3f} p={pv:.2e}", flush=True)


if __name__ == "__main__":
    main()
