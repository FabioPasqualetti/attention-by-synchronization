"""kws_rank_sharpening — Rank vs. readout-sharpening exponent p (eval-only, CPU).

Trained KWS readout-sharpening checkpoints exist at p in {1,2,4}. They are stock
KWSTransformer(attn_type='lohe',
d_osc=2, non-causal), seed 3, from keyword_spotting/readout_exponent.py.

For each p, over the KWS validation set we measure:
  (a) the SIMILARITY-operator effective rank  S = 1 + x*·anchor  (p-independent; bound <= d_osc+1 = 3),
  (b) the REALIZED readout attention-matrix effective rank  a = (1+cos)^p / norm  (sharpens with p),
  (c) the checkpoint's own KWS val accuracy.
Reported alongside the paper's 3-seed sharpening accuracies (results/kws_p_ablation_lohe_3seeds.json).

Resumable per-run JSON.
"""
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training import harness, paths  # noqa: E402
paths.ensure_paths()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from oscillator_attention import KWSTransformer  # noqa: E402

EXP = "kws_rank_sharpening"
ARCH = dict(n_feats=40, d_model=32, n_heads=2, n_layers=1, n_classes=10,
            T=49, d_osc=2, dropout=0.1)
PS = [1, 2, 4]
CKPT = os.path.join(paths.runs_root(), "kws",
                    "checkpoints_readout_sharpening")
PABLATION = os.path.join(paths.results_root(), "kws_p_ablation_lohe_3seeds.json")


def _spectrum_eff_rank(A):
    s = torch.linalg.svdvals(A.float()).cpu().numpy()
    s = s[s > 1e-10]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


@torch.no_grad()
def _sim_op(mod, x):
    """S = 1 + x*·anchor (unmasked; KWS is non-causal). rank <= d_osc+1."""
    B, T, _ = x.shape
    H, Dh = mod.n_heads, mod.d_head
    q = mod.W_q(x).view(B, T, H, Dh).transpose(1, 2)
    k = mod.W_k(x).view(B, T, H, Dh).transpose(1, 2)
    raw = torch.einsum('bhid,bhjd->bhij', q, k) / math.sqrt(Dh)
    W = F.softplus(raw)
    anc = mod.anchors[:, :T, :]
    h = torch.einsum('bhij,hjd->bhid', W, anc)
    xstar = F.normalize(h, dim=-1, eps=1e-8)
    cos = torch.einsum('bhid,hjd->bhij', xstar, anc)
    return 1.0 + cos  # (B,H,T,T)


@torch.no_grad()
def run_p(p, device):
    key = f"kws_p{p}"
    if harness.exists(EXP, key):
        return
    ckpt = os.path.join(CKPT, f"KWS_p{p}_s3_ep30.pt")
    if not os.path.exists(ckpt):
        print(f"kws_rank_sharpening p={p}: no ckpt {ckpt}", flush=True); return
    from training.kws_data import cached_loaders
    _, val, _, _ = cached_loaders(0, batch_size=64)
    model = KWSTransformer(**ARCH, attn_type="lohe", p=p)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
    model.to(device).eval()
    attn_mod = model.layers[0].attn

    sim_ranks, readout_ranks = [], []
    correct = total = 0
    captured = {}
    ph = attn_mod.register_forward_pre_hook(lambda m, a: captured.__setitem__("x", a[0].detach()))
    fh = attn_mod.register_forward_hook(lambda m, i, o: captured.__setitem__("attn", o[1].detach()))
    for feats, labels in val:
        if feats is None:
            continue
        feats = feats.to(device)
        logits = model(feats)
        correct += (logits.argmax(1) == labels.to(device)).sum().item()
        total += labels.size(0)
        a = captured["attn"]                       # realized readout attention (B,H,T,T)
        if a.dim() == 3:
            a = a.view(feats.shape[0], ARCH["n_heads"], ARCH["T"], ARCH["T"])
        sim = _sim_op(attn_mod, captured["x"])
        for b in range(a.shape[0]):
            for h in range(a.shape[1]):
                readout_ranks.append(_spectrum_eff_rank(a[b, h]))
                sim_ranks.append(_spectrum_eff_rank(sim[b, h]))
    ph.remove(); fh.remove()
    acc = correct / max(total, 1)

    import json
    pab = json.load(open(PABLATION)).get(str(p), {})
    payload = {
        "p": p, "seed": 3, "ckpt": ckpt,
        "sim_op_eff_rank": {"mean": float(np.mean(sim_ranks)), "std": float(np.std(sim_ranks))},
        "readout_attn_eff_rank": {"mean": float(np.mean(readout_ranks)), "std": float(np.std(readout_ranks))},
        "bound_d_osc_plus_1": ARCH["d_osc"] + 1,
        "ckpt_val_acc": acc,
        "paper_3seed_acc_mean": pab.get("mean"),
        "n_matrices": len(sim_ranks),
    }
    harness.save_result(EXP, key, payload)
    print(f"DONE kws_rank_sharpening p={p}: sim-op rank={payload['sim_op_eff_rank']['mean']:.3f} "
          f"readout rank={payload['readout_attn_eff_rank']['mean']:.3f} "
          f"acc={acc*100:.2f}% (paper mean {pab.get('mean')})", flush=True)
    harness.free_memory(device)


def main():
    device = torch.device("cpu")  # eval-only, CPU (do not contend with MPS training)
    print("kws_rank_sharpening device=cpu", flush=True)
    for p in PS:
        run_p(p, device)
    print("kws_rank_sharpening COMPLETE", flush=True)


if __name__ == "__main__":
    main()
