"""robustness_frequency_disorder — Frequency-disorder robustness (CPU, ODE path).

On trained KWS (d_osc=2) and TinyStories (d_osc=8) oscillator checkpoints, integrate the
Lohe dynamics with a per-oscillator natural-frequency (skew-symmetric) generator:

    dx_i/dt = A_i x_i + (I - x_i x_i^T) h_i ,   A_i = (M_i - M_i^T)/2 , M_i ~ N(0, s^2)^{DxD}

(For d_osc=2 this is a scalar rotation rate Omega_i ~ N(0,s^2): A=[[0,-Omega],[Omega,0]].)
A_i skew-symmetric preserves the sphere; we renormalize. Integrate from the undisturbed
fixed point x0 = normalize(h_i) to a long horizon T=30 (scipy RK45 rtol=atol=1e-6, Euler
fallback), read out attention from x(T), and measure the task metric vs s.

s in {0.01,0.05,0.1,0.2,0.5}, 5 draws each. Report metric vs s (mean +/- std), the tolerance
envelope, and an abrupt locking-loss threshold s* if one exists (largest drop between adjacent
s levels). ODE eval subset-capped (KWS <=512 samples, TS 100 seqs) — documented.

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
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from training.lohe_probe import _coupling, _sigma_of, LOHE_TYPES  # noqa: E402

EXP = "robustness_frequency_disorder"
CACHE = os.path.expanduser(os.path.join(paths.cache_root(), "tinystories"))
S_GRID = [0.01, 0.05, 0.1, 0.2, 0.5]
N_DRAWS = 5
T_HORIZON = 30.0
ODE_MAX_KWS = 512
ODE_MAX_TS = 100
FIGDIR = os.path.join(paths.REPO_ROOT, "figures", "diagnostics", EXP)


def _integrate_freq(g, x0, A, Tset, rtol=1e-6, atol=1e-6):
    """Integrate dx/dt = A x + (I - x x^T) g for M oscillators. g,x0:(M,D); A:(M,D,D)."""
    from scipy.integrate import solve_ivp
    M, D = g.shape

    def rhs(t, y):
        x = y.reshape(M, D)
        Ax = np.einsum('mij,mj->mi', A, x)
        radial = (x * g).sum(axis=1, keepdims=True)
        return (Ax + g - radial * x).reshape(-1)

    try:
        sol = solve_ivp(rhs, (0.0, float(Tset)), x0.reshape(-1), method="RK45",
                        rtol=rtol, atol=atol, t_eval=[float(Tset)])
        y = np.asarray(sol.y, dtype=float)
        if getattr(sol, "success", False) and y.ndim == 2 and y.shape[1] >= 1:
            return y[:, -1].reshape(M, D)
    except Exception:
        pass
    # Euler fallback (renormalized), dt=0.05
    N = max(1, int(round(float(Tset) / 0.05)))
    x = x0.copy(); dt = 0.05
    for _ in range(N):
        Ax = np.einsum('mij,mj->mi', A, x)
        radial = (x * g).sum(axis=1, keepdims=True)
        x = x + dt * (Ax + g - radial * x)
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    return x


class FreqDisorderLohe(nn.Module):
    """Wrap a trained Lohe module; replace analytic x* by frequency-disordered settling."""

    def __init__(self, src, s, rng_seed):
        super().__init__()
        self.src = src; self.s = s; self.rng_seed = rng_seed

    def forward(self, x, padding_mask=None):
        m = self.src
        B, T, _ = x.shape
        H, D_h, D = m.n_heads, m.d_head, m.d_osc
        q = m.W_q(x).view(B, T, H, D_h).transpose(1, 2)
        k = m.W_k(x).view(B, T, H, D_h).transpose(1, 2)
        raw = torch.einsum('bhid,bhjd->bhij', q, k) / math.sqrt(D_h)
        W = _coupling(_sigma_of(m), raw)
        if m.causal:
            W = W * m._causal_mask[:T, :T]
        if padding_mask is not None:
            W = W * (~padding_mask).float().view(B, 1, 1, T)
        anc = m.anchors[:, :T, :]
        h = torch.einsum('bhij,hjd->bhid', W, anc)              # (B,H,T,D)
        g = h.detach().cpu().numpy().reshape(-1, D)
        x0 = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-12)  # undisturbed FP
        rng = np.random.default_rng(self.rng_seed)
        Mrand = rng.normal(0.0, self.s, size=(g.shape[0], D, D))
        A = 0.5 * (Mrand - np.transpose(Mrand, (0, 2, 1)))      # skew-symmetric
        xset = _integrate_freq(g, x0, A, T_HORIZON)
        x_star = F.normalize(torch.from_numpy(xset).float().view(B, H, T, D), dim=-1, eps=1e-8)
        cos = torch.einsum('bhid,hjd->bhij', x_star, anc)
        attn = (1.0 + cos).clamp(min=0.0)
        if m.p != 1:
            attn = attn ** m.p
        if m.causal:
            attn = attn * m._causal_mask[:T, :T]
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        v = m.W_v(x).view(B, T, H, D_h).transpose(1, 2)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, H * D_h)
        return m.W_o(out), attn


def _swap(model, s, rng_seed):
    tgt = []
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, LOHE_TYPES):
                tgt.append((parent, name, child))
    for parent, name, child in tgt:
        setattr(parent, name, FreqDisorderLohe(child, s, rng_seed))
    return model


# ── task builders / metrics ──────────────────────────────────────────────────

def build_kws(ckpt):
    from oscillator_attention.kws_pe_model import KWSTransformerPE
    m = KWSTransformerPE(pe="none", attn_type="lohe", p=1, n_feats=40, d_model=32,
                         n_heads=2, n_layers=1, n_classes=10, T=49, d_osc=2, dropout=0.1)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
    return m.eval()


def build_ts(ckpt):
    from oscillator_attention.sigma_models import LoheLMSigma
    vocab, _, _, _ = load_tinystories(max_len=128)
    m = LoheLMSigma(vocab_size=len(vocab), d_osc=8, sigma="softplus", d_model=128,
                    n_heads=4, n_layers=2, d_ff=512, max_seq_len=128, dropout=0.1)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
    return m.eval()


@torch.no_grad()
def kws_acc(model):
    from training.kws_data import cached_loaders
    _, val, _, _ = cached_loaders(0, batch_size=64)
    correct = total = 0
    for feats, labels in val:
        if feats is None:
            continue
        correct += (model(feats).argmax(1) == labels).sum().item()
        total += labels.size(0)
        if total >= ODE_MAX_KWS:
            break
    return correct / max(total, 1)


def _ts_val():
    # via load_tinystories, not the raw cache: it resolves the shipped
    # data/tinystories_eval/ vocabulary, which is what indexes the released
    # checkpoints. Reading <cache>/tinystories directly fails on a clean clone.
    _, _, _, val_all = load_tinystories(max_len=128)
    val = val_all[:ODE_MAX_TS]
    toks = torch.zeros(len(val), 128, dtype=torch.long)
    for i, c in enumerate(val):
        c = c[:128]; toks[i, :len(c)] = torch.tensor(c, dtype=torch.long)
    return toks, (toks == 0)


@torch.no_grad()
def ts_ppl(model, toks, pad):
    ce = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    tot_loss = tot_tok = 0
    B = 25
    for s in range(0, toks.shape[0], B):
        x = toks[s:s + B]; p = pad[s:s + B]
        logits = model(x, padding_mask=p)
        tgt = x[:, 1:].contiguous(); lg = logits[:, :-1].contiguous()
        tot_loss += ce(lg.view(-1, lg.size(-1)), tgt.view(-1)).item()
        tot_tok += (tgt != 0).sum().item()
    return math.exp(tot_loss / max(tot_tok, 1))


def run_target(target, ckpt):
    if not os.path.exists(ckpt):
        print(f"robustness_frequency_disorder {target}: no ckpt {ckpt}", flush=True); return
    metric = "acc" if target == "kws" else "ppl"
    ts_data = None if target == "kws" else _ts_val()
    # baseline (s=0, undisturbed) for reference
    bkey = f"{target}_baseline"
    if not harness.exists(EXP, bkey):
        m = build_kws(ckpt) if target == "kws" else build_ts(ckpt)
        _swap(m, 0.0, 0)
        val = kws_acc(m) if target == "kws" else ts_ppl(m, *ts_data)
        harness.save_result(EXP, bkey, {"target": target, "s": 0.0, metric: val})
        print(f"DONE {bkey}: {metric}={val:.4f}", flush=True)
    for s in S_GRID:
        for d in range(N_DRAWS):
            key = f"{target}_s{s:g}_d{d}"
            if harness.exists(EXP, key):
                continue
            m = build_kws(ckpt) if target == "kws" else build_ts(ckpt)
            _swap(m, s, 1000 * d + int(s * 1000))
            val = kws_acc(m) if target == "kws" else ts_ppl(m, *ts_data)
            harness.save_result(EXP, key, {"target": target, "s": s, "draw": d, metric: val})
            print(f"DONE {key}: {metric}={val:.4f}", flush=True)
            harness.free_memory(None)


def _summarize_and_plot():
    import glob, json
    recs = [json.load(open(p)) for p in glob.glob(os.path.join(harness.RUNS_ROOT, EXP, "*.json"))]
    os.makedirs(FIGDIR, exist_ok=True)
    for target, metric in [("kws", "acc"), ("ts", "ppl")]:
        pts = {}
        base = None
        for r in recs:
            if r["target"] != target or metric not in r:
                continue
            if r["s"] == 0.0:
                base = r[metric]
            else:
                pts.setdefault(r["s"], []).append(r[metric])
        if not pts:
            continue
        xs = sorted(pts)
        means = [float(np.mean(pts[x])) for x in xs]
        stds = [float(np.std(pts[x])) for x in xs]
        # abrupt locking-loss s*: largest adjacent jump (for acc: drop; ppl: rise), incl baseline
        seq = ([base] if base is not None else []) + means
        sx = ([0.0] if base is not None else []) + xs
        jumps = [(abs(seq[i + 1] - seq[i]), sx[i + 1]) for i in range(len(seq) - 1)]
        sstar = max(jumps)[1] if jumps else None
        harness.save_result(EXP, f"{target}_summary", {
            "target": target, "metric": metric, "baseline": base,
            "s": xs, "mean": means, "std": stds, "locking_loss_s_star": sstar,
            "note": "s* = s level with the largest adjacent change in the metric"})
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3)
        if base is not None:
            ax.axhline(base, ls="--", color="gray", lw=1, label="s=0 (undisturbed)")
        if sstar is not None:
            ax.axvline(sstar, ls=":", color="crimson", lw=1, label=f"largest jump at s={sstar:g}")
        ax.set_xscale("log"); ax.set_xlabel("frequency-disorder scale s")
        ax.set_ylabel(metric.upper()); ax.set_title(f"robustness_frequency_disorder {target.upper()} — freq disorder")
        ax.legend(fontsize=7); fig.tight_layout()
        fig.savefig(os.path.join(FIGDIR, f"{target}_freqdisorder.pdf")); plt.close(fig)
        print(f"robustness_frequency_disorder {target}: s*={sstar}; fig -> {FIGDIR}/{target}_freqdisorder.pdf", flush=True)


def main():
    print("robustness_frequency_disorder device=cpu", flush=True)
    run_target("kws", os.path.join(harness.RUNS_ROOT, "kws_position_matched", "ckpt", "osc_none_s0_final.pt"))
    run_target("ts", os.path.join(harness.RUNS_ROOT, "lm_coupling_function", "ckpt", "ts_softplus_s0.pt"))
    _summarize_and_plot()
    print("robustness_frequency_disorder COMPLETE", flush=True)


if __name__ == "__main__":
    main()
