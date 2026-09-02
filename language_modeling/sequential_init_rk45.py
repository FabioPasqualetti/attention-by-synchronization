"""
Sequential vs random init on L5c (the Euler-ODE-trained model).

Backs no number reported in the paper. Kept because it is the producer of the
committed results/lm/sequential_init_rk45.json and documents a side experiment
on a second model; nothing in the manuscript cites its analytic PPL (118.95),
its Euler PPL (115.84), or any of its gaps.

The paper's WikiText-2 sequential-initialization claims -- "within 0.13 PPL"
and "closes the residual gap to 0.04 PPL" -- come from a DIFFERENT model, L6a
(analytic-trained, analytic PPL 110.2156), and are produced by
language_modeling/sequential_init_L6a.py. This file was previously annotated as
backing the 0.04 PPL claim; that attribution was wrong.

Uses the L5c model (trained with Euler ODE, KuramotoAttention on S^1) and
evaluates PPL under three inference modes:

  (a) Analytic:    analytic fixed point  theta* = arctan2(SUM W_ij sin phi_j,
                                                          SUM W_ij cos phi_j)
  (b) random init: theta_i(0) ~ Uniform[0, 2pi), evolved to T_max
  (c) sequential:  theta_i(0) = the previous position's INTEGRATED STATE AT
                   T_max (position 0 random) -- not its fixed point.

NOT RK45, despite the filename. The scalar S^1 ODE has a closed form, which
this script evaluates at T_max:
  theta(T) = theta* + 2 arctan(tan((theta(0) - theta*)/2) * exp(-||F|| * T_max))
This is the exact finite-horizon state, not a numerical approximation of it.
The filename is retained because the output file is a committed artifact.

The T_max=30 finite horizon causes incomplete convergence only for weakly
coupled tokens (||F|| ~ 0.1), where exp(-0.1 * 30) = 0.05 still allows ~5%
residual.

Reference result: results/lm/sequential_init_rk45.json
Prior Euler results at: results/lm/sequential_init.json (untouched)

Usage:
    python language_modeling/sequential_init_rk45.py
    python language_modeling/sequential_init_rk45.py --T_max 30 --seeds 0 1 2
    python language_modeling/sequential_init_rk45.py --smoke

Distinct from the TinyStories "reduces mean error 4.6x" experiment
(convergence/ts_convergence.py::verify_init_methods), which starts from the
previous token's analytic fixed point plus Gaussian jitter of scale 0.05.
Do not assume a change here applies to both.
"""

import argparse
import json
import math
import os
import statistics
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from training import paths  # noqa: E402

from oscillator_attention.transformer import SinusoidalPositionalEncoding
from training.data_utils import load_wikitext2, make_lm_loaders

RESULTS_DIR = os.path.join(paths.runs_root(), "lm")

L5C_CKPT = os.path.expanduser(
    os.path.join(paths.cache_root(), "checkpoints", "L5c_s0_ep30.pt")
)

TWO_PI = 2.0 * math.pi

# ── L5c model (KuramotoAttention + causal LM) ─────────────────────────────────

class _KuramotoAttn(nn.Module):
    """Scalar KuramotoAttention (S^1), as used in L5c."""
    def __init__(self, d_model=128, n_heads=4, T=50, N=500, dt=0.05, p=1, causal=True):
        super().__init__()
        dh = d_model // n_heads
        self.n_heads, self.d_head = n_heads, dh
        self.T, self.N, self.dt, self.p, self.causal = T, N, dt, p, causal
        self.W_f = nn.Linear(d_model, d_model, bias=False)
        self.W_g = nn.Linear(d_model, d_model, bias=False)
        self.alpha = nn.Parameter(torch.ones(n_heads))
        self.beta  = nn.Parameter(torch.zeros(n_heads))
        phi_init = (TWO_PI * torch.arange(T, dtype=torch.float32) / T
                    ).unsqueeze(0).expand(n_heads, -1).clone()
        self.phi = nn.Parameter(phi_init)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=True)

    def _coupling_weights(self, x, pad_mask=None):
        """Compute and return row-normalised W_ij: (B*H, T, T)."""
        B, T, _ = x.shape
        H, dh = self.n_heads, self.d_head
        fi = self.W_f(x).view(B,T,H,dh).transpose(1,2).reshape(B*H, T, dh)
        gj = self.W_g(x).view(B,T,H,dh).transpose(1,2).reshape(B*H, T, dh)
        raw = torch.bmm(fi, gj.transpose(1,2)) / (dh**0.5)
        alpha = self.alpha.unsqueeze(0).expand(B,H).reshape(B*H,1,1)
        beta  = self.beta.unsqueeze(0).expand(B,H).reshape(B*H,1,1)
        W = F.softplus(alpha * raw + beta)
        if self.causal:
            W = W * torch.tril(torch.ones(T, T, device=x.device))
        if pad_mask is not None:
            pad = (~pad_mask).float().view(B,1,T).expand(B,H,T).reshape(B*H,1,T)
            W = W * pad
        W = W / (W.sum(dim=2, keepdim=True) + 1e-8)
        return W

    def _readout(self, theta, phi, x, B, T):
        """Compute attn from theta angles, then value aggregation."""
        H, dh = self.n_heads, self.d_head
        cos_sim = torch.cos(theta.unsqueeze(2) - phi.unsqueeze(1))
        shifted = (1.0 + cos_sim).clamp(min=0.0) ** self.p
        if self.causal:
            shifted = shifted * torch.tril(torch.ones(T,T,device=x.device))
        attn = shifted / (shifted.sum(dim=2, keepdim=True) + 1e-8)
        V = self.W_v(x).view(B,T,H,dh).transpose(1,2).reshape(B*H, T, dh)
        out = torch.bmm(attn, V).view(B,H,T,dh).transpose(1,2).reshape(B,T,H*dh)
        return self.W_o(out), attn


class _L5cLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, T):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = _KuramotoAttn(d_model, n_heads, T)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout))

    def forward(self, x, pad_mask=None):  # standard Euler ODE
        B, T, _ = x.shape
        H, dh = self.attn.n_heads, self.attn.d_head
        normed = self.norm1(x)
        W = self.attn._coupling_weights(normed, pad_mask)
        phi = self.attn.phi[:,:T].unsqueeze(0).expand(B,H,T).reshape(B*H, T)
        theta = TWO_PI * torch.rand(B*H, T, device=x.device, dtype=x.dtype)
        for _ in range(self.attn.N):
            sin_diff = torch.sin(theta.unsqueeze(2) - phi.unsqueeze(1))
            theta = (theta - self.attn.dt * (W * sin_diff).sum(dim=2)) % TWO_PI
        out, _ = self.attn._readout(theta, phi, normed, B, T)
        x = x + out
        return x + self.ffn(self.norm2(x))


class L5cLM(nn.Module):
    def __init__(self, vocab_size=10000, d_model=128, n_heads=4, n_layers=2,
                 d_ff=512, max_seq_len=50, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc   = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)
        self.layers    = nn.ModuleList([
            _L5cLayer(d_model, n_heads, d_ff, dropout, max_seq_len)
            for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, pad_mask=None):
        h = self.pos_enc(self.embedding(x))
        for layer in self.layers:
            h = layer(h, pad_mask)
        return self.head(self.norm(h))


# ── Exact scalar ODE solution (equivalent to RK45 with rtol→0) ───────────────

def _exact_kuramoto_T(theta0, theta_star, K, T_max):
    """
    Exact solution of dθ/dt = -K sin(θ - θ*) at time T_max.
    All inputs are numpy arrays with the same shape.
    theta0, theta_star: angles in radians
    K: coupling strengths (scalar or array)
    """
    delta0 = theta0 - theta_star
    # Normalize to (-pi, pi]
    delta0 = (delta0 + np.pi) % (2*np.pi) - np.pi
    half = delta0 / 2.0
    tan_half = np.tan(half)
    exp_factor = np.exp(-np.clip(K * T_max, 0, 700))
    tan_final = tan_half * exp_factor
    delta_T = 2 * np.arctan(tan_final)
    return theta_star + delta_T


# ── Inference modes ───────────────────────────────────────────────────────────

@torch.no_grad()
def _compute_W_and_phi(model, x, pad_mask, layer_idx=0):
    """Compute row-normalised W: (B*H, T, T) and phi: (B*H, T) for given layer."""
    B, T = x.shape
    H = model.layers[layer_idx].attn.n_heads
    h = model.pos_enc(model.embedding(x))
    for i, layer in enumerate(model.layers):
        if i == layer_idx:
            normed = layer.norm1(h)
            W   = layer.attn._coupling_weights(normed, pad_mask)
            phi = layer.attn.phi[:, :T].unsqueeze(0).expand(B, H, T).reshape(B*H, T)
            return W, phi, normed
        # apply earlier layers in standard mode
        h = layer(h, pad_mask)
    raise ValueError(f"layer_idx {layer_idx} out of range")


@torch.no_grad()
def eval_ppl_l5c(model, loader, device, mode="euler", T_max=30.0, rng_seed=0):
    """
    Evaluate L5c PPL under a given inference mode.

    mode:
      'euler'      — standard Euler ODE (training mode, N=500 steps)
      'analytic'   — exact analytic fixed point (θ* = arctan2(ΣWsinφ, ΣWcosφ))
      'ode_random' — exact scalar ODE from random θ(0), finite T_max
      'ode_seq'    — exact scalar ODE, chained from the previous position's
                     INTEGRATED state at T_max (not its fixed point θ*)
    """
    model.eval()
    crit = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    total_loss, total_toks = 0.0, 0
    rng = np.random.default_rng(rng_seed)

    for batch in loader:
        x        = batch["tokens"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        B, T     = x.shape

        if mode == "euler":
            logits = model(x, pad_mask)
        else:
            # Custom inference: override attention computation
            h = model.pos_enc(model.embedding(x))

            for layer in model.layers:
                normed = layer.norm1(h)
                attn   = layer.attn
                H, dh  = attn.n_heads, attn.d_head
                W   = attn._coupling_weights(normed, pad_mask)   # (B*H, T, T)
                phi = attn.phi[:, :T].unsqueeze(0).expand(B, H, T).reshape(B*H, T)

                phi_np  = phi.cpu().numpy()                      # (B*H, T)
                W_np    = W.cpu().numpy()                        # (B*H, T, T)

                # Driving force: F_x, F_y = ΣW_ij cosφ_j, ΣW_ij sinφ_j
                F_x = (W_np * np.cos(phi_np[:, np.newaxis, :])).sum(axis=2)  # (B*H, T)
                F_y = (W_np * np.sin(phi_np[:, np.newaxis, :])).sum(axis=2)
                K   = np.sqrt(F_x**2 + F_y**2)                 # coupling strength
                theta_star = np.arctan2(F_y, F_x)               # (B*H, T)

                if mode == "analytic":
                    theta_T = theta_star

                elif mode == "ode_random":
                    theta0 = rng.uniform(0, TWO_PI, size=theta_star.shape)
                    theta_T = _exact_kuramoto_T(theta0, theta_star, K, T_max)

                elif mode == "ode_seq":
                    theta_T = np.zeros_like(theta_star)
                    # Position 0: random init
                    theta0_0 = rng.uniform(0, TWO_PI, size=(B*H,))
                    theta_T[:, 0] = _exact_kuramoto_T(
                        theta0_0, theta_star[:, 0], K[:, 0], T_max)
                    # Positions 1..T-1: init from previous x_star
                    for t in range(1, T):
                        # Previous converged angle = theta_T[:, t-1]
                        theta0_t = theta_T[:, t-1]
                        theta_T[:, t] = _exact_kuramoto_T(
                            theta0_t, theta_star[:, t], K[:, t], T_max)
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                theta_tensor = torch.tensor(theta_T, dtype=x.dtype, device=device)
                out, _ = attn._readout(theta_tensor, phi, normed, B, T)
                h = h + out
                h = h + layer.ffn(layer.norm2(h))

            logits = model.head(model.norm(h))

        logits_in = logits[:, :-1].reshape(-1, logits.size(-1))
        targets   = x[:, 1:].reshape(-1)
        mask      = ~pad_mask[:, 1:].reshape(-1)
        total_loss += crit(logits_in[mask], targets[mask]).item()
        total_toks += mask.sum().item()

    return math.exp(total_loss / max(total_toks, 1))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3 (exact finite-horizon): sequential vs random init on L5c")
    parser.add_argument("--T_max",  type=float, default=30.0,
                        help="Integration horizon (default 30)")
    parser.add_argument("--seeds",  type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--ckpt",   default=L5C_CKPT,
                        help="Path to L5c checkpoint")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--smoke",  action="store_true")
    args = parser.parse_args()

    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load data
    _, val_seqs, _, vocab = load_wikitext2()
    if args.smoke:
        val_seqs = val_seqs[:100]
    _, val_loader = make_lm_loaders([], val_seqs, 50, batch_size=32)

    # Load L5c model
    if not os.path.exists(args.ckpt):
        print(f"ERROR: L5c checkpoint not found: {args.ckpt}")
        sys.exit(1)
    model = L5cLM(vocab_size=len(vocab))
    model.load_state_dict(torch.load(args.ckpt, weights_only=False, map_location="cpu"))
    model = model.to(device)
    print(f"Loaded L5c checkpoint: {args.ckpt}")
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    print(f"\nExperiment 3 (exact finite-horizon): T_max={args.T_max}  seeds={args.seeds}")

    results = {}

    # (a) Analytic
    print("\n(a) Analytic ...", flush=True)
    ppl_analytic = eval_ppl_l5c(model, val_loader, device, "analytic")
    results["analytic"] = {"ppl": ppl_analytic}
    print(f"  PPL = {ppl_analytic:.2f}")

    # (b) Euler (training mode — sanity check, should match 115.83)
    print("\n(b) Euler ODE (training mode, sanity check) ...", flush=True)
    ppls_euler = []
    for s in args.seeds:
        torch.manual_seed(s)
        ppl = eval_ppl_l5c(model, val_loader, device, "euler", rng_seed=s)
        ppls_euler.append(ppl)
        print(f"  seed={s}: PPL = {ppl:.2f}", flush=True)
    results["euler"] = {
        "seeds": {str(s): p for s, p in zip(args.seeds, ppls_euler)},
        "mean": statistics.mean(ppls_euler),
        "std":  statistics.stdev(ppls_euler) if len(ppls_euler) > 1 else 0.0,
    }

    # (c) ODE random (T_max)
    print(f"\n(c) ODE random init (T_max={args.T_max}) ...", flush=True)
    ppls_random = []
    for s in args.seeds:
        ppl = eval_ppl_l5c(model, val_loader, device, "ode_random",
                            T_max=args.T_max, rng_seed=s)
        ppls_random.append(ppl)
        print(f"  seed={s}: PPL = {ppl:.2f}", flush=True)
    results["ode_random"] = {
        "seeds": {str(s): p for s, p in zip(args.seeds, ppls_random)},
        "mean": statistics.mean(ppls_random),
        "std":  statistics.stdev(ppls_random) if len(ppls_random) > 1 else 0.0,
    }

    # (d) ODE sequential (T_max) — 1 seed (deterministic given model)
    print(f"\n(d) ODE sequential init (T_max={args.T_max}) ...", flush=True)
    ppls_seq = []
    for s in [args.seeds[0]]:
        ppl = eval_ppl_l5c(model, val_loader, device, "ode_seq",
                            T_max=args.T_max, rng_seed=s)
        ppls_seq.append(ppl)
        print(f"  seed={s}: PPL = {ppl:.2f}", flush=True)
    results["ode_seq"] = {
        "seeds": {str(s): p for s, p in zip([args.seeds[0]], ppls_seq)},
        "mean": statistics.mean(ppls_seq),
        "std":  0.0,
    }

    results["config"] = {"T_max": args.T_max, "model": "L5c"}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "sequential_init_rk45.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    print("\n## Sequential-init results (WikiText-2, L5c, d_osc=2, exact finite-horizon)\n")
    print("| Mode | PPL |")
    print("|---|---|")
    print(f"| Analytic fixed point | {results['analytic']['ppl']:.2f} |")
    print(f"| Euler ODE (training) | {results['euler']['mean']:.2f} ± {results['euler']['std']:.2f} |")
    print(f"| Finite-horizon, random init (T={args.T_max}) | {results['ode_random']['mean']:.2f} ± {results['ode_random']['std']:.2f} |")
    print(f"| Finite-horizon, sequential init (T={args.T_max}) | {results['ode_seq']['mean']:.2f} |")

    gap_euler = results["euler"]["mean"] - results["analytic"]["ppl"]
    gap_random = results["ode_random"]["mean"] - results["analytic"]["ppl"]
    gap_seq = results["ode_seq"]["mean"] - results["analytic"]["ppl"]
    print(f"\nGaps vs analytic:")
    print(f"  Euler ODE:      +{gap_euler:.2f}")
    print(f"  Finite-horizon random init:     +{gap_random:.2f}")
    print(f"  Finite-horizon sequential init: +{gap_seq:.2f}")
    pct = (gap_euler - gap_seq) / gap_euler * 100 if gap_euler > 0 else 0
    print(f"  Sequential closes {pct:.0f}% of Euler gap")


if __name__ == "__main__":
    main()
