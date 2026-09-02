"""
Sequential vs random initialization on the WikiText-2 d_osc=2 model (L6a).

Backs two printed numbers: under random initialization the finite-horizon
solution recovers the analytic fixed-point perplexity within 0.13 PPL, and
sequential initialization closes the residual gap to 0.04 PPL.

Model: L6a -- LoheAttention, trained against the analytic fixed point, d_osc=2.
Its analytic validation PPL is 110.2156, i.e. the published WikiText-2 d_osc=2
seed-0 model. (The L5c experiment in sequential_init_rk45.py is a DIFFERENT
model, PPL 118.95, and backs no reported number.)

Three inference modes:
  (a) analytic:    standard forward pass through the analytic fixed point
  (b) random init: theta_i(0) ~ Uniform[0, 2pi), evolved to T_max
  (c) sequential:  theta_i(0) = the previous position's INTEGRATED STATE AT
                   T_max -- not the previous position's fixed point. See the
                   `theta_T[:, t-1]` argument in eval_ppl_l6a; passing
                   theta_star[:, t-1] there would be a different experiment.

NOT RK45, despite the output filename. At d_osc=2 the scalar Lohe ODE
    dtheta/dt = -K sin(theta - theta*)
admits a closed form, which this script evaluates at T_max:
    theta(T) = theta* + 2 arctan(tan((theta(0) - theta*)/2) * exp(-K*T))
That is the exact finite-horizon state rather than a numerical approximation of
it, so the measurement isolates the finite-horizon effect from integrator
error. The output filename sequential_init_rk45_L6a.json predates this
correction and is left unchanged because it is a committed artifact.

L6a driving forces have mean ||F|| ~ 15, so exp(-||F||*30) ~ 0 and the
finite-horizon state is at the fixed point for virtually every token.

Seeds: mode (c) always runs exactly one seed (args.seeds[0]) by construction.
The committed reference used five random-init seeds:
    python language_modeling/sequential_init_L6a.py --seeds 0 1 2 3 4
The --seeds default below is 0 1 2 and does NOT reproduce the committed file.

Reference result: results/lm/sequential_init_rk45_L6a.json
Writes to:        <runs>/lm/sequential_init_rk45_L6a.json

Distinct from the TinyStories "reduces mean error 4.6x" sequential-init
experiment (convergence/ts_convergence.py::verify_init_methods), which starts
from the previous token's analytic fixed point plus Gaussian jitter of scale
0.05. A change here does not apply to that one.
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

from oscillator_attention import LoheLanguageTransformer
from training.data_utils import load_wikitext2, make_lm_loaders

RESULTS_DIR = os.path.join(paths.runs_root(), "lm")

L6A_CKPT = os.path.expanduser(
    os.path.join(paths.cache_root(), "checkpoints", "L6a_s0_ep30.pt")
)

TWO_PI = 2.0 * math.pi
ARCH = dict(d_model=128, n_heads=4, n_layers=2, d_ff=512,
            max_seq_len=50, dropout=0.0)


# ── Exact scalar ODE solution on S^1 ─────────────────────────────────────────

def _exact_kuramoto_T(theta0: np.ndarray, theta_star: np.ndarray,
                      K: np.ndarray, T_max: float) -> np.ndarray:
    """
    Exact solution of dθ/dt = -K sin(θ - θ*) at time T_max.
    All inputs are numpy arrays with the same shape.
    """
    delta0 = (theta0 - theta_star + math.pi) % (2 * math.pi) - math.pi
    tan_half = np.tan(delta0 / 2.0)
    exp_factor = np.exp(-np.clip(K * T_max, 0, 700))
    delta_T = 2.0 * np.arctan(tan_half * exp_factor)
    return theta_star + delta_T


# ── Inference modes on L6a ───────────────────────────────────────────────────

@torch.no_grad()
def eval_ppl_l6a(model, loader, device, mode="analytic",
                 T_max=30.0, rng_seed=0):
    """
    Evaluate L6a PPL under a given inference mode.

    mode:
      'analytic'   — standard model forward (trains with analytic fixed point)
      'ode_random' — replace x* with exact scalar ODE from random θ(0)
      'ode_seq'    — replace x* with the finite-horizon ODE state, chained from
                     the previous position's INTEGRATED state at T_max (not its
                     fixed point; see the theta_T[:, t-1] argument below)
    """
    model.eval()
    crit = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    total_loss, total_toks = 0.0, 0
    rng = np.random.default_rng(rng_seed)

    for batch in loader:
        x        = batch["tokens"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        B, T     = x.shape

        if mode == "analytic":
            logits = model(x, padding_mask=pad_mask)
        else:
            h = model.pos_enc(model.embedding(x))

            for layer in model.layers:
                normed   = layer.norm1(h)
                attn_mod = layer.attn
                H, D_h, D_osc = attn_mod.n_heads, attn_mod.d_head, attn_mod.d_osc

                # Coupling weights (LoheAttention, unnormalized, causal)
                q = attn_mod.W_q(normed).view(B, T, H, D_h).transpose(1, 2)
                k = attn_mod.W_k(normed).view(B, T, H, D_h).transpose(1, 2)
                W = F.softplus(
                    torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(D_h))
                W = W * torch.tril(torch.ones(T, T, device=device))
                if pad_mask is not None:
                    W = W * (~pad_mask).float().view(B, 1, 1, T)

                anc     = attn_mod.anchors[:, :T, :]              # (H, T, D_osc)
                driving = torch.einsum("bhij,hjd->bhid", W, anc)  # (B,H,T,D_osc)

                # Convert to numpy for ODE
                drv_np = driving.cpu().numpy()  # (B, H, T, D_osc)
                BH = B * H

                if D_osc == 2:
                    # Use exact scalar S^1 ODE solution
                    drv_flat = drv_np.reshape(BH, T, 2)
                    F_x = drv_flat[:, :, 0]   # (B*H, T)
                    F_y = drv_flat[:, :, 1]
                    K         = np.sqrt(F_x**2 + F_y**2)
                    theta_star = np.arctan2(F_y, F_x)

                    if mode == "ode_random":
                        theta0 = rng.uniform(0, TWO_PI, size=theta_star.shape)
                        theta_T = _exact_kuramoto_T(theta0, theta_star, K, T_max)
                    else:  # ode_seq
                        theta_T = np.zeros_like(theta_star)
                        theta_T[:, 0] = _exact_kuramoto_T(
                            rng.uniform(0, TWO_PI, size=(BH,)),
                            theta_star[:, 0], K[:, 0], T_max)
                        for t in range(1, T):
                            theta_T[:, t] = _exact_kuramoto_T(
                                theta_T[:, t-1],
                                theta_star[:, t], K[:, t], T_max)

                    x_star_np = np.stack([np.cos(theta_T), np.sin(theta_T)],
                                         axis=-1)  # (B*H, T, 2)
                    x_star = torch.tensor(
                        x_star_np.reshape(B, H, T, 2),
                        dtype=driving.dtype, device=device)
                else:
                    # General case (d_osc > 2): use analytic as fallback
                    x_star = F.normalize(driving, dim=-1, eps=1e-8)

                # Cosine similarity with anchors → attention
                cos_sim = torch.einsum("bhid,hjd->bhij", x_star, anc)
                attn = (1.0 + cos_sim).clamp(min=0.0)
                attn = attn * torch.tril(torch.ones(T, T, device=device))
                attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)

                # Value aggregation
                v   = attn_mod.W_v(normed).view(B, T, H, D_h).transpose(1, 2)
                out = torch.einsum("bhij,bhjd->bhid", attn, v)
                out = out.transpose(1, 2).contiguous().view(B, T, H * D_h)
                out = attn_mod.W_o(out)

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
        description="Exp 3 (exact finite-horizon, L6a): sequential vs random init")
    parser.add_argument("--T_max",  type=float, default=30.0)
    parser.add_argument("--seeds",  type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--ckpt",   default=L6A_CKPT)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--smoke",  action="store_true")
    args = parser.parse_args()

    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    _, val_seqs, _, vocab = load_wikitext2()
    if args.smoke:
        val_seqs = val_seqs[:100]
    _, val_loader = make_lm_loaders([], val_seqs, 50, batch_size=32)

    if not os.path.exists(args.ckpt):
        print(f"ERROR: L6a checkpoint not found: {args.ckpt}")
        sys.exit(1)

    model = LoheLanguageTransformer(vocab_size=len(vocab), d_osc=2, **ARCH)
    model.load_state_dict(torch.load(args.ckpt, weights_only=False, map_location="cpu"))
    model = model.to(device)
    print(f"Loaded L6a: {args.ckpt}")

    # Also report driving force stats for reference
    with torch.no_grad():
        sample_batch = next(iter(val_loader))
        x  = sample_batch["tokens"].to(device)[:8]
        pm = sample_batch["pad_mask"].to(device)[:8]
        B, T = x.shape
        h  = model.pos_enc(model.embedding(x))
        l  = model.layers[0]
        nd = l.norm1(h)
        am = l.attn
        q  = am.W_q(nd).view(B, T, am.n_heads, am.d_head).transpose(1, 2)
        k  = am.W_k(nd).view(B, T, am.n_heads, am.d_head).transpose(1, 2)
        W  = F.softplus(torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(am.d_head))
        W  = W * torch.tril(torch.ones(T, T, device=device))
        if pm is not None:
            W = W * (~pm).float().view(B, 1, 1, T)
        anc = am.anchors[:, :T, :]
        drv = torch.einsum("bhij,hjd->bhid", W, anc)
        K   = drv.norm(dim=-1).cpu().float()
        print(f"L6a driving force ||F||: mean={K.mean():.2f} min={K.min():.3f} max={K.max():.2f}")
        print(f"  exp(-||F||*30) mean: {torch.exp(-K * 30).mean():.2e}")

    results = {}

    print(f"\nExp 3 (exact finite-horizon, L6a): T_max={args.T_max}  seeds={args.seeds}")

    # (a) Analytic
    print("\n(a) Analytic ...", flush=True)
    ppl_analytic = eval_ppl_l6a(model, val_loader, device, "analytic")
    results["analytic"] = {"ppl": ppl_analytic}
    print(f"  PPL = {ppl_analytic:.2f}")

    # (b) finite-horizon solution, random initialization
    print(f"\n(b) random init (T_max={args.T_max}) ...", flush=True)
    ppls_random = []
    for s in args.seeds:
        ppl = eval_ppl_l6a(model, val_loader, device, "ode_random",
                            T_max=args.T_max, rng_seed=s)
        ppls_random.append(ppl)
        print(f"  seed={s}: PPL = {ppl:.2f}", flush=True)
    results["ode_random"] = {
        "seeds": {str(s): p for s, p in zip(args.seeds, ppls_random)},
        "mean": statistics.mean(ppls_random),
        "std":  statistics.stdev(ppls_random) if len(ppls_random) > 1 else 0.0,
        "n":    len(ppls_random),
    }

    # (c) finite-horizon solution, sequential initialization
    print(f"\n(c) sequential init (T_max={args.T_max}) ...", flush=True)
    ppls_seq = []
    for s in [args.seeds[0]]:
        ppl = eval_ppl_l6a(model, val_loader, device, "ode_seq",
                            T_max=args.T_max, rng_seed=s)
        ppls_seq.append(ppl)
        print(f"  seed={s}: PPL = {ppl:.2f}", flush=True)
    results["ode_seq"] = {
        "seeds": {str(s): p for s, p in zip([args.seeds[0]], ppls_seq)},
        "mean": statistics.mean(ppls_seq), "std": 0.0,
    }

    results["config"] = {"T_max": args.T_max, "model": "L6a", "d_osc": 2}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "sequential_init_rk45_L6a.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    print("\n## Sequential-init results (WikiText-2, L6a, d_osc=2, exact finite-horizon)\n")
    print("| Mode | PPL |")
    print("|---|---|")
    print(f"| Analytic (training mode) | {results['analytic']['ppl']:.2f} |")
    print(f"| Finite-horizon, random init (T={args.T_max}) | {results['ode_random']['mean']:.2f} ± {results['ode_random']['std']:.2f} |")
    print(f"| Finite-horizon, sequential init (T={args.T_max}) | {results['ode_seq']['mean']:.2f} |")

    gap_r = results["ode_random"]["mean"] - results["analytic"]["ppl"]
    gap_s = results["ode_seq"]["mean"]    - results["analytic"]["ppl"]
    print(f"\nFinite-horizon random init vs analytic: {gap_r:+.2f} PPL")
    print(f"Finite-horizon sequential init vs analytic: {gap_s:+.2f} PPL")


if __name__ == "__main__":
    main()
