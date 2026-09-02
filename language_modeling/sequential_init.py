"""
Experiment 3: Sequential vs random initialization (WikiText-2, d_osc=2).

Evaluates a trained L6a (d_osc=2) model under three inference modes:
  (a) Analytic:        use the analytic fixed point (standard forward pass)
  (b) ODE, random:     integrate Lohe ODE from k(0) ~ Uniform(S^1), N=500 steps
  (c) ODE, sequential: init k_i(0) = converged k_{i-1}^* from previous position;
                        position 0 uses random init

Expected: sequential init sits between random-init ODE and analytic, closer to
analytic, confirming the discussion claim about closing the PPL gap.

Usage:
    python language_modeling/sequential_init.py
    python language_modeling/sequential_init.py --ckpt path/to/checkpoint.pt
    python language_modeling/sequential_init.py --seeds 0 1 2
    python language_modeling/sequential_init.py --smoke

Backs the paper's "sequential initialization ... closes the residual gap to 0.04 PPL"
(WikiText-2, d_osc=2). NOT the "reduces mean error 4.6x" claim, which is a different
experiment on TinyStories with a different scheme -- see
convergence/ts_convergence.py::verify_init_methods. This one chains the previous
position's CONVERGED state with no jitter; that one starts from the previous token's
analytic fixed point plus Gaussian jitter. Do not assume a change here applies to both.
"""

import argparse
import json
import math
import os
import statistics
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from training import paths  # noqa: E402

from oscillator_attention import LoheLanguageTransformer
from training.data_utils import load_wikitext2, make_lm_loaders

RESULTS_DIR = os.path.join(paths.runs_root(), "lm")

# Default location for the trained d_osc=2 checkpoints (not shipped; see README)
RESEARCH_CKPT_TEMPLATE = os.path.expanduser(
    os.path.join(paths.cache_root(), "checkpoints", "L6a_s{seed}_ep30.pt")
)

ARCH = dict(d_model=128, n_heads=4, n_layers=2, d_ff=512,
            max_seq_len=50, dropout=0.0)


# ── ODE inference utilities ────────────────────────────────────────────────────

@torch.no_grad()
def _compute_driving(attn_module, normed_x):
    """
    Compute driving force F_i = sum_{j<=i} W_ij * anchor_j for all positions.
    Returns (B, H, T, D_osc).
    """
    B, T, _ = normed_x.shape
    H, D_h, D_osc = attn_module.n_heads, attn_module.d_head, attn_module.d_osc

    q = attn_module.W_q(normed_x).view(B, T, H, D_h).transpose(1, 2)
    k = attn_module.W_k(normed_x).view(B, T, H, D_h).transpose(1, 2)
    W = F.softplus(torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(D_h))

    device = normed_x.device
    W = W * torch.tril(torch.ones(T, T, device=device))

    anc = attn_module.anchors[:, :T, :]   # (H, T, D_osc)
    driving = torch.einsum("bhij,hjd->bhid", W, anc)   # (B, H, T, D_osc)
    return driving


@torch.no_grad()
def _lohe_euler(x_init, driving, N=500, dt=0.05):
    """Euler integration of Lohe ODE. x_init: (..., D), driving: (..., D)."""
    x = F.normalize(x_init, dim=-1, eps=1e-8)
    for _ in range(N):
        radial = (x * driving).sum(dim=-1, keepdim=True)
        x = F.normalize(x + dt * (driving - radial * x), dim=-1, eps=1e-8)
    return x


@torch.no_grad()
def eval_ppl_with_mode(model, loader, device, mode="analytic", N=500, dt=0.05,
                        rng_seed=0):
    """
    Compute PPL under a specified inference mode.

    mode:
      'analytic'   — standard forward pass (analytic fixed point)
      'ode_random' — replace x_star with Euler-ODE from random init
      'ode_seq'    — replace x_star with Euler-ODE from sequential init
    """
    model.eval()
    crit = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    total_loss, total_toks = 0.0, 0
    torch.manual_seed(rng_seed)

    for batch in loader:
        x        = batch["tokens"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        B, T     = x.shape

        if mode == "analytic":
            logits = model(x, padding_mask=pad_mask)
        else:
            # Custom forward: replace analytic x_star with ODE result
            h = model.pos_enc(model.embedding(x))
            for layer in model.layers:
                normed   = layer.norm1(h)
                attn_mod = layer.attn
                H, D_h, D_osc = (attn_mod.n_heads, attn_mod.d_head,
                                  attn_mod.d_osc)

                q = attn_mod.W_q(normed).view(B, T, H, D_h).transpose(1, 2)
                k = attn_mod.W_k(normed).view(B, T, H, D_h).transpose(1, 2)
                W = F.softplus(
                    torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(D_h))
                W = W * torch.tril(torch.ones(T, T, device=device))
                if pad_mask is not None:
                    W = W * (~pad_mask).float().view(B, 1, 1, T)

                anc = attn_mod.anchors[:, :T, :]      # (H, T, D_osc)
                driving = torch.einsum("bhij,hjd->bhid", W, anc)  # (B,H,T,D)

                if mode == "ode_random":
                    x_init = torch.randn(B, H, T, D_osc, device=device)
                    x_star = _lohe_euler(x_init, driving, N, dt)
                else:  # ode_seq: init each token from previous x_star
                    x_init = torch.randn(B, H, 1, D_osc, device=device)
                    x_prev = F.normalize(x_init, dim=-1, eps=1e-8)  # pos 0 random
                    x_stars = []
                    for t in range(T):
                        drv_t = driving[:, :, t:t+1, :]   # (B, H, 1, D_osc)
                        # init from previous position's x_star
                        x_t = _lohe_euler(x_prev, drv_t, N, dt)
                        x_stars.append(x_t)
                        x_prev = x_t.detach()
                    x_star = torch.cat(x_stars, dim=2)   # (B, H, T, D_osc)

                cos_sim = torch.einsum("bhid,hjd->bhij", x_star, anc)
                attn = (1.0 + cos_sim).clamp(min=0.0)
                attn = attn * torch.tril(torch.ones(T, T, device=device))
                attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)

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
        description="Sequential vs random init study (WikiText-2, d_osc=2)")
    parser.add_argument("--ckpt",    default=None,
                        help="Path to checkpoint (default: RESEARCH_CKPT_TEMPLATE)")
    parser.add_argument("--seeds",   type=int, nargs="+", default=[0, 1, 2],
                        help="Seeds for random-init ODE modes")
    parser.add_argument("--N",       type=int, default=500,
                        help="ODE steps (default 500)")
    parser.add_argument("--dt",      type=float, default=0.05)
    parser.add_argument("--ckpt_seed", type=int, default=0,
                        help="Which research-code checkpoint to load (default 0)")
    parser.add_argument("--device",  default="mps")
    parser.add_argument("--smoke",   action="store_true")
    args = parser.parse_args()

    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load data
    train_seqs, val_seqs, _, vocab = load_wikitext2(max_seq_len=50)
    vocab_size = len(vocab)
    if args.smoke:
        val_seqs = val_seqs[:100]
    _, val_loader = make_lm_loaders(train_seqs, val_seqs, 50, batch_size=32)

    # Load model
    ckpt_path = args.ckpt or RESEARCH_CKPT_TEMPLATE.format(seed=args.ckpt_seed)
    if not os.path.exists(ckpt_path):
        print(f"ERROR: checkpoint not found at {ckpt_path}")
        print("Please train a WikiText-2 d_osc=2 model first:")
        print("  python training/train_wikitext.py --d_osc 2 --seed 0")
        sys.exit(1)

    model = LoheLanguageTransformer(vocab_size=vocab_size, d_osc=2, **ARCH)
    state = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device)
    print(f"Loaded checkpoint: {ckpt_path}")

    print(f"\nExperiment 3: Sequential vs random init (N={args.N}, dt={args.dt})")
    print(f"  ODE seeds: {args.seeds}")

    results = {}

    # (a) Analytic
    print("\n(a) Analytic ...", flush=True)
    ppl_analytic = eval_ppl_with_mode(model, val_loader, device, "analytic")
    results["analytic"] = {"ppl": ppl_analytic}
    print(f"  PPL (analytic) = {ppl_analytic:.2f}")

    # (b) ODE random init
    print("\n(b) ODE random init ...", flush=True)
    ppls_random = []
    for s in args.seeds:
        ppl = eval_ppl_with_mode(model, val_loader, device, "ode_random",
                                  args.N, args.dt, rng_seed=s)
        ppls_random.append(ppl)
        print(f"  seed={s}: PPL = {ppl:.2f}", flush=True)
    results["ode_random"] = {
        "seeds": {str(s): p for s, p in zip(args.seeds, ppls_random)},
        "mean": statistics.mean(ppls_random),
        "std":  statistics.stdev(ppls_random) if len(ppls_random) > 1 else 0.0,
    }

    # (c) ODE sequential init
    print("\n(c) ODE sequential init ...", flush=True)
    ppls_seq = []
    for s in args.seeds:
        ppl = eval_ppl_with_mode(model, val_loader, device, "ode_seq",
                                  args.N, args.dt, rng_seed=s)
        ppls_seq.append(ppl)
        print(f"  seed={s}: PPL = {ppl:.2f}", flush=True)
    results["ode_seq"] = {
        "seeds": {str(s): p for s, p in zip(args.seeds, ppls_seq)},
        "mean": statistics.mean(ppls_seq),
        "std":  statistics.stdev(ppls_seq) if len(ppls_seq) > 1 else 0.0,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "sequential_init.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # Summary table
    print("\n## Sequential init results (WikiText-2, d_osc=2)\n")
    print("| Mode | PPL |")
    print("|---|---|")
    print(f"| Analytic | {results['analytic']['ppl']:.2f} |")
    print(f"| ODE random init | {results['ode_random']['mean']:.2f} ± "
          f"{results['ode_random']['std']:.2f} |")
    print(f"| ODE sequential init | {results['ode_seq']['mean']:.2f} ± "
          f"{results['ode_seq']['std']:.2f} |")


if __name__ == "__main__":
    main()
