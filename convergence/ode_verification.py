"""Proposition 4 check: how often does a random initialization land near the antipode?

This measures the *geometry* of the antipodal basin on WikiText-2 -- the empirical
probability that a uniform z(0) falls within alpha_0 of -x*, against the closed form of
Proposition 4. It integrates nothing, and it does not produce Table 6: that is
convergence/ts_convergence.py (TinyStories, RK45). Nothing printed in the paper comes
from this script; it is kept because it verifies a proposition the paper proves.

Loads trained L6a/b/c models (d_osc in {2, 8, 32}) and for each:
  1. Computes the analytic fixed points x* on validation data.
  2. Draws random initial conditions k(0) ~ Uniform(S^{d-1}).
  3. Reports empirical fraction of tokens with angle(k(0), -x*) < alpha_0.
  4. Compares against Proposition 4 closed-form prediction.

alpha_0 values: {pi/8, pi/6, pi/4}.  Canonical threshold: pi/6.

Usage:
    python convergence/ode_verification.py
    python convergence/ode_verification.py --n_batches 20
    python convergence/ode_verification.py --smoke
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from training import paths  # noqa: E402
from training import harness  # noqa: E402

from oscillator_attention import LoheLanguageTransformer
from training.data_utils import load_wikitext2, make_lm_loaders

RESULTS_DIR = os.path.join(paths.runs_root(), "convergence")

RESEARCH_CKPT = os.path.expanduser(
    os.path.join(paths.cache_root(), "checkpoints", "{label}_s0_ep30.pt")
)
D_OSC_LABELS = {2: "L6a", 8: "L6b", 32: "L6c"}

ARCH_BASE = dict(d_model=128, n_heads=4, n_layers=2, d_ff=512,
                 max_seq_len=50, dropout=0.0)

ALPHA_VALUES = [math.pi / 8, math.pi / 6, math.pi / 4]
N_RAND_STARTS = 100   # random k(0) per token


# ── Proposition 4 ────────────────────────────────────────────────────────────

def antipodal_theory(alpha: float, d: int) -> float:
    from scipy.integrate import quad
    from scipy.special import gamma
    prefactor = gamma(d / 2) / (math.sqrt(math.pi) * gamma((d - 1) / 2))
    val, _ = quad(lambda t: math.sin(t) ** (d - 2), 0, alpha)
    return prefactor * val


# ── Empirical antipodal rate from trained model ────────────────────────────────

@torch.no_grad()
def empirical_antipodal_rate(model, val_loader, device, alpha0, n_batches,
                              n_rand_starts, layer_idx=0, rng_seed=0):
    """
    Compute fraction of (batch, head, token, random_start) tuples
    where angle(k(0), -x*) < alpha0.

    x* is the analytic fixed point from the trained model.
    k(0) is drawn uniformly from S^{d_osc-1}.
    """
    model.eval()
    torch.manual_seed(rng_seed)

    layer    = model.layers[layer_idx]
    attn_mod = layer.attn
    H, D_osc = attn_mod.n_heads, attn_mod.d_osc

    total_count = 0
    antipodal_count = 0

    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= n_batches:
            break
        x        = batch["tokens"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        B, T     = x.shape

        h      = model.pos_enc(model.embedding(x))
        normed = layer.norm1(h)

        q = attn_mod.W_q(normed).view(B, T, H, attn_mod.d_head).transpose(1, 2)
        k = attn_mod.W_k(normed).view(B, T, H, attn_mod.d_head).transpose(1, 2)
        W = F.softplus(
            torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(attn_mod.d_head))
        W = W * torch.tril(torch.ones(T, T, device=device))
        if pad_mask is not None:
            W = W * (~pad_mask).float().view(B, 1, 1, T)

        anc     = attn_mod.anchors[:, :T, :]               # (H, T, D_osc)
        driving = torch.einsum("bhij,hjd->bhid", W, anc)   # (B, H, T, D_osc)
        x_star  = F.normalize(driving, dim=-1, eps=1e-8)   # (B, H, T, D_osc)

        # neg x_star: the antipodal fixed point
        neg_xstar = -x_star  # (B, H, T, D_osc)

        # Draw random k(0) on S^{D_osc-1}: (n_rand, D_osc)
        k0 = torch.randn(n_rand_starts, D_osc, device=device)
        k0 = F.normalize(k0, dim=-1, eps=1e-8)

        # Angle between k0 and -x*: cos(angle) = k0^T (-x*)
        # Expand: x_star (B, H, T, D_osc) + k0 (n_rand, D_osc)
        # → (B, H, T, n_rand) cos similarities
        cos_k0_neg = torch.einsum("bhtd,nd->bhtn", neg_xstar, k0)
        cos_k0_neg = cos_k0_neg.clamp(-1.0, 1.0)
        angles     = torch.acos(cos_k0_neg)   # (B, H, T, n_rand)

        # Exclude padding positions
        real_mask = ~pad_mask.view(B, 1, T, 1).expand_as(
            angles[:, :, :, :1]).squeeze(-1)   # (B, H, T)
        angles_flat = angles[real_mask.unsqueeze(-1).expand_as(angles)]

        antipodal_count += (angles_flat < alpha0).sum().item()
        total_count     += angles_flat.numel()

    return antipodal_count / max(total_count, 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ODE convergence verification with explicit alpha_0")
    parser.add_argument("--n_batches",    type=int, default=10,
                        help="Number of validation batches to use")
    parser.add_argument("--n_rand_starts", type=int, default=100)
    parser.add_argument("--device",       default="mps")
    parser.add_argument("--smoke",        action="store_true")
    args = parser.parse_args()

    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    n_batches = 2 if args.smoke else args.n_batches
    n_rand    = 10 if args.smoke else args.n_rand_starts

    _, val_seqs, _, vocab = load_wikitext2(max_seq_len=50)
    vocab_size = len(vocab)
    _, val_loader = make_lm_loaders([], val_seqs, 50, batch_size=32)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = {}

    for d_osc, ckpt_label in D_OSC_LABELS.items():
        ckpt_path = RESEARCH_CKPT.format(label=ckpt_label)
        if not os.path.exists(ckpt_path):
            print(f"  Checkpoint not found: {ckpt_path} — skipping d_osc={d_osc}")
            continue

        print(f"\n{'='*60}")
        print(f"d_osc={d_osc}  checkpoint={ckpt_path}")
        print(f"{'='*60}")

        model = LoheLanguageTransformer(vocab_size=vocab_size, d_osc=d_osc,
                                        **ARCH_BASE)
        state = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        model.load_state_dict(state)
        model = model.to(device)

        d_results = {}
        for alpha0 in ALPHA_VALUES:
            emp = empirical_antipodal_rate(
                model, val_loader, device, alpha0, n_batches, n_rand)
            theory = antipodal_theory(alpha0, d_osc)
            d_results[float(alpha0)] = {
                "empirical":   emp,
                "theoretical": theory,
                "delta":       abs(emp - theory),
            }
            print(f"  alpha0={alpha0:.4f} ({math.degrees(alpha0):.0f}°): "
                  f"emp={emp:.6f}  theory={theory:.6f}  "
                  f"delta={abs(emp-theory):.6f}")

        all_results[str(d_osc)] = d_results

    out_path = os.path.join(RESULTS_DIR, "ode_verification_v2.json")
    # Make JSON-serialisable
    json_out = {}
    for d, inner in all_results.items():
        json_out[d] = {str(a): v for a, v in inner.items()}
    harness.guarded_dump(out_path, json_out)
    print(f"\nResults saved: {out_path}")

    # Summary table
    print(f"\n{'d_osc':>6}  {'alpha0':>10}  {'empirical':>12}  "
          f"{'theoretical':>14}  {'delta':>10}")
    for d_osc in D_OSC_LABELS:
        if str(d_osc) not in all_results:
            continue
        for alpha0 in ALPHA_VALUES:
            r = all_results[str(d_osc)][float(alpha0)]
            print(f"  {d_osc:>4}  {alpha0:>10.4f}  {r['empirical']:>12.6f}  "
                  f"{r['theoretical']:>14.6f}  {r['delta']:>10.6f}")

    print("\nExperiment 5 complete.")


if __name__ == "__main__":
    main()
