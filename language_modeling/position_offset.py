"""
Appendix C ablation L3 (analytic): position-dependent anchor rotation in Lohe.

Uses LoheRelativeAnchorAttention — analytic fixed-point version of the
relative-anchor modification, implemented in the Lohe framework (same
methodology as the headline WikiText-2 results).

Architecture: d_model=128, n_heads=4, n_layers=2, d_ff=512, max_seq_len=50,
              d_osc=2 (rotation only defined in 2D).
Optimizer:    AdamW, lr=5e-4, wd=1e-4, cosine LR, 30 epochs — identical to
              headline runs.
Seeds:        0, 1, 2 (3 seeds for mean ± std).

Expected: small PPL change vs the d_osc=2 Lohe baseline (~110 PPL, mean).

Usage:
    python language_modeling/position_offset.py
    python language_modeling/position_offset.py --seeds 0 1 2
    python language_modeling/position_offset.py --smoke
"""

import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch.nn as nn

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from training import harness, paths  # noqa: E402

from oscillator_attention import LoheRelativeAnchorAttention
from oscillator_attention.transformer import SinusoidalPositionalEncoding
from training.data_utils import (
    load_wikitext2, make_lm_loaders, train_lm_epoch, eval_ppl
)

RESULTS_DIR = os.path.join(paths.runs_root(), "lm_position_offset")
CKPT_DIR    = os.path.join(paths.runs_root(), "lm", "checkpoints")

ARCH  = dict(d_model=128, n_heads=4, n_layers=2, d_ff=512,
             max_seq_len=50, dropout=0.1)
TRAIN = dict(lr=5e-4, weight_decay=1e-4, batch_size=64, grad_clip=1.0,
             n_epochs=30)


# ── Minimal transformer wrapper ───────────────────────────────────────────────

class _L3Layer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, max_seq_len):
        super().__init__()
        d_head = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = LoheRelativeAnchorAttention(
            d_model, n_heads, d_head, max_seq_len, p=1, causal=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout))

    def forward(self, x, padding_mask=None):
        out, _ = self.attn(self.norm1(x), padding_mask=padding_mask)
        x = x + out
        return x + self.ffn(self.norm2(x))


class L3LanguageTransformer(nn.Module):
    """Autoregressive LM with LoheRelativeAnchorAttention (Appendix C, L3)."""

    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2,
                 d_ff=512, max_seq_len=50, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc   = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)
        self.layers    = nn.ModuleList([
            _L3Layer(d_model, n_heads, d_ff, dropout, max_seq_len)
            for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, padding_mask=None):
        h = self.pos_enc(self.embedding(x))
        for layer in self.layers:
            h = layer(h, padding_mask=padding_mask)
        return self.head(self.norm(h))


# ── Training ──────────────────────────────────────────────────────────────────

def run_one(seed, vocab_size, train_loader, val_loader, device, n_epochs, smoke):
    torch.manual_seed(seed)
    label = f"wt2_L3_analytic_s{seed}"

    model = L3LanguageTransformer(vocab_size=vocab_size, **ARCH).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [{label}] params: {n_params:,}", flush=True)

    opt   = torch.optim.AdamW(model.parameters(), lr=TRAIN["lr"],
                               weight_decay=TRAIN["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_epochs * len(train_loader))

    best_ppl   = float("inf")
    best_state = None
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        avg_loss = train_lm_epoch(model, train_loader, opt, sched,
                                   TRAIN["grad_clip"], device)
        val_ppl  = eval_ppl(model, val_loader, device)
        if val_ppl < best_ppl:
            best_ppl   = val_ppl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elapsed = (time.time() - t0) / 60
        print(f"  [{label}] ep {epoch}/{n_epochs} | loss={avg_loss:.4f} | "
              f"ppl={val_ppl:.2f} | best={best_ppl:.2f} | {elapsed:.1f}m",
              flush=True)

        if not smoke and epoch % 10 == 0:
            os.makedirs(CKPT_DIR, exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(CKPT_DIR, f"{label}_ep{epoch}.pt"))

    if not smoke and best_state is not None:
        os.makedirs(CKPT_DIR, exist_ok=True)
        torch.save(best_state, os.path.join(CKPT_DIR, f"{label}_best.pt"))

    # Extract trained delta statistics
    deltas = []
    for layer in model.layers:
        deltas.append(layer.attn.delta.detach().cpu())
    delta_all = torch.cat(deltas)
    delta_stats = {
        "min": delta_all.min().item(),
        "max": delta_all.max().item(),
        "std": delta_all.std().item(),
        "mean": delta_all.mean().item(),
    }

    return best_ppl, delta_stats


def main():
    parser = argparse.ArgumentParser(description="L3 analytic ablation (WikiText-2)")
    parser.add_argument("--seeds",  type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", default="mps")
    parser.add_argument("--smoke",  action="store_true")
    args = parser.parse_args()

    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"\n{'='*60}")
    print(f"L3 Appendix C (analytic): relative anchor rotation (WikiText-2)")
    print(f"  device: {device}  seeds: {args.seeds}  smoke: {args.smoke}")
    print(f"{'='*60}\n")

    train_seqs, val_seqs, _, vocab = load_wikitext2(max_seq_len=50)
    vocab_size = len(vocab)
    print(f"Vocab: {vocab_size}, train: {len(train_seqs)}, val: {len(val_seqs)}",
          flush=True)

    if args.smoke:
        train_seqs = train_seqs[:200]
        val_seqs   = val_seqs[:100]

    train_loader, val_loader = make_lm_loaders(
        train_seqs, val_seqs, 50, batch_size=TRAIN["batch_size"])

    n_epochs = 2 if args.smoke else TRAIN["n_epochs"]

    ppls, delta_stats_list = [], []
    for seed in args.seeds:
        print(f"\n{'─'*60}", flush=True)
        ppl, dstats = run_one(seed, vocab_size, train_loader, val_loader,
                              device, n_epochs, args.smoke)
        ppls.append(ppl)
        delta_stats_list.append(dstats)
        print(f"\nSeed {seed}: best PPL = {ppl:.2f}", flush=True)
        print(f"  delta stats: {dstats}", flush=True)

    lohe_d2_mean = 110.67   # 5-seed mean from stored results
    lohe_d2_s0   = 110.22   # seed 0 from stored results
    our_mean = statistics.mean(ppls)
    our_std  = statistics.stdev(ppls) if len(ppls) > 1 else 0.0

    print(f"\n{'='*60}")
    print(f"L3 ANALYTIC RESULTS")
    print(f"{'='*60}")
    print(f"  PPL per seed: {[f'{p:.2f}' for p in ppls]}")
    print(f"  PPL mean ± std: {our_mean:.2f} ± {our_std:.2f}")
    print(f"  Lohe d_osc=2 baseline: {lohe_d2_mean:.2f} (5-seed mean)")
    print(f"  Delta vs baseline (mean): {our_mean - lohe_d2_mean:+.2f} PPL")
    print(f"  Delta vs baseline (seed 0): {ppls[0] - lohe_d2_s0:+.2f} PPL")

    results = {
        "model": "L3_analytic_LoheRelativeAnchorAttention",
        "seeds": args.seeds,
        "val_ppl_per_seed": {str(s): p for s, p in zip(args.seeds, ppls)},
        "val_ppl_mean": our_mean,
        "val_ppl_std": our_std,
        "lohe_d2_baseline_mean": lohe_d2_mean,
        "lohe_d2_baseline_seed0": lohe_d2_s0,
        "delta_vs_baseline_mean": our_mean - lohe_d2_mean,
        "delta_stats_per_seed": delta_stats_list,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "appendix_c_L3_5seeds.json")
    harness.guarded_dump(out, results)
    print(f"\nResults saved: {out}")


if __name__ == "__main__":
    main()
