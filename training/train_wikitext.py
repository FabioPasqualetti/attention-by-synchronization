"""
Reproduces Table 5 (WikiText-2 d_osc scaling law).

Conditions:
  wt2   : LoheAttention d_osc in {2, 8, 32}  (analytic fixed point)
  softmax: SoftmaxAttention baseline

Architecture: d_model=128, n_heads=4, n_layers=2, d_ff=512, max_seq_len=50.

Usage:
    python training/train_wikitext.py --d_osc 2 --seed 0
    python training/train_wikitext.py --d_osc 8 --seed 0
    python training/train_wikitext.py --attn_type softmax --seed 0
    python training/train_wikitext.py --seeds 0 1 2 3 4 --d_osc 2
"""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from training import paths  # noqa: E402
from oscillator_attention import LoheLanguageTransformer
from oscillator_attention.attention import SoftmaxAttention
from training.data_utils import (
    load_wikitext2, make_lm_loaders, train_lm_epoch, eval_ppl
)

RESULTS_DIR = os.path.join(paths.runs_root(), "lm")
CKPT_DIR    = os.path.join(paths.runs_root(), "lm", "checkpoints")

# Architecture (matches L6 experiments)
ARCH = dict(d_model=128, n_heads=4, n_layers=2, d_ff=512,
            max_seq_len=50, dropout=0.1)

# Training (matches L6 config)
TRAIN = dict(lr=5e-4, weight_decay=1e-4, batch_size=64, grad_clip=1.0,
             n_epochs=30)


def make_softmax_lm(vocab_size):
    """Builds a softmax-attention LM with the same architecture as Lohe."""

    class _SoftmaxLayer(nn.Module):
        def __init__(self, d_model, n_heads, d_ff, dropout):
            super().__init__()
            self.norm1 = nn.LayerNorm(d_model)
            self.attn  = SoftmaxAttention(d_model, n_heads)
            self.norm2 = nn.LayerNorm(d_model)
            self.ffn   = nn.Sequential(
                nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_ff, d_model), nn.Dropout(dropout))

        def forward(self, x, padding_mask=None):
            out, _ = self.attn(self.norm1(x), padding_mask=padding_mask, causal=True)
            x = x + out
            return x + self.ffn(self.norm2(x))

    class _SoftmaxLM(nn.Module):
        def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff,
                     max_seq_len, dropout):
            super().__init__()
            import math
            from oscillator_attention.transformer import SinusoidalPositionalEncoding
            self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
            self.pos_enc   = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)
            self.layers    = nn.ModuleList([
                _SoftmaxLayer(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)])
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size)

        def forward(self, x, padding_mask=None):
            h = self.pos_enc(self.embedding(x))
            for layer in self.layers:
                h = layer(h, padding_mask=padding_mask)
            return self.head(self.norm(h))

    return _SoftmaxLM(vocab_size, **ARCH)


def run_one(d_osc, attn_type, seed, device, smoke_test=False):
    torch.manual_seed(seed)
    label = f"wt2_{'softmax' if attn_type == 'softmax' else f'd{d_osc}'}_s{seed}"

    train_seqs, val_seqs, _, vocab = load_wikitext2(
        max_seq_len=ARCH["max_seq_len"])
    vocab_size = len(vocab)

    if smoke_test:
        train_seqs = train_seqs[:200]
        val_seqs   = val_seqs[:100]

    train_loader, val_loader = make_lm_loaders(
        train_seqs, val_seqs, ARCH["max_seq_len"],
        batch_size=TRAIN["batch_size"])

    if attn_type == "softmax":
        model = make_softmax_lm(vocab_size).to(device)
    else:
        model = LoheLanguageTransformer(vocab_size=vocab_size, d_osc=d_osc,
                                        **ARCH).to(device)

    n_epochs = 2 if smoke_test else TRAIN["n_epochs"]
    opt   = torch.optim.AdamW(model.parameters(), lr=TRAIN["lr"],
                               weight_decay=TRAIN["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_epochs * len(train_loader))

    best_ppl = float("inf")
    for epoch in range(1, n_epochs + 1):
        avg_loss = train_lm_epoch(model, train_loader, opt, sched,
                                   TRAIN["grad_clip"], device)
        val_ppl  = eval_ppl(model, val_loader, device)
        if val_ppl < best_ppl:
            best_ppl = val_ppl
        print(f"[{label}] epoch {epoch}/{n_epochs} | loss={avg_loss:.4f} | "
              f"val_ppl={val_ppl:.2f}", flush=True)

        if not smoke_test and epoch % 10 == 0:
            os.makedirs(CKPT_DIR, exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(CKPT_DIR, f"{label}_ep{epoch}.pt"))

    return {"label": label, "d_osc": d_osc, "attn_type": attn_type,
            "seed": seed, "val_ppl": best_ppl}


def main():
    parser = argparse.ArgumentParser(description="Train WikiText-2 LM")
    parser.add_argument("--d_osc",     type=int, default=8,
                        choices=[2, 8, 32])
    parser.add_argument("--attn_type", default="kuramoto",
                        choices=["kuramoto", "softmax"])
    parser.add_argument("--seed",      type=int, default=0)
    parser.add_argument("--seeds",     type=int, nargs="+",
                        help="Run multiple seeds sequentially")
    parser.add_argument("--device",    default="mps")
    parser.add_argument("--smoke",     action="store_true")
    args = parser.parse_args()

    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    seeds = args.seeds if args.seeds else [args.seed]
    results = []
    for s in seeds:
        r = run_one(args.d_osc, args.attn_type, s, device, args.smoke)
        results.append(r)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = "softmax" if args.attn_type == "softmax" else f"d{args.d_osc}"
    out_path = os.path.join(RESULTS_DIR, f"wikitext2_{tag}_seeds.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    ppls = [r["val_ppl"] for r in results]
    import statistics
    if len(ppls) > 1:
        print(f"PPL: {statistics.mean(ppls):.2f} ± {statistics.stdev(ppls):.2f} "
              f"(n={len(ppls)})")
    else:
        print(f"PPL: {ppls[0]:.2f}")


if __name__ == "__main__":
    main()
