"""
Appendix C ablation L5: attention-head and d_model scaling on WikiText-2.

Configurations run (seed 0):
  n_h=8, d_model=128  → softmax + oscillator (d_h=16, half-wide heads)
  n_h=8, d_model=256  → softmax + oscillator (d_h=32, proportional scaling)

Baseline (n_h=4, d_model=128), both from results/wikitext2_d2_5seeds.json:
  softmax:       99.58 PPL  (its `softmax_ppl` field)
  oscillator d2: 110.22 PPL (its `seeds` field, seed 0)

Expected: n_h alone hurts both; n_h+d_model together improves both;
gap is unchanged in both comparisons.

Usage:
    python language_modeling/head_scaling.py
    python language_modeling/head_scaling.py --smoke
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from training import harness, paths  # noqa: E402

from oscillator_attention import LoheLanguageTransformer, SoftmaxAttention
from oscillator_attention.transformer import SinusoidalPositionalEncoding
from training.data_utils import (
    load_wikitext2, make_lm_loaders, train_lm_epoch, eval_ppl
)

RESULTS_DIR = os.path.join(paths.runs_root(), "lm_head_scaling")
CKPT_DIR    = os.path.join(paths.runs_root(), "lm", "checkpoints")

# Shared training settings (identical to headline runs)
TRAIN = dict(lr=5e-4, weight_decay=1e-4, batch_size=64, grad_clip=1.0,
             n_epochs=30)


# ── Softmax LM builder ────────────────────────────────────────────────────────

def make_softmax_lm(vocab_size, d_model, n_heads, n_layers, d_ff,
                    max_seq_len, dropout):

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(d_model)
            self.attn  = SoftmaxAttention(d_model, n_heads, max_seq_len)
            self.norm2 = nn.LayerNorm(d_model)
            self.ffn   = nn.Sequential(
                nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_ff, d_model), nn.Dropout(dropout))

        def forward(self, x, padding_mask=None):
            out, _ = self.attn(self.norm1(x), padding_mask=padding_mask,
                               causal=True)
            x = x + out
            return x + self.ffn(self.norm2(x))

    class _LM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
            self.pos_enc   = SinusoidalPositionalEncoding(d_model, max_seq_len,
                                                          dropout)
            self.layers    = nn.ModuleList([_Layer() for _ in range(n_layers)])
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size)

        def forward(self, x, padding_mask=None):
            h = self.pos_enc(self.embedding(x))
            for layer in self.layers:
                h = layer(h, padding_mask=padding_mask)
            return self.head(self.norm(h))

    return _LM()


# ── Single run ────────────────────────────────────────────────────────────────

def run_one(label, model, train_loader, val_loader, device, n_epochs, smoke):
    model = model.to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=TRAIN["lr"],
                               weight_decay=TRAIN["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_epochs * len(train_loader))

    best_ppl = float("inf")
    t0 = time.time()
    for epoch in range(1, n_epochs + 1):
        avg_loss = train_lm_epoch(model, train_loader, opt, sched,
                                   TRAIN["grad_clip"], device)
        val_ppl  = eval_ppl(model, val_loader, device)
        if val_ppl < best_ppl:
            best_ppl = val_ppl
        elapsed = (time.time() - t0) / 60
        print(f"[{label}] epoch {epoch}/{n_epochs} | loss={avg_loss:.4f} | "
              f"val_ppl={val_ppl:.2f} | elapsed={elapsed:.1f}m", flush=True)

        if not smoke and epoch % 10 == 0:
            os.makedirs(CKPT_DIR, exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(CKPT_DIR, f"{label}_ep{epoch}.pt"))

    return best_ppl


# ── Configurations ────────────────────────────────────────────────────────────

CONFIGS = [
    # (tag, n_heads, d_model, d_ff)
    ("n8_d128", 8, 128, 512),   # n_h=8, d_model=128  (d_h=16)
    ("n8_d256", 8, 256, 1024),  # n_h=8, d_model=256  (d_h=32, proportional)
]

BASE = dict(n_layers=2, max_seq_len=50, dropout=0.1)


def main():
    parser = argparse.ArgumentParser(description="L5 head/model scaling ablation")
    parser.add_argument("--seed",   type=int, default=0)
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
    print(f"L5 Appendix C ablation: head/model scaling (WikiText-2)")
    print(f"  device: {device}  seed: {args.seed}")
    print(f"  smoke: {args.smoke}")
    print(f"{'='*60}\n")

    train_seqs, val_seqs, _, vocab = load_wikitext2(max_seq_len=50)
    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size}, train seqs: {len(train_seqs)}, "
          f"val seqs: {len(val_seqs)}", flush=True)

    if args.smoke:
        train_seqs = train_seqs[:200]
        val_seqs   = val_seqs[:100]

    n_epochs = 2 if args.smoke else TRAIN["n_epochs"]

    results = {}
    # Also record the known baseline
    results["n4_d128_softmax"] = {"val_ppl": 99.58, "note": "from_stored_results"}
    results["n4_d128_osc_d2"]  = {"val_ppl": 110.22, "note": "from_stored_results_seed0"}

    for tag, n_heads, d_model, d_ff in CONFIGS:
        train_loader, val_loader = make_lm_loaders(
            train_seqs, val_seqs, 50, batch_size=TRAIN["batch_size"])

        for attn_type in ("softmax", "osc"):
            run_tag = f"{tag}_{attn_type}"
            torch.manual_seed(args.seed)
            print(f"\n{'─'*60}", flush=True)
            print(f"Running: {run_tag}", flush=True)

            if attn_type == "softmax":
                model = make_softmax_lm(
                    vocab_size, d_model, n_heads, BASE["n_layers"], d_ff,
                    BASE["max_seq_len"], BASE["dropout"])
            else:
                model = LoheLanguageTransformer(
                    vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
                    n_layers=BASE["n_layers"], d_ff=d_ff,
                    max_seq_len=BASE["max_seq_len"], dropout=BASE["dropout"],
                    d_osc=2)

            ppl = run_one(run_tag, model, train_loader, val_loader, device,
                          n_epochs, args.smoke)
            results[run_tag] = {"val_ppl": ppl, "n_heads": n_heads,
                                "d_model": d_model, "d_ff": d_ff,
                                "attn_type": attn_type, "seed": args.seed}
            print(f"\nBest PPL [{run_tag}]: {ppl:.2f}", flush=True)

    # Summary table
    print(f"\n{'='*60}")
    print("L5 SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<20} {'PPL':>8}  {'Note'}")
    print(f"{'─'*50}")
    for k, v in results.items():
        note = v.get("note", "")
        print(f"{k:<20} {v['val_ppl']:>8.2f}  {note}")

    # Gap analysis
    for tag, n_heads, d_model, d_ff in CONFIGS:
        sm_key  = f"{tag}_softmax"
        osc_key = f"{tag}_osc"
        if sm_key in results and osc_key in results:
            gap = results[osc_key]["val_ppl"] - results[sm_key]["val_ppl"]
            print(f"\n  Gap at {tag}: {gap:.2f} PPL  "
                  f"(softmax={results[sm_key]['val_ppl']:.2f}, "
                  f"osc={results[osc_key]['val_ppl']:.2f})")
    baseline_gap = 110.22 - 99.58
    print(f"  Baseline gap (n4_d128): {baseline_gap:.2f} PPL")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "appendix_c_L5_5seeds.json")
    harness.guarded_dump(out, results)
    print(f"\nResults saved: {out}")


if __name__ == "__main__":
    main()
