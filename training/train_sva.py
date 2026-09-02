"""
Canonical SVA training script — reproduces the paper's Table 3 (SVA accuracy).

Trains the minimum-hardware configuration: d_model=32, 1 attention head,
1 transformer layer, d_ff=64, d_osc=2 (14 total oscillators per sentence).

This script reproduces the results in:
  - Table 3: SVA accuracy at the minimum-hardware configuration
  - Table 2b: SVA frozen-WV ablation (via --freeze_wv flag)

Architecture matches the minimum-hardware configuration exactly. Canonical checkpoints
are saved to sva/checkpoints/ and match those committed
in the repository.

Usage:
    # Train one seed (softmax and oscillator):
    python training/train_sva.py --seed 0

    # Train paper-canonical 5 seeds:
    python training/train_sva.py --seeds 0 1 2 3 4

    # Train frozen-WV ablation (paper Table 2b):
    python training/train_sva.py --seeds 0 1 2 --freeze_wv --attn_type kuramoto

    # Smoke test (2 epochs, 200 sentences):
    python training/train_sva.py --seed 0 --smoke

Expected results (5 seeds each):
    Softmax:    92.2 ± 7.8% overall,  92.1 ± 7.9% hard
    Oscillator: 97.8 ± 0.2% overall,  97.4 ± 0.3% hard
The paper's SVA numbers are the 50-seed cohort in results/sva_seed_robustness/.
"""

import argparse
import json
import math
import os
import statistics
import sys

import torch
import torch.nn as nn

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

from oscillator_attention import SVATransformer
from training import sva_dataset as sva_ds
from training.sva_dataset import make_sva_loaders

# ── Canonical minimum-hardware architecture (d_model=32, 1h, 1L, d_ff=64) ───
ARCH = dict(d_model=32, n_heads=1, n_layers=1, d_ff=64, d_osc=2, max_seq_len=50)
TRAIN = dict(lr=5e-4, weight_decay=1e-4, n_epochs=20, grad_clip=1.0, batch_size=64)
CKPT_DIR = os.path.join(ROOT, "sva", "checkpoints")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    buckets = {"all": [0, 0], "simple": [0, 0], "hard": [0, 0]}
    for batch in loader:
        x        = batch["tokens"].to(device)
        labels   = batch["labels"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        verb_idx = batch["verb_idx"].to(device)
        logits   = model(x, padding_mask=pad_mask, verb_idx=verb_idx)
        preds    = (logits > 0).long()
        correct  = (preds == labels).cpu()
        for i in range(len(correct)):
            c = int(correct[i])
            buckets["all"][1] += 1; buckets["all"][0] += c
            if not batch["has_distractor"][i]:
                buckets["simple"][1] += 1; buckets["simple"][0] += c
            elif not batch["distractor_agrees"][i]:
                buckets["hard"][1] += 1; buckets["hard"][0] += c
    return {k: v[0] / v[1] if v[1] > 0 else 0.0 for k, v in buckets.items()}


def run_one(attn_type, seed, device, freeze_wv=False, smoke_test=False):
    torch.manual_seed(seed)
    label = f"sva_{'frozenV' if freeze_wv else attn_type}_s{seed}"
    n_epochs = 2 if smoke_test else TRAIN["n_epochs"]

    train_loader, val_loader = make_sva_loaders(
        TRAIN["batch_size"],
        n_train=200 if smoke_test else -1,
        n_val=50  if smoke_test else -1,
    )

    model = SVATransformer(
        vocab_size=len(sva_ds.VOCAB),
        attn_type=attn_type,
        d_osc=ARCH["d_osc"],
        d_model=ARCH["d_model"],
        n_heads=ARCH["n_heads"],
        n_layers=ARCH["n_layers"],
        d_ff=ARCH["d_ff"],
        max_seq_len=ARCH["max_seq_len"],
        dropout=0.1,
    ).to(device)

    if freeze_wv:
        for name, param in model.named_parameters():
            if "W_v" in name:
                param.requires_grad_(False)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt   = torch.optim.AdamW(trainable, lr=TRAIN["lr"],
                               weight_decay=TRAIN["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_epochs * len(train_loader))
    crit  = nn.BCEWithLogitsLoss()

    for epoch in range(1, n_epochs + 1):
        model.train()
        for batch in train_loader:
            x = batch["tokens"].to(device)
            labels = batch["labels"].to(device).float()
            pad_mask = batch["pad_mask"].to(device)
            verb_idx = batch["verb_idx"].to(device)
            opt.zero_grad()
            loss = crit(model(x, padding_mask=pad_mask, verb_idx=verb_idx), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, TRAIN["grad_clip"])
            opt.step()
            sched.step()
        accs = evaluate(model, val_loader, device)
        print(f"[{label}] ep {epoch}/{n_epochs} | "
              f"all={accs['all']*100:.1f}%  hard={accs['hard']*100:.1f}%",
              flush=True)

    final_accs = evaluate(model, val_loader, device)

    if not smoke_test:
        os.makedirs(CKPT_DIR, exist_ok=True)
        phase_tag = "frozenV" if freeze_wv else ("osc" if attn_type == "kuramoto" else "softmax")
        ckpt_path = os.path.join(CKPT_DIR, f"{phase_tag}_s{seed}_ep{n_epochs}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"[{label}] checkpoint → {ckpt_path}")

    return {
        "attn_type":  attn_type,
        "freeze_wv":  freeze_wv,
        "seed":       seed,
        "val_acc":    {k: round(v * 100, 2) for k, v in final_accs.items()},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train SVA at minimum-hardware configuration (paper canonical)")
    parser.add_argument("--attn_type", choices=["softmax", "kuramoto"],
                        default=None,
                        help="Attention type. Default: train both.")
    parser.add_argument("--seed",  type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--freeze_wv", action="store_true",
                        help="Freeze W_V for ablation (Table 2b)")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--smoke",  action="store_true")
    args = parser.parse_args()

    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    seeds = args.seeds if args.seeds is not None else (
        [args.seed] if args.seed is not None else [0])

    attn_types = ([args.attn_type] if args.attn_type is not None
                  else ["softmax", "kuramoto"])

    all_results = []
    for attn_type in attn_types:
        if args.freeze_wv and attn_type == "softmax":
            continue  # frozen-WV ablation only makes sense for oscillator
        for seed in seeds:
            r = run_one(attn_type, seed, device,
                        freeze_wv=args.freeze_wv,
                        smoke_test=args.smoke)
            all_results.append(r)
            print(f"  → all={r['val_acc']['all']}%  hard={r['val_acc']['hard']}%")

    if len(all_results) > 1:
        for attn_type in set(r["attn_type"] for r in all_results):
            subset = [r for r in all_results if r["attn_type"] == attn_type]
            hards = [r["val_acc"]["hard"] for r in subset]
            alls  = [r["val_acc"]["all"]  for r in subset]
            print(f"\n{attn_type}: hard={statistics.mean(hards):.1f}±{statistics.stdev(hards) if len(hards)>1 else 0:.1f}%"
                  f"  all={statistics.mean(alls):.1f}±{statistics.stdev(alls) if len(alls)>1 else 0:.1f}%")


if __name__ == "__main__":
    main()
