"""
Reproduces Table 2 (frozen W_V ablation on KWS).

Trains Kuramoto B1 (p=1) with W_v frozen at random initialization.
5 seeds.

Usage:
    python training/kws_frozen_wv_ablation.py --seeds 0 1 2 3 4
    python training/kws_frozen_wv_ablation.py --smoke
"""

import argparse
import json
import os
import statistics
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

from training import paths  # noqa: E402
from oscillator_attention import KWSTransformer

RESULTS_DIR = os.path.join(paths.runs_root(), "kws")
ARCH  = dict(n_feats=40, d_model=32, n_heads=2, n_layers=1, n_classes=10,
             T=49, d_osc=2, dropout=0.1, attn_type="lohe", p=1)
TRAIN = dict(lr=1e-3, weight_decay=1e-4, n_epochs=30, grad_clip=1.0)


def evaluate(model, val_loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for feats, labels in val_loader:
            if feats is None:
                continue
            feats, labels = feats.to(device), labels.to(device)
            preds = model(feats).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / max(total, 1)


def run_one(seed, device, smoke_test=False):
    torch.manual_seed(seed)
    label = f"kws_B1_frozenV_s{seed}"

    try:
        from dataset import get_loaders
    except ImportError:
        print("ERROR: KWS dataset not found. Install torchaudio."); return None

    train_loader, val_loader, _, _ = get_loaders(
        batch_size=64, smoke_test=smoke_test, seed=seed)

    model = KWSTransformer(**ARCH).to(device)

    # Freeze W_v
    for name, param in model.named_parameters():
        if "W_v" in name:
            param.requires_grad_(False)

    n_epochs = 2 if smoke_test else TRAIN["n_epochs"]
    opt   = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=TRAIN["lr"], weight_decay=TRAIN["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    best_acc = 0.0
    for epoch in range(1, n_epochs + 1):
        model.train()
        for feats, labels in train_loader:
            if feats is None:
                continue
            feats, labels = feats.to(device), labels.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(feats), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                TRAIN["grad_clip"])
            opt.step()
        sched.step()
        val_acc = evaluate(model, val_loader, device)
        if val_acc > best_acc:
            best_acc = val_acc
        print(f"[{label}] epoch {epoch}/{n_epochs} | val_acc={val_acc*100:.1f}%",
              flush=True)

    return {"label": label, "seed": seed, "val_acc": best_acc}


def main():
    parser = argparse.ArgumentParser(description="Frozen W_v ablation (KWS)")
    parser.add_argument("--seeds",  type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="mps")
    parser.add_argument("--smoke",  action="store_true")
    args = parser.parse_args()

    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    results = []
    for s in args.seeds:
        r = run_one(s, device, args.smoke)
        if r is not None:
            results.append(r)

    if results:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_path = os.path.join(RESULTS_DIR, "kws_frozen_wv.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        accs = [r["val_acc"] for r in results]
        std_str = f"{statistics.stdev(accs)*100:.1f}%" if len(accs) > 1 else "n/a"
        print(f"\nFrozen W_v: {statistics.mean(accs)*100:.1f}% ± {std_str}")
        print(f"Results saved: {out_path}")


if __name__ == "__main__":
    main()
