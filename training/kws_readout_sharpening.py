"""
Reproduces Table 6 (p-ablation on KWS).

Trains B1/B2/B3 (p=1/2/4) on KWS, 3 seeds each.
B0 (softmax baseline) also included for reference.

Usage:
    python training/kws_readout_sharpening.py
    python training/kws_readout_sharpening.py --p_values 1 2 4 --seeds 0 1 2
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
CKPT_DIR    = os.path.join(paths.runs_root(), "kws", "checkpoints_readout_sharpening")
ARCH_BASE = dict(n_feats=40, d_model=32, n_heads=2, n_layers=1, n_classes=10,
                 T=49, d_osc=2, dropout=0.1)
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


def run_one(p, seed, device, smoke_test=False, save_ckpt=True):
    torch.manual_seed(seed)
    try:
        from dataset import get_loaders
    except ImportError:
        return None

    train_loader, val_loader, _, _ = get_loaders(
        batch_size=64, smoke_test=smoke_test, seed=seed)

    model = KWSTransformer(**ARCH_BASE, attn_type="lohe", p=p).to(device)

    n_epochs = 2 if smoke_test else TRAIN["n_epochs"]
    opt   = torch.optim.Adam(model.parameters(), lr=TRAIN["lr"],
                              weight_decay=TRAIN["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    best_acc = 0.0
    best_state = None
    for epoch in range(1, n_epochs + 1):
        model.train()
        for feats, labels in train_loader:
            if feats is None:
                continue
            feats, labels = feats.to(device), labels.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(feats), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), TRAIN["grad_clip"])
            opt.step()
        sched.step()
        val_acc = evaluate(model, val_loader, device)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"  [p={p} seed={seed}] epoch {epoch}/{n_epochs} | "
              f"val_acc={val_acc*100:.1f}%", flush=True)

    if save_ckpt and not smoke_test and best_state is not None:
        os.makedirs(CKPT_DIR, exist_ok=True)
        ckpt_path = os.path.join(CKPT_DIR, f"KWS_p{p}_s{seed}_ep{n_epochs}.pt")
        torch.save(best_state, ckpt_path)
        print(f"  [p={p} seed={seed}] checkpoint → {ckpt_path}", flush=True)

    return best_acc


def main():
    parser = argparse.ArgumentParser(description="p-ablation on KWS")
    parser.add_argument("--p_values", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--seeds",    type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device",   default="mps")
    parser.add_argument("--smoke",    action="store_true")
    args = parser.parse_args()

    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    results = {}
    for p in args.p_values:
        seed_accs = {}
        for seed in args.seeds:
            acc = run_one(p, seed, device, args.smoke)
            if acc is not None:
                seed_accs[seed] = acc
        if seed_accs:
            accs = list(seed_accs.values())
            results[p] = {
                **{str(s): a for s, a in seed_accs.items()},
                "mean": statistics.mean(accs),
                "std":  statistics.stdev(accs) if len(accs) > 1 else 0.0,
            }
            print(f"p={p}: {results[p]['mean']*100:.1f}% ± "
                  f"{results[p]['std']*100:.1f}%")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "kws_readout_sharpening.json")
    # Merge with any existing results so that sequential single-seed runs
    # accumulate rather than overwrite each other.
    if os.path.exists(out_path):
        try:
            existing = json.load(open(out_path))
            for ex_p, ex_data in existing.items():
                key = int(ex_p) if str(ex_p).isdigit() else ex_p
                if key not in results:
                    results[key] = ex_data
                else:
                    for k, v in ex_data.items():
                        if str(k).isdigit() and str(k) not in results[key]:
                            results[key][str(k)] = v
                    all_v = [v for k, v in results[key].items() if str(k).isdigit()]
                    results[key]["mean"] = statistics.mean(all_v)
                    results[key]["std"]  = statistics.stdev(all_v) if len(all_v) > 1 else 0.0
        except (json.JSONDecodeError, KeyError, ValueError):
            pass  # corrupt file — start fresh
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    print("\n## KWS p-ablation\n")
    print("| p | Val Acc (mean ± std) |")
    print("|---|---|")
    for p in args.p_values:
        if p in results:
            print(f"| {p} | {results[p]['mean']*100:.1f}% ± "
                  f"{results[p]['std']*100:.1f}% |")


if __name__ == "__main__":
    main()
