"""
Reproduces Table 1 (KWS accuracy) and Table 6 (p ablation).

Conditions:
  B0: Softmax baseline
  B1: Kuramoto p=1
  B2: Kuramoto p=2
  B3: Kuramoto p=4

Architecture: d_model=32, n_heads=2, n_layers=1, T=49 (1-sec log-mel).
Training:     30 epochs, AdamW lr=1e-3, cosine LR, gradient clip.

Usage:
    python training/kws_train.py --condition B1 --seed 0
    python training/kws_train.py --seeds 0 1 2 3 4 --condition B1
    python training/kws_train.py --smoke --condition B1 --seed 0
"""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

# the KWS dataset module lives at training/kws_dataset.py
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

from training import paths  # noqa: E402
from oscillator_attention import KWSTransformer

RESULTS_DIR = os.path.join(paths.runs_root(), "kws")
CKPT_DIR    = os.path.join(paths.runs_root(), "kws", "checkpoints")

CONDITIONS = {
    "B0": dict(attn_type="softmax", p=1),
    "B1": dict(attn_type="lohe", p=1),
    "B2": dict(attn_type="lohe", p=2),
    "B3": dict(attn_type="lohe", p=4),
}

ARCH  = dict(n_feats=40, d_model=32, n_heads=2, n_layers=1, n_classes=10,
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


def run_one(condition, seed, device, smoke_test=False):
    torch.manual_seed(seed)
    cfg   = CONDITIONS[condition]
    label = f"kws_{condition}_s{seed}"

    try:
        from dataset import get_loaders
    except ImportError as e:
        print(f"ERROR: could not import dataset module: {e}")
        print("  Install torchaudio; Speech Commands downloads on first use.")
        return None

    train_loader, val_loader, _, _ = get_loaders(
        batch_size=64, smoke_test=smoke_test, seed=seed)

    model = KWSTransformer(**ARCH, **cfg).to(device)

    n_epochs = 2 if smoke_test else TRAIN["n_epochs"]
    opt   = torch.optim.Adam(model.parameters(), lr=TRAIN["lr"],
                              weight_decay=TRAIN["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    best_acc = 0.0
    for epoch in range(1, n_epochs + 1):
        model.train()
        for feats, labels in train_loader:
            if feats is None:
                continue
            feats, labels = feats.to(device), labels.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(feats), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), TRAIN["grad_clip"])
            opt.step()
        sched.step()
        val_acc = evaluate(model, val_loader, device)
        if val_acc > best_acc:
            best_acc = val_acc
        print(f"[{label}] epoch {epoch}/{n_epochs} | val_acc={val_acc*100:.1f}%",
              flush=True)
        if device.type == "mps":
            torch.mps.empty_cache()

    if not smoke_test:
        os.makedirs(CKPT_DIR, exist_ok=True)
        torch.save(model.state_dict(),
                   os.path.join(CKPT_DIR, f"{label}_final.pt"))

    return {"label": label, "condition": condition, "seed": seed,
            "val_acc": best_acc}


def main():
    parser = argparse.ArgumentParser(description="Train KWS transformer")
    parser.add_argument("--condition",  default="B1",
                        choices=list(CONDITIONS))
    parser.add_argument("--conditions", nargs="+", choices=list(CONDITIONS),
                        help="Run multiple conditions; overrides --condition")
    parser.add_argument("--seed",     type=int, default=0)
    parser.add_argument("--seeds",    type=int, nargs="+")
    parser.add_argument("--device",   default="mps")
    parser.add_argument("--smoke",    action="store_true")
    parser.add_argument("--out_path", default=None,
                        help="Override default output JSON path")
    args = parser.parse_args()

    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    seeds      = args.seeds if args.seeds else [args.seed]
    conditions = args.conditions if args.conditions else [args.condition]
    results = []
    for cond in conditions:
        for s in seeds:
            r = run_one(cond, s, device, args.smoke)
            if r is not None:
                results.append(r)

    if results:
        if args.out_path:
            out_path = args.out_path
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        else:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            cond_tag = "_".join(conditions)
            out_path = os.path.join(RESULTS_DIR, f"kws_{cond_tag}_seeds.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved: {out_path}")
        import statistics
        for cond in conditions:
            accs = [r["val_acc"] for r in results if r["condition"] == cond]
            if accs:
                if len(accs) > 1:
                    print(f"{cond}: {statistics.mean(accs)*100:.1f}% ± "
                          f"{statistics.stdev(accs)*100:.1f}%")
                else:
                    print(f"{cond}: {accs[0]*100:.1f}%")


if __name__ == "__main__":
    main()
