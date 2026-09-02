"""Frozen-W_V KWS ablation, 5 seeds, on the kws_position_matched no-PE cohort (Table 1).

Runs the frozen-W_V ablation on the kws_position_matched harness with pe='none', so the frozen-W_V
row of tab:kws sits in the same cohort as the position-matched no-PE numbers (osc 88.79,
softmax 87.71) rather than a separate one.

Recipe (identical to kws_position_matched except for the freeze):
  - mechanism: attn_type='lohe' (analytic Lohe fixed point), p=1, d_osc=2
  - freeze: every parameter whose name contains 'W_v', at random init
  - optimizer: Adam over trainable params only, lr=1e-3, wd=1e-4
  - schedule: CosineAnnealingLR(T_max=n_epochs=30), per-epoch
  - loss F.cross_entropy, grad_clip 1.0 (trainable params), batch 64,
    30 epochs, dropout 0.1, metric = best val acc over epochs
  - ARCH: n_feats=40, d_model=32, n_heads=2, n_layers=1, n_classes=10, T=49, d_osc=2
  - data: training.kws_data.cached_loaders(seed, batch_size=64)
  - model: kws_position_matched's KWSTransformerPE(pe='none'), which equals the stock
    KWSTransformer at pe='none' (verified in training/selftest.py)

Writes per-seed JSON to results/kws_frozen_value/. Resumable; run one seed at a time:
  python keyword_spotting/frozen_value.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training import harness, paths  # noqa: E402

paths.ensure_paths()
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from oscillator_attention.kws_pe_model import KWSTransformerPE  # noqa: E402
from training.kws_data import cached_loaders  # noqa: E402

EXP = "kws_frozen_value"
ARCH = dict(n_feats=40, d_model=32, n_heads=2, n_layers=1, n_classes=10,
            T=49, d_osc=2, dropout=0.1)  # == kws_position_matched
TRAIN = dict(lr=1e-3, weight_decay=1e-4, n_epochs=30, grad_clip=1.0)  # == kws_position_matched & code/
SEEDS = [0, 1, 2, 3, 4]


def _evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for feats, labels in loader:
            if feats is None:
                continue
            feats, labels = feats.to(device), labels.to(device)
            preds = model(feats).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)


def run_one(seed, device):
    torch.manual_seed(seed)
    train_loader, val_loader, _, _ = cached_loaders(seed, batch_size=64)

    model = KWSTransformerPE(attn_type="lohe", p=1, pe="none", **ARCH).to(device)

    # Freeze W_v at random init
    frozen = []
    for name, param in model.named_parameters():
        if "W_v" in name:
            param.requires_grad_(False)
            frozen.append(name)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=TRAIN["lr"], weight_decay=TRAIN["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TRAIN["n_epochs"])

    best_acc = 0.0
    for _ in range(TRAIN["n_epochs"]):
        model.train()
        for feats, labels in train_loader:
            if feats is None:
                continue
            feats, labels = feats.to(device), labels.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(feats), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, TRAIN["grad_clip"])
            opt.step()
        sched.step()
        best_acc = max(best_acc, _evaluate(model, val_loader, device))
        if device.type == "mps":
            torch.mps.empty_cache()
    return best_acc, frozen


def main():
    device = harness.pick_device("mps")
    print(f"{EXP} device={device}", flush=True)
    for seed in SEEDS:
        key = f"frozenwv_lohe_noPE_s{seed}"
        if harness.exists(EXP, key):
            print(f"skip {key}", flush=True); continue
        with harness.Timer() as t:
            acc, frozen = run_one(seed, device)
        harness.save_result(EXP, key, {
            "mech": "osc", "condition": "frozen_Wv", "pe": "none", "seed": seed,
            "val_acc": acc, "frozen_params": frozen,
            "wall_sec": round(t.wall, 1), "arch": ARCH, "train": TRAIN})
        print(f"DONE {key}: {acc*100:.2f}%  ({t.wall/60:.1f}m)", flush=True)
    print(f"{EXP} COMPLETE", flush=True)


if __name__ == "__main__":
    main()
