"""kws_position_matched — KWS position-matched baselines.

Full 2x3 grid (= the 6 conditions in the prompt):
  mechanism in {softmax, oscillator(lohe)}  x  pe in {none, sinusoidal, learned_abs}

The prompt enumerates a) softmax+learned, b) softmax+sin, c) osc+sin,
d) softmax+none & osc+none. The natural 6th cell completing the grid is
osc+learned; we run the full grid so every (mechanism, pe) pair is comparable.

All else identical to the KWS reference config (stock train.py ARCH/TRAIN):
  d_model=32, n_heads=2, n_layers=1, d_osc=2, T=49; Adam lr=1e-3, wd=1e-4,
  30 epochs, cosine LR (per-epoch), grad clip 1.0, batch 64. Metric: best val acc.

Training loop mirrors training/kws_train.py::run_one exactly, except the model is
KWSTransformerPE (which equals stock KWSTransformer when pe='none'; verified in selftest).

Resumable: writes results/kws_position_matched/<key>.json; skips existing.
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

EXP = "kws_position_matched"
ARCH = dict(n_feats=40, d_model=32, n_heads=2, n_layers=1, n_classes=10,
            T=49, d_osc=2, dropout=0.1)
TRAIN = dict(lr=1e-3, weight_decay=1e-4, n_epochs=30, grad_clip=1.0)
SEEDS = [0, 1, 2, 3, 4]
MECHS = {"softmax": "softmax", "osc": "lohe"}
PES = ["none", "sinusoidal", "learned_abs"]
CKPT = os.path.join(harness.RUNS_ROOT, EXP, "ckpt")


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


def run_one(mech, pe, seed, device, save_ckpt=True):
    from training.kws_data import cached_loaders
    torch.manual_seed(seed)
    train_loader, val_loader, _, _ = cached_loaders(seed, batch_size=64)

    model = KWSTransformerPE(attn_type=MECHS[mech], p=1, pe=pe, **ARCH).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=TRAIN["lr"],
                           weight_decay=TRAIN["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TRAIN["n_epochs"])

    best_acc = 0.0
    for epoch in range(1, TRAIN["n_epochs"] + 1):
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
        val_acc = _evaluate(model, val_loader, device)
        best_acc = max(best_acc, val_acc)
        if device.type == "mps":
            torch.mps.empty_cache()
    if save_ckpt:
        os.makedirs(CKPT, exist_ok=True)
        torch.save(model.state_dict(),
                   os.path.join(CKPT, f"{mech}_{pe}_s{seed}_final.pt"))
    return best_acc


def main():
    device = harness.pick_device("mps")
    print(f"kws_position_matched device={device}", flush=True)
    for mech in MECHS:
        for pe in PES:
            for seed in SEEDS:
                key = f"{mech}_{pe}_s{seed}"
                if harness.exists(EXP, key):
                    print(f"skip {key}", flush=True); continue
                with harness.Timer() as t:
                    # Save all checkpoints (kws_position_matched ckpts feed robustness_perturbations; all 6x5 feed kws_frame_shuffle).
                    acc = run_one(mech, pe, seed, device, save_ckpt=True)
                harness.save_result(EXP, key, {
                    "mech": mech, "pe": pe, "seed": seed,
                    "val_acc": acc, "wall_sec": round(t.wall, 1),
                    "arch": ARCH, "train": TRAIN})
                print(f"DONE {key}: {acc*100:.2f}%  ({t.wall/60:.1f}m)", flush=True)
    print("kws_position_matched COMPLETE", flush=True)


if __name__ == "__main__":
    main()
