"""Frozen-W_V SVA ablation at 50 seeds, on the sva_seed_robustness cohort (Table 2).

Runs the frozen-W_V ablation on the sva_seed_robustness minimum-hardware harness, so the frozen-W_V
row of tab:sva is directly comparable to the main 50-seed softmax/oscillator rows (one uniform
table) rather than to an earlier, larger architecture.

Recipe (identical to sva_seed_robustness except for the freeze):
  - mechanism: kuramoto, d_osc=2
  - freeze: every parameter whose name contains 'W_v', at random init
  - optimizer: AdamW over trainable params only, lr=5e-4, wd=1e-4
  - schedule: CosineAnnealingLR(T_max=n_epochs*len(train_loader))
  - loss BCEWithLogitsLoss, grad_clip 1.0 (trainable params), batch 64,
    20 epochs, dropout 0.1, full train/val (n_train=-1, n_val=-1)
  - eval: final-epoch buckets all/simple/hard; failure := hard < 85%
  - ARCH: d_model=32, n_heads=1, n_layers=1, d_ff=64  (the minimum-hardware config)

Writes per-seed JSON to results/sva_frozen_value/. Resumable; run one seed at a time:
  python sva/frozen_value.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training import harness, paths  # noqa: E402

paths.ensure_paths()
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

sva_ds = paths.load_sva_dataset()
make_sva_loaders = sva_ds.make_sva_loaders
from oscillator_attention import SVATransformer  # noqa: E402

EXP = "sva_frozen_value"
ARCH = dict(d_model=32, n_heads=1, n_layers=1, d_ff=64, d_osc=2, max_seq_len=50)  # == sva_seed_robustness main
WD = 1e-4
BATCH = 64
LR = 5e-4
N_EPOCHS = 20
SEEDS = list(range(50))  # 0..49, same seed set as the main SVA cohort


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    buckets = {"all": [0, 0], "simple": [0, 0], "hard": [0, 0]}
    for batch in loader:
        x = batch["tokens"].to(device)
        labels = batch["labels"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        verb_idx = batch["verb_idx"].to(device)
        logits = model(x, padding_mask=pad_mask, verb_idx=verb_idx)
        preds = (logits > 0).long()
        correct = (preds == labels).cpu()
        for i in range(len(correct)):
            c = int(correct[i])
            buckets["all"][1] += 1; buckets["all"][0] += c
            if not batch["has_distractor"][i]:
                buckets["simple"][1] += 1; buckets["simple"][0] += c
            elif not batch["distractor_agrees"][i]:
                buckets["hard"][1] += 1; buckets["hard"][0] += c
    return {k: (v[0] / v[1] if v[1] > 0 else 0.0) for k, v in buckets.items()}


def run_one(seed, device):
    torch.manual_seed(seed)
    train_loader, val_loader = make_sva_loaders(BATCH, n_train=-1, n_val=-1)
    model = SVATransformer(
        vocab_size=len(sva_ds.VOCAB), attn_type="kuramoto", dropout=0.1,
        **ARCH).to(device)

    # Freeze W_v at random init
    frozen = []
    for name, param in model.named_parameters():
        if "W_v" in name:
            param.requires_grad_(False)
            frozen.append(name)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=N_EPOCHS * len(train_loader))
    crit = nn.BCEWithLogitsLoss()
    for _ in range(N_EPOCHS):
        model.train()
        for batch in train_loader:
            x = batch["tokens"].to(device)
            labels = batch["labels"].to(device).float()
            pad_mask = batch["pad_mask"].to(device)
            verb_idx = batch["verb_idx"].to(device)
            opt.zero_grad()
            loss = crit(model(x, padding_mask=pad_mask, verb_idx=verb_idx), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
    accs = evaluate(model, val_loader, device)
    return {k: round(v * 100, 3) for k, v in accs.items()}, frozen


def main():
    device = harness.pick_device("mps")
    print(f"{EXP} device={device}", flush=True)
    for seed in SEEDS:
        key = f"frozenwv_kuramoto_s{seed}"
        if harness.exists(EXP, key):
            print(f"skip {key}", flush=True); continue
        with harness.Timer() as t:
            acc, frozen = run_one(seed, device)
        harness.save_result(EXP, key, {
            "attn_type": "kuramoto", "condition": "frozen_Wv", "seed": seed,
            "lr": LR, "n_epochs": N_EPOCHS, "val_acc": acc,
            "failure": acc["hard"] < 85.0, "frozen_params": frozen,
            "wall_sec": round(t.wall, 1), "arch": ARCH})
        harness.free_memory(device)
        print(f"DONE {key}: all={acc['all']:.2f} hard={acc['hard']:.2f}  "
              f"({t.wall/60:.1f}m)", flush=True)
    print(f"{EXP} COMPLETE", flush=True)


if __name__ == "__main__":
    main()
