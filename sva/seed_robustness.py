"""sva_seed_robustness — SVA seed expansion + sensitivity sweep.

Main: 30 seeds (0-29) each for softmax and oscillator at the min-hardware config
      (d_model=32, 1 head, 1 layer, d_ff=64, d_osc=2; AdamW lr=5e-4, wd=1e-4,
      20 epochs, batch 64, cosine per-batch). Metric: overall + hard accuracy.
      Failure := hard < 85%.

Sweep: lr in {2.5e-4, 5e-4, 1e-3} x epochs in {20, 40}, both mechanisms, 5 seeds each.
       The (lr=5e-4, ep=20) cell is covered by the first 5 seeds of the main run,
       so it is skipped here (10 additional cells).

Training loop mirrors training/train_sva.py::run_one (final-epoch eval),
parametrized by lr and n_epochs. Resumable per-run JSON in results/sva_seed_robustness/.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training import harness, paths  # noqa: E402

paths.ensure_paths()
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

sva_ds = paths.load_sva_dataset()  # training/sva_dataset.py, loaded under a unique name
make_sva_loaders = sva_ds.make_sva_loaders  # noqa: E402
from oscillator_attention import SVATransformer  # noqa: E402

EXP = "sva_seed_robustness"
ARCH = dict(d_model=32, n_heads=1, n_layers=1, d_ff=64, d_osc=2, max_seq_len=50)
WD = 1e-4
BATCH = 64
MAIN_SEEDS = list(range(50))  # pre-registered extension 30->50 seeds (marginal 30-seed Fisher p=0.08)
SWEEP_SEEDS = [0, 1, 2, 3, 4]
SWEEP_LRS = [2.5e-4, 5e-4, 1e-3]
SWEEP_EPOCHS = [20, 40]


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


def run_one(attn_type, seed, lr, n_epochs, device, save_ckpt=False, ckpt_path=None):
    torch.manual_seed(seed)
    train_loader, val_loader = make_sva_loaders(BATCH, n_train=-1, n_val=-1)
    model = SVATransformer(
        vocab_size=len(sva_ds.VOCAB), attn_type=attn_type, dropout=0.1,
        **ARCH).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_epochs * len(train_loader))
    crit = nn.BCEWithLogitsLoss()
    for _ in range(n_epochs):
        model.train()
        for batch in train_loader:
            x = batch["tokens"].to(device)
            labels = batch["labels"].to(device).float()
            pad_mask = batch["pad_mask"].to(device)
            verb_idx = batch["verb_idx"].to(device)
            opt.zero_grad()
            loss = crit(model(x, padding_mask=pad_mask, verb_idx=verb_idx), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
    accs = evaluate(model, val_loader, device)
    if save_ckpt and ckpt_path:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
    return {k: round(v * 100, 3) for k, v in accs.items()}


def _do(key, attn_type, seed, lr, n_epochs, device, **kw):
    if harness.exists(EXP, key):
        print(f"skip {key}", flush=True); return
    with harness.Timer() as t:
        acc = run_one(attn_type, seed, lr, n_epochs, device, **kw)
    harness.save_result(EXP, key, {
        "attn_type": attn_type, "seed": seed, "lr": lr, "n_epochs": n_epochs,
        "val_acc": acc, "failure": acc["hard"] < 85.0,
        "wall_sec": round(t.wall, 1), "arch": ARCH})
    harness.free_memory(device)  # release MPS cache between runs
    print(f"DONE {key}: all={acc['all']:.2f} hard={acc['hard']:.2f}  "
          f"({t.wall/60:.1f}m)", flush=True)


def main():
    device = harness.pick_device("mps")
    print(f"sva_seed_robustness device={device}", flush=True)
    ckpt_dir = os.path.join(harness.RUNS_ROOT, EXP, "ckpt")

    # Main 30-seed run
    for attn in ["softmax", "kuramoto"]:
        for seed in MAIN_SEEDS:
            key = f"main_{attn}_s{seed}"
            # keep a few oscillator ckpts for theory_degenerate_tokens (degenerate stats)
            save = (attn == "kuramoto" and seed < 3)
            _do(key, attn, seed, 5e-4, 20, device, save_ckpt=save,
                ckpt_path=os.path.join(ckpt_dir, f"{key}.pt"))

    # Sensitivity sweep (skip the 5e-4/20 cell — covered by main)
    for attn in ["softmax", "kuramoto"]:
        for lr in SWEEP_LRS:
            for ep in SWEEP_EPOCHS:
                if abs(lr - 5e-4) < 1e-12 and ep == 20:
                    continue
                for seed in SWEEP_SEEDS:
                    key = f"sweep_{attn}_lr{lr:g}_ep{ep}_s{seed}"
                    _do(key, attn, seed, lr, ep, device)
    print("sva_seed_robustness COMPLETE", flush=True)


if __name__ == "__main__":
    main()
