"""
SVA architecture sweep — find a configuration where attention is necessary.

Trains softmax (Phase 1) and optionally oscillator (Phase 2) SVA models
across 6 architecture configs. For each config, 3 seeds. Same training
procedure as the main SVA experiment.

Configs:
  A: d_model=64, n_heads=2, n_layers=2, d_ff=256  (paper baseline)
  B: d_model=64, n_heads=1, n_layers=2, d_ff=256
  C: d_model=64, n_heads=2, n_layers=1, d_ff=256
  D: d_model=64, n_heads=1, n_layers=1, d_ff=256
  E: d_model=32, n_heads=1, n_layers=1, d_ff=128
  F: d_model=16, n_heads=1, n_layers=1, d_ff=64

Usage:
    # Phase 1 (softmax sweep):
    python sva/architecture_sweep.py --phase 1 --configs A B C D E F --seeds 0 1 2

    # Phase 2 (oscillator at sweet-spot, e.g. config E):
    python sva/architecture_sweep.py --phase 2 --configs E --seeds 0 1 2

    # Phase 3 (frozen W_V at sweet-spot):
    python sva/architecture_sweep.py --phase 3 --configs E --seeds 0 1 2

    # Single config smoke test:
    python sva/architecture_sweep.py --phase 1 --configs F --seeds 0 --smoke
"""

import argparse
import json
import math
import os
import statistics
import sys
import time

import torch
import torch.nn as nn

# — three levels below the repo root, so three ".." hops are needed.
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ROOT = os.path.normpath(ROOT)
sys.path.insert(0, ROOT)
from training import harness, paths  # noqa: E402
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

from oscillator_attention import SVATransformer

RESULTS_DIR = os.path.join(paths.runs_root(), "sva")
CKPT_BASE   = os.path.join(paths.runs_root(), "sva", "checkpoints_arch_sweep")
LOG_PATH    = os.path.join(paths.runs_root(), "sva", "arch_sweep_log.md")
SWEEP_JSON  = os.path.join(RESULTS_DIR, "sva_arch_sweep_results.json")
HEARTBEAT   = os.path.join(paths.runs_root(), "sva", "architecture_sweep.log")

CONFIGS = {
    # Phase 1 configs (d_ff = 4×d_model)
    "A": dict(d_model=64, n_heads=2, n_layers=2),
    "B": dict(d_model=64, n_heads=1, n_layers=2),
    "C": dict(d_model=64, n_heads=2, n_layers=1),
    "D": dict(d_model=64, n_heads=1, n_layers=1),
    "E": dict(d_model=32, n_heads=1, n_layers=1),
    "F": dict(d_model=16, n_heads=1, n_layers=1),
    # Phase 4 configs (d_ff bottleneck sweep, h=1, L=1)
    "G": dict(d_model=32, n_heads=1, n_layers=1, d_ff_override=64),
    "H": dict(d_model=32, n_heads=1, n_layers=1, d_ff_override=32),
    "I": dict(d_model=32, n_heads=1, n_layers=1, d_ff_override=16),
    "J": dict(d_model=64, n_heads=1, n_layers=1, d_ff_override=128),
    "K": dict(d_model=64, n_heads=1, n_layers=1, d_ff_override=64),
    "L": dict(d_model=64, n_heads=1, n_layers=1, d_ff_override=32),
    # Check 3: +-15% ff perturbations around Config G (ff=64).
    # NOTE: the suffixes run OPPOSITE to the widths -- G_minus is the WIDER
    # network (ff=72) and G_plus the narrower (ff=56); they name the direction
    # of the capacity constraint, not of d_ff. Table 8 is keyed on d_ff, and
    # every stored run carries arch.d_ff, so select on that rather than on the
    # config name: selecting by name alone swaps two rows of the table.
    "G_minus": dict(d_model=32, n_heads=1, n_layers=1, d_ff_override=72),
    "G_plus":  dict(d_model=32, n_heads=1, n_layers=1, d_ff_override=56),
}

TRAIN = dict(lr=5e-4, weight_decay=1e-4, n_epochs=20, grad_clip=1.0, batch_size=64)
T_SVA = 9

_last_heartbeat = 0.0


def _heartbeat(msg):
    global _last_heartbeat
    now = time.time()
    if now - _last_heartbeat >= 1800:
        ts = __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M")
        with open(HEARTBEAT, "a") as f:
            f.write(f"[{ts}] {msg}\n")
        _last_heartbeat = now


def _anchor_spread_d2(model):
    spreads = {}
    for li, layer in enumerate(model.layers):
        attn = getattr(layer, "attn", None)
        if attn is None or not hasattr(attn, "anchors"):
            continue
        anc = attn.anchors.detach().cpu()
        for h in range(anc.shape[0]):
            angles = torch.atan2(anc[h, :T_SVA, 1], anc[h, :T_SVA, 0]) * 180.0 / math.pi
            spreads[f"L{li}H{h}"] = float(angles.max() - angles.min())
    return spreads


@torch.no_grad()
def evaluate_sva(model, loader, device):
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
    return {k: (v[0] / v[1] if v[1] > 0 else 0.0) for k, v in buckets.items()}


def run_one(cfg_letter, attn_type, seed, device, smoke_test=False,
            freeze_wv=False, d_osc=2, train_cfg=None, ckpt_suffix=""):
    cfg = CONFIGS[cfg_letter]
    d_model  = cfg["d_model"]
    n_heads  = cfg["n_heads"]
    n_layers = cfg["n_layers"]
    d_ff     = cfg.get("d_ff_override", d_model * 4)

    tc = train_cfg or TRAIN   # allow per-call override
    phase_tag = "softmax" if attn_type == "softmax" else ("frozenV" if freeze_wv else "osc")
    label = f"cfg{cfg_letter}_{phase_tag}{ckpt_suffix}_s{seed}"
    torch.manual_seed(seed)

    try:
        import dataset as sva_ds
        from dataset import make_sva_loaders
    except ImportError as e:
        print(f"ERROR: {e}"); return None

    n = 200 if smoke_test else -1
    train_loader, val_loader = make_sva_loaders(
        tc["batch_size"], n_train=n, n_val=50 if smoke_test else -1)

    vocab_size = len(sva_ds.VOCAB)
    model = SVATransformer(
        vocab_size=vocab_size, attn_type=attn_type, d_osc=d_osc,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        d_ff=d_ff, max_seq_len=50, dropout=0.1,
    ).to(device)

    def _weight_norms(m):
        """Frobenius norms for key weight matrices."""
        norms = {}
        for name, param in m.named_parameters():
            if any(k in name for k in ["W_q","W_k","W_v","W_o","ffn.0","ffn.3"]):
                norms[name] = float(param.detach().norm("fro"))
        return norms

    init_norms = _weight_norms(model)

    if freeze_wv:
        for name, param in model.named_parameters():
            if "W_v" in name:
                param.requires_grad_(False)

    spreads_init = _anchor_spread_d2(model) if attn_type != "softmax" else {}

    n_epochs = 2 if smoke_test else tc["n_epochs"]
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt   = torch.optim.AdamW(trainable, lr=tc["lr"], weight_decay=tc["weight_decay"])
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
            torch.nn.utils.clip_grad_norm_(trainable, tc["grad_clip"])
            opt.step()
            sched.step()

        accs = evaluate_sva(model, val_loader, device)
        sp   = _anchor_spread_d2(model)
        sp_str = ("  ".join(f"{k}={v:.0f}°" for k, v in sp.items())
                  if sp else "n/a")
        print(f"[{label}] ep {epoch}/{n_epochs} | "
              f"all={accs['all']*100:.1f}% hard={accs['hard']*100:.1f}% | "
              f"spread: {sp_str}", flush=True)
        _heartbeat(f"arch_sweep cfg={cfg_letter} {phase_tag} s{seed} "
                   f"ep{epoch}/{n_epochs} all={accs['all']*100:.1f}% "
                   f"hard={accs['hard']*100:.1f}%")

    final_accs  = evaluate_sva(model, val_loader, device)
    spreads_fin = _anchor_spread_d2(model) if attn_type != "softmax" else {}
    final_norms = _weight_norms(model)
    norm_ratios = {k: final_norms[k] / init_norms[k]
                   for k in init_norms if k in final_norms}

    if not smoke_test:
        ckpt_dir = os.path.join(CKPT_BASE, cfg_letter)
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir,
                                 f"{phase_tag}{ckpt_suffix}_s{seed}_ep{n_epochs}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"[{label}] checkpoint → {ckpt_path}", flush=True)

    return {
        "config":        cfg_letter,
        "attn_type":     attn_type,
        "freeze_wv":     freeze_wv,
        "seed":          seed,
        "val_acc":       final_accs,
        "anchor_spread_init":  spreads_init,
        "anchor_spread_final": spreads_fin,
        "weight_norm_ratios":  norm_ratios,
        "arch":          {"d_model": d_model, "n_heads": n_heads,
                          "n_layers": n_layers, "d_ff": d_ff},
    }


def _append_log(text):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(text + "\n")


def _load_sweep_results():
    if os.path.exists(SWEEP_JSON):
        return json.load(open(SWEEP_JSON))
    return {}


def _save_sweep_results(data):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    harness.guarded_dump(SWEEP_JSON, data)


def main():
    parser = argparse.ArgumentParser(description="SVA architecture sweep")
    parser.add_argument("--phase",        type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--configs",      nargs="+", default=list(CONFIGS.keys()))
    parser.add_argument("--seeds",        type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--d_osc",        type=int, default=2)
    parser.add_argument("--device",       default="mps")
    parser.add_argument("--smoke",        action="store_true")
    # Hyperparameter overrides for steelman variants
    parser.add_argument("--lr",           type=float, default=None,
                        help="Override learning rate (default: 5e-4)")
    parser.add_argument("--n_epochs",     type=int,   default=None,
                        help="Override number of epochs (default: 20)")
    parser.add_argument("--weight_decay", type=float, default=None,
                        help="Override weight decay (default: 1e-4)")
    parser.add_argument("--ckpt_suffix",  type=str,   default="",
                        help="Suffix appended to checkpoint filenames (e.g. '_lr2e4')")
    args = parser.parse_args()

    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Apply hyperparameter overrides (for steelman variants)
    train_override = dict(TRAIN)
    if args.lr           is not None: train_override["lr"]           = args.lr
    if args.n_epochs     is not None: train_override["n_epochs"]     = args.n_epochs
    if args.weight_decay is not None: train_override["weight_decay"] = args.weight_decay

    sweep = _load_sweep_results()
    _heartbeat(f"arch_sweep Phase {args.phase} START configs={args.configs} seeds={args.seeds}")

    # ── Phase 1: softmax sweep ────────────────────────────────────────────────
    if args.phase == 1:
        _append_log(f"\n## Phase 1 — Softmax sweep\n")
        for cfg_letter in args.configs:
            cfg = CONFIGS[cfg_letter]
            runs = []
            for seed in args.seeds:
                r = run_one(cfg_letter, "softmax", seed, device, args.smoke,
                            train_cfg=train_override, ckpt_suffix=args.ckpt_suffix)
                if r is not None:
                    runs.append(r)
            if not runs:
                continue
            all_accs  = [r["val_acc"]["all"]  for r in runs]
            hard_accs = [r["val_acc"]["hard"] for r in runs]
            am = statistics.mean(all_accs)  * 100
            as_ = (statistics.stdev(all_accs) * 100 if len(all_accs) > 1 else 0)
            hm = statistics.mean(hard_accs) * 100
            hs = (statistics.stdev(hard_accs) * 100 if len(hard_accs) > 1 else 0)
            arch = runs[0]["arch"]
            print(f"\nConfig {cfg_letter}: all={am:.1f}±{as_:.1f}%  "
                  f"hard={hm:.1f}±{hs:.1f}%  "
                  f"(d={arch['d_model']},h={arch['n_heads']},L={arch['n_layers']})")
            _append_log(
                f"| {cfg_letter} | d={arch['d_model']},h={arch['n_heads']},"
                f"L={arch['n_layers']},ff={arch['d_ff']} "
                f"| {am:.1f}±{as_:.1f}% | {hm:.1f}±{hs:.1f}% |"
            )
            sweep.setdefault("phase1", {})[cfg_letter] = {
                "runs": runs,
                "mean_all": am, "std_all": as_,
                "mean_hard": hm, "std_hard": hs,
            }
            _save_sweep_results(sweep)

    # ── Phase 2: oscillator at sweet-spot ────────────────────────────────────
    elif args.phase == 2:
        _append_log(f"\n## Phase 2 — Oscillator at config(s): {args.configs}\n")
        for cfg_letter in args.configs:
            runs = []
            for seed in args.seeds:
                r = run_one(cfg_letter, "kuramoto", seed, device,
                            args.smoke, d_osc=args.d_osc,
                            train_cfg=train_override, ckpt_suffix=args.ckpt_suffix)
                if r is not None:
                    runs.append(r)
            if not runs:
                continue
            all_accs  = [r["val_acc"]["all"]  for r in runs]
            hard_accs = [r["val_acc"]["hard"] for r in runs]
            am = statistics.mean(all_accs)  * 100
            as_ = (statistics.stdev(all_accs) * 100 if len(all_accs) > 1 else 0)
            hm = statistics.mean(hard_accs) * 100
            hs = (statistics.stdev(hard_accs) * 100 if len(hard_accs) > 1 else 0)
            print(f"\nConfig {cfg_letter} OSC: all={am:.1f}±{as_:.1f}%  "
                  f"hard={hm:.1f}±{hs:.1f}%")
            _append_log(f"| {cfg_letter} osc d={args.d_osc} "
                        f"| {am:.1f}±{as_:.1f}% | {hm:.1f}±{hs:.1f}% |")
            sweep.setdefault("phase2", {})[cfg_letter] = {
                "runs": runs,
                "mean_all": am, "std_all": as_,
                "mean_hard": hm, "std_hard": hs,
            }
            _save_sweep_results(sweep)

    # ── Phase 3: frozen W_V ───────────────────────────────────────────────────
    elif args.phase == 3:
        _append_log(f"\n## Phase 3 — Frozen W_V at config(s): {args.configs}\n")
        for cfg_letter in args.configs:
            runs = []
            for seed in args.seeds:
                r = run_one(cfg_letter, "kuramoto", seed, device,
                            args.smoke, freeze_wv=True, d_osc=args.d_osc,
                            train_cfg=train_override, ckpt_suffix=args.ckpt_suffix)
                if r is not None:
                    runs.append(r)
            if not runs:
                continue
            all_accs  = [r["val_acc"]["all"]  for r in runs]
            hard_accs = [r["val_acc"]["hard"] for r in runs]
            am = statistics.mean(all_accs)  * 100
            as_ = (statistics.stdev(all_accs) * 100 if len(all_accs) > 1 else 0)
            hm = statistics.mean(hard_accs) * 100
            hs = (statistics.stdev(hard_accs) * 100 if len(hard_accs) > 1 else 0)
            print(f"\nConfig {cfg_letter} frozenV: all={am:.1f}±{as_:.1f}%  "
                  f"hard={hm:.1f}±{hs:.1f}%")
            _append_log(f"| {cfg_letter} frozenV | {am:.1f}±{as_:.1f}% | {hm:.1f}±{hs:.1f}% |")
            sweep.setdefault("phase3", {})[cfg_letter] = {
                "runs": runs,
                "mean_all": am, "std_all": as_,
                "mean_hard": hm, "std_hard": hs,
            }
            _save_sweep_results(sweep)

    _heartbeat(f"arch_sweep Phase {args.phase} COMPLETE configs={args.configs}")
    print(f"\nSweep results saved to: {SWEEP_JSON}")


if __name__ == "__main__":
    main()
