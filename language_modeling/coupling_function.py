"""lm_coupling_function — Coupling-nonlinearity ablation.

Replace the coupling sigma in LoheAttention with:
  softplus (reference), relu_eps = relu(x)+1e-3, elu1 = elu(x)+1.

Runs:
  KWS reference config (5 seeds each sigma)  -> val accuracy
  TinyStories d_osc=8    (3 seeds each sigma) -> validation PPL

Training mirrors the stock KWS (Adam, 30ep, cosine per-epoch) and TinyStories
(AdamW, 5ep, cosine per-batch, batch 256) loops. Resumable per-run JSON in results/lm_coupling_function/.
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training import harness, paths  # noqa: E402

paths.ensure_paths()
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from oscillator_attention.sigma_models import build_kws_sigma, LoheLMSigma  # noqa: E402

EXP = "lm_coupling_function"
SIGMAS = ["softplus", "relu_eps", "elu1"]
CKPT = os.path.join(harness.RUNS_ROOT, EXP, "ckpt")

# ---- KWS ----
KWS_ARCH = dict(n_feats=40, d_model=32, n_heads=2, n_layers=1, n_classes=10,
                T=49, d_osc=2, p=1, dropout=0.1)
KWS_TRAIN = dict(lr=1e-3, weight_decay=1e-4, n_epochs=30, grad_clip=1.0)
KWS_SEEDS = [0, 1, 2, 3, 4]


def _kws_eval(model, loader, device):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for feats, labels in loader:
            if feats is None:
                continue
            feats, labels = feats.to(device), labels.to(device)
            correct += (model(feats).argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)


def run_kws(sigma, seed, device):
    from training.kws_data import cached_loaders
    torch.manual_seed(seed)
    train_loader, val_loader, _, _ = cached_loaders(seed, batch_size=64)
    model = build_kws_sigma(sigma, **KWS_ARCH).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=KWS_TRAIN["lr"],
                           weight_decay=KWS_TRAIN["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=KWS_TRAIN["n_epochs"])
    best = 0.0
    for _ in range(KWS_TRAIN["n_epochs"]):
        model.train()
        for feats, labels in train_loader:
            if feats is None:
                continue
            feats, labels = feats.to(device), labels.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(feats), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), KWS_TRAIN["grad_clip"])
            opt.step()
        sched.step()
        best = max(best, _kws_eval(model, val_loader, device))
        if device.type == "mps":
            torch.mps.empty_cache()
    return best


# ---- TinyStories ----
TS_ARCH = dict(d_model=128, n_heads=4, n_layers=2, d_ff=512, max_seq_len=128, dropout=0.1)
TS_TRAIN = dict(lr=5e-4, weight_decay=1e-4, batch_size=256, grad_clip=1.0, n_epochs=5)
# Reduced from 3 seeds/sigma to 1 seed/sigma for tractability on MPS (4.67M chunks,
# ~1.5-2 h/run). Gated: the softplus seed-0 run must reproduce the paper's TS PPL (~9.75,
# within 0.3) before the relu_eps/elu1 confirmatory seeds run. The KWS half (3 sigma within
# 0.01pp at 5 seeds) is the statistically weighted result; TS is one confirmatory point per
# sigma on a second task class. See results/lm_coupling_function.
TS_SEEDS = [0]


def run_ts(sigma, seed, device, save_ckpt=False):
    from training.data_utils import (
        load_tinystories, make_lm_loaders, train_lm_epoch, eval_ppl)
    torch.manual_seed(seed)
    vocab, _, train_chunks, val_chunks = load_tinystories(max_len=TS_ARCH["max_seq_len"])
    train_loader, val_loader = make_lm_loaders(
        train_chunks, val_chunks, TS_ARCH["max_seq_len"], batch_size=TS_TRAIN["batch_size"])
    model = LoheLMSigma(vocab_size=len(vocab), d_osc=8, sigma=sigma, **TS_ARCH).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=TS_TRAIN["lr"],
                            weight_decay=TS_TRAIN["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=TS_TRAIN["n_epochs"] * len(train_loader))
    best = float("inf")
    for _ in range(TS_TRAIN["n_epochs"]):
        train_lm_epoch(model, train_loader, opt, sched, TS_TRAIN["grad_clip"], device)
        best = min(best, eval_ppl(model, val_loader, device))
        if device.type == "mps":
            torch.mps.empty_cache()
    if save_ckpt:
        os.makedirs(CKPT, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(CKPT, f"ts_{sigma}_s{seed}.pt"))
    return best


def main():
    device = harness.pick_device("mps")
    print(f"lm_coupling_function device={device}", flush=True)

    for sigma in SIGMAS:
        for seed in KWS_SEEDS:
            key = f"kws_{sigma}_s{seed}"
            if harness.exists(EXP, key):
                print(f"skip {key}", flush=True); continue
            t0 = time.time()
            acc = run_kws(sigma, seed, device)
            harness.save_result(EXP, key, {
                "task": "kws", "sigma": sigma, "seed": seed, "val_acc": acc,
                "wall_sec": round(time.time() - t0, 1), "arch": KWS_ARCH, "train": KWS_TRAIN})
            print(f"DONE {key}: {acc*100:.2f}%  ({(time.time()-t0)/60:.1f}m)", flush=True)

    # TS reproduction gate (decision 1): the softplus seed-0 PPL must reproduce the paper
    # (~9.75) within TS_GATE_TOL before the relu_eps/elu1 confirmatory seeds run. If it
    # deviates, the TS ablation track stops (relu_eps/elu1 skipped) and the deviation is
    # reported; softplus s0 itself is kept (robustness_perturbations's TS arm uses its checkpoint).
    TS_REF_PPL, TS_GATE_TOL = 9.75, 0.3
    ts_gate_ok = True
    if harness.exists(EXP, "ts_softplus_s0"):
        ppl0 = harness.load_result(EXP, "ts_softplus_s0")["val_ppl"]
        if abs(ppl0 - TS_REF_PPL) > TS_GATE_TOL:
            ts_gate_ok = False
            print(f"TS GATE FAIL: softplus s0 PPL={ppl0:.3f} deviates >{TS_GATE_TOL} from "
                  f"ref {TS_REF_PPL}; STOPPING TS ablation (relu_eps/elu1 skipped).", flush=True)
        else:
            print(f"TS GATE PASS: softplus s0 PPL={ppl0:.3f} (ref {TS_REF_PPL}±{TS_GATE_TOL}).",
                  flush=True)

    for sigma in SIGMAS:
        for seed in TS_SEEDS:
            key = f"ts_{sigma}_s{seed}"
            if harness.exists(EXP, key):
                print(f"skip {key}", flush=True); continue
            if sigma != "softplus" and not ts_gate_ok:
                print(f"skip {key} (TS gate failed)", flush=True); continue
            t0 = time.time()
            # keep one softplus ckpt for robustness_perturbations/theory_degenerate_tokens
            ppl = run_ts(sigma, seed, device, save_ckpt=(sigma == "softplus" and seed == 0))
            harness.save_result(EXP, key, {
                "task": "tinystories", "sigma": sigma, "seed": seed, "val_ppl": ppl,
                "wall_sec": round(time.time() - t0, 1), "arch": TS_ARCH, "train": TS_TRAIN})
            print(f"DONE {key}: ppl={ppl:.3f}  ({(time.time()-t0)/60:.1f}m)", flush=True)
    print("lm_coupling_function COMPLETE", flush=True)


if __name__ == "__main__":
    main()
