"""Table 5 — KWS readout-sharpening p-ablation on the 5-seed kws_position_matched no-PE cohort.

Runs the readout-sharpening ablation on the kws_position_matched harness, so p=1 reproduces the
kws_position_matched osc/none cohort bit-exactly and p=2/p=4 are the same recipe with a sharpened
readout. Mechanism: analytic LoheAttention (KWSTransformerPE pe='none' == the stock
KWSTransformer).

Recipe == kws_position_matched (position_matched.py::run_one): Adam lr=1e-3 wd=1e-4, CosineAnnealingLR(T_max=30), 30 epochs,
grad_clip 1.0, batch 64, cross-entropy, best val acc; cached_loaders(seed). ARCH == kws_position_matched.

GATE: p=1 seed-0 must reproduce kws_position_matched osc/none seed-0 (results/kws_position_matched/osc_none_s0.json, 88.901%) within
3% rel tol (bit-exact expected). On GATE_FAIL: stop, write flag, do NOT run p2/p4.

p=1: only seed-0 is run (as the gate); the 5-seed p=1 row REUSES kws_position_matched osc/none (88.79±0.29).
p=2,p=4: fresh 5 seeds each (10 runs). Writes results/kws_readout_exponent/ (NEW dir; nothing overwritten).
Run: python keyword_spotting/readout_exponent.py
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

EXP = "kws_readout_exponent"
ARCH = dict(n_feats=40, d_model=32, n_heads=2, n_layers=1, n_classes=10,
            T=49, d_osc=2, dropout=0.1)  # == kws_position_matched
TRAIN = dict(lr=1e-3, weight_decay=1e-4, n_epochs=30, grad_clip=1.0)  # == kws_position_matched
SEEDS = [0, 1, 2, 3, 4]
GATE_TOL = 0.03


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


def run_one(p, seed, device):
    torch.manual_seed(seed)
    train_loader, val_loader, _, _ = cached_loaders(seed, batch_size=64)
    model = KWSTransformerPE(attn_type="lohe", p=p, pe="none", **ARCH).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=TRAIN["lr"], weight_decay=TRAIN["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TRAIN["n_epochs"])
    best = 0.0
    for _ in range(TRAIN["n_epochs"]):
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
        best = max(best, _evaluate(model, val_loader, device))
        if device.type == "mps":
            torch.mps.empty_cache()
    return best


def _save(p, seed, acc, extra=None):
    payload = {"p": p, "seed": seed, "val_acc": acc, "arch": ARCH, "train": TRAIN}
    if extra:
        payload.update(extra)
    harness.save_result(EXP, f"p{p}_s{seed}", payload)


def main():
    device = harness.pick_device("mps")
    print(f"{EXP} device={device}", flush=True)

    # ---- GATE: p=1 seed-0 vs kws_position_matched osc/none seed-0 ----
    ref = harness.load_reference("kws_position_matched", "osc_none_s0")["val_acc"]  # 0.889009
    if not harness.exists(EXP, "p1_s0"):
        with harness.Timer() as t:
            acc = run_one(1, 0, device)
        dev = abs(acc - ref) / ref
        ok = dev <= GATE_TOL
        _save(1, 0, acc, {"wall_sec": round(t.wall, 1), "gate_ref": ref,
                          "gate_rel_dev": dev, "gate_ok": ok,
                          "bit_exact": acc == ref, "GATE_FAIL": (not ok)})
        print(f"[GATE] p1_s0: acc={acc*100:.3f}% vs kws_position_matched {ref*100:.3f}% "
              f"(rel dev {dev*100:.4f}%, bit_exact={acc==ref}) -> {'OK' if ok else 'GATE_FAIL'}",
              flush=True)
        if not ok:
            print("GATE_FAIL: stopping A-KWS (p2/p4 not run).", flush=True)
            return
        harness.free_memory(device)
    else:
        g = harness.load_result(EXP, "p1_s0")
        if g.get("GATE_FAIL"):
            print("prior GATE_FAIL recorded; stopping.", flush=True); return
        print(f"skip gate p1_s0 (done: {g['val_acc']*100:.3f}%)", flush=True)

    # ---- p=2, p=4: fresh 5 seeds each ----
    for p in (2, 4):
        for seed in SEEDS:
            key = f"p{p}_s{seed}"
            if harness.exists(EXP, key):
                print(f"skip {key}", flush=True); continue
            with harness.Timer() as t:
                acc = run_one(p, seed, device)
            _save(p, seed, acc, {"wall_sec": round(t.wall, 1)})
            print(f"DONE {key}: {acc*100:.2f}%  ({t.wall/60:.1f}m)", flush=True)
            harness.free_memory(device)
    print(f"{EXP} COMPLETE", flush=True)


if __name__ == "__main__":
    main()
