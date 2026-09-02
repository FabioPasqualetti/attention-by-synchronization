"""Table 5 — TinyStories readout-sharpening p-ablation on the 5-seed cohort.

Runs the readout-sharpening ablation on the exact main TinyStories cohort recipe. Mechanism:
analytic LoheLanguageTransformer (== train_tinystories.py / the lm_dimension_scaling main cohort);
NOT LoheLMSigma, which is the coupling ablation.

Recipe == main TS cohort: LoheLanguageTransformer d_model=128 n_h=4 n_l=2 d_ff=512 max_seq_len=128
dropout=0.1; AdamW lr=5e-4 wd=1e-4, CosineAnnealingLR(T_max=n_epochs*len(loader)), 5 epochs,
batch 256, grad_clip 1.0; metric = best (min) val PPL over epochs (== train_tinystories).

GATE: for each d_osc, p=1 seed-0 must reproduce the lm_dimension_scaling main cohort mean within 3% rel tol
(d_osc=2 -> 10.911; d_osc=8 -> 9.784). On GATE_FAIL: stop, flag, run nothing further.

p=1: only the gate seed-0 per d_osc is run; the 5-seed p=1 rows REUSE lm_dimension_scaling
(TS d2 10.91±0.07, TS d8 9.78±0.06). p=2,p=4: fresh 5 seeds each × {d2,d8} = 20 runs.
Writes results/lm_readout_exponent/ (NEW dir; nothing overwritten). Resumable.
Run: python language_modeling/readout_exponent.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training import harness, paths  # noqa: E402

paths.ensure_paths()
import torch  # noqa: E402

from oscillator_attention import LoheLanguageTransformer  # noqa: E402
from training.data_utils import (  # noqa: E402
    load_tinystories, make_lm_loaders, train_lm_epoch, eval_ppl)

EXP = "lm_readout_exponent"
ARCH = dict(d_model=128, n_heads=4, n_layers=2, d_ff=512, max_seq_len=128, dropout=0.1)
TRAIN = dict(lr=5e-4, weight_decay=1e-4, batch_size=256, grad_clip=1.0, n_epochs=5)
SEEDS = [0, 1, 2, 3, 4]
GATE_TOL = 0.03
ESCALE_REF = {2: 10.911, 8: 9.784}  # lm_dimension_scaling ts_uniform scaling means (== Table 4)

_DATA = {}


def _loaders():
    if "loaders" not in _DATA:
        vocab, _, tr, va = load_tinystories(max_len=ARCH["max_seq_len"])
        _DATA["vocab"] = vocab
        _DATA["loaders"] = make_lm_loaders(tr, va, ARCH["max_seq_len"],
                                           batch_size=TRAIN["batch_size"])
    return _DATA["vocab"], _DATA["loaders"]


def run_one(d_osc, p, seed, device):
    torch.manual_seed(seed)
    vocab, (train_loader, val_loader) = _loaders()
    model = LoheLanguageTransformer(vocab_size=len(vocab), p=p, d_osc=d_osc, **ARCH).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=TRAIN["lr"], weight_decay=TRAIN["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TRAIN["n_epochs"] * len(train_loader))
    best = float("inf")
    for _ in range(TRAIN["n_epochs"]):
        train_lm_epoch(model, train_loader, opt, sched, TRAIN["grad_clip"], device)
        best = min(best, eval_ppl(model, val_loader, device))
        if device.type == "mps":
            torch.mps.empty_cache()
    return best


def _save(d_osc, p, seed, ppl, extra=None):
    payload = {"d_osc": d_osc, "p": p, "seed": seed, "val_ppl": ppl, "arch": ARCH, "train": TRAIN}
    if extra:
        payload.update(extra)
    harness.save_result(EXP, f"d{d_osc}_p{p}_s{seed}", payload)


def main():
    device = harness.pick_device("mps")
    print(f"{EXP} device={device}", flush=True)

    # ---- GATES: p=1 seed-0 for d_osc in {2,8} ----
    for d_osc in (2, 8):
        key = f"d{d_osc}_p1_s0"
        if harness.exists(EXP, key):
            g = harness.load_result(EXP, key)
            if g.get("GATE_FAIL"):
                print(f"prior GATE_FAIL {key}; stopping.", flush=True); return
            print(f"skip gate {key} (done: {g['val_ppl']:.3f})", flush=True); continue
        ref = ESCALE_REF[d_osc]
        with harness.Timer() as t:
            ppl = run_one(d_osc, 1, 0, device)
        dev = abs(ppl - ref) / ref
        ok = dev <= GATE_TOL
        _save(d_osc, 1, 0, ppl, {"wall_sec": round(t.wall, 1), "gate_ref": ref,
                                 "gate_rel_dev": dev, "gate_ok": ok, "GATE_FAIL": (not ok)})
        print(f"[GATE] {key}: ppl={ppl:.3f} vs lm_dimension_scaling {ref:.3f} "
              f"(rel dev {dev*100:.3f}%) -> {'OK' if ok else 'GATE_FAIL'}", flush=True)
        harness.free_memory(device)
        if not ok:
            print(f"GATE_FAIL d_osc={d_osc}: stopping A-LM.", flush=True); return

    # ---- p=2, p=4: fresh 5 seeds each for d_osc in {2,8} ----
    for p in (2, 4):
        for d_osc in (2, 8):
            for seed in SEEDS:
                key = f"d{d_osc}_p{p}_s{seed}"
                if harness.exists(EXP, key):
                    print(f"skip {key}", flush=True); continue
                with harness.Timer() as t:
                    ppl = run_one(d_osc, p, seed, device)
                _save(d_osc, p, seed, ppl, {"wall_sec": round(t.wall, 1)})
                print(f"DONE {key}: ppl={ppl:.4f}  ({t.wall/60:.1f}m)", flush=True)
                harness.free_memory(device)
    print(f"{EXP} COMPLETE", flush=True)


if __name__ == "__main__":
    main()
