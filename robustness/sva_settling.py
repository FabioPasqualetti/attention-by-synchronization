"""sva_settling — finite-settling residual for the SVA config-G checkpoints.

Produces the SVA column of the finite-settling table (paper Table `tab:settle`).
For each shipped config-G seed (sva/checkpoints/osc_s{0..4}_ep20.pt):

  baseline : analytic fixed-point accuracy on the full SVA validation set.
  settle   : RK45 integration (swap_lohe ode_T) from random z(0), read at
             T in {0.5,1,2,5,10,30}, N_DRAWS random initializations, same val set.

The residual reported for each T is |acc(T) - acc_analytic| / acc_analytic,
averaged over seeds x draws. Because the metric is accuracy, its resolution
(1 / (n_val * n_seeds) pooled) bounds the residual from below at small T; see the
caption of Table `tab:settle`. Baseline and settle use the SAME (full) val set,
so the comparison is like-for-like.

Deterministic on CPU (MPS disabled) so the shipped checkpoints reproduce exactly.
Run from the repo root:  python robustness/sva_settling.py
"""
import os
import sys
import json
import statistics

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sva")))

import torch
# Force CPU determinism: the RK45 path and the shipped checkpoints reproduce
# exactly on CPU; MPS is non-deterministic here.
torch.backends.mps.is_available = lambda: False

from training import harness  # noqa: E402
from training import sva_dataset as sva_ds  # noqa: E402
from training.sva_dataset import build_datasets, make_sva_loaders  # noqa: E402
from oscillator_attention import SVATransformer  # noqa: E402
from training.lohe_probe import swap_lohe, PerturbableLohe  # noqa: E402

EXP = "robustness_perturbations"
CKPT_DIR = os.path.join(harness.REPO_ROOT, "sva", "checkpoints")
CFG = dict(d_model=32, n_heads=1, n_layers=1, d_ff=64, d_osc=2, max_seq_len=50)
SEEDS = [0, 1, 2, 3, 4]
GRID = [0.5, 1, 2, 5, 10, 30]
N_DRAWS = 3
DEVICE = torch.device("cpu")


def _make_model():
    return SVATransformer(
        vocab_size=len(sva_ds.VOCAB), attn_type="kuramoto",
        d_osc=CFG["d_osc"], d_model=CFG["d_model"], n_heads=CFG["n_heads"],
        n_layers=CFG["n_layers"], d_ff=CFG["d_ff"], max_seq_len=CFG["max_seq_len"],
        dropout=0.0)


def _load(seed):
    m = _make_model()
    m.load_state_dict(torch.load(os.path.join(CKPT_DIR, f"osc_s{seed}_ep20.pt"),
                                 map_location=DEVICE, weights_only=True))
    return m.to(DEVICE)


@torch.no_grad()
def accuracy(model, loader):
    """Binary agree/disagree accuracy over the full SVA validation set."""
    model.eval()
    ncorrect = ntotal = 0
    for batch in loader:
        x = batch["tokens"].to(DEVICE)
        y = batch["labels"].to(DEVICE)
        pad = batch["pad_mask"].to(DEVICE)
        vi = batch["verb_idx"].to(DEVICE)
        preds = (model(x, padding_mask=pad, verb_idx=vi) > 0).long()
        ncorrect += int((preds == y).sum())
        ntotal += y.numel()
    return ncorrect / ntotal, ntotal


def main():
    _, val_loader = make_sva_loaders(batch_size=256, n_train=-1, n_val=-1)

    base_acc = {}
    curve = {T: [] for T in GRID}
    n_val = None

    for seed in SEEDS:
        # baseline: analytic fixed point (no swap), full val set
        m = _load(seed)
        b, n_val = accuracy(m, val_loader)
        base_acc[seed] = b
        harness.save_result(EXP, f"sva_baseline_s{seed}",
                            {"target": "sva", "type": "baseline", "seed": seed,
                             "acc": b, "n_val": n_val})
        # settle: RK45 at each horizon, N_DRAWS random inits
        for T in GRID:
            draws = []
            for d in range(N_DRAWS):
                mm = _load(seed)
                swap_lohe(mm, ode_T=T, rng_seed=7000 * d + int(T * 10))
                a, _ = accuracy(mm, val_loader)
                draws.append(a)
                harness.save_result(
                    EXP, f"sva_settle_T{T:g}_s{seed}_d{d}",
                    {"target": "sva", "type": "settle", "T": T, "seed": seed,
                     "draw": d, "acc": a, "n_val": n_val})
            curve[T].append(statistics.mean(draws))
        print(f"DONE seed {seed}: baseline acc={b:.4f}", flush=True)
        harness.free_memory(DEVICE)

    mb = statistics.mean(base_acc.values())
    resolution = 100.0 / (n_val * len(SEEDS))  # pooled one-flip resolution (%)
    print(f"\nSVA finite-settling residual (mean over {len(SEEDS)} seeds x "
          f"{N_DRAWS} draws); n_val={n_val}, pooled resolution "
          f"~{resolution:.4f}% ; analytic-FP baseline acc={mb:.4f}")
    print(f"{'T':>5} | {'settle acc':>10} | {'residual':>9}")
    for T in GRID:
        st = statistics.mean(curve[T])
        res = abs(st - mb) / abs(mb) * 100
        tag = "  (<res.)" if res < resolution else ""
        print(f"{T:>5} | {st:>10.4f} | {res:>8.3f}%{tag}")
    print("\nsva_settling COMPLETE", flush=True)


if __name__ == "__main__":
    main()
