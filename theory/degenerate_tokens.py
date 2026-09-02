"""theory_degenerate_tokens — Degenerate-set statistics.

For each available trained oscillator checkpoint, compute the empirical distribution
of ||h_i|| (h = sum_j W_ij anchor_j, the pre-normalization driving vector) over the
validation set (all tokens, all heads). Report min, percentiles {0.1,1,5,50}, and the
fraction below 0.01 (paper's degenerate threshold). Writes CSV + histogram PDF + JSON.

Checkpoints processed (only if present):
  KWS   oscillator d_osc=2   (from kws_position_matched osc/none seed0, else repro B1)
  SVA   oscillator d_osc=2   (from sva_seed_robustness main_kuramoto_s0)
  TinyStories d_osc=8        (from lm_coupling_function ts_softplus_s0)
  TinyStories d_osc=2        (from repo results/lm/checkpoints_ts if present)

Runs after kws_position_matched/sva_seed_robustness/lm_coupling_function. Resumable: skips a source whose results/theory_degenerate_tokens/<name>.json exists.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training import harness, paths  # noqa: E402

paths.ensure_paths()
import numpy as np  # noqa: E402
import torch  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from training.lohe_probe import HNormCapture  # noqa: E402

EXP = "theory_degenerate_tokens"
OUTDIR = os.path.join(harness.RUNS_ROOT, EXP)
FIGDIR = os.path.join(harness.REPO_ROOT, "figures", "diagnostics", "theory_degenerate_tokens")
RROOT = harness.RUNS_ROOT
REPO = harness.REPO_ROOT


def _stats(arr):
    arr = np.asarray(arr, dtype=np.float64)
    pct = {p: float(np.percentile(arr, p)) for p in (0.1, 1, 5, 50)}
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "pct": pct,
        "frac_below_0.01": float((arr < 0.01).mean()),
    }


def _write_csv(name, arr):
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, f"{name}_hnorm.csv")
    # write quantile summary + a subsample of raw values (cap 200k)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantile", "value"])
        for q in (0.0, 0.001, 0.01, 0.05, 0.5, 0.9, 0.99, 1.0):
            w.writerow([q, float(np.quantile(arr, q))])
    return path


def _hist(name, arr):
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, f"{name}_hnorm_hist.pdf")
    fig, ax = plt.subplots(figsize=(5, 3.2))
    lo = max(arr.min(), 1e-4)
    bins = np.logspace(np.log10(lo), np.log10(arr.max() + 1e-9), 60)
    ax.hist(np.clip(arr, lo, None), bins=bins, color="#3b6", alpha=0.85)
    ax.set_xscale("log")
    ax.axvline(0.01, color="crimson", ls="--", lw=1, label="degenerate thr 0.01")
    ax.set_xlabel(r"$\|h_i\|$"); ax.set_ylabel("count")
    ax.set_title(f"{name}: ||h|| over val set")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


# ── model + val-loader builders ──────────────────────────────────────────────

def build_kws():
    from oscillator_attention.kws_pe_model import KWSTransformerPE
    from training.kws_data import cached_loaders
    m = KWSTransformerPE(pe="none", attn_type="lohe", p=1,
                         n_feats=40, d_model=32, n_heads=2, n_layers=1,
                         n_classes=10, T=49, d_osc=2, dropout=0.1)
    _, val, _, _ = cached_loaders(0, batch_size=64)
    return m, val, "kws"


def build_sva(ckpt):
    from oscillator_attention import SVATransformer
    sva_ds = paths.load_sva_dataset()
    m = SVATransformer(vocab_size=len(sva_ds.VOCAB), attn_type="kuramoto",
                       d_model=32, n_heads=1, n_layers=1, d_ff=64, d_osc=2,
                       max_seq_len=50, dropout=0.1)
    _, val = sva_ds.make_sva_loaders(64, n_train=200, n_val=-1)
    return m, val, "sva"


def build_ts(d_osc):
    from oscillator_attention.sigma_models import LoheLMSigma
    from training.data_utils import load_tinystories, make_lm_loaders
    vocab, _, train_chunks, val_chunks = load_tinystories(max_len=128)
    m = LoheLMSigma(vocab_size=len(vocab), d_osc=d_osc, sigma="softplus",
                    d_model=128, n_heads=4, n_layers=2, d_ff=512,
                    max_seq_len=128, dropout=0.1)
    _, val = make_lm_loaders(train_chunks, val_chunks[:500], 128, batch_size=64)
    return m, val, "lm"


def process(name, builder, ckpt_path, device):
    if harness.exists(EXP, name):
        print(f"skip {name}", flush=True); return
    if not os.path.exists(ckpt_path):
        print(f"no ckpt for {name}: {ckpt_path}", flush=True); return
    model, val, task = builder()
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    model.to(device).eval()

    cap = HNormCapture(model)
    with torch.no_grad():
        for batch in val:
            if task == "kws":
                feats, labels = batch
                if feats is None:
                    continue
                cap.set_pad(None)
                model(feats.to(device))
            elif task == "sva":
                x = batch["tokens"].to(device)
                pad = batch["pad_mask"].to(device)
                cap.set_pad(pad)
                model(x, padding_mask=pad,
                      verb_idx=batch["verb_idx"].to(device))
            else:  # lm
                x = batch["tokens"].to(device)
                pad = batch["pad_mask"].to(device)
                cap.set_pad(pad)
                model(x, padding_mask=pad)
    cap.remove()
    arr = cap.all_norms()
    st = _stats(arr)
    csv_path = _write_csv(name, arr)
    pdf_path = _hist(name, arr)
    harness.save_result(EXP, name, {
        "name": name, "ckpt": ckpt_path, "stats": st,
        "csv": os.path.relpath(csv_path, REPO),
        "pdf": os.path.relpath(pdf_path, REPO)})
    print(f"DONE {name}: min={st['min']:.4g} p0.1={st['pct'][0.1]:.4g} "
          f"p50={st['pct'][50]:.4g} frac<0.01={st['frac_below_0.01']:.4g}", flush=True)


def main():
    device = harness.pick_device("mps")
    print(f"theory_degenerate_tokens device={device}", flush=True)
    sources = [
        ("kws_osc_d2",
         build_kws,
         os.path.join(RROOT, "kws_position_matched", "ckpt", "osc_none_s0_final.pt")),
        ("kws_osc_d2_repro",
         build_kws,
         os.path.join(RROOT, "repro", "ckpt", "kws_B1_s0_final.pt")),
        ("sva_osc_d2",
         lambda: build_sva(None),
         os.path.join(RROOT, "sva_seed_robustness", "ckpt", "main_kuramoto_s0.pt")),
        ("ts_osc_d8",
         lambda: build_ts(8),
         os.path.join(RROOT, "lm_coupling_function", "ckpt", "ts_softplus_s0.pt")),
        ("ts_osc_d2_repo",
         lambda: build_ts(2),
         os.path.join(paths.runs_root(), "lm", "checkpoints_ts", "ts_d2_s0_ep5.pt")),
    ]
    n_present = sum(os.path.exists(ckpt) for _, _, ckpt in sources)
    for name, builder, ckpt in sources:
        try:
            process(name, builder, ckpt, device)
        except Exception as e:
            print(f"ERROR {name}: {type(e).__name__}: {e}", flush=True)
        harness.free_memory(device)  # release MPS cache between checkpoints
    if n_present == 0:
        print("theory_degenerate_tokens SKIPPED: none of the checkpoints are "
              "present (not shipped); nothing was computed. Retrain with the "
              "included configs to regenerate.", flush=True)
        sys.exit(1)
    print("theory_degenerate_tokens COMPLETE", flush=True)


if __name__ == "__main__":
    main()
