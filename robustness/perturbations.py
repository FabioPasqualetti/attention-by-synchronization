"""robustness_perturbations — Physical-robustness simulation (inference-time perturbations on trained ckpts).

Targets:
  KWS oscillator d_osc=2   (kws_position_matched osc/none seed0)  -> accuracy on the FULL KWS val set
  TinyStories d_osc=8      (lm_coupling_function ts_softplus s0)   -> PPL on 100 TS val sequences

Perturbations (see lib/lohe_probe.PerturbableLohe):
  a) coupling mismatch  W_ij <- W_ij*exp(eps), eps~N(0,s^2), s in {.01,.05,.1,.2,.5}, 5 draws
  b) state noise        z* <- normalize(z*+eta), eta~N(0,s^2 I),   same grid,        5 draws
  c) finite settling    RK45 (rtol=atol=1e-6, CPU) from random z(0), read at
                        T in {.5,1,2,5,10,30}, 3 init draws

NOTE (documented deviation): the RK45 finite-settling sweep (c) is CPU-O(heavy). It is
evaluated on a fixed subset (ODE_MAX_KWS samples / ODE_MAX_TS sequences) to keep wall
time bounded; (a) and (b) use the full val set / 100 sequences as specified.

Writes per-condition JSON + a combined CSV and one matplotlib PDF per curve. Resumable.
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

from training.lohe_probe import swap_lohe  # noqa: E402

EXP = "robustness_perturbations"
RROOT = harness.RUNS_ROOT
REPO = harness.REPO_ROOT
FIGDIR = os.path.join(REPO, "figures", "diagnostics", "robustness_perturbations")
OUTDIR = os.path.join(RROOT, EXP)

SCALES = [0.01, 0.05, 0.1, 0.2, 0.5]
SETTLE_T = [0.5, 1, 2, 5, 10, 30]
N_DRAWS_NOISE = 5
N_DRAWS_INIT = 3
ODE_MAX_KWS = 512     # samples for RK45 settling (documented cap)
ODE_MAX_TS = 100      # sequences


# ── metric evaluators ────────────────────────────────────────────────────────

def kws_accuracy(model, loader, device, max_samples=None):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for feats, labels in loader:
            if feats is None:
                continue
            feats, labels = feats.to(device), labels.to(device)
            correct += (model(feats).argmax(1) == labels).sum().item()
            total += labels.size(0)
            if max_samples and total >= max_samples:
                break
    return correct / max(total, 1)


def ts_ppl(model, loader, device, max_seqs=None):
    import math
    model.eval(); tot_loss = tot_tok = seen = 0
    ce = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    with torch.no_grad():
        for batch in loader:
            x = batch["tokens"].to(device)
            pad = batch["pad_mask"].to(device)
            logits = model(x, padding_mask=pad)
            tgt = x[:, 1:].contiguous()
            lg = logits[:, :-1].contiguous()
            loss = ce(lg.view(-1, lg.size(-1)), tgt.view(-1))
            ntok = (tgt != 0).sum().item()
            tot_loss += loss.item(); tot_tok += ntok
            seen += x.size(0)
            if max_seqs and seen >= max_seqs:
                break
    return math.exp(tot_loss / max(tot_tok, 1))


# ── model builders (fresh model each eval; weights loaded from ckpt) ─────────

def build_kws(ckpt, device):
    from oscillator_attention.kws_pe_model import KWSTransformerPE
    from training.kws_data import cached_loaders
    m = KWSTransformerPE(pe="none", attn_type="lohe", p=1, n_feats=40,
                         d_model=32, n_heads=2, n_layers=1, n_classes=10,
                         T=49, d_osc=2, dropout=0.1)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
    _, val, _, _ = cached_loaders(0, batch_size=64)
    return m.to(device), val


def build_ts(ckpt, device):
    from oscillator_attention.sigma_models import LoheLMSigma
    from training.data_utils import load_tinystories, make_lm_loaders
    vocab, _, train_chunks, val_chunks = load_tinystories(max_len=128)
    m = LoheLMSigma(vocab_size=len(vocab), d_osc=8, sigma="softplus",
                    d_model=128, n_heads=4, n_layers=2, d_ff=512,
                    max_seq_len=128, dropout=0.1)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
    _, val = make_lm_loaders(train_chunks, val_chunks[:ODE_MAX_TS], 128, batch_size=50)
    return m.to(device), val


def _eval(target, model, loader, device, ode_subset=False):
    if target == "kws":
        return kws_accuracy(model, loader, device,
                            max_samples=ODE_MAX_KWS if ode_subset else None)
    return ts_ppl(model, loader, device,
                  max_seqs=ODE_MAX_TS if ode_subset else None)


def run_target(target, ckpt, device):
    if not os.path.exists(ckpt):
        print(f"robustness_perturbations {target}: no ckpt {ckpt}", flush=True); return None
    builder = build_kws if target == "kws" else build_ts
    metric = "acc" if target == "kws" else "ppl"
    rows = []

    def fresh():
        return builder(ckpt, device)

    # baseline (unperturbed analytic) on the FULL validation set: this is the
    # reference for the coupling- and state-noise panels, whose perturbed metrics
    # are also computed on the full set. The finite-settling panel uses its own
    # matched 512-example subset baseline (the robustness_frequency_disorder baseline);
    # see figures/robustness_grid.py, which pairs each panel with the reference on
    # that panel's own evaluation set.
    key = f"{target}_baseline"
    if not harness.exists(EXP, key):
        m, val = fresh()
        base = _eval(target, m, val, device)
        harness.save_result(EXP, key, {"target": target, "type": "baseline",
                                        metric: base})
        del m, val; harness.free_memory(device)
        print(f"DONE {key}: {metric}={base:.4f}", flush=True)

    # (a) coupling, (b) state
    for ptype, cfgkey in [("coupling", "coupling_s"), ("state", "state_s")]:
        for s in SCALES:
            for draw in range(N_DRAWS_NOISE):
                key = f"{target}_{ptype}_s{s:g}_d{draw}"
                if harness.exists(EXP, key):
                    continue
                m, val = fresh()
                swap_lohe(m, **{cfgkey: s}, rng_seed=1000 * draw + int(s * 1000))
                val_metric = _eval(target, m, val, device)
                harness.save_result(EXP, key, {
                    "target": target, "type": ptype, "scale": s, "draw": draw,
                    metric: val_metric})
                del m, val; harness.free_memory(device)
                print(f"DONE {key}: {metric}={val_metric:.4f}", flush=True)

    # (c) finite settling (RK45, CPU subset)
    for Tset in SETTLE_T:
        for draw in range(N_DRAWS_INIT):
            key = f"{target}_settle_T{Tset:g}_d{draw}"
            if harness.exists(EXP, key):
                continue
            m, val = fresh()
            m = m.to("cpu")  # RK45 path is numpy/CPU
            swap_lohe(m, ode_T=Tset, rng_seed=7000 * draw + int(Tset * 10))
            val_metric = _eval(target, m, val, torch.device("cpu"), ode_subset=True)
            harness.save_result(EXP, key, {
                "target": target, "type": "settle", "T": Tset, "draw": draw,
                "subset": ODE_MAX_KWS if target == "kws" else ODE_MAX_TS,
                metric: val_metric})
            del m, val; harness.free_memory(device)
            print(f"DONE {key}: {metric}={val_metric:.4f}", flush=True)
    return metric


def _aggregate_and_plot():
    """Collect all robustness_perturbations JSONs into CSV + one PDF per (target, perturbation type)."""
    import glob
    import json
    recs = []
    for p in glob.glob(os.path.join(OUTDIR, "*.json")):
        with open(p) as f:
            recs.append(json.load(f))
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(FIGDIR, exist_ok=True)
    csv_path = os.path.join(OUTDIR, "perturbation_grid.csv")
    fields = ["target", "type", "scale", "T", "draw", "acc", "ppl", "subset"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            w.writerow(r)

    for target in ("kws", "ts"):
        metric = "acc" if target == "kws" else "ppl"
        base = [r[metric] for r in recs
                if r["target"] == target and r["type"] == "baseline"]
        base = base[0] if base else None
        # The settle arm is evaluated on the matched subset, so it needs that panel's own
        # reference -- the robustness_frequency_disorder baseline -- not the full-val one
        # above. Drawing the full-val line here is the mispairing the comment in
        # run_target() warns about; figures/robustness_grid.py pairs them correctly.
        base_settle = None
        if harness.reference_exists("robustness_frequency_disorder", f"{target}_baseline"):
            base_settle = harness.load_reference(
                "robustness_frequency_disorder", f"{target}_baseline").get(metric)
        for ptype, xkey in [("coupling", "scale"), ("state", "scale"), ("settle", "T")]:
            pts = {}
            for r in recs:
                if r["target"] == target and r["type"] == ptype:
                    pts.setdefault(r[xkey], []).append(r[metric])
            if not pts:
                continue
            xs = sorted(pts)
            means = [np.mean(pts[x]) for x in xs]
            stds = [np.std(pts[x]) for x in xs]
            fig, ax = plt.subplots(figsize=(5, 3.2))
            ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3)
            ref = base_settle if ptype == "settle" else base
            if ref is not None:
                ax.axhline(ref, color="gray", ls="--", lw=1,
                           label="unperturbed (matched subset)" if ptype == "settle"
                                 else "unperturbed")
            ax.set_xlabel("noise scale s" if ptype != "settle" else "settling time T")
            ax.set_ylabel(metric.upper())
            ax.set_title(f"{target.upper()} — {ptype}")
            if ptype != "settle":
                ax.set_xscale("log")
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(os.path.join(FIGDIR, f"{target}_{ptype}.pdf"))
            plt.close(fig)
    print(f"robustness_perturbations aggregated -> {csv_path} and {FIGDIR}/*.pdf", flush=True)


def main():
    device = harness.pick_device("mps")
    print(f"robustness_perturbations device={device}", flush=True)
    kws_ckpt = os.path.join(RROOT, "kws_position_matched", "ckpt", "osc_none_s0_final.pt")
    ts_ckpt = os.path.join(RROOT, "lm_coupling_function", "ckpt", "ts_softplus_s0.pt")
    ran_kws = run_target("kws", kws_ckpt, device)
    ran_ts = run_target("ts", ts_ckpt, device)
    if ran_kws is None and ran_ts is None:
        print("robustness_perturbations SKIPPED: neither checkpoint is present "
              "(these are not shipped); nothing was recomputed. Retrain with the "
              "included configs to regenerate the raw metrics.", flush=True)
        sys.exit(1)
    _aggregate_and_plot()
    print("robustness_perturbations COMPLETE", flush=True)


if __name__ == "__main__":
    main()
