"""Dimensional bottleneck at the hardware-minimal case d_osc=2 (TinyStories).

Train oscillator TS d_osc=2 seed 0 (ref config), gate PPL ~= paper d2 scaling point (10.947 +/-0.3),
then measure its per-head attention spectra with the same method as dimensional_bottleneck.py and
check the rank <= d_osc+1 = 3 structural bound. Also records the measured d2 PPL for the r=3 point
of the truncation table. Resumable.
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training import harness, paths  # noqa: E402
paths.ensure_paths()

import torch  # noqa: E402

EXP = "lm_dimensional_bottleneck"
REPO = paths.REPO_ROOT
CKPT = os.path.join(harness.RUNS_ROOT, EXP, "ckpt")
D2_REF_PPL = 10.947
GATE_TOL = 0.3

_e8 = paths.load_py(os.path.join(REPO, "language_modeling", "dimensional_bottleneck.py"),
                    "dimensional_bottleneck_mod")


def train_d2(device):
    if harness.exists(EXP, "train_d2_s0"):
        return harness.load_result(EXP, "train_d2_s0")
    ts = paths.load_py(os.path.join(REPO, "training", "train_tinystories.py"),
                       "ts_train_mod")
    os.makedirs(CKPT, exist_ok=True)
    ts.CKPT_DIR = CKPT
    t0 = time.time()
    r = ts.run_one(2, 0, device)
    gate_ok = abs(r["val_ppl"] - D2_REF_PPL) <= GATE_TOL
    payload = {"d_osc": 2, "seed": 0, "val_ppl": r["val_ppl"], "ref_ppl": D2_REF_PPL,
               "gate_ok": bool(gate_ok), "wall_sec": round(time.time() - t0, 1)}
    harness.save_result(EXP, "train_d2_s0", payload)
    harness.free_memory(device)
    tag = "PASS" if gate_ok else "FAIL"
    print(f"lm_dimensional_bottleneck d_osc=2 PPL={r['val_ppl']:.3f} (ref {D2_REF_PPL}) GATE {tag}", flush=True)
    return payload


def spectra_d2():
    if harness.exists(EXP, "spectra_osc_d2"):
        return
    ckpt = os.path.join(CKPT, "ts_d2_s0_ep5.pt")
    if not os.path.exists(ckpt):
        # stock train_tinystories saves as ts_d{d_osc}_s{seed}_ep5.pt
        cands = [p for p in os.listdir(CKPT) if p.startswith("ts_d2_s0") and p.endswith(".pt")] if os.path.isdir(CKPT) else []
        if cands:
            ckpt = os.path.join(CKPT, sorted(cands)[-1])
        else:
            print(f"lm_dimensional_bottleneck spectra: no d2 ckpt in {CKPT}", flush=True); return
    toks, pad, vocab_size = _e8.load_val()
    model = _e8.build_osc(2, vocab_size)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
    model.eval()
    effs, prs, sim_effs = _e8.capture_spectra(model, toks, pad, is_softmax=False)
    payload = {"name": "osc_d2", "mechanism": "oscillator", "d_osc": 2, "bound_rank": 3,
               "effective_rank_causal_attn": _e8._agg(effs),
               "participation_ratio": _e8._agg(prs),
               "effective_rank_unmasked_similarity": (_e8._agg(sim_effs) if sim_effs else None),
               "n_val_seqs": int(toks.shape[0]), "ckpt": ckpt}
    harness.save_result(EXP, "spectra_osc_d2", payload)
    sim = payload["effective_rank_unmasked_similarity"]
    print(f"lm_dimensional_bottleneck DONE spectra osc_d2: causal-attn eff_rank="
          f"{payload['effective_rank_causal_attn']['mean']:.2f} | unmasked-similarity eff_rank="
          f"{sim['mean']:.2f} (bound d_osc+1=3)", flush=True)


def main():
    device = harness.pick_device("mps")
    print(f"lm_dimensional_bottleneck device={device}", flush=True)
    tr = train_d2(device)
    if not tr.get("gate_ok"):
        print("lm_dimensional_bottleneck: GATE_FAIL noted; still computing spectra.", flush=True)
    spectra_d2()
    print("lm_dimensional_bottleneck COMPLETE", flush=True)


if __name__ == "__main__":
    main()
