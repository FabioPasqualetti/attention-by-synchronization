"""lm_dimensional_bottleneck rank-truncation on the FULL TinyStories validation set (not the 128-seq subset).

Re-runs the post-hoc SVD rank-truncation of the softmax checkpoint
(results/lm_dimensional_bottleneck/ckpt/ts_softmax_s0_ep5.pt) at ranks r in {3,5,9,17,33},
evaluating perplexity on the ENTIRE TinyStories validation set, so all three series in the right
panel of the rank figure (truncated softmax / oscillator / reduced-Q/K softmax) are on the SAME
eval set. Reuses the truncation machinery (TruncatedSoftmax, build_softmax, eval_ppl) from
dimensional_bottleneck.py verbatim — no training, no model changes.

**This is the published truncation series.** The subset series `truncation_r*.json` written by
dimensional_bottleneck.py section (c) is superseded; see this directory's README.md.

Resumable per-r JSON -> results/lm_dimensional_bottleneck/truncation_full_val_r{r}.json.
"""
import os
import pickle
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training import harness, paths  # noqa: E402
from training.data_utils import load_tinystories  # noqa: E402
paths.ensure_paths()
import torch  # noqa: E402

# reuse the truncation machinery verbatim (TruncatedSoftmax, build_softmax, eval_ppl, constants)
e8 = paths.load_py(os.path.join(os.path.dirname(__file__), "dimensional_bottleneck.py"),
                   "dimensional_bottleneck_mod")
EXP = "lm_dimensional_bottleneck"

# --- robust SVD: on the full val set some softmax-attention rows are near-degenerate (very short /
# heavily padded sequences -> repeated singular values), and LAPACK gesdd (torch/numpy default)
# occasionally fails to converge. Scope a robust wrapper over torch.linalg.svd for THIS process
# only: on any failure, redo the batch matrix-by-matrix, falling back to the QR-based gesvd driver
# (scipy) which is stable for repeated singular values. Same mathematical SVD -> identical result;
# no change to dimensional_bottleneck.py. ---
import numpy as np  # noqa: E402
import scipy.linalg as _sla  # noqa: E402

_orig_svd = torch.linalg.svd


def _robust_svd(A, full_matrices=True, **kw):
    try:
        return _orig_svd(A, full_matrices=full_matrices, **kw)
    except Exception:
        dev, dt = A.device, A.dtype
        Ac = A.detach().cpu().numpy().astype("float64")           # (N, T, T); cast on CPU (MPS has no f64)
        Us, Ss, Vhs = [], [], []
        for m in Ac:
            try:
                u, s, vh = np.linalg.svd(m, full_matrices=full_matrices)
            except Exception:
                u, s, vh = _sla.svd(m, full_matrices=full_matrices, lapack_driver="gesvd")
            Us.append(u); Ss.append(s); Vhs.append(vh)
        def _t(x):  # float32 on CPU, then to the original device (MPS rejects float64)
            return torch.from_numpy(np.stack(x).astype("float32")).to(dev)
        return _t(Us), _t(Ss), _t(Vhs)


torch.linalg.svd = _robust_svd


def load_full_val():
    """Full TinyStories val set (every chunk) — same tokenisation as e8.load_val but no [:N_VAL]."""
    # via load_tinystories, not the raw cache: it resolves the shipped
    # data/tinystories_eval/ vocabulary, which is what indexes the released
    # checkpoints. Reading <cache>/tinystories directly fails on a clean clone.
    vocab, _, _, val = load_tinystories(max_len=e8.MAXLEN)   # ALL sequences
    ML = e8.MAXLEN
    toks = torch.zeros(len(val), ML, dtype=torch.long)
    for i, c in enumerate(val):
        c = c[:ML]
        toks[i, :len(c)] = torch.tensor(c, dtype=torch.long)
    pad = (toks == 0)
    return toks, pad, len(vocab)


def main():
    device = harness.pick_device("mps")
    toks, pad, vocab_size = load_full_val()
    print(f"lm_dimensional_bottleneck full-val truncation: device={device}, {toks.shape[0]} val sequences "
          f"(subset run used {e8.N_VAL})", flush=True)
    smx_ckpt = os.path.join(e8.CKPT, "ts_softmax_s0_ep5.pt")
    assert os.path.exists(smx_ckpt), f"missing softmax checkpoint: {smx_ckpt}"
    for r in e8.RANKS:
        key = f"truncation_full_val_r{r}"
        if harness.exists(EXP, key):
            print(f"skip {key}", flush=True)
            continue
        model = e8.build_softmax(vocab_size)
        model.load_state_dict(torch.load(smx_ckpt, map_location="cpu", weights_only=False))
        for layer in model.layers:
            layer.attn = e8.TruncatedSoftmax(layer.attn, r)   # SVD-truncate to rank r at inference
        model.to(device).eval()
        ppl = e8.eval_ppl(model, toks, pad, device)
        harness.save_result(EXP, key, {"rank": r, "ppl": ppl, "matched_d_osc": r - 1,
                                       "eval": "full_val", "n_val_seqs": int(toks.shape[0]),
                                       "ckpt": "ts_softmax_s0_ep5.pt"})
        print(f"DONE {key} (d_osc≈{r-1}): PPL={ppl:.4f}", flush=True)
        harness.free_memory(device)
    print("lm_dimensional_bottleneck full-val truncation COMPLETE", flush=True)


if __name__ == "__main__":
    main()
