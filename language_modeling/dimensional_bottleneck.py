"""lm_dimensional_bottleneck — Dimensional bottleneck: measured rank + causal truncation.

(a) Train one softmax TinyStories run (ref config, seed 0); gate val PPL == 8.544 ± 0.3.
(b) Measured per-head attention singular-value spectra over >=100 val sequences for softmax
    (from a) and oscillator d_osc in {4,8,16} (existing checkpoints only — do NOT train new
    oscillator LMs); report entropy-based effective rank + participation ratio (mean +/- std),
    vs the oscillator structural bound rank <= d_osc+1.
(c) On the softmax checkpoint, SVD-truncate each post-softmax per-head attention matrix to
    rank r in {3,5,9,17,33} (= d_osc+1 for d_osc in {2,4,8,16,32}), renormalize rows, measure
    val PPL(r) -> truncation_r{r}.json.

    NOTE: (c) evaluates on the 128-sequence subset loaded here, and is SUPERSEDED. The
    published truncation series is truncation_full_val_r{r}.json, produced by
    dimensional_bottleneck_truncation.py on the FULL validation set -- that is the series the
    paper reports and figures/rank_truncation.py reads. (c) is kept for provenance; see
    results/lm_dimensional_bottleneck/README.md.

Resumable per-run JSON.
"""
import math
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training import harness, paths  # noqa: E402
from training.data_utils import load_tinystories  # noqa: E402
paths.ensure_paths()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

EXP = "lm_dimensional_bottleneck"
CKPT = os.path.join(harness.RUNS_ROOT, EXP, "ckpt")
CACHE = os.path.expanduser(os.path.join(paths.cache_root(), "tinystories"))
SOFTMAX_REF_PPL = 8.544
GATE_TOL = 0.3
RANKS = [3, 5, 9, 17, 33]              # = d_osc+1 for d_osc in {2,4,8,16,32}
N_VAL = 128                            # >= 100 val sequences
MAXLEN = 128
ARCH = dict(d_model=128, n_heads=4, n_layers=2, d_ff=512, max_seq_len=128, dropout=0.1)
# Existing oscillator TS checkpoints (state_dicts). d2/d32 absent -> excluded (no new training).
OSC_CKPTS = {
    4:  os.path.join(paths.results_root(), "lm_5pt_runs", "checkpoints",
                     "ts_dosc4_seed0", "ts_d4_s0_ep5.pt"),
    8:  os.path.join(harness.RUNS_ROOT, "lm_coupling_function", "ckpt", "ts_softplus_s0.pt"),
    16: os.path.join(paths.results_root(), "lm_5pt_runs", "checkpoints",
                     "ts_dosc16_seed0", "ts_d16_s0_ep5.pt"),
}
# Oscillator TS PPL(d_osc) means for the overlay (paper scaling data).
OSC_PPL = {2: 10.947, 4: None, 8: 9.763, 16: None, 32: 9.106}  # d4/d16 filled from 5pt files


def load_val():
    # via load_tinystories, not the raw cache: it resolves the shipped
    # data/tinystories_eval/ vocabulary, which is what indexes the released
    # checkpoints. Reading <cache>/tinystories directly fails on a clean clone.
    vocab, _, _, val_all = load_tinystories(max_len=MAXLEN)
    val = val_all[:N_VAL]
    toks = torch.zeros(len(val), MAXLEN, dtype=torch.long)
    for i, c in enumerate(val):
        c = c[:MAXLEN]
        toks[i, :len(c)] = torch.tensor(c, dtype=torch.long)
    pad = (toks == 0)
    return toks, pad, len(vocab)


def _spectrum_stats(A):
    """A: (L,L) tensor -> (effective_rank, participation_ratio)."""
    s = torch.linalg.svdvals(A.float()).cpu().numpy()
    s = s[s > 1e-10]
    if s.size == 0:
        return 0.0, 0.0
    p = s / s.sum()
    eff = float(np.exp(-(p * np.log(p)).sum()))
    pr = float((s.sum() ** 2) / (s ** 2).sum())
    return eff, pr


@torch.no_grad()
def _osc_similarity(mod, x, pad):
    """UNMASKED oscillator similarity S_ij = (1 + x*_i·anchor_j), rank <= d_osc+1 by
    construction (structural bound). x* uses the causal coupling, but the similarity is
    evaluated for all (i,j) pairs (unmasked), so its rank is bounded."""
    B, T, _ = x.shape
    H, Dh = mod.n_heads, mod.d_head
    q = mod.W_q(x).view(B, T, H, Dh).transpose(1, 2)
    k = mod.W_k(x).view(B, T, H, Dh).transpose(1, 2)
    raw = torch.einsum('bhid,bhjd->bhij', q, k) / math.sqrt(Dh)
    W = F.softplus(raw)
    if getattr(mod, "causal", False):
        W = W * mod._causal_mask[:T, :T]
    if pad is not None:
        W = W * (~pad).float().view(B, 1, 1, T)
    anc = mod.anchors[:, :T, :]
    h = torch.einsum('bhij,hjd->bhid', W, anc)
    xstar = F.normalize(h, dim=-1, eps=1e-8)
    cos = torch.einsum('bhid,hjd->bhij', xstar, anc)   # (B,H,T,T) UNMASKED
    # Structural similarity BEFORE the readout clamp/power: (1 + x*·anchor) = ones + X* @ Anchor^T
    # has rank <= d_osc+1 exactly. (The readout's clamp(min=0) is nonlinear and can lift the
    # rank when trained anchors are non-unit; the causal mask lifts it further — both reported
    # via the realized causal-attn effective rank.)
    return 1.0 + cos


@torch.no_grad()
def capture_spectra(model, toks, pad, is_softmax):
    attn_modules = [layer.attn for layer in model.layers]
    H = ARCH["n_heads"]
    effs, prs, sim_effs = [], [], []
    outs, ins = [], []
    hs = [am.register_forward_hook(lambda m, i, o: outs.append(o[1].detach()))
          for am in attn_modules]
    phs = [am.register_forward_pre_hook(lambda m, a: ins.append(a[0].detach()))
           for am in attn_modules]
    B = 32
    for start in range(0, toks.shape[0], B):
        outs.clear(); ins.clear()
        tb = toks[start:start + B]; pb = pad[start:start + B]
        model(tb, padding_mask=pb)
        lengths = (~pb).sum(1).tolist()
        for li, a in enumerate(outs):                 # realized (causal) attention per layer
            if a.dim() == 3:                          # softmax (B*H,T,T)
                a = a.view(tb.shape[0], H, MAXLEN, MAXLEN)
            sim = None if is_softmax else _osc_similarity(attn_modules[li], ins[li], pb)
            for b in range(a.shape[0]):
                L = int(lengths[b])
                if L < 2:
                    continue
                for h in range(a.shape[1]):
                    e, pr = _spectrum_stats(a[b, h, :L, :L])
                    effs.append(e); prs.append(pr)
                    if sim is not None:
                        se, _ = _spectrum_stats(sim[b, h, :L, :L])
                        sim_effs.append(se)
    for hd in hs + phs:
        hd.remove()
    return effs, prs, sim_effs


def _agg(vals):
    v = np.asarray(vals, dtype=float)
    return dict(mean=float(v.mean()), std=float(v.std()), n=int(v.size),
                min=float(v.min()), max=float(v.max()))


def build_osc(d_osc, vocab_size):
    from oscillator_attention.sigma_models import LoheLMSigma
    return LoheLMSigma(vocab_size=vocab_size, d_osc=d_osc, sigma="softplus", **ARCH)


def build_softmax(vocab_size):
    ts = paths.load_py(os.path.join(paths.REPO_ROOT, "training",
                                    "train_tinystories.py"), "ts_train_mod")
    return ts.make_softmax_lm(vocab_size)


def spectra_one(name, model, ckpt, toks, pad, vocab_size, d_osc, is_softmax):
    """Measure one checkpoint's spectra. Returns "present" (committed result already there),
    "computed" (measured now), or "no_ckpt" (skipped: checkpoints are not shipped)."""
    key = f"spectra_{name}"
    if harness.exists(EXP, key):
        print(f"lm_dimensional_bottleneck spectra {name}: result already present — skip",
              flush=True)
        return "present"
    if not os.path.exists(ckpt):
        print(f"lm_dimensional_bottleneck spectra {name}: no ckpt {ckpt} — SKIPPED",
              flush=True)
        return "no_ckpt"
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
    model.eval()
    effs, prs, sim_effs = capture_spectra(model, toks, pad, is_softmax)
    payload = {"name": name, "mechanism": "softmax" if is_softmax else "oscillator",
               "d_osc": d_osc, "bound_rank": (None if is_softmax else d_osc + 1),
               "effective_rank_causal_attn": _agg(effs), "participation_ratio": _agg(prs),
               "effective_rank_unmasked_similarity": (_agg(sim_effs) if sim_effs else None),
               "n_val_seqs": int(toks.shape[0]), "ckpt": ckpt}
    harness.save_result(EXP, key, payload)
    b = payload["bound_rank"]
    sim = payload["effective_rank_unmasked_similarity"]
    msg = (f"DONE spectra {name}: causal-attn eff_rank="
           f"{payload['effective_rank_causal_attn']['mean']:.2f}"
           f"±{payload['effective_rank_causal_attn']['std']:.2f} "
           f"PR={payload['participation_ratio']['mean']:.2f}")
    if sim is not None:
        msg += f" | unmasked-similarity eff_rank={sim['mean']:.2f} (bound d_osc+1={b})"
    print(msg, flush=True)
    return "computed"


# ── (c) SVD-truncated softmax ────────────────────────────────────────────────

class TruncatedSoftmax(torch.nn.Module):
    """Wrap a stock SoftmaxAttention; SVD-truncate the post-softmax attention to rank r,
    clamp negatives, renormalize rows to sum 1, then aggregate values."""

    def __init__(self, src, r):
        super().__init__()
        self.src = src
        self.r = r

    def forward(self, x, padding_mask=None, causal=True):
        m = self.src
        B, T, _ = x.shape
        H, dh = m.n_heads, m.d_head
        Q = m.W_q(x).view(B, T, H, dh).transpose(1, 2).reshape(B * H, T, dh)
        K = m.W_k(x).view(B, T, H, dh).transpose(1, 2).reshape(B * H, T, dh)
        V = m.W_v(x).view(B, T, H, dh).transpose(1, 2).reshape(B * H, T, dh)
        logits = torch.bmm(Q, K.transpose(1, 2)) / m.scale
        if causal:
            logits = logits.masked_fill(m._causal_mask[:T, :T].unsqueeze(0), float("-inf"))
        if padding_mask is not None:
            mask = padding_mask.unsqueeze(1).expand(B, H, T).reshape(B * H, 1, T)
            logits = logits.masked_fill(mask, float("-inf"))
        attn = torch.softmax(logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        # SVD-truncate each (B*H) TxT matrix to rank r
        U, S, Vh = torch.linalg.svd(attn.float(), full_matrices=False)
        r = min(self.r, S.shape[-1])
        S2 = S.clone(); S2[..., r:] = 0.0
        trunc = (U * S2.unsqueeze(-2)) @ Vh
        trunc = trunc.clamp(min=0.0)
        trunc = trunc / trunc.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        trunc = trunc.to(V.dtype)
        out = torch.bmm(trunc, V)
        out = out.view(B, H, T, dh).transpose(1, 2).reshape(B, T, H * dh)
        return m.W_o(out), trunc


@torch.no_grad()
def eval_ppl(model, toks, pad, device):
    model.eval()
    ce = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    tot_loss = tot_tok = 0
    B = 32
    for s in range(0, toks.shape[0], B):
        x = toks[s:s + B].to(device); p = pad[s:s + B].to(device)
        logits = model(x, padding_mask=p)
        tgt = x[:, 1:].contiguous(); lg = logits[:, :-1].contiguous()
        tot_loss += ce(lg.view(-1, lg.size(-1)), tgt.view(-1)).item()
        tot_tok += (tgt != 0).sum().item()
    return math.exp(tot_loss / max(tot_tok, 1))


# Marker written with every (c) result so the superseded subset series can never be mistaken
# for the published full-validation-set one. Kept in sync with the committed JSONs.
SUPERSEDED_NOTE = (
    "SUPERSEDED: 128-sequence validation subset. The published series is "
    "truncation_full_val_r*.json (full 47,385-sequence validation set), which is what "
    "figures/rank_truncation.py reads and what the paper reports. Retained for provenance "
    "only -- do not compare these PPLs with the oscillator or rank-match series, which are "
    "evaluated on the full set. See this directory's README.md.")


def truncation(vocab_size, toks, pad, device):
    """Returns "present" if every rank is already committed, "computed" if any was measured now,
    or "no_ckpt" if the softmax checkpoint is absent."""
    smx_ckpt = os.path.join(CKPT, "ts_softmax_s0_ep5.pt")
    if not os.path.exists(smx_ckpt):
        if all(harness.exists(EXP, f"truncation_r{r}") for r in RANKS):
            print("lm_dimensional_bottleneck (c): results already present — skip truncation",
                  flush=True)
            return "present"
        print("lm_dimensional_bottleneck (c): no softmax ckpt — truncation SKIPPED", flush=True)
        return "no_ckpt"
    status = "present"
    for r in RANKS:
        key = f"truncation_r{r}"
        if harness.exists(EXP, key):
            continue
        model = build_softmax(vocab_size)
        model.load_state_dict(torch.load(smx_ckpt, map_location="cpu", weights_only=False))
        for layer in model.layers:
            layer.attn = TruncatedSoftmax(layer.attn, r)
        model.to(device).eval()
        ppl = eval_ppl(model, toks, pad, device)
        harness.save_result(EXP, key, {"rank": r, "ppl": ppl,
                                       "matched_d_osc": r - 1,
                                       "osc_ppl_at_d": OSC_PPL.get(r - 1),
                                       "eval": "val_subset_128",
                                       "n_val_seqs": int(toks.shape[0]),
                                       "status": "superseded",
                                       "superseded_by": f"truncation_full_val_r{r}.json",
                                       "note": SUPERSEDED_NOTE})
        print(f"DONE truncation r={r} (d_osc≈{r-1}): PPL={ppl:.3f}", flush=True)
        status = "computed"
        harness.free_memory(device)
    return status


def _fill_osc_ppl():
    import glob, json
    for d, patt in [(4, "ts_dosc4_seed*.json"), (16, "ts_dosc16_seed*.json")]:
        vals = []
        for p in glob.glob(os.path.join(paths.results_root(), "lm_5pt_runs", patt)):
            try:
                vals.append(json.load(open(p))["val_ppl"])
            except Exception:
                pass
        if vals:
            OSC_PPL[d] = float(np.mean(vals))


def main():
    device = harness.pick_device("mps")
    print(f"lm_dimensional_bottleneck device={device}", flush=True)
    _fill_osc_ppl()
    toks, pad, vocab_size = load_val()

    # (a) softmax train + gate
    if not harness.exists(EXP, "softmax_train"):
        ts = paths.load_py(os.path.join(paths.REPO_ROOT, "training",
                                        "train_tinystories.py"), "ts_train_mod")
        os.makedirs(CKPT, exist_ok=True)
        ts.CKPT_DIR = CKPT
        t0 = time.time()
        r = ts.run_one(0, 0, device)
        ppl = r["val_ppl"]
        gate_ok = abs(ppl - SOFTMAX_REF_PPL) <= GATE_TOL
        harness.save_result(EXP, "softmax_train", {
            "val_ppl": ppl, "ref_ppl": SOFTMAX_REF_PPL, "gate_ok": gate_ok,
            "wall_sec": round(time.time() - t0, 1)})
        print(f"lm_dimensional_bottleneck (a) softmax PPL={ppl:.3f} gate_ok={gate_ok}"
              + ("" if gate_ok else "  GATE FAIL — skipping softmax spectra/truncation"),
              flush=True)
        harness.free_memory(device)
    gate_ok = harness.load_result(EXP, "softmax_train").get("gate_ok", False) \
        if harness.exists(EXP, "softmax_train") else False

    # (b) spectra: oscillator (always) + softmax (if gate passed)
    status = {}
    for d_osc, ck in OSC_CKPTS.items():
        status[f"spectra_osc_d{d_osc}"] = spectra_one(
            f"osc_d{d_osc}", build_osc(d_osc, vocab_size), ck, toks, pad,
            vocab_size, d_osc, is_softmax=False)
        harness.free_memory(device)
    if gate_ok:
        status["spectra_softmax"] = spectra_one(
            "softmax", build_softmax(vocab_size),
            os.path.join(CKPT, "ts_softmax_s0_ep5.pt"), toks, pad,
            vocab_size, None, is_softmax=True)

    # (c) truncation (needs softmax ckpt + gate)
    if gate_ok:
        status["truncation"] = truncation(vocab_size, toks, pad, device)

    # Report honestly. Oscillator/softmax checkpoints are not shipped, so on a clean clone the
    # committed JSONs satisfy every section and nothing is recomputed — that is success. A
    # section whose result is missing AND whose checkpoint is absent was genuinely skipped;
    # never print COMPLETE over that.
    skipped = sorted(k for k, v in status.items() if v == "no_ckpt")
    if skipped:
        print("lm_dimensional_bottleneck INCOMPLETE: no result and no checkpoint for "
              + ", ".join(skipped)
              + ". Checkpoints are not shipped; retrain with the included configs "
                "(see README) to regenerate these sections.", flush=True)
        sys.exit(1)
    print("lm_dimensional_bottleneck COMPLETE", flush=True)


if __name__ == "__main__":
    main()
