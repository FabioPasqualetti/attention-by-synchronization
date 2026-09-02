"""Integrator cross-validation (CPU-only; safe to run alongside MPS training).

For one trained KWS checkpoint and one TinyStories checkpoint, integrate the Lohe
dynamics of the first attention layer from IDENTICAL random initial conditions to T=30 with:
  (a) stock Euler  oscillator_attention.ode.lohe_ode_steps (dt=0.05, N=600)
  (b) fixed RK45   training.lohe_probe._integrate_ode_rk45 (scipy solve_ivp)
and compare the final unit states to each other and to the analytic fixed point h/||h||.
Reports max/median angular deviation (deg) and the max attention-weight difference.

TinyStories val is loaded directly from the dev-cache pickle (val_chunks.pkl, ~10 MB) to
avoid loading the 4.67M-chunk train set (which would duplicate ~7 GB and pressure lm_coupling_function-TS).
"""
import math
import os
import pickle
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training import paths  # noqa: E402
paths.ensure_paths()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from oscillator_attention.ode import lohe_ode_steps  # noqa: E402
from training.lohe_probe import _integrate_ode_rk45  # noqa: E402

DEV = torch.device("cpu")
RROOT = paths.runs_root()


def layer_h(mod, x, pad=None):
    """Replicate LoheAttention driving h = sum_j W_ij anchor_j for one module."""
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
    return h, anc, W


def attn_from(xstar, anc, p, causal_mask=None):
    cos = torch.einsum('bhid,hjd->bhij', xstar, anc)
    a = (1.0 + cos).clamp(min=0.0)
    if p != 1:
        a = a ** p
    if causal_mask is not None:
        a = a * causal_mask
    return a / a.sum(-1, keepdim=True).clamp(min=1e-8)


def angdev_deg(a, b, mask=None):
    """Angular deviation (deg) between unit vectors along last dim; mask (B,H,T) bool."""
    c = (a * b).sum(-1).clamp(-1, 1)
    d = torch.rad2deg(torch.arccos(c))
    if mask is not None:
        d = d[mask]
    else:
        d = d.reshape(-1)
    return float(d.max()), float(d.median())


def capture_input(model, target):
    box = {}
    h = target.register_forward_pre_hook(lambda m, a: box.__setitem__("x", a[0]))
    return box, h


def run_case(name, model, target, x_batch, pad, p):
    box, handle = capture_input(model, target)
    with torch.no_grad():
        if pad is None:
            model(x_batch)
        else:
            model(x_batch, padding_mask=pad)
    handle.remove()
    xin = box["x"].detach()
    with torch.no_grad():
        h, anc, W = layer_h(target, xin, pad)
        h, anc = h.detach(), anc.detach()
    B, H, T, D = h.shape
    # shared random init on the sphere
    g = torch.Generator().manual_seed(12345)
    x0 = torch.randn(B, H, T, D, generator=g)
    x0 = F.normalize(x0, dim=-1)
    # (a) Euler, (b) RK45, analytic
    x_euler = lohe_ode_steps(x0.clone(), h, N=600, dt=0.05)
    hf = h.reshape(-1, D).numpy(); x0f = x0.reshape(-1, D).numpy()
    x_rk45 = torch.from_numpy(_integrate_ode_rk45(hf, x0f, 30.0)).float().reshape(B, H, T, D)
    x_rk45 = F.normalize(x_rk45, dim=-1)
    x_an = F.normalize(h, dim=-1)

    mask = None
    if pad is not None:
        mask = (~pad).view(B, 1, T).expand(B, H, T)
    cm = target._causal_mask[:T, :T] if getattr(target, "causal", False) else None

    pairs = {
        "Euler vs RK45":     (x_euler, x_rk45),
        "Euler vs analytic": (x_euler, x_an),
        "RK45 vs analytic":  (x_rk45, x_an),
    }
    # degenerate tokens: driving norm ||h|| < 0.01 (analytic fixed point ill-defined there).
    hnorm = h.norm(dim=-1)                              # (B, H, T)
    valid = mask if mask is not None else torch.ones(B, H, T, dtype=torch.bool)
    degenerate = hnorm < 0.01
    keep = valid & (~degenerate)                        # non-degenerate valid tokens
    n_valid = int(valid.sum().item())
    n_excl = int((valid & degenerate).sum().item())
    frac_excl = n_excl / max(n_valid, 1)

    print(f"\n=== {name} (d_osc={D}, {B}x{H}x{T} oscillators; "
          f"degenerate excluded {n_excl}/{n_valid} = {100*frac_excl:.2f}%) ===")
    metrics = {}
    for label, (a, b) in pairs.items():
        rmx, rmd = angdev_deg(a, b, valid)              # raw: all valid tokens
        mmx, mmd = angdev_deg(a, b, keep) if keep.any() else (0.0, 0.0)  # masked: non-degenerate
        aa, ab = attn_from(a, anc, p, cm), attn_from(b, anc, p, cm)
        dattn_raw = (aa - ab).abs()[valid].max().item()
        dattn_msk = (aa - ab).abs()[keep].max().item() if keep.any() else 0.0
        print(f"  {label:20s}: raw med={rmd:.4f}° max={rmx:.4f}° | "
              f"masked med={mmd:.4f}° max={mmx:.4f}°")
        metrics[label] = {
            "raw_median_deg": round(float(rmd), 4), "raw_max_deg": round(float(rmx), 4),
            "masked_median_deg": round(float(mmd), 4), "masked_max_deg": round(float(mmx), 4),
            "raw_max_abs_delta_attn": float(dattn_raw),
            "masked_max_abs_delta_attn": float(dattn_msk)}
    return {"name": name, "d_osc": int(D), "n_oscillators": f"{B}x{H}x{T}",
            "degenerate_mask_criterion": "norm_h < 0.01",
            "n_valid_tokens": n_valid, "n_excluded_degenerate": n_excl,
            "fraction_excluded": round(frac_excl, 4), "pairs": metrics}


def kws_case():
    from oscillator_attention.kws_pe_model import KWSTransformerPE
    from training.kws_data import cached_loaders
    e1 = paths.load_py(os.path.join(paths.REPO_ROOT, "keyword_spotting", "position_matched.py"), "e1x")
    m = KWSTransformerPE(attn_type="lohe", p=1, pe="none", **e1.ARCH)
    m.load_state_dict(torch.load(os.path.join(RROOT, "kws_position_matched", "ckpt", "osc_none_s0_final.pt"),
                                 map_location="cpu", weights_only=False))
    m.eval()
    _, val, _, _ = cached_loaders(0, batch_size=64)
    feats, _ = next(iter(val))
    return run_case("KWS osc d2", m, m.layers[0].attn, feats, None, p=1)


def ts_case():
    from oscillator_attention.sigma_models import LoheLMSigma
    cache = os.path.expanduser(os.path.join(paths.cache_root(), "tinystories"))
    vocab = torch.load(os.path.join(cache, "vocab.pt"), weights_only=False)["vocab"]
    with open(os.path.join(cache, "val_chunks.pkl"), "rb") as f:
        val = pickle.load(f)[:8]
    maxlen = 128
    toks = torch.zeros(len(val), maxlen, dtype=torch.long)
    for i, c in enumerate(val):
        c = c[:maxlen]; toks[i, :len(c)] = torch.tensor(c, dtype=torch.long)
    pad = (toks == 0)
    m = LoheLMSigma(vocab_size=len(vocab), d_osc=8, sigma="softplus",
                    d_model=128, n_heads=4, n_layers=2, d_ff=512, max_seq_len=128, dropout=0.1)
    m.load_state_dict(torch.load(os.path.join(RROOT, "lm_coupling_function", "ckpt", "ts_softplus_s0.pt"),
                                 map_location="cpu", weights_only=False))
    m.eval()
    return run_case("TinyStories osc d8", m, m.layers[0].attn, toks, pad, p=1)


def main():
    import json
    out = {"description": "Integrator independence: Euler vs RK45 vs analytic fixed point; "
           "angle deviation (deg) and max|delta attn| between integrators on trained models.",
           "cases": []}
    import sys
    for label, fn in (("KWS", kws_case), ("TinyStories", ts_case)):
        try:
            out["cases"].append(fn())
        except Exception as e:
            print(f"[skip {label}] {type(e).__name__}: {e}")
            out["cases"].append({"name": label, "skipped": f"{type(e).__name__}: {e}"})
    d = os.path.join(RROOT, "lm_integrator_independence")
    os.makedirs(d, exist_ok=True)
    dest = os.path.join(d, "integrator_independence.json")
    failed = [c for c in out["cases"] if "skipped" in c]
    if failed:
        # This driver needs trained checkpoints (not shipped; regenerated by
        # training). On a clean clone the cases raise, and overwriting the
        # prior result with error stubs would destroy work this run did not redo.
        # Refuse: write to a scratch sibling and exit non-zero instead. (dest is
        # under runs/, so this protects the driver's own output, not the
        # reference tree -- drivers cannot write there at all.)
        alt = dest + ".partial"
        with open(alt, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\n{len(failed)} case(s) failed {[c['name'] for c in failed]}; refusing to "
              f"overwrite {dest}. Wrote {alt} and exiting non-zero.")
        sys.exit(1)
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
