"""ODE convergence verification on TinyStories -- the producer for Table 6 (tab:convergence)
and the right-hand panel of Figure 7 (fig:convergence).

For each trained TinyStories oscillator model (d_osc in {2, 8, 32}) this integrates the Lohe
ODE from random initial conditions and measures how often it reaches the analytic fixed
point. Per the paper: RK45 via scipy.solve_ivp with rtol = atol = 1e-4, T_max = 30, and 5
random initializations of z(0) per oscillator. Convergence is the final-time error
err = ||z(T_max) - z*||_2; "Converged" counts err < 0.01 and "apparent-antipodal" counts
err > 0.1 (slow convergence consistent with initialization near -z*).

Sample: N_SEQS is compared against a running count of sequences that advances a batch at a
time, so with batch_size 16 the loop consumes TWO batches -- 32 validation chunks -- before
it stops. The oscillator count is (valid position, head) pairs of the LAST attention layer:
2,895 positions x 4 heads = 11,580, and 11,580 x 5 initializations = 57,900 ODE solves.
This is preserved exactly as the published numbers were produced.

The d_osc=8 and d_osc=32 checkpoints are not distributed; train them with
training/train_tinystories.py and pass --ckptdir. The d_osc=2 checkpoint ships, so that row
of Table 6 needs no arguments and no download.

Run:  python convergence/ts_convergence.py --d_osc 2          # shipped checkpoint
      python convergence/ts_convergence.py --ckptdir <dir>    # all three
"""
import argparse
import json
import math
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training import harness, paths  # noqa: E402
paths.ensure_paths()
from training.data_utils import load_tinystories, SeqDataset  # noqa: E402
from oscillator_attention import LoheLanguageTransformer      # noqa: E402

EXP = "convergence"
CKPT_NAME = {2: "TS2_d2", 8: "TS3_d8", 32: "TS4_d32"}
# The d_osc=2 checkpoint ships, under the name Figure 5 uses for it. Look there when a
# --ckptdir copy is not present, so `--d_osc 2` needs no arguments on a clean clone.
SHIPPED_CKPT = {2: os.path.join(paths.REPO_ROOT, "figures", "lib", "models", "TS_d2.pt")}


def resolve_ckpt(d_osc, ckptdir, ckpt):
    """Path to the checkpoint for d_osc, or None. --ckpt wins, then --ckptdir, then the
    shipped copy."""
    if ckpt:
        return ckpt
    if ckptdir:
        p = os.path.join(ckptdir, CKPT_NAME[d_osc] + ".pt")
        if os.path.exists(p):
            return p
    return SHIPPED_CKPT.get(d_osc)
N_SEQS, BATCH, N_RAND_INIT = 20, 16, 5
SEQ_NOISE = 0.05          # jitter on the previous token's fixed point, sequential init
T_MAX, TOL = 30.0, 1e-4
ARCH = dict(d_model=128, n_heads=4, n_layers=2, d_ff=512, max_seq_len=128, dropout=0.0)


@torch.no_grad()
def compute_h_and_analytic(model, tokens, pad_mask):
    """Driving vector h and analytic fixed point x* = normalize(h), for every
    (head, position) of the LAST attention layer (h_out is reassigned per layer)."""
    B, T = tokens.shape
    h_emb = model.pos_enc(model.embedding(tokens))
    h_out = None
    for layer in model.layers:
        attn = layer.attn
        normed = layer.norm1(h_emb)
        H, D_h = attn.n_heads, attn.d_head
        q = attn.W_q(normed).view(B, T, H, D_h).transpose(1, 2)
        k = attn.W_k(normed).view(B, T, H, D_h).transpose(1, 2)
        W = F.softplus(torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(D_h))
        W = W * torch.tril(torch.ones(T, T))
        if pad_mask is not None:
            W = W * (~pad_mask).float().view(B, 1, 1, T)
        anc = attn.anchors[:, :T, :]
        h_out = torch.einsum("bhij,hjd->bhid", W, anc)
        x_star = F.normalize(h_out, dim=-1, eps=1e-8)
        cos = torch.einsum("bhid,hjd->bhij", x_star, anc)
        aw = ((1.0 + cos).clamp(min=0.0) * torch.tril(torch.ones(T, T)))
        aw = aw / aw.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        v = attn.W_v(normed).view(B, T, H, D_h).transpose(1, 2)
        out = torch.einsum("bhij,bhjd->bhid", aw, v)
        h_emb = h_emb + attn.W_o(out.transpose(1, 2).contiguous().view(B, T, H * D_h))
        h_emb = h_emb + layer.ffn(layer.norm2(h_emb))
    return h_out, F.normalize(h_out, dim=-1, eps=1e-8)


def _rhs_single(t, y, h):
    """dz/dt = (I - z z^T) h, with z the projection of y onto the sphere."""
    x = y / max(np.linalg.norm(y), 1e-10)
    return h - np.dot(x, h) * x


def integrate(h, x0):
    """One solve_ivp per oscillator -- batching was counterproductive, since a single
    near-antipodal problem drags the shared step size down for every other one."""
    M = h.shape[0]
    xf = np.empty_like(x0)
    total_nfev, n_failed = 0, 0
    for i in range(M):
        hi = h[i]
        # RHS renormalises y before projecting. On the sphere this equals
        # h - (y.h)y, but the solver steps off the sphere between stages, so the two
        # forms differ there and give different step-size control -- and therefore
        # different nfev. This is the form the published numbers were produced with.
        sol = solve_ivp(_rhs_single, (0.0, T_MAX), x0[i], method="RK45",
                        rtol=TOL, atol=TOL, args=(hi,), dense_output=False)
        y = sol.y[:, -1]
        n = np.linalg.norm(y)
        xf[i] = y / max(n, 1e-10)
        total_nfev += int(sol.nfev)
        n_failed += 0 if sol.success else 1
    return xf, total_nfev, n_failed


def rand_sphere(shape):
    v = np.random.randn(*shape)
    return v / np.linalg.norm(v, axis=-1, keepdims=True).clip(min=1e-10)


def verify(model, loader, d_osc):
    errors, nfevs = [], []
    n_failed_total, n_total, seq_count = 0, 0, 0
    H = model.layers[0].attn.n_heads
    for batch in loader:
        if seq_count >= N_SEQS:
            break
        tokens, pad_mask = batch["tokens"], batch["pad_mask"]
        B = tokens.size(0)
        h, xa = compute_h_and_analytic(model, tokens, pad_mask)
        BH, T = B * H, tokens.size(1)
        pm = pad_mask.unsqueeze(1).expand(B, H, T).reshape(BH, T).numpy()
        valid = ~pm
        h_valid = h.reshape(BH, T, d_osc).numpy()[valid]
        xa_valid = xa.reshape(BH, T, d_osc).numpy()[valid]
        N_valid = h_valid.shape[0]
        for _ in range(N_RAND_INIT):
            xf, nfev, n_fail = integrate(h_valid, rand_sphere((N_valid, d_osc)))
            errors.append(np.linalg.norm(xf - xa_valid, axis=-1))
            nfevs.append(nfev)
            n_failed_total += n_fail
            n_total += N_valid
        seq_count += B
    errors = np.concatenate(errors)
    n_solves = errors.size                       # = oscillators x N_RAND_INIT
    return {
        "mean_error": float(errors.mean()),
        "max_error": float(errors.max()),
        "frac_lt_001": float((errors < 0.01).mean()),
        "antipodal_rate": float((errors > 0.1).mean()),
        # As published. n_total already counts each initialization, so dividing by
        # N_RAND_INIT again scales this down by a further factor of 5; kept so the
        # committed JSON reproduces. Use mean_nfev_per_solve for the real figure.
        "mean_nfev_per_token": float(sum(nfevs) / (n_total * N_RAND_INIT + 1e-8)),
        "mean_nfev_per_solve": float(sum(nfevs) / n_solves),
        "frac_failed": float(n_failed_total / max(n_solves, 1)),
        "n_oscillators": int(n_solves // N_RAND_INIT),
        "n_token_positions": int(n_solves // N_RAND_INIT // 4),   # oscillators / heads
        "n_solves": int(n_solves),
    }


def verify_init_methods(model, loader, d_osc):
    """Random vs sequential initialization, one init each.

    Backs the paper's "sequential initialization reduces mean error 4.6x at d_osc=2"
    (Section 3, TinyStories). NOT the "closes the residual gap to 0.04 PPL" claim, which is
    a different experiment on WikiText-2 with a different scheme -- see
    language_modeling/sequential_init.py. This one starts from the previous token's
    ANALYTIC fixed point plus SEQ_NOISE jitter; that one chains the previous position's
    CONVERGED state with no jitter. Do not assume a change here applies to both.

    Sequential seeds each position from the previous position's analytic fixed point
    (plus SEQ_NOISE jitter) instead of from a uniform draw. This is the arm behind the
    paper's "sequential initialization reduces mean error 4.6x at d_osc=2 at no
    additional cost".

    Note the divisor here is the oscillator count, NOT oscillators x initializations --
    this arm runs a single initialization per method -- so the nfev figure in this block
    is already per solve. It is reported as `mean_nfev_per_solve`; the reference JSON
    stored it as `mean_nfev_per_token`, the same name the `uniqueness` block uses for a
    value scaled differently, which is why it was renamed.
    """
    rand_errs, seq_errs, rand_nfevs, seq_nfevs = [], [], [], []
    seq_count = 0
    H = model.layers[0].attn.n_heads
    for batch in loader:
        if seq_count >= N_SEQS:
            break
        tokens, pad_mask = batch["tokens"], batch["pad_mask"]
        B = tokens.size(0)
        h, xa = compute_h_and_analytic(model, tokens, pad_mask)
        BH, T = B * H, tokens.size(1)
        h_flat = h.reshape(BH, T, d_osc).numpy()
        xa_flat = xa.reshape(BH, T, d_osc).numpy()
        valid = ~pad_mask.unsqueeze(1).expand(B, H, T).reshape(BH, T).numpy()
        h_v, xa_v = h_flat[valid], xa_flat[valid]

        xf_r, nfev_r, _ = integrate(h_v, rand_sphere((h_v.shape[0], d_osc)))
        rand_errs.append(np.linalg.norm(xf_r - xa_v, axis=-1))
        rand_nfevs.append(nfev_r)

        x_seq = np.zeros_like(h_flat)
        x_seq[:, 0, :] = rand_sphere((BH, d_osc))
        for i in range(1, T):
            prev = xa_flat[:, i - 1, :]
            x_seq[:, i, :] = prev + np.random.randn(*prev.shape) * SEQ_NOISE
            x_seq[:, i, :] /= np.linalg.norm(x_seq[:, i, :], axis=-1,
                                             keepdims=True).clip(min=1e-10)
        xf_s, nfev_s, _ = integrate(h_v, x_seq[valid])
        seq_errs.append(np.linalg.norm(xf_s - xa_v, axis=-1))
        seq_nfevs.append(nfev_s)
        seq_count += B

    rand_errs = np.concatenate(rand_errs)
    seq_errs = np.concatenate(seq_errs)
    n_tok = rand_errs.shape[0]
    return {
        "random": {"mean": float(rand_errs.mean()),
                   "frac_lt_001": float((rand_errs < 0.01).mean()),
                   "mean_nfev_per_solve": sum(rand_nfevs) / max(n_tok, 1)},
        "sequential": {"mean": float(seq_errs.mean()),
                       "frac_lt_001": float((seq_errs < 0.01).mean()),
                       "mean_nfev_per_solve": sum(seq_nfevs) / max(n_tok, 1)},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckptdir",
                    help="directory holding TS2_d2.pt / TS3_d8.pt / TS4_d32.pt")
    ap.add_argument("--ckpt", help="a single checkpoint file, for one --d_osc")
    ap.add_argument("--d_osc", type=int, nargs="+", default=[2, 8, 32])
    ap.add_argument("--seed", type=int, default=None,
                    help="seed the initial conditions (the published run did not)")
    args = ap.parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    vocab, _, _, val_chunks = load_tinystories(max_len=ARCH["max_seq_len"])
    loader = DataLoader(SeqDataset(val_chunks[:N_SEQS * BATCH], ARCH["max_seq_len"]),
                        batch_size=BATCH, shuffle=False, num_workers=0)
    out = {}
    if args.ckpt and len(args.d_osc) != 1:
        ap.error("--ckpt takes exactly one --d_osc")
    for d in args.d_osc:
        ck = resolve_ckpt(d, args.ckptdir, args.ckpt)
        if not ck or not os.path.exists(ck):
            print(f"{EXP}: no checkpoint {ck} -- SKIPPED (not distributed)", flush=True)
            continue
        model = LoheLanguageTransformer(vocab_size=len(vocab), d_osc=d, **ARCH)
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(sd.get("state_dict", sd) if isinstance(sd, dict) else sd,
                              strict=False)
        model.eval()
        r = verify(model, loader, d)
        r["d_osc"] = d
        im = verify_init_methods(model, loader, d)
        out[CKPT_NAME[d]] = {"d_osc": d, "uniqueness": r, "init_methods": im}
        ratio = im["random"]["mean"] / max(im["sequential"]["mean"], 1e-12)
        print(f"DONE {CKPT_NAME[d]}: converged={r['frac_lt_001']*100:.1f}% "
              f"antipodal={r['antipodal_rate']*100:.2f}% "
              f"nfev/solve={r['mean_nfev_per_solve']:.0f} "
              f"| init_methods rand/seq mean-err ratio={ratio:.2f}x", flush=True)
    if not out:
        print(f"{EXP} SKIPPED: no checkpoints present; none are distributed.", flush=True)
        sys.exit(1)
    harness.guarded_dump(harness.result_path(EXP, "TS_verify_adaptive"), out)
    print(f"{EXP} COMPLETE", flush=True)


if __name__ == "__main__":
    main()
