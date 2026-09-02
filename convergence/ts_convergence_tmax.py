"""Convergence vs integration horizon on TinyStories -- the producer for the left-hand
panel of Figure 7 (fig:convergence).

Integrates the Lohe ODE for the d_osc=2 TinyStories model at T_max = 30, then re-integrates
only the runs that had not converged at successively longer horizons (100, 500, 1000, 5000),
recording the cumulative converged fraction at each. Convergence is err = ||z(T) - z*||_2
below 0.01. RK45, rtol = atol = 1e-4.

The sample matches convergence/ts_convergence.py exactly: N_SEQS is compared against a count
that advances a batch at a time, so with batch_size 16 the loop consumes 32 validation
chunks -- 2,895 positions x 4 heads = 11,580 oscillators, x 5 initializations = 57,900 solves.

Unlike ts_convergence.py this seeds the initial conditions (np.random.seed(42)), so it
reproduces its committed JSON exactly rather than to within sampling error.

The d_osc=2 checkpoint ships (it is the one Figure 5 uses), so this runs on a clean clone
with no arguments and no download.

Run:  python convergence/ts_convergence_tmax.py
"""
import argparse
import json
import math
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training import harness, paths  # noqa: E402
paths.ensure_paths()
from training.data_utils import load_tinystories, SeqDataset  # noqa: E402
from oscillator_attention import LoheLanguageTransformer      # noqa: E402

_here = os.path.dirname(os.path.abspath(__file__))
_core = paths.load_py(os.path.join(_here, "ts_convergence.py"), "ts_convergence_mod")

EXP = "convergence"
N_SEQS, BATCH, N_RAND_INIT = _core.N_SEQS, _core.BATCH, _core.N_RAND_INIT
T_MAX_BASE, T_MAX_RETRY, ERR_THRESH, TOL = 30.0, [100.0, 500.0, 1000.0, 5000.0], 0.01, 1e-4
ARCH = _core.ARCH


def _rhs(t, y, h):
    x = y / max(np.linalg.norm(y), 1e-10)
    return h - np.dot(x, h) * x


def integrate_one(x_init, h, t_max):
    sol = solve_ivp(_rhs, (0.0, t_max), x_init.astype(np.float64), method="RK45",
                    rtol=TOL, atol=TOL, args=(h,), dense_output=False)
    y = sol.y[:, -1]
    return y / max(np.linalg.norm(y), 1e-10)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=os.path.join(paths.REPO_ROOT, "figures",
                    "lib", "models", "TS_d2.pt"),
                    help="path to the d_osc=2 checkpoint (defaults to the shipped one)")
    args = ap.parse_args()
    if not os.path.exists(args.ckpt):
        print(f"{EXP}: no checkpoint {args.ckpt} -- SKIPPED (not distributed)", flush=True)
        sys.exit(1)

    vocab, _, _, val_chunks = load_tinystories(max_len=ARCH["max_seq_len"])
    loader = DataLoader(SeqDataset(val_chunks[:N_SEQS * BATCH], ARCH["max_seq_len"]),
                        batch_size=BATCH, shuffle=False, num_workers=0)
    model = LoheLanguageTransformer(vocab_size=len(vocab), d_osc=2, **ARCH)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(sd.get("state_dict", sd) if isinstance(sd, dict) else sd,
                          strict=False)
    model.eval()

    all_h, all_xa, seq_count = [], [], 0
    H = model.layers[0].attn.n_heads
    for batch in loader:
        if seq_count >= N_SEQS:
            break
        tokens, pm = batch["tokens"], batch["pad_mask"]
        B = tokens.size(0)
        h, xa = _core.compute_h_and_analytic(model, tokens, pm)
        BH, T = B * H, tokens.size(1)
        valid = ~pm.unsqueeze(1).expand(B, H, T).reshape(BH, T).numpy()
        all_h.append(h.reshape(BH, T, 2).numpy()[valid])
        all_xa.append(xa.reshape(BH, T, 2).numpy()[valid])
        seq_count += B
    h_all = np.concatenate(all_h, 0)
    xa_all = np.concatenate(all_xa, 0)
    N_valid = h_all.shape[0]
    total = N_valid * N_RAND_INIT
    print(f"[tmax] {N_valid:,} oscillators x {N_RAND_INIT} inits = {total:,} solves",
          flush=True)

    np.random.seed(42)   # the published run seeded here; keep it so this reproduces
    failures, success_count = [], 0
    for rep in range(N_RAND_INIT):
        print(f"  init {rep + 1}/{N_RAND_INIT} ...", flush=True)
        for i in range(N_valid):
            v = np.random.randn(2)
            x_init = v / max(np.linalg.norm(v), 1e-10)
            err = float(np.linalg.norm(integrate_one(x_init, h_all[i], T_MAX_BASE) - xa_all[i]))
            if err < ERR_THRESH:
                success_count += 1
            else:
                # angle of the initial condition from the antipode -x*, the diagnostic
                # behind angle_breakdown_failures: a failure starting within a degree or
                # two of -x* is the slow-escape case Proposition 4 predicts.
                cos_a = float(np.dot(x_init, xa_all[i]))
                ang = math.degrees(math.acos(max(-1.0, min(1.0, -cos_a))))
                failures.append({"x_init": x_init, "h": h_all[i], "xa": xa_all[i],
                                 "angle_from_anti": ang})

    frac_by_tmax = {int(T_MAX_BASE): success_count / total}
    resolved = np.zeros(len(failures), dtype=bool)
    for t_max in T_MAX_RETRY:
        still = [i for i, r in enumerate(resolved) if not r]
        if not still:
            frac_by_tmax[int(t_max)] = 1.0
            continue
        print(f"[tmax] retry T_max={t_max:g}: {len(still)} remaining ...", flush=True)
        for idx in still:
            f = failures[idx]
            err = float(np.linalg.norm(integrate_one(f["x_init"], f["h"], t_max) - f["xa"]))
            if err < ERR_THRESH:
                resolved[idx] = True
        frac_by_tmax[int(t_max)] = float((success_count + resolved.sum()) / total)
        print(f"  T_max={t_max:g}: {frac_by_tmax[int(t_max)]:.4f} converged", flush=True)

    ang = np.array([f["angle_from_anti"] for f in failures]) if failures else np.zeros(0)
    sf = np.array([f["angle_from_anti"] for f, r in zip(failures, resolved) if not r]) \
        if failures else np.zeros(0)
    out = {"model": "TS2_d2", "d_osc": 2,
           # N_SEQS is the configured constant; the loop advances a batch at a time and
           # stops at the first batch boundary at or past it, so 32 are processed.
           "n_seqs_configured": N_SEQS, "n_seqs_processed": int(seq_count),
           "n_rand_init": N_RAND_INIT,
           "n_token_positions": int(N_valid // 4), "n_oscillators": int(N_valid),
           "n_total": int(total),
           "frac_by_tmax": {str(k): v for k, v in sorted(frac_by_tmax.items())},
           "n_still_failing_after_tmax5000": int((~resolved).sum()),
           "angle_breakdown_failures": {f"lt_{t}deg": int((ang < t).sum())
                                        for t in (1, 5, 10, 30, 90)},
           "still_failing_angle_stats": ({"min_deg": float(sf.min()),
                                          "mean_deg": float(sf.mean())}
                                         if sf.size else None)}
    harness.guarded_dump(harness.result_path(EXP, "L6_ode_verify_extended"), out)
    print(f"{EXP} tmax sweep COMPLETE", flush=True)


if __name__ == "__main__":
    main()
