"""
Single-run driver for the 5-point d_osc scaling-law extension (TMLR §4.2).

Runs ONE (dataset, d_osc, seed) and writes a rich per-run artifact set under
results/lm_5pt_runs/. Training/eval are NOT reimplemented: this calls the exact
`run_one` from train_wikitext.py / train_tinystories.py, so the backbone,
optimizer, LR schedule, tokenizer, train/val split, epoch budget, and PPL
evaluation are identical to the existing {2,8,32} runs. Only d_osc changes.

Per run it writes:
  <out>/<name>.json            metrics + provenance (schema below)
  <out>/logs/<name>.log        full stdout/stderr
  <out>/checkpoints/<name>/    periodic checkpoints (deletable after metrics)
where <name> = {wt2|ts}_dosc{D}_seed{S}.

Idempotent: if <name>.json already exists with a val_ppl, the run is skipped
(use --force to re-run). This makes the sweep safely resumable.

Usage:
    python results/lm_5pt_runs/scripts/run_one_5pt.py --dataset wt2 --d_osc 4 --seed 0
"""

import argparse
import contextlib
import datetime
import json
import os
import re
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from training import train_wikitext as W       # noqa: E402
from training import train_tinystories as T     # noqa: E402

DEFAULT_OUT = os.path.join(ROOT, "runs", "lm_5pt_runs")
_LINE = re.compile(r"loss=([0-9.]+)\s*\|\s*val_ppl=([0-9.]+)")


class _Tee:
    """Write to several streams at once (live console + per-run log file)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def get_device(name):
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["wt2", "ts"], required=True)
    ap.add_argument("--d_osc", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    name = f"{args.dataset}_dosc{args.d_osc}_seed{args.seed}"
    out_dir = os.path.abspath(args.out)
    log_dir = os.path.join(out_dir, "logs")
    ckpt_dir = os.path.join(out_dir, "checkpoints", name)
    json_path = os.path.join(out_dir, f"{name}.json")
    log_path = os.path.join(log_dir, f"{name}.log")
    for d in (out_dir, log_dir, ckpt_dir):
        os.makedirs(d, exist_ok=True)

    if os.path.exists(json_path) and not args.force:
        try:
            prev = json.load(open(json_path))
            if prev.get("val_ppl") is not None:
                print(f"[skip] {name}: already done (val_ppl="
                      f"{prev['val_ppl']:.4f}). Use --force to re-run.",
                      flush=True)
                return
        except Exception:
            pass

    device = get_device(args.device)

    # Redirect the original run_one's per-epoch checkpoints into this run's dir.
    # run_one resolves CKPT_DIR as a module global at call time, so reassigning
    # the module attribute is sufficient and changes no training behavior.
    W.CKPT_DIR = ckpt_dir
    T.CKPT_DIR = ckpt_dir

    started = datetime.datetime.now()
    t0 = time.time()
    print(f"[start] {name} on {device} at {started.isoformat(timespec='seconds')}",
          flush=True)

    with open(log_path, "w") as lf:
        tee = _Tee(lf, sys.__stdout__)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            print(f"# {name} | device={device} | torch={torch.__version__} | "
                  f"started={started.isoformat(timespec='seconds')}")
            if args.dataset == "wt2":
                r = W.run_one(args.d_osc, "kuramoto", args.seed, device)
            else:
                r = T.run_one(args.d_osc, args.seed, device)

    wall = time.time() - t0
    finished = datetime.datetime.now()

    # Recover per-epoch histories from the captured log.
    train_loss_history, val_ppl_history = [], []
    for line in open(log_path):
        m = _LINE.search(line)
        if m:
            train_loss_history.append(float(m.group(1)))
            val_ppl_history.append(float(m.group(2)))

    result = {
        "dataset": args.dataset,
        "d_osc": args.d_osc,
        "seed": args.seed,
        "val_ppl": r["val_ppl"],
        "train_loss_history": train_loss_history,
        "val_ppl_history": val_ppl_history,
        "wall_clock_seconds": round(wall, 1),
        "machine": "M1_iMac",
        "pytorch_version": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "label": r.get("label"),
        "device": str(device),
        "note": ("backbone/optim/schedule/tokenizer/split/eval identical to "
                 "existing runs via original run_one; only d_osc changed"),
    }
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[done] {name}: val_ppl={r['val_ppl']:.4f} | "
          f"{wall/60:.1f}min | -> {json_path}", flush=True)


if __name__ == "__main__":
    main()
