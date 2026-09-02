"""lm_dimensional_bottleneck — score-rank-matched softmax control (TinyStories).

Softmax attention with the per-head Q/K projection reduced to d_qk dimensions, so the pre-softmax
score matrix has rank <= d_qk per head, matching the oscillator's pre-mask similarity ceiling at
d_osc = d_qk. V and the output projection keep their full per-head dimension (d_model // n_heads),
so only the *score* rank is constrained — everything else (architecture, optimizer, schedule,
epochs, seed conventions) is the softmax reference of make_softmax_lm / run_one in
training/train_tinystories.py: d_model=128, n_heads=4, n_layers=2, d_ff=512, 5 epochs.

Runs: d_qk in {2,4,8,16} x seeds {0..4} — the five-seed cohort the paper reports (means
9.41 / 9.18 / 8.96 / 8.83, against the unmodified d_qk=32 model at 8.67). Only the seed differs
across the runs of a given d_qk. Per-run val PPL -> results/lm_dimensional_bottleneck/; results
already present are skipped, so the sweep is resumable and never overwrites a committed run.

Framing: d_qk=2 gives score rank <= 2 per head, matching the oscillator similarity ceiling at
d_osc=2, while retaining softmax's rank-lifting exponential. The comparison against oscillator
d_osc=2 (PPL 10.95) and full softmax (8.70) separates how much of the oscillator gap is score rank
vs. readout. NOTE: score rank and realized attention rank are different objects for softmax — the
exponential lifts the rank of the realized attention matrix above the score rank.
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training import harness, paths  # noqa: E402
paths.ensure_paths()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from oscillator_attention.transformer import SinusoidalPositionalEncoding  # noqa: E402
from training.data_utils import (  # noqa: E402
    load_tinystories, make_lm_loaders, train_lm_epoch, eval_ppl,
)

EXP = "lm_dimensional_bottleneck"
CKPT = os.path.join(harness.RUNS_ROOT, EXP, "ckpt")

# identical to train_tinystories.ARCH / TRAIN
ARCH = dict(d_model=128, n_heads=4, n_layers=2, d_ff=512, max_seq_len=128, dropout=0.1)
TRAIN = dict(lr=5e-4, weight_decay=1e-4, batch_size=256, grad_clip=1.0, n_epochs=5)

# (d_qk, seed) jobs — the full five-seed cohort: d_qk in {2,4,8,16} x seeds 0-4, ordered so the two
# headline dimensions (2 and 8) finish first. Each run is ~3 h; run_one reseeds from `seed` alone,
# so a job's result does not depend on this order or on what ran before it in the process.
# Completed runs are skipped, so a restart never repeats or overwrites one.
JOBS = [(d, s) for s in range(5) for d in (2, 8, 4, 16)]

# reference numbers for the RESULTS framing (means from the 5-seed uniform corpus)
OSC_D2_PPL = 10.95       # oscillator d_osc=2
FULL_SOFTMAX_PPL = 8.70  # full softmax (d_h = d_model//n_heads = 32)
# Sanity bracket for a newly-swept config: PPL should fall monotonically between the d_h=2 mean
# (9.418) and the full-model 5-seed baseline (8.6715, lm_dimension_scaling ts_softmax). A point outside is
# REPORTED/flagged, NOT treated as an error or a stop.
BRACKET_LO, BRACKET_HI = 8.6715, 9.418


class RankMatchedSoftmaxAttention(nn.Module):
    """Softmax attention with score rank <= d_qk per head.

    W_q, W_k project d_model -> n_heads * d_qk (small), so each head's logits = Q Kᵀ / sqrt(d_qk)
    has rank <= d_qk. W_v, W_o are full-width (per-head d_v = d_model // n_heads), so the value
    aggregation and output are unchanged relative to the standard softmax head.
    """

    def __init__(self, d_model, n_heads, d_qk, max_seq_len=2048):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_qk = d_qk                       # per-head query/key dim (score rank ceiling)
        self.d_v = d_model // n_heads          # per-head value dim (unchanged)
        self.scale = d_qk ** 0.5
        self.W_q = nn.Linear(d_model, n_heads * d_qk, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_qk, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=True)
        self.register_buffer("_causal_mask",
                             torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool),
                                        diagonal=1), persistent=False)

    def forward(self, x, padding_mask=None, causal=False):
        B, T, _ = x.shape
        H, dqk, dv = self.n_heads, self.d_qk, self.d_v
        Q = self.W_q(x).view(B, T, H, dqk).transpose(1, 2).reshape(B * H, T, dqk)
        K = self.W_k(x).view(B, T, H, dqk).transpose(1, 2).reshape(B * H, T, dqk)
        V = self.W_v(x).view(B, T, H, dv).transpose(1, 2).reshape(B * H, T, dv)
        logits = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        if causal:
            logits = logits.masked_fill(self._causal_mask[:T, :T].unsqueeze(0), float("-inf"))
        if padding_mask is not None:
            mask = padding_mask.unsqueeze(1).expand(B, H, T).reshape(B * H, 1, T)
            logits = logits.masked_fill(mask, float("-inf"))
        attn = F.softmax(logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        out = torch.bmm(attn, V).view(B, H, T, dv).transpose(1, 2).reshape(B, T, H * dv)
        return self.W_o(out), attn


def build_lm(vocab_size, d_qk):
    """Identical to make_softmax_lm but with RankMatchedSoftmaxAttention (score rank <= d_qk)."""
    class _Layer(nn.Module):
        def __init__(self, d_model, n_heads, d_ff, dropout):
            super().__init__()
            self.norm1 = nn.LayerNorm(d_model)
            self.attn = RankMatchedSoftmaxAttention(d_model, n_heads, d_qk,
                                                    max_seq_len=ARCH["max_seq_len"])
            self.norm2 = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_ff, d_model), nn.Dropout(dropout))

        def forward(self, x, padding_mask=None):
            out, _ = self.attn(self.norm1(x), padding_mask=padding_mask, causal=True)
            x = x + out
            return x + self.ffn(self.norm2(x))

    class _LM(nn.Module):
        def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len, dropout):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
            self.pos_enc = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)
            self.layers = nn.ModuleList([_Layer(d_model, n_heads, d_ff, dropout)
                                         for _ in range(n_layers)])
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size)

        def forward(self, x, padding_mask=None):
            h = self.pos_enc(self.embedding(x))
            for layer in self.layers:
                h = layer(h, padding_mask=padding_mask)
            return self.head(self.norm(h))

    return _LM(vocab_size, **ARCH)


def run_one(d_qk, seed, device):
    """Mirrors train_tinystories.run_one exactly (same optimizer/schedule/epochs)."""
    torch.manual_seed(seed)
    label = f"ts_softmax_dqk{d_qk}_s{seed}"
    vocab, idx2word, train_chunks, val_chunks = load_tinystories(max_len=ARCH["max_seq_len"])
    vocab_size = len(vocab)
    train_loader, val_loader = make_lm_loaders(train_chunks, val_chunks, ARCH["max_seq_len"],
                                               batch_size=TRAIN["batch_size"])
    model = build_lm(vocab_size, d_qk).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=TRAIN["lr"],
                            weight_decay=TRAIN["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=TRAIN["n_epochs"] * len(train_loader))
    best_ppl = float("inf")
    run_start = time.time()
    for epoch in range(1, TRAIN["n_epochs"] + 1):
        t0 = time.time()
        avg_loss = train_lm_epoch(model, train_loader, opt, sched, TRAIN["grad_clip"], device)
        val_ppl = eval_ppl(model, val_loader, device)
        best_ppl = min(best_ppl, val_ppl)
        print(f"[{label}] epoch {epoch}/{TRAIN['n_epochs']} | loss={avg_loss:.4f} | "
              f"val_ppl={val_ppl:.2f} | ep={(time.time()-t0)/60:.1f}min | "
              f"total={(time.time()-run_start)/60:.1f}min", flush=True)
        if epoch % 5 == 0:
            os.makedirs(CKPT, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(CKPT, f"{label}_ep{epoch}.pt"))
        if device.type == "mps":
            torch.mps.empty_cache()
    return best_ppl


def main():
    device = harness.pick_device("mps")
    print(f"lm_dimensional_bottleneck device={device}", flush=True)
    for d_qk, seed in JOBS:
        key = f"rankmatch_dqk{d_qk}_s{seed}"
        if harness.exists(EXP, key):
            print(f"skip {key}", flush=True)
            continue
        t0 = time.time()
        ppl = run_one(d_qk, seed, device)
        in_bracket = BRACKET_LO <= ppl <= BRACKET_HI
        payload = {"d_qk": d_qk, "score_rank_ceiling": d_qk, "seed": seed,
                   "val_ppl": ppl, "n_heads": ARCH["n_heads"],
                   "d_v_per_head": ARCH["d_model"] // ARCH["n_heads"],
                   "osc_d2_ppl": OSC_D2_PPL, "full_softmax_ppl": FULL_SOFTMAX_PPL,
                   "bracket": [BRACKET_LO, BRACKET_HI], "bracket_ok": in_bracket,
                   "wall_sec": round(time.time() - t0, 1)}
        harness.save_result(EXP, key, payload)
        if in_bracket:
            print(f"[bracket] {key}: ppl={ppl:.4f} within [{BRACKET_LO}, {BRACKET_HI}] ✓", flush=True)
        else:
            print(f"[bracket] {key}: ppl={ppl:.4f} OUTSIDE [{BRACKET_LO}, {BRACKET_HI}] ⚠ — FLAG "
                  f"(monotone expectation d_h=2 mean 9.418 -> full-model 8.6715); reported, not an error",
                  flush=True)
        print(f"DONE {key}: PPL={ppl:.4f} ({(time.time()-t0)/60:.1f}m)", flush=True)
        harness.free_memory(device)
    print("lm_dimensional_bottleneck COMPLETE", flush=True)


if __name__ == "__main__":
    main()
