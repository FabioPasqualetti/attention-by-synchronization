# `results/lm_dimensional_bottleneck/` — two truncation series, one of them superseded

This directory backs the appendix dimensional-bottleneck study (Figure 10, `fig:rank`). It holds
**two** post-hoc SVD-truncation series over the same ranks. Only one of them is the published one.

| series | evaluation set | PPL at r=3 → r=33 | status |
|---|---|---|---|
| `truncation_full_val_r{3,5,9,17,33}.json` | **full** TinyStories validation set (47,385 sequences) | 49.74 → 10.60 | **published** — this is what the paper reports and what `figures/rank_truncation.py` reads |
| `truncation_r{3,5,9,17,33}.json` | 128-sequence validation subset | 45.22 → 7.92 | **superseded** — provenance only, read by nothing |

The two disagree because they are evaluated on different sets, most visibly at r=33 (7.92 on the
subset vs. 10.60 on full validation). The subset run made the truncated softmax appear to *cross
below* the trained mechanisms at high rank; on the full validation set it stays above them at every
rank. Only the full-validation series is comparable with the oscillator and rank-match series, which
are also evaluated on the full set.

Each superseded file carries `"status": "superseded"`, `"superseded_by"`, `"eval": "val_subset_128"`
and an explanatory `"note"`, so the distinction survives even if a file is read on its own.

## Producers

| files | driver |
|---|---|
| `truncation_full_val_r*.json` | `language_modeling/dimensional_bottleneck_truncation.py` (full validation set) |
| `truncation_r*.json` | `language_modeling/dimensional_bottleneck.py`, section (c) (128-sequence subset) |
| `spectra_osc_d{2,4,8,16}.json`, `spectra_softmax.json` | `language_modeling/dimensional_bottleneck.py` section (b), `dimensional_bottleneck_dim2.py` (d_osc=2) |
| `rankmatch_dqk{2,4,8,16}_s{0..4}.json` | `language_modeling/dimensional_bottleneck_rankmatch.py` |
| `softmax_train.json`, `train_d2_s0.json` | `language_modeling/dimensional_bottleneck.py` section (a), `dimensional_bottleneck_dim2.py` |

## Field note: `rankmatch_dqk*_s*.json`

The five-seed means are 9.41 / 9.18 / 8.96 / 8.83 for d_qk = 2 / 4 / 8 / 16, with the unmodified
d_qk=32 model at 8.67.

Not every file carries the same metadata: seeds 0–1 record the score-rank ceiling explicitly as
`score_rank_ceiling`, seeds 2–4 record only `d_qk`. The two are equal by construction, so `d_qk` is
the field to read; `figures/rank_truncation.py` falls back to it. The measured `val_ppl` is present
and comparable in all of them.
