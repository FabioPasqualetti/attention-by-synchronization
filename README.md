# Attention by Synchronization in Coupled Oscillator Networks

Code, results, and figure scripts for the TMLR paper *Attention by Synchronization in
Coupled Oscillator Networks*.

We replace the softmax attention mechanism with a Kuramoto–Lohe oscillator ODE whose
fixed point yields row-stochastic attention weights — eliminating both the softmax
nonlinearity and the learned query matrix. Oscillators are initialized at fixed anchor
positions on a hypersphere and driven by key–anchor similarities; their analytic fixed
point provides a differentiable, hardware-friendly attention map. A readout sharpening
exponent *p* interpolates between uniform and peaked attention; we use *p*=1 in the
headline experiments because it is the hardware-native readout. We validate on keyword
spotting (KWS), subject–verb agreement (SVA), WikiText-2 and TinyStories language
modeling, matching or exceeding softmax baselines across all tasks.

```bibtex
@article{pasqualetti2026attention,
  title   = {Attention by Synchronization in Coupled Oscillator Networks},
  author  = {Pasqualetti, Fabio and Guo, Taosha},
  journal = {Transactions on Machine Learning Research (TMLR)},
  year    = {2026}
}
```

The repository layout mirrors the paper: a reader who sees a table, figure, or number can
find the code that produces it from the directory name.

## Layout

```
oscillator_attention/   model package (attention, transformer, ODE, model variants)
training/               shared training backends, data loaders, run harness, path config
data/                   dataset generators + fetcher (corpora are downloaded, not vendored)
keyword_spotting/       Section 4.1 — keyword spotting (Google Speech Commands)
sva/                    Section 4.1 — subject-verb agreement  (+ sva/checkpoints/, config-G)
language_modeling/      Section 4.2 & Appendix C — TinyStories / WikiText-2 / PTB
convergence/            Section 3 — ODE convergence
cost/                   Section 5 — operation accounting
robustness/             Appendix — perturbation grid, frequency disorder
theory/                 Section 3 — antipodal escape, degenerate tokens
figures/                one script per paper figure (+ shared style)
results/                per-experiment result JSONs (the published numbers)
```

Every experiment driver imports the shared `training` package (harness, data, path config)
and the `oscillator_attention` model package. There is one training backend, not copies.

## Two result trees: `results/` and `runs/`

**`results/` is the reference** — the run behind the paper, tracked in git. Every table and
figure regenerates from it, and every number can be verified, **without re-running any
training**.

**`runs/` is yours.** It is gitignored and absent on a clean clone. Every driver writes there,
so re-running an experiment cannot touch the published numbers; your output lands at the same
relative path as ours and comparing is:

```
diff results/lm_dimension_scaling/ts_uniform.json runs/lm_dimension_scaling/ts_uniform.json
```

Figure and analysis scripts **read `results/` by default**, so the figures regenerate on a clean
clone and can be checked against the published PDFs. To redraw them from your own run instead:

```
OSCILLATOR_RESULTS=runs python figures/dimension_scaling.py
```

`OSCILLATOR_RESULTS` moves the read root, `OSCILLATOR_RUNS` the write root; both default under
the repo. Checkpoints follow the write root (`runs/<experiment>/ckpt/`) and are never shipped.

### Data

Corpora are **downloaded, not redistributed**. The cache root is the `OSCILLATOR_DATA_CACHE`
environment variable, defaulting to `data/cache/`. One place configures all of it.

```
python data/fetch_data.py        # WikiText-2 + TinyStories via HuggingFace `datasets`
```

Expected cache layout and approximate sizes:

| corpus | location | how obtained | size |
|---|---|---|---|
| WikiText-2 | `<cache>/wikitext2/` | `data/fetch_data.py` (HuggingFace `wikitext-2-raw-v1`) | ~15 MB |
| TinyStories — **evaluation** | `data/tinystories_eval/{vocab.pt,val_chunks.pkl}` | **shipped** — no download | ~10 MB |
| TinyStories — training split | `<cache>/tinystories/train_chunks.pkl` | `data/fetch_data.py` (HuggingFace `roneneldan/TinyStories`) | ~1 GB |
| Speech Commands | `<cache>/speech_commands/` | auto-downloaded by `torchaudio` on first KWS run | ~5 GB |
| **Penn Treebank** | `<cache>/ptb/ptb_maxlen50.pt` | **user-provided** — LDC-licensed, not redistributable | ~2 MB |
| SVA | `data/sva/*.jsonl` | shipped; regenerable via `training/sva_dataset.py` (fixed seeds) | ~8 MB |

Penn Treebank is under LDC license and is **not** downloaded; place your own licensed,
tokenized copy at `<cache>/ptb/ptb_maxlen50.pt`.

**Which TinyStories split gets evaluated.** `data/tinystories_eval/` ships the vocabulary
and validation split the published numbers were computed on, and it **takes precedence over
anything `data/fetch_data.py` writes**. The fetcher rebuilds from the current upstream
corpus, which does not reproduce that split, so if the fetched copy won, running the setup
above would quietly move every evaluation — and the `results/` vs `runs/` diff — onto
different data. Fetching exists to obtain the training split, which is too large to ship.

To evaluate on a different split deliberately, pass it explicitly:
`load_tinystories(data_dir=...)`. That is the only way to override, and a `data_dir` that
does not resolve raises rather than falling back.

### Checkpoints

Model checkpoints are **not** shipped (~605 MB) and are regenerated by training. Scripts that
need them (`convergence/ode_verification.py`, `language_modeling/sequential_init*.py`,
`language_modeling/integrator_independence.py`) load from `<cache>/checkpoints/`; train the
corresponding model first (per-seed LM training ~= 1-4 h on one GPU/MPS device).

Two sets **do** ship, because a paper table or figure depends on them directly and they are small:
the config-G SVA checkpoints (`sva/checkpoints/`, < 1 MB — Table 3 and Figure 4) and the
TinyStories d_osc=2 demo checkpoint (`figures/lib/models/TS_d2.pt`, 9.9 MB, with its vocabulary —
Figure 5). Both figures therefore regenerate from a clean clone with no training.

## Paper -> code map

### Tables

| Table | driver(s) | reference file(s) | your run lands at |
|---|---|---|---|
| 1 — KWS accuracy | `keyword_spotting/position_matched.py`, `frozen_value.py` | `results/kws_position_matched`, `results/kws_frozen_value` | `runs/kws_position_matched`, `runs/kws_frozen_value` |
| 2 — SVA (50-seed) | `sva/seed_robustness.py`, `sva/frozen_value.py` | `results/sva_seed_robustness`, `results/sva_frozen_value` | `runs/sva_seed_robustness`, `runs/sva_frozen_value` |
| 3 — verb attention | `sva/verb_attention.py`, `sva/verb_attention_softmax.py` | `results/sva_verb_attention` (+ shipped `sva/checkpoints/`) | `runs/sva_verb_attention` |
| 4 — LM dimension scaling | `language_modeling/dimension_scaling.py` (+ `_analysis.py`) | `results/lm_dimension_scaling` | `runs/lm_dimension_scaling` |
| 5 — readout exponent | `keyword_spotting/readout_exponent.py`, `language_modeling/readout_exponent.py` | `results/kws_readout_exponent`, `results/lm_readout_exponent` | `runs/kws_readout_exponent`, `runs/lm_readout_exponent` |
| 6 — ODE convergence | `convergence/ts_convergence.py` | `results/convergence/TS_verify_adaptive.json` | `runs/convergence` |
| 7 — operation accounting | `cost/operation_accounting.py` | `results/cost_operation_accounting` | `runs/cost_operation_accounting` |
| 8 — SVA d_ff sweep (App.) | `sva/architecture_sweep.py` | `results/sva/sva_arch_sweep_5seeds.json` | `runs/sva/` |
| 9 — finite settling (App.) | `robustness/perturbations.py`, `robustness/sva_settling.py` | `results/robustness_perturbations` (`*_settle_*`) | `runs/robustness_perturbations` |

Other appendix numbers: coupling ablation -> `language_modeling/coupling_function*.py`
(`results/lm_coupling_function`); OOS extrapolation -> `language_modeling/{tinystories,wikitext}_extrapolation*.py`
(`results/lm_{tinystories,wikitext}_extrapolation`); dimensional bottleneck ->
`language_modeling/dimensional_bottleneck*.py` (`results/lm_dimensional_bottleneck`); attention
entropy -> `language_modeling/attention_entropy*.py` (`results/lm_attention_entropy`);
degenerate tokens -> `theory/degenerate_tokens.py` (`results/theory_degenerate_tokens`);
robustness -> `robustness/{perturbations,frequency_disorder}.py`
(`results/robustness_{perturbations,frequency_disorder}`); coupling amplification,
position offset, and head scaling -> `language_modeling/{coupling_amplification,position_offset,head_scaling}.py`.

### Figures

Each script writes `figures/output/<stem>.pdf`, where `<stem>` is the name `main.tex` includes the
figure by. Details and exact inputs are in `figures/README.md`.

| Figure | script | output stem | reads |
|---|---|---|---|
| 2 — Antipodal escape | `figures/antipodal_escape.py` | `fig_antipodal_validation` | `results/theory_validation` |
| 3 — Bidirectional accuracy | `figures/bidirectional_accuracy.py` | `fig_bidirectional` | `results/kws_position_matched`, `results/sva_seed_robustness` |
| 4 — SVA verb attention | `figures/sva_verb_attention.py` | `fig_sva_attention` | `sva/checkpoints/` (shipped) |
| 5 — TinyStories demo | `figures/tinystories_demo.py` | `fig_tinystories_demo` | `figures/lib/models/TS_d2.pt` (shipped) |
| 6 — Dimension scaling | `figures/dimension_scaling.py` | `fig_scaling_law` | `results/lm_dimension_scaling` |
| 7 — ODE convergence | `figures/convergence.py` | `fig_convergence_tmax_dosc` | `results/convergence` (produced by `convergence/ts_convergence_tmax.py` and `ts_convergence.py`) |
| 8 — Robustness grid | `figures/robustness_grid.py` | `fig_robustness_grid` | `results/robustness_perturbations`, `results/robustness_frequency_disorder` |
| 9 — Attention entropy + seeds | `figures/attention_entropy.py` | `fig_entropy_seeds` | `results/lm_attention_entropy`, `results/sva_seed_robustness` |
| 10 — Rank / truncation | `figures/rank_truncation.py` | `fig_rank_truncation` | `results/lm_dimensional_bottleneck`, `results/lm_dimension_scaling` |

## Requirements

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch, torchaudio, `datasets`, scipy, matplotlib. Every command in this repository is written
as `python <script>`, which is the environment's interpreter once that venv is active. A CUDA
or Apple-MPS device is used when available; all analysis and figure scripts run on CPU.
