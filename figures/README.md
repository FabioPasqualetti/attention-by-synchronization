# `figures/` — one script per paper figure

Each script regenerates one figure of the paper from the committed `results/` JSONs (plus, for two
of them, the shipped checkpoints). Every script:

* is run **from the repo root**: `python figures/<script>.py`;
* reads only from `results/` (repo-root-relative), hardcoding no plotted value;
* writes `figures/output/<stem>.pdf` (and, for all but `tinystories_demo.py`, a `.png` preview),
  where **`<stem>` is exactly the name `main.tex` includes the figure by** — so an output PDF can be
  copied into the paper's `figures/` directory unrenamed.

`figures/output/` is gitignored: the PDFs are build artifacts, and the paper's copies live in the
paper repository.

## Figure index

| Paper figure | Script | Output stem | Reads |
|---|---|---|---|
| Fig. 2 — antipodal escape | `antipodal_escape.py` | `fig_antipodal_validation` | `results/theory_validation/antipodal_data.json` |
| Fig. 3 — KWS + SVA accuracy | `bidirectional_accuracy.py` | `fig_bidirectional` | `results/kws_position_matched/`, `results/sva_seed_robustness/` |
| Fig. 4 — SVA verb attention | `sva_verb_attention.py` | `fig_sva_attention` | `sva/checkpoints/`, `data/sva/sva_test.jsonl` |
| Fig. 5 — TinyStories demo | `tinystories_demo.py` | `fig_tinystories_demo` | `figures/lib/models/TS_d2.pt`, `figures/lib/cache/vocab_ts.pt` |
| Fig. 6 — d_osc scaling law | `dimension_scaling.py` | `fig_scaling_law` | `results/lm_dimension_scaling/{ts,wt2}_uniform.json` |
| Fig. 7 — ODE convergence | `convergence.py` | `fig_convergence_tmax_dosc` | `results/convergence/{L6_ode_verify_extended,TS_verify_adaptive}.json` |
| Fig. 8 — robustness grid | `robustness_grid.py` | `fig_robustness_grid` | `results/robustness_perturbations/`, `results/robustness_frequency_disorder/` |
| Fig. 9 — entropy + seed spread | `attention_entropy.py` | `fig_entropy_seeds` | `results/lm_attention_entropy/raw_*.json`, `results/sva_seed_robustness/` |
| Fig. 10 — rank and truncation | `rank_truncation.py` | `fig_rank_truncation` | `results/lm_dimensional_bottleneck/`, `results/lm_dimension_scaling/ts_uniform.json` |

Figure 1 (the architecture schematic) is drawn in the manuscript, not generated here.

## Regenerating every figure

```bash
for s in antipodal_escape bidirectional_accuracy sva_verb_attention tinystories_demo \
         dimension_scaling convergence robustness_grid attention_entropy rank_truncation; do
    python figures/$s.py
done
```

Seven of the nine need nothing but the committed result JSONs and run in seconds on CPU. The two
that load a model — `sva_verb_attention.py` (Fig. 4) and `tinystories_demo.py` (Fig. 5) — use
checkpoints that **are shipped** (`sva/checkpoints/`, < 1 MB; `figures/lib/models/TS_d2.pt`, 9.9 MB),
so they also run from a clean clone with no training. `tinystories_demo.py` prints the predicted
word and its probability (`named`, 56.59 %), which is the number quoted in the paper.

## Style

| module | used by | purpose |
|---|---|---|
| `style.py` | the seven results-only scripts | shared rcParams, semantic colors, the standard figure sizes, and `save()` (the single output path into `figures/output/`) |
| `plot_style.py` | `sva_verb_attention.py`, `tinystories_demo.py` | the older style module those two were written against; the color values agree with `style.py` |
| `bidirectional_common.py` | `bidirectional_accuracy.py`, `attention_entropy.py` | shared KWS/SVA loaders and bar / box drawing, so the main-text and appendix panels cannot diverge |

Figure text is typeset with LaTeX (Latin Modern, matching the paper body) when a working LaTeX
installation is found; both style modules probe for one in `apply_style()` and fall back to a
Computer-Modern serif otherwise. The fallback changes typography only, never the plotted values.

## `figures/lib/` — helpers for Figures 4 and 5

| File | Purpose |
|---|---|
| `dataset_sva.py` | SVA vocabulary and tokenizer (same vocabulary as training) |
| `lang_config.yaml` | model hyperparameters for the SVA / LM research configs |
| `model_backend.py` | `KuramotoWordPredictor` / `SoftmaxWordPredictor` for the TinyStories demo |
| `oscillator_attention_L6.py` | `LoheLanguageTransformer` used by `model_backend.py` |
| `demo_config.yaml` | TinyStories demo model config |
| `assets/style.py` | visual constants for the demo figure |
| `cache/vocab_ts.pt` | TinyStories vocabulary (shipped, ~200 KB) |
| `models/TS_d2.pt` | TinyStories d_osc=2 demo checkpoint (shipped, 9.9 MB) |
