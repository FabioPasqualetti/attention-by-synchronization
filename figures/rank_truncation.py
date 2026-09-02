"""Dimensional bottleneck: rank bound and truncation -> figures/output/fig_rank_truncation.pdf.

  (i)  effective rank of the PRE-MASK similarity operator vs d_osc in {2,4,8,16}, with the
       rank<=d_osc+1 structural bound as a reference line and the realized POST-MASK (causal)
       attention rank (~40) shown for contrast.
  (ii) TinyStories PPL vs rank, three series:
        - softmax SVD-truncated POST-HOC AT INFERENCE to rank r in {3,5,9,17,33};
        - oscillator PPL(d_osc) TRAINED UNDER its rank ceiling, plotted at matched rank r=d_osc+1,
          as the 5-seed means (+/- SEM) from the scaling re-analysis;
        - softmax TRAINED UNDER REDUCED Q/K SCORE RANK (the rank-match control): d_qk in
          {2,4,8,16} plotted at its score-rank ceiling (score rank <= d_qk), plus the unmodified
          d_qk=32 model. (log y-axis.)
     Note: score rank != realized attention rank for softmax (the exponential lifts realized rank);
     the rank-match points are placed at their SCORE-rank ceiling.

Inputs:
  results/lm_dimensional_bottleneck/spectra_osc_d{2,4,8,16}.json
     (effective_rank_unmasked_similarity{mean,std}, effective_rank_causal_attn{mean}, d_osc, bound_rank)
  results/lm_dimensional_bottleneck/truncation_full_val_r{3,5,9,17,33}.json  (rank, ppl -> post-hoc-truncated
     softmax curve, evaluated on the FULL TinyStories val set -- SAME eval set as the oscillator and
     rank-match series, so all three panel-(ii) curves are directly comparable in absolute PPL.
     The unlabelled truncation_r*.json series in the same directory is a superseded 128-sequence
     subset run -- see results/lm_dimensional_bottleneck/README.md -- and is NOT read here)
  results/lm_dimension_scaling/ts_uniform.json  (scaling_ppls -> oscillator 5-seed mean+/-SEM PPL(d_osc))
  results/lm_dimensional_bottleneck/rankmatch_dqk{2,4,8,16}_s{0..4}.json  (d_qk, val_ppl)

Run standalone from the repo root:
  python figures/rank_truncation.py
"""
import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import style as ps



def _load(rel):
    """`rel` is relative to the results root (see style.results_root)."""
    return json.load(open(os.path.join(ps.results_root(), rel)))


def spectra():
    """Oscillator effective-rank rows, one per d_osc."""
    rows = []
    paths = sorted(set(glob.glob(os.path.join(
        ps.results_root(), "lm_dimensional_bottleneck/spectra_osc_d*.json"))))
    for p in paths:
        d = json.load(open(p))
        rows.append((d["d_osc"], d["bound_rank"],
                     d["effective_rank_unmasked_similarity"]["mean"],
                     d["effective_rank_unmasked_similarity"]["std"],
                     d["effective_rank_causal_attn"]["mean"]))
    return sorted(rows)


def truncation():
    rows = []
    for p in glob.glob(os.path.join(ps.results_root(), "lm_dimensional_bottleneck/truncation_full_val_r*.json")):
        d = json.load(open(p))
        rows.append((d["rank"], d["ppl"], d["matched_d_osc"], d.get("osc_ppl_at_d")))
    return sorted(rows)


def oscillator_uniform():
    """Oscillator PPL(d_osc): 5-seed mean +/- SEM from the uniform re-analysis, mapped to the
    matched rank r = d_osc + 1. Returns sorted [(rank, mean, sem), ...]."""
    d = _load("lm_dimension_scaling/ts_uniform.json")
    rows = []
    for k, ppls in d["scaling_ppls"].items():
        a = np.asarray(ppls, float)
        rows.append((int(k) + 1, float(a.mean()), float(a.std(ddof=1) / np.sqrt(a.size))))
    return sorted(rows)


def rankmatch_points():
    """Score-rank-matched softmax: per-d_qk mean +/- std val PPL at its score-rank ceiling.
    Returns sorted [(score_rank, mean, std), ...] for d_qk in {2,4,8,16} (5 seeds each).

    The score-rank ceiling equals d_qk by construction; seeds 0-1 record it explicitly as
    `score_rank_ceiling`, seeds 2-4 record only `d_qk`, so fall back to it."""
    groups = {}
    for p in glob.glob(os.path.join(ps.results_root(), "lm_dimensional_bottleneck/rankmatch_dqk*_s*.json")):
        d = json.load(open(p))
        groups.setdefault(d.get("score_rank_ceiling", d["d_qk"]), []).append(d["val_ppl"])
    out = []
    for r, v in groups.items():
        a = np.asarray(v, float)
        out.append((r, float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0))
    return sorted(out)


def full_softmax_point():
    """d_qk=32 = unmodified full model: the softmax 5-seed baseline (lm_dimensional_bottleneck
    softmax_train + the lm_dimension_scaling ts_softmax seeds 1-4). Returns (32, mean, std)."""
    ppls = [_load("lm_dimensional_bottleneck/softmax_train.json")["val_ppl"]]
    for s in (1, 2, 3, 4):
        ppls.append(_load(f"lm_dimension_scaling/ts_softmax_s{s}.json")["val_ppl"])
    a = np.asarray(ppls, float)
    return 32, float(a.mean()), float(a.std(ddof=1))


def main():
    ps.apply_style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=ps.TWO_PANEL)
    fig.subplots_adjust(wspace=0.44)

    # (i) rank vs d_osc
    rows = spectra()
    dz = [r[0] for r in rows]
    a1.errorbar(dz, [r[2] for r in rows], yerr=[r[3] for r in rows], fmt="o-",
                color=ps.COLOR_OSCILLATOR, capsize=3, label="similarity operator (before masking)")
    a1.plot(dz, [r[1] for r in rows], "--", color=ps.COLOR_ACCENT, lw=1.3,
            label=r"rank bound $d_{\rm osc}\!+\!1$")
    a1.plot(dz, [r[4] for r in rows], "s:", color=ps.COLOR_SOFTMAX, lw=1.2,
            markersize=4, label="attention matrix (after masking)")
    a1.set_xscale("log", base=2)
    a1.set_xticks(dz); a1.set_xticklabels([str(x) for x in dz])
    a1.set_xlabel(r"$d_{\rm osc}$")
    a1.set_ylabel("effective rank")
    a1.set_title("measured effective rank", loc="left", fontweight="bold")
    # free zone: mid-left, below the flat realized-rank line (~40) and above the low rising curves
    a1.legend(fontsize=6.8, loc="center left", framealpha=0.95)

    # (ii) rank-matched PPL: post-hoc-truncated softmax, 5-seed oscillator, reduced-Q/K softmax control
    tr = truncation()
    ranks = [r[0] for r in tr]
    a2.plot(ranks, [r[1] for r in tr], "o-", color=ps.COLOR_SOFTMAX,
            label="softmax, truncated at inference")
    # oscillator: 5-seed mean +/- SEM (matches the scaling figure's whiskers), at rank r=d_osc+1
    osc = oscillator_uniform()
    a2.errorbar([r[0] for r in osc], [r[1] for r in osc], yerr=[r[2] for r in osc],
                fmt="s-", color=ps.COLOR_OSCILLATOR, capsize=3, elinewidth=1.0,
                label=r"oscillator (trained)")
    # rank-match control: softmax trained under reduced Q/K score rank, at its score-rank ceiling —
    # the full sweep d_qk in {2,4,8,16} PLUS the d_qk=32 endpoint (= the unmodified full model). Open
    # diamonds (same size as other markers) joined by a THIN DASHED line = a visual grouping, NOT a
    # fit; the d_qk=32 endpoint is drawn FILLED to mark it as the unmodified full model. Whiskers =
    # per-point std over the 5 seeds, where visible.
    full = full_softmax_point()
    curve = rankmatch_points() + [full]  # [(d_qk, mean, std), ...], d_qk = 2,4,8,16,32
    cx = [r[0] for r in curve]; cy = [r[1] for r in curve]; cerr = [r[2] for r in curve]
    a2.errorbar(cx, cy, yerr=cerr, ls="--", lw=0.9, marker="D", markersize=5,
                markerfacecolor="none", markeredgecolor=ps.COLOR_HIGHLIGHT, markeredgewidth=1.3,
                color=ps.COLOR_HIGHLIGHT, ecolor=ps.COLOR_HIGHLIGHT, capsize=2.5, elinewidth=0.8,
                zorder=6, label="softmax, reduced Q/K dimension (trained)")
    a2.plot([full[0]], [full[1]], marker="D", markersize=5, markerfacecolor=ps.COLOR_HIGHLIGHT,
            markeredgecolor=ps.COLOR_HIGHLIGHT, zorder=7)  # d_qk=32 = unmodified full model
    # log y-axis with PLAIN number ticks (no scientific notation / offset)
    a2.set_yscale("log")
    a2.set_ylim(7.6, 52)
    a2.yaxis.set_major_locator(mticker.FixedLocator([8, 10, 15, 20, 30, 50]))
    _sf = mticker.ScalarFormatter(); _sf.set_scientific(False); _sf.set_useOffset(False)
    a2.yaxis.set_major_formatter(_sf)
    a2.yaxis.set_minor_locator(mticker.NullLocator())
    a2.yaxis.set_minor_formatter(mticker.NullFormatter())
    a2.set_xticks(ranks)
    a2.set_xticklabels([rf"{r[0]}" for r in tr])
    a2.set_xlabel("score-rank ceiling")
    a2.set_ylabel("TinyStories PPL")
    a2.set_title("perplexity at matched rank ceiling", loc="left", fontweight="bold")
    # free zone: upper-right triangle — the truncated-softmax curve has descended out of it and the
    # oscillator / rank-match band sits low; short labels keep the box within this clear region
    a2.legend(fontsize=6.4, loc="upper right", framealpha=0.95)
    ps.save(fig, "fig_rank_truncation")


if __name__ == "__main__":
    main()
