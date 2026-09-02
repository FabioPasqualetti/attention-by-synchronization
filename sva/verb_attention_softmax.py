"""
Softmax error analysis — the softmax mirror of verb_attention.py.

Three questions:
  1. Subject-word breakdown: are softmax errors also concentrated on fish/deer?
  2. Per-seed comparison: identify the catastrophic-failure seed (lowest hard
     accuracy) and separate it from the four successful seeds.
  3. Attention vs logit breakdown: is "attention collapse" a per-seed property
     or a mixing artifact from averaging the catastrophic seed with others?

Run from the repo root:
    python sva/verb_attention_softmax.py
"""

import collections
import json
import math
import os
import statistics
import sys

import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from training import harness, paths  # noqa: E402
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

from oscillator_attention import SVATransformer
from training import sva_dataset as sva_ds
from training.sva_dataset import build_datasets, make_sva_loaders, NOUNS_SG, NOUNS_PL, VOCAB

CKPT_DIR = os.path.join(ROOT, "sva", "checkpoints")
OUT_JSON  = os.path.join(paths.runs_root(), "sva_verb_attention", "sva_softmax_misclass.json")
CFG       = dict(d_model=32, n_heads=1, n_layers=1, d_ff=64, d_osc=2, max_seq_len=50)
SEEDS     = [0, 1, 2, 3, 4]

# Position labels for hard sentences (BOS the SUBJ PREP the DIST VERB . EOS)
POS_NAMES = {0: "BOS", 1: "the_s", 2: "SUBJ", 3: "PREP",
             4: "the_d", 5: "DIST", 6: "VERB", 7: ".", 8: "EOS"}


def _make_model(attn_type):
    return SVATransformer(
        vocab_size=len(sva_ds.VOCAB), attn_type=attn_type,
        d_osc=CFG["d_osc"], d_model=CFG["d_model"],
        n_heads=CFG["n_heads"], n_layers=CFG["n_layers"],
        d_ff=CFG["d_ff"], max_seq_len=CFG["max_seq_len"],
        dropout=0.0,
    )


@torch.no_grad()
def collect_hard_records(model, loader, val_raw, device, attn_type):
    """
    Returns list of dicts for every hard sentence in the val set.
    Works for both softmax and kuramoto (hook path differs).
    """
    model.eval()
    _attn_cache = []

    if attn_type == "softmax":
        def _attn_hook(module, inp, out):
            # out = (output, attn_weights) where attn_weights: (B*H, T, T)
            H = module.n_heads
            attn_bh = out[1].detach().cpu()
            B = attn_bh.shape[0] // H
            T = attn_bh.shape[-1]
            _attn_cache.append(attn_bh.view(B, H, T, T))
    else:
        def _attn_hook(module, inp, out):
            _attn_cache.append(out[1].detach().cpu())

    handle = model.layers[0].attn.register_forward_hook(_attn_hook)

    records = []
    global_idx = 0

    for batch in loader:
        bsz = len(batch["tokens"])
        _attn_cache.clear()

        x        = batch["tokens"].to(device)
        labels   = batch["labels"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        verb_idx = batch["verb_idx"].to(device)
        logits   = model(x, padding_mask=pad_mask, verb_idx=verb_idx)
        preds    = (logits > 0).long()
        alpha_b  = _attn_cache[0]  # (B, H, T, T)

        for i in range(bsz):
            raw_idx = global_idx + i
            raw     = val_raw[raw_idx]

            if not (raw["has_distractor"] and raw["distractor_agrees"] is False):
                continue

            vi = int(verb_idx[i])
            si, di = 2, 5
            a_m      = alpha_b[i].mean(0)   # (T, T)
            n_toks   = int((~batch["pad_mask"][i]).sum().item())
            attn_row = [round(float(a_m[vi, p]), 4) for p in range(n_toks)]

            a_vs = float(a_m[vi, si])
            a_vd = float(a_m[vi, di])
            gap  = a_vs - a_vd

            logit_v = float(logits[i].item())
            correct = bool(preds[i] == labels[i])
            label   = int(labels[i])

            toks      = raw["tokens"]
            subj_word = toks[raw["subject_idx"]]
            dist_word = toks[raw["distractor_idx"]]
            verb_word = toks[raw["verb_idx"]]
            num_invar = (subj_word in NOUNS_SG) and (subj_word in NOUNS_PL)

            records.append({
                "val_idx":   raw_idx,
                "correct":   correct,
                "label":     label,
                "logit":     round(logit_v, 4),
                "subj_word": subj_word,
                "dist_word": dist_word,
                "verb_word": verb_word,
                "num_invar": num_invar,   # fish/deer flag
                "a_vs":      round(a_vs, 4),
                "a_vd":      round(a_vd, 4),
                "gap":       round(gap, 4),
                "attn_row":  attn_row,
                "n_tokens":  n_toks,
            })

        global_idx += bsz

    handle.remove()
    return records


def _gap_stats(recs):
    gaps = [r["gap"] for r in recs]
    if not gaps:
        return {}
    return {
        "n": len(gaps),
        "mean": round(statistics.mean(gaps), 4),
        "std":  round(statistics.stdev(gaps) if len(gaps) > 1 else 0.0, 4),
        "n_negative": sum(1 for g in gaps if g < 0),
        "n_ge05":     sum(1 for g in gaps if g >= 0.05),
    }


def _logit_stats(recs):
    if not recs:
        return {}
    absl = [abs(r["logit"]) for r in recs]
    return {
        "n": len(absl),
        "abs_mean":  round(statistics.mean(absl), 4),
        "abs_std":   round(statistics.stdev(absl) if len(absl) > 1 else 0.0, 4),
        "n_close":   sum(1 for v in absl if v < 0.5),
        "n_conf":    sum(1 for v in absl if v >= 1.0),
    }


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() \
             else torch.device("cpu")
    print(f"Device: {device}")
    print("Heartbeat: starting softmax per-seed collection")

    val_raw = build_datasets()["val"]
    _, val_loader = make_sva_loaders(batch_size=256, n_train=-1, n_val=-1)

    # ── Collect records for all 5 seeds ──────────────────────────────────────
    seed_records = {}
    seed_hard_acc = {}
    for seed in SEEDS:
        print(f"\n  Seed {seed} — softmax")
        m = _make_model("softmax")
        src = os.path.join(CKPT_DIR, f"softmax_s{seed}_ep20.pt")
        m.load_state_dict(torch.load(src, map_location=device, weights_only=True))
        m = m.to(device)
        recs = collect_hard_records(m, val_loader, val_raw, device, "softmax")
        n_hard  = len(recs)
        n_right = sum(r["correct"] for r in recs)
        hard_acc = n_hard and n_right / n_hard
        print(f"    hard={n_hard}  correct={n_right}  hard_acc={hard_acc:.4f}")
        seed_records[seed] = recs
        seed_hard_acc[seed] = round(hard_acc, 4)

    # Identify catastrophic seed (lowest hard accuracy)
    cat_seed = min(SEEDS, key=lambda s: seed_hard_acc[s])
    good_seeds = [s for s in SEEDS if s != cat_seed]
    print(f"\nCatastrophic seed: {cat_seed} (hard_acc={seed_hard_acc[cat_seed]:.4f})")
    print(f"Good seeds: {good_seeds}")

    print("\nHeartbeat: collection done — running analysis")

    results = {"seed_hard_acc": seed_hard_acc, "catastrophic_seed": cat_seed}

    # ── Q1: Subject-word breakdown ────────────────────────────────────────────
    print("\n══ Q1: Subject-word breakdown of softmax errors ══")

    # All wrong records pooled (all 5 seeds)
    all_wrong_all = []
    for seed in SEEDS:
        for r in seed_records[seed]:
            if not r["correct"]:
                all_wrong_all.append({**r, "seed": seed})

    all_correct_all = []
    for seed in SEEDS:
        for r in seed_records[seed]:
            if r["correct"]:
                all_correct_all.append({**r, "seed": seed})

    subj_cnt = collections.Counter(r["subj_word"] for r in all_wrong_all)
    total_wrong = len(all_wrong_all)
    n_numinvar  = sum(r["num_invar"] for r in all_wrong_all)
    n_numinvar_correct = sum(r["num_invar"] for r in all_correct_all)

    print(f"Total wrong (pooled 5 seeds): {total_wrong}")
    print(f"Number-invariant noun as subject: {n_numinvar}/{total_wrong} "
          f"({100*n_numinvar/total_wrong:.1f}%)")
    print(f"Number-invariant noun in correct: {n_numinvar_correct}/{len(all_correct_all)} "
          f"({100*n_numinvar_correct/len(all_correct_all):.1f}%)")
    print(f"\nTop-15 subject words on wrong sentences:")
    subj_all_cnt = collections.Counter(
        r["subj_word"] for seed in SEEDS for r in seed_records[seed])
    for w, cnt in subj_cnt.most_common(15):
        rate = cnt / max(1, subj_all_cnt[w])
        invar = "num-invar" if (w in NOUNS_SG and w in NOUNS_PL) else ""
        print(f"  {w:12s} errors={cnt:4d}  total={subj_all_cnt[w]:5d}  "
              f"rate={rate:.3f}  {invar}")

    results["Q1"] = {
        "n_wrong_pooled": total_wrong,
        "n_num_invariant_wrong": n_numinvar,
        "frac_num_invariant_wrong": round(n_numinvar / max(1, total_wrong), 3),
        "n_num_invariant_correct": n_numinvar_correct,
        "frac_num_invariant_correct": round(n_numinvar_correct / max(1, len(all_correct_all)), 3),
        "top_subject_words": {w: {"n_errors": cnt, "n_total": subj_all_cnt[w],
                                   "rate": round(cnt / max(1, subj_all_cnt[w]), 3),
                                   "num_invar": (w in NOUNS_SG and w in NOUNS_PL)}
                              for w, cnt in subj_cnt.most_common(15)},
    }

    # ── Q2: Per-seed comparison ───────────────────────────────────────────────
    print("\n══ Q2: Per-seed comparison (catastrophic vs good seeds) ══")

    # Catastrophic seed
    cat_wrong   = [r for r in seed_records[cat_seed] if not r["correct"]]
    cat_correct = [r for r in seed_records[cat_seed] if r["correct"]]
    cat_wrong_subj = collections.Counter(r["subj_word"] for r in cat_wrong)
    cat_numinvar_wrong = sum(r["num_invar"] for r in cat_wrong)
    print(f"\nCatastrophic seed {cat_seed}: {len(cat_wrong)} wrong / {len(seed_records[cat_seed])} hard")
    print(f"  num-invariant subject: {cat_numinvar_wrong}/{len(cat_wrong)} "
          f"({100*cat_numinvar_wrong/max(1,len(cat_wrong)):.1f}%)")
    print(f"  Top-5 subject words on wrong: {list(cat_wrong_subj.most_common(5))}")

    # Good seeds pooled
    good_wrong = []
    good_correct = []
    for s in good_seeds:
        good_wrong.extend(r for r in seed_records[s] if not r["correct"])
        good_correct.extend(r for r in seed_records[s] if r["correct"])
    good_numinvar_wrong = sum(r["num_invar"] for r in good_wrong)
    good_subj_cnt = collections.Counter(r["subj_word"] for r in good_wrong)
    print(f"\nGood seeds {good_seeds}: {len(good_wrong)} wrong "
          f"(mean={len(good_wrong)/len(good_seeds):.1f} per seed)")
    print(f"  num-invariant subject: {good_numinvar_wrong}/{len(good_wrong)} "
          f"({100*good_numinvar_wrong/max(1,len(good_wrong)):.1f}%)")
    print(f"  Top-5 subject words on wrong: {list(good_subj_cnt.most_common(5))}")

    # Are catastrophic errors spread across sentences or concentrated?
    # Compare: how many unique val_idx in cat wrong vs expected if random
    cat_wrong_idx = set(r["val_idx"] for r in cat_wrong)
    n_hard_total  = len(seed_records[cat_seed])
    print(f"\nCatastrophic seed: {len(cat_wrong_idx)} unique sentences wrong "
          f"out of {n_hard_total} hard sentences ({100*len(cat_wrong_idx)/n_hard_total:.1f}%)")

    # Do good seeds fail on the same fish/deer sentences as each other?
    good_wrong_idx_per_seed = {s: set(r["val_idx"] for r in seed_records[s]
                                       if not r["correct"]) for s in good_seeds}
    print(f"\nGood-seed pairwise overlap (wrong-sentence Jaccard):")
    all_j = []
    for i, s1 in enumerate(good_seeds):
        for s2 in good_seeds[i+1:]:
            inter = len(good_wrong_idx_per_seed[s1] & good_wrong_idx_per_seed[s2])
            union = len(good_wrong_idx_per_seed[s1] | good_wrong_idx_per_seed[s2])
            jac = inter / max(1, union)
            all_j.append(jac)
            print(f"  seed{s1}∩seed{s2}: inter={inter}  union={union}  J={jac:.3f}")

    results["Q2"] = {
        "catastrophic_seed": {
            "seed": cat_seed,
            "hard_acc": seed_hard_acc[cat_seed],
            "n_wrong": len(cat_wrong),
            "n_hard": len(seed_records[cat_seed]),
            "n_unique_wrong_sentences": len(cat_wrong_idx),
            "frac_unique_wrong": round(len(cat_wrong_idx) / n_hard_total, 3),
            "frac_num_invariant_wrong": round(cat_numinvar_wrong / max(1, len(cat_wrong)), 3),
            "top_subject_words": dict(cat_wrong_subj.most_common(10)),
            "gap_stats_wrong": _gap_stats(cat_wrong),
        },
        "good_seeds": {
            "seeds": good_seeds,
            "n_wrong_pooled": len(good_wrong),
            "mean_wrong_per_seed": round(len(good_wrong) / len(good_seeds), 1),
            "frac_num_invariant_wrong": round(good_numinvar_wrong / max(1, len(good_wrong)), 3),
            "top_subject_words": dict(good_subj_cnt.most_common(10)),
            "pairwise_jaccard_wrong": {
                f"seed{s1}_seed{s2}": round(
                    len(good_wrong_idx_per_seed[s1] & good_wrong_idx_per_seed[s2]) /
                    max(1, len(good_wrong_idx_per_seed[s1] | good_wrong_idx_per_seed[s2])), 3)
                for i, s1 in enumerate(good_seeds)
                for s2 in good_seeds[i+1:]
            },
            "gap_stats_wrong": _gap_stats(good_wrong),
        },
    }

    # ── Q3: Attention vs logit breakdown ─────────────────────────────────────
    print("\n══ Q3: Attention collapse vs logit ambiguity ══")

    # Thresholds: "collapsed attention" = gap < 0.1 (below correct-sentence mean)
    GAP_THRESH = 0.10

    def _attn_vs_logit(wrong_recs, label):
        if not wrong_recs:
            print(f"  {label}: no wrong records")
            return {}
        collapsed = [r for r in wrong_recs if r["gap"] < GAP_THRESH]
        maintained = [r for r in wrong_recs if r["gap"] >= GAP_THRESH]
        close_logit_all = sum(1 for r in wrong_recs if abs(r["logit"]) < 0.5)
        gaps = [r["gap"] for r in wrong_recs]
        print(f"  {label}: n={len(wrong_recs)}")
        print(f"    gap: mean={statistics.mean(gaps):.4f}  "
              f"n_collapsed(gap<{GAP_THRESH})={len(collapsed)}/{len(wrong_recs)}  "
              f"n_maintained={len(maintained)}/{len(wrong_recs)}")
        print(f"    |logit|<0.5 (close call): {close_logit_all}/{len(wrong_recs)}")
        if maintained:
            close_m = sum(1 for r in maintained if abs(r["logit"]) < 0.5)
            print(f"    maintained-attn: {len(maintained)} sentences, "
                  f"{close_m} close-call logits ({100*close_m/len(maintained):.0f}%)")
        return {
            "n": len(wrong_recs),
            "gap_stats": _gap_stats(wrong_recs),
            "logit_stats": _logit_stats(wrong_recs),
            "n_collapsed_attn": len(collapsed),
            "n_maintained_attn": len(maintained),
            "frac_collapsed": round(len(collapsed) / max(1, len(wrong_recs)), 3),
        }

    print(f"\nCatastrophic seed {cat_seed}:")
    q3_cat = _attn_vs_logit(cat_wrong, f"cat_seed{cat_seed}_wrong")
    print(f"\nGood seeds (pooled):")
    q3_good = _attn_vs_logit(good_wrong, "good_seeds_wrong")
    print(f"\nAll wrong pooled:")
    q3_all = _attn_vs_logit(all_wrong_all, "all_wrong_pooled")

    # Per-seed gap and logit means for transparency
    print(f"\nPer-seed gap and |logit| on wrong predictions:")
    per_seed = {}
    for seed in SEEDS:
        wrong_s = [r for r in seed_records[seed] if not r["correct"]]
        corr_s  = [r for r in seed_records[seed] if r["correct"]]
        if wrong_s:
            gaps_w = [r["gap"] for r in wrong_s]
            gaps_c = [r["gap"] for r in corr_s]
            absl_w = [abs(r["logit"]) for r in wrong_s]
            absl_c = [abs(r["logit"]) for r in corr_s]
            mark = " ← CATASTROPHIC" if seed == cat_seed else ""
            print(f"  seed {seed}{mark}:  n_wrong={len(wrong_s)}"
                  f"  gap_wrong={statistics.mean(gaps_w):.4f}"
                  f"  gap_corr={statistics.mean(gaps_c):.4f}"
                  f"  |logit|_wrong={statistics.mean(absl_w):.4f}"
                  f"  |logit|_corr={statistics.mean(absl_c):.4f}")
            per_seed[str(seed)] = {
                "n_wrong": len(wrong_s), "n_correct": len(corr_s),
                "hard_acc": seed_hard_acc[seed],
                "gap_wrong_mean": round(statistics.mean(gaps_w), 4),
                "gap_correct_mean": round(statistics.mean(gaps_c), 4),
                "abs_logit_wrong_mean": round(statistics.mean(absl_w), 4),
                "abs_logit_correct_mean": round(statistics.mean(absl_c), 4),
                "frac_collapsed_attn": round(
                    sum(1 for r in wrong_s if r["gap"] < GAP_THRESH) / len(wrong_s), 3),
                "frac_close_logit": round(
                    sum(1 for r in wrong_s if abs(r["logit"]) < 0.5) / len(wrong_s), 3),
            }
        else:
            print(f"  seed {seed}: 0 wrong")
            per_seed[str(seed)] = {"n_wrong": 0}

    results["Q3"] = {
        "gap_threshold": GAP_THRESH,
        "catastrophic_seed_wrong": q3_cat,
        "good_seeds_wrong":        q3_good,
        "all_wrong_pooled":        q3_all,
        "per_seed":                per_seed,
    }

    # Table 3 provenance: per-split (correct/wrong) verb->subject (a_vs) and
    # verb->distractor (a_vd) attention, as the unweighted mean of per-seed means
    # (each seed one observation) — the estimator Table 3 uses for both rows.
    def _table3_split(want_correct):
        pv, pd = {}, {}
        for s in SEEDS:
            recs = [r for r in seed_records[s] if r["correct"] == want_correct]
            if recs:
                pv[s] = statistics.mean(r["a_vs"] for r in recs)
                pd[s] = statistics.mean(r["a_vd"] for r in recs)
        mvs, mvd = statistics.mean(pv.values()), statistics.mean(pd.values())
        return {"a_vs": round(mvs, 4), "a_vd": round(mvd, 4), "gap": round(mvs - mvd, 4),
                "a_vs_between_seed_std": round(statistics.stdev(pv.values()), 4),
                "a_vd_between_seed_std": round(statistics.stdev(pd.values()), 4)}
    results["table3_mean_attn"] = {
        "estimator": "unweighted mean of per-seed means",
        "correct": _table3_split(True),
        "wrong": _table3_split(False),
    }

    # ── Save ──────────────────────────────────────────────────────────────────
    harness.guarded_dump(OUT_JSON, results)
    print(f"\n[done] → {OUT_JSON}")
    print("Heartbeat: complete")


if __name__ == "__main__":
    main()
