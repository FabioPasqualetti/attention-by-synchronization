"""
Deep analysis of oscillator misclassifications on hard SVA sentences.
Addresses four questions:
  1. Sentence-level: which words, subject/distractor classes, positions
  2. Attention-level: full attn distribution, gap width, unexpected high-attn tokens
  3. Value/readout: verb_h magnitude/direction, logit margin distribution
  4. Per-seed: are the same sentences failing across seeds?

Run from the repo root:
    python sva/verb_attention.py
"""

import json, math, os, statistics, sys, collections
import torch
import torch.nn.functional as F

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from training import harness, paths  # noqa: E402
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

from oscillator_attention import SVATransformer
from training import sva_dataset as sva_ds
from training.sva_dataset import build_datasets, make_sva_loaders, NOUNS_SG, NOUNS_PL, VOCAB

CKPT_DIR = os.path.join(ROOT, "sva", "checkpoints")
OUT_JSON  = os.path.join(paths.runs_root(), "sva_verb_attention", "sva_misclass_analysis.json")
CFG       = dict(d_model=32, n_heads=1, n_layers=1, d_ff=64, d_osc=2, max_seq_len=50)
SEEDS     = [0, 1, 2, 3, 4]
IDX2WORD  = {i: w for i, w in enumerate(VOCAB)}


def _make_model(attn_type):
    return SVATransformer(
        vocab_size=len(sva_ds.VOCAB), attn_type=attn_type,
        d_osc=CFG["d_osc"], d_model=CFG["d_model"],
        n_heads=CFG["n_heads"], n_layers=CFG["n_layers"],
        d_ff=CFG["d_ff"], max_seq_len=CFG["max_seq_len"],
        dropout=0.0,
    )


@torch.no_grad()
def collect_hard_records(model, loader, val_raw, device):
    """
    Returns list of dicts for every hard sentence, with full diagnostics.
    Includes attention weights, verb_h, logit, and original sentence metadata.
    """
    model.eval()

    # Hooks: attention weights and post-norm verb representation
    _attn_cache  = []
    _verb_h_list = []  # per-sample verb_h (d_model,)

    def _attn_hook(module, inp, out):
        # LoheAttention: out = (B, H, T, T)
        _attn_cache.append(out[1].detach().cpu())

    def _norm_hook(module, inp, out):
        # self.norm output: (B, T, d_model) — stored whole, indexed by verb_idx after
        _verb_h_list.append(out.detach().cpu())

    h_attn = model.layers[0].attn.register_forward_hook(_attn_hook)
    h_norm = model.norm.register_forward_hook(_norm_hook)

    records = []
    global_idx = 0  # tracks position in val_raw

    for batch in loader:
        bsz = len(batch["tokens"])
        _attn_cache.clear()
        _verb_h_list.clear()

        x        = batch["tokens"].to(device)
        labels   = batch["labels"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        verb_idx = batch["verb_idx"].to(device)
        logits   = model(x, padding_mask=pad_mask, verb_idx=verb_idx)
        preds    = (logits > 0).long()

        alpha_b  = _attn_cache[0]           # (B, H, T, T)
        norm_out = _verb_h_list[0]           # (B, T, d_model)

        for i in range(bsz):
            raw_idx = global_idx + i
            raw     = val_raw[raw_idx]

            if not (raw["has_distractor"] and raw["distractor_agrees"] is False):
                continue  # not a hard sentence

            vi  = int(verb_idx[i])          # verb position (BOS-adjusted)
            si  = 2                         # subject at pos 2
            di  = 5                         # distractor at pos 5

            # Full attention row at verb position, head-averaged
            a_m     = alpha_b[i].mean(0)    # (T, T)
            attn_row = a_m[vi, :].tolist()  # length T (padded)

            # Sequence length (non-pad tokens)
            n_toks  = int((~batch["pad_mask"][i]).sum().item())

            a_vs  = float(a_m[vi, si])
            a_vd  = float(a_m[vi, di])
            gap   = a_vs - a_vd

            # Non-subject/distractor peak
            other_attn = [(pos, float(a_m[vi, pos]))
                          for pos in range(n_toks) if pos not in (si, di)]
            other_peak_pos, other_peak_val = max(other_attn, key=lambda x: x[1])

            # Verb_h and logit
            verb_h  = norm_out[i, vi, :].tolist()       # (d_model,)
            logit_v = float(logits[i].item())
            correct = bool(preds[i] == labels[i])
            label   = int(labels[i])

            # Words (from raw sentence)
            toks    = raw["tokens"]  # e.g. ["the","cat","on","the","dogs","sits","."]
            subj_word = toks[raw["subject_idx"]]
            dist_word = toks[raw["distractor_idx"]]
            verb_word = toks[raw["verb_idx"]]
            subj_sg   = subj_word in NOUNS_SG
            dist_sg   = dist_word in NOUNS_SG
            correct_form = "sg" if label == 1 and subj_sg else (
                           "pl" if label == 1 and not subj_sg else (
                           "pl" if label == 0 and subj_sg else "sg"))
            # label=1 → verb agrees with subject
            # So if label=1 and subj_sg → should use SG verb; label=0 and subj_sg → PL verb

            records.append({
                "val_idx":      raw_idx,
                "correct":      correct,
                "label":        label,
                "logit":        round(logit_v, 4),
                "subj_word":    subj_word,
                "dist_word":    dist_word,
                "verb_word":    verb_word,
                "subj_sg":      subj_sg,
                "dist_sg":      dist_sg,
                "a_vs":         round(a_vs, 4),
                "a_vd":         round(a_vd, 4),
                "gap":          round(gap, 4),
                "other_peak_pos": other_peak_pos,
                "other_peak_val": round(other_peak_val, 4),
                "attn_row":     [round(v, 4) for v in attn_row[:n_toks]],
                "verb_h":       [round(v, 5) for v in verb_h],
                "n_tokens":     n_toks,
            })

        global_idx += bsz

    h_attn.remove()
    h_norm.remove()
    return records


def analyse_all_seeds(device):
    splits   = build_datasets()
    val_raw  = splits["val"]
    _, val_loader = make_sva_loaders(batch_size=256, n_train=-1, n_val=-1)

    all_seed_records = {}  # seed → list of hard-sentence records
    for seed in SEEDS:
        print(f"\n── Seed {seed} ──")
        m = _make_model("kuramoto")
        src = os.path.join(CKPT_DIR, f"osc_s{seed}_ep20.pt")
        m.load_state_dict(torch.load(src, map_location=device, weights_only=True))
        m = m.to(device)
        recs = collect_hard_records(m, val_loader, val_raw, device)
        wrong = [r for r in recs if not r["correct"]]
        print(f"  hard={len(recs)}  wrong={len(wrong)}")
        all_seed_records[seed] = recs

    return all_seed_records


def report(all_seed_records):
    out = {}

    # ── Q1: Sentence-level analysis ──────────────────────────────────────────
    print("\n══════════════════════════════════════════════")
    print("Q1 — Sentence-level analysis of wrong predictions")
    print("══════════════════════════════════════════════")

    # Collect all wrong records (pooled)
    all_wrong = []
    for seed, recs in all_seed_records.items():
        for r in recs:
            if not r["correct"]:
                all_wrong.append({**r, "seed": seed})

    print(f"Total wrong records (pooled 5 seeds): {len(all_wrong)}")

    # Subject word distribution on wrong
    subj_counter = collections.Counter(r["subj_word"] for r in all_wrong)
    dist_counter = collections.Counter(r["dist_word"] for r in all_wrong)
    verb_counter = collections.Counter(r["verb_word"] for r in all_wrong)
    print(f"\nTop-10 subject words on wrong sentences:")
    for w, n in subj_counter.most_common(10):
        cls = "SG" if w in NOUNS_SG else "PL"
        print(f"  {w:12s} ({cls}) × {n}")
    print(f"\nTop-10 distractor words on wrong sentences:")
    for w, n in dist_counter.most_common(10):
        cls = "SG" if w in NOUNS_SG else "PL"
        print(f"  {w:12s} ({cls}) × {n}")
    print(f"\nVerb words on wrong sentences:")
    for w, n in verb_counter.most_common():
        print(f"  {w:12s} × {n}")

    # Are subject and distractor in same number class? (They should NOT be on hard sentences)
    same_class_wrong = sum(1 for r in all_wrong if r["subj_sg"] == r["dist_sg"])
    print(f"\nSubj/dist SAME number class: {same_class_wrong}/{len(all_wrong)} (expected 0 by construction)")

    # Subject class distribution on wrong vs correct
    all_correct = [r for seed, recs in all_seed_records.items()
                   for r in recs if r["correct"]]
    wrong_subj_sg = sum(r["subj_sg"] for r in all_wrong) / len(all_wrong)
    corr_subj_sg  = sum(r["subj_sg"] for r in all_correct) / len(all_correct)
    print(f"\nFraction singular subject — wrong: {wrong_subj_sg:.3f}  correct: {corr_subj_sg:.3f}")
    wrong_label1  = sum(r["label"]==1 for r in all_wrong) / len(all_wrong)
    corr_label1   = sum(r["label"]==1 for r in all_correct) / len(all_correct)
    print(f"Fraction label=1 (agree)    — wrong: {wrong_label1:.3f}  correct: {corr_label1:.3f}")

    # Are wrong sentences clustered on specific subjects?
    # Compute per-noun error rate vs exposure
    subj_all = collections.Counter(r["subj_word"] for recs in all_seed_records.values()
                                   for r in recs)
    print(f"\nPer-noun error rate (top 10 by error count):")
    noun_errors = [(w, subj_counter[w], subj_all[w])
                   for w in subj_counter if subj_all[w] > 0]
    noun_errors.sort(key=lambda x: -x[1])
    for w, nerr, ntot in noun_errors[:10]:
        cls = "SG" if w in NOUNS_SG else "PL"
        print(f"  {w:12s} ({cls}) errors={nerr}  total={ntot}  rate={nerr/ntot:.3f}")

    out["Q1"] = {
        "n_wrong_pooled": len(all_wrong),
        "top_subject_words": dict(subj_counter.most_common(15)),
        "top_distractor_words": dict(dist_counter.most_common(15)),
        "verb_words": dict(verb_counter),
        "same_class_wrong": same_class_wrong,
        "frac_subj_sg_wrong": round(wrong_subj_sg, 3),
        "frac_subj_sg_correct": round(corr_subj_sg, 3),
        "frac_label1_wrong": round(wrong_label1, 3),
        "frac_label1_correct": round(corr_label1, 3),
        "per_noun_errors": {w: {"n_errors": nerr, "n_total": ntot, "rate": round(nerr/ntot,3)}
                             for w, nerr, ntot in noun_errors[:15]},
    }

    # ── Q2: Attention-level analysis ─────────────────────────────────────────
    print("\n══════════════════════════════════════════════")
    print("Q2 — Attention distribution on wrong sentences")
    print("══════════════════════════════════════════════")

    wrong_gaps    = [r["gap"] for r in all_wrong]
    correct_gaps  = [r["gap"] for r in all_correct]
    print(f"Gap (a_vs - a_vd) on wrong:   mean={statistics.mean(wrong_gaps):.4f}  "
          f"std={statistics.stdev(wrong_gaps):.4f}  "
          f"min={min(wrong_gaps):.4f}  max={max(wrong_gaps):.4f}")
    print(f"Gap (a_vs - a_vd) on correct: mean={statistics.mean(correct_gaps):.4f}  "
          f"std={statistics.stdev(correct_gaps):.4f}")

    # Histogram of wrong gaps
    neg_gap  = sum(1 for g in wrong_gaps if g < 0)
    small_gap = sum(1 for g in wrong_gaps if 0 <= g < 0.05)
    large_gap = sum(1 for g in wrong_gaps if g >= 0.05)
    print(f"Wrong sentences by gap: negative={neg_gap} small(0-0.05)={small_gap} large(≥0.05)={large_gap}")

    # What token gets the OTHER peak attention on wrong sentences?
    pos_names = {0: "BOS", 1: "the(subj)", 2: "SUBJ", 3: "PREP", 4: "the(dist)", 5: "DIST", 6: "VERB", 7: "EOS"}
    other_peak_counter = collections.Counter(r["other_peak_pos"] for r in all_wrong)
    print(f"\nOther-peak position (non-subj, non-dist) on wrong sentences:")
    for pos, cnt in sorted(other_peak_counter.items()):
        pname = pos_names.get(pos, f"pos{pos}")
        print(f"  pos {pos} ({pname}): {cnt}/{len(all_wrong)}")

    # Mean attention by token position on wrong vs correct
    # Hard sentences always have 9 tokens: BOS the SUBJ PREP the DIST VERB . EOS
    print(f"\nMean attention from VERB to each position:")
    print(f"  pos  token       wrong-mean   correct-mean")
    for pos in range(9):
        pname = pos_names.get(pos, f"pos{pos}")
        w_attn = [r["attn_row"][pos] for r in all_wrong if len(r["attn_row"]) > pos]
        c_attn = [r["attn_row"][pos] for r in all_correct if len(r["attn_row"]) > pos]
        w_m = statistics.mean(w_attn) if w_attn else float("nan")
        c_m = statistics.mean(c_attn) if c_attn else float("nan")
        marker = " ←" if pos in (2, 5) else ""
        print(f"  {pos:3d}  {pname:12s}  {w_m:.4f}       {c_m:.4f}{marker}")

    # Primary estimator = unweighted mean of the per-seed means (each seed is one
    # observation, regardless of its error count), matching Table 3's caption. The
    # record-pooled value is retained as an additional field for reference.
    _ps_wrong = {s: statistics.mean(r["gap"] for r in recs if not r["correct"])
                 for s, recs in all_seed_records.items()
                 if any(not r["correct"] for r in recs)}
    _ps_correct = {s: statistics.mean(r["gap"] for r in recs if r["correct"])
                   for s, recs in all_seed_records.items()
                   if any(r["correct"] for r in recs)}
    out["Q2"] = {
        "wrong_gap": {"mean": round(statistics.mean(_ps_wrong.values()),4),
                       "estimator": "unweighted mean of per-seed means",
                       "between_seed_std": round(statistics.stdev(_ps_wrong.values()),4),
                       "per_seed_mean": {str(s): round(v,4) for s,v in sorted(_ps_wrong.items())},
                       "pooled_mean": round(statistics.mean(wrong_gaps),4),
                       "pooled_std":  round(statistics.stdev(wrong_gaps),4),
                       "min":  round(min(wrong_gaps),4), "max": round(max(wrong_gaps),4),
                       "n_negative": neg_gap, "n_small": small_gap, "n_large": large_gap},
        "correct_gap": {"mean": round(statistics.mean(_ps_correct.values()),4),
                         "estimator": "unweighted mean of per-seed means",
                         "between_seed_std": round(statistics.stdev(_ps_correct.values()),4),
                         "per_seed_mean": {str(s): round(v,4) for s,v in sorted(_ps_correct.items())},
                         "pooled_mean": round(statistics.mean(correct_gaps),4),
                         "pooled_std": round(statistics.stdev(correct_gaps),4)},
        "other_peak_pos_counts": dict(other_peak_counter),
        "mean_attn_by_pos": {
            "wrong":   {pos_names.get(p,f"pos{p}"): round(statistics.mean(
                [r["attn_row"][p] for r in all_wrong if len(r["attn_row"])>p]),4)
                for p in range(9)},
            "correct": {pos_names.get(p,f"pos{p}"): round(statistics.mean(
                [r["attn_row"][p] for r in all_correct if len(r["attn_row"])>p]),4)
                for p in range(9)},
        },
    }

    # ── Q3: Value/readout analysis ────────────────────────────────────────────
    print("\n══════════════════════════════════════════════")
    print("Q3 — Value/readout analysis")
    print("══════════════════════════════════════════════")

    wrong_logits   = [r["logit"] for r in all_wrong]
    correct_logits = [r["logit"] for r in all_correct]

    # Logit distribution
    print(f"Logit on wrong   sentences: mean={statistics.mean(wrong_logits):.4f}  "
          f"std={statistics.stdev(wrong_logits):.4f}  "
          f"min={min(wrong_logits):.4f}  max={max(wrong_logits):.4f}")
    print(f"Logit on correct sentences: mean={statistics.mean(correct_logits):.4f}  "
          f"std={statistics.stdev(correct_logits):.4f}  "
          f"min={min(correct_logits):.4f}  max={max(correct_logits):.4f}")

    # Wrong: signed margin (how far from 0 in the wrong direction)
    # label=1 → correct is positive logit; label=0 → correct is negative logit
    margins_wrong = []
    for r in all_wrong:
        # Margin magnitude: how confident was the wrong prediction?
        margins_wrong.append(abs(r["logit"]))
    margins_correct = [abs(r["logit"]) for r in all_correct]

    print(f"|logit| on wrong:   mean={statistics.mean(margins_wrong):.4f}  "
          f"std={statistics.stdev(margins_wrong):.4f}")
    print(f"|logit| on correct: mean={statistics.mean(margins_correct):.4f}  "
          f"std={statistics.stdev(margins_correct):.4f}")

    # Are wrong predictions confident (large |logit|) or marginal (small |logit|)?
    close_wrong = sum(1 for m in margins_wrong if m < 0.5)
    conf_wrong  = sum(1 for m in margins_wrong if m >= 1.0)
    print(f"Wrong predictions: close (|logit|<0.5)={close_wrong}/{len(margins_wrong)}  "
          f"confident (|logit|≥1.0)={conf_wrong}/{len(margins_wrong)}")

    # Verb_h magnitude on wrong vs correct
    wrong_h_norms   = [math.sqrt(sum(x**2 for x in r["verb_h"])) for r in all_wrong]
    correct_h_norms = [math.sqrt(sum(x**2 for x in r["correct"] and r.get("verb_h", []) or []))
                       for r in all_correct if r.get("verb_h")]
    correct_h_norms = [math.sqrt(sum(x**2 for x in r["verb_h"])) for r in all_correct]
    print(f"‖verb_h‖ on wrong:   mean={statistics.mean(wrong_h_norms):.4f}  "
          f"std={statistics.stdev(wrong_h_norms):.4f}")
    print(f"‖verb_h‖ on correct: mean={statistics.mean(correct_h_norms):.4f}  "
          f"std={statistics.stdev(correct_h_norms):.4f}")

    # Correlation: does higher gap → more confident correct prediction?
    # Only on correct sentences
    corr_gaps   = [r["gap"] for r in all_correct]
    corr_logits_abs = [abs(r["logit"]) for r in all_correct]
    # Signed: if label=1, logit should be positive; compute "signed_correct_logit"
    signed_correct = [r["logit"] if r["label"]==1 else -r["logit"] for r in all_correct]
    n = len(corr_gaps)
    mg, ml = sum(corr_gaps)/n, sum(signed_correct)/n
    num = sum((g-mg)*(l-ml) for g,l in zip(corr_gaps, signed_correct))
    dg  = math.sqrt(sum((g-mg)**2 for g in corr_gaps))
    dl  = math.sqrt(sum((l-ml)**2 for l in signed_correct))
    r_gl = num/(dg*dl) if dg*dl > 1e-12 else float("nan")
    print(f"\nCorrelation gap ↔ signed-correct-logit (correct sentences): r={r_gl:.4f}")

    out["Q3"] = {
        "wrong_logit":  {"mean": round(statistics.mean(wrong_logits),4),
                          "std": round(statistics.stdev(wrong_logits),4),
                          "min": round(min(wrong_logits),4),
                          "max": round(max(wrong_logits),4)},
        "correct_logit": {"mean": round(statistics.mean(correct_logits),4),
                           "std": round(statistics.stdev(correct_logits),4)},
        "abs_logit_wrong": {"mean": round(statistics.mean(margins_wrong),4),
                             "n_close": close_wrong, "n_confident": conf_wrong},
        "abs_logit_correct": {"mean": round(statistics.mean(margins_correct),4)},
        "verb_h_norm_wrong": round(statistics.mean(wrong_h_norms),4),
        "verb_h_norm_correct": round(statistics.mean(correct_h_norms),4),
        "r_gap_logit_correct": round(r_gl, 4),
    }

    # ── Q4: Per-seed analysis ─────────────────────────────────────────────────
    print("\n══════════════════════════════════════════════")
    print("Q4 — Per-seed analysis: same sentences failing?")
    print("══════════════════════════════════════════════")

    # Wrong val_idx per seed
    wrong_idx_per_seed = {}
    for seed, recs in all_seed_records.items():
        wrong_idx_per_seed[seed] = set(r["val_idx"] for r in recs if not r["correct"])
        print(f"  Seed {seed}: {len(wrong_idx_per_seed[seed])} wrong sentences  "
              f"val_idx range [{min(wrong_idx_per_seed[seed])}, {max(wrong_idx_per_seed[seed])}]"
              if wrong_idx_per_seed[seed] else f"  Seed {seed}: 0 wrong sentences")

    all_wrong_idx = set().union(*wrong_idx_per_seed.values())
    always_wrong  = set.intersection(*wrong_idx_per_seed.values())
    only_one_seed = {idx for idx in all_wrong_idx
                     if sum(idx in s for s in wrong_idx_per_seed.values()) == 1}
    multi_seed    = all_wrong_idx - only_one_seed

    print(f"\nTotal unique sentences that fail in ≥1 seed: {len(all_wrong_idx)}")
    print(f"Fail in ALL 5 seeds ('intrinsically hard'): {len(always_wrong)}")
    print(f"Fail in ≥2 seeds: {len(multi_seed)}")
    print(f"Fail in exactly 1 seed ('seed-specific'): {len(only_one_seed)}")

    # Distribution: how many seeds does each failing sentence fail in?
    seed_count_dist = collections.Counter(
        sum(idx in s for s in wrong_idx_per_seed.values())
        for idx in all_wrong_idx
    )
    print(f"\nDistribution (n_seeds × n_sentences):")
    for n_seeds in sorted(seed_count_dist):
        print(f"  fails in {n_seeds} seed(s): {seed_count_dist[n_seeds]} sentences")

    # Inspect always-wrong sentences
    if always_wrong:
        print(f"\nSentences failing in all 5 seeds ({len(always_wrong)} total):")
        for idx in sorted(always_wrong)[:20]:
            raw = splits_val[idx]
            toks = raw["tokens"]
            seed0_rec = next(r for r in all_seed_records[0] if r["val_idx"]==idx)
            print(f"  val[{idx:4d}]: {' '.join(toks):50s} "
                  f"label={raw['label']}  gap={seed0_rec['gap']:.3f}  logit={seed0_rec['logit']:.3f}")
    else:
        print("No sentence fails in all 5 seeds.")

    # Pairwise overlap between seeds
    print(f"\nPairwise overlap (Jaccard) of wrong-sentence sets:")
    for i, s1 in enumerate(SEEDS):
        for s2 in SEEDS[i+1:]:
            inter = len(wrong_idx_per_seed[s1] & wrong_idx_per_seed[s2])
            union = len(wrong_idx_per_seed[s1] | wrong_idx_per_seed[s2])
            jac = inter/union if union > 0 else 0
            print(f"  seed{s1}∩seed{s2}: {inter} shared / {union} total  Jaccard={jac:.3f}")

    out["Q4"] = {
        "wrong_idx_per_seed": {str(s): sorted(wrong_idx_per_seed[s]) for s in SEEDS},
        "n_always_wrong": len(always_wrong),
        "n_only_one_seed": len(only_one_seed),
        "n_unique_fail": len(all_wrong_idx),
        "seed_count_distribution": {str(k): v for k, v in seed_count_dist.items()},
        "always_wrong_idx": sorted(always_wrong),
        "pairwise_jaccard": {
            f"seed{s1}_seed{s2}":
                round(len(wrong_idx_per_seed[s1] & wrong_idx_per_seed[s2]) /
                      max(1, len(wrong_idx_per_seed[s1] | wrong_idx_per_seed[s2])), 3)
            for i, s1 in enumerate(SEEDS) for s2 in SEEDS[i+1:]
        },
    }

    return out


# Need val_raw accessible inside report(); pass via closure
splits_val = None


def main():
    global splits_val
    device = torch.device("mps") if torch.backends.mps.is_available() \
             else torch.device("cpu")
    print(f"Device: {device}")
    print("Heartbeat: starting per-seed collection (10 models × 1153 hard sentences)")

    splits_val = build_datasets()["val"]
    all_seed_records = analyse_all_seeds(device)

    print("\nHeartbeat: collection done, running analysis")
    result = report(all_seed_records)

    harness.guarded_dump(OUT_JSON, result)
    print(f"\n[done] → {OUT_JSON}")
    print("Heartbeat: complete")


if __name__ == "__main__":
    main()
