"""cost_operation_accounting — Analytical per-inference cost accounting (analysis only, CPU).

Per-configuration operation counts for KWS / SVA / TinyStories. No end-to-end energy
figures: those are a device property. Per-operation energies enter only as the
multiply-add weighting used to compare an exponential against a multiply-add.

counts() returns the raw per-stage operation counts. Two breakdowns are printed for
reference: an all-digital one and a hybrid one in which the fixed point is reached by
physical equilibration.

stage_mae() produces the paper's Table 7: the attention stage only, in multiply-add
equivalents, for three implementations (see the comment above it). It is the aggregation
the paper uses; note that it does not charge coupling-programming writes, since how
couplings are loaded into an array is device-dependent and not modelled here.

Per-pair terms use the number of query-key pairs actually evaluated: T^2 for a
bidirectional model, T(T+1)/2 for a causal one. TinyStories is causal, so its per-pair
counts are masked accordingly.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training import harness  # noqa: E402

EXP = "cost_operation_accounting"

# (T, n_h, d_h, d_osc, d_model, causal). SVA T = representative padded sentence length.
CONFIGS = {
    "KWS":         dict(T=49,  n_h=2, d_h=16, d_osc=2, d_model=32,  causal=False),
    "SVA":         dict(T=9,   n_h=1, d_h=32, d_osc=2, d_model=32,  causal=False),
    "TinyStories": dict(T=128, n_h=4, d_h=32, d_osc=8, d_model=128, causal=True),
}
# Dimensionless settling horizon (normalized units of the trained dynamics), measured (robustness_perturbations).
T_SETTLE = {"KWS": 5, "SVA": 5, "TinyStories": 30}
T_SETTLE_NOTE = {
    "KWS": "measured (robustness_perturbations): metric flat/converged by T≈1–5",
    "SVA": "measured (robustness_perturbations SVA settling arm): relative residual "
           "vs the analytic fixed point is below the accuracy resolution by T=5",
    "TinyStories": "measured (robustness_perturbations): within 0.14% of analytic FP by T≈30",
}
# --- Energy weighting: multiply-add equivalents (MAE) -------------------------
# Operation counts alone treat an exponential and a multiply-add as one unit each,
# which understates softmax: its cost is concentrated in the exponential.
#
# Horowitz, ISSCC 2014 ("Computing's Energy Problem"), 45 nm: FP32 add = 0.9 pJ,
# FP32 multiply = 3.7 pJ, hence one multiply-accumulate = 4.6 pJ.
# Park & Park, arXiv:2603.12934, Table "Softmax exponential unit cost": a digital
# exponential by Taylor series = 10 FP MACs ~ 46 pJ at FP32/45 nm (INT8: ~2.3 pJ).
# Ratio exponential : MAC = 10 : 1.
MAC_PJ         = 4.6   # pJ per FP32 multiply-accumulate (3.7 multiply + 0.9 add)
EXP_MAC_EQUIV  = 10    # multiply-adds per digital exponential (Taylor series)
RELU_MAC_EQUIV = 1     # relu/elu coupling: one compare-and-select per pair

CAVEAT = ("Physical latency equals the settling horizon divided by the effective coupling "
          "rate, a design parameter; the binding constraints are fabrication precision and "
          "noise, whose tolerated envelopes are measured in robustness_perturbations (coupling mismatch, state noise).")


def counts(cfg, task):
    T, H, dh, do, dm = cfg["T"], cfg["n_h"], cfg["d_h"], cfg["d_osc"], cfg["d_model"]
    # Query-key pairs actually evaluated. A causal model computes only the lower
    # triangle, T(T+1)/2 per head, so every per-pair term is masked. Both
    # mechanisms evaluate their nonlinearity elementwise on the same score
    # matrix (exp for softmax, softplus or relu for the oscillator), so the mask
    # applies identically to both. Per-token terms (the projections, and the one
    # row reduction and one division per query) do not depend on the pair count
    # and are unaffected.
    P = T * (T + 1) // 2 if cfg["causal"] else T * T
    qk_macs = 2 * T * dm * (H * dh) + H * P * dh          # W_q,W_k + QK^T
    value_macs = T * dm * (H * dh) + H * P * dh + T * (H * dh) * dm  # W_v + A·V + W_o
    return dict(
        T=T, H=H, d_h=dh, d_osc=do, d_model=dm, causal=cfg["causal"], P=P,
        qk_macs=qk_macs, value_macs=value_macs,
        exp=H * P,                        # softmax: T exps/query; osc softplus (Path A)
        fixed_point_macs=H * P * do,      # Σ_j W_ij anchor_j
        readout_i_macs=H * P * do,        # (i) component readout: d_osc MACs/pair
        readout_ii_cos=H * P,             # (ii) phase readout: cosine evals (T^2*n_h)
        coupling_writes=H * P,            # coupling-programming writes (T^2*n_h)
        # Normalization, counted per operation rather than per query. A row sum
        # over k terms is k-1 additions, so summing every row costs P-T additions
        # per head; each of the P weights is then divided by its row sum. Both
        # mechanisms normalize this way, so both carry the same two terms. The
        # oscillator additionally normalizes h onto the sphere: d_osc squares and
        # d_osc divisions per query, which is a reduction over d_osc, not over T.
        row_sum_adds=H * (P - T),
        weight_divisions=H * P,
        sphere_norm_ops=H * T * 2 * do,
        T_settle=T_SETTLE[task], T_settle_note=T_SETTLE_NOTE[task],
    )


def _f(n):
    return f"{n:,}"


def markdown(all_c):
    L = ["### cost_operation_accounting — Per-inference cost accounting: two deployment paths\n",
         "Operation counts only — **no energy**. Per attention layer; dense pair count "
         "P = pairs actually evaluated (T² bidirectional, T(T+1)/2 causal). "
         "Front-end (QK/coupling) MACs and value-path MACs are **identical** "
         "between mechanisms.\n"]
    for task, c in all_c.items():
        hdr = (f"(T={c['T']}, n_h={c['H']}, d_h={c['d_h']}, d_osc={c['d_osc']}, "
               f"d_model={c['d_model']}, causal={c['causal']}, pairs={_f(c['P'])})")

        # ---- Path A ----
        L.append(f"\n#### {task} — Path A: all-digital evaluation (paper's train/eval convention)\n{hdr}\n")
        L.append("| op | softmax | oscillator (analytic) |")
        L.append("|---|---|---|")
        L.append(f"| QK / coupling front-end MACs | {_f(c['qk_macs'])} | {_f(c['qk_macs'])} |")
        L.append(f"| exp evaluations | {_f(c['exp'])} (T exps/query) | {_f(c['exp'])} (softplus) |")
        L.append(f"| fixed-point MACs (Σ W·anchor) | — | {_f(c['fixed_point_macs'])} |")
        L.append(f"| readout MACs (cos s_ij) | — | {_f(c['readout_i_macs'])} |")
        L.append(f"| row sum, adds (P-T) | {_f(c['row_sum_adds'])} | {_f(c['row_sum_adds'])} |")
        L.append(f"| divisions, one per weight (P) | {_f(c['weight_divisions'])} | {_f(c['weight_divisions'])} |")
        L.append(f"| normalization onto the sphere | — | {_f(c['sphere_norm_ops'])} |")
        L.append(f"| value-path MACs (V,A·V,O) | {_f(c['value_macs'])} | {_f(c['value_macs'])} |")
        soft_macs = c['qk_macs'] + c['value_macs']
        osc_macs = c['qk_macs'] + c['value_macs'] + c['fixed_point_macs'] + c['readout_i_macs']
        L.append(f"| **total MACs** | **{_f(soft_macs)}** | **{_f(osc_macs)}** |")
        L.append(f"| **total exp / row-sum adds / divisions** | **{_f(c['exp'])} / {_f(c['row_sum_adds'])} / {_f(c['weight_divisions'])}** | **{_f(c['exp'])} / {_f(c['row_sum_adds'])} / {_f(c['weight_divisions'])}** |")
        L.append("\n_Path A **favors softmax** in digital op count (the oscillator adds fixed-point "
                 "and readout MACs on top of the shared front-end). This is the training/evaluation "
                 "convention — not the proposed deployment._\n")

        # ---- Path B ----
        L.append(f"\n#### {task} — Path B: hybrid deployment (proposed)\n{hdr}\n")
        L.append("| stage | softmax (digital) | oscillator (hybrid) |")
        L.append("|---|---|---|")
        L.append(f"| coupling / QK front-end MACs | {_f(c['qk_macs'])} | {_f(c['qk_macs'])} |")
        L.append(f"| coupling-programming writes (T²·n_h) | — | {_f(c['coupling_writes'])} |")
        L.append(f"| exp evaluations | {_f(c['exp'])} (T²) | **0** |")
        L.append(f"| readout (i) component/vector — d_osc MACs/pair | — | {_f(c['readout_i_macs'])} MACs |")
        L.append(f"| readout (ii) phase — T²·n_h cosine evals | — | {_f(c['readout_ii_cos'])} cosines |")
        L.append(f"| row sum, adds (P-T) | {_f(c['row_sum_adds'])} | {_f(c['row_sum_adds'])} |")
        L.append(f"| divisions, one per weight (P) | {_f(c['weight_divisions'])} | {_f(c['weight_divisions'])} |")
        L.append(f"| value-path MACs (V,A·V,O) | {_f(c['value_macs'])} | {_f(c['value_macs'])} |")
        L.append(f"| **physical stage** (not ops/cycles) | — | settling horizon T_settle ≈ **{c['T_settle']}** (dimensionless, normalized units; {c['T_settle_note']}) |")
        L.append("\nReadout variants (both computed; **choice left open**, TBD by the fixed-point "
                 "implementation): **(i)** component/vector readout — s_ij as d_osc MACs per pair, "
                 "anchors precomputed (assumes state components measured directly, e.g. I/Q "
                 "demodulation in the scalar case); **(ii)** phase readout — T²·n_h cosine evaluations.\n")
        L.append(f"> {CAVEAT}\n")

    # ---- Attention stage in multiply-add equivalents (the paper's Table 7) ----
    L.append("\n### Attention stage in multiply-add equivalents (paper Table 7)\n")
    L.append(f"One exponential is counted as {EXP_MAC_EQUIV} multiply-adds and every other "
             f"operation as one (Horowitz, ISSCC 2014; Park & Park, arXiv:2603.12934). "
             f"Only the step where the mechanisms differ is counted; the projections, "
             f"pairwise couplings and value products around it are identical and excluded. "
             f"Coupling-programming writes are not charged.\n")
    for label, w in [("ReLU coupling", RELU_MAC_EQUIV), ("softplus coupling", EXP_MAC_EQUIV)]:
        L.append(f"\n**With {label}.**\n")
        L.append("| implementation | " + " | ".join(all_c) + " |")
        L.append("|---|" + "---|" * len(all_c))
        m = {t: stage_mae(c, coupling_weight=w) for t, c in all_c.items()}
        for name, key in [("softmax", "softmax_mae"),
                          ("oscillator, all digital", "all_digital_mae"),
                          ("oscillator, equilibration physical", "equilibration_physical_mae"),
                          ("oscillator, equilibration and readout physical", "both_physical_mae")]:
            L.append(f"| {name} | " + " | ".join(_f(m[t][key]) for t in all_c) + " |")
        L.append("| **softmax / oscillator** | " + " | ".join(
            "%.2fx / %.2fx / %.2fx" % (m[t]["ratio_all_digital"],
                                       m[t]["ratio_equilibration_physical"],
                                       m[t]["ratio_both_physical"]) for t in all_c) + " |")
    L.append("\nWith softplus coupling the last row equals the softmax row exactly: both "
             "reduce to the same expression, one exponential per pair plus one row sum and "
             "one division. Verified by integer equality in softplus_identity(): "
             f"{all(softplus_identity(c) for c in all_c.values())}.\n")

    L.append("\n**Takeaway.** Front-end (QK/coupling) and value-path MAC counts are identical "
             "between mechanisms in every config. Path A (all-digital) favors softmax. Path B "
             "(proposed hybrid) removes softmax's T² exponentials entirely (exp = 0), moves the "
             "fixed-point computation into a physical equilibration stage characterized only by a "
             "dimensionless settling horizon, and leaves the digital side with coupling writes + "
             "readout + one reduction + one division/token + value MACs.\n")
    return "\n".join(L)


def main():
    all_c = {task: counts(cfg, task) for task, cfg in CONFIGS.items()}
    for task, c in all_c.items():
        harness.save_result(EXP, task, c)
    md_path = os.path.join(harness.RUNS_ROOT, EXP, "cost_tables.md")
    with open(md_path, "w") as f:
        f.write(markdown(all_c))
    print(markdown(all_c))
    print(f"\ncost_operation_accounting COMPLETE -> {md_path}", flush=True)


if __name__ == "__main__":
    main()


# --- Attention-stage cost in multiply-add equivalents -------------------------
# Three implementations of the step where the mechanisms differ, all with ReLU
# coupling (the ablation licenses it: relu(x)+1e-3 matches softplus, 9.856 vs
# 9.784 on TinyStories, inside the pooled between-seed spread).
#
#   A  all digital                : coupling nonlinearity + fixed point + readout
#                                   + both normalizations (sphere and readout).
#   B  equilibration physical     : the array performs the equilibration, which
#                                   removes the fixed point and the normalization
#                                   onto the sphere; the readout stays digital.
#   C  equilibration and readout  : the array also measures the inner products
#      both physical                z*^T r_j. The affine normalization (row sum
#                                   and division) stays digital on the back-end.
#                                   This is the design stated in Remark 1 of the
#                                   paper ("inner products on the sphere are the
#                                   natural observable from oscillator hardware
#                                   ... the linear normalization requires only
#                                   division, which is cheap to perform digitally
#                                   on the back-end"). The row sum is NOT moved
#                                   into the array; doing so would give 4,900 on
#                                   KWS instead of 4,998, and would contradict
#                                   Remark 1.
#
# Path C leaves softmax and the oscillator with the same expression shape: one
# nonlinearity per pair, one row sum, one division. The whole difference is then
# the nonlinearity weight, so with softplus coupling (weight EXP_MAC_EQUIV) the
# two are exactly equal, not approximately -- see softplus_identity().
def stage_mae(c, coupling_weight=None):
    """Attention stage only (the stage where the mechanisms differ), in MAE.

    coupling_weight defaults to RELU_MAC_EQUIV; pass EXP_MAC_EQUIV for the
    softplus variant the trained models actually use.
    """
    w = RELU_MAC_EQUIV if coupling_weight is None else coupling_weight
    norm = c["row_sum_adds"] + c["weight_divisions"]
    soft = c["exp"] * EXP_MAC_EQUIV + norm
    a = (c["exp"] * w + c["fixed_point_macs"] + c["sphere_norm_ops"]
         + c["readout_i_macs"] + norm)
    b = c["exp"] * w + c["readout_i_macs"] + norm
    c_ = c["exp"] * w + norm
    return dict(softmax_mae=soft, all_digital_mae=a,
                equilibration_physical_mae=b, both_physical_mae=c_,
                ratio_all_digital=soft / a, ratio_equilibration_physical=soft / b,
                ratio_both_physical=soft / c_,
                softmax_pj=soft * MAC_PJ, all_digital_pj=a * MAC_PJ,
                equilibration_physical_pj=b * MAC_PJ, both_physical_pj=c_ * MAC_PJ)


def softplus_identity(c):
    """With softplus coupling, Path C and softmax are the SAME expression.

    Both reduce to exp*EXP_MAC_EQUIV + row_sum_adds + weight_divisions, so the
    ratio is exactly 1, by identity rather than by rounding.
    """
    m = stage_mae(c, coupling_weight=EXP_MAC_EQUIV)
    return m["softmax_mae"] == m["both_physical_mae"]
