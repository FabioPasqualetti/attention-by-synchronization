"""cost_operation_accounting — Analytical per-inference cost accounting (analysis only, CPU, no training).

Two-path accounting per config (KWS / SVA / TinyStories). NO energy (joules) anywhere.

Path A — all-digital evaluation (the paper's training/eval convention): full digital op
counts, softmax vs oscillator analytic path. Favors softmax in digital op count.

Path B — hybrid deployment (proposed): the digital side keeps coupling front-end MACs +
coupling-programming writes (T^2*n_h) + readout + row reduction + one division/token + value
MACs, and has exp = 0 (vs softmax's T^2). The fixed-point computation is NOT a digital op
column — it is performed by physical equilibration, characterized only by a dimensionless
settling horizon T_settle in normalized units of the trained dynamics (from robustness_perturbations; NOT cycles/dt).
Two readout variants are reported (both computed): (i) component/vector readout (d_osc MACs
per pair, anchors precomputed) and (ii) phase readout (T^2*n_h cosine evaluations).

Counts use dense pair count P = T^2 (matching the T^2*n_h convention in the review). TinyStories
is causal, so its realized valid-pair count is ~T^2/2 — noted, not applied, for convention parity.
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
CAVEAT = ("Physical latency equals the settling horizon divided by the effective coupling "
          "rate, a design parameter; the binding constraints are fabrication precision and "
          "noise, whose tolerated envelopes are measured in robustness_perturbations (coupling mismatch, state noise).")


def counts(cfg, task):
    T, H, dh, do, dm = cfg["T"], cfg["n_h"], cfg["d_h"], cfg["d_osc"], cfg["d_model"]
    P = T * T  # dense pair count (T^2)
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
        reduction_soft=H * T, reduction_osc=2 * H * T,
        division_soft=H * T, division_osc=2 * H * T, division_hybrid=H * T,
        T_settle=T_SETTLE[task], T_settle_note=T_SETTLE_NOTE[task],
    )


def _f(n):
    return f"{n:,}"


def markdown(all_c):
    L = ["### cost_operation_accounting — Per-inference cost accounting: two deployment paths\n",
         "Operation counts only — **no energy**. Per attention layer; dense pair count "
         "P = T² (the T²·n_h convention; TinyStories is causal so realized valid pairs ≈ T²/2 — "
         "noted, not applied). Front-end (QK/coupling) MACs and value-path MACs are **identical** "
         "between mechanisms.\n"]
    for task, c in all_c.items():
        hdr = (f"(T={c['T']}, n_h={c['H']}, d_h={c['d_h']}, d_osc={c['d_osc']}, "
               f"d_model={c['d_model']}, causal={c['causal']}, P=T²={_f(c['P'])})")

        # ---- Path A ----
        L.append(f"\n#### {task} — Path A: all-digital evaluation (paper's train/eval convention)\n{hdr}\n")
        L.append("| op | softmax | oscillator (analytic) |")
        L.append("|---|---|---|")
        L.append(f"| QK / coupling front-end MACs | {_f(c['qk_macs'])} | {_f(c['qk_macs'])} |")
        L.append(f"| exp evaluations | {_f(c['exp'])} (T exps/query) | {_f(c['exp'])} (softplus) |")
        L.append(f"| fixed-point MACs (Σ W·anchor) | — | {_f(c['fixed_point_macs'])} |")
        L.append(f"| readout MACs (cos s_ij) | — | {_f(c['readout_i_macs'])} |")
        L.append(f"| global reduction (Σ over T) | {_f(c['reduction_soft'])} | {_f(c['reduction_osc'])} |")
        L.append(f"| division | {_f(c['division_soft'])} | {_f(c['division_osc'])} |")
        L.append(f"| value-path MACs (V,A·V,O) | {_f(c['value_macs'])} | {_f(c['value_macs'])} |")
        soft_macs = c['qk_macs'] + c['value_macs']
        osc_macs = c['qk_macs'] + c['value_macs'] + c['fixed_point_macs'] + c['readout_i_macs']
        L.append(f"| **total MACs** | **{_f(soft_macs)}** | **{_f(osc_macs)}** |")
        L.append(f"| **total exp / reduction / division** | **{_f(c['exp'])} / {_f(c['reduction_soft'])} / {_f(c['division_soft'])}** | **{_f(c['exp'])} / {_f(c['reduction_osc'])} / {_f(c['division_osc'])}** |")
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
        L.append(f"| row reduction (Σ over T) | {_f(c['reduction_soft'])} | {_f(c['reduction_osc']//2)} |")
        L.append(f"| division (one per token) | {_f(c['division_soft'])} | {_f(c['division_hybrid'])} |")
        L.append(f"| value-path MACs (V,A·V,O) | {_f(c['value_macs'])} | {_f(c['value_macs'])} |")
        L.append(f"| **physical stage** (not ops/cycles) | — | settling horizon T_settle ≈ **{c['T_settle']}** (dimensionless, normalized units; {c['T_settle_note']}) |")
        L.append("\nReadout variants (both computed; **choice left open**, TBD by the fixed-point "
                 "implementation): **(i)** component/vector readout — s_ij as d_osc MACs per pair, "
                 "anchors precomputed (assumes state components measured directly, e.g. I/Q "
                 "demodulation in the scalar case); **(ii)** phase readout — T²·n_h cosine evaluations.\n")
        L.append(f"> {CAVEAT}\n")

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
