### cost_operation_accounting — Per-inference cost accounting: two deployment paths

Operation counts only — **no energy**. Per attention layer; dense pair count P = pairs actually evaluated (T² bidirectional, T(T+1)/2 causal). Front-end (QK/coupling) MACs and value-path MACs are **identical** between mechanisms.


#### KWS — Path A: all-digital evaluation (paper's train/eval convention)
(T=49, n_h=2, d_h=16, d_osc=2, d_model=32, causal=False, pairs=2,401)

| op | softmax | oscillator (analytic) |
|---|---|---|
| QK / coupling front-end MACs | 177,184 | 177,184 |
| exp evaluations | 4,802 (T exps/query) | 4,802 (softplus) |
| fixed-point MACs (Σ W·anchor) | — | 9,604 |
| readout MACs (cos s_ij) | — | 9,604 |
| row sum, adds (P-T) | 4,704 | 4,704 |
| divisions, one per weight (P) | 4,802 | 4,802 |
| normalization onto the sphere | — | 392 |
| value-path MACs (V,A·V,O) | 177,184 | 177,184 |
| **total MACs** | **354,368** | **373,576** |
| **total exp / row-sum adds / divisions** | **4,802 / 4,704 / 4,802** | **4,802 / 4,704 / 4,802** |

_Path A **favors softmax** in digital op count (the oscillator adds fixed-point and readout MACs on top of the shared front-end). This is the training/evaluation convention — not the proposed deployment._


#### KWS — Path B: hybrid deployment (proposed)
(T=49, n_h=2, d_h=16, d_osc=2, d_model=32, causal=False, pairs=2,401)

| stage | softmax (digital) | oscillator (hybrid) |
|---|---|---|
| coupling / QK front-end MACs | 177,184 | 177,184 |
| coupling-programming writes (T²·n_h) | — | 4,802 |
| exp evaluations | 4,802 (T²) | **0** |
| readout (i) component/vector — d_osc MACs/pair | — | 9,604 MACs |
| readout (ii) phase — T²·n_h cosine evals | — | 4,802 cosines |
| row sum, adds (P-T) | 4,704 | 4,704 |
| divisions, one per weight (P) | 4,802 | 4,802 |
| value-path MACs (V,A·V,O) | 177,184 | 177,184 |
| **physical stage** (not ops/cycles) | — | settling horizon T_settle ≈ **5** (dimensionless, normalized units; measured (robustness_perturbations): metric flat/converged by T≈1–5) |

Readout variants (both computed; **choice left open**, TBD by the fixed-point implementation): **(i)** component/vector readout — s_ij as d_osc MACs per pair, anchors precomputed (assumes state components measured directly, e.g. I/Q demodulation in the scalar case); **(ii)** phase readout — T²·n_h cosine evaluations.

> Physical latency equals the settling horizon divided by the effective coupling rate, a design parameter; the binding constraints are fabrication precision and noise, whose tolerated envelopes are measured in robustness_perturbations (coupling mismatch, state noise).


#### SVA — Path A: all-digital evaluation (paper's train/eval convention)
(T=9, n_h=1, d_h=32, d_osc=2, d_model=32, causal=False, pairs=81)

| op | softmax | oscillator (analytic) |
|---|---|---|
| QK / coupling front-end MACs | 21,024 | 21,024 |
| exp evaluations | 81 (T exps/query) | 81 (softplus) |
| fixed-point MACs (Σ W·anchor) | — | 162 |
| readout MACs (cos s_ij) | — | 162 |
| row sum, adds (P-T) | 72 | 72 |
| divisions, one per weight (P) | 81 | 81 |
| normalization onto the sphere | — | 36 |
| value-path MACs (V,A·V,O) | 21,024 | 21,024 |
| **total MACs** | **42,048** | **42,372** |
| **total exp / row-sum adds / divisions** | **81 / 72 / 81** | **81 / 72 / 81** |

_Path A **favors softmax** in digital op count (the oscillator adds fixed-point and readout MACs on top of the shared front-end). This is the training/evaluation convention — not the proposed deployment._


#### SVA — Path B: hybrid deployment (proposed)
(T=9, n_h=1, d_h=32, d_osc=2, d_model=32, causal=False, pairs=81)

| stage | softmax (digital) | oscillator (hybrid) |
|---|---|---|
| coupling / QK front-end MACs | 21,024 | 21,024 |
| coupling-programming writes (T²·n_h) | — | 81 |
| exp evaluations | 81 (T²) | **0** |
| readout (i) component/vector — d_osc MACs/pair | — | 162 MACs |
| readout (ii) phase — T²·n_h cosine evals | — | 81 cosines |
| row sum, adds (P-T) | 72 | 72 |
| divisions, one per weight (P) | 81 | 81 |
| value-path MACs (V,A·V,O) | 21,024 | 21,024 |
| **physical stage** (not ops/cycles) | — | settling horizon T_settle ≈ **5** (dimensionless, normalized units; measured (robustness_perturbations SVA settling arm): relative residual vs the analytic fixed point is below the accuracy resolution by T=5) |

Readout variants (both computed; **choice left open**, TBD by the fixed-point implementation): **(i)** component/vector readout — s_ij as d_osc MACs per pair, anchors precomputed (assumes state components measured directly, e.g. I/Q demodulation in the scalar case); **(ii)** phase readout — T²·n_h cosine evaluations.

> Physical latency equals the settling horizon divided by the effective coupling rate, a design parameter; the binding constraints are fabrication precision and noise, whose tolerated envelopes are measured in robustness_perturbations (coupling mismatch, state noise).


#### TinyStories — Path A: all-digital evaluation (paper's train/eval convention)
(T=128, n_h=4, d_h=32, d_osc=8, d_model=128, causal=True, pairs=8,256)

| op | softmax | oscillator (analytic) |
|---|---|---|
| QK / coupling front-end MACs | 5,251,072 | 5,251,072 |
| exp evaluations | 33,024 (T exps/query) | 33,024 (softplus) |
| fixed-point MACs (Σ W·anchor) | — | 264,192 |
| readout MACs (cos s_ij) | — | 264,192 |
| row sum, adds (P-T) | 32,512 | 32,512 |
| divisions, one per weight (P) | 33,024 | 33,024 |
| normalization onto the sphere | — | 8,192 |
| value-path MACs (V,A·V,O) | 5,251,072 | 5,251,072 |
| **total MACs** | **10,502,144** | **11,030,528** |
| **total exp / row-sum adds / divisions** | **33,024 / 32,512 / 33,024** | **33,024 / 32,512 / 33,024** |

_Path A **favors softmax** in digital op count (the oscillator adds fixed-point and readout MACs on top of the shared front-end). This is the training/evaluation convention — not the proposed deployment._


#### TinyStories — Path B: hybrid deployment (proposed)
(T=128, n_h=4, d_h=32, d_osc=8, d_model=128, causal=True, pairs=8,256)

| stage | softmax (digital) | oscillator (hybrid) |
|---|---|---|
| coupling / QK front-end MACs | 5,251,072 | 5,251,072 |
| coupling-programming writes (T²·n_h) | — | 33,024 |
| exp evaluations | 33,024 (T²) | **0** |
| readout (i) component/vector — d_osc MACs/pair | — | 264,192 MACs |
| readout (ii) phase — T²·n_h cosine evals | — | 33,024 cosines |
| row sum, adds (P-T) | 32,512 | 32,512 |
| divisions, one per weight (P) | 33,024 | 33,024 |
| value-path MACs (V,A·V,O) | 5,251,072 | 5,251,072 |
| **physical stage** (not ops/cycles) | — | settling horizon T_settle ≈ **30** (dimensionless, normalized units; measured (robustness_perturbations): within 0.14% of analytic FP by T≈30) |

Readout variants (both computed; **choice left open**, TBD by the fixed-point implementation): **(i)** component/vector readout — s_ij as d_osc MACs per pair, anchors precomputed (assumes state components measured directly, e.g. I/Q demodulation in the scalar case); **(ii)** phase readout — T²·n_h cosine evaluations.

> Physical latency equals the settling horizon divided by the effective coupling rate, a design parameter; the binding constraints are fabrication precision and noise, whose tolerated envelopes are measured in robustness_perturbations (coupling mismatch, state noise).


### Attention stage in multiply-add equivalents (paper Table 7)

One exponential is counted as 10 multiply-adds and every other operation as one (Horowitz, ISSCC 2014; Park & Park, arXiv:2603.12934). Only the step where the mechanisms differ is counted; the projections, pairwise couplings and value products around it are identical and excluded. Coupling-programming writes are not charged.


**With ReLU coupling.**

| implementation | KWS | SVA | TinyStories |
|---|---|---|---|
| softmax | 57,526 | 963 | 395,776 |
| oscillator, all digital | 33,908 | 594 | 635,136 |
| oscillator, equilibration physical | 23,912 | 396 | 362,752 |
| oscillator, equilibration and readout physical | 14,308 | 234 | 98,560 |
| oscillator, affine sum and division physical | 4,802 | 81 | 33,024 |
| **softmax / oscillator** | 1.70x / 2.41x / 4.02x / 11.98x | 1.62x / 2.43x / 4.12x / 11.89x | 0.62x / 1.09x / 4.02x / 11.98x |

**With softplus coupling.**

| implementation | KWS | SVA | TinyStories |
|---|---|---|---|
| softmax | 57,526 | 963 | 395,776 |
| oscillator, all digital | 77,126 | 1,323 | 932,352 |
| oscillator, equilibration physical | 67,130 | 1,125 | 659,968 |
| oscillator, equilibration and readout physical | 57,526 | 963 | 395,776 |
| oscillator, affine sum and division physical | 48,020 | 810 | 330,240 |
| **softmax / oscillator** | 0.75x / 0.86x / 1.00x / 1.20x | 0.73x / 0.86x / 1.00x / 1.19x | 0.42x / 0.60x / 1.00x / 1.20x |

With softplus coupling the third row equals the softmax row exactly: both reduce to the same expression, one exponential per pair plus one row sum and one division. Verified by integer equality in softplus_identity(): True.


**Takeaway.** Front-end (QK/coupling) and value-path MAC counts are identical between mechanisms in every config. Path A (all-digital) favors softmax. Path B (proposed hybrid) removes softmax's T² exponentials entirely (exp = 0), moves the fixed-point computation into a physical equilibration stage characterized only by a dimensionless settling horizon, and leaves the digital side with coupling writes + readout + one reduction + one division/token + value MACs.
