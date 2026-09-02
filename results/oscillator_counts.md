# Active free-oscillator counts

Formula (paper, Section 5 / cost discussion):

> the mechanism requires T scalar oscillators per head, i.e. **n_h × T × (d_osc − 1)**;
> this counts the active free oscillators (the anchors are fixed forcing points set by
> learned parameters, not evolving elements).

Each free oscillator lives on the sphere S^{d_osc−1}, which has **d_osc − 1** scalar
degrees of freedom — hence the `(d_osc − 1)` factor, not `d_osc`. The count is the size
of one attention layer's free-oscillator array; **n_layers is not a factor** (this matches
the paper, e.g. WikiText-2 is a 2-layer model but its count is 4×50×31, not doubled).

| Experiment | Paper table | n_h | T | d_osc | Oscillators = n_h × T × (d_osc−1) |
|---|---|---|---|---|---|
| KWS (main result) | Table 1 | 2 | 49 | 2 | 2 × 49 × 1 = **98** |
| SVA Config A (original) | Table 2 | 2 | 9 | 2 | 2 × 9 × 1 = **18** |
| **SVA Config G (min-hardware, §4.1)** | Table 2 | **1** | **9** | **2** | 1 × 9 × 1 = **9** |
| WikiText-2 d_osc=2 | Table 4 | 4 | 50 | 2 | 4 × 50 × 1 = **200** |
| WikiText-2 d_osc=8 | Table 4 | 4 | 50 | 8 | 4 × 50 × 7 = **1,400** |
| WikiText-2 d_osc=32 | Table 4 | 4 | 50 | 32 | 4 × 50 × 31 = **6,200** |
| TinyStories d_osc=2 | Table 4 | 4 | 128 | 2 | 4 × 128 × 1 = **512** |
| TinyStories d_osc=8 | Table 4 | 4 | 128 | 8 | 4 × 128 × 7 = **3,584** |
| TinyStories d_osc=32 | Table 4 | 4 | 128 | 32 | 4 × 128 × 31 = **15,872** |

The three counts the paper states explicitly — KWS 98 (2 heads, T=49), SVA Config G 9
(1×9×1), and WikiText-2 at d_osc=32 ≈6,200 (n_h=4, T=50) — all reproduce exactly under
this formula.

## Notes

- **T for KWS**: 49 log-mel frames (1-second clip at 20 ms shift).
- **T for SVA**: 9 tokens per hard sentence (paper §4.1; Config G reports 1×9×1 = 9).
- **T for LM**: context length T=50 for WikiText-2 and T=128 for TinyStories (causal).
- **d_osc − 1**: degrees of freedom of a unit vector on S^{d_osc−1}; at d_osc=2 each
  oscillator is a single scalar phase, so the factor is 1.

## Config G anchor parameter count

Config G (d_model=32, n_heads=1, n_layers=1, d_osc=2, max_seq_len=50):
- Total anchor parameters: n_heads × max_seq_len × d_osc = 1 × 50 × 2 = **100**
- Active free oscillators on a 9-token hard sentence: 1 × 9 × 1 = **9**
- This is the minimum-hardware oscillator configuration tested.

## Config G in context

At Config G, **9 active oscillators** reach a lower training-failure rate than softmax on
hard SVA -- 3/50 seeds below the 85% criterion versus 12/50 -- which is a reduction in
failure rate, not immunity: the oscillator still fails on 3 seeds. It is the smallest
oscillator model tested on a linguistic-structure task. The 9-oscillator LoheAttention
adds 100 anchor values (softmax adds none).
