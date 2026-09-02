"""Bidirectional accuracy composite -> figures/output/fig_bidirectional.pdf.

Side-by-side grouped-bar panels: (a) Keyword Spotting -- grouped bars over
PE x {softmax, oscillator} (from kws_position_matched); (b) Subject-Verb Agreement -- grouped bars
over the three metrics {Overall, Simple, Hard} x {softmax, oscillator} (from sva_seed_robustness);
bars = 50-seed means, +/-1 std error bars (upper whisker truncated at 100 %). Drawing code lives in
bidirectional_common.py, shared with the appendix seed-distribution panel in attention_entropy.py so
the two never diverge. Failure counts go in the caption; the 50-seed box-and-whisker distribution is
the right panel of fig_entropy_seeds.

Inputs:
  (a) results/kws_position_matched/{mech}_{pe}_s{seed}.json   (val_acc in [0,1])
  (b) results/sva_seed_robustness/main_{softmax,kuramoto}_s{0..49}.json  (val_acc.hard, %)

Run standalone from the repo root:
  python figures/bidirectional_accuracy.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import style as ps
import bidirectional_common as c


def main():
    ps.apply_style()
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=ps.TWO_PANEL,
        gridspec_kw={"width_ratios": [1, 1]})
    fig.subplots_adjust(wspace=0.34, bottom=0.13, top=0.88)
    c.draw_kws_bars(ax1, show_ylabel=True, title="Keyword Spotting")
    c.draw_sva_grouped(ax2, show_ylabel=True, title="Subject--Verb Agreement")
    ps.save(fig, "fig_bidirectional")
    fc = c.failure_counts()
    print(f"SVA failures <85%: "
          f"softmax {fc['softmax'][0]}/{fc['softmax'][1]}, "
          f"oscillator {fc['oscillator'][0]}/{fc['oscillator'][1]}")


if __name__ == "__main__":
    main()
