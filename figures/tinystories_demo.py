"""TinyStories oscillator-dynamics demo -> figures/output/fig_tinystories_demo.pdf.

Bespoke 3-panel layout (outside the SINGLE/TWO_PANEL standard, by design): the prompt with the
predicted next word, the oscillator state space with the settling trajectory, and the resulting
attention row. Loads the shipped d_osc=2 demo checkpoint figures/lib/models/TS_d2.pt and vocabulary
figures/lib/cache/vocab_ts.pt.

Run standalone from the repo root:
  python figures/tinystories_demo.py
"""

import math
import os
import sys
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
LIB  = os.path.join(ROOT, "figures", "lib")  # reuse existing lib/ assets
sys.path.insert(0, HERE)
sys.path.insert(0, LIB)

from plot_style import apply_style  # noqa: E402
import yaml  # noqa: E402
import torch  # noqa: E402
from model_backend import make_predictor  # noqa: E402

OUT_PATH = os.path.join(ROOT, "figures", "output", "fig_tinystories_demo.pdf")

# Semantic colors.
ANCHOR_COLOR     = "#E69F00"   # orange (Okabe-Ito) — anchors r_j
SETTLED_COLOR    = "#0072B2"   # blue   (Okabe-Ito) — z_i^*
ACTIVE_COLOR     = "#009E73"   # bluish green       — active oscillator
SPRING_COLOR     = "#B8860B"   # amber — coupling springs


# The left panel's oscillator start point is a DISPLAY choice, not a measurement: model_backend
# draws it from np.random 90-150 degrees away from x* so the convergence trail is visible. Seed it
# so the figure is reproducible run to run. Nothing measured depends on this -- the settled fixed
# points, the attention heatmap and the next-word probabilities are all deterministic.
TRAJECTORY_SEED = 0


def main():
    np.random.seed(TRAJECTORY_SEED)
    apply_style()
    # Restore an "axes.grid": False default for the oscillator/heatmap axes —
    # the global grid is on, but the oscillator circle needs a clean canvas.
    # We toggle it off per-axis below.

    with open(os.path.join(LIB, "demo_config.yaml")) as f:
        cfg = yaml.safe_load(f)
    for mkey in cfg.get("models", {}):
        rel = cfg["models"][mkey].get("checkpoint", "")
        if rel.startswith("models/"):
            cfg["models"][mkey]["checkpoint"] = os.path.join(LIB, rel)
        vp = cfg["models"][mkey].get("vocab_path", "")
        if vp.startswith("cache/"):
            cfg["models"][mkey]["vocab_path"] = os.path.join(LIB, vp)

    device    = "mps" if torch.backends.mps.is_available() else "cpu"
    model     = make_predictor("ts_d2", cfg, device_str=device)
    prompt    = "Once upon a time there was a little girl"
    token_ids = model.encode(prompt)

    xs, attn, logits = model.get_fixed_points(token_ids)
    _, word, _       = model.sample_next_token(logits, temperature=0.0)
    _, _,    top5    = model.sample_next_token(logits, temperature=1.0)
    print(f"[fig_tinystories_demo_v2] predicted word: '{word}'  "
          f"top-5: {[(w, f'{p*100:.2f}%') for w, p in top5[:5]]}")

    td        = model.get_ode_trajectory(x_prev=None)
    last_attn = attn[-1].mean(0)
    T         = last_attn.shape[0]

    fig = plt.figure(figsize=(11, 4.4), facecolor="white")
    gs  = gridspec.GridSpec(1, 3, figure=fig,
                            left=0.02, right=0.99, top=0.91, bottom=0.10,
                            wspace=0.28,
                            width_ratios=[0.40, 0.35, 0.25])

    ax_osc  = fig.add_subplot(gs[0])
    ax_attn = fig.add_subplot(gs[1])
    ax_pred = fig.add_subplot(gs[2])

    # ── Oscillator circle ────────────────────────────────────────────────────
    ax_osc.set_xlim(-1.55, 1.55); ax_osc.set_ylim(-1.55, 1.55)
    ax_osc.set_aspect("equal"); ax_osc.axis("off")

    theta_c = np.linspace(0, 2*np.pi, 300)
    ax_osc.plot(np.cos(theta_c), np.sin(theta_c),
                color="#cccccc", lw=1.4, zorder=0)

    # Anchor squares
    a2d_used = None
    if model.last_ancs is not None:
        a2d  = model.last_ancs[:min(T, 50), :2]
        nrms = np.linalg.norm(a2d, axis=1, keepdims=True)
        a2d  = a2d / np.maximum(nrms, 1e-8)
        a2d_used = a2d
        ax_osc.scatter(a2d[:, 0], a2d[:, 1], marker="s", s=40,
                       color=ANCHOR_COLOR, alpha=0.65, zorder=2,
                       edgecolors="none")

    # Settled oscillators (blue)
    show_words      = ["once", "time", "little", "girl"]
    words_in_prompt = prompt.lower().split()
    for i, (x_star, tok_word) in enumerate(zip(xs, words_in_prompt)):
        if tok_word not in show_words:
            continue
        pos   = x_star[:2]
        age   = len(words_in_prompt) - 1 - i
        size  = max(30, 90 - age * 10)
        alpha = max(0.35, 0.85 - age * 0.06)
        ax_osc.scatter(pos[0], pos[1], s=size, color=SETTLED_COLOR,
                       alpha=alpha, zorder=4, edgecolors="none")
        ang    = math.atan2(float(pos[1]), float(pos[0]))
        lx, ly = 1.35*math.cos(ang), 1.35*math.sin(ang)
        ax_osc.plot([pos[0]*1.06, lx*0.88], [pos[1]*1.06, ly*0.88],
                    color="#aaaaaa", lw=0.7, zorder=3)
        ax_osc.text(lx, ly, tok_word, ha="center", va="center",
                    fontsize=8, color="#333333", zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none",
                              boxstyle="round,pad=0.18", alpha=0.88))

    # Trajectory
    if td is not None:
        traj_2d   = td[0]; ancs_2d = td[3]
        n         = len(traj_2d)
        # Sample pos_active early in the trajectory (15% through), so the
        # active oscillator is still well away from its destination and the
        # coupling springs have visible length.  At idx=0.45 it sits almost
        # on top of its dominant anchor.
        idx       = int(0.15 * (n - 1))
        pos_active = traj_2d[idx]

        # Coupling springs (amber): draw ALL nonzero couplings, with both
        # opacity and linewidth scaled by the (normalized) coupling weight.
        # This makes the dominant connection most visible while still
        # showing the rest of the structure -- a more honest depiction of
        # the "spring force" intent than picking only top-3.
        if model.last_coupling is not None and ancs_2d is not None:
            coupling = model.last_coupling
            wmax     = coupling.max() + 1e-8
            for j in range(len(coupling)):
                if j >= len(ancs_2d):
                    continue
                w_norm = float(coupling[j]) / wmax
                if w_norm < 0.10:
                    continue
                ax_osc.plot([pos_active[0], ancs_2d[j, 0]],
                            [pos_active[1], ancs_2d[j, 1]],
                            color=SPRING_COLOR,
                            alpha=0.30 + 0.55 * w_norm,
                            lw=0.7 + 1.6 * w_norm, zorder=3)

        # ODE trail (green, fading from light to saturated as it advances).
        # Spans trajectory fraction 0 -> idx_frac (matches pos_active).
        # Alpha sweeps 0.30 -> 0.90 (was 0.03 -> 0.55, too faint at the
        # start) and lw 2.2 so the trail is clearly visible.
        idx_frac = 0.15
        n_seg    = 18
        for ti in range(1, n_seg + 1):
            t0 = (ti - 1) / n_seg * idx_frac
            t1 = ti       / n_seg * idx_frac
            i0 = max(0, min(int(t0 * (n - 1)), n - 2))
            i1 = max(0, min(int(t1 * (n - 1)), n - 1))
            alpha_t = 0.30 + 0.60 * (ti / n_seg)
            ax_osc.plot([traj_2d[i0, 0], traj_2d[i1, 0]],
                        [traj_2d[i0, 1], traj_2d[i1, 1]],
                        color=ACTIVE_COLOR, alpha=alpha_t,
                        lw=2.2, zorder=5)

        # "start" annotation at the beginning of the visible trail.
        start_pt = traj_2d[0]
        ang_s = math.atan2(float(start_pt[1]), float(start_pt[0]))
        ax_osc.annotate(
            "start",
            xy=tuple(start_pt),
            xytext=(start_pt[0] + 0.18 * math.cos(ang_s + 0.6),
                    start_pt[1] + 0.18 * math.sin(ang_s + 0.6)),
            fontsize=7.5, color="#1f6b3d", style="italic",
            ha="center", va="center",
            arrowprops=dict(arrowstyle="-", color="#1f6b3d",
                            lw=0.7, alpha=0.8))

        # Dashed destination ring
        dest = traj_2d[-1]
        ax_osc.add_patch(plt.Circle(dest, 0.09, fill=False,
                                    color=ACTIVE_COLOR, lw=1.5, ls="--",
                                    alpha=0.55))

        # Active oscillator dot (no callout label — the legend already
        # identifies it as "Active osc.", and a callout here would conflict
        # with the surrounding word labels).
        ax_osc.scatter(pos_active[0], pos_active[1], s=130,
                       color=ACTIVE_COLOR, zorder=7,
                       edgecolors="#0d5c2a", linewidths=1)

    ax_osc.set_title("Oscillator dynamics", fontsize=10, pad=6,
                     color="#333333", loc="left", fontweight="bold")
    # Redundant prompt caption removed (the figure caption in main.tex
    # already conveys the prompt).

    # In-panel legend explaining markers.
    legend_handles = [
        mlines.Line2D([], [], marker="s", color=ANCHOR_COLOR,
                      linestyle="None", markersize=7, alpha=0.85,
                      label=r"Anchor $r_j$"),
        mlines.Line2D([], [], marker="o", color=SETTLED_COLOR,
                      linestyle="None", markersize=7, alpha=0.85,
                      label=r"Settled $z_i^*$"),
        mlines.Line2D([], [], marker="o", color=ACTIVE_COLOR,
                      linestyle="None", markersize=8,
                      markeredgecolor="#0d5c2a", markeredgewidth=0.8,
                      label="Active osc."),
        mlines.Line2D([], [], color=SPRING_COLOR, lw=1.6, alpha=0.85,
                      label=r"Coupling $w_{ij}$"),
    ]
    ax_osc.legend(handles=legend_handles, loc="lower left",
                  bbox_to_anchor=(-0.02, 0.78),
                  frameon=True, fontsize=7.5,
                  handlelength=1.2, handletextpad=0.5,
                  borderpad=0.4)

    # ── Attention heatmap ────────────────────────────────────────────────────
    ax_attn.set_title("Attention (last layer)", fontsize=9,
                      loc="left", fontweight="bold")
    ax_attn.grid(False)
    T_start = 1 if model.decode(token_ids[0]) in ("<bos>", "<BOS>") else 0
    win     = T - T_start
    sub     = last_attn[T_start:, T_start:]
    raw_words   = [model.decode(t) for t in token_ids[T_start:]]
    word_counts = Counter(raw_words)
    seen_counts = {}
    words_win   = []
    for w in raw_words:
        if word_counts[w] > 1:
            n = seen_counts.get(w, 0) + 1
            seen_counts[w] = n
            # Render duplicates as proper mathtext subscripts.
            words_win.append(rf"${w}_{{{n}}}$")
        else:
            words_win.append(w)
    im = ax_attn.imshow(sub, cmap="viridis", aspect="auto",
                        interpolation="nearest", vmin=0)
    ax_attn.set_xticks(range(win))
    ax_attn.set_xticklabels(words_win, fontsize=8, rotation=40,
                            ha="right", color="#333333")
    ax_attn.set_yticks(range(win))
    ax_attn.set_yticklabels(words_win, fontsize=8, color="#333333")
    cb = plt.colorbar(im, ax=ax_attn, shrink=0.8)
    cb.ax.tick_params(labelsize=8)
    ax_attn.add_patch(mpatches.Rectangle(
        (-0.5, win - 1.5), win, 1,
        fill=False, edgecolor=ANCHOR_COLOR, lw=1.2))

    # ── Top-5 predictions ────────────────────────────────────────────────────
    ax_pred.set_title("Next word predictions", fontsize=9,
                      loc="left", fontweight="bold")
    bar_words  = [w for w, _ in top5[:5]]
    bar_probs  = [p * 100 for _, p in top5[:5]]
    colors_bar = [ACTIVE_COLOR] + [SETTLED_COLOR] * 4
    ax_pred.barh([4 - i for i in range(5)], bar_probs,
                 color=colors_bar, alpha=0.80, edgecolor="none")
    for i, (w, p) in enumerate(zip(bar_words, bar_probs)):
        ax_pred.text(max(bar_probs) * 0.02, 4 - i,
                     f" {w}  {p:.2f}\\%", va="center",
                     fontsize=10, color="#222222")
    ax_pred.set_xlim(0, max(bar_probs) * 1.35)
    ax_pred.set_xlabel("Probability (\\%)", fontsize=9)
    ax_pred.set_yticks([])
    ax_pred.grid(True, axis="x", color="#e0e0e0", lw=0.5)

    fig.patch.set_facecolor("white")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, format="pdf", facecolor="white", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
