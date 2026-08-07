"""Figure 6: the three-family comparison -- what transfers and what doesn't.

Panel A  selectivity: the load-bearing effect reproduces everywhere, the
         reasoning/prediction dissociation only on Claude.
Panel B  swap vs broadcast: near-identical swap on the two open models, ~2x
         apart on broadcast.
Panel C  white bear, protocol-matched: three families, three points.
"""
# NOTE: paths under /data3 (model weights, fitted lenses in /data3/fan-test/jlens_out)
# are machine-specific to the original 8xB200 host; adjust for your environment.
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
TEAL, ORANGE, GRAY = "#0891b2", "#c2410c", "#8a9aa4"
PURPLE = "#7c3aed"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e8edf0", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.facecolor": "white",
})
MODELS = ["Claude\n(paper)", "DeepSeek-V4\n4-stream mHC", "GLM-5.2\n1-stream MoE"]
x = np.arange(3)
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13, 3.6))

# ---- A: selectivity
w = 0.36
reason_drop = [np.nan, 1 - 0.100 / 0.589, 1 - 0.167 / 0.611]   # relative collapse
agree = [0.92, 0.508, 0.498]                                    # >0.90 reported for Claude
# Claude's collapse is reported qualitatively in the paper, not as a rate we can
# put on this axis -- draw it hatched and unnumbered rather than inventing a value.
heights = [0.0 if np.isnan(v) else v for v in reason_drop]
b1 = a1.bar(x - w / 2, heights, w, color=TEAL,
            label="Reasoning collapse under ablation")
a1.bar([x[0] - w / 2], [1.0], w, color="none", edgecolor=TEAL, hatch="///",
       linewidth=1.0)
b2 = a1.bar(x + w / 2, agree, w, color=ORANGE,
            label="Automatic prediction RETAINED")
a1.axhline(0.90, color=GRAY, ls="--", lw=0.9)
a1.text(2.42, 0.92, "selective", fontsize=7.6, color=GRAY, ha="right")
for r, v in zip(b1, reason_drop, strict=False):
    a1.text(r.get_x() + r.get_width() / 2,
            (1.0 if np.isnan(v) else r.get_height()) + 0.02,
            "reported,\nnot rated" if np.isnan(v) else f"{v:.2f}",
            ha="center", fontsize=7.4)
for r, v in zip(b2, agree, strict=False):
    a1.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.02,
            ">0.90" if v == 0.92 else f"{v:.3f}", ha="center", fontsize=7.4)
a1.set_xticks(x, MODELS, fontsize=8)
a1.set_ylim(0, 1.45)
a1.set_title("A · Load-bearing everywhere,\nselective only on Claude", fontsize=10)
a1.legend(frameon=False, fontsize=7.4, loc="upper center", ncols=1)

# ---- B: swap vs broadcast
swap = [0.62, 0.434, 0.455]        # paper midpoint of 0.54-0.70
bcast = [0.526, 0.208, 0.375]
b3 = a2.bar(x - w / 2, swap, w, color=TEAL, label="Swap (immediate answer)")
b4 = a2.bar(x + w / 2, bcast, w, color=PURPLE, label="Broadcast (other downstream fn)")
for bars in (b3, b4):
    for r in bars:
        a2.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.012,
                f"{r.get_height():.3f}", ha="center", fontsize=7.6)
a2.set_xticks(x, MODELS, fontsize=8)
a2.set_ylim(0, 0.86)
a2.set_title("B · Swap matches across open models,\nbroadcast does not", fontsize=10)
a2.legend(frameon=False, fontsize=7.4, loc="upper right")
a2.annotate("", xy=(1.18, 0.208), xytext=(2.18, 0.375),
            arrowprops=dict(arrowstyle="<->", color=GRAY, lw=0.9))
a2.text(1.68, 0.335, "1.8x", fontsize=8, color=GRAY, ha="center",
        bbox=dict(fc="white", ec="none", pad=1.2))

# ---- C: white bear
wb = [0.35, 0.000, 0.917]
cols = [GRAY, ORANGE, TEAL]
b5 = a3.bar(x, wb, 0.5, color=cols)
for r, v in zip(b5, wb, strict=False):
    a3.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.02,
            "partial\n(reported)" if v == 0.35 else f"{v:.3f}",
            ha="center", fontsize=7.8)
a3.set_xticks(x, MODELS, fontsize=8)
a3.set_ylim(0, 1.18)
a3.set_title("C · \"Don't think about X\":\nthree families, three behaviours", fontsize=10, pad=14)
a3.set_ylabel("Concept still active while answering")
a3.text(0.5, -0.46, "V4 and GLM measured under an identical protocol: generated\n"
        "positions only, verbalization-controlled, absent-concept floor 0.000",
        transform=a3.transAxes, ha="center", fontsize=7.2, color=GRAY)
for ax, letter in zip((a1, a2, a3), "ABC", strict=False):
    ax.text(-0.09, 1.16, letter, transform=ax.transAxes, fontsize=12,
            fontweight="bold")
fig.suptitle("Three model families: the workspace transfers, its division of labour does not",
             y=1.10, fontsize=11.5)
fig.tight_layout()
fig.subplots_adjust(bottom=0.28)
fig.savefig(f"{OUT}/fig6_three_families.png", dpi=170, bbox_inches="tight")
print("fig6 written")
