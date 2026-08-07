"""Figures for the write-up, modeled on the paper's, drawn from our measurements.

Paper figure -> ours:
  Fig 28 (workspace onset/offset, 4 panels)  -> fig1 from exp62_stats.json
  Fig 29 (ignition: share heatmap + width)   -> fig2 from exp63_ignition.json
  Fig 22/25 (ablation + experiential)        -> fig3 from measured 5.3/5.4 numbers
  Fig 17 (arithmetic rank trajectory)        -> fig4 from exp35_traj_clean.txt
  (methods comparison, appendix)             -> fig5 measured six-set table

Style follows the paper: white ground, light gray grid, muted band shading,
panel letters, direct labels instead of legends where possible. Palette is the
pair validated earlier for CVD safety: teal #0891b2 / orange #c2410c.
"""
# NOTE: paths under /data3 (model weights, fitted lenses in /data3/fan-test/jlens_out)
# are machine-specific to the original 8xB200 host; adjust for your environment.

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
S = os.path.join(_HERE, "results")            # measured inputs, committed
OUT = os.path.join(_HERE, "..", "figures")  # repo figures/ (blog images)
os.makedirs(OUT, exist_ok=True)

TEAL, ORANGE, GRAY = "#0891b2", "#c2410c", "#8a9aa4"
BAND = (19, 39)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e8edf0", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.facecolor": "white",
})


def shade_band(ax):
    ax.axvspan(BAND[0], BAND[1], color=TEAL, alpha=0.07, zorder=0)


# ---------------------------------------------------------------- fig 1: onset/offset
rows = json.load(open(f"{S}/exp62_stats.json"))
L = [r["layer"] for r in rows]
fig, axes = plt.subplots(1, 4, figsize=(12.5, 2.9))
panels = [
    ("top1", "Top-1 accuracy", None),
    ("kurt", "Excess kurtosis", None),
    ("excess", "Excess autocorrelation", 0.0),
    ("eff_dim", "Effective dimensionality", None),
]
for ax, (key, title, hline) in zip(axes, panels, strict=False):
    ax.plot(L, [r[key] for r in rows], color=TEAL, lw=1.8,
            marker="o", ms=2.6, mec="none")
    shade_band(ax)
    if hline is not None:
        ax.axhline(hline, color=GRAY, lw=0.8, ls="--")
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel("Layer")
    ax.set_xlim(-1, 43)
for ax, letter in zip(axes, "ABCD", strict=False):
    ax.text(-0.08, 1.12, letter, transform=ax.transAxes,
            fontsize=12, fontweight="bold")
axes[0].annotate("workspace\nL19–39", xy=(29, axes[0].get_ylim()[1] * 0.75),
                 ha="center", fontsize=8.5, color=TEAL)
fig.suptitle("Workspace onset and offset on DeepSeek-V4-Flash-0731 "
             "(cf. paper Fig. 28)", y=1.06, fontsize=11.5)
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_onset_offset.png", dpi=170, bbox_inches="tight")
plt.close(fig)
print("fig1 done")

# ---------------------------------------------------------------- fig 2: ignition
ig = json.load(open(f"{S}/exp63_ignition.json"))
layers = sorted(int(k) for k in ig)
alphas = np.linspace(0, 1, 11)
M = np.array([ig[str(l)]["means"] for l in layers])       # [n_layers, 11]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.4),
                             gridspec_kw={"width_ratios": [1.35, 1]})
im = a1.imshow(M, aspect="auto", origin="lower", cmap="RdBu_r",
               vmin=0, vmax=1,
               extent=[alphas[0] - .05, alphas[-1] + .05,
                       layers[0] - .5, layers[-1] + .5])
a1.axhline(BAND[0], color="k", lw=0.9, ls="--")
a1.axhline(BAND[1], color="k", lw=0.9, ls="--")
a1.text(1.02, BAND[0], "L19", va="center", fontsize=8.5,
        transform=a1.get_yaxis_transform())
a1.text(1.02, BAND[1], "L39", va="center", fontsize=8.5,
        transform=a1.get_yaxis_transform())
a1.set_xlabel("Mixture weight α toward concept A")
a1.set_ylabel("Layer")
a1.set_title("Reciprocal-rank share of concept A", fontsize=10.5)
a1.grid(False)
fig.colorbar(im, ax=a1, fraction=0.04, pad=0.1)

widths = [ig[str(l)]["width"] for l in layers]
a2.plot([w if w == w else None for w in widths], layers, color=ORANGE, lw=1.8,
        marker="o", ms=3.4, mec="none")
a2.axhspan(BAND[0], BAND[1], color=TEAL, alpha=0.07, zorder=0)
# mark layers that never cross 10->90 with open circles at the right edge
for l, w in zip(layers, widths, strict=False):
    if w != w:  # NaN
        a2.plot(1.02, l, marker="o", ms=3.4, mfc="white", mec=GRAY,
                clip_on=False)
a2.text(0.97, 4, "never\ncrosses", fontsize=8, color=GRAY, ha="right")
a2.set_xlabel("10→90% transition width in α\n(lower = sharper commitment)")
a2.set_ylabel("Layer")
a2.set_xlim(0, 1.05)
a2.set_title("Ignition sharpness", fontsize=10.5)
for ax, letter in zip((a1, a2), "AB", strict=False):
    ax.text(-0.1, 1.09, letter, transform=ax.transAxes,
            fontsize=12, fontweight="bold")
fig.suptitle("Commitment to one reading of an ambiguous token "
             "(cf. paper Fig. 29)", y=1.05, fontsize=11.5)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_ignition.png", dpi=170, bbox_inches="tight")
plt.close(fig)
print("fig2 done")

# ---------------------------------------------------------------- fig 3: ablation
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.3))
conds = ["unablated", "J-space\nablated", "random\ncontrol"]
multihop = [0.589, 0.100, 0.589]
agree = [1.000, 0.508, 0.922]
x = np.arange(3)
w = 0.36
b1 = a1.bar(x - w / 2, multihop, w, color=TEAL, label="Multi-hop accuracy")
b2 = a1.bar(x + w / 2, agree, w, color=ORANGE,
            label="Pretraining top-1 agreement")
for bars in (b1, b2):
    for r in bars:
        a1.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.02,
                f"{r.get_height():.3f}", ha="center", fontsize=8.3)
a1.set_xticks(x, conds)
a1.set_ylim(0, 1.42)
a1.set_title("Ablating top-10 lens directions, L19–39", fontsize=10.5)
a1.legend(frameon=False, fontsize=8.5, loc="upper right", ncols=1,
          borderaxespad=0.1)

conds2 = ["unablated", "J-space\nL19–25", "J-space\nL19–39", "random\nL19–39"]
exp = [0.778, 0.889, 0.222, 1.000]
story = [0.333, 0.667, 0.000, 0.333]
x2 = np.arange(4)
b3 = a2.bar(x2 - w / 2, exp, w, color=TEAL, label="Experiential language")
b4 = a2.bar(x2 + w / 2, story, w, color=ORANGE, label="Story quality (control)")
for bars in (b3, b4):
    for r in bars:
        a2.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.02,
                f"{r.get_height():.2f}", ha="center", fontsize=8.3)
a2.set_xticks(x2, conds2, fontsize=8.6)
a2.set_ylim(0, 1.42)
a2.set_title("Experiential-language ablation (LLM-graded)", fontsize=10.5)
a2.legend(frameon=False, fontsize=8.5, loc="upper right")
for ax, letter in zip((a1, a2), "AB", strict=False):
    ax.text(-0.09, 1.09, letter, transform=ax.transAxes,
            fontsize=12, fontweight="bold")
fig.suptitle("The workspace is load-bearing — but not selective "
             "(cf. paper Figs. 22, 25)", y=1.06, fontsize=11.5)
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_ablation.png", dpi=170, bbox_inches="tight")
plt.close(fig)
print("fig3 done")

# ---------------------------------------------------------------- fig 4: arithmetic
traj = np.loadtxt(f"{S}/exp35_traj_clean.txt")   # layer, r21, r42, r49
fig, ax = plt.subplots(figsize=(7.2, 3.4))
labels = ["21  (4+17)", "42  (·2)", "49  (+7)"]
colors = [TEAL, ORANGE, "#7c3aed"]
for i, (lab, c) in enumerate(zip(labels, colors, strict=False)):
    ax.plot(traj[:, 0], traj[:, i + 1] + 1, color=c, lw=1.8,
            marker="o", ms=2.6, mec="none", label=lab)
shade_band(ax)
ax.set_yscale("log")
ax.invert_yaxis()
ax.set_xlabel("Layer")
ax.set_ylabel("Lens rank of intermediate (log, 1 = top)")
ax.set_xlim(-1, 43)
ax.legend(frameon=False, fontsize=9, title="calc: (4+17)*2+7 =",
          title_fontsize=9, loc="lower left")
# staggered so the three labels cannot collide
for val, layer, tx, ty in (("21", 27, 21.0, 2.6), ("42", 30, 29.0, 14.0),
                           ("49", 36, 37.5, 2.6)):
    ax.annotate(f"{val} → rank 1 @ L{layer}", xy=(layer, 1),
                xytext=(tx, ty), fontsize=8,
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8))
ax.annotate("consumed intermediates\nare cleared", xy=(37, 8e4),
            xytext=(25.5, 2.2e4), fontsize=8, color=GRAY,
            arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8))
ax.set_title("Unspoken intermediates resolve in computational order "
             "(cf. paper Fig. 17)", fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT}/fig4_arithmetic.png", dpi=170, bbox_inches="tight")
plt.close(fig)
print("fig4 done")

# ---------------------------------------------------------------- fig 5: methods
sets = ["assoc.", "typo", "multihop", "multiling.", "order-ops", "poetry"]
# All three columns use the same English-only target sets, matching the
# write-up's table. A separate multilingual-augmented run raises every
# method's multihop/multilingual scores without changing their ordering.
jlens = [0.069, 0.125, 0.294, 0.105, 0.145, 0.031]
# logit column from the SAME eval run as the J-lens column (English-only
# target sets). An earlier revision took it from the tuned-lens eval run,
# whose config differs (its deterministic vanilla scores diverge on 4/6
# sets) -- a cross-run splice caught in editorial review.
logit = [0.010, 0.052, 0.358, 0.213, 0.291, 0.031]
tuned = [0.000, 0.010, 0.233, 0.157, 0.191, 0.000]
x = np.arange(6)
w = 0.26
fig, ax = plt.subplots(figsize=(8.6, 3.3))
ax.bar(x - w, jlens, w, color=TEAL, label="J-lens")
ax.bar(x, logit, w, color=ORANGE, label="Logit lens")
ax.bar(x + w, tuned, w, color=GRAY, label="Tuned lens (converged)")
ax.set_xticks(x, sets)
ax.set_ylabel("pass@1")
ax.legend(frameon=False, fontsize=9)
ax.set_title("Method comparison on the six lens-quality sets — the tuned lens "
             "collapses exactly on unverbalized content", fontsize=10.5)
# mark the unverbalized sets
for xi in (0, 1, 5):
    ax.annotate("unspoken", xy=(xi, max(jlens[xi], logit[xi]) + 0.015),
                ha="center", fontsize=7.5, color=GRAY)
fig.tight_layout()
fig.savefig(f"{OUT}/fig5_methods.png", dpi=170, bbox_inches="tight")
plt.close(fig)
print("fig5 done")
print("ALL FIGURES WRITTEN to", OUT)
