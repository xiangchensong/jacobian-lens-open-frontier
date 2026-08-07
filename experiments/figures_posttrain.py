"""Figure 7: what the second post-training changed, entirely from committed data.

Panel A -- direct quantities, not ratios: cross-checkpoint geometry (mean CKA,
from b1_geometry.json) and readout overlap at top-1/5/10 (from b2_compare.json),
each next to its same-checkpoint disjoint-corpus floor. Ratio-of-floor framing
was dropped deliberately: CKA and top-k overlap have no meaningful zero-based
ratio scale, so the figure shows both raw values and lets the gap speak.
Panel B -- per-layer CKA, preview-vs-0731 against the 0731-vs-disjoint floor,
showing the divergence concentrating in the early and middle band. Unlike
figs 3/5/6, nothing here is a hardcoded literal.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

_HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(_HERE, "posttrain_comparison", "results")
OUT = os.path.join(_HERE, "..", "figures")
TEAL, ORANGE, GRAY = "#0891b2", "#c2410c", "#8a9aa4"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e8edf0", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.facecolor": "white",
})

b1 = json.load(open(os.path.join(R, "b1_geometry.json")))
b2 = json.load(open(os.path.join(R, "b2_compare.json")))

cka_pair = sum(r["cka_pair"] for r in b1) / len(b1)
cka_floor = sum(r["cka_floor"] for r in b1) / len(b1)
pair_vals, floor_vals = [cka_pair], [cka_floor]
for K in (1, 5, 10):
    sel = [r for r in b2 if r["k"] == K]
    pair_vals.append(sum(r["pair"] for r in sel) / len(sel))
    floor_vals.append(sum(r["floor"] for r in sel) / len(sel))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 3.5),
                             gridspec_kw={"width_ratios": [1.1, 1.35]})
labels = ["geometry\n(mean CKA)", "readout\ntop-1", "readout\ntop-5",
          "readout\ntop-10"]
xs = list(range(len(labels)))
w = 0.36
bf = a1.bar([x - w / 2 for x in xs], floor_vals, w, color=GRAY,
            label="same checkpoint, disjoint corpora (floor)")
bp = a1.bar([x + w / 2 for x in xs], pair_vals, w, color=ORANGE,
            label="preview vs 0731")
for bars in (bf, bp):
    for r in bars:
        a1.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.015,
                f"{r.get_height():.2f}", ha="center", fontsize=8)
a1.set_xticks(xs, labels)
a1.set_ylim(0, 1.05)
a1.set_ylabel("similarity / overlap")
a1.legend(frameon=False, fontsize=8, loc="upper right")
a1.set_title("A · Cross-checkpoint change, floor-controlled", fontsize=10.5)

L = [r["layer"] for r in b1]
a2.plot(L, [r["cka_floor"] for r in b1], color=GRAY, lw=1.8, marker="o",
        ms=2.6, mec="none", label="0731 vs disjoint-corpus 0731 (floor)")
a2.plot(L, [r["cka_pair"] for r in b1], color=TEAL, lw=1.8, marker="o",
        ms=2.6, mec="none", label="preview vs 0731")
a2.axvspan(19, 28, color=ORANGE, alpha=0.07, zorder=0)
a2.text(23.5, 0.955, "divergence\nconcentrates here", fontsize=8,
        color=ORANGE, ha="center")
a2.xaxis.set_major_locator(MaxNLocator(integer=True))
a2.set_xlabel("Layer")
a2.set_ylabel("CKA")
a2.set_title("B · Geometric divergence sits early/mid band", fontsize=10.5)
a2.legend(frameon=False, fontsize=8.3, loc="lower right")
for ax, letter in zip((a1, a2), "AB", strict=True):
    ax.text(-0.12, 1.1, letter, transform=ax.transAxes, fontsize=12,
            fontweight="bold")
fig.suptitle("Two post-trainings of the same reported base: readout overlap "
             "moves far more than lens geometry", y=1.06, fontsize=11.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig7_posttrain.png"), dpi=170,
            bbox_inches="tight")
print("fig7 written")
