#!/usr/bin/env python3
"""
Generate improvement-over-DP-SGD bar chart for MICCAI 2026 paper.
Grouped bars: methods x datasets, showing delta Dice at eps=8.0.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Data: Dice improvement over DP-SGD at eps=8.0 (in percentage points)
# ---------------------------------------------------------------------------
methods = ["AdaDPS\n(oracle)", "AdaDPS\n(synth.)", "Shampoo\n(oracle)", "Shampoo\n(synth.)"]

# Kvasir: dp=0.453
kvasir_delta = [
    0.513 - 0.453,  # AdaDPS oracle
    0.510 - 0.453,  # AdaDPS synth
    0.503 - 0.453,  # Shampoo oracle
    0.509 - 0.453,  # Shampoo synth
]
kvasir_delta = [d * 100 for d in kvasir_delta]  # to pp

# Pet: dp=0.697
pet_delta = [
    0.661 - 0.697,  # AdaDPS oracle (NEGATIVE)
    0.697 - 0.697,  # AdaDPS synth
    0.715 - 0.697,  # Shampoo oracle
    0.714 - 0.697,  # Shampoo synth
]
pet_delta = [d * 100 for d in pet_delta]  # to pp

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

fig, ax = plt.subplots(figsize=(4.0, 2.2))

x = np.arange(len(methods))
width = 0.35

bars1 = ax.bar(x - width/2, kvasir_delta, width, label="Kvasir-SEG",
               color="#2d7bb6", edgecolor="white", linewidth=0.5)
bars2 = ax.bar(x + width/2, pet_delta, width, label="Oxford-IIIT Pet",
               color="#d62728", edgecolor="white", linewidth=0.5)

# Zero line
ax.axhline(0, color="black", linewidth=0.5, zorder=0)

# Value labels
for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.15,
            f"{h:+.1f}", ha="center", va="bottom", fontsize=7)
for bar in bars2:
    h = bar.get_height()
    offset = 0.15 if h >= 0 else -0.15
    va = "bottom" if h >= 0 else "top"
    ax.text(bar.get_x() + bar.get_width()/2, h + offset,
            f"{h:+.1f}", ha="center", va=va, fontsize=7)

ax.set_ylabel(r"$\Delta$ Dice (pp) vs DP-SGD")
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend(loc="upper left", frameon=True, edgecolor="lightgray")
ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
ax.set_ylim(-5, 8)

plt.tight_layout()
fig.savefig("paper/improvement_bars.pdf")
fig.savefig("paper/improvement_bars.png")
print("Saved paper/improvement_bars.pdf and paper/improvement_bars.png")
