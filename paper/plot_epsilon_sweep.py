#!/usr/bin/env python3
"""
Generate epsilon-sweep figure for MICCAI 2026 paper.
Two-panel plot: Kvasir-SEG (left) and Oxford-IIIT Pet (right).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Data (mean ± std, 3 seeds, 30 epochs)
# ---------------------------------------------------------------------------
epsilons = [1, 2, 4, 8]

# Kvasir-SEG
kvasir = {
    "DP-SGD":           {"mean": [0.2719, 0.2873, 0.4418, 0.4527],
                         "std":  [0.0046, 0.0076, 0.0069, 0.0114]},
    "AdaDPS (synth.)":  {"mean": [0.3367, 0.4520, 0.4947, 0.5100],
                         "std":  [0.0282, 0.0087, 0.0041, 0.0021]},
    "Shampoo (oracle)": {"mean": [0.2927, 0.4407, 0.4881, 0.5030],
                         "std":  [0.0250, 0.0156, 0.0037, 0.0043]},
    "Shampoo (synth.)": {"mean": [0.2818, 0.4430, 0.4903, 0.5091],
                         "std":  [0.0079, 0.0030, 0.0059, 0.0064]},
}
kvasir_baseline = {"mean": 0.7372, "std": 0.0157}

# Oxford-IIIT Pet
pet = {
    "DP-SGD":           {"mean": [0.6572, 0.6705, 0.6866, 0.6966],
                         "std":  [0.0059, 0.0087, 0.0035, 0.0042]},
    "AdaDPS (synth.)":  {"mean": [0.6587, 0.6734, 0.6865, 0.6968],
                         "std":  [0.0074, 0.0030, 0.0042, 0.0060]},
    "Shampoo (oracle)": {"mean": [0.6643, 0.6817, 0.7035, 0.7145],
                         "std":  [0.0041, 0.0017, 0.0116, 0.0094]},
    "Shampoo (synth.)": {"mean": [0.6673, 0.6831, 0.6919, 0.7136],
                         "std":  [0.0077, 0.0015, 0.0010, 0.0063]},
}
pet_baseline = {"mean": 0.8588, "std": 0.0041}

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
COLORS = {
    "DP-SGD":           "#888888",
    "AdaDPS (synth.)":  "#e07b39",
    "Shampoo (oracle)": "#2d7bb6",
    "Shampoo (synth.)": "#d62728",
}
MARKERS = {
    "DP-SGD":           "o",
    "AdaDPS (synth.)":  "s",
    "Shampoo (oracle)": "^",
    "Shampoo (synth.)": "D",
}
LINESTYLES = {
    "DP-SGD":           "--",
    "AdaDPS (synth.)":  "-.",
    "Shampoo (oracle)": "-",
    "Shampoo (synth.)": "-",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.5), sharey=False)

x = np.array(epsilons)

def plot_panel(ax, data, baseline, title, ylim):
    for method in ["DP-SGD", "AdaDPS (synth.)", "Shampoo (oracle)", "Shampoo (synth.)"]:
        m = np.array(data[method]["mean"])
        s = np.array(data[method]["std"])
        ax.errorbar(x, m, yerr=s,
                     label=method,
                     color=COLORS[method],
                     marker=MARKERS[method],
                     linestyle=LINESTYLES[method],
                     markersize=5, linewidth=1.3, capsize=3, capthick=1)

    # Baseline band
    bm, bs = baseline["mean"], baseline["std"]
    ax.axhspan(bm - bs, bm + bs, alpha=0.10, color="green", zorder=0)
    ax.axhline(bm, color="green", linewidth=0.8, linestyle=":", alpha=0.7,
               label="Non-private", zorder=0)

    ax.set_xlabel(r"Privacy budget $\varepsilon$")
    ax.set_ylabel("Dice score")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xticks(epsilons)
    ax.set_xticklabels([str(e) for e in epsilons])
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3, linewidth=0.5)

plot_panel(ax1, kvasir, kvasir_baseline,
           "Kvasir-SEG (polyp)", (0.22, 0.80))
plot_panel(ax2, pet, pet_baseline,
           "Oxford-IIIT Pet", (0.62, 0.90))

# Single shared legend below
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5,
           bbox_to_anchor=(0.5, -0.08), frameon=False, columnspacing=1.0)

plt.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig("paper/epsilon_sweep.pdf")
fig.savefig("paper/epsilon_sweep.png")
print("Saved paper/epsilon_sweep.pdf and paper/epsilon_sweep.png")
