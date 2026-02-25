"""
plot_precision_recall_pairs.py

Grouped bar chart: models paired by similar size,
each pair shows precision and recall side by side.
Style matches the light-blue reference aesthetic.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
pairs = [
    ("deepseek-r1:1.5b", "qwen3:1.7b"),
    ("deepseek-r1:8b",   "qwen3:8b"),
    ("deepseek-r1:14b",  "qwen3:14b"),
    ("deepseek-r1:32b",  "qwen3:30b"),
]

metrics = {
    "deepseek-r1:1.5b": {"precision": 0.400, "recall": 0.320},
    "deepseek-r1:8b":   {"precision": 0.100, "recall": 0.040},
    "deepseek-r1:14b":  {"precision": 0.967, "recall": 0.330},
    "deepseek-r1:32b":  {"precision": 1.000, "recall": 0.370},
    "qwen3:1.7b":       {"precision": 1.000, "recall": 0.280},
    "qwen3:8b":         {"precision": 1.000, "recall": 0.240},
    "qwen3:14b":        {"precision": 1.000, "recall": 0.340},
    "qwen3:30b":        {"precision": 1.000, "recall": 0.490},
}

size_labels = ["~1.5B", "~8B", "~14B", "~30B"]

# ── Colors (matching reference style) ─────────────────────────────────────────
c_ds_prec = "#8fadd4"   # light steel blue
c_ds_rec  = "#4a72a8"   # medium blue
c_qw_prec = "#2e5090"   # dark blue
c_qw_rec  = "#1a2f5a"   # navy

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))

bg = "#d8e2f0"
fig.patch.set_facecolor(bg)
ax.set_facecolor(bg)

n = len(pairs)
bar_w = 0.15
x = np.arange(n)

for i, (ds_model, qw_model) in enumerate(pairs):
    ds = metrics[ds_model]
    qw = metrics[qw_model]

    bars = [
        (x[i] - 1.5 * bar_w, ds["precision"], c_ds_prec),
        (x[i] - 0.5 * bar_w, ds["recall"],    c_ds_rec),
        (x[i] + 0.5 * bar_w, qw["precision"], c_qw_prec),
        (x[i] + 1.5 * bar_w, qw["recall"],    c_qw_rec),
    ]

    for xpos, val, color in bars:
        ax.bar(xpos, val, width=bar_w * 0.88, color=color,
               edgecolor="none", zorder=3)

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(size_labels, fontsize=11, color="#2d3748", fontweight="bold")
ax.set_xlabel("Model Size", fontsize=12, color="#2d3748",
              fontweight="bold", labelpad=12)
ax.set_ylabel("Score", fontsize=12, color="#2d3748",
              fontweight="bold", labelpad=10)

ax.set_ylim(0, 1.08)
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
ax.tick_params(axis="y", labelsize=9, colors="#4a5568")
ax.tick_params(axis="x", length=0)

for spine in ax.spines.values():
    spine.set_visible(False)

# ── Legend ─────────────────────────────────────────────────────────────────────
legend_items = [
    Patch(facecolor=c_ds_prec, label="DeepSeek-R1 Precision"),
    Patch(facecolor=c_ds_rec,  label="DeepSeek-R1 Recall"),
    Patch(facecolor=c_qw_prec, label="Qwen3 Precision"),
    Patch(facecolor=c_qw_rec,  label="Qwen3 Recall"),
]
ax.legend(handles=legend_items, loc="upper left", frameon=False,
          fontsize=8.5, labelcolor="#2d3748", ncol=1)

plt.tight_layout()
plt.savefig("precision_recall_pairs.png", dpi=250, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved → precision_recall_pairs.png")
plt.close()