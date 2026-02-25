"""
plot_judge_passrate.py

Grouped bar chart: LLM-as-Judge pass rates (answer_support & clinical_relevance)
paired by model size, DeepSeek-R1 vs Qwen3.

Expects a CSV with columns: rep, gene, model, num_cited, answer_support, clinical_relevance_support
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from pathlib import Path


def plot_judge_passrate(
    csv_path: str = "C://Users//aemekkawi//Documents//GitHub//matching-system//evaluation_results//field_support_scores.csv",
    output_path: str = "judge_passrate.png",
    dpi: int = 250,
):
    df = pd.read_csv(csv_path)

    BG = "#d8e2f0"

    pairs = [
        ("deepseek-r1:1.5b", "qwen3:1.7b"),
        ("deepseek-r1:8b",   "qwen3:8b"),
        ("deepseek-r1:14b",  "qwen3:14b"),
        ("deepseek-r1:32b",  "qwen3:30b"),
    ]
    size_labels = ["~1.5B", "~8B", "~14B", "~30B"]

    all_models = [m for p in pairs for m in p]

    # Compute pass rates
    pass_rates = {}
    for model in all_models:
        mdf = df[df["model"] == model]
        n = len(mdf)
        pass_rates[model] = {
            "answer":   mdf["answer_support"].sum() / n if n else 0,
            "clinical": mdf["clinical_relevance_support"].sum() / n if n else 0,
        }

    # Colors
    c_ds_ans  = "#8fadd4"
    c_ds_clin = "#4a72a8"
    c_qw_ans  = "#2e5090"
    c_qw_clin = "#1a2f5a"

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    n_pairs = len(pairs)
    bar_w = 0.15
    x = np.arange(n_pairs)

    for i, (ds_m, qw_m) in enumerate(pairs):
        ds = pass_rates[ds_m]
        qw = pass_rates[qw_m]
        bars_data = [
            (x[i] - 1.5 * bar_w, ds["answer"],   c_ds_ans),
            (x[i] - 0.5 * bar_w, ds["clinical"],  c_ds_clin),
            (x[i] + 0.5 * bar_w, qw["answer"],    c_qw_ans),
            (x[i] + 1.5 * bar_w, qw["clinical"],  c_qw_clin),
        ]
        for xp, val, col in bars_data:
            ax.bar(xp, val, width=bar_w * 0.88, color=col, edgecolor="none", zorder=3)

    # Axes
    ax.set_xticks(x)
    ax.set_xticklabels(size_labels, fontsize=11, color="#2d3748", fontweight="bold")
    ax.set_xlabel("Model Size", fontsize=12, color="#2d3748", fontweight="bold", labelpad=10)
    ax.set_ylabel("Pass Rate", fontsize=12, color="#2d3748", fontweight="bold", labelpad=10)
    ax.set_ylim(0, 1.)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.tick_params(axis="y", labelsize=9, colors="#4a5568")
    ax.tick_params(axis="x", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Legend at top center, well above bars
    legend_items = [
        Patch(facecolor=c_ds_ans,  label="DS-R1 Answer Support"),
        Patch(facecolor=c_ds_clin, label="DS-R1 Clinical Relevance"),
        Patch(facecolor=c_qw_ans,  label="Qwen3 Answer Support"),
        Patch(facecolor=c_qw_clin, label="Qwen3 Clinical Relevance"),
    ]
    ax.legend(
        handles=legend_items, loc="upper center", frameon=False,
        fontsize=8.5, labelcolor="#2d3748", ncol=2,
        bbox_to_anchor=(0.5, 1.0),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    plot_judge_passrate()