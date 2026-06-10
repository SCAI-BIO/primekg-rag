from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import log


MODEL_GROUPS = [
    ("0.6b", [("qwen3:0.6b", "#e45756")]),
    ("~1b", [("llama3.2:1b", "#54a24b")]),
    ("~1.5b", [("deepseek-r1:1.5b", "#4c78a8"), ("qwen3:1.7b", "#e45756")]),
    ("3b", [("llama3.2:3b", "#54a24b")]),
    ("8b", [("deepseek-r1:8b", "#4c78a8"), ("qwen3:8b", "#e45756"), ("llama3.1:8b", "#54a24b")]),
    ("14b", [("deepseek-r1:14b", "#4c78a8"), ("qwen3:14b", "#e45756")]),
    ("~32b", [("deepseek-r1:32b", "#4c78a8"), ("qwen3:30b", "#e45756")]),
]

FAMILY_COLORS = {
    "deepseek-r1": "#4c78a8",
    "qwen3": "#e45756",
    "llama3.1": "#54a24b",
    "llama3.2": "#54a24b",
}


def _get_family(model_name: str) -> str:
    return model_name.rsplit(":", 1)[0]


def _short_model_name(model_name: str) -> str:
    return (
        model_name.split(":")[0]
        .replace("deepseek-r1", "DS")
        .replace("qwen3", "QW")
        .replace("llama3.2", "LL3.2")
        .replace("llama3.1", "LL3.1")
    )


def _bar_by_model(
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    out_name: str,
    output_dir: Path,
    ylim=(0, 1.1),
):
    means = df.groupby("model")[metric].mean()
    present = set(means.index)

    positions, vals, tick_labels, colors = [], [], [], []
    pos = 0

    for size_lbl, models_in_tier in MODEL_GROUPS:
        tier_has_any = False

        for model_name, color in models_in_tier:
            if model_name in present:
                positions.append(pos)
                vals.append(means[model_name])
                tick_labels.append(f"{_short_model_name(model_name)} {size_lbl}")
                colors.append(color)
                pos += 1
                tier_has_any = True

        if tier_has_any:
            pos += 0.5

    if not vals:
        return

    fig, ax = plt.subplots(figsize=(max(9, len(vals) * 0.8), 5))
    bars = ax.bar(positions, vals, width=0.7, color=colors, alpha=0.85)

    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.02,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)

    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color="#4c78a8", label="DeepSeek-R1"),
            Patch(color="#e45756", label="Qwen3"),
            Patch(color="#54a24b", label="Llama"),
        ],
        fontsize=8,
    )

    fig.tight_layout()

    out = output_dir / out_name
    fig.savefig(out, dpi=150)
    plt.close(fig)

    log.info(f"Saved {out}")


def plot_metric_by_model(df: pd.DataFrame, metric: str, title: str, out_name: str, output_dir: Path):
    _bar_by_model(df, metric, title, metric, out_name, output_dir)


def _boxplot_by_model(
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    out_name: str,
    output_dir: Path,
    ylim=(-0.05, 1.15),
):
    present = set(df["model"].unique())

    positions, data, tick_labels, colors = [], [], [], []
    pos = 1

    for size_lbl, models_in_tier in MODEL_GROUPS:
        tier_has_any = False

        for model_name, color in models_in_tier:
            if model_name in present:
                vals = df[df["model"] == model_name][metric].values

                if len(vals) > 0:
                    data.append(vals)
                    positions.append(pos)
                    tick_labels.append(f"{_short_model_name(model_name)} {size_lbl}\nn={len(vals)}")
                    colors.append(color)
                    pos += 1
                    tier_has_any = True

        if tier_has_any:
            pos += 0.5

    if not data:
        log.warning(f"No data for boxplot: {title}")
        return

    fig, ax = plt.subplots(figsize=(max(12, len(data) * 0.95), 6))
    bp = ax.boxplot(
        data,
        positions=positions,
        patch_artist=True,
        widths=0.6,
    )

    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, rotation=0, ha="center", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25)

    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color="#4c78a8", label="DeepSeek-R1"),
            Patch(color="#e45756", label="Qwen3"),
            Patch(color="#54a24b", label="Llama"),
        ]
    )

    fig.tight_layout()

    out = output_dir / out_name
    fig.savefig(out, dpi=150)
    plt.close(fig)

    log.info(f"Saved {out}")


def plot_beeswarm_by_model(
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    out_name: str,
    output_dir: Path,
    ylim=(-0.05, 1.15),
):
    present = set(df["model"].unique())

    ordered_models = []
    tick_labels = []

    for size_lbl, models_in_tier in MODEL_GROUPS:
        for model_name, _ in models_in_tier:
            if model_name in present:
                ordered_models.append(model_name)
                n_points = int((df["model"] == model_name).sum())
                tick_labels.append(f"{_short_model_name(model_name)} {size_lbl}\nn={n_points}")

    if not ordered_models:
        log.warning(f"No data for beeswarm plot: {title}")
        return

    plot_df = df[df["model"].isin(ordered_models)].copy()
    plot_df["model"] = pd.Categorical(
        plot_df["model"],
        categories=ordered_models,
        ordered=True,
    )

    plot_df["family"] = plot_df["model"].astype(str).apply(_get_family)

    fig, ax = plt.subplots(figsize=(max(14, len(ordered_models) * 1.05), 6))

    sns.swarmplot(
        data=plot_df,
        x="model",
        y=metric,
        hue="family",
        size=2,
        linewidth=0,
        ax=ax,
    )

    ax.set_xticks(range(len(ordered_models)))
    ax.set_xticklabels(tick_labels, rotation=0, ha="center", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25)

    if ax.get_legend():
        ax.get_legend().remove()

    fig.tight_layout()

    out = output_dir / out_name
    fig.savefig(out, dpi=150)
    plt.close(fig)

    log.info(f"Saved {out}")


def plot_distribution_bin_heatmap(
    bin_df: pd.DataFrame,
    metric: str,
    title: str,
    out_name: str,
    output_dir: Path,
):
    percent_cols = [c for c in bin_df.columns if c.startswith("percent_")]

    if not percent_cols:
        log.warning(f"No percent bin columns found for heatmap: {title}")
        return

    plot_df = bin_df.copy()
    plot_df["model_order"] = pd.Categorical(
        plot_df["model"],
        categories=[
            model_name
            for _, models_in_tier in MODEL_GROUPS
            for model_name, _ in models_in_tier
            if model_name in set(plot_df["model"])
        ],
        ordered=True,
    )
    plot_df = plot_df.sort_values("model_order")

    heatmap_df = plot_df.set_index("model")[percent_cols]
    heatmap_df.columns = [c.replace("percent_", "") for c in percent_cols]

    fig, ax = plt.subplots(figsize=(12, max(6, len(heatmap_df) * 0.45)))

    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".1f",
        cmap="viridis",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "% of rows"},
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel(f"{metric} bin")
    ax.set_ylabel("Model")

    fig.tight_layout()

    out = output_dir / out_name
    fig.savefig(out, dpi=150)
    plt.close(fig)

    log.info(f"Saved {out}")


def plot_boxplot_precision(run_df: pd.DataFrame, output_dir: Path):
    _boxplot_by_model(
        run_df,
        "pmid_precision",
        "PMID Precision per Gene x Rep",
        "Precision",
        "boxplot_pmid_precision.png",
        output_dir,
    )


def plot_boxplot_recall(run_df: pd.DataFrame, output_dir: Path):
    _boxplot_by_model(
        run_df,
        "pmid_recall",
        "PMID Recall per Gene x Rep",
        "Recall",
        "boxplot_pmid_recall.png",
        output_dir,
    )


def plot_boxplot_structure(run_df: pd.DataFrame, output_dir: Path):
    _boxplot_by_model(
        run_df,
        "fields_present",
        "Structure Adherence per Gene x Rep",
        "Fields Present (0-4)",
        "boxplot_structure_fields.png",
        output_dir,
        ylim=(-0.2, 4.6),
    )


def plot_structure_by_model(df: pd.DataFrame, output_dir: Path):
    cols = [
        "answer_present",
        "key_mechanisms_present",
        "limitations_present",
        "clinical_relevance_present",
    ]

    labels_short = ["answer", "mechanisms", "limitations", "clinical_rel"]

    g = df.groupby("model")[cols].mean()
    present = set(g.index)

    tiers = []

    for size_lbl, models_in_tier in MODEL_GROUPS:
        tier_models = [m for m, _ in models_in_tier if m in present]
        if tier_models:
            tiers.append((size_lbl, tier_models))

    if not tiers:
        return

    fig, axes = plt.subplots(
        1,
        len(tiers),
        figsize=(4 * len(tiers), 5),
        sharey=True,
        squeeze=False,
    )

    for idx, (size_lbl, tier_models) in enumerate(tiers):
        ax = axes[0][idx]

        n_models = len(tier_models)
        x = list(range(len(cols)))
        width = 0.8 / n_models

        for m_idx, model_name in enumerate(tier_models):
            offsets = [
                xi + (m_idx - n_models / 2 + 0.5) * width
                for xi in x
            ]

            model_vals = [g.loc[model_name, c] for c in cols]
            family = model_name.rsplit(":", 1)[0]
            color = FAMILY_COLORS.get(family, "#999999")

            ax.bar(
                offsets,
                model_vals,
                width,
                label=_short_model_name(model_name),
                color=color,
                alpha=0.85,
            )

        ax.set_title(size_lbl, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels_short, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1.15)

        if idx == 0:
            ax.set_ylabel("Pass Rate")
            ax.legend(fontsize=7)

    fig.suptitle("Structure Section Presence by Model", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out = output_dir / "structure_sections_by_model.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    log.info(f"Saved {out}")


def plot_structure_breakdown(df: pd.DataFrame, output_dir: Path):
    cols = [
        "answer_present",
        "key_mechanisms_present",
        "limitations_present",
        "clinical_relevance_present",
    ]

    labels = ["answer", "mechanisms", "limitations", "clinical_rel"]
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]

    models = sorted(df["model"].unique())
    n = len(models)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 4 * nrows),
        squeeze=False,
    )

    for idx, model in enumerate(models):
        ax = axes[idx // ncols][idx % ncols]
        mdf = df[df["model"] == model]

        rates = [mdf[c].mean() for c in cols]
        counts = [int(mdf[c].sum()) for c in cols]

        bars = ax.bar(labels, rates, color=colors)

        for bar, rate, count in zip(bars, rates, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                rate + 0.03,
                f"{count}/{len(mdf)}",
                ha="center",
                fontsize=9,
                fontweight="bold",
            )

        ax.set_title(model, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.2)
        ax.set_ylabel("Pass Rate")
        ax.tick_params(axis="x", rotation=30)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Structure Field Breakdown by Model", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = output_dir / "structure_field_breakdown.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    log.info(f"Saved {out}")


def plot_scaling_curve(
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    out_name: str,
    output_dir: Path,
):
    size_map = {
        "qwen3:0.6b": 0.6,
        "llama3.2:1b": 1.0,
        "deepseek-r1:1.5b": 1.5,
        "qwen3:1.7b": 1.7,
        "llama3.2:3b": 3.0,
        "deepseek-r1:8b": 8.0,
        "qwen3:8b": 8.0,
        "llama3.1:8b": 8.0,
        "deepseek-r1:14b": 14.0,
        "qwen3:14b": 14.0,
        "deepseek-r1:32b": 32.0,
        "qwen3:30b": 30.0,
    }

    family_styles = {
        "deepseek-r1": {
            "color": "#4c78a8",
            "marker": "o",
            "label": "DeepSeek-R1",
        },
        "qwen3": {
            "color": "#1a2f5a",
            "marker": "s",
            "label": "Qwen3",
        },
        "llama": {
            "color": "#e57a2a",
            "marker": "^",
            "label": "Llama",
        },
    }

    means = df.groupby("model")[metric].mean()

    families = {}

    for model_name, mean_val in means.items():
        if model_name not in size_map:
            continue

        family = model_name.rsplit(":", 1)[0]

        if family.startswith("llama"):
            family = "llama"

        if family not in families:
            families[family] = {
                "sizes": [],
                "vals": [],
                "labels": [],
            }

        families[family]["sizes"].append(size_map[model_name])
        families[family]["vals"].append(mean_val)
        families[family]["labels"].append(f"{mean_val:.2f}")

    all_sizes = sorted(set(s for d in families.values() for s in d["sizes"]))
    size_to_pos = {s: i for i, s in enumerate(all_sizes)}

    fig, ax = plt.subplots(figsize=(10, 6))

    bg = "#d8e2f0"
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    for family, data in sorted(families.items()):
        style = family_styles.get(
            family,
            {
                "color": "#999",
                "marker": "o",
                "label": family,
            },
        )

        pairs = sorted(zip(data["sizes"], data["vals"], data["labels"]))

        sizes = [p[0] for p in pairs]
        vals = [p[1] for p in pairs]
        labels = [p[2] for p in pairs]
        positions = [size_to_pos[s] for s in sizes]

        ax.plot(
            positions,
            vals,
            marker=style["marker"],
            color=style["color"],
            label=style["label"],
            linewidth=2,
            markersize=8,
            zorder=3,
        )

        for px, v, lbl in zip(positions, vals, labels):
            ax.annotate(
                lbl,
                (px, v),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
                fontweight="bold",
                color=style["color"],
            )

    ax.set_xlabel("Model Size (B parameters)", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(-0.05, 1.15)

    ax.set_xticks(range(len(all_sizes)))
    ax.set_xticklabels([f"{s}B" for s in all_sizes], fontsize=9)
    ax.set_xlim(-0.5, len(all_sizes) - 0.5)

    ax.legend(fontsize=9, frameon=False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    out = output_dir / out_name
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)

    log.info(f"Saved {out}")
