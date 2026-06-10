import json
from pathlib import Path

import pandas as pd

from .config import log


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "answer_present",
        "key_mechanisms_present",
        "limitations_present",
        "clinical_relevance_present",
    ]

    tmp = df.copy()
    tmp["fields_present"] = tmp[cols].sum(axis=1)

    p = tmp["pmid_precision"]
    r = tmp["pmid_recall"]

    tmp["pmid_f1"] = (2 * p * r / (p + r)).fillna(0.0)

    return tmp


def save_outputs(evals, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(evals)

    json_path = output_dir / "eval_detailed.json"
    csv_path = output_dir / "eval_summary.csv"

    df.to_json(json_path, orient="records", indent=2)
    df.to_csv(csv_path, index=False)

    log.info(f"Saved {json_path}")
    log.info(f"Saved {csv_path}")

    return df


def save_gene_level_outputs(gene_df: pd.DataFrame, output_dir: Path):
    run_json_path = output_dir / "eval_gene_level.json"
    run_csv_path = output_dir / "eval_gene_level.csv"

    gene_df.to_json(run_json_path, orient="records", indent=2)
    gene_df.to_csv(run_csv_path, index=False)

    log.info(f"Saved {run_json_path}")
    log.info(f"Saved {run_csv_path}")


def save_evidence_store(evidence_store: list, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / "eval_evidence_store.jsonl"

    with open(path, "w", encoding="utf-8") as f:
        for entry in evidence_store:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    log.info(f"Saved evidence store ({len(evidence_store)} entries) -> {path}")


def count_metric_distribution_bins(
    df: pd.DataFrame,
    metric: str,
    output_dir: Path,
    bin_width: float = 0.1,
) -> pd.DataFrame:
    """
    Count observations per model in fixed bins from 0.0 to 1.0.

    Designed for bounded metrics such as pmid_precision, pmid_recall, and pmid_f1.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 100)
    print(f"DISTRIBUTION COUNTS FOR: {metric}")
    print("=" * 100)

    if metric not in df.columns:
        raise ValueError(f"Metric column not found: {metric}")

    if bin_width <= 0 or bin_width > 1:
        raise ValueError("bin_width must be greater than 0 and less than or equal to 1")

    n_bins = int(round(1 / bin_width))
    bins = [round(i * bin_width, 3) for i in range(n_bins + 1)]

    if bins[-1] != 1.0:
        bins.append(1.0)

    labels = []
    for i in range(len(bins) - 1):
        left = bins[i]
        right = bins[i + 1]
        label = f"{left:.1f}-1.0" if right == 1.0 else f"{left:.1f}-<{right:.1f}"
        labels.append(label)

    rows = []

    for model in sorted(df["model"].unique()):
        mdf = df[df["model"] == model].copy()
        n_total = len(mdf)
        values = pd.to_numeric(mdf[metric], errors="coerce").clip(lower=0.0, upper=1.0)

        binned = pd.cut(
            values,
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=False,
        ).astype("object")

        binned.loc[values == 1.0] = labels[-1]
        counts = binned.value_counts().reindex(labels, fill_value=0)

        row = {
            "model": model,
            "n_total": n_total,
        }

        for label in labels:
            count = int(counts[label])
            percent = round((count / n_total) * 100, 2) if n_total else 0.0

            row[f"count_{label}"] = count
            row[f"percent_{label}"] = percent

        rows.append(row)

    out_df = pd.DataFrame(rows)
    print(out_df.to_string(index=False))

    out_csv = output_dir / f"{metric}_distribution_bins.csv"
    out_df.to_csv(out_csv, index=False)

    log.info(f"Saved {out_csv}")

    return out_df
