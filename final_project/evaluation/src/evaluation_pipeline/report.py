from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    ENABLE_LLM_JUDGE,
    EXPECTED_GENES,
    EXPECTED_MODELS,
    EXPECTED_REPS,
    EXPECTED_ROWS,
    RESULTS_DIR,
    log,
)


STRUCTURE_FIELDS = [
    "answer_present",
    "key_mechanisms_present",
    "limitations_present",
    "clinical_relevance_present",
]


def _fmt_float(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "None\n"

    table = df[columns].copy()
    for col in table.columns:
        if pd.api.types.is_float_dtype(table[col]):
            table[col] = table[col].map(lambda v: _fmt_float(v))

    header = "| " + " | ".join(table.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(table.columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[col]) for col in table.columns) + " |"
        for _, row in table.iterrows()
    ]

    return "\n".join([header, separator, *rows]) + "\n"


def _metric_summary(run_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "pmid_precision",
        "pmid_recall",
        "pmid_f1",
        "structure_score",
        "fields_present",
        "response_time",
    ]
    available = [m for m in metrics if m in run_df.columns]

    summary = (
        run_df.groupby("model")[available]
        .mean()
        .reset_index()
        .sort_values(["pmid_f1", "pmid_precision"], ascending=False)
    )

    summary["rows"] = summary["model"].map(run_df.groupby("model").size())
    ordered_cols = ["model", "rows"] + available
    return summary[ordered_cols]


def _dataset_quality(df: pd.DataFrame) -> dict:
    rows_per_rep = df.groupby("rep").size()
    expected_rows_per_rep = EXPECTED_MODELS * EXPECTED_GENES
    bad_reps = rows_per_rep[rows_per_rep != expected_rows_per_rep]

    actual = df.groupby(["model", "gene"])["rep"].apply(set).reset_index()
    actual["missing_reps"] = actual["rep"].apply(
        lambda reps: sorted(EXPECTED_REPS - reps)
    )
    actual["num_missing"] = actual["missing_reps"].apply(len)

    return {
        "rows": len(df),
        "unique_reps": df["rep"].nunique(),
        "unique_models": df["model"].nunique(),
        "unique_genes": df["gene"].nunique(),
        "bad_reps": bad_reps.reset_index(name="rows"),
        "missing_model_gene_reps": actual[actual["num_missing"] > 0][
            ["model", "gene", "num_missing", "missing_reps"]
        ],
    }


def _artifact_list(output_dir: Path) -> list[str]:
    preferred = [
        "eval_detailed.json",
        "eval_summary.csv",
        "eval_evidence_store.jsonl",
        "eval_gene_level.json",
        "eval_gene_level.csv",
        "pmid_precision_distribution_bins.csv",
        "pmid_recall_distribution_bins.csv",
        "pmid_f1_distribution_bins.csv",
        "pmid_precision_distribution_bins_heatmap.png",
        "pmid_recall_distribution_bins_heatmap.png",
        "pmid_f1_distribution_bins_heatmap.png",
        "structure_sections_by_model.png",
        "pmid_precision_by_model.png",
        "pmid_recall_by_model.png",
        "boxplot_pmid_precision.png",
        "boxplot_pmid_recall.png",
        "boxplot_structure_fields.png",
        "pmid_precision_scaling.png",
        "pmid_recall_scaling.png",
        "pmid_f1_scaling.png",
        "structure_field_breakdown.png",
    ]

    return [name for name in preferred if (output_dir / name).exists()]


def save_evaluation_report(
    df: pd.DataFrame,
    run_df: pd.DataFrame,
    output_dir: Path,
    precision_bins: pd.DataFrame,
    recall_bins: pd.DataFrame,
    f1_bins: pd.DataFrame,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    quality = _dataset_quality(df)
    model_summary = _metric_summary(run_df)
    field_counts = df[STRUCTURE_FIELDS].sum().reset_index()
    field_counts.columns = ["field", "present_count"]
    field_counts["missing_count"] = len(df) - field_counts["present_count"]
    field_counts["present_rate"] = field_counts["present_count"] / len(df)

    missing_fields = df[df[STRUCTURE_FIELDS].sum(axis=1) < len(STRUCTURE_FIELDS)]
    artifact_names = _artifact_list(output_dir)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# PrimeKG RAG Evaluation Report",
        "",
        f"Generated: {now}",
        "",
        "## What this evaluation does",
        "",
        "1. Loads saved RAG result JSON files from the configured results directory.",
        "2. Keeps only expected repetitions from the configured evaluation range.",
        "3. Computes structure checks for the required response sections.",
        "4. Extracts PMID citations from the generated answer text.",
        "5. Compares cited PMIDs against retrieved PMIDs to compute citation grounding metrics.",
        "6. Optionally runs LLM fidelity and validity judges when ENABLE_LLM_JUDGE is enabled and an API key is available.",
        "7. Saves detailed row-level outputs, gene-level outputs, metric distribution CSVs, evidence-store JSONL, plots, and this report.",
        "",
        "## Configuration",
        "",
        f"- Results directory: `{RESULTS_DIR}`",
        f"- Expected repetitions: {len(EXPECTED_REPS)}",
        f"- Expected models: {EXPECTED_MODELS}",
        f"- Expected genes: {EXPECTED_GENES}",
        f"- Expected rows: {EXPECTED_ROWS}",
        f"- LLM judge enabled: {ENABLE_LLM_JUDGE}",
        f"- Chroma collection: `{COLLECTION_NAME}`",
        f"- Chroma directory: `{CHROMA_DIR}`",
        "",
        "## Dataset checks",
        "",
        f"- Evaluated rows: {quality['rows']}",
        f"- Unique repetitions: {quality['unique_reps']}",
        f"- Unique models: {quality['unique_models']}",
        f"- Unique genes: {quality['unique_genes']}",
        f"- Missing or extra rows versus expected: {quality['rows'] - EXPECTED_ROWS}",
        "",
        "### Repetitions with unexpected row counts",
        "",
        _markdown_table(quality["bad_reps"], ["rep", "rows"]),
        "### Missing repetition coverage by model and gene",
        "",
        _markdown_table(
            quality["missing_model_gene_reps"].head(50),
            ["model", "gene", "num_missing", "missing_reps"],
        ),
        "## Metric definitions",
        "",
        "- `structure_score`: 1 when all required sections are present; otherwise 0.",
        "- `fields_present`: number of required sections present, from 0 to 4.",
        "- `num_cited`: number of PMID citations found in the generated output.",
        "- `num_grounded`: cited PMIDs that also appeared in retrieved PMIDs.",
        "- `num_retrieved`: number of retrieved PMIDs attached to the RAG result.",
        "- `pmid_precision`: grounded cited PMIDs divided by cited PMIDs.",
        "- `pmid_recall`: grounded cited PMIDs divided by retrieved PMIDs.",
        "- `pmid_f1`: harmonic mean of PMID precision and recall.",
        "- `pmid_fidelity_score`: LLM-judge faithfulness score over checked PMIDs; 0 when judging is disabled or nothing is checked.",
        "- `validity_score`: LLM-judge validity score; 0 when judging is disabled.",
        "",
        "## Structure completeness",
        "",
        _markdown_table(
            field_counts,
            ["field", "present_count", "missing_count", "present_rate"],
        ),
        f"Rows with at least one missing required field: {len(missing_fields)}",
        "",
        "## Model-level summary",
        "",
        _markdown_table(
            model_summary,
            [
                "model",
                "rows",
                "pmid_precision",
                "pmid_recall",
                "pmid_f1",
                "structure_score",
                "fields_present",
                "response_time",
            ],
        ),
        "## Citation distribution files",
        "",
        f"- Precision distribution rows: {len(precision_bins)}",
        f"- Recall distribution rows: {len(recall_bins)}",
        f"- F1 distribution rows: {len(f1_bins)}",
        "",
        "## Generated artifacts",
        "",
    ]

    if artifact_names:
        lines.extend(f"- `{name}`" for name in artifact_names)
    else:
        lines.append("No known artifacts were found yet.")

    lines.extend([
        "",
        "## How to read the results",
        "",
        "Start with `eval_summary.csv` for row-level metrics, then use this report to check whether the dataset is complete. The precision, recall, and F1 plots show whether each model cites evidence that came from retrieval. The structure plots show whether each model follows the expected answer format. If LLM judging is enabled, inspect `eval_evidence_store.jsonl` for the evidence used by the fidelity and validity judges.",
        "",
    ])

    report_path = output_dir / "evaluation_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    log.info(f"Saved {report_path}")
    return report_path
