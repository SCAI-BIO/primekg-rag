import pandas as pd
from pathlib import Path
import textwrap

INPUT_CSV = Path(__file__).parent / "question_top10_matches.csv"
SUMMARY_CSV = Path(__file__).parent / "question_top10_summary.csv"
REPORT_MD = Path(__file__).parent / "pubmed_match_report.md"

TRUNCATE_Q = 140  # chars for question preview in tables


def _short(text: str, n: int = TRUNCATE_Q) -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip().replace("\n", " ")
    return (t[: n - 1] + "…") if len(t) > n else t


def main():
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    if df.empty:
        raise ValueError("Input CSV is empty")

    # Ensure expected columns exist
    required = {
        "question_id",
        "question",
        "rank",
        "weighted_similarity",
        "matched_pmid",
        "matched_title",
        "matched_topic",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input: {sorted(missing)}")

    # Per-question aggregates
    q_groups = df.groupby("question_id", as_index=False)
    per_q = (
        q_groups.apply(
            lambda g: pd.Series(
                {
                    "question": g["question"].iloc[0],
                    "n_matches": len(g),
                    "top1_similarity": g.loc[g["rank"].idxmin(), "weighted_similarity"],
                    "top1_title": g.loc[g["rank"].idxmin(), "matched_title"],
                    "top1_pmid": g.loc[g["rank"].idxmin(), "matched_pmid"],
                    "top1_topic": g.loc[g["rank"].idxmin(), "matched_topic"],
                    "mean_top10": g["weighted_similarity"].mean(),
                    "median_top10": g["weighted_similarity"].median(),
                    "std_top10": g["weighted_similarity"].std(ddof=0),
                }
            )
        )
        .reset_index(drop=True)
    )

    # Overall metrics
    total_questions = per_q.shape[0]
    total_rows = df.shape[0]
    avg_top1 = per_q["top1_similarity"].mean()
    avg_mean_top10 = per_q["mean_top10"].mean()

    # Per-topic aggregates across all top10s
    topic_df = df.copy()
    topic_df["matched_topic"].fillna("", inplace=True)
    per_topic = (
        topic_df.groupby("matched_topic", as_index=False)
        .agg(
            count_in_top10=("matched_pmid", "count"),
            unique_pmids=("matched_pmid", pd.Series.nunique),
            avg_similarity=("weighted_similarity", "mean"),
        )
        .sort_values(["count_in_top10", "avg_similarity"], ascending=[False, False])
    )

    # Save per-question summary CSV
    per_q_out = per_q.copy()
    per_q_out["question_preview"] = per_q_out["question"].map(_short)
    per_q_out = per_q_out[
        [
            "question_id",
            "question_preview",
            "top1_similarity",
            "mean_top10",
            "median_top10",
            "std_top10",
            "top1_title",
            "top1_pmid",
            "top1_topic",
            "n_matches",
        ]
    ]
    per_q_out.to_csv(SUMMARY_CSV, index=False, float_format="%.4f")

    # Build Markdown report
    lines = []
    lines.append("# PubMed Retrieval Report")
    lines.append("")
    lines.append(f"- __Input__: `{INPUT_CSV.name}`")
    lines.append(f"- __Questions__: {total_questions}")
    lines.append(f"- __Rows (questions × topK)__: {total_rows}")
    lines.append(f"- __Avg Top-1 similarity__: {avg_top1:.4f}")
    lines.append(f"- __Avg Mean Top-10 similarity__: {avg_mean_top10:.4f}")
    lines.append("")

    lines.append("## Top-1 Summary per Question")
    lines.append("")
    # Smaller table for quick view
    small = per_q_out.copy()
    lines.append(small.head(50).to_markdown(index=False))
    if per_q_out.shape[0] > 50:
        lines.append("")
        lines.append(f"Showing first 50 of {per_q_out.shape[0]} questions.")

    lines.append("")
    lines.append("## Topic Coverage (across all top-10 matches)")
    lines.append("")
    lines.append(per_topic.head(50).to_markdown(index=False))
    if per_topic.shape[0] > 50:
        lines.append("")
        lines.append(f"Showing first 50 of {per_topic.shape[0]} topics.")

    # Save report
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {SUMMARY_CSV}")
    print(f"Saved: {REPORT_MD}")


if __name__ == "__main__":
    main()
