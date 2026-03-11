import json
import logging
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# ============================================================================
# Logging
# ============================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# ============================================================================
# Config
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = Path("C:/Users/aemekkawi/Desktop/h")
OUTPUT_DIR = BASE_DIR / "evaluation_results"
JUDGE_MODEL = "gpt-4o-mini"

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ============================================================================
# Helpers
# ============================================================================

def extract_pmids_from_text(text: str):
    """
    Extract PMID citations from inline text.

    Handles:
      [PMID:17884271]
      [PMID:17884271, PMID:24511450]
      PMID:17884271
      PMID 17884271

    Returns deduplicated list like:
      ["17884271", "24511450"]
    """
    if not text:
        return []

    seen, out = set(), []

    # 1) bracket-aware extraction
    bracket_chunks = re.findall(r"\[([^\]]+)\]", text)
    for chunk in bracket_chunks:
        hits = re.findall(r"PMID[:\s]*(\d+)", chunk, flags=re.IGNORECASE)
        for h in hits:
            if h not in seen:
                seen.add(h)
                out.append(h)

    # 2) fallback: catch any PMID outside brackets too
    hits = re.findall(r"PMID[:\s]*(\d+)", text, flags=re.IGNORECASE)
    for h in hits:
        if h not in seen:
            seen.add(h)
            out.append(h)

    return out


def extract_triples_from_text(text: str):
    """
    Extract triple IDs from inline citations.

    Handles:
      [T1]
      [T2, T3, T4]
      [PMID:17884271][T5]
      mixed bracket contents

    Returns deduplicated list like:
      ["T1", "T2", "T3"]
    """
    if not text:
        return []

    bracket_chunks = re.findall(r"\[([^\]]+)\]", text)
    seen, out = set(), []

    for chunk in bracket_chunks:
        hits = re.findall(r"T\d+", chunk, flags=re.IGNORECASE)
        for h in hits:
            h = h.upper()
            if h not in seen:
                seen.add(h)
                out.append(h)

    return out


def dedupe_str_list(xs):
    seen, out = set(), []
    for x in xs or []:
        s = str(x).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def get_combined_output_text(parsed: dict) -> str:
    """
    Combine all answer-bearing text fields for citation extraction.
    """
    if not isinstance(parsed, dict):
        return ""

    parts = [
        parsed.get("answer", ""),
        parsed.get("clinical_relevance", ""),
    ]

    kms = parsed.get("key_mechanisms", [])
    if isinstance(kms, list):
        for km in kms:
            if isinstance(km, dict):
                parts.append(km.get("description", ""))

    return " ".join(p for p in parts if p)


def load_all_results(results_dir: Path):
    results = []
    for jf in sorted(results_dir.glob("rep_*/results_*.json")):
        try:
            data = json.loads(Path(jf).read_text(encoding="utf-8"))
        except Exception as e:
            log.warning(f"Skipping {jf}: {e}")
            continue

        rep = int(jf.parent.name.replace("rep_", ""))
        for r in data:
            r["rep"] = rep
        results.extend(data)

        log.info(f"Loaded {len(data)} from {jf}")

    log.info(f"Total results: {len(results)}")
    return results


# ============================================================================
# STRUCTURE — 4 independent flags
# ============================================================================

def eval_structure_sections(result: dict):
    p = result.get("parsed")
    if not isinstance(p, dict):
        return {
            "answer_present": 0,
            "key_mechanisms_present": 0,
            "limitations_present": 0,
            "clinical_relevance_present": 0,
        }
    return {
        "answer_present": 1 if p.get("answer") else 0,
        "key_mechanisms_present": 1 if p.get("key_mechanisms") else 0,
        "limitations_present": 1 if p.get("limitations") else 0,
        "clinical_relevance_present": 1 if p.get("clinical_relevance") else 0,
    }


# ============================================================================
# PMID CITATIONS — precision + recall (grounded wrt retrieved_pmids)
# ============================================================================

def eval_citations(result: dict):
    p = result.get("parsed")
    if not isinstance(p, dict):
        return 0, 0, 0, 0.0, 0.0

    combined_text = get_combined_output_text(p)
    cited = extract_pmids_from_text(combined_text)

    retrieved = dedupe_str_list(result.get("retrieved_pmids", []))

    cited_n = len(cited)
    retrieved_n = len(retrieved)

    if cited_n == 0:
        return 0, 0, retrieved_n, 0.0, 0.0

    retrieved_set = set(retrieved)
    grounded_n = sum(1 for pmid in cited if pmid in retrieved_set)

    precision = grounded_n / cited_n if cited_n else 0.0
    recall = grounded_n / retrieved_n if retrieved_n else 0.0

    return cited_n, grounded_n, retrieved_n, round(precision, 3), round(recall, 3)
# ============================================================================
# KG TRIPLE CITATIONS — precision + recall (grounded wrt kg_index)
# ============================================================================

def eval_triplets(result: dict):
    """
    KG citation metrics:
      - cited triples = T# found in generated text
      - available triples = T# available in kg_index for that run
      - grounded triples = cited triples that exist in kg_index
    """
    p = result.get("parsed")
    if not isinstance(p, dict):
        return 0, 0, 0, 0.0, 0.0

    combined_text = get_combined_output_text(p)
    cited = extract_triples_from_text(combined_text)

    kg_index = result.get("kg_index", {}) or {}
    available = [f"T{k}" for k in kg_index.keys()]
    available = dedupe_str_list(available)

    cited_n = len(cited)
    available_n = len(available)

    if cited_n == 0:
        return 0, 0, available_n, 0.0, 0.0

    available_set = set(available)
    grounded_n = sum(1 for t in cited if t in available_set)

    precision = grounded_n / cited_n if cited_n else 0.0
    recall = grounded_n / available_n if available_n else 0.0

    return cited_n, grounded_n, available_n, round(precision, 3), round(recall, 3)


# ============================================================================
# Judges (optional; require OPENAI_API_KEY)
# ============================================================================

def _judge(system_prompt: str, parsed: dict):
    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(parsed)},
            ],
            temperature=0,
        )
        out = json.loads(resp.choices[0].message.content)
        return int(out.get("score", 0)), str(out.get("reason", "")).strip()
    except Exception:
        return 0, "judge_error"


def judge_grounding(parsed: dict):
    return _judge(
        "Evaluate whether cited literature and KG evidence support the claims. Return JSON: {score:0|1, reason:'<=20 words'}",
        parsed,
    )


def judge_validity(parsed: dict):
    return _judge(
        "Are the claims scientifically valid? Return JSON: {score:0|1, reason:'<=20 words'}",
        parsed,
    )


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(results):
    evals = []
    for r in results:
        struct = eval_structure_sections(r)

        cited_n, grounded_n, retrieved_n, pmid_precision, pmid_recall = eval_citations(r)
        trip_cited_n, trip_grounded_n, trip_available_n, kg_precision, kg_recall = eval_triplets(r)

        parsed = r.get("parsed")
        if isinstance(parsed, dict) and os.getenv("OPENAI_API_KEY"):
            g_score, g_reason = judge_grounding(parsed)
            v_score, v_reason = judge_validity(parsed)
        else:
            g_score, g_reason = 0, ""
            v_score, v_reason = 0, ""

        struct_score = 1 if all(v == 1 for v in struct.values()) else 0

        evals.append({
            "rep": r.get("rep", 0),
            "gene": r.get("gene", ""),
            "model": r.get("model", ""),
            "status": r.get("status", ""),
            "response_time": round(float(r.get("response_time", 0) or 0), 1),

            # structure flags
            **struct,
            "structure_score": struct_score,

            # PMID metrics
            "num_cited": cited_n,
            "num_grounded": grounded_n,
            "num_retrieved": retrieved_n,
            "pmid_precision": pmid_precision,
            "pmid_recall": pmid_recall,

            # KG metrics
            "num_triplets_cited": trip_cited_n,
            "num_triplets_grounded": trip_grounded_n,
            "num_triplets_available": trip_available_n,
            "kg_precision": kg_precision,
            "kg_recall": kg_recall,

            # judges
            "grounding_score": g_score,
            "grounding_reason": g_reason,
            "validity_score": v_score,
            "validity_reason": v_reason,
        })

    return evals


# ============================================================================
# Save + Plots
# ============================================================================

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


def _paired_bar(df: pd.DataFrame, metric: str, title: str, ylabel: str, out_name: str, output_dir: Path, ylim=(0, 1.1)):
    """
    Grouped bar chart: DeepSeek vs Qwen at each size tier.
    """
    means = df.groupby("model")[metric].mean()

    pairs = [
        ("~1.5b", "deepseek-r1:1.5b", "qwen3:1.7b"),
        ("8b", "deepseek-r1:8b", "qwen3:8b"),
        ("14b", "deepseek-r1:14b", "qwen3:14b"),
        ("~32b", "deepseek-r1:32b", "qwen3:30b"),
    ]

    present = means.index.tolist()
    pairs = [(lbl, ds, qw) for lbl, ds, qw in pairs if ds in present or qw in present]

    labels = [p[0] for p in pairs]
    ds_vals = [means.get(p[1], 0) for p in pairs]
    qw_vals = [means.get(p[2], 0) for p in pairs]

    x = list(range(len(labels)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_ds = ax.bar([i - width / 2 for i in x], ds_vals, width, label="DeepSeek-R1", color="#4c78a8")
    bars_qw = ax.bar([i + width / 2 for i in x], qw_vals, width, label="Qwen3", color="#e45756")

    for bars in [bars_ds, bars_qw]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.02,
                    f"{h:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Model Size")
    ax.legend()
    fig.tight_layout()

    out = output_dir / out_name
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info(f"Saved {out}")


def plot_structure_by_model(df: pd.DataFrame, output_dir: Path):
    """
    Grouped bar: structure section presence — DeepSeek vs Qwen per size.
    """
    cols = ["answer_present", "key_mechanisms_present", "limitations_present", "clinical_relevance_present"]
    labels_short = ["answer", "mechanisms", "limitations", "clinical_rel"]

    pairs = [
        ("~1.5b", "deepseek-r1:1.5b", "qwen3:1.7b"),
        ("8b", "deepseek-r1:8b", "qwen3:8b"),
        ("14b", "deepseek-r1:14b", "qwen3:14b"),
        ("~32b", "deepseek-r1:32b", "qwen3:30b"),
    ]

    g = df.groupby("model")[cols].mean()
    present = g.index.tolist()
    pairs = [(lbl, ds, qw) for lbl, ds, qw in pairs if ds in present or qw in present]

    n_pairs = len(pairs)
    fig, axes = plt.subplots(1, n_pairs, figsize=(4 * n_pairs, 5), sharey=True, squeeze=False)

    for idx, (size_lbl, ds, qw) in enumerate(pairs):
        ax = axes[0][idx]
        x = list(range(len(cols)))
        width = 0.35

        ds_vals = [g.loc[ds, c] if ds in present else 0 for c in cols]
        qw_vals = [g.loc[qw, c] if qw in present else 0 for c in cols]

        ax.bar([i - width / 2 for i in x], ds_vals, width, label="DeepSeek-R1", color="#4c78a8")
        ax.bar([i + width / 2 for i in x], qw_vals, width, label="Qwen3", color="#e45756")

        ax.set_title(size_lbl, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels_short, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1.15)
        if idx == 0:
            ax.set_ylabel("Pass Rate")
            ax.legend(fontsize=8)

    fig.suptitle("Structure Section Presence: DeepSeek-R1 vs Qwen3 by Size", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out = output_dir / "structure_sections_by_model.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info(f"Saved {out}")


def plot_metric_by_model(df: pd.DataFrame, metric: str, title: str, out_name: str, output_dir: Path):
    """
    Grouped bar: metric comparison — DeepSeek vs Qwen at each size tier.
    """
    _paired_bar(df, metric, title, metric, out_name, output_dir)


SIZE_PAIRS = [
    ("~1.5b", "deepseek-r1:1.5b", "qwen3:1.7b"),
    ("8b", "deepseek-r1:8b", "qwen3:8b"),
    ("14b", "deepseek-r1:14b", "qwen3:14b"),
    ("~32b", "deepseek-r1:32b", "qwen3:30b"),
]


def plot_boxplot_precision(df: pd.DataFrame, output_dir: Path):
    """
    Boxplot: PMID precision across repetitions, paired by size tier.
    """
    present = df["model"].unique().tolist()
    pairs = [(lbl, ds, qw) for lbl, ds, qw in SIZE_PAIRS if ds in present or qw in present]

    positions, data, tick_labels, colors = [], [], [], []
    pos = 1
    for lbl, ds, qw in pairs:
        if ds in present:
            data.append(df[df["model"] == ds]["pmid_precision"].values)
            positions.append(pos)
            tick_labels.append(f"DS {lbl}")
            colors.append("#4c78a8")
            pos += 1
        if qw in present:
            data.append(df[df["model"] == qw]["pmid_precision"].values)
            positions.append(pos)
            tick_labels.append(f"QW {lbl}")
            colors.append("#e45756")
            pos += 1
        pos += 0.5

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, positions=positions, patch_artist=True, widths=0.6)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right")
    ax.set_title("PMID Precision Variance Across Repetitions")
    ax.set_ylabel("Precision")
    ax.set_ylim(-0.05, 1.15)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#4c78a8", label="DeepSeek-R1"),
                       Patch(color="#e45756", label="Qwen3")])

    fig.tight_layout()
    out = output_dir / "boxplot_pmid_precision.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info(f"Saved {out}")


def plot_boxplot_kg_precision(df: pd.DataFrame, output_dir: Path):
    """
    Boxplot: KG triplet precision across repetitions, paired by size tier.
    """
    present = df["model"].unique().tolist()
    pairs = [(lbl, ds, qw) for lbl, ds, qw in SIZE_PAIRS if ds in present or qw in present]

    positions, data, tick_labels, colors = [], [], [], []
    pos = 1
    for lbl, ds, qw in pairs:
        if ds in present:
            data.append(df[df["model"] == ds]["kg_precision"].values)
            positions.append(pos)
            tick_labels.append(f"DS {lbl}")
            colors.append("#4c78a8")
            pos += 1
        if qw in present:
            data.append(df[df["model"] == qw]["kg_precision"].values)
            positions.append(pos)
            tick_labels.append(f"QW {lbl}")
            colors.append("#e45756")
            pos += 1
        pos += 0.5

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, positions=positions, patch_artist=True, widths=0.6)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right")
    ax.set_title("KG Triplet Precision Variance Across Repetitions")
    ax.set_ylabel("Precision")
    ax.set_ylim(-0.05, 1.15)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#4c78a8", label="DeepSeek-R1"),
                       Patch(color="#e45756", label="Qwen3")])

    fig.tight_layout()
    out = output_dir / "boxplot_kg_precision.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info(f"Saved {out}")


def plot_boxplot_structure(df: pd.DataFrame, output_dir: Path):
    """
    Boxplot: fields present per run (0–4) across repetitions, paired by size tier.
    """
    cols = ["answer_present", "key_mechanisms_present", "limitations_present", "clinical_relevance_present"]
    df = df.copy()
    df["fields_present"] = df[cols].sum(axis=1)

    present = df["model"].unique().tolist()
    pairs = [(lbl, ds, qw) for lbl, ds, qw in SIZE_PAIRS if ds in present or qw in present]

    positions, data, tick_labels, colors = [], [], [], []
    pos = 1
    for lbl, ds, qw in pairs:
        if ds in present:
            data.append(df[df["model"] == ds]["fields_present"].values)
            positions.append(pos)
            tick_labels.append(f"DS {lbl}")
            colors.append("#4c78a8")
            pos += 1
        if qw in present:
            data.append(df[df["model"] == qw]["fields_present"].values)
            positions.append(pos)
            tick_labels.append(f"QW {lbl}")
            colors.append("#e45756")
            pos += 1
        pos += 0.5

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, positions=positions, patch_artist=True, widths=0.6)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    for p, d in zip(positions, data):
        total = int(sum(d))
        max_possible = len(d) * 4
        ax.text(p, 4.15, f"{total}/{max_possible}", ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right")
    ax.set_title("Structure Adherence: Fields Present per Run (max 4)")
    ax.set_ylabel("Fields Present (0–4)")
    ax.set_ylim(-0.2, 4.6)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#4c78a8", label="DeepSeek-R1"),
                       Patch(color="#e45756", label="Qwen3")])

    fig.tight_layout()
    out = output_dir / "boxplot_structure_fields.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info(f"Saved {out}")


def plot_structure_breakdown(df: pd.DataFrame, output_dir: Path):
    """
    Grid: one subplot per model, 4 bars showing pass rate of each field.
    """
    cols = ["answer_present", "key_mechanisms_present", "limitations_present", "clinical_relevance_present"]
    labels = ["answer", "mechanisms", "limitations", "clinical_rel"]
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]

    models = sorted(df["model"].unique())
    n = len(models)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)

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


# ============================================================================
# Main
# ============================================================================

def main():
    if not RESULTS_DIR.exists():
        log.error(f"Missing results dir: {RESULTS_DIR}")
        return

    if not os.getenv("OPENAI_API_KEY"):
        log.warning("OPENAI_API_KEY not set — grounding/validity judges will be skipped.")

    results = load_all_results(RESULTS_DIR)
    if not results:
        log.error("No results found.")
        return

    evals = evaluate(results)
    df = save_outputs(evals, OUTPUT_DIR)

    plot_structure_by_model(df, OUTPUT_DIR)
    plot_metric_by_model(df, "pmid_precision", "PMID Precision by Model", "pmid_precision_by_model.png", OUTPUT_DIR)
    plot_metric_by_model(df, "pmid_recall", "PMID Recall by Model", "pmid_recall_by_model.png", OUTPUT_DIR)
    plot_metric_by_model(df, "kg_precision", "KG Triplet Precision by Model", "kg_precision_by_model.png", OUTPUT_DIR)
    plot_metric_by_model(df, "kg_recall", "KG Triplet Recall by Model", "kg_recall_by_model.png", OUTPUT_DIR)

    plot_boxplot_precision(df, OUTPUT_DIR)
    plot_boxplot_kg_precision(df, OUTPUT_DIR)
    plot_boxplot_structure(df, OUTPUT_DIR)
    plot_structure_breakdown(df, OUTPUT_DIR)

    log.info("Done.")


if __name__ == "__main__":
    main()