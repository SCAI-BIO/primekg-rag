from .chroma_access import connect_chroma
from .config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    ENABLE_LLM_JUDGE,
    OUTPUT_DIR,
    RESULTS_DIR,
    log,
)
from .evaluator import evaluate
from .loaders import load_all_results
from .outputs import (
    add_derived_columns,
    count_metric_distribution_bins,
    save_evidence_store,
    save_gene_level_outputs,
    save_outputs,
)
from .plots import (
    plot_boxplot_precision,
    plot_boxplot_recall,
    plot_boxplot_structure,
    plot_distribution_bin_heatmap,
    plot_metric_by_model,
    plot_scaling_curve,
    plot_structure_breakdown,
    plot_structure_by_model,
)
from .report import save_evaluation_report
from .validation import print_dataset_checks


def main():
    if not RESULTS_DIR.exists():
        log.error(f"Missing results dir: {RESULTS_DIR}")
        return

    collection = None

    if ENABLE_LLM_JUDGE:
        collection = connect_chroma(CHROMA_DIR, COLLECTION_NAME)

    results = load_all_results(RESULTS_DIR)

    if not results:
        log.error("No results found.")
        return

    models_found = set(r.get("model", "") for r in results)

    log.info(
        f"Loaded {len(results)} results across "
        f"{len(models_found)} models: {sorted(models_found)}"
    )

    evals, evidence_store = evaluate(results, collection=collection)

    df = save_outputs(evals, OUTPUT_DIR)

    fields = [
        "answer_present",
        "key_mechanisms_present",
        "limitations_present",
        "clinical_relevance_present",
    ]

    print("\nTOTAL EVALUATED ROWS:", len(df))

    print("\nFIELD COUNTS:")
    print(df[fields].sum())

    print("\nMISSING FIELD COUNTS:")
    print((1 - df[fields]).sum())

    print("\nROWS WITH MISSING FIELDS:")
    print(
        df[df[fields].sum(axis=1) < 4][
            ["model", "gene", "rep"] + fields
        ]
    )

    print_dataset_checks(df)

    save_evidence_store(evidence_store, OUTPUT_DIR)

    run_df = add_derived_columns(df)
    save_gene_level_outputs(run_df, OUTPUT_DIR)

    precision_bins = count_metric_distribution_bins(run_df, "pmid_precision", OUTPUT_DIR)
    recall_bins = count_metric_distribution_bins(run_df, "pmid_recall", OUTPUT_DIR)
    f1_bins = count_metric_distribution_bins(run_df, "pmid_f1", OUTPUT_DIR)

    plot_distribution_bin_heatmap(
        precision_bins,
        "pmid_precision",
        "PMID Precision Distribution by Model",
        "pmid_precision_distribution_bins_heatmap.png",
        OUTPUT_DIR,
    )

    plot_distribution_bin_heatmap(
        recall_bins,
        "pmid_recall",
        "PMID Recall Distribution by Model",
        "pmid_recall_distribution_bins_heatmap.png",
        OUTPUT_DIR,
    )

    plot_distribution_bin_heatmap(
        f1_bins,
        "pmid_f1",
        "PMID F1 Distribution by Model",
        "pmid_f1_distribution_bins_heatmap.png",
        OUTPUT_DIR,
    )

    plot_structure_by_model(df, OUTPUT_DIR)

    plot_metric_by_model(
        df,
        "pmid_precision",
        "PMID Precision by Model",
        "pmid_precision_by_model.png",
        OUTPUT_DIR,
    )

    plot_metric_by_model(
        df,
        "pmid_recall",
        "PMID Recall by Model",
        "pmid_recall_by_model.png",
        OUTPUT_DIR,
    )

    plot_boxplot_precision(run_df, OUTPUT_DIR)
    plot_boxplot_recall(run_df, OUTPUT_DIR)
    plot_boxplot_structure(run_df, OUTPUT_DIR)

    plot_scaling_curve(
        run_df,
        "pmid_precision",
        "PMID Precision Scaling by Model Family",
        "Mean PMID Precision",
        "pmid_precision_scaling.png",
        OUTPUT_DIR,
    )

    plot_scaling_curve(
        run_df,
        "pmid_recall",
        "PMID Recall Scaling by Model Family",
        "Mean PMID Recall",
        "pmid_recall_scaling.png",
        OUTPUT_DIR,
    )

    plot_scaling_curve(
        run_df,
        "pmid_f1",
        "PMID F1 Scaling by Model Family",
        "Mean PMID F1",
        "pmid_f1_scaling.png",
        OUTPUT_DIR,
    )

    plot_structure_breakdown(df, OUTPUT_DIR)

    save_evaluation_report(
        df,
        run_df,
        OUTPUT_DIR,
        precision_bins,
        recall_bins,
        f1_bins,
    )

    log.info("Done.")
