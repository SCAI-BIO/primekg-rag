# Final Project

This folder is a clean copy of the source files that matter for the final
project. It leaves out local data, generated corpora, ChromaDB folders, logs,
HTML exports, and evaluation outputs.

There are two related pieces here.

## Streamlit Disease Tool

Folder: `streamlit_tool/`

This is the interactive disease and subgraph tool. It lets a user explore
PrimeKG paths, retrieve PubMed evidence, and generate evidence-backed analysis.

Files:

- `app.py` - main Streamlit app.
- `prompt_template.txt` - prompt used for analysis generation.
- `export_utils.py` - exports selected paths and relationships.
- `pmc_parser.py` - downloads PubMed abstracts for fixed disease/topic queries.
- `path_paper_downloader.py` - builds PubMed queries from shortest-path nodes.
- `pmc_embed.py` - embeds PubMed abstracts into ChromaDB.
- `retrieve_node_papers.py` - retrieves papers for a selected graph node.
- `setup_databases.py` - prepares the app databases.

## LLM Benchmarking

Folder: `benchmarking/`

This is the local LLM benchmark for gene-focused TRD/MDD RAG. It validates
structured model output and checks whether local LLMs can answer using retrieved
PubMed evidence.

Files:

- `rag_pipeline.py` - benchmark pipeline.
- `schema.py` - structured output validation.
- `chroma_db.py` - builds the benchmark ChromaDB collection from `pubmed_records_unique.json`.

Corpus scripts:

- `rag/pubmed_parser.py` - creates `pubmed_trd_new.jsonl`.
- `rag/extras.py` - creates `pubmed_trd_extras.jsonl`.
- `rag/increasing_corpus.py` - creates `pubmed_gene_depression.jsonl`.
- `rag/final_corpus.py` - combines the generated corpora and embeds the final corpus.

The benchmark retrieval query is:

```text
Role of {gene} in treatment-resistant major depressive disorder and antidepressant response mechanisms
```

## Evaluation

Folder: `evaluation/`

This is the source code for evaluating benchmark outputs. It is included here
without generated result folders.

Files:

- `evaluation.py` - command entry point for the modular evaluation pipeline.
- `src/evaluation_pipeline/` - loaders, validators, metrics, plots, reports, and LLM-judge helpers.
- `fedility.py` - older/standalone fidelity-checking script kept with the evaluation sources.
- `pmid_heatmaps.py` - creates PMID-level evaluation heatmaps from saved result tables.
- `pmid_top_cited.py` - summarizes the most-cited PMIDs per gene from evaluation outputs.

## Data Note

The corpus files themselves are intentionally not included. They are generated
outputs and should stay local:

- `pubmed_records_unique.json`
- `pubmed_trd_new.jsonl`
- `pubmed_trd_extras.jsonl`
- `pubmed_gene_depression.jsonl`
- `pubmed_final_corpus*.json`
- `chroma_data/`
- `chroma_expanded_corpus*/`
- `evaluation_results*/`
- `kg_triplet_judgment_results/`

Keep `.env`, KG CSVs, database folders, logs, and evaluation result folders out
of git as well.
