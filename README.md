# PrimeKG RAG

This repo currently holds two related projects that share some data and naming,
but they are doing different jobs.

## 1. Streamlit Disease Tool

This is the interactive app for exploring disease subgraphs, shortest paths, and
related PubMed evidence.

The main app lives in `primekg-rag/app.py`. It uses PrimeKG-style graph files,
local ChromaDB collections, and a prompt template to help users inspect medical
relationships and generate evidence-backed summaries.

Useful files:

- `primekg-rag/app.py` - the Streamlit interface.
- `primekg-rag/prompt_template.txt` - the main analysis prompt.
- `primekg-rag/export_utils.py` - exports selected paths and relationship data.
- `primekg-rag/pmc_parser.py` - downloads PubMed abstracts for fixed disease or topic queries.
- `primekg-rag/path_paper_downloader.py` - builds PubMed queries from shortest-path nodes and downloads matching papers.
- `primekg-rag/pmc_embed.py` - embeds PubMed abstracts into ChromaDB.
- `primekg-rag/retrieve_node_papers.py` - retrieves papers for a selected node from the local PubMed collection.
- `primekg-rag/setup_databases.py` - downloads or prepares the local databases used by the app.
- `primekg-rag/tool.md` - human working notes for the tool. This may later become separate docs, for example `tool.md` and `benchmarking.md`, or one combined README.

## 2. LLM Benchmarking

This is the local LLM benchmark for gene-focused treatment-resistant depression
and major depressive disorder RAG. The benchmark starts from a PubMed corpus,
embeds it, retrieves evidence for each gene, and validates structured LLM
outputs.

Useful files:

- `rag_pipeline.py` - runs the benchmark pipeline and asks the model about each gene.
- `schema.py` - validates the structured output expected from the LLM.
- `chroma_db.py` - builds the benchmark ChromaDB collection from `pubmed_records_unique.json`.
- `rag/pubmed_parser.py` - downloads the broad TRD/MDD PubMed corpus into `pubmed_trd_new.jsonl`.
- `rag/extras.py` - downloads an extra focused TRD/MDD corpus for the benchmark genes into `pubmed_trd_extras.jsonl`.
- `rag/increasing_corpus.py` - downloads the current gene-focused depression corpus into `pubmed_gene_depression.jsonl`.
- `rag/final_corpus.py` - combines the generated JSONL corpora, deduplicates by PMID, writes a final corpus JSON, and embeds it into ChromaDB.

The retrieval query used by the current benchmark pipeline is:

```text
Role of {gene} in treatment-resistant major depressive disorder and antidepressant response mechanisms
```

## Corpus Flow

There are two corpus flows in the repo.

The current benchmark flow used by `chroma_db.py` is:

```text
pubmed_records_unique.json
-> chroma_db.py
-> chroma_data/pubmed_trd_mdd
-> rag_pipeline.py
```

The source scripts that create the expanded PubMed corpora are:

```text
rag/pubmed_parser.py
-> pubmed_trd_new.jsonl

rag/extras.py
-> pubmed_trd_extras.jsonl

rag/increasing_corpus.py
-> pubmed_gene_depression.jsonl

rag/final_corpus.py
-> pubmed_final_corpus_v2.json
-> chroma_expanded_corpus_v2/
```

`rag/final_corpus.py` also looks for `pubmed_gene_aliases.jsonl` if it exists.
That file is treated as optional; no matching downloader script is currently in
this repo.

## What Not To Push

Please keep generated data and local databases out of git. In particular, do not
push `.env`, raw KG CSVs, PubMed JSON/JSONL outputs, ChromaDB folders, evaluation
result folders, logs, or generated HTML/CSV summaries. The `.gitignore` includes
the common local outputs for both projects.

## Running Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit tool:

```bash
cd primekg-rag
streamlit run app.py
```

Build the benchmark ChromaDB from the existing corpus file:

```bash
python chroma_db.py
```

Run the benchmark:

```bash
python rag_pipeline.py
```

The code is research tooling, not a clinical decision system. Treat outputs as
evidence to inspect, not as medical advice.
