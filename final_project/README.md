# PrimeKG RAG

This project combines PrimeKG pathways and PubMed evidence to generate
gene-focused explanations for treatment-resistant major depressive
disorder (TR-MDD).

It includes a Streamlit Knowledge Graph Interpretation Tool, a local LLM benchmark,
and evaluation scripts.

The tool uses PrimeKG and PubMed evidence to help interpret relationships
between biomedical entities. Users can explore disease subgraphs, inspect
connecting paths, retrieve related papers, and generate explanations
grounded in the selected evidence.

## Project structure

| Folder | Contents |
| --- | --- |
| `streamlit_tool/` | Disease subgraph exploration, paper retrieval, and analysis |
| `benchmarking/` | PubMed corpus preparation, database creation, and LLM benchmarking |
| `evaluation/` | Structure and citation metrics, AI faithfulness checks, and plots |

## Installation

From `final_project/`, create a Python environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install pymed "pydantic>=2"
```

The benchmark uses local models through Ollama. Ollama must be running,
and the models listed in `benchmarking/rag_pipeline.py` must be downloaded.

The Streamlit analysis and the separate AI evaluation use an OpenAI API
key. It can be supplied through the environment or a local `.env` file:

```text
OPENAI_API_KEY=your_api_key
```

## Knowledge graph

### Downloading PrimeKG

The original PrimeKG dataset is available from
[Harvard Dataverse](https://doi.org/10.7910/DVN/IXA7BM).

Download `kg.csv` from the dataset's Files section.
Further documentation is available on the
[PrimeKG project page](https://zitniklab.hms.harvard.edu/projects/PrimeKG/).

### Extracting gene pathways

PrimeKG was loaded into Neo4j, and Cypher queries were used to find
two-hop connections between the 16 SHAP-selected genes and major
depressive disorder.

The intermediate entities included genes/proteins, drugs, diseases,
and effects/phenotypes. Relationships were searched in either direction
to identify connecting paths.

The results were exported as `pathways.csv` with these columns:

gene, intermediate_node, intermediate_type, relationship_1, relationship_2

The benchmark removes duplicate rows and assigns each pathway a reference
such as `[T1]` or `[T2]` for use in the generated explanations.

Place the exported file in `benchmarking/` before running
`rag_pipeline.py`. The original CSV-export query is not included here.

Duplicate paths are removed when the benchmark loads the file.
Each path is assigned a reference such as `[T1]` or `[T2]`, allowing
the generated explanation to cite the graph evidence used.

Place `pathways.csv` in `benchmarking/`. The benchmark reads this export;
it does not query Neo4j during model generation. The original extraction
queries are not currently included in this folder.

## PubMed corpus and database creation

The retrieval databases can be built from downloaded PubMed abstracts.
The Streamlit tool and the benchmark use separate collections.

### Benchmark corpus

Run the following commands from `final_project/benchmarking/`:

```bash
python rag/pubmed_parser.py
python rag/extras.py
python rag/increasing_corpus.py
python rag/final_corpus.py
```

| Script | Purpose |
| --- | --- |
| `rag/pubmed_parser.py` | Downloads a broad TRD/MDD corpus |
| `rag/extras.py` | Downloads additional gene-focused TRD/MDD papers |
| `rag/increasing_corpus.py` | Downloads depression-related papers for the selected genes |
| `rag/final_corpus.py` | Combines the corpora, removes duplicate PMIDs, and builds the expanded database |

The gene-focused scripts require `features.json` in the working
directory. This file is a JSON object whose gene entries contain
`"type": "gene"`.

The downloaded corpus files are:

```text
pubmed_trd_new.jsonl
pubmed_trd_extras.jsonl
pubmed_gene_depression.jsonl
```

`final_corpus.py` also includes `pubmed_gene_aliases.jsonl` if available.
Missing input corpora are skipped with a warning.

It produces:

```text
pubmed_final_corpus_v2.json
chroma_expanded_corpus_v2/
```

The expanded collection is named `pubmed_depression_expanded` and uses
`all-MiniLM-L6-v2` embeddings. Running this script rebuilds that collection.

To use it in `benchmarking/rag_pipeline.py`, set:

```python
CHROMA_DIR = BASE_DIR / "chroma_expanded_corpus_v2"
COLLECTION_NAME = "pubmed_depression_expanded"
```

### Alternative benchmark ingestion

`benchmarking/chroma_db.py` builds a database from the existing
`pubmed_records_unique.json` corpus:

```bash
python chroma_db.py
```

It writes to `chroma_data/pubmed_trd_mdd/`, using the collection name
`pubmed_trd_mdd`.

This is a separate ingestion route. The database path and collection
name in `rag_pipeline.py` must match the database being used.

### Streamlit PubMed database

In `streamlit_tool/`:

- `pmc_parser.py` downloads abstracts for disease and topic queries.
- `pmc_embed.py` embeds the downloaded JSON into the `pubmed_abstracts`
  collection.
- `path_paper_downloader.py` downloads and embeds papers related to
  selected graph paths.

`pmc_embed.py` looks for `pubmed_abstracts_full_*.json` in
`final_project/` and creates `final_project/pubmed_db/`.

The app defaults to `streamlit_tool/pubmed_db/`. When using the database
created by `pmc_embed.py`, set `PUBMED_DB_PATH` to its actual location.

## Streamlit disease tool

The app supports:

- Exploring disease subgraphs and their relationships.
- Inspecting paths between biomedical entities.
- Retrieving related PubMed evidence.
- Generating evidence-backed analysis.
- Exporting selected paths and relationships.

Main files:

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit interface |
| `prompt_template.txt` | Analysis prompt |
| `export_utils.py` | Path and relationship exports |
| `pmc_parser.py` | PubMed abstract download |
| `pmc_embed.py` | PubMed database creation |
| `path_paper_downloader.py` | Path-based paper retrieval and embedding |
| `retrieve_node_papers.py` | Paper retrieval for selected nodes |
| `setup_databases.py` | Download of prepared app databases |

### Using prepared databases

Prepared app databases are available through the Zenodo archive referenced
by `setup_databases.py`:

https://zenodo.org/records/17121721

This provides an alternative to rebuilding the available database
components locally.

From `final_project/`, run:

```bash
cd streamlit_tool
python setup_databases.py
streamlit run app.py
```

The app opens at http://localhost:8501.

The setup script downloads and extracts the archive each time it runs.
After setup, start the app directly:

```bash
streamlit run app.py
```

Building the PubMed database alone does not create the app's graph
subgraphs, node metadata, or question mappings. These are separate
inputs supplied through the prepared data or local preprocessing.

## Local LLM benchmark

The benchmark combines the exported graph pathways with retrieved
PubMed abstracts and generates a structured analysis for each gene.

The selected genes are:

```text
MAOA, DRD4, HTR2C, ADRA1A, ADRA2A, OPRM1, OPRK1, KCND2,
KCND3, KCNQ2, KCNQ3, CHRNA3, CHRNB4, CHRM1, ABAT, GPT2
```

The retrieval query is:

```text
Role of {gene} in treatment-resistant major depressive disorder and antidepressant response mechanisms
```

### Configuration

| Setting | Value |
| --- | --- |
| Model families | Qwen3, DeepSeek-R1, and Llama |
| Number of models | 12 |
| Genes | 16 |
| Repetitions | 100 |
| Retrieved abstracts | 10 per query |
| Temperature | 0.1 |
| Maximum predicted tokens | 2000 |
| Ollama endpoint | `http://localhost:11434/api/generate` |

A complete run contains 19,200 gene-model outputs.

Before running, check that:

- `pathways.csv` is in `benchmarking/`.
- The selected ChromaDB collection has been created.
- The database path and collection name match the benchmark configuration.
- Ollama is running with the required models available.

From `final_project/benchmarking/`, run:

```bash
python rag_pipeline.py
```

Results are saved in repetition folders:

```text
rag_results/rep_01/
rag_results/rep_02/
...
rag_results/rep_100/
```

The JSON outputs contain the retrieved PMIDs, graph evidence index,
raw model response, parsed response, validation status, and response
time. CSV summaries are saved alongside them.

Existing model result files are skipped when the script is restarted.

`schema.py` defines the expected response structure:

- Gene.
- Answer.
- Key mechanisms.
- Relevant graph references and PMIDs.
- Limitations.
- Clinical relevance.

## Evaluation

### Structure and citation evaluation

`evaluation/evaluation.py` runs the evaluation modules in
`evaluation/src/evaluation_pipeline/`.

It checks the required response sections and calculates PMID precision,
recall, and F1. It also generates summary tables, plots, and reports.

The expected dataset contains:

```text
100 repetitions × 12 models × 16 genes = 19,200 outputs
```

Set `RESULTS_DIR` in
`evaluation/src/evaluation_pipeline/config.py` to the directory
containing the benchmark's `rep_*` folders.

From `final_project/evaluation/`, run:

```bash
python evaluation.py
```

Outputs are saved in `evaluation_results/`.

The normal evaluation uses:

```python
ENABLE_LLM_JUDGE = False
```

PMID precision measures the proportion of cited PMIDs present in the
retrieved evidence. PMID recall measures the proportion of retrieved
PMIDs cited in the response. F1 combines the two.

These metrics check citation membership. They do not establish whether
a cited paper supports the associated claim.

### AI faithfulness evaluation

`evaluation/fedility.py` performs a separate AI check on selected results
using `gpt-4o-mini`.

It compares claims containing graph references, such as `[T1]`, with
the corresponding pathways stored in `kg_index`. The prompt inside
the script defines the faithfulness and claim-support judgments.

Set `RESULTS_DIR` in this script to the benchmark results directory.
An `OPENAI_API_KEY` is required.

The saved configuration uses a small test input:

```python
TEST_MODE = True
TEST_MAX_FILES = 1
TEST_MAX_ROWS_PER_FILE = 1
```

Adjust these limits for the intended subset, or disable test mode
to process the full input selected by the script.

From `final_project/evaluation/`, run:

```bash
python fedility.py
```

Outputs are saved in `kg_triplet_judgment_results/`.

### Additional analysis

| File | Purpose |
| --- | --- |
| `pmid_heatmaps.py` | PMID-level heatmaps from saved evaluation tables |
| `pmid_top_cited.py` | Frequently cited PMIDs per gene |

## Data and reproducibility

Large datasets, local databases, credentials, and generated results
are stored separately from the source code.

PubMed downloads can change over time. Reproducing an existing run
requires the same corpus, graph exports, model versions, and experiment
settings.

This project is research software. Generated explanations require
review and are not validated clinical recommendations.

## References

- [PrimeKG project](https://zitniklab.hms.harvard.edu/projects/PrimeKG/)
- [PrimeKG dataset](https://doi.org/10.7910/DVN/IXA7BM)
- [PrimeKG publication](https://doi.org/10.1038/s41597-023-01960-3)
- [Prepared app databases](https://zenodo.org/records/17121721)
