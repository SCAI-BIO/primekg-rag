# This is a two-stage pipeline to prepare data for AI analysis.
# 1. MAPPING: Maps questions to the most semantically similar nodes in the `node_db`.
# 2. SUBGRAPH EXTRACTION: Extracts focused subgraphs for each matched node from `kg.csv`.

import logging
from pathlib import Path
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import duckdb
from tqdm import tqdm
import sys
import re

# --- Configuration ---
BASE_DIR = Path(__file__).parent
LOG_FILE_PATH = BASE_DIR / "data_preparation.log"

# --- Stage 1: Mapping Config ---
QUESTIONS_CSV_PATH = BASE_DIR / "questions_for_mapping.csv"
NODE_DB_PATH = BASE_DIR / "node_db"
MAPPING_OUTPUT_CSV_PATH = BASE_DIR / "question_to_node_mappings.csv"
MAPPINGS_DB_PATH = BASE_DIR / "mappings_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
NODE_COLLECTION_NAME = "node_embeddings"
MAPPINGS_COLLECTION_NAME = "question_to_node_mappings"

# --- Stage 2: Subgraph Extraction Config ---
KG_CSV_PATH = BASE_DIR / "kg.csv"
SUBGRAPH_OUTPUT_DIR = BASE_DIR / "subgraphs"

# --- Logging Configuration ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(LOG_FILE_PATH, mode="w", encoding="utf-8")
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)
stream_handler = logging.StreamHandler(sys.stdout)
stream_formatter = logging.Formatter("%(message)s")
stream_handler.setFormatter(stream_formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def preprocess_question(text: str) -> str:
    """Cleans up a question to remove noise before embedding."""
    text = text.lower()
    text = re.sub(r"current\??|past\??|status options\??|lifetime:|v1.2", "", text)
    text = re.sub(r'[?:\'"]', "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def map_questions_to_nodes():
    """Stage 1: Reads questions, queries the node_db, stores the mappings, and returns a list of unique best-match node names."""
    logger.info("--- STAGE 1: Mapping Questions to Knowledge Graph Nodes ---")
    if not QUESTIONS_CSV_PATH.is_file():
        logger.critical(f"Questions file not found: '{QUESTIONS_CSV_PATH}'")
        return None
    questions_df = pd.read_csv(QUESTIONS_CSV_PATH)
    questions = questions_df["question_text"].dropna().astype(str).tolist()
    processed_questions = [preprocess_question(q) for q in questions]
    logger.info(f"Loaded and processed {len(questions)} questions.")
    if not NODE_DB_PATH.is_dir():
        logger.critical(f"Node database not found at '{NODE_DB_PATH}'.")
        return None
    try:
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )
        node_client = chromadb.PersistentClient(path=str(NODE_DB_PATH))
        node_collection = node_client.get_collection(
            name=NODE_COLLECTION_NAME, embedding_function=embedding_fn
        )

        mappings_client = chromadb.PersistentClient(path=str(MAPPINGS_DB_PATH))
        mappings_collection = mappings_client.get_or_create_collection(
            name=MAPPINGS_COLLECTION_NAME
        )

    except Exception as e:
        logger.critical(f"Could not connect to ChromaDB: {e}")
        return None

    results = node_collection.query(
        query_texts=processed_questions,
        n_results=1,
        include=["documents", "metadatas", "distances"],
    )
    all_mappings = []
    if results:
        for i, q in enumerate(questions):
            if (
                results["documents"]
                and len(results["documents"]) > i
                and results["documents"][i]
            ):
                dist = results["distances"][i][0]
                similarity = 1 - dist
                doc = results["documents"][i][0]
                meta = results["metadatas"][i][0]
                all_mappings.append(
                    {
                        "original_question": q,
                        "best_match_node_name": doc,
                        "similarity_score": f"{similarity:.4f}",
                        "node_id": meta.get("id"),
                        "node_type": meta.get("type"),
                    }
                )
    if not all_mappings:
        logger.warning("No mappings were found.")
        return None

    mappings_df = pd.DataFrame(all_mappings)
    mappings_df.to_csv(MAPPING_OUTPUT_CSV_PATH, index=False)

    logger.info(f"Storing {len(mappings_df)} mappings in the database...")
    mappings_collection.upsert(
        ids=[f"map_{i}" for i in range(len(mappings_df))],
        documents=mappings_df["original_question"].tolist(),
        metadatas=mappings_df.to_dict("records"),
    )

    logger.info(f"(SUCCESS) Stage 1 complete. Mappings saved to CSV and database.")
    return mappings_df["best_match_node_name"].unique().tolist()


def extract_subgraphs_for_nodes(nodes_to_process: list):
    """Stage 2: Takes a list of node names and extracts a deduplicated subgraph for each one."""
    logger.info("\n--- STAGE 2: Extracting Subgraphs for Matched Nodes ---")
    SUBGRAPH_OUTPUT_DIR.mkdir(exist_ok=True)
    if not KG_CSV_PATH.is_file():
        logger.critical(f"Knowledge graph file not found: '{KG_CSV_PATH}'.")
        return
    logger.info(f"Extracting subgraphs for {len(nodes_to_process)} unique nodes...")
    for node_name in tqdm(nodes_to_process, desc="Extracting Subgraphs"):
        try:
            safe_node_name = str(node_name).replace("'", "''")
            subgraph_df = duckdb.query(
                f"""SELECT * FROM read_csv_auto('{KG_CSV_PATH}', ignore_errors=true) WHERE x_name = '{safe_node_name}' OR y_name = '{safe_node_name}';"""
            ).to_df()
            if not subgraph_df.empty:
                needs_swap = subgraph_df["x_name"] > subgraph_df["y_name"]
                for col in ["id", "name", "type", "source"]:
                    x_col, y_col = f"x_{col}", f"y_{col}"
                    if x_col in subgraph_df.columns and y_col in subgraph_df.columns:
                        x_orig, y_orig = (
                            subgraph_df.loc[needs_swap, x_col].copy(),
                            subgraph_df.loc[needs_swap, y_col].copy(),
                        )
                        subgraph_df.loc[needs_swap, x_col] = y_orig
                        subgraph_df.loc[needs_swap, y_col] = x_orig
                subgraph_df.drop_duplicates(inplace=True)
                safe_filename = "".join(
                    c for c in str(node_name) if c.isalnum() or c in " _-"
                ).rstrip()
                output_path = SUBGRAPH_OUTPUT_DIR / f"{safe_filename}_subgraph.csv"
                subgraph_df.to_csv(output_path, index=False)
        except Exception as e:
            logger.error(f"An error occurred for node '{node_name}': {e}")
            continue
    logger.info(
        f"(SUCCESS) Stage 2 complete. Subgraph files created in '{SUBGRAPH_OUTPUT_DIR}'."
    )


if __name__ == "__main__":
    # Run Stage 1
    best_match_nodes = map_questions_to_nodes()

    # Run Stage 2 only if Stage 1 was successful
    if best_match_nodes:
        extract_subgraphs_for_nodes(best_match_nodes)

    logger.info("\n--- Data Preparation Pipeline Finished ---")
