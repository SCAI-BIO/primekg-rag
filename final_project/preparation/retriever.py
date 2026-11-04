# # This is a two-stage pipeline to prepare data for AI analysis.
# # 1. MAPPING: Maps questions to the most semantically similar nodes in the `node_db`.
# # 2. SUBGRAPH EXTRACTION: Extracts focused subgraphs for each matched node from `kg.csv`.

# import logging
# from pathlib import Path
# import pandas as pd
# import chromadb
# from chromadb.utils import embedding_functions
# import duckdb
# from tqdm import tqdm
# import sys
# import re

# # --- Configuration ---
# BASE_DIR = Path(__file__).parent
# LOG_FILE_PATH = BASE_DIR / "data_preparation.log"

# # --- Stage 1: Mapping Config ---
# QUESTIONS_CSV_PATH = BASE_DIR / "questions_for_mapping.csv"
# NODE_DB_PATH = BASE_DIR / "node_db"
# MAPPING_OUTPUT_CSV_PATH = BASE_DIR / "question_to_node_mappings.csv"
# MAPPINGS_DB_PATH = BASE_DIR / "mappings_db"
# EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# NODE_COLLECTION_NAME = "node_embeddings"
# MAPPINGS_COLLECTION_NAME = "question_to_node_mappings"

# # --- Stage 2: Subgraph Extraction Config ---
# KG_CSV_PATH = BASE_DIR / "kg.csv"
# SUBGRAPH_OUTPUT_DIR = BASE_DIR / "subgraphs"

# # --- Logging Configuration ---
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# file_handler = logging.FileHandler(LOG_FILE_PATH, mode="w", encoding="utf-8")
# file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
# file_handler.setFormatter(file_formatter)
# stream_handler = logging.StreamHandler(sys.stdout)
# stream_formatter = logging.Formatter("%(message)s")
# stream_handler.setFormatter(stream_formatter)
# logger.addHandler(file_handler)
# logger.addHandler(stream_handler)


# def preprocess_question(text: str) -> str:
#     """Cleans up a question to remove noise before embedding."""
#     text = text.lower()
#     text = re.sub(r"current\??|past\??|status options\??|lifetime:|v1.2", "", text)
#     text = re.sub(r'[?:\'"]', "", text)
#     text = re.sub(r"\s+", " ", text).strip()
#     return text


# def map_questions_to_nodes():
#     """Stage 1: Reads questions, queries the node_db, stores the mappings, and returns a list of unique best-match node names."""
#     logger.info("--- STAGE 1: Mapping Questions to Knowledge Graph Nodes ---")
#     if not QUESTIONS_CSV_PATH.is_file():
#         logger.critical(f"Questions file not found: '{QUESTIONS_CSV_PATH}'")
#         return None
#     questions_df = pd.read_csv(QUESTIONS_CSV_PATH)
#     questions = questions_df["question_text"].dropna().astype(str).tolist()
#     processed_questions = [preprocess_question(q) for q in questions]
#     logger.info(f"Loaded and processed {len(questions)} questions.")
#     if not NODE_DB_PATH.is_dir():
#         logger.critical(f"Node database not found at '{NODE_DB_PATH}'.")
#         return None
#     try:
#         embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
#             model_name=EMBEDDING_MODEL_NAME
#         )
#         node_client = chromadb.PersistentClient(path=str(NODE_DB_PATH))
#         node_collection = node_client.get_collection(
#             name=NODE_COLLECTION_NAME, embedding_function=embedding_fn
#         )

#         mappings_client = chromadb.PersistentClient(path=str(MAPPINGS_DB_PATH))
#         mappings_collection = mappings_client.get_or_create_collection(
#             name=MAPPINGS_COLLECTION_NAME
#         )

#     except Exception as e:
#         logger.critical(f"Could not connect to ChromaDB: {e}")
#         return None

#     results = node_collection.query(
#         query_texts=processed_questions,
#         n_results=1,
#         include=["documents", "metadatas", "distances"],
#     )
#     all_mappings = []
#     if results:
#         for i, q in enumerate(questions):
#             if (results["documents"] and len(results["documents"]) > i and results["documents"][i]):
#                 dist = results["distances"][i][0]
#                 similarity = 1 - dist
#                 doc = results["documents"][i][0]
#                 meta = results["metadatas"][i][0]
#                 all_mappings.append(
#                     {
#                         "original_question": q,
#                         "best_match_node_name": doc,
#                         "similarity_score": f"{similarity:.4f}",
#                         "node_id": meta.get("id"),
#                         "node_type": meta.get("type"),
#                     }
#                 )
#     if not all_mappings:
#         logger.warning("No mappings were found.")
#         return None

#     mappings_df = pd.DataFrame(all_mappings)
#     mappings_df.to_csv(MAPPING_OUTPUT_CSV_PATH, index=False)

#     logger.info(f"Storing {len(mappings_df)} mappings in the database...")
#     mappings_collection.upsert(
#         ids=[f"map_{i}" for i in range(len(mappings_df))],
#         documents=mappings_df["original_question"].tolist(),
#         metadatas=mappings_df.to_dict("records"),
#     )

#     logger.info("(SUCCESS) Stage 1 complete. Mappings saved to CSV and database.")
#     return mappings_df["best_match_node_name"].unique().tolist()


# def extract_subgraphs_for_nodes(nodes_to_process: list):
#     """Stage 2: Takes a list of node names and extracts a deduplicated subgraph for each one."""
#     logger.info("\n--- STAGE 2: Extracting Subgraphs for Matched Nodes ---")
#     SUBGRAPH_OUTPUT_DIR.mkdir(exist_ok=True)
#     if not KG_CSV_PATH.is_file():
#         logger.critical(f"Knowledge graph file not found: '{KG_CSV_PATH}'.")
#         return
#     logger.info(f"Extracting subgraphs for {len(nodes_to_process)} unique nodes...")
#     for node_name in tqdm(nodes_to_process, desc="Extracting Subgraphs"):
#         try:
#             safe_node_name = str(node_name).replace("'", "''")
#             query = (
#                 f"SELECT * FROM read_csv_auto('{KG_CSV_PATH}', ignore_errors=true) "
#                 f"WHERE x_name = '{safe_node_name}' OR y_name = '{safe_node_name}';"
#             )
#             subgraph_df = duckdb.query(query).to_df()
#             if not subgraph_df.empty:
#                 needs_swap = subgraph_df["x_name"] > subgraph_df["y_name"]
#                 for col in ["id", "name", "type", "source"]:
#                     x_col, y_col = f"x_{col}", f"y_{col}"
#                     if x_col in subgraph_df.columns and y_col in subgraph_df.columns:
#                         x_orig, y_orig = (
#                             subgraph_df.loc[needs_swap, x_col].copy(),
#                             subgraph_df.loc[needs_swap, y_col].copy(),
#                         )
#                         subgraph_df.loc[needs_swap, x_col] = y_orig
#                         subgraph_df.loc[needs_swap, y_col] = x_orig
#                 subgraph_df.drop_duplicates(inplace=True)
#                 safe_filename = "".join(
#                     c for c in str(node_name) if c.isalnum() or c in " _-"
#                 ).rstrip()
#                 output_path = SUBGRAPH_OUTPUT_DIR / f"{safe_filename}_subgraph.csv"
#                 subgraph_df.to_csv(output_path, index=False)
#         except Exception as e:
#             logger.error(f"An error occurred for node '{node_name}': {e}")
#             continue
#     logger.info(
#         f"(SUCCESS) Stage 2 complete. Subgraph files created in '{SUBGRAPH_OUTPUT_DIR}'."
#     )


# if __name__ == "__main__":
#     # Run Stage 1
#     best_match_nodes = map_questions_to_nodes()

#     # Run Stage 2 only if Stage 1 was successful
#     if best_match_nodes:
#         extract_subgraphs_for_nodes(best_match_nodes)

#     logger.info("\n--- Data Preparation Pipeline Finished ---")

# This is a two-stage pipeline to prepare data for AI analysis.
# 1. MAPPING: Maps questions to the most semantically similar nodes in the `node_db`.
# 2. SUBGRAPH EXTRACTION: Extracts focused subgraphs for each matched node from `kg.csv`, now including 2-hop neighborhoods.

import logging
from pathlib import Path
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import duckdb
from tqdm import tqdm
import sys
import re
import numpy as np

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
SUBGRAPH_OUTPUT_DIR = BASE_DIR / "new_subgraphs"  # Changed from 'subgraphs' to 'new_subgraphs'

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
    """
    Stage 1: Maps questions to nodes in the knowledge graph using pre-computed embeddings.
    - Loads questions from questions_for_mapping.csv
    - Uses pre-computed node embeddings from node_db
    - Computes cosine similarity between questions and node names
    - Shows all similarity scores without filtering
    - Returns list of unique best-match node names
    """
    logger.info("--- STAGE 1: Mapping Questions to Knowledge Graph Nodes ---")
    
    # Load and process questions
    if not QUESTIONS_CSV_PATH.is_file():
        logger.critical(f"Questions file not found: '{QUESTIONS_CSV_PATH}'")
        return None
    
    # Load questions
    questions_df = pd.read_csv(QUESTIONS_CSV_PATH, header=None, names=['question'])
    questions = questions_df['question'].dropna().astype(str).tolist()
    logger.info(f"Loaded {len(questions)} questions.")
    
    # Initialize embedding function
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
    
    # Connect to node collection
    try:
        node_client = chromadb.PersistentClient(path=str(NODE_DB_PATH))
        node_collection = node_client.get_collection(
            name=NODE_COLLECTION_NAME,
            embedding_function=embedding_fn
        )
        logger.info(f"Connected to node collection: {NODE_COLLECTION_NAME}")
    except Exception as e:
        logger.critical(f"Could not connect to node collection: {e}")
        return None
    
    # Get all node embeddings, metadatas, and documents (often the human-readable name)
    logger.info("Retrieving node embeddings and metadata...")
    node_data = node_collection.get(include=["embeddings", "metadatas", "documents"])
    
    if not node_data["ids"]:
        logger.critical("No nodes found in the node collection.")
        return None
    
    node_embeddings = node_data["embeddings"]
    node_metadatas = node_data["metadatas"]
    node_ids = node_data["ids"]
    node_docs = node_data.get("documents", None)
    
    # Process each question
    results = []
    
    for question in questions:
        # Embed the question
        question_embedding = embedding_fn([question])[0]
        
        # Calculate cosine similarity with all nodes
        similarities = []
        for i, node_embedding in enumerate(node_embeddings):
            # Calculate cosine similarity
            similarity = np.dot(question_embedding, node_embedding) / (
                np.linalg.norm(question_embedding) * np.linalg.norm(node_embedding)
            )
            # Robustly pick a human-readable node name
            meta = node_metadatas[i] if i < len(node_metadatas) else {}
            doc_name = node_docs[i] if (node_docs and i < len(node_docs)) else None
            def pick_name(meta_obj, doc_val, fallback_id):
                # Try document first, then common metadata keys
                candidates = [
                    doc_val,
                    meta_obj.get("name") if isinstance(meta_obj, dict) else None,
                    meta_obj.get("label") if isinstance(meta_obj, dict) else None,
                    meta_obj.get("preferred_name") if isinstance(meta_obj, dict) else None,
                    meta_obj.get("display_name") if isinstance(meta_obj, dict) else None,
                ]
                for c in candidates:
                    if isinstance(c, str) and c.strip():
                        return c.strip()
                return fallback_id
            node_name = pick_name(meta, doc_name, node_ids[i])
            similarities.append((node_name, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get only the best match
        if similarities:
            best_node, best_score = similarities[0]
            results.append({
                'question': question,
                'best_match_node': best_node,
                'similarity_score': best_score
            })
            logger.info(f"Question: {question[:50]}... -> Best match: {best_node} (similarity: {best_score:.4f})")
        else:
            logger.warning(f"No matches found for question: {question}")
    
    # Save results to CSV
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv('question_node_matches.csv', index=False, float_format='%.4f')
        logger.info(f"Saved {len(results)} question-node matches to question_node_matches.csv")
        
        # Get unique nodes for next stage
        unique_nodes = list(set(r['best_match_node'] for r in results))
        logger.info(f"Found {len(unique_nodes)} unique nodes to process in Stage 2.")
        return unique_nodes
    
    logger.warning("No valid matches found above the similarity threshold.")
    return []


def get_node_embeddings():
    """Initialize and return the node embeddings collection."""
    try:
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )
        node_client = chromadb.PersistentClient(path=str(NODE_DB_PATH))
        return node_client.get_collection(
            name=NODE_COLLECTION_NAME,
            embedding_function=embedding_fn
        )
    except Exception as e:
        logger.critical(f"Could not connect to node collection: {e}")
        return None

def find_matching_nodes(query: str, threshold: float = 0.7) -> tuple[set, float]:
    """
    Find the best matching node in the knowledge graph for the query.
    Uses cosine similarity of embeddings to find the best match.
    
    Args:
        query: The node name to find a match for
        threshold: Minimum similarity score (0-1) to consider a match
        
    Returns:
        tuple: (set containing the best matching node name or empty set if no good match, 
               similarity score of the best match or 0 if no match)
    """
    node_collection = get_node_embeddings()
    if not node_collection:
        return set(), 0.0
    
    try:
        # Get the single best match
        results = node_collection.query(
            query_texts=[query],
            n_results=1,
            include=["documents", "distances"]
        )
        
        if not results or not results['documents'][0]:
            logger.warning(f"No matches found for query: '{query}'")
            return set(), 0.0
        
        # Get the best match and its similarity score
        best_match = results['documents'][0][0]
        similarity = 1 - results['distances'][0][0]  # Convert distance to similarity
        
        if similarity >= threshold:
            logger.info(f"Best match for '{query}': '{best_match}' (similarity: {similarity:.3f})")
            return {best_match}, similarity
        else:
            logger.warning(f"No good match found for '{query}'. Best match '{best_match}' has similarity {similarity:.3f} < {threshold}")
            return set(), similarity
    
    except Exception as e:
        logger.error(f"Error finding matching nodes for '{query}': {e}")
        return set(), 0.0
        return set()

def extract_subgraphs_for_nodes(nodes_to_process: list):
    """
    Stage 2: Takes a list of node names and extracts their subgraphs from kg.csv.
    Uses the best match from node_db without any threshold filtering.
    
    Args:
        nodes_to_process: List of node names to extract subgraphs for
    """
    logger.info("\n--- STAGE 2: Extracting Subgraphs for Matched Nodes ---")
    SUBGRAPH_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    if not KG_CSV_PATH.is_file():
        logger.critical(f"Knowledge graph file not found: '{KG_CSV_PATH}'.")
        return
    
    # Skip pandas loading to avoid memory pressure - DuckDB will read CSV directly
    logger.info(f"Will query knowledge graph from {KG_CSV_PATH} using DuckDB...")
    
    # Get actual node names from the node_db collection
    try:
        node_client = chromadb.PersistentClient(path=str(NODE_DB_PATH))
        node_collection = node_client.get_collection(
            name=NODE_COLLECTION_NAME,
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL_NAME
            )
        )
        logger.info(f"Connected to node collection: {NODE_COLLECTION_NAME}")
    except Exception as e:
        logger.critical(f"Could not connect to node collection: {e}")
        return
    
    # Get actual node names from metadata for each node ID
    actual_node_names = []
    for node_id in nodes_to_process:
        try:
            results = node_collection.get(ids=[node_id], include=["metadatas", "documents"])
            if results and results["metadatas"] and results["metadatas"][0]:
                metadata = results["metadatas"][0]
                # Try different possible keys for the actual node name
                actual_name = metadata.get("name") or metadata.get("node_name") or metadata.get("original_name")
                
                # If no name in metadata, try the document content
                if not actual_name and results["documents"] and results["documents"][0]:
                    actual_name = results["documents"][0]
                
                # If still no name, extract from node_id (remove 'node_' prefix if present)
                if not actual_name:
                    actual_name = node_id.replace("node_", "") if node_id.startswith("node_") else node_id
                
                actual_node_names.append(actual_name)
                logger.info(f"Node ID '{node_id}' -> Actual name: '{actual_name}'")
            else:
                logger.warning(f"No metadata found for node ID: {node_id}")
                # Extract from node_id as fallback
                actual_name = node_id.replace("node_", "") if node_id.startswith("node_") else node_id
                actual_node_names.append(actual_name)
        except Exception as e:
            logger.error(f"Error looking up node {node_id}: {e}")
            actual_name = node_id.replace("node_", "") if node_id.startswith("node_") else node_id
            actual_node_names.append(actual_name)
    
    if not actual_node_names:
        logger.warning("No valid nodes found to process. Exiting.")
        return
    
    # Initialize DuckDB and create a view over the CSV directly (avoid registering huge pandas DF)
    try:
        con = duckdb.connect(database=":memory:")
        con.execute("PRAGMA threads=1")  # Single-threaded to avoid Windows segfaults
        con.execute("PRAGMA memory_limit='2GB'")
        # DuckDB does not allow prepared parameters in DDL here; inline-escape the path
        csv_path_sql = str(KG_CSV_PATH).replace("'", "''")
        con.execute(
            f"CREATE OR REPLACE VIEW kg AS SELECT * FROM read_csv_auto('{csv_path_sql}', header=true, all_varchar=true)"
        )
        logger.info("DuckDB connection established and kg view created from CSV.")
        # Get schema for empty CSV creation
        schema_df = con.execute("SELECT * FROM kg LIMIT 0").df()
    except Exception as e:
        logger.error(f"Failed to initialize DuckDB over CSV: {e}")
        return

    for node_name in tqdm(actual_node_names, desc="Extracting Subgraphs"):
        try:
            logger.info(f"\nProcessing node: '{node_name}'")
            name_lc = str(node_name).strip().lower()

            # Build candidate ID set from metadata and node_id string
            candidate_ids = set()
            try:
                # Try to retrieve metadata again for this node to fetch its 'id'
                results = node_collection.get(ids=[node_name], include=["metadatas"])
                if results and results["metadatas"] and results["metadatas"][0]:
                    meta_id = results["metadatas"][0].get("id")
                    if meta_id:
                        candidate_ids.add(str(meta_id))
            except Exception as _:
                pass
            # Parse numeric tokens from node_name in case it's composite like node_8187_2050_...
            for tok in re.findall(r"\d+", str(node_name)):
                candidate_ids.add(tok)

            # 1-hop by name: exact case-insensitive match using LOWER on CSV columns
            q_by_name = "select * from kg where lower(x_name) = ? or lower(y_name) = ?"
            one_hop = con.execute(q_by_name, [name_lc, name_lc]).df()

            # If no edges by name, try ID-based matching
            if one_hop.empty and candidate_ids:
                q_by_id = (
                    "select * from kg where x_id in (select * from unnest(?)) or y_id in (select * from unnest(?))"
                )
                one_hop = con.execute(q_by_id, [list(candidate_ids), list(candidate_ids)]).df()

            if one_hop.empty:
                # If no 1-hop edges, create an empty df with correct columns
                all_edges = pd.DataFrame(columns=schema_df.columns)
                logger.info(f"No 1-hop edges found for '{node_name}'.")
            else:
                logger.info(f"Found {len(one_hop)} 1-hop edges for '{node_name}'")
                all_edges = one_hop

            # Create output directory if it doesn't exist
            SUBGRAPH_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

            # Save the subgraph (even if empty)
            safe_filename = "".join(c for c in str(node_name) if c.isalnum() or c in " _-").rstrip()
            if not safe_filename:
                safe_filename = f"node_{hash(node_name) % 10000}"
            output_path = SUBGRAPH_OUTPUT_DIR / f"{safe_filename}_subgraph.csv"

            all_edges.to_csv(output_path, index=False)
            logger.info(
                f"Saved {'empty ' if all_edges.empty else ''}subgraph to: {output_path}"
            )

        except Exception as e:
            logger.error(f"An error occurred for node '{node_name}': {e}")
            # Still create an empty CSV file for this node
            try:
                safe_filename = "".join(c for c in str(node_name) if c.isalnum() or c in " _-").rstrip()
                if not safe_filename:
                    safe_filename = f"node_{hash(node_name) % 10000}"
                output_path = SUBGRAPH_OUTPUT_DIR / f"{safe_filename}_subgraph.csv"
                empty_df = pd.DataFrame(columns=schema_df.columns)
                empty_df.to_csv(output_path, index=False)
                logger.info(f"Created empty CSV due to error: {output_path}")
            except Exception as csv_error:
                logger.error(f"Failed to create empty CSV for '{node_name}': {csv_error}")
            continue
    logger.info(
        f"(SUCCESS) Stage 2 complete. 2-hop subgraph files created in '{SUBGRAPH_OUTPUT_DIR}'."
    )
    logger.info(f"Subgraph files saved to: {SUBGRAPH_OUTPUT_DIR.absolute()}")

if __name__ == "__main__":
    # Run Stage 1: Map questions to best-match nodes using embeddings (no threshold)
    best_match_nodes = map_questions_to_nodes()

    # NOTE: Stage 2 disabled per request. Only mapping is executed when running this script.
    # To re-enable subgraph extraction, uncomment the lines below.
    # # Run Stage 2: Extract subgraphs via fast DuckDB lookups on kg.csv
    # if best_match_nodes:
    #     extract_subgraphs_for_nodes(best_match_nodes)

    logger.info("\n--- Data Preparation Pipeline Finished ---")
