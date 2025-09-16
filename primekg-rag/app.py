import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import chromadb
from typing import List, Dict, Any, Optional, Tuple
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import time
import re
from export_utils import export_detailed_path
import openai
import csv
from pyvis.network import Network
import streamlit.components.v1 as components
from pathlib import Path
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
BASE_DIR = Path(__file__).parent
SUBGRAPHS_DIR = BASE_DIR / "new_subgraphs"
NODES_CSV_PATH = BASE_DIR / "nodes.csv"
ANALYSIS_DB_PATH = BASE_DIR / "analyses_db"
ANALYSIS_COLLECTION_NAME = "medical_analyses"
OPENAI_MODEL_NAME = "gpt-4o"
QUESTION_NODE_MAPPING_PATH = BASE_DIR / "best_question_matches.csv"
PUBMED_DB_PATH = BASE_DIR / "pubmed_db"

# Enhanced Retrieval Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_PAPERS = 20
SIMILARITY_THRESHOLD = 0.5
TITLE_WEIGHT = 0.8
ABSTRACT_WEIGHT = 0.2
ENHANCE_QUERY_FOR_DIAGNOSTIC_THERAPEUTIC = True

# Initialize OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    openai_client = openai.OpenAI(api_key=openai_api_key)
else:
    st.warning("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")
    openai_client = None

# --- Graph Visualization Settings ---
MAX_NODES = 25
LABEL_MAX_LENGTH = 20
NODE_COLORS = {
    "gene/protein": "#4CAF50",
    "drug": "#2196F3",
    "disease": "#9C27B0",
    "phenotype": "#00BCD4",
    "pathway": "#FFC107",
    "molecular_function": "#F44336",
    "biological_process": "#E91E63",
    "cellular_component": "#673AB7",
    "compound": "#8BC34A",
    "chemical_compound": "#8BC34A",
    "biological_entity": "#FFEB3B",
    "exposure": "#FF9800",
    "symptom": "#CDDC39",
    "default": "#607D8B",
}

# --- Analysis Rendering Helpers ---
def render_analysis_with_collapse(markdown_text: str, bullet_threshold: int = 10, targets: list[str] | None = None):
    """Render the analysis markdown, collapsing long sections.

    - targets: list of section titles (level-3) to collapse if they have > bullet_threshold bullets.
      Defaults to ["Key Clinical Relationships"].
    """
    try:
        import re
        if targets is None:
            targets = ["Key Clinical Relationships"]

        # Light CSS for readability
        st.markdown(
            """
            <style>
            .analysis-text { max-width: 980px; word-wrap: break-word; overflow-wrap: anywhere; }
            .analysis-text p { line-height: 1.6; }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Normalize line breaks and convert bare numeric citations to PMID form
        def _norm_pmids(s: str) -> str:
            def repl(m: re.Match) -> str:
                inside = m.group(1)
                if re.search(r"PMID\s*:\s*\d+", inside, flags=re.IGNORECASE):
                    return f"({inside})"
                return "(" + re.sub(r"\b\d{5,10}\b", lambda nm: f"PMID:{nm.group(0)}", inside) + ")"
            return re.sub(r"\(([^)]+)\)", repl, s)
        text = _norm_pmids(markdown_text.replace('\r\n', '\n').replace('\r', '\n'))

        def process_once(src_text: str, section_title: str) -> str:
            # Build header regex
            header_pattern = re.compile(rf"^###\s+{re.escape(section_title)}\s*$", re.IGNORECASE | re.MULTILINE)
            header_match = header_pattern.search(src_text)
            if not header_match:
                return src_text

            header_start = header_match.start()
            header_end = header_match.end()
            pre = src_text[:header_start]
            # End of section is next level-3 header
            next_header = re.search(r"^###\s+.+$", src_text[header_end:], flags=re.MULTILINE)
            section_end_idx = header_end + (next_header.start() if next_header else len(src_text) - header_end)
            section = src_text[header_end:section_end_idx]
            post = src_text[section_end_idx:]

            lines = [ln for ln in section.split('\n')]
            bullet_idxs = [i for i, ln in enumerate(lines) if ln.lstrip().startswith(('-', '*', '•'))]
            # If no bullets, treat each non-empty, non-header line as an item
            if not bullet_idxs:
                item_idxs = [i for i, ln in enumerate(lines) if ln.strip() and not ln.strip().startswith('###')]
            else:
                item_idxs = bullet_idxs
            if len(item_idxs) <= bullet_threshold:
                return src_text

            items = [lines[i] for i in item_idxs]
            first = items[:bullet_threshold]
            rest = items[bullet_threshold:]

            # Rebuild section content with expander for rest
            rebuilt = []
            rebuilt.append(f"### {section_title}")
            if first:
                rebuilt.append("\n".join(first))
            if rest:
                # Use a placeholder that Streamlit can render as expander in the final pass
                rebuilt.append(f"\n<EXPANDER title=\"Show {len(rest)} more\">\n" + "\n".join(rest) + "\n</EXPANDER>\n")

            new_section = "\n".join(rebuilt)
            return pre + new_section + post

        # Apply processing for each target sequentially
        processed = text
        for title in targets:
            processed = process_once(processed, title)

        # Render processed text, converting our EXPANDER placeholders to real expanders
        # Split on placeholders to interleave expanders
        parts = processed.split("<EXPANDER")
        # First part before any expander
        st.markdown(f"<div class='analysis-text'>\n{convert_pmids_to_links(parts[0])}\n</div>", unsafe_allow_html=True)
        for chunk in parts[1:]:
            # Extract title and body
            try:
                title_start = chunk.index('title="') + len('title="')
                title_end = chunk.index('"', title_start)
                title = chunk[title_start:title_end]
                body_start = chunk.index('>') + 1
                body_end = chunk.index('</EXPANDER>')
                body = chunk[body_start:body_end]
            except Exception:
                # Fallback: render raw
                st.markdown(convert_pmids_to_links("<EXPANDER" + chunk), unsafe_allow_html=True)
                continue
            with st.expander(title, expanded=False):
                st.markdown(convert_pmids_to_links(body), unsafe_allow_html=True)
            remainder = chunk[body_end + len('</EXPANDER>'):]
            if remainder.strip():
                st.markdown(convert_pmids_to_links(remainder), unsafe_allow_html=True)
    except Exception:
        st.markdown(convert_pmids_to_links(markdown_text), unsafe_allow_html=True)

# --- Helper Functions ---
def get_subgraph_files():
    """Finds all available subgraph CSV files from disk (legacy fallback)."""
    if not SUBGRAPHS_DIR.is_dir():
        return []
    return sorted([f.name for f in SUBGRAPHS_DIR.glob("*.csv")])

@st.cache_data
def list_analyzed_subgraphs(collection: chromadb.Collection | None) -> list[str]:
    """Return list of subgraph CSV filenames that have stored analyses in ChromaDB.

    We derive the subgraph CSV name from the stored analysis metadata 'filename',
    which is expected to look like '<condition>_subgraph_analysis.txt'.
    """
    if not collection:
        return []
    try:
        total = collection.count()
        analyzed_csvs: set[str] = set()
        batch = 100
        for offset in range(0, total, batch):
            res = collection.get(limit=batch, offset=offset, include=["metadatas"])  # type: ignore[arg-type]
            for meta in (res.get("metadatas") or []):
                if not meta:
                    continue
                analysis_filename = meta.get("filename") or meta.get("analysis_filename")
                condition = meta.get("condition")
                # Prefer explicit csv filename if present in metadata
                csv_name = meta.get("csv_filename") or meta.get("source_csv")
                if csv_name:
                    analyzed_csvs.add(csv_name)
                    continue
                # Else derive from analysis file or condition
                if analysis_filename and analysis_filename.endswith("_subgraph_analysis.txt"):
                    prefix = analysis_filename.replace("_subgraph_analysis.txt", "")
                    analyzed_csvs.add(f"{prefix}_subgraph.csv")
                elif condition:
                    analyzed_csvs.add(f"{condition}_subgraph.csv")
        return sorted(analyzed_csvs)
    except Exception as e:
        st.warning(f"Could not list analyzed subgraphs from DB: {e}")
        return []

def truncate_label(text, max_length):
    """Shortens text for display in the graph."""
    if len(str(text)) > max_length:
        return str(text)[:max_length - 3] + "..."
    return str(text)

@st.cache_data
def load_node_types():
    """Loads nodes.csv and creates a mapping from node name to node type."""
    if not NODES_CSV_PATH.is_file():
        st.warning(f"'{NODES_CSV_PATH.name}' not found. Cannot filter by node type.")
        return {}, []
    try:
        df_nodes = pd.read_csv(NODES_CSV_PATH)
        node_type_map = pd.Series(df_nodes.type.values, index=df_nodes.name).to_dict()
        unique_types = sorted(df_nodes["type"].unique().tolist())
        return node_type_map, unique_types
    except Exception as e:
        st.error(f"Could not read or process '{NODES_CSV_PATH.name}': {e}")
        return {}, []

@st.cache_resource
def get_analysis_collection():
    """Connects to ChromaDB containing the AI analyses."""
    try:
        # Initialize the ChromaDB client with the new API
        client = chromadb.PersistentClient(path=str(ANALYSIS_DB_PATH))
        
        # First, check if the collection exists by listing all collections
        collections = client.list_collections()
        collection_names = [c.name for c in collections]
        
        if ANALYSIS_COLLECTION_NAME in collection_names:
            # Collection exists, get it
            collection = client.get_collection(name=ANALYSIS_COLLECTION_NAME)
            return collection
        else:
            # Collection doesn't exist, create it
            st.warning(f"Creating new '{ANALYSIS_COLLECTION_NAME}' collection as it doesn't exist yet.")
            collection = client.create_collection(
                name=ANALYSIS_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            st.info(f"Successfully created new empty '{ANALYSIS_COLLECTION_NAME}' collection.")
            return collection
            
    except Exception as e:
        st.error(f"Could not connect to or create Analyses Database: {e}")
        import traceback
        st.error(f"Error details: {traceback.format_exc()}")
        return None

@st.cache_resource
def get_pubmed_collection():
    """Connects to the PubMed ChromaDB collection."""
    try:
        db_path = os.getenv("PUBMED_DB_PATH", str(PUBMED_DB_PATH))
        client = chromadb.PersistentClient(path=db_path)
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        collection = client.get_collection(
            name="pubmed_abstracts",
            embedding_function=embedding_function
        )
        st.info("PubMed evidence database connected. 🧬")
        return collection
    except Exception as e:
        st.error(f"Could not connect to PubMed DB: {e}")
        st.warning("Chatbot will use graph data only.")
        return None

# --- MODULAR FUNCTIONS ---
def retrieve_from_pubmed(query_text: str, collection: chromadb.Collection, k: int = 10, save_csv: bool = True, pool_n: int = 50) -> list:
    """
    Retrieves top-k most similar PubMed articles.
    """
    if not collection:
        return []
    try:
        # First, retrieve a larger candidate pool from ChromaDB
        results = collection.query(
            query_texts=[query_text],
            n_results=pool_n,
            include=["documents", "metadatas", "distances"]
        )
        docs = results.get("documents", [[]])[0]
        if not docs:
            st.info("No relevant documents found in the PubMed database for your query.")
            return []

        metadatas = results.get("metadatas", [[]])[0] or [{}]*len(docs)
        titles = [ (metadatas[i] if i < len(metadatas) else {}).get("title", "") for i in range(len(docs)) ]

        # Compute embeddings for query, titles, and abstracts using the same model
        emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        q_emb = np.array(emb_fn([query_text])[0], dtype=float)
        title_embs = [np.array(e, dtype=float) for e in emb_fn(titles)] if titles else []
        abstract_embs = [np.array(e, dtype=float) for e in emb_fn(docs)] if docs else []

        def cos(a: np.ndarray, b: np.ndarray) -> float:
            denom = (np.linalg.norm(a) * np.linalg.norm(b))
            return float(np.dot(a, b) / denom) if denom else 0.0

        # Compute separate sims and weighted combo: 80% title, 20% abstract
        WEIGHT_TITLE = 0.8
        WEIGHT_ABSTRACT = 0.2
        combined = []
        for i in range(len(docs)):
            t_sim = cos(q_emb, title_embs[i]) if i < len(title_embs) and titles[i].strip() else 0.0
            a_sim = cos(q_emb, abstract_embs[i]) if i < len(abstract_embs) and docs[i].strip() else 0.0
            combo = WEIGHT_TITLE * t_sim + WEIGHT_ABSTRACT * a_sim
            combined.append({
                "rank": i + 1,  # preliminary rank before re-sorting
                "pmid": (metadatas[i] if i < len(metadatas) else {}).get("pmid", "N/A"),
                "title": titles[i] if i < len(titles) else "",
                "abstract": docs[i],
                "title_similarity": round(t_sim, 4),
                "abstract_similarity": round(a_sim, 4),
                "combined_similarity": round(combo, 4),
            })

        # Re-rank by combined_similarity and keep top-k
        combined.sort(key=lambda x: x["combined_similarity"], reverse=True)
        topk = combined[:k]
        # Reassign ranks after sorting
        for idx, rec in enumerate(topk, start=1):
            rec["rank"] = idx

        # Optionally save to CSV
        if save_csv:
            try:
                out_dir = BASE_DIR / "pubmed_results"
                out_dir.mkdir(parents=True, exist_ok=True)
                # Sanitize query for filename
                slug = "".join(c for c in query_text if c.isalnum() or c in " _-\t").strip().replace(" ", "_")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = out_dir / f"pubmed_top10_{slug[:60]}_{timestamp}.csv"
                pd.DataFrame(topk).to_csv(out_path, index=False)
                st.success(f"Saved top {k} PubMed matches with similarity to '{out_path.name}'.")
            except Exception as csv_e:
                st.warning(f"Could not save PubMed results to CSV: {csv_e}")

        return topk
    except Exception as e:
        st.error(f"Error querying PubMed DB: {e}")
        return []

def format_subgraph_for_chat_prompt(df: pd.DataFrame) -> str:
    """Formats subgraph into a readable list of facts."""
    required_cols = ['x_name', 'x_type', 'y_name', 'y_type', 'display_relation']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"❌ Subgraph is missing required columns: {missing}")
        return "Subgraph data is invalid or missing required fields."

    facts = []
    for _, row in df.iterrows():
        # Flexible detection of IDs and sources
        def pick(colnames):
            for c in colnames:
                if c in df.columns and pd.notna(row.get(c, None)) and str(row.get(c)).strip():
                    return str(row.get(c)).strip()
            return ""

        x_id = pick(["x_id", "x_identifier", "x_curie", "x_node_id", "x_uid"])
        y_id = pick(["y_id", "y_identifier", "y_curie", "y_node_id", "y_uid"])
        x_src = pick(["x_source", "x_db", "x_namespace", "x_provider", "x_source_db"])
        y_src = pick(["y_source", "y_db", "y_namespace", "y_provider", "y_source_db"])
        edge_src = pick(["edge_source", "relation_source", "predicate_source", "edge_provider", "source"])

        # Compose annotations
        def annot(name, _id, src):
            parts = []
            if _id:
                parts.append(_id)
            if src:
                parts.append(f"@{src}")
            return f"{name} ({' '.join(parts)})" if parts else name

        x_annot = annot(str(row['x_name']), x_id, x_src)
        y_annot = annot(str(row['y_name']), y_id, y_src)
        rel = str(row['display_relation'])
        edge_annot = f"; source={edge_src}" if edge_src else ""

        facts.append(f"{x_annot} —[{rel}{edge_annot}]-> {y_annot}")
    return "\n".join(facts)

def generate_subgraph_summary_with_evidence(df: pd.DataFrame, filename: str) -> None:
    """Generate a summary of the subgraph with evidence displayed in collapsible sections."""
    if df.empty:
        st.warning("No data available for summary.")
        return
    
    required_cols = ['x_name', 'x_type', 'y_name', 'y_type', 'display_relation']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"❌ Subgraph is missing required columns: {missing}")
        return
    
    # Extract the main condition from filename
    condition = filename.replace('_subgraph.csv', '').replace('_', ' ').title()
    
    st.markdown(f"### 📊 Subgraph Summary: {condition}")
    st.markdown("---")
    
    # Basic statistics
    total_relationships = len(df)
    unique_nodes = set(df['x_name'].tolist() + df['y_name'].tolist())
    unique_node_types = set(df['x_type'].tolist() + df['y_type'].tolist())
    unique_relations = df['display_relation'].unique()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Relationships", total_relationships)
    with col2:
        st.metric("Unique Nodes", len(unique_nodes))
    with col3:
        st.metric("Node Types", len(unique_node_types))
    with col4:
        st.metric("Relation Types", len(unique_relations))
    
    # Group relationships by type
    relation_groups = df.groupby('display_relation')
    
    st.markdown("### 🔍 Relationship Evidence")
    
    for relation_type, group in relation_groups:
        with st.expander(f"**{relation_type}** ({len(group)} relationships)", expanded=False):
            st.markdown(f"**Relationship Type:** {relation_type}")
            st.markdown(f"**Number of instances:** {len(group)}")
            st.markdown("**Evidence:**")
            
            # Create a formatted list of relationships
            evidence_list = []
            for _, row in group.iterrows():
                evidence_list.append(
                    f"• **{row['x_name']}** ({row['x_type']}) → **{row['y_name']}** ({row['y_type']})"
                )
            
            # Display in a text area for better readability
            evidence_text = "\n".join(evidence_list)
            st.text_area(
                "Relationships:",
                value=evidence_text,
                height=min(200, len(group) * 20 + 50),
                disabled=True,
                key=f"evidence_{relation_type}_{len(group)}"
            )
    
    # Node type analysis
    with st.expander("📋 Node Type Analysis", expanded=False):
        node_type_counts = {}
        for node_type in unique_node_types:
            count_x = len(df[df['x_type'] == node_type])
            count_y = len(df[df['y_type'] == node_type])
            node_type_counts[node_type] = count_x + count_y
        
        st.markdown("**Node Types in this subgraph:**")
        for node_type, count in sorted(node_type_counts.items(), key=lambda x: x[1], reverse=True):
            st.markdown(f"• **{node_type}**: {count} occurrences")
    
    # Most connected nodes
    with st.expander("🌟 Most Connected Nodes", expanded=False):
        node_connections = {}
        for node in unique_nodes:
            count_x = len(df[df['x_name'] == node])
            count_y = len(df[df['y_name'] == node])
            node_connections[node] = count_x + count_y
        
        top_nodes = sorted(node_connections.items(), key=lambda x: x[1], reverse=True)[:10]
        st.markdown("**Top 10 most connected nodes:**")
        for node, count in top_nodes:
            st.markdown(f"• **{node}**: {count} connections")

def create_pubmed_link(pmid: str) -> str:
    """Creates a clickable PubMed link from a PMID.
    
    Args:
        pmid: The PubMed ID to create a link for
        
    Returns:
        str: HTML anchor tag with the PubMed link
    """
    if not pmid or str(pmid).lower() == 'n/a':
        return 'N/A'
    return f'<a href="https://pubmed.ncbi.nlm.nih.gov/{pmid}/" target="_blank">{pmid}</a>'



def convert_pmids_to_links(text: str) -> str:
    """Convert PMID references into clickable PubMed links."""
    if not text:
        return ""
        
    import re
    from html import escape
    
    # First, escape any existing HTML to prevent XSS
    text = escape(str(text))
    
    # PubMed link template
    def create_pmid_link(pmid: str) -> str:
        return f'<a href="https://pubmed.ncbi.nlm.nih.gov/{pmid}/" target="_blank" rel="noopener noreferrer">PMID:{pmid}</a>'
    
    # Handle PMID:1234 format
    text = re.sub(
        r'(?<!\d)(PMID:?\s*)(\d+)', 
        lambda m: create_pmid_link(m.group(2)),
        text,
        flags=re.IGNORECASE
    )
    
    # Handle (PMID: 1234, 5678) format
    def replace_pmid_group(match):
        pmids = [p.strip() for p in match.group(1).split(',')]
        links = []
        for pmid in pmids:
            if pmid.isdigit():
                links.append(create_pmid_link(pmid))
            else:
                links.append(pmid)
        return f"({', '.join(links)})"
    
    text = re.sub(
        r'\(PMID:?\s*([\d\s,]+)\)',
        replace_pmid_group,
        text,
        flags=re.IGNORECASE
    )
    
    return text

# --- Enhanced Retrieval Functions ---
@st.cache_resource
def get_embedding_model():
    """Get the embedding model for similarity calculations."""
    return SentenceTransformer(EMBEDDING_MODEL)

def enhance_query_for_diagnostic_therapeutic(subgraph_name: str) -> str:
    """Enhance the subgraph name query to focus on diagnostic and therapeutic implications."""
    if not ENHANCE_QUERY_FOR_DIAGNOSTIC_THERAPEUTIC:
        return subgraph_name
    
    # Clean the subgraph name
    clean_name = subgraph_name.replace('_subgraph', '').replace('_', ' ').strip()
    
    # Add diagnostic and therapeutic focus terms
    enhanced_query = f"{clean_name} diagnosis diagnostic criteria therapeutic treatment therapy clinical implications"
    
    return enhanced_query

def calculate_weighted_cosine_similarity(query_embedding: List[float], title_embedding: List[float], abstract_embedding: List[float]) -> float:
    """Calculate weighted cosine similarity between query and paper (80% title, 20% abstract)."""
    try:
        # Convert to numpy arrays for easier computation
        query_vec = np.array(query_embedding, dtype=float)
        title_vec = np.array(title_embedding, dtype=float)
        abstract_vec = np.array(abstract_embedding, dtype=float)
        
        # Calculate cosine similarity for title we can use here sklearn and just use similarity function instead of the full formula (dot product/magnitude)
        title_similarity = np.dot(query_vec, title_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(title_vec))
        
        # Calculate cosine similarity for abstract
        abstract_similarity = np.dot(query_vec, abstract_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(abstract_vec))
        
        # Calculate weighted similarity (80% title, 20% abstract)
        weighted_similarity = (TITLE_WEIGHT * title_similarity) + (ABSTRACT_WEIGHT * abstract_similarity)
        
        return float(weighted_similarity)
        
    except Exception as e:
        st.error(f"Error calculating weighted similarity: {e}")
        return 0.0

def retrieve_similar_papers_dynamic(subgraph_name: str, pubmed_collection, top_k: int = TOP_K_PAPERS, threshold: float = SIMILARITY_THRESHOLD) -> List[Dict[str, Any]]:
    """Retrieve similar papers from PubMed database using weighted cosine similarity (80% title, 20% abstract)."""
    if not pubmed_collection:
        st.warning("PubMed collection not available")
        return []
    
    try:
        # Get embedding model
        embedding_model = get_embedding_model()
        
        # Enhance query for diagnostic/therapeutic focus
        enhanced_query = enhance_query_for_diagnostic_therapeutic(subgraph_name)
        
        # Generate embedding for enhanced query
        query_embedding = embedding_model.encode(enhanced_query, convert_to_tensor=False).tolist()
        
        # Get a larger pool of candidates for re-ranking
        pool_size = min(top_k * 3, 100)  # Get 3x more candidates for better selection
        
        # Query the PubMed collection
        results = pubmed_collection.query(
            query_embeddings=[query_embedding],
            n_results=pool_size,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results or 'ids' not in results or not results['ids']:
            st.warning(f"No results found for {subgraph_name}")
            return []
        
        # Generate embeddings for titles and abstracts
        titles = [results['metadatas'][0][i].get('title', '') for i in range(len(results['ids'][0]))]
        abstracts = results['documents'][0]
        
        # Generate embeddings for titles and abstracts
        title_embeddings = embedding_model.encode(titles, convert_to_tensor=False)
        abstract_embeddings = embedding_model.encode(abstracts, convert_to_tensor=False)
        
        # Calculate weighted similarities and filter by threshold
        papers = []
        for i in range(len(results['ids'][0])):
            try:
                # Calculate individual similarities first
                query_vec = np.array(query_embedding, dtype=float)
                title_vec = np.array(title_embeddings[i], dtype=float)
                abstract_vec = np.array(abstract_embeddings[i], dtype=float)
                
                # Calculate weighted similarity (80% title, 20% abstract)
                title_similarity = float(np.dot(query_vec, title_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(title_vec)))
                abstract_similarity = float(np.dot(query_vec, abstract_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(abstract_vec)))
                weighted_similarity = (TITLE_WEIGHT * title_similarity) + (ABSTRACT_WEIGHT * abstract_similarity)
                
                # Filter by threshold
                if weighted_similarity >= threshold:
                    paper = {
                        'pmid': results['ids'][0][i],
                        'title': titles[i],
                        'abstract': abstracts[i],
                        'similarity': weighted_similarity
                    }
                    papers.append(paper)
                    
            except Exception as e:
                st.warning(f"Error processing paper {i}: {e}")
                continue
        
        # Sort by weighted similarity and return top_k
        papers.sort(key=lambda x: x['similarity'], reverse=True)
        
        return papers[:top_k]
        
    except Exception as e:
        st.error(f"Error retrieving papers for {subgraph_name}: {e}")
        return []

def format_subgraph_context_for_analysis(df: pd.DataFrame) -> str:
    """Format subgraph data for the analysis prompt."""
    if df.empty:
        return "No subgraph data available."
    
    # Create a structured representation of the subgraph
    context_lines = []
    
    # Add nodes information
    nodes = set()
    if 'x_name' in df.columns:
        nodes.update(df['x_name'].dropna().unique())
    if 'y_name' in df.columns:
        nodes.update(df['y_name'].dropna().unique())
    
    if nodes:
        context_lines.append(f"**Nodes in subgraph:** {', '.join(sorted(nodes))}")
        context_lines.append("")
    
    # Add relationships
    if all(col in df.columns for col in ['x_name', 'display_relation', 'y_name']):
        context_lines.append("**Relationships:**")
        for _, row in df[['x_name', 'display_relation', 'y_name']].dropna().iterrows():
            context_lines.append(f"- {row['x_name']} --{row['display_relation']}--> {row['y_name']}")
    
    return "\n".join(context_lines)

def format_research_papers_for_analysis(papers: List[Dict[str, Any]]) -> str:
    """Format research papers for the analysis prompt."""
    if not papers:
        return "No relevant research papers found."
    
    papers_text = []
    for i, paper in enumerate(papers, 1):
        abstract_preview = paper['abstract'][:500]
        if len(paper['abstract']) > 500:
            abstract_preview += '...'
        
        papers_text.append(f"""
**Paper {i} (PMID: {paper['pmid']})**
Title: {paper['title']}
Similarity: {paper['similarity']:.4f}
Abstract: {abstract_preview}
""")
    
    return "\n".join(papers_text)

def generate_dynamic_analysis(subgraph_name: str, df: pd.DataFrame, papers: List[Dict[str, Any]]) -> str:
    """Generate analysis using Gemini with enhanced retrieval."""
    try:
        # System prompt with diagnostic/therapeutic focus and clinical decision support
        system_prompt = (
            """You are a Principal Knowledge Graph Analyst producing clinician-friendly factual reports. """
            "You strictly follow the provided structured data and retrieved research papers. "
            "You do not use any external sources or prior knowledge beyond what is provided.\n\n"
            "**Core Directives:**\n"
            "1. **Strict Data Adherence:** Use ONLY the information in the `<SUBGRAPH_CONTEXT>` and `<RESEARCH_PAPERS>` blocks.\n"
            "2. **No Hallucination:** Do not invent, infer, or embellish relationships.\n"
            "3. **Traceability:** Every fact must be directly traceable to the provided data or research papers.\n"
            "4. **Refusal on Missing Data:** If a relationship is not in the data, omit it.\n"
            "5. **Complete Response:** You MUST always provide all sections below.\n"
            "6. **Citation Requirements:** Always cite specific PMIDs when referencing research findings.\n"
            "7. **Disorder Relationships:** Pay special attention to relationships between Major Depressive Disorder (MDD), Bipolar Disorder, and Schizophrenia when present in the data.\n\n"
            "**MANDATORY OUTPUT FORMAT - INCLUDE ALL SECTIONS:**\n\n"
            "### Evidence\n"
            "List all verifiable relationships from the subgraph data, each on its own bullet point.\n\n"
            "### Analysis\n"
            "Write a comprehensive clinical analysis of the subgraph relationships, their patterns, "
            "and significance for clinicians based solely on the provided data and research papers. "
            "Focus on potential diagnostic implications and therapeutic considerations. "
            "Include specific citations to research papers using PMID format.\n\n"
            "### Clinical Decision Support\n"
            "Provide actionable insights including:\n"
            "- **Diagnostic Considerations:** Key diagnostic criteria and differential diagnosis points based on the subgraph relationships\n"
            "- **Treatment Selection Framework:** Evidence-based treatment approaches, including normality and gold standard treatments most commonly used\n"
            "- **Monitoring Recommendations:** Key parameters to monitor during treatment and follow-up\n"
            "- **Disorder Relationships:** When MDD, Bipolar Disorder, or Schizophrenia are present, explain their interconnections, shared pathways, and clinical implications for diagnosis and treatment\n\n"
            "### References\n"
            "List all referenced research papers with their PMIDs and brief descriptions of their relevance.\n\n"
            "**CRITICAL: Your response is incomplete if it does not contain all four sections above.**"
        )
        
        # Format the data
        subgraph_context = format_subgraph_context_for_analysis(df)
        research_papers = format_research_papers_for_analysis(papers)
        
        # Build the complete prompt
        user_prompt = f"""**TASK:** Generate a complete clinician-friendly report based on the provided 
knowledge graph subgraph data and relevant research papers.

**SUBGRAPH CONTEXT:**
{subgraph_context}

**RESEARCH PAPERS:**
{research_papers}

**MANDATORY REQUIREMENT:** Your response must contain exactly four sections:
1. ### Evidence (with bullet points of all relationships from subgraph)
2. ### Analysis (with clinical interpretation using both subgraph and research data)
3. ### Clinical Decision Support (with actionable insights and disorder relationships)
4. ### References (with all cited papers and their PMIDs)

**OUTPUT TEMPLATE - Fill in each section completely:**

### Evidence
[Extract and list every relationship from the subgraph data above as bullet points]

### Analysis
[Provide a comprehensive clinical analysis of the relationships, their patterns, and significance for clinicians. 
Focus specifically on potential diagnostic implications and therapeutic considerations based on the retrieved research papers. 
Use specific citations to research papers when making claims. Base analysis solely on the provided subgraph relationships and research papers.]

### Clinical Decision Support
[Provide actionable insights including:
- Diagnostic Considerations: Key diagnostic criteria and differential diagnosis points
- Treatment Selection Framework: Evidence-based treatment approaches, including normality and gold standard treatments most commonly used
- Monitoring Recommendations: Key parameters to monitor during treatment and follow-up
- Disorder Relationships: When MDD, Bipolar Disorder, or Schizophrenia are present, explain their interconnections, shared pathways, and clinical implications]

### References
[List all referenced research papers with their PMIDs and brief descriptions of their relevance to the analysis]

**REMINDER: Complete all four sections above. Use only the data provided in the context and research papers.**
"""
        
        # Generate analysis with OpenAI
        if not openai_client:
            return "OpenAI client not initialized. Please check your API key."
            
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=4000,
            temperature=0.1
        )
        
        if response and response.choices and response.choices[0].message.content:
            return response.choices[0].message.content
        else:
            return "The model returned an empty response."
            
    except Exception as e:
        return f"Error generating analysis: {str(e)}"

def generate_chat_response(
    context: str,
    question: str,
    chat_history: list,
    pubmed_collection: chromadb.Collection,
    current_subgraph: str = None
):
    """Generates a response using both subgraph and PubMed evidence.
    
    Args:
        context: Formatted subgraph context
        question: User's question
        chat_history: List of previous messages
        pubmed_collection: ChromaDB collection for PubMed
        current_subgraph: Name of the current subgraph being viewed
    """
    # Format the chat history
    formatted_history = "\n".join(
        f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
        for msg in chat_history
    )
    
    # Enhance the PubMed query with subgraph context
    pubmed_query = question
    if current_subgraph:
        # Add the subgraph name to focus the search
        pubmed_query = f"{question} related to {current_subgraph}"
    
    pubmed_evidence = retrieve_from_pubmed(pubmed_query, pubmed_collection, k=5, save_csv=False)

    # Combine system instructions with the user prompt
    prompt = (
        "You are a knowledgeable and articulate biomedical research assistant. Provide clear, conversational responses "
        "that synthesize information from the provided knowledge graph and PubMed evidence. "
        "Maintain a professional yet approachable tone, as if explaining to a colleague. "
        "Structure your response as follows:\n\n"
        "1. Provide a concise, direct answer to the question first\n"
        "2. Elaborate with relevant details and context\n"
        "3. At the end, include a 'References' section with all sources in the format:\n"
        "   - [PMID:12345678] Brief description of what this reference supports\n\n"
        "<KNOWLEDGE_GRAPH_CONTEXT>\n{context}\n</KNOWLEDGE_GRAPH_CONTEXT>\n\n"
        "<PUBMED_EVIDENCE>\n{pubmed_evidence}\n</PUBMED_EVIDENCE>\n\n"
        "<CONVERSATION_HISTORY>\n{formatted_history}\n</CONVERSATION_HISTORY>\n\n"
        "Question: {question}\n\n"
        "Guidelines:\n"
        "- Write in complete, flowing sentences\n"
        "- Use transition words to connect ideas naturally\n"
        "- Avoid robotic phrasing like 'based on the evidence'\n"
        "- Group related information together logically\n"
        "- Keep technical terms but explain them in context\n"
        "- End with a brief summary or next steps when appropriate"
    ).format(
        context=context,
        pubmed_evidence=pubmed_evidence,
        formatted_history=formatted_history,
        question=question
    )

    try:
        if not openai_client:
            return "OpenAI client not initialized. Please check your API key."
            
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.1
        )
        
        if response and response.choices and response.choices[0].message.content:
            return convert_pmids_to_links(response.choices[0].message.content)
        else:
            return "The model returned an empty response."
    except Exception as e:
        return f"Error communicating with OpenAI: {e}"

@st.cache_data
def load_precomputed_paths():
    """Load and process the pre-computed paths from CSV."""
    try:
        # Look for the file in the parent directory
        import os
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(parent_dir, 'kg_shortest_paths.csv')
        
        paths_df = pd.read_csv(csv_path)
        # Convert path strings to lists for easier manipulation
        paths_df['Path_Nodes'] = paths_df['Path'].str.split(' → ')
        return paths_df
    except Exception as e:
        st.error(f"Error loading pre-computed paths: {e}")
        return pd.DataFrame()

# Import the new downloader module
from path_paper_downloader import PathPaperDownloader

def download_pubmed_papers_for_path(path_nodes: List[str], papers_per_node: int, max_total_papers: int, 
                                   include_intermediate: bool, include_pmc_metadata: bool, pubmed_collection) -> Dict[str, Any]:
    """
    Download PubMed papers for nodes in a shortest path using the new downloader module.
    
    Args:
        path_nodes: List of node names in the path
        papers_per_node: Number of papers to retrieve per node
        max_total_papers: Maximum total papers to download
        include_intermediate: Whether to include intermediate nodes
        pubmed_collection: ChromaDB collection with PubMed data (not used in new implementation)
        
    Returns:
        Dictionary with download results and summary
    """
    try:
        # Initialize the new downloader
        downloader = PathPaperDownloader()
        
        # Download papers with PMC metadata and MeSH terms, focusing on associations
        results = downloader.download_papers_for_path(
            path_nodes=path_nodes,
            papers_per_node=papers_per_node,
            max_total_papers=max_total_papers,
            include_intermediate=include_intermediate,
            include_pmc_metadata=include_pmc_metadata,  # Use the parameter from UI
            focus_on_associations=True  # Focus on associations between source and target
        )
        
        if results['success']:
            # Embed papers into ChromaDB
            embed_results = downloader.embed_papers_to_chromadb(results['papers'])
            
            if embed_results['success']:
                results['embedding_success'] = True
                results['embedded_count'] = embed_results['embedded_count']
                results['collection_name'] = embed_results['collection_name']
            else:
                results['embedding_success'] = False
                results['embedding_error'] = embed_results['error']
            
            # Get sample papers for display (top 5 by publication date)
            sample_papers = sorted(results['papers'], key=lambda x: x.get('publication_date', ''), reverse=True)[:5]
            results['sample_papers'] = sample_papers
            
        return results
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'total_papers': 0,
            'summary': [],
            'sample_papers': []
        }

def save_path_papers_results(download_result: Dict[str, Any], source_node: str, target_node: str):
    """Save the downloaded papers results to a file using the new downloader."""
    try:
        # Initialize downloader to use its save method
        downloader = PathPaperDownloader()
        filepath = downloader.save_results(download_result, source_node, target_node)
        st.success(f"Results saved to: {filepath}")
        
    except Exception as e:
        st.error(f"Error saving results: {str(e)}")

# --- Streamlit App Layout ---
st.set_page_config(layout="wide")
st.title("🧠 Knowledge Graph AI Analyst")
st.markdown("Select a topic to explore its knowledge subgraph and chat with an AI powered by PubMed and your graph.")

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_subgraph_df" not in st.session_state:
    st.session_state.current_subgraph_df = pd.DataFrame()
if "selected_file" not in st.session_state:
    st.session_state.selected_file = None

# --- Load API Key ---
# OpenAI API key is already configured above

# --- Load Question-to-Node Mappings ---
try:
    if not QUESTION_NODE_MAPPING_PATH.exists():
        st.warning(f"Question-to-node mappings file not found at: {QUESTION_NODE_MAPPING_PATH}")
    else:
        mappings_df = pd.read_csv(QUESTION_NODE_MAPPING_PATH)
        st.subheader("🔍 Filter Question-to-Node Mappings by Similarity")
        
        # Debug: Show available columns
        st.info(f"Available columns in CSV: {', '.join(mappings_df.columns)}")
        
        if "similarity_score" not in mappings_df.columns:
            st.error(f"Error: 'similarity_score' column not found in {QUESTION_NODE_MAPPING_PATH}")
            st.error(f"Available columns are: {', '.join(mappings_df.columns)}")
        else:
            min_similarity = st.slider(
                "Minimum Similarity Score", min_value=0.0, max_value=1.0, value=0.3, step=0.01
            )
            filtered_df = mappings_df[mappings_df["similarity_score"] >= min_similarity]
            st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True, height=500)
except Exception as e:
    st.error(f"Error loading question-to-node mappings: {str(e)}")
    import traceback
    st.error(f"Error details: {traceback.format_exc()}")

# --- Load Subgraphs and Databases ---
analysis_collection = get_analysis_collection()
subgraph_files = get_subgraph_files()
node_type_map, _ = load_node_types()
pubmed_collection = get_pubmed_collection()

if not subgraph_files:
    st.warning(f"No subgraph files found in '{SUBGRAPHS_DIR}'.")
else:
    selected_file = st.selectbox(
        "📂 Select a Subgraph to Analyze:",
        options=subgraph_files,
        key="subgraph_selector",
    )

    if selected_file and selected_file != st.session_state.selected_file:
        st.session_state.selected_file = selected_file
        st.session_state.messages = []
        # Reset analysis state when new subgraph is selected
        st.session_state.generate_analysis = False
        st.session_state.current_analysis = None
        st.session_state.current_papers = None
        try:
            file_path = SUBGRAPHS_DIR / selected_file
            df_raw = pd.read_csv(file_path)

            # 🔍 Validate required columns
            required_cols = ['x_name', 'x_type', 'y_name', 'y_type', 'display_relation']
            missing_cols = [col for col in required_cols if col not in df_raw.columns]
            if missing_cols:
                st.error(f"❌ Your file '{selected_file}' is missing required columns: {missing_cols}")
                st.stop()

            # ✅ Only keep what we need
            st.session_state.current_subgraph_df = df_raw[required_cols].dropna()
        except Exception as e:
            st.error(f"Could not read or process subgraph file: {e}")
            st.session_state.current_subgraph_df = pd.DataFrame()
        st.rerun()

    if st.session_state.selected_file:
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "AI Expert Analysis", "Subgraph Summary", "Chat with AI Analyst", "Interactive Graph", "Raw Data",
            "Path Finder"
        ])

        with tab1:
            st.subheader("🔬 LLM Summary")
            
            # Check if we have subgraph data
            if st.session_state.current_subgraph_df.empty:
                st.warning("No subgraph data available for analysis.")
            else:
                # Get subgraph name for analysis
                subgraph_name = st.session_state.selected_file.replace('_subgraph.csv', '')
                
                # Add a button to generate/regenerate analysis
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🚀 Generate Dynamic Analysis", type="primary", use_container_width=True):
                        st.session_state.generate_analysis = True
                
                # Check if we should generate analysis
                if st.session_state.get('generate_analysis', False):
                    with st.spinner("🔍 Retrieving relevant research papers..."):
                        # Retrieve similar papers using enhanced retrieval
                        papers = retrieve_similar_papers_dynamic(
                            subgraph_name, 
                            pubmed_collection, 
                            top_k=TOP_K_PAPERS, 
                            threshold=SIMILARITY_THRESHOLD
                        )
                        
                        if papers:
                            st.success(f"✅ Retrieved {len(papers)} relevant papers")
                            
                            # Show paper details in an expander
                            with st.expander(f"📚 Retrieved Papers ({len(papers)} papers)", expanded=False):
                                for i, paper in enumerate(papers[:10], 1):  # Show first 10
                                    st.markdown(f"""
                                    **{i}. {paper['title']}**
                                    - PMID: {create_pubmed_link(paper['pmid'])}
                                    - Similarity: {paper['similarity']:.4f}
                                    - Abstract: {paper['abstract'][:200]}...
                                    """, unsafe_allow_html=True)
                                
                                if len(papers) > 10:
                                    st.info(f"... and {len(papers) - 10} more papers")
                        else:
                            st.warning("⚠️ No relevant papers found above the similarity threshold")
                    
                    with st.spinner("🤖 Generating AI analysis with OpenAI..."):
                        # Generate dynamic analysis
                        analysis_text = generate_dynamic_analysis(
                            subgraph_name, 
                            st.session_state.current_subgraph_df, 
                            papers
                        )
                        
                        if analysis_text and not analysis_text.startswith("Error"):
                            st.session_state.current_analysis = analysis_text
                            st.session_state.current_papers = papers
                            #st.success("✅ Analysis generated successfully!")
                        else:
                            st.error(f"❌ Failed to generate analysis: {analysis_text}")
                
                # Display the analysis if available
                if st.session_state.get('current_analysis'):
                    st.markdown("### Analysis Results")
                    st.markdown("---")
                    
                    # Parse the analysis to extract sections
                    analysis_text = st.session_state.current_analysis
                    
                    # Extract Evidence section
                    evidence_match = re.search(r'### Evidence\s*\n(.*?)(?=### |$)', analysis_text, re.DOTALL | re.IGNORECASE)
                    evidence_section = evidence_match.group(1).strip() if evidence_match else ""
                    
                    # Extract Analysis section
                    analysis_match = re.search(r'### Analysis\s*\n(.*?)(?=### |$)', analysis_text, re.DOTALL | re.IGNORECASE)
                    analysis_section = analysis_match.group(1).strip() if analysis_match else ""
                    
                    # Extract Clinical Decision Support section
                    clinical_support_match = re.search(r'### Clinical Decision Support\s*\n(.*?)(?=### |$)', analysis_text, re.DOTALL | re.IGNORECASE)
                    clinical_support_section = clinical_support_match.group(1).strip() if clinical_support_match else ""
                    
                    # Extract References section
                    references_match = re.search(r'### References\s*\n(.*?)(?=### |$)', analysis_text, re.DOTALL | re.IGNORECASE)
                    references_section = references_match.group(1).strip() if references_match else ""
                    
                    # Display References in expander (moved to top)
                    if st.session_state.get('current_papers'):
                        with st.expander("📚 References", expanded=False):
                            for i, paper in enumerate(st.session_state.current_papers, 1):
                                st.markdown(f"""
                                **{i}. {paper['title']}**
                                - PMID: {create_pubmed_link(paper['pmid'])}
                                - Similarity: {paper['similarity']:.4f}
                                - Abstract: {paper['abstract'][:300]}{'...' if len(paper['abstract']) > 300 else ''}
                                """, unsafe_allow_html=True)
                                st.markdown("---")

                    # Display Evidence in expander
                    if evidence_section:
                        with st.expander("📋 Evidence", expanded=True):
                            st.markdown(convert_pmids_to_links(evidence_section), unsafe_allow_html=True)

                    # Display Analysis in expander
                    if analysis_section:
                        with st.expander("🔬 Analysis", expanded=True):
                            st.markdown(convert_pmids_to_links(analysis_section), unsafe_allow_html=True)

                    # Display Clinical Decision Support in expander
                    if clinical_support_section:
                        with st.expander("🏥 Clinical Decision Support", expanded=True):
                            st.markdown(convert_pmids_to_links(clinical_support_section), unsafe_allow_html=True)
                else:
                    st.info("👆 Click 'Generate Dynamic Analysis' to create an AI-powered analysis based on the selected subgraph and relevant research papers.")

        with tab2:
            if not st.session_state.current_subgraph_df.empty:
                generate_subgraph_summary_with_evidence(
                    st.session_state.current_subgraph_df, 
                    st.session_state.selected_file
                )
            else:
                st.warning("No subgraph data available for summary.")

        with tab3:
            current_subgraph = st.session_state.selected_file.replace('_subgraph.csv', '')
            st.subheader(f"💬 Chat about `{current_subgraph}`")
            st.info(f"ℹ️ Chat context is limited to the `{current_subgraph}` subgraph. "
                   "For questions about other conditions, please select the appropriate subgraph first.")
            
            # Initialize chat history if not exists
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            # Display chat messages
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"], unsafe_allow_html=True)
            
            # Chat input
            if prompt := st.chat_input(f"Ask about {current_subgraph}..."):
                # Add user message to chat history
                user_message = {"role": "user", "content": prompt}
                st.session_state.chat_history.append(user_message)
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt, unsafe_allow_html=True)
                
                # Generate and display assistant response
                with st.chat_message("assistant"):
                    with st.spinner("🧠 Analyzing..."):
                        try:
                            # Get context from the current subgraph only
                            context = format_subgraph_for_chat_prompt(st.session_state.current_subgraph_df)
                            
                            # Generate response with clear scope
                            response = generate_chat_response(
                                context=context,
                                question=prompt,
                                chat_history=st.session_state.chat_history[:-1],  # Exclude current message
                                pubmed_collection=pubmed_collection,
                                current_subgraph=current_subgraph  # Pass current subgraph name
                            )
                            
                            # Ensure response stays on topic
                            if current_subgraph.lower() not in response.lower():
                                response = f"Based on the {current_subgraph} subgraph: " + response
                            
                            # Display and store the response
                            st.markdown(response, unsafe_allow_html=True)
                            st.session_state.chat_history.append({
                                "role": "assistant", 
                                "content": response
                            })
                            
                        except Exception as e:
                            error_msg = f"I encountered an error while analyzing the {current_subgraph} subgraph. Please try rephrasing your question."
                            st.error(error_msg)
                            st.session_state.chat_history.append({
                                "role": "assistant", 
                                "content": error_msg
                            })

        with tab4:
            st.subheader(f"📊 Interactive Graph: {st.session_state.selected_file}")
            df = st.session_state.current_subgraph_df
            if not df.empty:
                net = Network(height="600px", width="100%", bgcolor="#F0F0F0", font_color="black", directed=True)
                for _, row in df.head(MAX_NODES).iterrows():
                    src, src_type = row["x_name"], row["x_type"]
                    dst, dst_type = row["y_name"], row["y_type"]
                    rel = row["display_relation"]
                    src_color = NODE_COLORS.get(src_type, NODE_COLORS["default"])
                    dst_color = NODE_COLORS.get(dst_type, NODE_COLORS["default"])
                    net.add_node(src, label=truncate_label(src, LABEL_MAX_LENGTH), title=f"{src}\nType: {src_type}", color=src_color)
                    net.add_node(dst, label=truncate_label(dst, LABEL_MAX_LENGTH), title=f"{dst}\nType: {dst_type}", color=dst_color)
                    net.add_edge(src, dst, label=rel, title=rel)
                net.save_graph("graph.html")
                with open("graph.html", "r", encoding="utf-8") as f:
                    components.html(f.read(), height=610, scrolling=True)
            else:
                st.warning("No graph data to display.")

        with tab5:
            st.header("📊 Raw Data")
            if not st.session_state.current_subgraph_df.empty:
                st.dataframe(st.session_state.current_subgraph_df)
                
                # Add download button for the subgraph
                csv = st.session_state.current_subgraph_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Subgraph as CSV",
                    data=csv,
                    file_name=st.session_state.selected_file,
                    mime="text/csv"
                )
            else:
                st.warning("No raw subgraph data to display.")

        with tab6:
            st.header("🔍 Find Path Between Nodes")
            
            # Load pre-computed paths
            paths_df = load_precomputed_paths()
            
            if paths_df.empty:
                st.error("Could not load pre-computed paths. Please make sure 'kg_shortest_paths.csv' is in the correct location.")
            else:
                # Get unique source and target nodes
                source_nodes = sorted(paths_df['Source_Node'].unique().tolist())
                
                # Create two columns for the node selectors
                col1, col2 = st.columns(2)
                
                with col1:
                    with st.expander("🔘 Select Source Node", expanded=True):
                        source_node = st.selectbox(
                            "Source Node",
                            source_nodes,
                            key="source_node_selector",
                            index=0
                        )
                
                with col2:
                    with st.expander("🎯 Select Target Node", expanded=True):
                        # Get available targets for the selected source
                        available_targets = paths_df[paths_df['Source_Node'] == source_node]['Target_Node'].unique()
                        target_node = st.selectbox(
                            "Target Node",
                            available_targets,
                            key="target_node_selector",
                            index=0 if len(available_targets) > 0 else None
                        )
                
                # Display the path information if both nodes are selected
                if source_node and target_node and len(available_targets) > 0:
                    # Get the path information
                    path_info = paths_df[
                        (paths_df['Source_Node'] == source_node) & 
                        (paths_df['Target_Node'] == target_node)
                    ].iloc[0]
                    
                    # Display status and path length
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Status", path_info['Status'])
                    with col2:
                        st.metric("Path Length", f"{path_info['Path_Length']} steps")
                    
                    # Display the path
                    st.subheader("Path")
                    st.code(path_info['Path'], language="plaintext")
                    
                    # Display detailed path if available
                    if 'Detailed_Path' in path_info and pd.notna(path_info['Detailed_Path']):
                        st.subheader("Detailed Path")
                        # Split the detailed path into individual steps and clean up formatting
                        detailed_steps = [step.strip() for step in path_info['Detailed_Path'].split('•') if step.strip()]
                        for step in detailed_steps:
                            # Create columns for better alignment of the arrow
                            if '-->' in step:
                                source, target = step.split('-->', 1)
                                col1, col2 = st.columns([1, 10])
                                with col1:
                                    st.markdown("•")
                                with col2:
                                    st.markdown(f"**{source.strip()}** → {target.strip()}")
                            else:
                                st.markdown(f"• {step}")
                    
                    # Add export button
                    if st.button("💾 Export Path to CSV"):
                        try:
                            output_file = export_detailed_path(path_info)
                            st.success(f'Successfully saved path to {output_file}')
                        except Exception as e:
                            st.error(f'Error exporting path: {str(e)}')
                    
                    # New PubMed Download Task
                    st.subheader("📚 Download PubMed Papers for Path")
                    st.info("Download research papers that discuss the **associations and relationships** between the source and target nodes in this path to provide context for analysis.")
                    
                    # Get path nodes
                    path_nodes = path_info['Path'].split(' → ')
                    
                    # Display path nodes
                    st.write("**Path nodes to search for papers:**")
                    for i, node in enumerate(path_nodes):
                        if i == 0:
                            st.write(f"🔘 **Source:** {node}")
                        elif i == len(path_nodes) - 1:
                            st.write(f"🎯 **Target:** {node}")
                        else:
                            st.write(f"🔗 **Intermediate:** {node}")
                    
                    # Configuration options
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        papers_per_node = st.number_input("Papers per node", min_value=5, max_value=50, value=10, help="Number of papers to download for each node")
                    with col2:
                        max_total_papers = st.number_input("Max total papers", min_value=20, max_value=200, value=50, help="Maximum total papers to download")
                    with col3:
                        include_intermediate = st.checkbox("Include intermediate nodes", value=True, help="Download papers for intermediate nodes in the path")
                    with col4:
                        include_pmc_metadata = st.checkbox("Include PMC metadata", value=True, help="Fetch PMC IDs and URLs for full text access")
                    
                    # Download button
                    if st.button("📥 Download PubMed Papers", type="primary"):
                        with st.spinner("Downloading PubMed papers for path nodes..."):
                            try:
                                download_result = download_pubmed_papers_for_path(
                                    path_nodes=path_nodes,
                                    papers_per_node=papers_per_node,
                                    max_total_papers=max_total_papers,
                                    include_intermediate=include_intermediate,
                                    include_pmc_metadata=include_pmc_metadata,
                                    pubmed_collection=pubmed_collection
                                )
                                
                                if download_result['success']:
                                    st.success(f"✅ Successfully downloaded {download_result['total_papers']} papers!")
                                    
                                    # Show embedding status
                                    if download_result.get('embedding_success'):
                                        st.success(f"📚 Embedded {download_result['embedded_count']} papers into ChromaDB collection: `{download_result['collection_name']}`")
                                    else:
                                        st.warning(f"⚠️ Embedding failed: {download_result.get('embedding_error', 'Unknown error')}")
                                    
                                    # Display summary
                                    st.subheader("📊 Download Summary")
                                    summary_df = pd.DataFrame(download_result['summary'])
                                    st.dataframe(summary_df, use_container_width=True)
                                    
                                    # Display all papers with enhanced information
                                    if download_result.get('papers'):
                                        st.subheader(f"📄 All Papers ({len(download_result['papers'])} total)")
                                        
                                        # Sort papers by publication date (newest first)
                                        sorted_papers = sorted(download_result['papers'], 
                                                            key=lambda x: x.get('publication_date', ''), 
                                                            reverse=True)
                                        
                                        for i, paper in enumerate(sorted_papers):
                                            # Create expander title with paper number and key info
                                            title_preview = paper['title'][:60] + "..." if len(paper['title']) > 60 else paper['title']
                                            expander_title = f"📖 {i+1}. {title_preview}"
                                            
                                            with st.expander(expander_title):
                                                col1, col2 = st.columns(2)
                                                
                                                with col1:
                                                    st.write(f"**PMID:** {paper['pmid']}")
                                                    st.write(f"**Journal:** {paper.get('journal', 'N/A')}")
                                                    st.write(f"**Publication Date:** {paper.get('publication_date', 'N/A')}")
                                                    if paper.get('pmc_id'):
                                                        st.write(f"**PMC ID:** {paper['pmc_id']}")
                                                
                                                with col2:
                                                    st.write(f"**Source:** {paper.get('source_node', 'N/A')}")
                                                    if paper.get('search_term'):
                                                        st.write(f"**Search Term:** {paper.get('search_term', 'N/A')}")
                                                    if paper.get('authors'):
                                                        authors_display = ', '.join(paper['authors'][:3])
                                                        if len(paper['authors']) > 3:
                                                            authors_display += f" (+{len(paper['authors'])-3} more)"
                                                        st.write(f"**Authors:** {authors_display}")
                                                
                                                # MeSH terms
                                                if paper.get('all_mesh_terms'):
                                                    mesh_display = ', '.join(paper['all_mesh_terms'][:8])
                                                    if len(paper['all_mesh_terms']) > 8:
                                                        mesh_display += f" (+{len(paper['all_mesh_terms'])-8} more)"
                                                    st.write(f"**MeSH Terms:** {mesh_display}")
                                                
                                                # Abstract
                                                abstract_text = paper.get('abstract', 'No abstract available')
                                                if len(abstract_text) > 500:
                                                    st.write(f"**Abstract:** {abstract_text[:500]}...")
                                                    with st.expander("View Full Abstract"):
                                                        st.write(abstract_text)
                                                else:
                                                    st.write(f"**Abstract:** {abstract_text}")
                                                
                                                # Links
                                                col1, col2 = st.columns(2)
                                                with col1:
                                                    st.write(f"**PubMed:** [View Paper](https://pubmed.ncbi.nlm.nih.gov/{paper['pmid']}/)")
                                                with col2:
                                                    if paper.get('pmc_url'):
                                                        st.write(f"**PMC:** [View Full Text]({paper['pmc_url']})")
                                        
                                        # Add a summary at the bottom
                                        st.info(f"📊 **Summary:** Found {len(download_result['papers'])} papers total. "
                                               f"Papers are sorted by publication date (newest first).")
                                    
                                    # Save results option
                                    if st.button("💾 Save Results to File"):
                                        save_path_papers_results(download_result, source_node, target_node)
                                        st.success("Results saved successfully!")
                                        
                                else:
                                    st.error(f"❌ Download failed: {download_result['error']}")
                                    
                            except Exception as e:
                                st.error(f"❌ Error during download: {str(e)}")
                    
                    # Add some space at the bottom
                    st.markdown("<br><br>", unsafe_allow_html=True)
