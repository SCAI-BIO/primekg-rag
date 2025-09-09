import streamlit as st
import pandas as pd
from pyvis.network import Network
import streamlit.components.v1 as components
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv
from datetime import datetime
import numpy as np
import google.generativeai as genai

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
BASE_DIR = Path(__file__).parent
SUBGRAPHS_DIR = BASE_DIR / "new_subgraphs"
NODES_CSV_PATH = BASE_DIR / "nodes.csv"
ANALYSIS_DB_PATH = BASE_DIR / "analyses_db"
ANALYSIS_COLLECTION_NAME = "medical_analyses"
GEMINI_MODEL_NAME = "gemini-1.5-pro"
QUESTION_NODE_MAPPING_PATH = BASE_DIR / "best_question_matches.csv"

# Initialize Gemini
gemini_api_key = os.getenv("GOOGLE_API_KEY")
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
else:
    st.warning("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")

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

        text = markdown_text.replace('\r\n', '\n').replace('\r', '\n')

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
        st.markdown(parts[0], unsafe_allow_html=True)
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
                st.markdown("<EXPANDER" + chunk, unsafe_allow_html=True)
                continue
            with st.expander(title, expanded=False):
                st.markdown(body)
            remainder = chunk[body_end + len('</EXPANDER>'):]
            if remainder.strip():
                st.markdown(remainder, unsafe_allow_html=True)
    except Exception:
        st.markdown(markdown_text, unsafe_allow_html=True)

# --- Helper Functions ---
@st.cache_data
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
        db_path = os.getenv("PUBMED_DB_PATH")
        if not db_path:
            st.error("PUBMED_DB_PATH not set in .env file.")
            return None
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
    """Convert PMID references into '(PMID:xxxx) (PubMed link)' format without nesting issues."""
    import re

    def format_pmid(pmid: str) -> str:
        return f"(PMID:{pmid}) (<a href=\"https://pubmed.ncbi.nlm.nih.gov/{pmid}/\" target=\"_blank\" rel=\"noopener noreferrer\">PubMed</a>)"

    # Handle groups like (PMID: 123, 456, 789)
    def replace_pmid_group(match):
        content = match.group(0)[1:-1]  # strip outer parentheses
        pmids = []
        for part in content.split(","):
            part = part.strip()
            if part.startswith("PMID:"):
                part = part[5:].strip()
            if part.isdigit():
                pmids.append(part)
        if not pmids:
            return match.group(0)
        # return each PMID separately, not nested
        return " ".join(format_pmid(pmid) for pmid in pmids)

    # Replace groups first
    text = re.sub(r"\(PMID:\s*\d+(?:\s*,\s*\d+)*\)", replace_pmid_group, text)

    # Replace single PMIDs outside parentheses
    text = re.sub(
        r"PMID:\s*(\d{1,8})(?![\d/])",
        lambda m: format_pmid(m.group(1)),
        text
    )

    return text

def generate_chat_response(
    context: str,
    question: str,
    chat_history: list,
    pubmed_collection: chromadb.Collection
):
    """Generates a response using both subgraph and PubMed evidence."""
    # Format the chat history
    formatted_history = "\n".join(
        f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
        for msg in chat_history
    )
    
    pubmed_evidence = retrieve_from_pubmed(question, pubmed_collection, k=3, save_csv=False)

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
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        response = model.generate_content(prompt)
        
        if response.text:
            return convert_pmids_to_links(response.text)
        else:
            return "The model returned an empty response."
    except Exception as e:
        return f"Error communicating with Gemini: {e}"

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
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    st.error("❌ Could not find GOOGLE_API_KEY in your .env file. Please add it.")
    st.stop()

try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"❌ Could not configure Gemini API: {e}")
    st.stop()

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
            st.subheader("🔬 AI Expert Analysis")
            if analysis_collection:
                try:
                    # First try to find by exact analysis filename derived from the CSV name
                    expected_analysis = st.session_state.selected_file.replace('_subgraph.csv', '_subgraph_analysis.txt')
                    result = analysis_collection.get(
                        where={"filename": expected_analysis}
                    )
                    
                    # If not found, try to find by partial filename match
                    if not result or 'documents' not in result or not result['documents']:
                        # Extract the main condition from the filename (e.g., 'Panic attack' from 'Panic attack_subgraph.csv')
                        condition = st.session_state.selected_file.replace('_subgraph.csv', '').lower()
                        
                        # Get all analyses to search through
                        all_analyses = analysis_collection.get()
                        
                        if all_analyses and 'metadatas' in all_analyses and all_analyses['metadatas']:
                            # Try to find a matching analysis by stored condition or by filename
                            for i, meta in enumerate(all_analyses['metadatas']):
                                if not meta:
                                    continue
                                meta_filename = (meta.get('filename') or '').lower()
                                meta_condition = (meta.get('condition') or '').lower()
                                if meta_condition == condition or condition in meta_filename:
                                    # Found a match
                                    st.success(f"Found matching analysis for '{condition}' in '{meta.get('filename','unknown')}'")
                                    result = {
                                        'documents': [all_analyses['documents'][i]] if 'documents' in all_analyses else ["No content available"],
                                        'metadatas': [meta]
                                    }
                                    break
                    
                    # Display the analysis if found
                    if result and 'documents' in result and result['documents']:
                        st.markdown("### Analysis Results")
                        st.markdown("---")
                        
                        # Render analysis with collapsible sections if long
                        render_analysis_with_collapse(
                            result['documents'][0],
                            bullet_threshold=10,
                            targets=["Key Clinical Relationships", "Therapeutic Insights"]
                        )
                    else:
                        st.warning(f"No analysis found for '{st.session_state.selected_file}'")
                        
                        # Show available analyses for reference
                        all_analyses = analysis_collection.get()
                        if all_analyses and 'metadatas' in all_analyses and all_analyses['metadatas']:
                            st.info("### Analyses")
                            st.write("The following analyses are available in the database:")
                            
                            # Create a simple list of available analyses
                            analyses_text = "\n\n".join([
                                f"- {meta.get('condition', 'Unknown')} ({meta.get('filename', 'No filename')})"
                                for meta in all_analyses['metadatas']
                                if meta and 'filename' in meta
                            ])
                            
                            st.text_area(
                                "Available analyses:",
                                value=analyses_text,
                                height=200,
                                disabled=True
                            )
                except Exception as e:
                    st.error(f"Error querying analysis database: {e}")
            else:
                st.error("Analysis database is unavailable. Could not connect to ChromaDB.")
                st.info("This could be because the database is empty or there was an error connecting to it.")

        with tab2:
            if not st.session_state.current_subgraph_df.empty:
                generate_subgraph_summary_with_evidence(
                    st.session_state.current_subgraph_df, 
                    st.session_state.selected_file
                )
            else:
                st.warning("No subgraph data available for summary.")

        with tab3:
            st.subheader(f"💬 Chat about `{st.session_state.selected_file}`")
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"], unsafe_allow_html=True)

            if prompt := st.chat_input("Ask a question about this subgraph..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt, unsafe_allow_html=True)
                with st.chat_message("assistant"):
                    with st.spinner("🧠 Thinking..."):
                        context = format_subgraph_for_chat_prompt(st.session_state.current_subgraph_df)
                        response = generate_chat_response(
                            context, prompt, st.session_state.messages[:-1], pubmed_collection
                        )
                        st.markdown(response, unsafe_allow_html=True)
                        st.session_state.messages.append({"role": "assistant", "content": response})

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
                    
                    # Add some space at the bottom
                    st.markdown("<br><br>", unsafe_allow_html=True)