import streamlit as st
import pandas as pd
from pyvis.network import Network
import streamlit.components.v1 as components
import os
import chromadb

# --- Configuration ---
MATCHES_FILE = "qa_to_node_matches_improved.csv"
SUBGRAPHS_DIR = "subgraphs"
ANALYSIS_DB_PATH = "analyses_db"
ANALYSIS_COLLECTION_NAME = "subgraph_analyses"
MAX_NODES = 15
LABEL_MAX_LENGTH = 15
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


# --- Helper Functions ---
@st.cache_data
def load_matches_file():
    """Loads the CSV file with the Q&A matches."""
    if not os.path.exists(MATCHES_FILE):
        st.error(f"Matches file not found: '{MATCHES_FILE}'")
        return pd.DataFrame()
    return pd.read_csv(MATCHES_FILE)


@st.cache_data
def get_subgraph_files():
    """Finds all available subgraph CSV files."""
    if not os.path.exists(SUBGRAPHS_DIR):
        return []
    return sorted([f for f in os.listdir(SUBGRAPHS_DIR) if f.endswith(".csv")])


@st.cache_resource
def get_analysis_collection():
    """Connects to the ChromaDB collection containing the AI analyses."""
    try:
        client = chromadb.PersistentClient(path=ANALYSIS_DB_PATH)
        return client.get_collection(name=ANALYSIS_COLLECTION_NAME)
    except Exception as e:
        st.error(f"Could not connect to the Analyses Database: {e}")
        return None


def truncate_label(text, max_length):
    """Shortens text for display in the graph."""
    if len(text) > max_length:
        return text[: max_length - 3] + "..."
    return text


# --- Streamlit App Layout ---
st.set_page_config(layout="wide")
st.title("Knowledge Graph & AI Analyst")

# --- Part 1: Display the Q&A Matches Table ---
st.subheader("1. Q&A Matched to Prime KG")
matches_df = load_matches_file()
if not matches_df.empty:
    st.dataframe(matches_df)
else:
    st.warning("Could not load the matches file.")

st.divider()

# --- Part 2: Visualize Subgraph and Display AI Analysis ---
st.subheader("2. Subgraph Explorer & AI Analysis")
subgraph_files = get_subgraph_files()
analysis_collection = get_analysis_collection()

if not subgraph_files:
    st.warning(f"No subgraph files found in the '{SUBGRAPHS_DIR}' folder.")
else:
    selected_file = st.selectbox(
        "Select a Subgraph to Analyze:", options=subgraph_files
    )

    if selected_file:
        # Display the Interactive Graph
        st.markdown(f"#### Interactive Graph for `{selected_file}`")
        file_path = os.path.join(SUBGRAPHS_DIR, selected_file)
        full_subgraph_df = pd.read_csv(file_path)

        nodes_in_graph = set()
        edges_for_graph = []
        for _, row in full_subgraph_df.iterrows():
            new_nodes_count = len(set([row["x_name"], row["y_name"]]) - nodes_in_graph)
            if len(nodes_in_graph) + new_nodes_count > MAX_NODES:
                continue
            nodes_in_graph.add(row["x_name"])
            nodes_in_graph.add(row["y_name"])
            edges_for_graph.append(row)

        trimmed_subgraph_df = pd.DataFrame(edges_for_graph)

        if not trimmed_subgraph_df.empty:
            net = Network(
                height="500px",
                width="100%",
                bgcolor="#F0F0F0",
                font_color="black",
                directed=True,
            )
            for _, row in trimmed_subgraph_df.iterrows():
                src, src_type = row["x_name"], row["x_type"]
                dst, dst_type = row["y_name"], row["y_type"]
                rel = row["display_relation"]

                src_color = NODE_COLORS.get(src_type, NODE_COLORS["default"])
                dst_color = NODE_COLORS.get(dst_type, NODE_COLORS["default"])
                src_label = truncate_label(src, LABEL_MAX_LENGTH)
                dst_label = truncate_label(dst, LABEL_MAX_LENGTH)
                src_title = f"{src}\nType: {src_type}"
                dst_title = f"{dst}\nType: {dst_type}"

                net.add_node(src, label=src_label, title=src_title, color=src_color)
                net.add_node(dst, label=dst_label, title=dst_title, color=dst_color)
                net.add_edge(src, dst, label=rel, title=rel)

            net.show_buttons(filter_=["physics"])
            try:
                net.save_graph("final_graph.html")
                with open("final_graph.html", "r", encoding="utf-8") as f:
                    components.html(f.read(), height=510, scrolling=True)
            except Exception as e:
                st.error(f"Could not generate visualization: {e}")
        else:
            st.warning("No relationships could be displayed within the node limit.")

        st.divider()

        # Display the Pre-Generated AI Analysis
        st.subheader("AI Expert Analysis")
        if analysis_collection:
            with st.spinner("Retrieving analysis..."):
                retrieved_analysis = analysis_collection.get(ids=[selected_file])

                if retrieved_analysis and retrieved_analysis["documents"]:
                    st.markdown(retrieved_analysis["documents"][0])
                else:
                    st.warning("No pre-generated analysis found for this subgraph.")
        else:
            st.error("Analysis database not available.")
