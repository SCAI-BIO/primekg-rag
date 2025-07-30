import streamlit as st
import pandas as pd
from pyvis.network import Network
import streamlit.components.v1 as components
from pathlib import Path
import chromadb
import ollama

# --- Configuration ---
BASE_DIR = Path(__file__).parent
SUBGRAPHS_DIR = BASE_DIR / "subgraphs"
NODES_CSV_PATH = BASE_DIR / "nodes.csv"
# **NEW**: Added back the path to the analysis database
ANALYSIS_DB_PATH = BASE_DIR / "analyses_db"
ANALYSIS_COLLECTION_NAME = "subgraph_analyses"
OLLAMA_MODEL_NAME = "deepseek-r1:14b"

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


# --- Helper Functions ---
@st.cache_data
def get_subgraph_files():
    """Finds all available subgraph CSV files."""
    if not SUBGRAPHS_DIR.is_dir():
        return []
    return sorted([f.name for f in SUBGRAPHS_DIR.glob("*.csv")])


def truncate_label(text, max_length):
    """Shortens text for display in the graph."""
    if len(str(text)) > max_length:
        return str(text)[: max_length - 3] + "..."
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


# **NEW**: Added back the function to connect to the analysis DB
@st.cache_resource
def get_analysis_collection():
    """Connects to ChromaDB containing the AI analyses."""
    try:
        client = chromadb.PersistentClient(path=str(ANALYSIS_DB_PATH))
        return client.get_collection(name=ANALYSIS_COLLECTION_NAME)
    except Exception as e:
        st.error(f"Could not connect to Analyses Database: {e}")
        return None


# --- Functions for the Conversational Chatbot ---
def format_subgraph_for_chat_prompt(df: pd.DataFrame) -> str:
    """Formats a subgraph into a simple, readable list of facts for the LLM."""
    facts = []
    for index, row in df.iterrows():
        fact = f"The entity '{row['x_name']}' has '{row['display_relation']}' relationship with '{row['y_name']}'."
        facts.append(fact)
    return "\n".join(facts)


def generate_chat_response(context: str, question: str, chat_history: list) -> str:
    """Calls the Ollama model with a conversational prompt including history."""
    formatted_history = ""
    for msg in chat_history:
        if msg["role"] == "user":
            formatted_history += f"Previous question: {msg['content']}\n"
        else:
            formatted_history += f"Previous answer: {msg['content']}\n"

    prompt = f"""You are an AI research assistant.
Your knowledge is strictly limited to the following facts from a knowledge graph.
Do not use any outside information.
<KNOWLEDGE_GRAPH_CONTEXT>
{context}
</KNOWLEDGE_GRAPH_CONTEXT>
<CONVERSATION_HISTORY>
{formatted_history}
</CONVERSATION_HISTORY>
Based only on the context and conversation history, answer the following new question: "{question}"
If the context does not contain the answer, state that the information is not available in the provided data.
"""
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1},
        )
        return response["message"]["content"]
    except Exception as e:
        return f"Error communicating with the AI model: {e}"


# --- Streamlit App Layout ---
st.set_page_config(layout="wide")
st.title("Knowledge Graph AI Analyst")
st.markdown(
    "Select a topic to load its knowledge subgraph, then ask any question about it."
)

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_subgraph_df" not in st.session_state:
    st.session_state.current_subgraph_df = pd.DataFrame()
if "selected_file" not in st.session_state:
    st.session_state.selected_file = None

# --- Main App Logic ---
subgraph_files = get_subgraph_files()
node_type_map, unique_types = load_node_types()
analysis_collection = get_analysis_collection()  # Connect to the analysis DB

if not subgraph_files:
    st.warning(f"No subgraph files found in '{SUBGRAPHS_DIR}'.")
else:
    filter_options = ["All Types"] + unique_types
    selected_type_filter = st.selectbox(
        "Filter subgraphs by node type:", options=filter_options
    )

    if selected_type_filter != "All Types":
        filtered_files = [
            f
            for f in subgraph_files
            if node_type_map.get(
                Path(f).stem.replace("_subgraph", "").replace("_", " ")
            ) == selected_type_filter
        ]
        subgraphs_to_display = filtered_files
    else:
        subgraphs_to_display = subgraph_files

    if not subgraphs_to_display:
        st.warning(f"No subgraphs found for the type '{selected_type_filter}'.")
    else:
        selected_file = st.selectbox(
            "Select a Subgraph to Analyze:",
            options=subgraphs_to_display,
            key="subgraph_selector",
        )

        if selected_file != st.session_state.selected_file:
            st.session_state.selected_file = selected_file
            st.session_state.messages = []
            try:
                file_path = SUBGRAPHS_DIR / selected_file
                st.session_state.current_subgraph_df = pd.read_csv(file_path)
            except Exception as e:
                st.error(f"Could not read subgraph file: {e}")
                st.session_state.current_subgraph_df = pd.DataFrame()
            st.rerun()

        if selected_file:
            # **NEW**: The full four-tab layout
            tab1, tab2, tab3, tab4 = st.tabs(
                [
                    "AI Expert Analysis",
                    "Chat with AI Analyst",
                    "Interactive Graph",
                    "Raw Data",
                ]
            )

            # --- Tab 1: The Pre-computed AI Analysis ---
            with tab1:
                st.subheader("AI Expert Analysis")
                if analysis_collection:
                    with st.spinner("Retrieving analysis..."):
                        try:
                            retrieved_analysis = analysis_collection.get(
                                ids=[selected_file]
                            )
                            if retrieved_analysis and retrieved_analysis["documents"]:
                                st.markdown(retrieved_analysis["documents"][0])
                            else:
                                st.warning(
                                    "No pre-generated analysis found for this subgraph."
                                )
                        except Exception as e:
                            st.error(f"Could not retrieve analysis from database: {e}")
                else:
                    st.error("Analysis database is not available.")

            # --- Tab 2: The Conversational Chatbot ---
            with tab2:
                st.subheader(f"Chat about `{selected_file}`")
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                if prompt := st.chat_input("Ask a follow-up question..."):
                    st.session_state.messages.append(
                        {"role": "user", "content": prompt}
                    )
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            context = format_subgraph_for_chat_prompt(
                                st.session_state.current_subgraph_df
                            )
                            response = generate_chat_response(
                                context, prompt, st.session_state.messages[:-1]
                            )
                            st.markdown(response)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": response}
                            )

            # --- Tab 3: Display the Interactive Graph ---
            with tab3:
                st.subheader(f"Interactive Graph for `{selected_file}`")
                if not st.session_state.current_subgraph_df.empty:
                    net = Network(
                        height="600px",
                        width="100%",
                        bgcolor="#F0F0F0",
                        font_color="black",
                        directed=True,
                    )
                    for _, row in st.session_state.current_subgraph_df.head(
                        MAX_NODES
                    ).iterrows():
                        src, src_type, dst, dst_type, rel = (
                            row["x_name"],
                            row["x_type"],
                            row["y_name"],
                            row["y_type"],
                            row["display_relation"],
                        )
                        src_color, dst_color = NODE_COLORS.get(
                            src_type, NODE_COLORS["default"]
                        ), NODE_COLORS.get(dst_type, NODE_COLORS["default"])
                        net.add_node(
                            src,
                            label=truncate_label(src, LABEL_MAX_LENGTH),
                            title=f"{src}\nType: {src_type}",
                            color=src_color,
                        )
                        net.add_node(
                            dst,
                            label=truncate_label(dst, LABEL_MAX_LENGTH),
                            title=f"{dst}\nType: {dst_type}",
                            color=dst_color,
                        )
                        net.add_edge(src, dst, label=rel, title=rel)
                    net.save_graph("graph.html")
                    with open("graph.html", "r", encoding="utf-8") as f:
                        components.html(f.read(), height=610, scrolling=True)
                else:
                    st.warning("Subgraph data is empty.")

            # --- Tab 4: Display the Raw Data Table ---
            with tab4:
                st.subheader("Raw Subgraph Data")
                st.dataframe(st.session_state.current_subgraph_df)
