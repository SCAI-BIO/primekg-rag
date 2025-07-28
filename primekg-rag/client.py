# import streamlit as st
# import pandas as pd
# from pyvis.network import Network
# import streamlit.components.v1 as components
# import os
# import chromadb

# # --- Configuration ---
# MATCHES_FILE = "qa_to_node_matches_improved.csv"
# SUBGRAPHS_DIR = "subgraphs"
# ANALYSIS_DB_PATH = "analyses_db"
# ANALYSIS_COLLECTION_NAME = "subgraph_analyses"
# MAX_NODES = 15
# LABEL_MAX_LENGTH = 15
# NODE_COLORS = {
#     "gene/protein": "#4CAF50", "drug": "#2196F3", "disease": "#9C27B0",
#     "phenotype": "#00BCD4", "pathway": "#FFC107", 
#     "molecular_function": "#F44336", "biological_process": "#E91E63",
#     "cellular_component": "#673AB7", "compound": "#8BC34A",
#     "chemical_compound": "#8BC34A", "biological_entity": "#FFEB3B",
#     "exposure": "#FF9800", "symptom": "#CDDC39",
#     "default": "#607D8B"
# }

# # --- Helper Functions ---
# @st.cache_data
# def load_matches_file():
#     """Loads the CSV file with the Q&A matches."""
#     if not os.path.exists(MATCHES_FILE):
#         st.error(f"Matches file not found: '{MATCHES_FILE}'")
#         return pd.DataFrame()
#     return pd.read_csv(MATCHES_FILE)

# @st.cache_data
# def get_subgraph_files():
#     """Finds all available subgraph CSV files."""
#     if not os.path.exists(SUBGRAPHS_DIR):
#         return []
#     return sorted([f for f in os.listdir(SUBGRAPHS_DIR) if f.endswith('.csv')])

# @st.cache_resource
# def get_analysis_collection():
#     """Connects to the ChromaDB collection containing the AI analyses."""
#     try:
#         client = chromadb.PersistentClient(path=ANALYSIS_DB_PATH)
#         return client.get_collection(name=ANALYSIS_COLLECTION_NAME)
#     except Exception as e:
#         st.error(f"Could not connect to the Analyses Database: {e}")
#         return None

# def truncate_label(text, max_length):
#     """Shortens text for display in the graph."""
#     if len(text) > max_length:
#         return text[:max_length-3] + "..."
#     return text

# # --- Streamlit App Layout ---
# st.set_page_config(layout="wide")
# st.title("Knowledge Graph & AI Analyst")

# # --- Part 1: Display the Q&A Matches Table ---
# st.subheader("1. Q&A Matched to Prime KG")
# matches_df = load_matches_file()
# if not matches_df.empty:
#     st.dataframe(matches_df)
# else:
#     st.warning("Could not load the matches file.")

# st.divider()

# # --- Part 2: Visualize Subgraph and Display AI Analysis ---
# st.subheader("2. Subgraph Explorer & AI Analysis")
# subgraph_files = get_subgraph_files()
# analysis_collection = get_analysis_collection()

# if not subgraph_files:
#     st.warning(f"No subgraph files found in the '{SUBGRAPHS_DIR}' folder.")
# else:
#     selected_file = st.selectbox(
#         "Select a Subgraph to Analyze:",
#         options=subgraph_files
#     )

#     if selected_file:
#         # Display the Interactive Graph
#         st.markdown(f"#### Interactive Graph for `{selected_file}`")
#         file_path = os.path.join(SUBGRAPHS_DIR, selected_file)
#         full_subgraph_df = pd.read_csv(file_path)
        
#         nodes_in_graph = set()
#         edges_for_graph = []
#         for _, row in full_subgraph_df.iterrows():
#             new_nodes_count = len(set([row['x_name'], row['y_name']]) - nodes_in_graph)
#             if len(nodes_in_graph) + new_nodes_count > MAX_NODES:
#                 continue
#             nodes_in_graph.add(row['x_name'])
#             nodes_in_graph.add(row['y_name'])
#             edges_for_graph.append(row)
        
#         trimmed_subgraph_df = pd.DataFrame(edges_for_graph)

#         if not trimmed_subgraph_df.empty:
#             net = Network(height="500px", width="100%", bgcolor="#F0F0F0", font_color="black", directed=True)
#             for _, row in trimmed_subgraph_df.iterrows():
#                 src, src_type = row['x_name'], row['x_type']
#                 dst, dst_type = row['y_name'], row['y_type']
#                 rel = row['display_relation']
                
#                 src_color = NODE_COLORS.get(src_type, NODE_COLORS['default'])
#                 dst_color = NODE_COLORS.get(dst_type, NODE_COLORS['default'])
#                 src_label = truncate_label(src, LABEL_MAX_LENGTH)
#                 dst_label = truncate_label(dst, LABEL_MAX_LENGTH)
#                 src_title = f"{src}\nType: {src_type}"
#                 dst_title = f"{dst}\nType: {dst_type}"
                
#                 net.add_node(src, label=src_label, title=src_title, color=src_color)
#                 net.add_node(dst, label=dst_label, title=dst_title, color=dst_color)
#                 net.add_edge(src, dst, label=rel, title=rel)
            
#             net.show_buttons(filter_=['physics'])
#             try:
#                 net.save_graph("final_graph.html")
#                 with open("final_graph.html", "r", encoding="utf-8") as f:
#                     components.html(f.read(), height=510, scrolling=True)
#             except Exception as e:
#                 st.error(f"Could not generate visualization: {e}")
#         else:
#             st.warning("No relationships could be displayed within the node limit.")

#         st.divider()

#         # Display the Pre-Generated AI Analysis
#         st.subheader("AI Expert Analysis")
#         if analysis_collection:
#             with st.spinner("Retrieving analysis..."):
#                 retrieved_analysis = analysis_collection.get(ids=[selected_file])
                
#                 if retrieved_analysis and retrieved_analysis['documents']:
#                     st.markdown(retrieved_analysis['documents'][0])
#                 else:
#                     st.warning("No pre-generated analysis found for this subgraph.")
#         else:
#             st.error("Analysis database not available.")


import streamlit as st
import pandas as pd
from pyvis.network import Network
import streamlit.components.v1 as components
import os
import chromadb
from sentence_transformers import SentenceTransformer, util
import numpy as np
import re
import torch # Sentence-transformers uses PyTorch
import ollama # For local LLM integration

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths for subgraphs and the database relative to the script's location.
ANALYSIS_COLLECTION_NAME = "subgraph_analyses"
OLLAMA_MODEL_NAME = "deepseek-r1:14b"

# --- Configuration ---
MATCHES_FILE = os.path.join(BASE_DIR,"qa_to_node_matches_improved.csv")
SUBGRAPHS_DIR = os.path.join(BASE_DIR,"subgraphs")
ANALYSIS_DB_PATH = os.path.join(BASE_DIR, "analyses_db")

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
# Define the model we'll use for embeddings
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
# Define the local model for generation
OLLAMA_MODEL_NAME = "deepseek-r1:14b"



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
    if not isinstance(text, str):
        text = str(text)
    if len(text) > max_length:
        return text[: max_length - 3] + "..."
    return text

<<<<<<< HEAD
# --- Chatbot Helper Functions ---

@st.cache_resource
def load_embedding_model():
    """Loads the Sentence Transformer model from cache."""
    return SentenceTransformer(EMBEDDING_MODEL)

def split_into_sentences(text):
    """Splits text into a list of sentences."""
    if not text or not isinstance(text, str):
        return []
    sentences = re.split(r'(?<=[.?!])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def get_most_relevant_sentence(question, context_sentences, model):
    """
    Finds the most relevant sentence using a Sentence Transformer model.
    """
    if not context_sentences:
        return None, 0.0

    # Encode the question and context sentences into embeddings
    question_embedding = model.encode(question, convert_to_tensor=True)
    context_embeddings = model.encode(context_sentences, convert_to_tensor=True)
    
    # Calculate cosine similarity
    cosine_scores = util.cos_sim(question_embedding, context_embeddings)
    
    # Find the index of the highest score
    most_similar_index = torch.argmax(cosine_scores)
    max_similarity_score = cosine_scores[0][most_similar_index].item()
    
    return context_sentences[most_similar_index], max_similarity_score

def generate_llm_response(question, context_sentence):
    """
    Calls a locally running Ollama model to answer a question
    based on a specific piece of context.
    """
    # Construct the prompt for the local LLM, instructing it to be concise and use only the provided context.
    prompt = f"""You are an AI assistant. Answer the user's question based ONLY on the following context. Do not use any outside knowledge. Be concise.

Context: "{context_sentence}"

Question: "{question}"

Answer:"""

    try:
        # Make the API call to the local Ollama server
        response = ollama.chat(
            model=OLLAMA_MODEL_NAME,
            messages=[
                {'role': 'user', 'content': prompt}
            ]
        )
        # Extract the content from the response
        return response['message']['content']
    except Exception as e:
        st.error(f"Error connecting to Ollama model '{OLLAMA_MODEL_NAME}': {e}")
        return "I'm sorry, I'm having trouble connecting to my reasoning model. Please ensure Ollama is running and the model is available."

=======
>>>>>>> 0a1a1d5724d2b2d238e128c2a15b60c4e316bfbd

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="STRATA DSS")
st.title("STRATA: Knowledge Graph & AI Analyst 🧠")

# Load the embedding model once
embedding_model = load_embedding_model()

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
<<<<<<< HEAD
        "Select a Subgraph to Analyze:",
        options=subgraph_files,
        key='selected_file_key' # Give a key to track changes
=======
        "Select a Subgraph to Analyze:", options=subgraph_files
>>>>>>> 0a1a1d5724d2b2d238e128c2a15b60c4e316bfbd
    )

    # Initialize session state variables
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'analysis_context' not in st.session_state:
        st.session_state.analysis_context = ""

    if selected_file:
        # --- Graph Visualization Logic ---
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
<<<<<<< HEAD
                src, src_type = row['x_name'], str(row['x_type'])
                dst, dst_type = row['y_name'], str(row['y_type'])
                rel = str(row['display_relation'])
                
                src_color = NODE_COLORS.get(src_type, NODE_COLORS['default'])
                dst_color = NODE_COLORS.get(dst_type, NODE_COLORS['default'])
=======
                src, src_type = row["x_name"], row["x_type"]
                dst, dst_type = row["y_name"], row["y_type"]
                rel = row["display_relation"]

                src_color = NODE_COLORS.get(src_type, NODE_COLORS["default"])
                dst_color = NODE_COLORS.get(dst_type, NODE_COLORS["default"])
>>>>>>> 0a1a1d5724d2b2d238e128c2a15b60c4e316bfbd
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

        # --- AI Analysis Retrieval ---
        st.subheader("AI Expert Analysis")
        if analysis_collection:
            with st.spinner("Retrieving analysis..."):
                if st.session_state.current_file != selected_file:
                    st.session_state.current_file = selected_file
                    st.session_state.messages = []
                    st.session_state.analysis_context = ""

                retrieved_analysis = analysis_collection.get(ids=[selected_file])
<<<<<<< HEAD
                
                if retrieved_analysis and retrieved_analysis['documents']:
                    analysis_text = retrieved_analysis['documents'][0]
                    st.session_state.analysis_context = analysis_text
                    st.markdown(analysis_text)
=======

                if retrieved_analysis and retrieved_analysis["documents"]:
                    st.markdown(retrieved_analysis["documents"][0])
>>>>>>> 0a1a1d5724d2b2d238e128c2a15b60c4e316bfbd
                else:
                    st.warning("No pre-generated analysis found for this subgraph.")
                    st.session_state.analysis_context = ""
        else:
            st.error("Analysis database not available.")
<<<<<<< HEAD

# --- Part 3: Interactive Chat with AI Analyst ---
if st.session_state.get('analysis_context'):
    st.divider()
    st.subheader("3. Interactive Chat with AI Analyst")

    if not st.session_state.messages:
        st.session_state.messages.append({"role": "assistant", "content": "Ask me a follow-up question about the analysis above."})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Your question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                context_sentences = split_into_sentences(st.session_state.analysis_context)
                
                relevant_sentence, score = get_most_relevant_sentence(prompt, context_sentences, embedding_model)
                
                SIMILARITY_THRESHOLD = 0.5 
                
                if score > SIMILARITY_THRESHOLD:
                    # **UPGRADED LOGIC**
                    # Now we call the generative model to answer the question
                    # using the retrieved sentence as context.
                    response = generate_llm_response(prompt, relevant_sentence)
                else:
                    response = "I'm sorry, I couldn't find a direct answer to your question within the provided analysis. Could you try rephrasing or asking about a specific concept mentioned in the text?"

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
=======
>>>>>>> 0a1a1d5724d2b2d238e128c2a15b60c4e316bfbd
