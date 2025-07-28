import os
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.llms import Ollama

# --- Configuration ---
SUBGRAPHS_DIR = "subgraphs"
ANALYSIS_DB_PATH = "analyses_db"
ANALYSIS_COLLECTION_NAME = "subgraph_analyses"
OLLAMA_MODEL_NAME = "deepseek-r1:14b"
RAG_DB_PATH = "rag_db"
RAG_COLLECTION_NAME = "subgraph_relations"

# --- RAG Class Definition ---


class OllamaRAG:
    """A self-contained class to handle the RAG process with Ollama."""

    def __init__(self, model_name, db_path, collection_name):
        self.llm = self._load_ollama_llm(model_name)
        self.collection = self._get_chroma_collection(db_path, collection_name)

    def _load_ollama_llm(self, model_name):
        print(f"--- Initializing connection to Ollama model: {model_name} ---")
        try:
            return Ollama(model=model_name)
        except Exception as e:
            print(f" Error initializing Ollama: {e}")
            return None

    def _get_chroma_collection(self, db_path, collection_name):
        print(f"---  Connecting to RAG DB at: {db_path} ---")
        try:
            client = chromadb.PersistentClient(path=db_path)
            return client.get_collection(name=collection_name)
        except Exception as e:
            print(f"Failed to connect to ChromaDB: {e}")
            return None

    def get_expert_explanation(self, topic: str, n_results: int = 10):
        if not self.llm or not self.collection:
            return "System not initialized properly."

        results = self.collection.query(query_texts=[topic], n_results=n_results)
        if not results["documents"] or not results["documents"][0]:
            return f"No context found for topic: {topic}"

        context_block = "\n".join(f"- {sentence}" for sentence in results["documents"][0])

        prompt = f"""
You are a leading AI research analyst specializing in bioinformatics and systems biology.
Your task is to analyze a set of relationships from a knowledge graph and provide
a high-level summary of their implications.

**Provided Context from Knowledge Graph:**
{context_block}

**Your Analysis Task:**
Based *only* on the context provided, generate a professional analysis of the connections related to "{topic}".
Structure your response with the following sections:
1.  **Introduction:** Briefly state the central theme emerging from the data.
2.  **Key Findings:** Synthesize the specific relationships into 2-3 key thematic points.
3.  **Overall Implications:** Conclude with a high-level summary of what these connections imply
about the complex nature of the topic.

**Professional Analysis:**
"""
        response = self.llm.invoke(prompt)
        return response


# --- Main Execution Logic ---


def generate_all_analyses():
    """
    Generates an LLM analysis for each subgraph and stores it in a new DB.
    """
    print("--- Starting Analysis Generation and Storage ---")

    rag_system = OllamaRAG(
        model_name=OLLAMA_MODEL_NAME,
        db_path=RAG_DB_PATH,
        collection_name=RAG_COLLECTION_NAME,
    )
    if not rag_system.llm or not rag_system.collection:
        print("Could not initialize RAG system. Aborting.")
        return

    analysis_client = chromadb.PersistentClient(path=ANALYSIS_DB_PATH)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    analysis_collection = analysis_client.get_or_create_collection(
        name=ANALYSIS_COLLECTION_NAME, embedding_function=embedding_func
    )

    subgraph_files = [f for f in os.listdir(SUBGRAPHS_DIR) if f.endswith(".csv")]
    if not subgraph_files:
        print(f"No subgraph files found in '{SUBGRAPHS_DIR}'.")
        return

    print(f"Generating and storing analyses for {len(subgraph_files)} subgraphs...")
    for filename in tqdm(subgraph_files, desc="Analyzing Subgraphs"):
        topic = filename.replace("_subgraph.csv", "").replace("_", " ")

        print(f"\nGenerating analysis for: {topic}")
        analysis_text = rag_system.get_expert_explanation(topic)

        if analysis_text and "No context found" not in analysis_text:
            analysis_collection.add(ids=[filename], documents=[analysis_text], metadatas=[{"topic": topic}])
            print(f" Stored analysis for: {topic}")

    print("\n--- Process Complete. All analyses have been generated and stored. ---")
    print(f"Total analyses stored: {analysis_collection.count()}")


if __name__ == "__main__":
    generate_all_analyses()
