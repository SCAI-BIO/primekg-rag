# This script is the data ingestion pipeline for the STRATA DSS.
# It reads subgraph CSVs, generates AI analyses using a local LLM (Ollama),
# and stores them in the 'analyses_db' ChromaDB collection, which is required
# by the main Streamlit application. Progress is logged to 'analysis_generation.log'.

import os
import chromadb
import pandas as pd
import ollama
import logging
from tqdm import tqdm

# --- Configuration ---
# **UPDATED FOR PORTABILITY**
# Get the absolute path of the directory where this script is located.
# This makes all other paths relative to the script, so it works on any computer.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths for subgraphs and the database relative to the script's location.
SUBGRAPHS_DIR = os.path.join(BASE_DIR, "subgraphs")
ANALYSIS_DB_PATH = os.path.join(BASE_DIR, "analyses_db")
LOG_FILE_PATH = os.path.join(BASE_DIR, "analysis_generation.log")

ANALYSIS_COLLECTION_NAME = "subgraph_analyses"
OLLAMA_MODEL_NAME = "deepseek-r1:14b"

# --- Logging Configuration ---
# Sets up a log file to track the script's progress and any errors.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=LOG_FILE_PATH,  # Use the new portable path
    filemode="w",  # 'w' overwrites the log file on each run. Use 'a' to append.
)
logger = logging.getLogger(__name__)

# --- Core Functions ---


def format_subgraph_for_prompt(df: pd.DataFrame) -> str:
    """
    Definition: Takes a DataFrame representing a subgraph and formats it into a
    numbered, human-readable list of facts. This structured text is then used
    as the context block in the prompt for the LLM, making it easy for the model
    to cite its sources.
    """
    facts = []
    for index, row in df.iterrows():
        fact = f"{index + 1}. ({row['x_name']}, {row['display_relation']}, {row['y_name']})"
        facts.append(fact)
    return "\n".join(facts)


def generate_analysis_for_subgraph(subgraph_facts: str, topic: str) -> str:
    """
    Definition: This function acts as the "Evidence Synthesizer." It takes the
    formatted facts and a topic, injects them into a carefully engineered prompt,
    and calls the local Ollama model to generate a professional analysis. The prompt
    is designed to be highly constraining to prevent hallucination.
    """
    # This prompt template is designed to be highly constraining, forcing the LLM
    # to act as a deterministic "data-to-text" engine.
    prompt_template = """The following is the content of a file named 'knowledge_graph_context.txt':

---
{context_block}
---

Based only on the file content provided, answer the following question:

"You are a leading AI research analyst specializing in bioinformatics and systems biology.
Your function is to be a deterministic data-to-text engine.
Based on the context provided, generate a professional analysis of the connections related to '{topic}'.

Strictly adhere to these rules:
- Do not infer or add any information not explicitly present in the context.
- Do not hallucinate connections or implications.
- Avoid all ambiguity.
- For every factual statement in your analysis, you MUST cite the specific fact number from the context that supports it.
Use the format `[Source: Fact X]`.

Structure your response with the following sections:
1.  **Central Theme:** Summarize the central theme of the provided data in a single, concise sentence.
2.  **Key Relationships:** Synthesize the specific relationships from the data into 2-3 numbered points.
Each point must end with citations.
3.  **Conclusive Summary:** Provide a high-level summary of what these connections explicitly represent.
Each sentence in the summary must be cited."

**Professional Analysis:**
"""
    final_prompt = prompt_template.format(context_block=subgraph_facts, topic=topic)

    try:
        # Make the API call to the local Ollama server.
        # A low temperature is critical for reducing creativity and ensuring factuality.
        response = ollama.chat(
            model=OLLAMA_MODEL_NAME,
            messages=[{"role": "user", "content": final_prompt}],
            options={"temperature": 0.1},
        )
        return response["message"]["content"]
    except Exception as e:
        logger.error(f"Ollama API call failed for topic '{topic}': {e}")
        return f"Error: Could not generate analysis for {topic}."


# --- Main Execution Block ---
if __name__ == "__main__":
    logger.info("--- Starting Analysis Generation and Database Creation Pipeline ---")

    if not os.path.exists(SUBGRAPHS_DIR):
        logger.error(
            f"The '{SUBGRAPHS_DIR}' directory was not found. Please ensure it exists and contains your CSV files."
        )
        exit()

    # 1. Set up the ChromaDB client and create the collection.
    # This will create the folder and files if they don't exist.
    logger.info(f"Setting up ChromaDB at '{ANALYSIS_DB_PATH}'...")
    client = chromadb.PersistentClient(path=ANALYSIS_DB_PATH)

    # This line CREATES the collection your main app is looking for.
    analysis_collection = client.get_or_create_collection(name=ANALYSIS_COLLECTION_NAME)
    logger.info(f"Successfully created or connected to collection: '{ANALYSIS_COLLECTION_NAME}'")

    # 2. Find all subgraph files to process.
    subgraph_files = [f for f in os.listdir(SUBGRAPHS_DIR) if f.endswith(".csv")]
    if not subgraph_files:
        logger.warning(f"No .csv files found in the '{SUBGRAPHS_DIR}' directory.")
        exit()

    logger.info(f"Found {len(subgraph_files)} subgraphs to analyze.")

    # 3. Process each file, generate an analysis, and prepare it for storage.
    analyses_to_store = []
    ids_to_store = []

    for filename in tqdm(subgraph_files, desc="Generating AI Analyses"):
        file_path = os.path.join(SUBGRAPHS_DIR, filename)
        try:
            subgraph_df = pd.read_csv(file_path)

            # The topic is derived from the filename for context (e.g., "file_name.csv" -> "file name").
            topic = os.path.splitext(filename)[0].replace("_", " ")

            # Format the subgraph data for the prompt.
            subgraph_facts = format_subgraph_for_prompt(subgraph_df)

            # Generate the analysis using the LLM.
            logger.info(f"Generating analysis for topic: {topic}")
            analysis_text = generate_analysis_for_subgraph(subgraph_facts, topic)

            if not analysis_text.startswith("Error:"):
                analyses_to_store.append(analysis_text)
                ids_to_store.append(filename)  # Use the filename as the unique ID in the database.

        except Exception as e:
            logger.error(f"Failed to process file {filename}: {e}")

    # 4. Upsert all generated analyses into ChromaDB in a single efficient batch.
    if analyses_to_store:
        logger.info(f"Storing {len(analyses_to_store)} generated analyses in ChromaDB...")
        analysis_collection.upsert(documents=analyses_to_store, ids=ids_to_store)
        logger.info("All analyses have been successfully stored.")
    else:
        logger.warning("No analyses were generated to store.")

    logger.info("--- Pipeline Finished ---")
    print(
        f"\nThe '{ANALYSIS_DB_PATH}' database and '{ANALYSIS_COLLECTION_NAME}' collection are now ready."
    )
    print("You should now be able to run your main Streamlit applicatio")
