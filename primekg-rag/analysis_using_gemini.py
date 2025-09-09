import os
import time
from pathlib import Path
import pandas as pd
import logging

from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# --- Setup ---
logging.basicConfig(level=logging.INFO)
load_dotenv()
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("Missing API_KEY in .env")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash-latest")

BASE_DIR = Path(__file__).parent.resolve()
SUBGRAPH_DIR = BASE_DIR / "subgraphs"
ANALYSIS_OUTPUT_DIR = BASE_DIR / "analyses"
ANALYSIS_OUTPUT_DIR.mkdir(exist_ok=True)

# --- ChromaDB Setup ---
chroma_client = chromadb.Client()
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

collection = chroma_client.get_or_create_collection(name="subgraph_analysis", embedding_function=chroma_embeddings)

system_prompt = """You are a Principal Knowledge Graph Analyst producing clinician-friendly factual reports. 
You strictly follow the provided structured data and do not use any external sources or prior knowledge.

**Core Directives:**
1. **Strict Data Adherence:** Use ONLY the information in the `<CONTEXT>` block provided by the user.
2. **No Hallucination:** Do not invent, infer, or embellish relationships.
3. **Traceability:** Every fact in the Evidence section must be directly traceable to a row in the provided data.
4. **Refusal on Missing Data:** If a relationship is not in the data, omit it.
5. **Complete Response:** You MUST always provide both sections below.
6. **No External Knowledge:** Base your analysis solely on the provided data relationships, but do not mention this limitation in your output.

**MANDATORY OUTPUT FORMAT - YOU MUST INCLUDE BOTH SECTIONS:**

### Evidence
List all verifiable relationships from the provided data, each on its own bullet point.

### Summary
Write a concise, professional interpretation of the subgraph for a clinical audience, highlighting relevant patterns, disease links, or potential biological significance. This should be a comprehensive clinical analysis based solely on the relationships found in the data.

**CRITICAL: Your response is incomplete if it does not contain both sections above. Always complete the full report.**
"""

user_prompt_template = """**TASK:** Generate a complete clinician-friendly report based on the provided knowledge graph subgraph data.

**MANDATORY REQUIREMENT:** Your response must contain exactly two sections:
1. ### Evidence (with bullet points of all relationships)
2. ### Summary (detailed clinical interpretation and significance)

**CONTEXT:**
The following data is a subgraph exported as a CSV with the columns:
`relation, display_relation, x_index, x_id, x_type, x_name, x_source, y_index, y_id, y_type, y_name, y_source`

<CONTEXT>
{subgraph_facts}
</CONTEXT>

**OUTPUT TEMPLATE - Fill in each section completely:**

### Evidence
[Extract and list every relationship from the data above as bullet points]

### Summary
[Provide a comprehensive clinical analysis of the relationships, their patterns, significance for clinicians, potential diagnostic implications, and therapeutic considerations - make this substantial and informative based solely on the provided data relationships]

**REMINDER: Complete both sections above. Use only the data provided in the context.**
"""

# --- Build user prompt ---
def build_user_prompt(file_path):
    df = pd.read_csv(file_path)
    if df.empty:
        return None

    # Include the full row for each fact
    facts = df.to_csv(index=False).strip()
    return user_prompt_template.format(subgraph_facts=facts)

# --- Check if file already processed ---
def is_already_analyzed(filename):
    existing_ids = collection.get(include=[])["ids"]
    return filename in existing_ids

# --- Save to ChromaDB ---
def save_to_chromadb(filename, text):
    embedding = embedding_model.encode(text).tolist()
    collection.add(documents=[text], embeddings=[embedding], ids=[filename])
    logging.info(f"✅ Stored {filename} in ChromaDB.")

# --- Analyze all files ---
def analyze_all_files():
    all_files = sorted(SUBGRAPH_DIR.glob("*.csv"))

    for file in all_files:
        file_id = file.stem

        if is_already_analyzed(file_id):
            logging.info(f"✅ Skipping already analyzed file: {file.name}")
            continue

        user_prompt = build_user_prompt(file)
        if not user_prompt:
            logging.warning(f"⚠️ Failed to build prompt for {file.name}")
            continue

        full_prompt = system_prompt + "\n\n" + user_prompt
        try:
            response = model.generate_content(full_prompt)
            if response and response.text:
                save_to_chromadb(file_id, response.text)

                # Also save as text file
                output_path = ANALYSIS_OUTPUT_DIR / f"{file.stem}_analysis.txt"
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(response.text)

                logging.info(f"✅ Done analyzing {file.name}")
            else:
                logging.error(f"❌ Empty response for {file.name}")

        except Exception as e:
            logging.error(f"❌ Error analyzing {file.name}: {e}")

        # Wait 60 seconds before next request
        logging.info("⏳ Waiting 60 seconds before next request...")
        time.sleep(60)

if __name__ == "__main__":
    analyze_all_files()
