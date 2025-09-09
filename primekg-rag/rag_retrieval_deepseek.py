# This script is the final stage of the pipeline. It reads the subgraph CSVs
# created by the data preparation script, generates a detailed, evidence-backed
# analysis for each one using an Ollama LLM, and stores the results in a new
# ChromaDB collection called 'analyses_db'.

import logging
from pathlib import Path
import pandas as pd
import chromadb
from tqdm import tqdm
import sys
import ollama

# --- Configuration ---
BASE_DIR = Path(__file__).parent
LOG_FILE_PATH = BASE_DIR / "ai_analysis.log"

# --- Input Directory ---
# This script reads from the output of the data preparation pipeline
SUBGRAPH_INPUT_DIR = BASE_DIR / "subgraphs"

# --- Output Database ---
ANALYSIS_DB_PATH = BASE_DIR / "analyses_db"
ANALYSIS_COLLECTION_NAME = "subgraph_analyses"
OLLAMA_MODEL_NAME = "deepseek-r1:14b"  # Or your preferred Ollama model

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


def format_subgraph_for_prompt(df: pd.DataFrame) -> str:
    """
    Formats a subgraph into a detailed, numbered list of facts,
    including entity types and evidence sources for maximum context.
    """
    facts = []
    for index, row in df.iterrows():
        x_type = row.get("x_type", "entity")
        y_type = row.get("y_type", "entity")
        sources = sorted(
            list(set([row.get("x_source", "N/A"), row.get("y_source", "N/A")]))
        )
        source_str = "/".join(sources)

        fact = (
            f"Fact {index + 1}: The {x_type} '{row['x_name']}' "
            f"has a '{row['display_relation']}' relationship with "
            f"the {y_type} '{row['y_name']}'. "
            f"[Evidence Source: {source_str}]"
        )
        facts.append(fact)
    return "\n".join(facts)


def generate_analysis_for_subgraph(subgraph_facts: str, topic: str) -> str:
    """
    Generates a highly structured, evidence-backed analysis using
    a sophisticated and constraining prompt.
    """
    system_prompt = """**CRITICAL INSTRUCTIONS:**
1. **Multi-Source Evidence Integration:**
   - Primary: Knowledge graph relationships `[KG: ID: y_id, Source: y_source]`
   - Secondary: Peer-reviewed research `[PMID: xxxxxxx]`
   - Tertiary: Established medical knowledge (only when explicitly relevant)

2. **Analytical Depth Required:**
   - Perform causal reasoning between related entities
   - Identify and resolve potential contradictions in evidence
   - Highlight consensus and controversies in the literature
   - Assess the strength and quality of each piece of evidence

3. **Clinical & Research Context:**
   - Relate findings to current clinical practice guidelines
   - Note the level of evidence for each claim (RCT, meta-analysis, etc.)
   - Consider potential biases in the available evidence

**MANDATORY OUTPUT FORMAT:**

## Advanced Clinical Analysis: {condition}

### 1. Comprehensive Disease Profile
[Integrate epidemiological, clinical, and molecular perspectives:
- Current diagnostic criteria and clinical subtypes
- Known risk factors and prognostic indicators
- Pathophysiological mechanisms with supporting evidence
- Recent advances in disease understanding]

### 2. Evidence-Based Relationship Analysis
[For each significant relationship:
- State the relationship in clinical terms
- Provide all supporting KG evidence with quality assessment
- Note any conflicting evidence or knowledge gaps
- Assess clinical significance and confidence level]"""

    user_prompt = f"""**ANALYSIS REQUEST**

**Condition of Interest:** {topic}

**KNOWLEDGE GRAPH EVIDENCE:**
{subgraph_facts}

**ANALYTICAL TASKS:**
1. Perform a critical synthesis of the provided knowledge graph relationships
2. Identify and elaborate on the most clinically significant patterns
3. Evaluate the strength and consistency of the evidence
4. Generate novel insights through evidence integration
5. Provide specific, actionable recommendations

**REQUIRED ELEMENTS:**
- Clear distinction between established facts and novel hypotheses
- Explicit assessment of evidence quality
- Identification of knowledge gaps and research opportunities
- Practical clinical implications
- Specific citations for all claims

**OUTPUT GUIDELINES:**
- Use clear, hierarchical organization
- Stick to the sources
- Highlight key takeaways in bullet points
- Maintain scientific rigor while being clinically relevant
- Use plain language explanations for complex concepts"""

    # Combine prompts and make API call
    final_prompt = f"{system_prompt}\n\n{user_prompt}"
    
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL_NAME,
            messages=[{"role": "user", "content": final_prompt}],
            options={"temperature": 0.05},
        )
        return response["message"]["content"]
    except Exception as e:
        logger.error(f"Ollama API call failed for topic '{topic}': {e}")
        return f"Error: Could not generate analysis for {topic}."


if __name__ == "__main__":
    logger.info("--- Starting AI Analysis Pipeline ---")
    ANALYSIS_DB_PATH.mkdir(exist_ok=True)

    if not SUBGRAPH_INPUT_DIR.is_dir():
        logger.error(
            f"The '{SUBGRAPH_INPUT_DIR}' directory was not found. Please run the data preparation script first."
        )
        sys.exit(1)

    # Set up the ChromaDB for storing analyses
    try:
        client = chromadb.PersistentClient(path=str(ANALYSIS_DB_PATH))
        analysis_collection = client.get_or_create_collection(
            name=ANALYSIS_COLLECTION_NAME
        )
        logger.info(
            f"Successfully connected to collection: '{ANALYSIS_COLLECTION_NAME}'"
        )
    except Exception as e:
        logger.critical(f"Could not connect to the analysis ChromaDB: {e}")
        sys.exit(1)

    subgraph_files = [f for f in SUBGRAPH_INPUT_DIR.glob("*.csv")]
    if not subgraph_files:
        logger.warning(f"No .csv files found in '{SUBGRAPH_INPUT_DIR}' to analyze.")
        sys.exit(1)

    logger.info(f"Found {len(subgraph_files)} subgraphs to analyze.")
    analyses_to_store, ids_to_store = [], []

    for file_path in tqdm(subgraph_files, desc="Generating AI Analyses"):
        try:
            subgraph_df = pd.read_csv(file_path)
            if subgraph_df.empty:
                logger.warning(f"Subgraph file {file_path.name} is empty, skipping.")
                continue
            topic = file_path.stem.replace("_subgraph", "").replace("_", " ")

            subgraph_facts = format_subgraph_for_prompt(subgraph_df)
            analysis_text = generate_analysis_for_subgraph(subgraph_facts, topic)

            if not analysis_text.startswith("Error:"):
                analyses_to_store.append(analysis_text)
                ids_to_store.append(file_path.name)  # Use filename as the ID
        except Exception as e:
            logger.error(f"Failed to process file {file_path.name}: {e}")

    # Upsert all generated analyses into ChromaDB in a single batch
    if analyses_to_store:
        logger.info(
            f"Storing {len(analyses_to_store)} generated analyses in ChromaDB..."
        )
        analysis_collection.upsert(documents=analyses_to_store, ids=ids_to_store)
        logger.info("All analyses have been successfully stored.")

    logger.info(
        f"(SUCCESS) AI Analysis complete. Analyses stored in '{ANALYSIS_DB_PATH}'."
    )
