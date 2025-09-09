import os
import time
from pathlib import Path
import pandas as pd
import logging
from collections import defaultdict, Counter

from dotenv import load_dotenv
import google.generativeai as genai
import chromadb

# --- Setup ---
logging.basicConfig(level=logging.INFO)
load_dotenv()
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("Missing API_KEY in .env")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash-latest")

BASE_DIR = Path(__file__).parent.resolve()
SUBGRAPH_DIR = BASE_DIR / "new_subgraphs"
ANALYSIS_OUTPUT_DIR = BASE_DIR / "analyses"
ANALYSIS_DB_PATH = BASE_DIR / "analyses_db"
ANALYSIS_OUTPUT_DIR.mkdir(exist_ok=True)

def analyze_subgraph_structure(df: pd.DataFrame) -> dict:
    """Analyze the subgraph structure and extract key statistics and patterns."""
    if df.empty:
        return {}
    
    # Basic statistics
    total_relationships = len(df)
    unique_nodes = set(df['x_name'].tolist() + df['y_name'].tolist())
    unique_node_types = set(df['x_type'].tolist() + df['y_type'].tolist())
    unique_relations = df['display_relation'].unique()
    
    # Node connection analysis
    node_connections = defaultdict(int)
    for _, row in df.iterrows():
        node_connections[row['x_name']] += 1
        node_connections[row['y_name']] += 1
    
    # Relationship type analysis
    relation_counts = Counter(df['display_relation'])
    
    # Node type analysis
    node_type_counts = defaultdict(int)
    for _, row in df.iterrows():
        node_type_counts[row['x_type']] += 1
        node_type_counts[row['y_type']] += 1
    
    # Group relationships by type for evidence
    relation_groups = df.groupby('display_relation')
    
    return {
        'total_relationships': total_relationships,
        'unique_nodes': len(unique_nodes),
        'unique_node_types': len(unique_node_types),
        'unique_relations': len(unique_relations),
        'node_connections': dict(node_connections),
        'relation_counts': dict(relation_counts),
        'node_type_counts': dict(node_type_counts),
        'relation_groups': relation_groups,
        'top_connected_nodes': sorted(node_connections.items(), key=lambda x: x[1], reverse=True)[:10]
    }

def format_evidence_from_subgraph(df: pd.DataFrame, stats: dict) -> str:
    """Format evidence directly from subgraph relationships, including source IDs."""
    evidence_sections = []
    
    # Group relationships by type and format them
    for relation_type, group in stats['relation_groups']:
        evidence_sections.append(f"\n**{relation_type} Relationships ({len(group)} instances):**")
        
        for _, row in group.iterrows():
            evidence_sections.append(
                f"• {row['x_name']} ({row['x_type']}) → {row['y_name']} ({row['y_type']}) [ID: {row['y_id']}, Source: {row['y_source']}]"
            )
    
    return "\n".join(evidence_sections)

def build_structured_prompt(file_path: Path) -> tuple:
    """Build a structured prompt with pre-analyzed subgraph data."""
    df = pd.read_csv(file_path)
    if df.empty:
        return None, None

    # If the dataframe is too large, sample it to avoid performance issues
    ROW_LIMIT = 10000
    if len(df) > ROW_LIMIT:
        logging.warning(f"Subgraph {file_path.name} is too large ({len(df)} rows). Sampling down to {ROW_LIMIT} rows.")
        df = df.sample(n=ROW_LIMIT, random_state=42) # Use a fixed random state for reproducibility
    
    # Analyze the subgraph structure
    stats = analyze_subgraph_structure(df)
    
    # Format evidence from the actual data
    evidence_text = format_evidence_from_subgraph(df, stats)
    
    # Truncate evidence if it's too long to avoid API errors
    if len(evidence_text) > 100000:
        evidence_text = evidence_text[:100000] + "\n... [TRUNCATED DUE TO LENGTH] ..."
    
    # Extract condition name from filename
    condition = file_path.stem.replace('_subgraph', '').replace('_', ' ').title()
    
    system_prompt = f"""You are a biomedical knowledge graph analyst. Your task is to generate a clinical summary from a knowledge subgraph.

**CRITICAL INSTRUCTIONS:**
1.  **Cite Everything:** You MUST cite every piece of information in your summary. Each statement you make must be followed by its evidence citation in the format `[ID: y_id, Source: y_source]`.
2.  **No External Knowledge:** Use ONLY the relationships provided in the 'EVIDENCE FROM SUBGRAPH' section. Do not use any outside knowledge.
3.  **Strictly Adhere to Format:** Your entire response must follow the specified markdown format.

**EXAMPLE OF A CORRECTLY CITED SENTENCE:**
- Agoraphobia is treated by Fluvoxamine `[ID: DB00176, Source: DrugBank]`.

**MANDATORY OUTPUT FORMAT:**

## Clinical Summary: {condition}

### Key Clinical Relationships
[Provide a bulleted list of the most clinically significant relationships. Each bullet point MUST end with an evidence citation.]

### Therapeutic Insights
[Summarize any potential treatments or drug interactions found in the data. Each statement MUST end with an evidence citation.]

### Associated Conditions & Phenotypes
[List any diseases or phenotypes associated with the main condition. Each statement MUST end with an evidence citation.]

### Biological Context
[Describe any genes, proteins, or biological processes linked to the condition. Each statement MUST end with an evidence citation.]

**Remember: Every single claim requires a direct citation from the evidence provided.**
"""

    user_prompt = f"""**SUBGRAPH DATA FOR ANALYSIS:**

**Condition:** {condition}
**Source File:** {file_path.name}

**EVIDENCE FROM SUBGRAPH:**
{evidence_text}

**STATISTICAL SUMMARY:**
- Most connected entities: {', '.join([f"{name} ({count} connections)" for name, count in stats['top_connected_nodes'][:5]])}
- Relationship type distribution: {', '.join([f"{rel}: {count}" for rel, count in sorted(stats['relation_counts'].items(), key=lambda x: x[1], reverse=True)[:5]])}
- Entity type distribution: {', '.join([f"{etype}: {count}" for etype, count in sorted(stats['node_type_counts'].items(), key=lambda x: x[1], reverse=True)[:5]])}

Please generate the complete analysis following the exact format specified in the system prompt.
"""

    return system_prompt, user_prompt

def save_to_new_analyses_db(filename: str, analysis_text: str, condition: str):
    """Save the analysis to the new analyses database."""
    try:
        # Connect to the analyses database
        client = chromadb.PersistentClient(path=str(ANALYSIS_DB_PATH))
        
        # Get or create the collection
        try:
            collection = client.get_collection(name="medical_analyses")
        except:
            collection = client.create_collection(
                name="medical_analyses",
                metadata={"hnsw:space": "cosine"}
            )
        
        # Prepare metadata
        metadata = {
            "filename": filename,
            "condition": condition,
            "content_length": len(analysis_text)
        }
        
        # Add to collection
        collection.add(
            documents=[analysis_text],
            metadatas=[metadata],
            ids=[f"analysis_{filename}"]
        )
        
        logging.info(f"✅ Saved {filename} to analyses database")
        
    except Exception as e:
        logging.error(f"❌ Error saving to analyses database: {e}")

def analyze_all_files():
    """Analyze all subgraph files and generate improved analyses."""
    all_files = sorted(SUBGRAPH_DIR.glob("*.csv"))
    
    if not all_files:
        logging.error("No CSV files found in subgraphs directory")
        return
    
    logging.info(f"Found {len(all_files)} subgraph files to analyze")
    
    for i, file in enumerate(all_files, 1):
        logging.info(f"Processing {i}/{len(all_files)}: {file.name}")
        
        try:
            system_prompt, user_prompt = build_structured_prompt(file)
            if not system_prompt or not user_prompt:
                logging.warning(f"⚠️ Failed to build prompt for {file.name}")
                continue
            
            # Generate analysis with Gemini
            full_prompt = system_prompt + "\n\n" + user_prompt
            response = model.generate_content(full_prompt)
            
            if response and response.text:
                # Save as text file
                output_path = ANALYSIS_OUTPUT_DIR / f"{file.stem}_analysis.txt"
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(response.text)
                
                # Save to analyses database
                condition = file.stem.replace('_subgraph', '').replace('_', ' ').title()
                save_to_new_analyses_db(
                    filename=output_path.name,
                    analysis_text=response.text,
                    condition=condition
                )
                
                logging.info(f"✅ Completed analysis for {file.name}")
            else:
                logging.error(f"❌ Empty response for {file.name}")
                
        except Exception as e:
            logging.error(f"❌ Error analyzing {file.name}: {e}")
            continue
        
        # Rate limiting - wait between requests
        if i < len(all_files):  # Don't wait after the last file
            logging.info("⏳ Waiting 10 seconds before next request...")
            time.sleep(10)
    
    logging.info("🎉 Analysis complete! All files processed.")

if __name__ == "__main__":
    analyze_all_files()
