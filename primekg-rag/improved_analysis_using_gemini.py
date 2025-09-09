import os
import time
from pathlib import Path
import pandas as pd
import logging
from collections import defaultdict, Counter

from dotenv import load_dotenv
import requests
import chromadb
from sentence_transformers import SentenceTransformer

# --- Setup ---
logging.basicConfig(level=logging.INFO)
load_dotenv()

# DeepSeek API configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "http://localhost:11434/api/generate"  # Local Ollama server

if not DEEPSEEK_API_KEY:
    logging.warning("DEEPSEEK_API_KEY not found in .env, using local model")
    DEEPSEEK_API_KEY = ""

BASE_DIR = Path(__file__).parent.resolve()
SUBGRAPH_DIR = BASE_DIR / "new_subgraphs"
ANALYSIS_OUTPUT_DIR = BASE_DIR / "analyses"
ANALYSIS_DB_PATH = BASE_DIR / "analyses_db"
PUBMED_DB_PATH = BASE_DIR / "pubmed_db"
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
    
    # Get relevant PubMed papers
    pubmed_context = get_pubmed_context(condition)
    
    system_prompt = f"""You are a biomedical knowledge graph analyst with expertise in interpreting research literature. Your task is to generate a comprehensive clinical summary by integrating knowledge from both a knowledge subgraph and relevant scientific literature.

**CRITICAL INSTRUCTIONS:**
1. **Cite Everything:** Every statement must be supported by either:
   - Knowledge graph evidence: `[KG: ID: y_id, Source: y_source]`
   - Research evidence: `[PMID: xxxxxxx]` (from provided papers)
2. **Prioritize Evidence:** Use knowledge graph data as primary evidence, supplemented by research papers for context.
3. **Be Precise:** Only make claims that can be directly supported by the provided evidence.

**MANDATORY OUTPUT FORMAT:**

## Clinical Summary: {condition}

### Disease Overview
[Provide a 2-3 paragraph overview of the condition based on the most relevant research papers. Include key epidemiological data, clinical presentation, and current understanding of the condition. Cite research papers using PMIDs.]

### Key Clinical Relationships
[Bullet points of the most clinically significant relationships from the knowledge graph. Each point must include a KG citation.]

### Therapeutic Insights
[Summarize potential treatments, focusing on evidence from the knowledge graph. Include any drug mechanisms or interactions found. Add relevant context from research papers when available.]

### Biological Mechanisms
[Describe genes, proteins, and pathways involved, with emphasis on KG evidence. Use research papers to explain mechanisms and biological context.]

### Research Context
[Highlight 3-5 key findings from the most relevant papers that provide additional context not fully captured in the knowledge graph.]

**Remember:** All claims must be directly supported by either knowledge graph evidence or cited research papers."""

    user_prompt = f"""**CONTEXT FOR ANALYSIS:**

**Condition:** {condition}
**Source File:** {file_path.name}

**RELEVANT RESEARCH LITERATURE (Top 50 most relevant papers):**
{pubmed_context}

**EVIDENCE FROM KNOWLEDGE GRAPH:**
{evidence_text}

**STATISTICAL SUMMARY:**
- Most connected entities: {', '.join([f"{name} ({count} connections)" for name, count in stats['top_connected_nodes'][:5]])}
- Relationship type distribution: {', '.join([f"{rel}: {count}" for rel, count in sorted(stats['relation_counts'].items(), key=lambda x: x[1], reverse=True)[:5]])}
- Entity type distribution: {', '.join([f"{etype}: {count}" for etype, count in sorted(stats['node_type_counts'].items(), key=lambda x: x[1], reverse=True)[:5]])}

**INSTRUCTIONS:**
1. First, identify the most relevant papers that provide context about {condition}.
2. Analyze the knowledge graph evidence, focusing on relationships that are supported by multiple sources.
3. Integrate insights from both the knowledge graph and research papers, clearly citing your sources.
4. Highlight any discrepancies or gaps between the knowledge graph and current research."""

    return system_prompt, user_prompt

def get_pubmed_context(condition: str, top_k: int = 50) -> str:
    """Retrieve relevant PubMed papers for the given condition."""
    try:
        # Connect to the pubmed database
        client = chromadb.PersistentClient(path=str(PUBMED_DB_PATH))
        collection = client.get_collection("pubmed_abstracts")
        
        # Get the embedding of the condition
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode(condition).tolist()
        
        # Search for similar documents
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas"]
        )
        
        # Format the results
        context = []
        for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
            title = meta.get('title', 'No title available')
            pmid = meta.get('pmid', 'N/A')
            abstract = doc
            
            context.append(f"{i}. [PMID: {pmid}] {title}\n   {abstract[:500]}{'...' if len(abstract) > 500 else ''}")
        
        return "\n\n".join(context)
        
    except Exception as e:
        logging.error(f"Error retrieving PubMed context: {str(e)}")
        return "[Could not retrieve PubMed context due to an error]"

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

def is_ollama_running() -> bool:
    """Check if the Ollama server is running and DeepSeek model is available."""
    try:
        # Check server status
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code != 200:
            logging.error("Ollama server is not running or not accessible")
            return False
            
        # Check if DeepSeek model is available
        models = response.json().get("models", [])
        deepseek_available = any("deepseek" in model.get("name", "").lower() for model in models)
        
        if not deepseek_available:
            logging.error("DeepSeek model is not available. Please pull it with: ollama pull deepseek-r1:14b")
            return False
            
        return True
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Error connecting to Ollama server: {e}")
        return False

def analyze_all_files():
    """Analyze all subgraph files and generate improved analyses."""
    # Check if Ollama server is running and DeepSeek is available
    if not is_ollama_running():
        logging.error("❌ Please start the Ollama server and ensure the DeepSeek model is available")
        logging.info("\nTo fix this, run these commands in your terminal:")
        logging.info("1. Start the Ollama server: ollama serve")
        logging.info("2. In a new terminal, pull the model: ollama pull deepseek-r1:14b")
        return
        
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
            
            # Generate analysis with DeepSeek
            full_prompt = f"""{system_prompt}
            
            {user_prompt}"""
            
            # Prepare the request data
            data = {
                "model": "deepseek-r1:14b",
                "prompt": full_prompt,
                "stream": False
            }
            
            # Make the API request
            response = requests.post(
                DEEPSEEK_API_URL,
                json=data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            response_data = response.json()
            analysis_text = response_data.get("response", "")
            
            if analysis_text:
                # Save as text file
                output_path = ANALYSIS_OUTPUT_DIR / f"{file.stem}_analysis.txt"
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(analysis_text)
                
                # Save to analyses database
                condition = file.stem.replace('_subgraph', '').replace('_', ' ').title()
                save_to_new_analyses_db(
                    filename=output_path.name,
                    analysis_text=analysis_text,
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
