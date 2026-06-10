import chromadb
from chromadb.utils import embedding_functions
import json
import logging
from tqdm import tqdm
from pathlib import Path
import os
import sys

# --- Configuration ---
# Get the parent directory path
PARENT_DIR = Path(__file__).parent.parent

# Find the most recent JSON file from the provided list
def find_latest_json():
    # List of known JSON files in order of newest to oldest
    known_files = [
        "pubmed_abstracts_full_20250908_102008.json",
        "pubmed_abstracts_full_20250903_111320.json",
        "pubmed_abstracts_full_20250903_072116.json"
    ]
    
    # Check each file in order and return the first one that exists
    for filename in known_files:
        file_path = PARENT_DIR / filename
        if file_path.exists():
            logging.info(f"Found JSON file: {file_path}")
            return file_path
    
    # If no known files found, try to find any matching file
    json_files = list(PARENT_DIR.glob("pubmed_abstracts_full_*.json"))
    if json_files:
        latest_file = max(json_files, key=os.path.getmtime)
        logging.warning(f"Using fallback to find latest file: {latest_file}")
        return latest_file
        
    raise FileNotFoundError("No PubMed JSON files found in the parent directory")

# The path to the latest JSON file containing the PubMed abstracts
PUBMED_JSON_FILE = find_latest_json()
# The path where the persistent database will be stored
DB_PATH = str(PARENT_DIR / "pubmed_db")  # Changed to use absolute path in project root
# The name of the collection within the database
COLLECTION_NAME = "pubmed_abstracts"
# The embedding model to use. Must match the one in your Streamlit app.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PARENT_DIR / "pmc_embed.log")
    ]
)

def get_collection_stats(collection):
    """Get and log collection statistics."""
    count = collection.count()
    logging.info(f"Current collection size: {count:,} documents")
    return count

def main():
    """
    Main function to load data from a JSON file and insert it into a persistent ChromaDB.
    """
    # Create database directory if it doesn't exist
    os.makedirs(DB_PATH, exist_ok=True)
    
    # --- 1. Initialize Database Client ---
    logging.info(f"Initializing persistent ChromaDB client at: {DB_PATH}")
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # --- 2. Create or Get Collection ---
    logging.info(f"Using embedding model: {MODEL_NAME}")
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODEL_NAME
    )
    
    logging.info(f"Getting or creating collection: {COLLECTION_NAME}")
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )
    
    # Log initial collection stats
    initial_count = get_collection_stats(collection)
    
    # --- 3. Load Data from JSON File ---
    json_path = Path(PUBMED_JSON_FILE)
    logging.info(f"Loading data from: {json_path}")
    
    if not json_path.is_file():
        logging.error(f"JSON file not found at '{json_path}'. Please check the path.")
        return
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            pubmed_data = json.load(f)
        logging.info(f"Successfully loaded data from {json_path.name}")
    except Exception as e:
        logging.error(f"Error loading JSON file: {str(e)}")
        return
    
    # --- 4. Prepare and Insert Data ---
    logging.info("Preparing documents for insertion into ChromaDB...")
    
    # Get existing document IDs to avoid duplicates
    existing_ids = set(collection.get()['ids'])
    logging.info(f"Found {len(existing_ids):,} existing documents in the collection")
    
    # Prepare documents for insertion
    documents = []
    metadatas = []
    ids = []
    
    for topic, articles in tqdm(pubmed_data.items(), desc="Processing Topics"):
        for article in articles:
            pmid = str(article.get("pmid"))
            title = article.get("title", "")
            abstract = article.get("abstract", "")
            
            # Skip if document already exists
            if pmid in existing_ids:
                continue
                
            # Combine title and abstract for a richer document content
            text_content = f"{title}\n\n{abstract}"

            if not pmid or not text_content.strip():
                logging.warning(f"Skipping article with missing pmid or content for topic '{topic}'.")
                continue

            ids.append(pmid)
            documents.append(text_content)
            metadatas.append({
                "topic": topic,
                "pmid": pmid,
                "title": title
            })
    
    # --- 5. Insert New Documents ---
    if not ids:
        logging.info("No new documents to add.")
        return
        
    logging.info(f"Adding {len(ids):,} new documents to the collection")
    
    # Insert in batches to avoid memory issues
    batch_size = 100
    for i in tqdm(range(0, len(ids), batch_size), desc="Inserting documents"):
        batch_ids = ids[i:i+batch_size]
        batch_docs = documents[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]
        
        try:
            collection.upsert(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metas
            )
        except Exception as e:
            logging.error(f"Error inserting batch {i//batch_size + 1}: {str(e)}")
    
    # Log final collection stats
    final_count = get_collection_stats(collection)
    logging.info(f"Added {final_count - initial_count:,} new documents to the collection")
    logging.info(f"Total documents in collection: {final_count:,}")

if __name__ == "__main__":
    main()