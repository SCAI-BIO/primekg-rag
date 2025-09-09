import chromadb
from chromadb.utils import embedding_functions
import json
import logging
from tqdm import tqdm
from pathlib import Path

# --- Configuration ---
# The name of your JSON file containing the PubMed abstracts
PUBMED_JSON_FILE = "pubmed_abstracts.json"
# The path where the persistent database will be stored
DB_PATH = "./pubmed_db"
# The name of the collection within the database
COLLECTION_NAME = "pubmed_abstracts"
# The embedding model to use. Must match the one in your Streamlit app.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    """
    Main function to load data from a JSON file and insert it into a persistent ChromaDB.
    """
    # --- 1. Initialize Database Client ---
    logging.info(f"Initializing persistent ChromaDB client at: {DB_PATH}")
    # Use PersistentClient to save the database to disk
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

    # --- 3. Load Data from JSON File ---
    json_path = Path(PUBMED_JSON_FILE)
    if not json_path.is_file():
        logging.error(f"JSON file not found at '{PUBMED_JSON_FILE}'. Please create it.")
        return

    logging.info(f"Loading data from '{json_path.name}'...")
    with open(json_path, "r", encoding="utf-8") as f:
        pubmed_data = json.load(f)

    # --- 4. Prepare and Insert Data ---
    logging.info("Preparing documents for insertion into ChromaDB...")
    ids_to_insert = []
    documents_to_insert = []
    metadatas_to_insert = []

    for topic, articles in tqdm(pubmed_data.items(), desc="Processing Topics"):
        for article in articles:
            pmid = article.get("pmid")
            title = article.get("title", "")
            abstract = article.get("abstract", "")
            
            # Combine title and abstract for a richer document content
            text_content = f"{title}\n\n{abstract}"

            if not pmid or not text_content.strip():
                logging.warning(f"Skipping article with missing pmid or content for topic '{topic}'.")
                continue

            ids_to_insert.append(str(pmid))
            documents_to_insert.append(text_content)
            metadatas_to_insert.append({
                "topic": topic,
                "pmid": pmid,
                "title": title
            })
    
    # --- 5. Add to Collection in Batches ---
    if not ids_to_insert:
        logging.info("No new documents to insert.")
        return

    batch_size = 100 # Process 100 documents at a time
    logging.info(f"Adding {len(ids_to_insert)} documents to collection in batches of {batch_size}...")
    
    for i in tqdm(range(0, len(ids_to_insert), batch_size), desc="Inserting Batches"):
        batch_ids = ids_to_insert[i:i+batch_size]
        batch_documents = documents_to_insert[i:i+batch_size]
        batch_metadatas = metadatas_to_insert[i:i+batch_size]
        
        collection.add(
            ids=batch_ids,
            documents=batch_documents,
            metadatas=batch_metadatas
        )

    logging.info("✅ Data loading complete. The database is ready.")

if __name__ == "__main__":
    main()