import os
import chromadb
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).parent
ANALYSIS_DB_PATH = BASE_DIR / "analyses_db"
COLLECTION_NAME = "medical_analyses"

def clear_and_recreate_analyses_db():
    """Clear the existing analyses database and recreate it."""
    try:
        # Connect to ChromaDB
        client = chromadb.PersistentClient(path=str(ANALYSIS_DB_PATH))
        
        # Delete existing collection if it exists
        try:
            client.delete_collection(COLLECTION_NAME)
            logging.info(f"✅ Deleted existing collection: {COLLECTION_NAME}")
        except Exception as e:
            logging.info(f"No existing collection to delete: {e}")
        
        # Create new empty collection
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        logging.info(f"✅ Created new empty collection: {COLLECTION_NAME}")
        
        return True
        
    except Exception as e:
        logging.error(f"❌ Error clearing analyses database: {e}")
        return False

if __name__ == "__main__":
    print("🧹 Clearing existing analyses database...")
    if clear_and_recreate_analyses_db():
        print("✅ Database cleared successfully!")
        print("🚀 Now run: python improved_analysis_using_gemini.py")
        print("   to generate new evidence-based analyses.")
    else:
        print("❌ Failed to clear database.")
