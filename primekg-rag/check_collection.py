import chromadb
from pathlib import Path
import os
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def check_collection():
    """Check the number of documents in the ChromaDB collection."""
    # Possible database locations to check
    possible_paths = [
        "./pubmed_db",  # Current directory
        "../pubmed_db",  # Parent directory
        "./primekg-rag/pubmed_db",  # Subdirectory
        "../primekg-rag/pubmed_db",  # Parent's subdirectory
        str(Path.home() / ".cache/chroma")  # Default ChromaDB location
    ]
    
    collection_name = "pubmed_abstracts"
    found = False
    
    for db_path in possible_paths:
        try:
            path = Path(db_path).resolve()
            if not path.exists():
                logging.info(f"Database not found at: {path}")
                continue
                
            logging.info(f"\nChecking database at: {path}")
            client = chromadb.PersistentClient(path=str(path))
            
            # List all collections in this database
            collections = client.list_collections()
            logging.info(f"Found {len(collections)} collection(s) in this database")
            
            for collection_info in collections:
                try:
                    collection = client.get_collection(name=collection_info.name)
                    count = collection.count()
                    logging.info(f"\nCollection: {collection_info.name}")
                    logging.info(f"Location: {path}")
                    logging.info(f"Total documents: {count:,}")
                    
                    # Get metadata from first document if available
                    results = collection.get(limit=1)
                    if results.get('metadatas'):
                        logging.info("\nSample document metadata:")
                        for key, value in results['metadatas'][0].items():
                            logging.info(f"- {key}: {value}")
                    
                    found = True
                    
                except Exception as e:
                    logging.error(f"Error accessing collection {collection_info.name}: {str(e)}")
            
        except Exception as e:
            logging.error(f"Error checking database at {db_path}: {str(e)}")
    
    if not found:
        logging.error("\nNo valid ChromaDB collections found in any standard location.")
        logging.info("\nPlease check if you're running the script from the correct directory.")
        logging.info("The database should be in a 'pubmed_db' directory in your project root.")

if __name__ == "__main__":
    check_collection()
