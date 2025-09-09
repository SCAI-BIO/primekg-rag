import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
import pandas as pd

# Configuration
BASE_DIR = Path(__file__).parent
QUESTION_DB_PATH = BASE_DIR / "question_db"
NODE_DB_PATH = BASE_DIR / "node_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Initialize embedding function
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)

def inspect_database(db_path: Path, collection_name: str, description: str):
    """Inspect and print database contents."""
    print(f"\n=== {description} ===")
    print(f"Database path: {db_path}")
    
    try:
        # Connect to the database
        client = chromadb.PersistentClient(path=str(db_path))
        
        # Get collection names
        collections = client.list_collections()
        print(f"Collections: {[c.name for c in collections]}")
        
        if not collections:
            print("No collections found.")
            return
            
        # Get the main collection
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_fn
        )
        
        # Get a sample of items
        items = collection.get(limit=5)
        
        # Print summary
        total_items = collection.count()
        print(f"Total items: {total_items}")
        
        if total_items > 0:
            print(f"\nAll {total_items} items:")
            # Get all items with embeddings
            items = collection.get(
                include=["documents", "metadatas", "embeddings"]
            )
            
            # Show all items with their embeddings
            for i, (doc_id, doc, metadata, embedding) in enumerate(zip(
                items['ids'],
                items['documents'],
                items['metadatas'],
                items['embeddings']
            )):
                print(f"\nItem {i+1}/{total_items}:")
                print(f"ID: {doc_id}")
                print(f"Content: {doc}")
                print(f"Metadata: {metadata}")
                print(f"Embedding (first 5 dims): {embedding[:5]}...")
                print(f"Total dimensions: {len(embedding)}")
        
    except Exception as e:
        print(f"Error inspecting database: {e}")

if __name__ == "__main__":
    # Inspect question database
    inspect_database(
        QUESTION_DB_PATH, 
        "question_embeddings",
        "Question Database"
    )
    
    # Inspect node database
    inspect_database(
        NODE_DB_PATH,
        "node_embeddings",
        "Node Database"
    )
    
    print("\nTo see more items, you can query the databases directly using ChromaDB's API.")
