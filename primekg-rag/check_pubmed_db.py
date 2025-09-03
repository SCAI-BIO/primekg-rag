import chromadb
import json
from pathlib import Path

def inspect_pubmed_db():
    """Inspect the pubmed_db to see collections and their structure."""
    db_path = Path("./pubmed_db")
    if not db_path.exists():
        print(f"Database not found at {db_path}")
        return
    
    print(f"Inspecting ChromaDB at: {db_path}")
    client = chromadb.PersistentClient(path=str(db_path))
    
    collections = client.list_collections()
    print(f"\nFound {len(collections)} collections:")
    
    for collection in collections:
        print(f"\n{'='*50}")
        print(f"Collection: {collection.name}")
        print(f"{'='*50}")
        
        col = client.get_collection(collection.name)
        
        # Get count
        try:
            count = col.count()
            print(f"Total items: {count}")
        except Exception as e:
            print(f"Could not get count: {e}")
        
        # Get sample data
        try:
            sample = col.get(include=["documents", "metadatas", "embeddings"], limit=2)
            
            print(f"\nSample IDs: {sample.get('ids', [])[:2]}")
            
            docs = sample.get('documents', [])
            if docs:
                print(f"Sample documents (first 100 chars):")
                for i, doc in enumerate(docs[:2]):
                    print(f"  [{i}]: {str(doc)[:100]}...")
            
            metas = sample.get('metadatas', [])
            if metas:
                print(f"Sample metadata keys:")
                for i, meta in enumerate(metas[:2]):
                    if isinstance(meta, dict):
                        print(f"  [{i}]: {list(meta.keys())}")
                        # Show a sample of metadata values
                        for key, value in list(meta.items())[:3]:
                            val_str = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                            print(f"    {key}: {val_str}")
            
            embeddings = sample.get('embeddings', [])
            if embeddings:
                print(f"Embeddings info:")
                print(f"  Count: {len(embeddings)}")
                if embeddings:
                    print(f"  First embedding shape: {len(embeddings[0]) if embeddings[0] else 'None'}")
                    
        except Exception as e:
            print(f"Error getting sample data: {e}")

if __name__ == "__main__":
    inspect_pubmed_db()
