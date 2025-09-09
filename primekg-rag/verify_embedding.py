import chromadb
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def check_database(db_path):
    """Check a specific database location and return collection info."""
    try:
        db_path = Path(db_path).resolve()
        if not db_path.exists():
            logging.warning(f"Database not found at: {db_path}")
            return None
            
        logging.info(f"\nChecking database at: {db_path}")
        client = chromadb.PersistentClient(path=str(db_path))
        
        # Get all collections in this database
        collections = client.list_collections()
        
        if not collections:
            logging.info("No collections found in this database")
            return None
            
        results = {}
        for collection in collections:
            col = client.get_collection(name=collection.name)
            count = col.count()
            results[collection.name] = {
                'path': str(db_path),
                'count': count,
                'sample': col.get(limit=1)
            }
            logging.info(f"Found collection '{collection.name}' with {count:,} documents")
            
        return results
        
    except Exception as e:
        logging.error(f"Error checking database at {db_path}: {str(e)}")
        return None

def main():
    # Check all possible database locations
    db_paths = [
        "./pubmed_db",
        "../pubmed_db",
        "./primekg-rag/pubmed_db",
        "../primekg-rag/pubmed_db",
        str(Path.home() / ".cache/chroma")
    ]
    
    all_results = {}
    for path in db_paths:
        result = check_database(path)
        if result:
            all_results.update(result)
    
    # Print summary
    print("\n" + "="*80)
    print("DATABASE SUMMARY")
    print("="*80)
    
    if not all_results:
        print("No valid databases found!")
        return
    
    for col_name, data in all_results.items():
        print(f"\nCOLLECTION: {col_name}")
        print("-" * 50)
        print(f"Location: {data['path']}")
        print(f"Document count: {data['count']:,}")
        
        if data['count'] > 0 and 'sample' in data and data['sample'].get('documents'):
            print("\nSample document:")
            print(f"ID: {data['sample']['ids'][0]}")
            print(f"Title: {data['sample']['metadatas'][0].get('title', 'N/A')}")
            print(f"Topic: {data['sample']['metadatas'][0].get('topic', 'N/A')}")
            preview = data['sample']['documents'][0]
            print(f"Preview: {preview[:200]}..." if len(preview) > 200 else f"Preview: {preview}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
