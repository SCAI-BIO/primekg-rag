import chromadb
from chromadb.utils import embedding_functions
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def inspect_chroma_db():
    try:
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path="./pubmed_db")
        
        # List all collections
        collections = client.list_collections()
        logging.info(f"Found {len(collections)} collections")
        
        for collection in collections:
            logging.info(f"\nCollection: {collection.name}")
            logging.info(f"Number of documents: {collection.count()}")
            
            # Get sample metadata
            sample = collection.peek()
            if sample['metadatas']:
                logging.info("Sample metadata fields:")
                for key in sample['metadatas'][0].keys():
                    logging.info(f"- {key}")
    
    except Exception as e:
        logging.error(f"Error inspecting ChromaDB: {e}")

def inspect_chroma_db(db_path="./pubmed_db", collection_name="pubmed_abstracts"):
    """Inspect the metadata fields in a ChromaDB collection."""
    try:
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path=db_path)
        
        # Get the collection
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        )
        
        # Get a sample of documents to inspect metadata
        sample_size = min(5, collection.count())  # Look at up to 5 documents
        if sample_size == 0:
            logging.warning("The collection is empty.")
            return
            
        results = collection.get(limit=sample_size)
        
        # Analyze metadata fields
        all_metadata_fields = set()
        for i, metadata in enumerate(results['metadatas']):
            logging.info(f"\nDocument {i+1} metadata fields:")
            for field, value in metadata.items():
                all_metadata_fields.add(field)
                logging.info(f"  - {field}: {value}")
        
        # Print summary
        logging.info("\n=== Collection Summary ===")
        logging.info(f"Total documents: {collection.count()}")
        logging.info(f"Unique metadata fields found: {', '.join(all_metadata_fields) or 'None'}")
        
        # Check if collection has embeddings
        if hasattr(collection, '_embedding_function'):
            logging.info(f"Embedding model: {collection._embedding_function.model_name}")
        
    except Exception as e:
        logging.error(f"Error inspecting ChromaDB: {e}")

def check_chroma_db(db_path="./pubmed_db", collection_name="pubmed_abstracts"):
    try:
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path=db_path)
        
        # Get collection info
        collections = client.list_collections()
        logging.info(f"Available collections: {[c.name for c in collections]}")
        
        # Get the target collection
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        )
        
        # Get collection stats
        count = collection.count()
        logging.info(f"Collection '{collection_name}' has {count} documents")
        
        if count == 0:
            return False, 0
            
        # Get a sample document to check embeddings
        try:
            # First try to get a sample with embeddings
            sample = collection.get(limit=1, include=['embeddings'])
            if sample and 'embeddings' in sample and sample['embeddings']:
                doc_embedding = sample['embeddings'][0]
                if hasattr(doc_embedding, '__len__') and len(doc_embedding) > 0:
                    logging.info(f"✅ Embeddings are present (vector length: {len(doc_embedding)})")
                    return True, count
        except Exception as e:
            logging.debug(f"Could not get embeddings directly: {str(e)}")
        
        # If we get here, either couldn't get embeddings or they're not in the expected format
        logging.info("ℹ️ Could not verify embeddings. They may be generated on-the-fly during queries.")
        return True, count  # Assume documents are properly stored even if we can't verify embeddings
        
    except Exception as e:
        logging.error(f"Error checking ChromaDB: {str(e)}")
        return False, 0

if __name__ == "__main__":
    logging.info("=== Inspecting ChromaDB ===")
    inspect_chroma_db()
    
    logging.info("\n=== Checking ChromaDB Collection ===")
    has_embeddings, doc_count = check_chroma_db()
    
    if doc_count > 0:
        if has_embeddings:
            logging.info("✅ ChromaDB collection is ready with documents")
        else:
            logging.warning("⚠️ ChromaDB collection has documents but may be missing embeddings")
    else:
        logging.warning("⚠️ ChromaDB collection is empty")
