import pandas as pd
import os
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
import hashlib

import os

# --- Configuration ---

# Get the absolute path of the directory where the script is located.
# This makes all other paths relative to the script, so it works on any computer.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the data source paths relative to the script's location.
# This assumes your CSV files are in the same directory as your script.
DATA_SOURCES = {
    "qa_facts": os.path.join(BASE_DIR, "mini_sample_cleaned.csv"),
    "nodes": os.path.join(BASE_DIR, "nodes.csv")
}

# Define the ChromaDB path relative to the script's location.
# This will create a folder named 'primekg_unified_db_asis' inside your project.
CHROMA_DB_PATH = os.path.join(BASE_DIR, 'primekg_unified_db_asis')
CHROMA_COLLECTION_NAME = 'unified_knowledge_asis'

def setup_database_asis():
    """
    Reads nodes and Q&A facts from CSVs "as-is" and embeds them into ChromaDB
    using the Cosine Similarity metric.
    """
    print("--- TASK 1: Starting Database Setup (As-Is with Cosine Similarity) ---")
    all_docs = []

    for source_name, file_path in DATA_SOURCES.items():
        if not os.path.exists(file_path):
            print(f"ERROR: File not found at '{file_path}'. Skipping.")
            continue
        
        print(f"🔎 Loading data from: {source_name} ({file_path})")
        df = pd.read_csv(file_path)
        df.dropna(inplace=True) # Remove rows with any missing values

        if source_name == 'qa_facts':
            # For the Q&A file, embed the 'Question', store 'Answer' in metadata
            df['document'] = df['Question']
            df['id'] = df.apply(lambda row: f"qa_{hashlib.md5(row['Question'].encode()).hexdigest()}", axis=1)
            df['metadata'] = df.apply(lambda row: {'source': source_name, 'question': row['Question'], 'answer': row['Answer']}, axis=1)
            
        elif source_name == 'nodes':
            # For the nodes file, embed the 'name'
            df['document'] = df['name']
            # Use the provided 'id' and prefix it to ensure uniqueness across sources
            df['id'] = df['id'].apply(lambda x: f"node_{x}")
            df['metadata'] = df.apply(lambda row: {'source': source_name, 'name': row['name'], 'type': row['type']}, axis=1)

        all_docs.append(df[['id', 'document', 'metadata']])

    # Combine all documents into a single DataFrame
    if not all_docs:
        print("ERROR: No data sources were loaded. Aborting.")
        return
        
    final_df = pd.concat(all_docs).drop_duplicates(subset='id').reset_index(drop=True)
    print(f"Found {len(final_df)} total unique documents to embed.")

    # Prepare data for ChromaDB
    documents = final_df['document'].tolist()
    ids = final_df['id'].tolist()
    metadatas = final_df['metadata'].tolist()

    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    # Get or create the collection, specifying cosine similarity
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"} # Use cosine distance
    )
    
    print(f"Adding {len(ids)} documents to ChromaDB...")
    # Add documents to the collection in batches for efficiency
    batch_size = 4096
    for i in tqdm(range(0, len(ids), batch_size), desc="Adding documents"):
        collection.add(
            ids=ids[i:i+batch_size], 
            documents=documents[i:i+batch_size], 
            metadatas=metadatas[i:i+batch_size]
        )

    print("\n--- Task 1 Complete: Unified Database is ready. ---")

if __name__ == "__main__":
    setup_database_asis()