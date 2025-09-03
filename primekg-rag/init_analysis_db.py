import os
import chromadb
from pathlib import Path
import pandas as pd

# Configuration
BASE_DIR = Path(__file__).parent
ANALYSES_DIR = BASE_DIR / "analyses"
ANALYSIS_DB_PATH = BASE_DIR / "analyses_db"
COLLECTION_NAME = "medical_analyses"

def get_analysis_files():
    """Get all analysis text files from the analyses directory."""
    if not ANALYSES_DIR.exists():
        print(f"Error: Analyses directory not found at {ANALYSES_DIR}")
        return []
    
    analysis_files = list(ANALYSES_DIR.glob("*.txt"))
    print(f"Found {len(analysis_files)} analysis files in {ANALYSES_DIR}")
    return analysis_files

def read_analysis_file(file_path):
    """Read the content of an analysis file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Extract the condition name from the filename
        # Format: "Condition_2hop_subgraph_analysis.txt"
        condition = file_path.stem.replace("_2hop_subgraph_analysis", "").replace("_", " ").title()
        
        return {
            "filename": file_path.name,
            "condition": condition,
            "content": content,
            "content_length": len(content)
        }
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def init_analysis_db():
    """Initialize the analysis database with analysis files."""
    try:
        # Get all analysis files
        analysis_files = get_analysis_files()
        if not analysis_files:
            print("No analysis files found. Exiting.")
            return False
        
        # Read all analysis files
        analyses = []
        for file_path in analysis_files:
            analysis = read_analysis_file(file_path)
            if analysis:
                analyses.append(analysis)
        
        if not analyses:
            print("No valid analyses to add to the database.")
            return False
            
        print(f"Successfully read {len(analyses)} analysis files.")
        
        # Create or connect to the ChromaDB
        client = chromadb.PersistentClient(path=str(ANALYSIS_DB_PATH))
        
        # Delete the existing collection if it exists
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection: {COLLECTION_NAME}")
        except Exception as e:
            print(f"No existing collection to delete: {e}")
        
        # Create a new collection
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Created new collection: {COLLECTION_NAME}")
        
        # Prepare documents for the database
        documents = []
        metadatas = []
        ids = []
        
        for i, analysis in enumerate(analyses):
            documents.append(analysis["content"])
            metadatas.append({
                "filename": analysis["filename"],
                "condition": analysis["condition"],
                "content_length": analysis["content_length"]
            })
            ids.append(f"analysis_{i+1}")
            print(f"Prepared analysis: {analysis['condition']} ({analysis['content_length']} chars)")
        
        # Add the documents to the collection
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Successfully added {len(analyses)} analyses to the database.")
        
        # Verify the data was added
        count = collection.count()
        print(f"Total analyses in database: {count}")
        
        return True
        
    except Exception as e:
        print(f"Error initializing analysis database: {e}")
        import traceback
        print(f"Error details: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("Initializing analysis database...")
    if init_analysis_db():
        print("Analysis database initialized successfully!")
    else:
        print("Failed to initialize analysis database.")
        exit(1)
