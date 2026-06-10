import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import json

# Configuration
PUBMED_DB_PATH = "pubmed_db"
NODE_CSV = "best_question_matches.csv"
OUTPUT_DIR = "node_papers"
TOP_K = 10

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

class NodePaperRetriever:
    def __init__(self, db_path: str = PUBMED_DB_PATH):
        """Initialize the retriever with ChromaDB and sentence transformer."""
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_collection("pubmed_abstracts")
    
    def get_top_papers(self, node_name: str, top_k: int = TOP_K) -> list:
        """Retrieve top-k most similar papers for a given node name."""
        try:
            # Generate embedding for the node name
            query_embedding = self.model.encode([node_name], convert_to_numpy=True).tolist()
            
            # Query the collection with include=['documents', 'metadatas', 'distances']
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            papers = []
            for i in range(len(results['ids'][0])):
                # Get the abstract from documents if available, otherwise use the title
                abstract = results['documents'][0][i]
                title = results['metadatas'][0][i].get('title', '')
                
                # If abstract is empty or same as title, try to get it from metadata
                if not abstract or abstract == title:
                    abstract = results['metadatas'][0][i].get('abstract', '')
                
                paper = {
                    'rank': i + 1,
                    'pmid': results['ids'][0][i],
                    'title': title,
                    'abstract': abstract,
                    'similarity': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'publication_date': results['metadatas'][0][i].get('publication_date', ''),
                    'journal': results['metadatas'][0][i].get('journal', '')
                }
                papers.append(paper)
                
            return papers
            
        except Exception as e:
            print(f"Error retrieving papers for {node_name}: {str(e)}")
            return []

def main():
    # Initialize retriever
    retriever = NodePaperRetriever()
    
    # Read node names from CSV
    try:
        df = pd.read_csv(NODE_CSV)
        nodes = df['node_name'].unique()
        print(f"Found {len(nodes)} unique nodes in {NODE_CSV}")
    except Exception as e:
        print(f"Error reading {NODE_CSV}: {str(e)}")
        return
    
    # Process each node
    for node in tqdm(nodes, desc="Processing nodes"):
        # Skip empty node names
        if pd.isna(node) or not str(node).strip():
            continue
            
        # Get top papers
        papers = retriever.get_top_papers(node, TOP_K)
        
        if not papers:
            print(f"No papers found for node: {node}")
            continue
        
        # Create a safe filename from node name
        safe_node_name = "".join([c if c.isalnum() or c in ' -_' else '_' for c in str(node)]).strip()
        output_file = os.path.join(OUTPUT_DIR, f"{safe_node_name}.json")
        
        # Save to JSON
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'node': node,
                    'papers': papers
                }, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(papers)} papers for node: {node}")
        except Exception as e:
            print(f"Error saving papers for {node}: {str(e)}")

if __name__ == "__main__":
    main()
    print("\nProcessing complete. Check the 'node_papers' directory for output files.")
