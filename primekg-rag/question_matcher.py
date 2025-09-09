import os
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
import logging
from typing import List, Dict, Tuple
import numpy as np

# --- Configuration ---
BASE_DIR = Path(__file__).parent
QUESTIONS_CSV_PATH = BASE_DIR / "questions_for_mapping.csv"
NODE_DB_PATH = BASE_DIR / "node_db"
QUESTION_DB_PATH = BASE_DIR / "question_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
NODE_COLLECTION = "node_embeddings"
QUESTION_COLLECTION = "question_embeddings"

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class QuestionMatcher:
    def __init__(self):
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        self.node_client = None
        self.question_client = None
        
    def initialize_databases(self):
        """Initialize connections to both databases."""
        try:
            # Connect to node database
            self.node_client = chromadb.PersistentClient(path=str(NODE_DB_PATH))
            
            # Create/connect to question database
            self.question_client = chromadb.PersistentClient(path=str(QUESTION_DB_PATH))
            
            logger.info("Successfully connected to databases")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize databases: {e}")
            return False
    
    def load_questions(self) -> List[Dict]:
        """Load questions from CSV file."""
        if not QUESTIONS_CSV_PATH.exists():
            logger.error(f"Questions file not found: {QUESTIONS_CSV_PATH}")
            return []
            
        df = pd.read_csv(QUESTIONS_CSV_PATH, header=None, names=['question'])
        questions = df['question'].dropna().astype(str).tolist()
        logger.info(f"Loaded {len(questions)} questions")
        return [{'id': str(i), 'text': q} for i, q in enumerate(questions)]
    
    def embed_questions(self):
        """Embed questions and store in ChromaDB."""
        questions = self.load_questions()
        if not questions:
            return False
            
        try:
            # Create or get collection
            collection = self.question_client.get_or_create_collection(
                name=QUESTION_COLLECTION,
                embedding_function=self.embedding_fn
            )
            
            # Add questions to collection
            collection.upsert(
                ids=[q['id'] for q in questions],
                documents=[q['text'] for q in questions],
                metadatas=[{'source': 'question_mapping'} for _ in questions]
            )
            
            logger.info(f"Successfully embedded {len(questions)} questions")
            return True
            
        except Exception as e:
            logger.error(f"Failed to embed questions: {e}")
            return False
    
    def find_similar_nodes(self, question: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find most similar nodes to a question."""
        try:
            # Get question collection
            question_collection = self.question_client.get_collection(
                name=QUESTION_COLLECTION,
                embedding_function=self.embedding_fn
            )
            
            # Get node collection
            node_collection = self.node_client.get_collection(
                name=NODE_COLLECTION,
                embedding_function=self.embedding_fn
            )
            
            # Get question embedding
            question_embedding = self.embedding_fn([question])[0]
            
            # Get all node embeddings and metadata
            node_data = node_collection.get(include=["embeddings", "metadatas", "documents"])
            
            if not node_data["ids"]:
                logger.warning("No nodes found in the node collection")
                return []
            
            # Convert to numpy for efficient computation
            node_embeddings = np.array(node_data["embeddings"])
            query_embedding = np.array(question_embedding).reshape(1, -1)
            
            # Calculate cosine similarities
            similarities = np.dot(node_embeddings, query_embedding.T).flatten()
            
            # Get top matches
            top_indices = np.argsort(similarities)[::-1][:top_k]
            results = []
            
            for idx in top_indices:
                node_id = node_data["ids"][idx]
                node_name = node_data["documents"][idx] if node_data["documents"] else f"Node {node_id}"
                score = float(similarities[idx])
                results.append((node_name, score))
            
            return results
            
        except Exception as e:
            logger.error(f"Error finding similar nodes: {e}")
            return []
    
    def process_all_questions(self, output_file: str) -> bool:
        """Process all questions and save best matches to CSV."""
        questions = self.load_questions()
        if not questions:
            return False
        
        results = []
        for question in questions:
            similar_nodes = self.find_similar_nodes(question['text'])
            if similar_nodes:
                results.append({
                    'question': question['text'],
                    'best_match': similar_nodes[0][0],
                    'score': similar_nodes[0][1]
                })
        
        try:
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False)
            return True
        except Exception as e:
            logger.error(f"Failed to save results to CSV: {e}")
            return False

def main():
    """Main function to run the question matching pipeline."""
    # Initialize matcher
    matcher = QuestionMatcher()
    if not matcher.initialize_databases():
        print("Failed to initialize databases")
        return
    
    # Process all questions and save best matches to CSV
    output_file = "best_question_matches.csv"
    if matcher.process_all_questions(output_file):
        print(f"\nSuccess! Best matches saved to: {output_file}")
        
        # Display the first few rows of the generated CSV
        try:
            import pandas as pd
            df = pd.read_csv(output_file)
            print("\nFirst few matches:")
            print(df.head())
        except Exception as e:
            print(f"\nCould not display CSV preview: {e}")
    else:
        print("\nFailed to process questions")

if __name__ == "__main__":
    main()
