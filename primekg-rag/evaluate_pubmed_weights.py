import os
import logging
import numpy as np
import pandas as pd
import chromadb
from tqdm import tqdm
from typing import List, Dict, Tuple
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ollama

# Configuration
BASE_DIR = Path(__file__).parent
PUBMED_DB_PATH = str(BASE_DIR / "pubmed_db")
COLLECTION_NAME = "pubmed_abstracts"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = BASE_DIR / "weight_evaluation"
OUTPUT_DIR.mkdir(exist_ok=True)

# Weight combinations to test (title, abstract)
WEIGHT_COMBINATIONS = [
    (1.0, 0.0),   # Title only
    (0.8, 0.2),   # Current setting
    (0.5, 0.5),   # Equal weights
    (0.2, 0.8),   # More weight to abstract
    (0.0, 1.0)    # Abstract only
]

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize models
model = SentenceTransformer(MODEL_NAME)

def load_questions() -> List[Dict[str, str]]:
    """Return a list of test questions about mental health conditions."""
    return [
        {"question": "what is Major Depressive Disorder?", "topic": "MDD"},
        {"question": "What are the risk factors for developing Suicidal Behavior Disorder?", "topic": "Suicidal Behavior"},
        {"question": "How is Bipolar I Disorder different from Bipolar II Disorder?", "topic": "Bipolar Disorders"},
        {"question": "What are the diagnostic criteria for Panic Disorder?", "topic": "Panic Disorder"},
        {"question": "What is Social Anxiety Disorder?", "topic": "Social Anxiety"}
    ]

def get_collection():
    """Initialize and return the ChromaDB collection."""
    client = chromadb.PersistentClient(path=PUBMED_DB_PATH)
    return client.get_collection(COLLECTION_NAME)

def get_top_k_matches(collection, query_embedding: np.ndarray, title_embeddings: np.ndarray, 
                     abstract_embeddings: np.ndarray, title_weight: float, abstract_weight: float, k: int = 5) -> List[Dict]:
    """Retrieve top-k matching documents using weighted similarity."""
    # Calculate similarities
    title_sims = cosine_similarity(query_embedding, title_embeddings)[0]
    abstract_sims = cosine_similarity(query_embedding, abstract_embeddings)[0]
    
    # Combine with weights
    combined_scores = (title_weight * title_sims) + (abstract_weight * abstract_sims)
    
    # Get top-k indices
    top_k_indices = np.argsort(combined_scores)[-k:][::-1]
    
    return [{"index": i, "score": combined_scores[i]} for i in top_k_indices]

def evaluate_with_ai(question: str, document_text: str) -> float:
    """Use Ollama to evaluate relevance of document to question (0-1)."""
    try:
        prompt = f"""Analyze the relevance of the following document to the question and provide a relevance score between 0 and 1, 
        where:
        - 0.9-1.0: Perfectly relevant, directly answers the question
        - 0.7-0.89: Highly relevant, addresses most aspects of the question
        - 0.5-0.69: Somewhat relevant, contains related information
        - 0.3-0.49: Slightly relevant, only tangentially related
        - 0.1-0.29: Mostly irrelevant, minimal connection
        - 0.0: Completely irrelevant
        
        Question: {question}
        
        Document: {document_text[:2000]}
        
        Provide your response in this exact format: SCORE: X.XX
        Where X.XX is your relevance score between 0 and 1."""
        
        response = ollama.generate(
            model="llama3:latest",
            prompt=prompt,
            options={"temperature": 0.2}  # Slight randomness for more nuanced scores
        )
        
        # Extract numeric score using more robust parsing
        response_text = response['response'].strip()
        logger.debug(f"AI Response: {response_text}")
        
        # Try to find a number between 0 and 1 in the response
        import re
        match = re.search(r'0\.\d+|1\.0*', response_text)
        if match:
            score = float(match.group())
            return max(0.0, min(1.0, score))  # Clamp between 0 and 1
        return 0.5  # Default score if parsing fails
            
    except Exception as e:
        logger.error(f"AI evaluation failed: {e}")
        return 0.5  # Neutral score on error

def get_ai_weight_rationale(results_summary: pd.DataFrame) -> str:
    """Get AI's explanation for the weight analysis based on all results."""
    # Format the results for the prompt
    results_str = "\n".join(
        f"Weights {row['title_weight']:.1f}:{row['abstract_weight']:.1f} | "
        f"Avg. Cosine: {row['cosine_score_mean']:.3f} | "
        f"AI Relevance: {row['ai_relevance_mean']:.3f} ± {row['ai_relevance_std']:.3f}"
        for _, row in results_summary.iterrows()
    )
    
    prompt = f"""You are a retrieval system expert analyzing PubMed document retrieval performance.
    
    Below are the evaluation results for different weight combinations of title vs. abstract in document retrieval:
    {results_str}
    
    Please analyze these results and provide a concise 3-4 paragraph explanation that covers:
    1. Which weight combination performed best and why
    2. The trade-offs between using titles vs. abstracts in medical literature retrieval
    3. Recommendations for when to use different weight combinations
    
    Focus on the relationship between the weights and the evaluation metrics (cosine similarity and AI relevance).
    """
    
    try:
        response = ollama.generate(
            model="llama3:latest",
            prompt=prompt,
            options={"temperature": 0.3}
        )
        return response['response'].strip()
    except Exception as e:
        logger.error(f"Failed to get AI rationale: {e}")
        return "Rationale not available"

def main():
    # Load data
    questions = load_questions()
    collection = get_collection()
    
    # Get all documents
    docs = collection.get(include=["metadatas", "embeddings"])
    doc_ids = docs["ids"]
    titles = [m.get("title", "") for m in docs["metadatas"]]
    abstracts = [m.get("abstract", "") for m in docs["metadatas"]]
    title_embeddings = np.array(docs["embeddings"])
    
    # Pre-compute abstract embeddings
    logger.info("Computing abstract embeddings...")
    abstract_embeddings = model.encode(abstracts, show_progress_bar=True)
    
    results = []
    
    # Process each question
    for q_data in tqdm(questions, desc="Processing questions"):
        question = q_data["question"]
        q_embedding = model.encode([question])
        
        # Test each weight combination
        for title_w, abstract_w in WEIGHT_COMBINATIONS:
            # Get top matches using cosine similarity
            matches = get_top_k_matches(
                collection, q_embedding, title_embeddings, abstract_embeddings, 
                title_w, abstract_w, k=2  # Reduced to 2 for faster evaluation
            )
            
            # Evaluate each match with AI
            for match in matches:
                doc_idx = match["index"]
                doc_text = f"Title: {titles[doc_idx]}\nAbstract: {abstracts[doc_idx]}"
                
                # Get AI relevance score
                relevance = evaluate_with_ai(question, doc_text)
                
                results.append({
                    "question": question,
                    "topic": q_data["topic"],
                    "title_weight": title_w,
                    "abstract_weight": abstract_w,
                    "doc_id": doc_ids[doc_idx],
                    "title": titles[doc_idx],
                    "cosine_score": match["score"],
                    "ai_relevance": relevance
                })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "weight_evaluation_results.csv", index=False)
    
    # Calculate and print summary statistics
    summary = results_df.groupby(['title_weight', 'abstract_weight']).agg({
        'cosine_score': 'mean',
        'ai_relevance': ['mean', 'std', 'count']  # Add std and count for AI relevance
    }).reset_index()
    
    # Flatten multi-index columns
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
    
    print("\n=== Evaluation Results ===")
    print("Weights (T:A) | Avg. Cosine | AI Relevance (mean±std) | Samples")
    print("-" * 70)
    for _, row in summary.iterrows():
        print(f"{row['title_weight']:.1f}:{row['abstract_weight']:.1f}\t"
              f"{row['cosine_score_mean']:.3f}\t"
              f"{row['ai_relevance_mean']:.3f} ± {row['ai_relevance_std']:.3f}\t"
              f"{int(row['ai_relevance_count'])}")
    
    # Find best weight combination based on AI relevance
    best_row = summary.loc[summary['ai_relevance_mean'].idxmax()]
    print("\n=== Best Weight Combination ===")
    print(f"Best weights: Title={best_row['title_weight']:.1f}, Abstract={best_row['abstract_weight']:.1f}")
    print(f"Average AI Relevance: {best_row['ai_relevance_mean']:.3f} ± {best_row['ai_relevance_std']:.3f}")
    
    # Get comprehensive AI analysis of all results
    print("\n=== AI Analysis of Results ===")
    rationale = get_ai_weight_rationale(summary)
    print(f"\nRationale:\n{rationale}")
    
    # Save the rationale to the results file
    results_df['ai_rationale'] = rationale
    results_df.to_csv(OUTPUT_DIR / "weight_evaluation_results.csv", index=False)
    
    logger.info(f"Evaluation complete! Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
