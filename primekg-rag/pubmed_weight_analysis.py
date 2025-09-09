# pubmed_weight_analysis.py

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Set
from pathlib import Path

# Configure logging and display
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
pd.set_option('display.max_colwidth', 100)

# Configuration
BASE_DIR = Path(__file__).parent
DB_PATH = str(BASE_DIR / "pubmed_db")
COLLECTION_NAME = "pubmed_abstracts"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # Dimension for all-MiniLM-L6-v2
OUTPUT_DIR = BASE_DIR / "analysis_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Initialize the sentence transformer model
model = SentenceTransformer(MODEL_NAME)

# Weight combinations to test (title_weight, abstract_weight)
WEIGHT_COMBINATIONS = [
    (1.0, 0.0),   # Title only
    (0.9, 0.1),
    (0.8, 0.2),   # Current setting
    (0.7, 0.3),
    (0.5, 0.5),   # Equal weights
    (0.3, 0.7),
    (0.2, 0.8),
    (0.1, 0.9),
    (0.0, 1.0)    # Abstract only
]

def load_questions() -> List[str]:
    """Load questions from subgraph filenames, checking if files exist and contain data."""
    subgraph_dir = BASE_DIR / "new_subgraphs"
    questions = set()
    
    # Terms that shouldn't be used with "causes"
    non_causal_terms = {'tics', 'behavioral', 'abnormality'}
    
    # Get all subgraph files
    for file_path in subgraph_dir.glob("*_subgraph.csv"):
        # Skip 2-hop subgraphs
        if "2hop" in str(file_path).lower():
            continue
            
        # Verify file exists and has content
        if not file_path.is_file() or file_path.stat().st_size == 0:
            logger.warning(f"Skipping empty or missing file: {file_path}")
            continue
            
        # Extract the main topic from filename
        topic = file_path.stem.replace('_subgraph', '')
        topic_lower = topic.lower()
        
        # Skip treatment-related terms
        if any(term in topic_lower for term in ['treatment', 'therapy', 'medication']):
            continue
            
        # Verify the CSV has content
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                logger.warning(f"Skipping empty subgraph: {file_path}")
                continue
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            continue
            
        # Generate "What is X?" question
        what_is = f"What is {topic}?"
        questions.add(what_is)
        
        # Generate "What causes X?" question if appropriate
        if not any(term in topic_lower for term in non_causal_terms):
            what_causes = f"What causes {topic_lower}?"
            questions.add(what_causes)
    
    # Convert to list and sort for consistency
    questions = sorted(list(questions))
    logger.info(f"Generated {len(questions)} questions from subgraph files")
    return questions

def get_collection():
    """Initialize and return the ChromaDB collection."""
    client = chromadb.PersistentClient(path=DB_PATH)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODEL_NAME
    )
    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )

def calculate_weighted_similarity(
    query_emb: np.ndarray, 
    title_emb: np.ndarray, 
    abstract_emb: np.ndarray,
    title_weight: float,
    abstract_weight: float
) -> float:
    """Calculate weighted similarity score between query and document."""
    def safe_cosine(a, b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)
    
    title_sim = safe_cosine(query_emb, title_emb)
    abstract_sim = safe_cosine(query_emb, abstract_emb)
    return (title_weight * title_sim) + (abstract_weight * abstract_sim)

def evaluate_weight_combinations(collection, questions: List[str], k: int = 10) -> pd.DataFrame:
    """Evaluate different weight combinations using all-MiniLM-L6-v2 embeddings and cosine similarity."""
    results = []
    
    # Get all documents and their data
    data = collection.get(include=["metadatas", "documents"])
    documents = data["documents"]
    metadatas = data["metadatas"]
    
    # Pre-compute document embeddings for titles and abstracts
    logger.info("Computing document embeddings...")
    title_texts = []
    abstract_texts = []
    doc_info = []
    
    for doc, meta in zip(documents, metadatas):
        title = meta.get("title", "")
        abstract = meta.get("abstract", "")
        title_texts.append(title)
        abstract_texts.append(abstract)
        doc_info.append({
            "pmid": meta.get("pmid", ""),
            "title": title,
            "abstract": abstract
        })
    
    # Batch process title and abstract embeddings
    logger.info("Embedding titles...")
    title_embeddings = model.encode(title_texts, show_progress_bar=True, batch_size=32)
    
    logger.info("Embedding abstracts...")
    abstract_embeddings = model.encode(abstract_texts, show_progress_bar=True, batch_size=32)
    
    # Process each question
    logger.info("Processing questions...")
    for q_idx, question in enumerate(tqdm(questions, desc="Questions")):
        # Get question embedding
        query_embedding = model.encode([question])[0]
        
        # Calculate cosine similarity with all documents for each weight combination
        for title_weight, abstract_weight in tqdm(WEIGHT_COMBINATIONS, desc="Weight combinations", leave=False):
            if not np.isclose(title_weight + abstract_weight, 1.0):
                logger.warning(f"Skipping invalid weights: title={title_weight}, abstract={abstract_weight}")
                continue
                
            # Calculate weighted scores
            scores = []
            for i in range(len(doc_info)):
                # Calculate title and abstract similarities
                title_sim = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    title_embeddings[i].reshape(1, -1)
                )[0][0]
                
                abstract_sim = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    abstract_embeddings[i].reshape(1, -1)
                )[0][0]
                
                # Apply weights
                weighted_score = (title_weight * title_sim) + (abstract_weight * abstract_sim)
                scores.append((i, weighted_score))
            
            # Sort by score and get top-k
            scores.sort(key=lambda x: x[1], reverse=True)
            top_k = scores[:k]
            
            # Store results
            for rank, (doc_idx, score) in enumerate(top_k, 1):
                results.append({
                    "question_idx": q_idx,
                    "question": question,
                    "title_weight": title_weight,
                    "abstract_weight": abstract_weight,
                    "pmid": doc_info[doc_idx]["pmid"],
                    "title": doc_info[doc_idx]["title"],
                    "similarity_score": score,
                    "rank": rank
                })
    
    return pd.DataFrame(results)

def analyze_results(results_df: pd.DataFrame) -> Dict:
    """Analyze the retrieval results and return metrics.
    
    Args:
        results_df: DataFrame containing retrieval results with columns:
            - question_idx: Index of the question
            - question: The question text
            - title_weight: Weight given to title similarity
            - abstract_weight: Weight given to abstract similarity
            - pmid: PubMed ID of the retrieved document
            - title: Title of the retrieved document
            - similarity_score: Combined similarity score
            - rank: Rank of the document in the results (1-based)
            
    Returns:
        Dictionary containing:
            - precision_at_k: Precision at k for k=1,3,5,10
            - map: Mean Average Precision
    """
    analysis = {}
    
    # Group by weight combination and question
    grouped = results_df.groupby(["title_weight", "abstract_weight", "question_idx"])
    
    # Calculate precision@k for different k values
    for k in [1, 3, 5, 10]:
        # For each question and weight combination, count relevant docs in top-k
        precisions = results_df[results_df["rank"] <= k].groupby(
            ["title_weight", "abstract_weight", "question_idx"]
        ).size().reset_index(name='hits')
        
        # Calculate precision (hits/k) for each question
        precisions['precision'] = precisions['hits'] / k
        
        # Average precision across questions for each weight combination
        mean_precisions = precisions.groupby(
            ["title_weight", "abstract_weight"]
        )['precision'].mean()
        
        analysis[f"precision_at_{k}"] = mean_precisions
    
    # Calculate mean average precision (MAP)
    def calculate_ap(group):
        """Calculate Average Precision for a single question and weight combination."""
        # Sort documents by rank
        group = group.sort_values("rank")
        
        # Calculate precision at each rank
        precisions = []
        for k in range(1, 11):  # Up to rank 10
            # Precision@k = (# relevant docs in top k) / k
            prec_at_k = (group["rank"] <= k).mean()
            precisions.append(prec_at_k)
            
        # Average precision is the mean of precisions at each rank
        return np.mean(precisions) if precisions else 0.0
    
    # Calculate MAP across all questions for each weight combination
    analysis["map"] = results_df.groupby(
        ["title_weight", "abstract_weight", "question_idx"]
    ).apply(calculate_ap).groupby(level=[0, 1]).mean()
    
    return analysis

def plot_metrics(analysis: Dict):
    """Plot the evaluation metrics with clear visualizations.
    
    Args:
        analysis: Dictionary containing analysis results with metrics for each weight combination.
        
    Returns:
        DataFrame containing all metrics for further analysis.
    """
    # Convert to DataFrame for easier plotting
    metrics = []
    for title_weight, abstract_weight in WEIGHT_COMBINATIONS:
        # Skip if weights don't sum to 1 (shouldn't happen with current WEIGHT_COMBINATIONS)
        if not np.isclose(title_weight + abstract_weight, 1.0):
            continue
            
        row = {
            "title_weight": title_weight,
            "abstract_weight": abstract_weight,
            "weight_ratio": f"{title_weight:.1f}:{abstract_weight:.1f}"
        }
        
        # Add all metrics to the row
        for metric_name, metric_values in analysis.items():
            try:
                row[metric_name] = metric_values.get((title_weight, abstract_weight), 0)
            except (KeyError, AttributeError):
                # Handle case where metric_values is a Series with MultiIndex
                try:
                    row[metric_name] = metric_values.loc[(title_weight, abstract_weight)]
                except (KeyError, AttributeError):
                    row[metric_name] = 0
        metrics.append(row)
    
    metrics_df = pd.DataFrame(metrics)
    
    # Plot precision@k with improved styling
    plt.figure(figsize=(14, 8))
    
    # Colors for different k values
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, k in enumerate([1, 3, 5, 10]):
        plt.plot(
            metrics_df["weight_ratio"],
            metrics_df[f"precision_at_{k}"],
            marker='o',
            linestyle='-',
            linewidth=2,
            markersize=8,
            color=colors[i],
            label=f'P@{k}',
            alpha=0.9
        )
    
    plt.title('Precision@k for Different Title/Abstract Weight Combinations', 
              fontsize=16, pad=20)
    plt.xlabel('Title Weight : Abstract Weight', fontsize=14, labelpad=15)
    plt.ylabel('Precision', fontsize=14, labelpad=15)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title='k', fontsize=12, title_fontsize=12,
              bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Save high-quality figure
    plt.savefig(OUTPUT_DIR / 'precision_at_k.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot MAP with improved styling
    plt.figure(figsize=(14, 7))
    
    # Sort by MAP for better visualization
    map_df = metrics_df.sort_values('map', ascending=False)
    
    # Create bar plot
    bars = plt.bar(
        map_df["weight_ratio"], 
        map_df["map"],
        color='#2ecc71',
        alpha=0.8,
        edgecolor='#27ae60',
        linewidth=1.5
    )
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.01,
            f'{height:.3f}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    plt.title('Mean Average Precision (MAP) by Title/Abstract Weight Combination', 
              fontsize=16, pad=20)
    plt.xlabel('Title Weight : Abstract Weight', fontsize=14, labelpad=15)
    plt.ylabel('MAP', fontsize=14, labelpad=15)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, min(1.1, max(map_df["map"]) * 1.2))  # Add some padding
    plt.grid(True, linestyle='--', alpha=0.6, axis='y')
    plt.tight_layout()
    
    # Save high-quality figure
    plt.savefig(OUTPUT_DIR / 'map_scores.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return metrics_df

def main():
    """Main function to run the analysis."""
    logger.info("Starting PubMed weight analysis...")
    
    try:
        # Load data
        questions = load_questions()
        logger.info(f"Sample questions: {questions[:5]}...")
        
        collection = get_collection()
        
        # Run evaluation
        logger.info("Starting evaluation...")
        results_df = evaluate_weight_combinations(collection, questions)
        
        # Analyze results
        logger.info("Analyzing results...")
        analysis = analyze_results(results_df)
        
        # Plot and save results
        logger.info("Generating plots...")
        metrics_df = plot_metrics(analysis)
        
        # Save raw results
        results_df.to_csv(OUTPUT_DIR / 'raw_results.csv', index=False)
        metrics_df.to_csv(OUTPUT_DIR / 'metrics_summary.csv', index=False)
        
        logger.info(f"Analysis complete! Results saved to '{OUTPUT_DIR}'")
        logger.info(f"Optimal weights (highest MAP):")
        best = metrics_df.loc[metrics_df['map'].idxmax()]
        logger.info(f"  Title: {best['title_weight']:.1f}, Abstract: {best['abstract_weight']:.1f}")
        logger.info(f"  MAP: {best['map']:.4f}")
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()

# Create requirements.txt if it doesn't exist
if not (BASE_DIR / "requirements.txt").exists():
    with open(BASE_DIR / "requirements.txt", "w") as f:
        f.write("""chromadb>=0.3.0
sentence-transformers>=2.2.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
tqdm>=4.60.0
""")
