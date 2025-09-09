import os
from pathlib import Path
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict, Any, Tuple
import json
import requests
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from collections import defaultdict
import numpy as np

# Download required NLTK data
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create file handler which logs even debug messages
fh = logging.FileHandler('subgraph_analysis.log')
fh.setLevel(logging.DEBUG)

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(ch)
logger.addHandler(fh)

logger.info("Script started with debug logging")

# Configuration
BASE_DIR = Path(__file__).parent.resolve()
SUBGRAPH_DIR = BASE_DIR / "new_subgraphs"
PUBMED_DB_PATH = BASE_DIR / "pubmed_db"
ANALYSIS_OUTPUT_DIR = BASE_DIR / "analyses"
ANALYSIS_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Model configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEEPSEEK_API_URL = "http://localhost:11434/api/generate"
TOP_K_PAPERS = 50

class SubgraphRetriever:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.chroma_client = chromadb.PersistentClient(path=str(PUBMED_DB_PATH))
        self.wordnet_cache = defaultdict(dict)  # Cache for WordNet lookups
        
        # Initialize or get collections
        try:
            self.pubmed_collection = self.chroma_client.get_collection("pubmed_abstracts")
        except ValueError:
            logger.error("PubMed collection not found. Please initialize the PubMed database first.")
            raise
            
        try:
            self.subgraph_collection = self.chroma_client.get_or_create_collection(
                name="subgraph_embeddings",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            logger.error(f"Failed to initialize subgraph collection: {e}")
            raise
    
    def embed_subgraph(self, file_path: Path) -> List[float]:
        """Generate embedding for a subgraph using only its title (filename)."""
        try:
            # Use only the filename (without extension) as the text to embed
            subgraph_title = file_path.stem
            logger.debug(f"Embedding subgraph title: {subgraph_title}")
            
            # Generate embedding for the title only
            return self.embedder.encode(subgraph_title, convert_to_tensor=False).tolist()
            
        except Exception as e:
            logger.error(f"Error embedding subgraph {file_path}: {e}")
            raise
    
    def store_subgraph_embedding(self, file_path: Path, embedding: List[float]):
        """Store subgraph embedding in ChromaDB."""
        try:
            subgraph_id = file_path.stem
            self.subgraph_collection.upsert(
                ids=[subgraph_id],
                embeddings=[embedding],
                metadatas=[{"source": str(file_path)}],
                documents=[subgraph_id]  # Store the subgraph ID as document content
            )
            logger.info(f"Stored embedding for {subgraph_id}")
        except Exception as e:
            logger.error(f"Error storing embedding for {file_path}: {e}")
            raise
    
    def get_semantic_similarity(self, term1: str, term2: str) -> float:
        """Calculate semantic similarity between two terms using WordNet."""
        if term1.lower() == term2.lower():
            return 1.0
            
        # Check cache first
        cache_key = (term1.lower(), term2.lower())
        if cache_key in self.wordnet_cache:
            return self.wordnet_cache[cache_key]
            
        # Tokenize and get synsets for each term
        def get_synsets(term):
            tokens = word_tokenize(term.lower())
            pos_tags = pos_tag(tokens)
            
            synsets = []
            for word, pos in pos_tags:
                # Map POS tag to WordNet POS tag
                wn_pos = None
                if pos.startswith('N'):
                    wn_pos = wn.NOUN
                elif pos.startswith('V'):
                    wn_pos = wn.VERB
                elif pos.startswith('J'):
                    wn_pos = wn.ADJ
                elif pos.startswith('R'):
                    wn_pos = wn.ADV
                    
                if wn_pos:
                    synsets.extend(wn.synsets(word, pos=wn_pos))
                else:
                    synsets.extend(wn.synsets(word))
            
            return synsets or [wn.synset('entity.n.01')]  # Default to entity if no synsets found
        
        synsets1 = get_synsets(term1)
        synsets2 = get_synsets(term2)
        
        if not synsets1 or not synsets2:
            return 0.0
            
        # Calculate maximum path similarity between all synset pairs
        max_sim = 0.0
        for s1 in synsets1:
            for s2 in synsets2:
                try:
                    sim = s1.path_similarity(s2) or 0.0
                    if sim > max_sim:
                        max_sim = sim
                except:
                    continue
        
        # Cache the result
        self.wordnet_cache[cache_key] = max_sim
        return max_sim

    def calculate_enhanced_similarity(self, subgraph_title: str, paper_title: str, base_similarity: float) -> float:
        """Enhance the base similarity score with semantic and exact match bonuses."""
        # Exact match bonus
        if subgraph_title.lower() in paper_title.lower():
            return min(1.0, base_similarity + 0.3)  # Cap at 1.0
            
        # Split into terms and calculate semantic similarity
        subgraph_terms = set(word_tokenize(subgraph_title.lower()))
        paper_terms = set(word_tokenize(paper_title.lower()))
        
        # Remove stopwords and short words
        stopwords = set(nltk.corpus.stopwords.words('english'))
        subgraph_terms = {t for t in subgraph_terms if t not in stopwords and len(t) > 2}
        paper_terms = {t for t in paper_terms if t not in stopwords and len(t) > 2}
        
        if not subgraph_terms or not paper_terms:
            return base_similarity
            
        # Calculate maximum semantic similarity between term pairs
        max_term_similarities = []
        for st in subgraph_terms:
            max_sim = max([self.get_semantic_similarity(st, pt) for pt in paper_terms] + [0])
            max_term_similarities.append(max_sim)
            
        # Calculate average of maximum similarities
        semantic_bonus = sum(max_term_similarities) / len(max_term_similarities) * 0.2  # Scale the bonus
        
        return min(1.0, base_similarity + semantic_bonus)

    def retrieve_similar_papers(self, subgraph_id: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve top-k similar papers for a subgraph with enhanced similarity scoring."""
        try:
            logger.info(f"Retrieving similar papers for {subgraph_id}")
            
            # Get the subgraph embedding
            results = self.subgraph_collection.get(
                ids=[subgraph_id],
                include=["embeddings", "metadatas"]
            )
            
            if results is None or 'embeddings' not in results or results['embeddings'] is None or len(results['embeddings']) == 0:
                logger.error(f"No embedding found for subgraph {subgraph_id}")
                return []
                
            subgraph_embedding = results['embeddings'][0]
            
            if subgraph_embedding is None:
                logger.error(f"Empty embedding for {subgraph_id}")
                return []
                
            if hasattr(subgraph_embedding, 'tolist'):
                subgraph_embedding = subgraph_embedding.tolist()
            
            # Get more results than needed to allow for re-ranking
            query_results = self.pubmed_collection.query(
                query_embeddings=[subgraph_embedding],
                n_results=min(top_k * 2, 20),  # Get more results for re-ranking
                include=["documents", "metadatas", "distances"]
            )
            
            if query_results is None or 'ids' not in query_results or not query_results['ids'] or len(query_results['ids']) == 0:
                logger.warning(f"No results found for {subgraph_id}")
                return []
            
            # Process and re-rank results
            papers = []
            for i in range(len(query_results['ids'][0])):
                try:
                    paper = {
                        'pmid': query_results['ids'][0][i],
                        'title': query_results['metadatas'][0][i].get('title', ''),
                        'abstract': query_results['documents'][0][i],
                        'similarity': 1.0 - query_results['distances'][0][i]  # Convert distance to similarity
                    }
                    
                    # Enhance similarity score based on title matching
                    paper['similarity'] = self.calculate_enhanced_similarity(
                        subgraph_id.replace('_subgraph', ''),  # Remove _subgraph suffix
                        paper['title'],
                        paper['similarity']
                    )
                    
                    papers.append(paper)
                except Exception as e:
                    logger.error(f"Error processing paper {i}: {e}")
                    continue
            
            # Sort by enhanced similarity and return top-k
            papers.sort(key=lambda x: x['similarity'], reverse=True)
            return papers[:top_k]
            
        except Exception as e:
            logger.error(f"Error in retrieve_similar_papers for {subgraph_id}: {str(e)}", exc_info=True)
            return []
    
    def retrieve_papers_by_query(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve papers directly using a text query."""
        try:
            # Generate embedding for the query
            query_embedding = self.embedder.encode(query, convert_to_tensor=False).tolist()
            
            # Query the PubMed collection
            results = self.pubmed_collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            papers = []
            for i in range(len(results['ids'][0])):
                paper = {
                    'pmid': results['ids'][0][i],
                    'title': results['metadatas'][0][i].get('title', ''),
                    'abstract': results['documents'][0][i],
                    'similarity': 1.0 - results['distances'][0][i],
                    'source': 'corpus'
                }
                papers.append(paper)
                
            return papers
            
        except Exception as e:
            logger.error(f"Error retrieving papers by query: {e}")
            return []

    def retrieve_combined_results(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Retrieve combined results from both corpus and subgraphs.
        Prioritizes corpus results over subgraph results.
        """
        # First, try to get papers from the corpus
        corpus_results = self.retrieve_papers_by_query(query, top_k=top_k)
        
        # If we got enough results from corpus, return them
        if len(corpus_results) >= top_k:
            return {
                'results': corpus_results[:top_k],
                'source': 'corpus_only',
                'total_found': len(corpus_results)
            }
        
        # If not enough corpus results, try to get subgraph results
        try:
            # Try to find a matching subgraph
            query_embedding = self.embedder.encode(query, convert_to_tensor=False).tolist()
            
            # Query subgraph collection
            subgraph_results = self.subgraph_collection.query(
                query_embeddings=[query_embedding],
                n_results=1,  # Get top matching subgraph
                include=["metadatas", "documents", "distances"]
            )
            
            if subgraph_results and 'ids' in subgraph_results and subgraph_results['ids']:
                best_subgraph_id = subgraph_results['ids'][0][0]
                subgraph_similarity = 1.0 - subgraph_results['distances'][0][0]
                
                # If we have a good enough match, get papers from this subgraph
                if subgraph_similarity > 0.5:  # Threshold can be adjusted
                    subgraph_papers = self.retrieve_similar_papers(
                        best_subgraph_id, 
                        top_k=top_k - len(corpus_results)
                    )
                    
                    # Mark subgraph papers with their source
                    for paper in subgraph_papers:
                        paper['source'] = 'subgraph'
                    
                    # Combine results (corpus first, then subgraph)
                    combined_results = corpus_results + subgraph_papers
                    
                    return {
                        'results': combined_results[:top_k],
                        'source': 'combined',
                        'corpus_count': len(corpus_results),
                        'subgraph_count': len(subgraph_papers),
                        'subgraph_id': best_subgraph_id,
                        'subgraph_similarity': subgraph_similarity
                    }
            
            # If no good subgraph match or other issues, just return corpus results
            return {
                'results': corpus_results,
                'source': 'corpus_only',
                'total_found': len(corpus_results)
            }
            
        except Exception as e:
            logger.error(f"Error in combined retrieval: {e}")
            return {
                'results': corpus_results,
                'source': 'corpus_only',
                'total_found': len(corpus_results),
                'error': str(e)
            }

    def analyze_with_deepseek(self, subgraph_id: str, papers: List[Dict[str, Any]], subgraph_path: Path) -> str:
        """Generate analysis using DeepSeek model."""
        try:
            # Read subgraph data
            df = pd.read_csv(subgraph_path)
            
            # Format the prompt
            prompt = self._build_analysis_prompt(subgraph_id, papers, df)
            
            # Call DeepSeek API
            response = requests.post(
                DEEPSEEK_API_URL,
                json={
                    "model": "deepseek-r1:14b",
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.1,
                    "max_tokens": 4000
                },
                timeout=300  # 5 minutes timeout
            )
            
            response.raise_for_status()
            return response.json().get("response", "")
            
        except Exception as e:
            logger.error(f"Error analyzing with DeepSeek: {e}")
            return f"Error generating analysis: {str(e)}"
    
    def _build_analysis_prompt(self, subgraph_id: str, papers: List[Dict[str, Any]], df: pd.DataFrame) -> str:
        """Build a detailed prompt for DeepSeek analysis."""
        # Format subgraph information
        nodes = set()
        if 'x_name' in df.columns:
            nodes.update(df['x_name'].dropna().unique())
        if 'y_name' in df.columns:
            nodes.update(df['y_name'].dropna().unique())
        
        relations = []
        if all(col in df.columns for col in ['x_name', 'relation', 'y_name']):
            relations = [f"{row['x_name']} - {row['relation']} - {row['y_name']}" 
                        for _, row in df[['x_name', 'relation', 'y_name']].dropna().iterrows()]
        
        # Format papers information
        papers_text = ""
        for i, paper in enumerate(papers, 1):
            papers_text += f"""
            Paper {i} (PMID: {paper['pmid']}):
            Title: {paper['title']}
            Abstract: {paper['abstract'][:500]}...
            Similarity Score: {paper['similarity']:.4f}
            """
        
        # Build the full prompt
        prompt = f"""You are a biomedical knowledge graph analyst. Your task is to analyze a subgraph about "{subgraph_id}" 
        in the context of relevant research papers. 
        
        SUBGRAPH INFORMATION:
        - Nodes: {', '.join(nodes)}
        - Relationships: {', '.join(relations)}
        
        RELEVANT RESEARCH PAPERS:
        {papers_text}
        
        TASKS:
        1. Provide a comprehensive analysis of the subgraph in the context of the provided research.
        2. Identify key relationships and their potential biological/clinical significance.
        3. Highlight any novel insights or connections suggested by the subgraph.
        4. Note any contradictions or gaps between the subgraph and research literature.
        5. Provide specific citations to papers when making claims (use PMID:XXX format).
        
        Format your response with clear sections and citations.
        """
        
        return prompt

def save_paper_matches(output_dir: Path, subgraph_name: str, papers: List[Dict[str, Any]]) -> Path:
    """Save paper matches to a JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{subgraph_name}_matches.json"
    
    if not papers:
        logger.warning(f"No papers to save for {subgraph_name}")
        return output_file
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(papers)} paper matches to {output_file}")
    except Exception as e:
        logger.error(f"Error saving paper matches for {subgraph_name}: {e}")
    
    return output_file

def process_all_subgraphs():
    """Process all subgraph files and save paper matches."""
    logger.info("Initializing SubgraphRetriever...")
    retriever = SubgraphRetriever()
    
    # Create output directory
    RESULTS_DIR = BASE_DIR / "paper_matches"
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    
    # Get all subgraph files
    subgraph_files = list(SUBGRAPH_DIR.glob("*.csv"))
    if not subgraph_files:
        logger.error(f"No subgraph files found in {SUBGRAPH_DIR}")
        return
    
    logger.info(f"Found {len(subgraph_files)} subgraph files to process")
    
    for i, file_path in enumerate(subgraph_files, 1):
        try:
            subgraph_id = file_path.stem
            logger.info(f"Processing {i}/{len(subgraph_files)}: {subgraph_id}")
            
            # Embed and store subgraph
            embedding = retriever.embed_subgraph(file_path)
            retriever.store_subgraph_embedding(file_path, embedding)
            
            # Retrieve and save similar papers
            papers = retriever.retrieve_similar_papers(subgraph_id)
            save_paper_matches(RESULTS_DIR, subgraph_id, papers)
            
            logger.info(f"✅ Completed {subgraph_id}")
            
        except Exception as e:
            logger.error(f"❌ Error processing {file_path.name}: {e}", exc_info=True)
            continue

def debug_chromadb():
    """Debug function to check ChromaDB collections."""
    try:
        logger.info("Checking ChromaDB...")
        client = chromadb.PersistentClient(path=str(PUBMED_DB_PATH))
        collections = client.list_collections()
        logger.info(f"Found {len(collections)} collections:")
        for col in collections:
            logger.info(f"- {col.name}: {col.count()} items")
            if col.name == "pubmed_abstracts":
                try:
                    sample = col.peek()
                    logger.info(f"  Sample items: {sample}")
                except Exception as e:
                    logger.error(f"  Error peeking collection: {e}")
    except Exception as e:
        logger.error(f"Error checking ChromaDB: {e}")

def search_knowledge(query: str, top_k: int = 10) -> Dict[str, Any]:
    """
    Search both corpus and subgraphs with a single query.
    Returns combined results with metadata.
    """
    retriever = SubgraphRetriever()
    return retriever.retrieve_combined_results(query, top_k)

def retrieve_from_pubmed(query: str, top_k: int = 10) -> Dict[str, Any]:
    """
    Retrieve papers directly from the PubMed database in ChromaDB.
    
    Args:
        query: The search query string
        top_k: Maximum number of results to return
        
    Returns:
        Dictionary containing:
        - results: List of matching papers with metadata
        - total_found: Total number of matches found
        - error: Error message if any
    """
    try:
        # Initialize the retriever
        retriever = SubgraphRetriever()
        
        # Get papers from the corpus
        papers = retriever.retrieve_papers_by_query(query, top_k=top_k)
        
        return {
            'results': papers,
            'total_found': len(papers),
            'status': 'success'
        }
        
    except Exception as e:
        logger.error(f"Error retrieving from PubMed: {str(e)}", exc_info=True)
        return {
            'results': [],
            'total_found': 0,
            'error': str(e),
            'status': 'error'
        }

def print_results(results: Dict[str, Any], query: str):
    """Pretty print the search results."""
    print(f"\nSearch results for: {query}")
    print("=" * 80)
    
    if 'error' in results:
        print(f"Error: {results['error']}")
        return
    
    if not results.get('results'):
        print("No results found.")
        return
    
    print(f"Found {results['total_found']} papers:\n")
    
    for i, paper in enumerate(results['results'], 1):
        print(f"{i}. {paper['title']}")
        print(f"   PMID: {paper['pmid']}")
        print(f"   Similarity: {paper['similarity']:.3f}")
        print(f"   Source: {paper.get('source', 'corpus')}")
        abstract_preview = paper['abstract'][:200] + ('...' if len(paper['abstract']) > 200 else '')
        print(f"   Abstract: {abstract_preview}")
        print("-" * 80)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Search PubMed database')
    parser.add_argument('--query', type=str, help='Search query', default='cancer treatment')
    parser.add_argument('--top-k', type=int, help='Number of results to return', default=5)
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
        debug_chromadb()
    
    # Perform the search
    results = retrieve_from_pubmed(args.query, top_k=args.top_k)
    
    # Print results
    print_results(results, args.query)
    
    # If no results, suggest checking the query or database
    if not results.get('results'):
        print("\nNo results found. Please check your query or database connection.")
        print("You can run with --debug flag to check the database status.")
