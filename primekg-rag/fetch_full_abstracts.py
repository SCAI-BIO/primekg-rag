import json
import argparse
import chromadb
from pathlib import Path
from typing import List, Dict, Any
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
BASE_DIR = Path(__file__).parent
PUBMED_DB_PATH = BASE_DIR / "pubmed_db"
PAPER_MATCHES_DIR = BASE_DIR / "paper_matches"

def get_chroma_collection():
    """Initialize and return the ChromaDB collection for PubMed abstracts."""
    try:
        client = chromadb.PersistentClient(path=str(PUBMED_DB_PATH))
        return client.get_collection("pubmed_abstracts")
    except Exception as e:
        logger.error(f"Error connecting to ChromaDB: {e}")
        raise

def fetch_full_abstracts(pmids: List[str]) -> Dict[str, Dict[str, str]]:
    """Fetch full abstracts from ChromaDB using PMIDs."""
    if not pmids:
        return {}
        
    try:
        collection = get_chroma_collection()
        results = collection.get(
            ids=pmids,
            include=["documents", "metadatas"]
        )
        
        # Combine results into a dictionary with PMID as key
        papers = {}
        if results and 'ids' in results:
            for i, pmid in enumerate(results['ids']):
                papers[pmid] = {
                    'abstract': results['documents'][i] if 'documents' in results and i < len(results['documents']) else '',
                    'title': results['metadatas'][i].get('title', '') if 'metadatas' in results and i < len(results['metadatas']) else ''
                }
        return papers
        
    except Exception as e:
        logger.error(f"Error fetching abstracts: {e}")
        return {}

def load_and_enrich_matches(subgraph_name: str) -> List[Dict[str, Any]]:
    """Load matches from JSON and enrich with full abstracts from ChromaDB."""
    # Ensure the subgraph name has the correct suffix
    if not subgraph_name.endswith('_subgraph'):
        subgraph_name = f"{subgraph_name}_subgraph"
    
    # Load the pre-computed matches
    matches_file = PAPER_MATCHES_DIR / f"{subgraph_name}_matches.json"
    try:
        with open(matches_file, 'r', encoding='utf-8') as f:
            matches = json.load(f)
    except FileNotFoundError:
        logger.error(f"No matches file found at {matches_file}")
        return []
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in {matches_file}")
        return []
    
    if not matches:
        return []
    
    # Extract PMIDs and fetch full abstracts
    pmids = [str(match.get('pmid', '')) for match in matches if 'pmid' in match]
    paper_details = fetch_full_abstracts(pmids)
    
    # Enrich matches with full abstracts
    enriched_matches = []
    for match in matches:
        pmid = str(match.get('pmid', ''))
        if pmid in paper_details:
            enriched_match = {
                **match,  # Keep original match data
                'full_abstract': paper_details[pmid].get('abstract', match.get('abstract', '')),
                'full_title': paper_details[pmid].get('title', match.get('title', ''))
            }
            enriched_matches.append(enriched_match)
    
    return enriched_matches

def print_enriched_matches(matches: List[Dict[str, Any]]):
    """Print enriched matches in a readable format."""
    if not matches:
        print("No matches found.")
        return
        
    print(f"\nFound {len(matches)} papers with full abstracts:")
    print("=" * 100)
    
    for i, match in enumerate(matches, 1):
        title = match.get('full_title') or match.get('title', 'No title')
        abstract = match.get('full_abstract') or match.get('abstract', 'No abstract available')
        
        print(f"\n{i}. {title}")
        print(f"   PMID: {match.get('pmid', 'N/A')}")
        if 'similarity' in match:
            print(f"   Similarity: {match['similarity']:.3f}")
        print(f"   Abstract: {abstract}")
        print("-" * 100)

def main():
    parser = argparse.ArgumentParser(description='Fetch full abstracts for papers in a subgraph')
    parser.add_argument('subgraph', 
                       help='Name of the subgraph (with or without _subgraph suffix)', 
                       default='personality disorder', 
                       nargs='?')
    parser.add_argument('--output', help='Save results to JSON file')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Load and enrich matches
    logger.info(f"Loading and enriching matches for: {args.subgraph}")
    matches = load_and_enrich_matches(args.subgraph)
    
    # Print results
    print_enriched_matches(matches)
    
    # Save to file if requested
    if args.output and matches:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(matches, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()
