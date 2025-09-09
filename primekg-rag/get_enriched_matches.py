import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
BASE_DIR = Path(__file__).parent
PAPER_MATCHES_DIR = BASE_DIR / "paper_matches"

def get_paper_details_from_pmids(pmids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Retrieve full paper details from ChromaDB using PMIDs.
    
    Args:
        pmids: List of PubMed IDs to look up
        
    Returns:
        Dictionary mapping PMIDs to paper details (title, abstract, etc.)
    """
    try:
        # Initialize ChromaDB client
        chroma_client = chromadb.PersistentClient(path=str(BASE_DIR / "chroma_db"))
        collection = chroma_client.get_collection("pubmed_abstracts")
        
        # Query by PMIDs
        results = collection.get(
            ids=pmids,
            include=["metadatas", "documents"]
        )
        
        # Map results by PMID
        paper_details = {}
        if results and 'ids' in results:
            for i, pmid in enumerate(results['ids']):
                paper_details[pmid] = {
                    'title': results['metadatas'][i].get('title', '') if 'metadatas' in results and i < len(results['metadatas']) else '',
                    'abstract': results['documents'][i] if 'documents' in results and i < len(results['documents']) else '',
                    'pmid': pmid
                }
        
        return paper_details
        
    except Exception as e:
        logger.error(f"Error fetching paper details: {e}")
        return {}

def get_enriched_matches(subgraph_name: str) -> List[Dict[str, Any]]:
    """
    Get pre-computed matches with full paper details.
    
    Args:
        subgraph_name: Name of the subgraph (with or without '_subgraph' suffix)
        
    Returns:
        List of enriched paper matches with full details
    """
    try:
        # Ensure the subgraph name has the correct suffix
        if not subgraph_name.endswith('_subgraph'):
            subgraph_name = f"{subgraph_name}_subgraph"
            
        # Load the matches
        matches_file = PAPER_MATCHES_DIR / f"{subgraph_name}_matches.json"
        if not matches_file.exists():
            logger.error(f"No matches file found at {matches_file}")
            return []
            
        with open(matches_file, 'r', encoding='utf-8') as f:
            matches = json.load(f)
            
        if not matches:
            return []
            
        # Get full details for each paper
        paper_details = get_paper_details_from_pmids([str(m.get('pmid', '')) for m in matches if 'pmid' in m])
        
        # Combine with original matches
        enriched_matches = []
        for match in matches:
            if 'pmid' in match and match['pmid'] in paper_details:
                enriched_match = {**match, **paper_details[match['pmid']]}
                enriched_matches.append(enriched_match)
        
        return enriched_matches
        
    except Exception as e:
        logger.error(f"Error enriching matches: {e}")
        return []

def print_enriched_matches(matches: List[Dict[str, Any]]):
    """Print enriched matches in a readable format."""
    if not matches:
        print("No matches found.")
        return
        
    print(f"\nFound {len(matches)} papers:")
    print("=" * 80)
    
    for i, match in enumerate(matches, 1):
        print(f"\n{i}. {match.get('title', 'No title')}")
        print(f"   PMID: {match.get('pmid', 'N/A')}")
        if 'similarity' in match:
            print(f"   Similarity: {match['similarity']:.3f}")
        abstract = match.get('abstract', 'No abstract available')
        print(f"   Abstract: {abstract[:200]}{'...' if len(abstract) > 200 else ''}")
        print("-" * 80)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Retrieve enriched paper matches for a subgraph')
    parser.add_argument('subgraph', help='Name of the subgraph (with or without _subgraph suffix)', 
                       default='personality disorder', nargs='?')
    parser.add_argument('--output', help='Save results to JSON file')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Get the enriched matches
    print(f"Retrieving enriched matches for: {args.subgraph}")
    matches = get_enriched_matches(args.subgraph)
    
    # Print the results
    print_enriched_matches(matches)
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(matches, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")

if __name__ == "__main__":
    main()
