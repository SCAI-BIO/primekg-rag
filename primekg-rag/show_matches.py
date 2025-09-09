import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

def load_matches(subgraph_name: str) -> List[Dict[str, Any]]:
    """Load pre-computed matches from JSON file."""
    # Ensure the subgraph name has the correct suffix
    if not subgraph_name.endswith('_subgraph'):
        subgraph_name = f"{subgraph_name}_subgraph"
    
    # Construct the file path
    matches_file = Path(__file__).parent / "paper_matches" / f"{subgraph_name}_matches.json"
    
    try:
        with open(matches_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: No matches file found at {matches_file}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {matches_file}")
        return []

def print_matches(matches: List[Dict[str, Any]]):
    """Print matches in a readable format."""
    if not matches:
        print("No matches found.")
        return
        
    print(f"\nFound {len(matches)} papers:")
    print("=" * 100)
    
    for i, match in enumerate(matches, 1):
        print(f"\n{i}. {match.get('title', 'No title')}")
        print(f"   PMID: {match.get('pmid', 'N/A')}")
        if 'similarity' in match:
            print(f"   Similarity: {match['similarity']:.3f}")
        abstract = match.get('abstract', 'No abstract available')
        print(f"   Abstract: {abstract}")
        print("-" * 100)

def main():
    parser = argparse.ArgumentParser(description='Display pre-computed paper matches for a subgraph')
    parser.add_argument('subgraph', 
                       help='Name of the subgraph (with or without _subgraph suffix)', 
                       default='personality disorder', 
                       nargs='?')
    parser.add_argument('--output', help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Load the matches
    matches = load_matches(args.subgraph)
    
    # Print the results
    print_matches(matches)
    
    # Save to file if requested
    if args.output and matches:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(matches, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")

if __name__ == "__main__":
    main()
