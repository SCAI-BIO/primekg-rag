import json
import argparse
import time
import os
from pathlib import Path
from typing import List, Dict, Any
import logging
from Bio import Entrez

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
BASE_DIR = Path(__file__).parent
PAPER_MATCHES_DIR = BASE_DIR / "paper_matches"
OUTPUT_DIR = BASE_DIR / "papers"
Entrez.email = "aemekkawi@example.com"  # Replace with your email

def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def fetch_pubmed_abstracts(pmids: List[str], batch_size: int = 5) -> Dict[str, Dict[str, str]]:
    """Fetch full abstracts from PubMed using Entrez API."""
    if not pmids:
        return {}
        
    papers = {}
    
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]
        logger.info(f"Fetching batch {i//batch_size + 1}/{(len(pmids)-1)//batch_size + 1}")
        
        try:
            handle = Entrez.efetch(
                db="pubmed",
                id=",".join(batch),
                retmode="xml",
                retmax=batch_size
            )
            records = Entrez.read(handle)
            handle.close()
            
            for record in records.get('PubmedArticle', []):
                try:
                    article = record['MedlineCitation']['Article']
                    pmid = str(record['MedlineCitation']['PMID'])
                    
                    # Get abstract text
                    abstract_text = ""
                    if 'Abstract' in article and 'AbstractText' in article['Abstract']:
                        if isinstance(article['Abstract']['AbstractText'], list):
                            abstract_text = " ".join(
                                str(section) for section in article['Abstract']['AbstractText']
                                if section and str(section).strip()
                            )
                        else:
                            abstract_text = str(article['Abstract']['AbstractText'])
                    
                    papers[pmid] = {
                        'title': str(article.get('ArticleTitle', '')),
                        'abstract': abstract_text,
                        'journal': str(article.get('Journal', {}).get('Title', '')),
                        'authors': [
                            {
                                'lastname': str(author.get('LastName', '')),
                                'forename': str(author.get('ForeName', '')),
                                'initials': str(author.get('Initials', ''))
                            }
                            for author in article.get('AuthorList', [])
                            if 'LastName' in author
                        ],
                        'pmid': pmid,
                        'doi': next(
                            (str(article_id) for article_id in record['PubmedData'].get('ArticleIdList', [])
                             if hasattr(article_id, 'attributes') and article_id.attributes.get('IdType') == 'doi'),
                            ''
                        )
                    }
                    
                except Exception as e:
                    logger.error(f"Error processing record: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error fetching batch: {e}")
            
        time.sleep(0.5)
        
    return papers

def process_matches_file(matches_file: Path):
    """Process a single matches file and save papers."""
    # Extract subgraph name from filename
    subgraph_name = matches_file.stem.replace('_matches', '').replace('_subgraph', '').replace('_', ' ').title()
    logger.info(f"\nProcessing subgraph: {subgraph_name}")
    
    try:
        with open(matches_file, 'r', encoding='utf-8') as f:
            matches = json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in {matches_file}")
        return []
    
    if not matches:
        logger.info(f"No matches found in {matches_file}")
        return []
    
    # Extract PMIDs and fetch full abstracts
    pmids = [str(match.get('pmid', '')) for match in matches if 'pmid' in match]
    logger.info(f"Found {len(pmids)} papers to process")
    
    paper_details = fetch_pubmed_abstracts(pmids)
    
    # Save each paper
    saved_files = []
    for match in matches:
        pmid = str(match.get('pmid', ''))
        if pmid in paper_details:
            paper_data = {
                **paper_details[pmid],
                'similarity_score': match.get('similarity', 0),
                'source_subgraph': matches_file.stem.replace('_matches', ''),
                'retrieval_date': time.strftime('%Y-%m-%d'),
                'topics': [subgraph_name]
            }
            
            # Create a safe filename
            safe_title = "".join(c if c.isalnum() or c in ' -_' else '_' for c in paper_data['title'])
            safe_title = safe_title[:100]
            filename = f"{pmid}_{safe_title}.json"
            filepath = OUTPUT_DIR / filename
            
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(paper_data, f, indent=2, ensure_ascii=False)
                saved_files.append(filepath)
                logger.debug(f"Saved: {filename}")
            except Exception as e:
                logger.error(f"Error saving paper {pmid}: {e}")
    
    return saved_files

def main():
    parser = argparse.ArgumentParser(description='Fetch and save papers from PubMed for subgraph matches')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Ensure output directory exists
    ensure_output_dir()
    
    # Process all match files
    all_saved = []
    for match_file in PAPER_MATCHES_DIR.glob("*_matches.json"):
        saved = process_matches_file(match_file)
        all_saved.extend(saved)
    
    logger.info(f"\nProcessing complete. Saved {len(all_saved)} papers in total.")

if __name__ == "__main__":
    main()
