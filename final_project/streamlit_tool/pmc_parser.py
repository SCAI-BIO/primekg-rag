import logging
from Bio import Entrez
from tqdm import tqdm
import time
import json
import chromadb
from chromadb.utils import embedding_functions
import datetime
import glob
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer

# Configure Entrez
Entrez.email = "anamekawy@gmail.com"
Entrez.tool = "PubMedScraperTool"

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Get PMIDs for a given term
def fetch_pmids(term, retmax=1500):
    logging.info(f"Fetching up to {retmax} PMIDs for term: {term}")
    try:
        # Use XML format so Entrez.read() can parse it
        handle = Entrez.esearch(db="pubmed", term=term, retmax=retmax, retmode="xml")
        results = Entrez.read(handle)
        handle.close()
        pmids = results['IdList']
        logging.info(f"Successfully fetched {len(pmids)} papers for term: {term}")
        return pmids
    except Exception as e:
        logging.error(f"Error fetching PMIDs for term {term}: {str(e)}")
        return []

# Fetch abstracts given a list of PMIDs
def fetch_abstracts(pmids):
    abstracts = []
    for i in tqdm(range(0, len(pmids), 100), desc="Fetching Abstracts"):
        batch = pmids[i:i+100]
        try:
            handle = Entrez.efetch(db="pubmed", id=batch, retmode="xml")
            records = Entrez.read(handle)
            handle.close()

            for article in records.get("PubmedArticle", []):
                try:
                    article_data = article['MedlineCitation']['Article']
                    title = article_data.get('ArticleTitle', '')
                    abstract_texts = article_data.get('Abstract', {}).get('AbstractText', [''])
                    # AbstractText can be a list of parts; join if needed
                    if isinstance(abstract_texts, list):
                        abstract = ' '.join(str(part) for part in abstract_texts)
                    else:
                        abstract = str(abstract_texts)
                    abstracts.append({
                        "pmid": str(article['MedlineCitation']['PMID']),
                        "title": str(title),
                        "abstract": abstract
                    })
                except Exception as e:
                    logging.warning(f"Skipping article due to parse error: {e}")
        except Exception as e:
            logging.error(f"Error fetching batch: {e}")
        time.sleep(0.5)  # NCBI rate limit compliance
    return abstracts

# Update existing ChromaDB collection with new abstracts
def update_chromadb(abstracts, db_path="./pubmed_db", collection_name="pubmed_abstracts", batch_size=500, max_workers=4):
    """ Update existing ChromaDB collection with new abstracts efficiently.
    
    Args:
        abstracts: List of document dictionaries with 'pmid', 'title', 'abstract', and 'topic'
        db_path: Path to the ChromaDB database (default: "./pubmed_db")
        collection_name: Name of the collection to update (default: "pubmed_abstracts")
        batch_size: Number of documents to process in each batch (default: 500)
        max_workers: Number of parallel workers for processing (default: 4)
    """
    try:
        # Initialize ChromaDB client with the existing database
        client = chromadb.PersistentClient(path=db_path)
        
        # Load the sentence transformer model for embeddings
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Get the existing collection with the embedding function
        try:
            collection = client.get_collection(
                name=collection_name,
                embedding_function=model.encode
            )
            logging.info(f"Using existing collection: {collection_name}")
        except Exception as e:
            logging.error(f"Failed to access collection {collection_name}: {e}")
            logging.info("Please ensure the collection exists in the database.")
            return 0
            
        # Get all existing IDs
        existing_ids = set(collection.get(include=[], limit=100000)['ids'])
        
        # Filter out duplicates before processing
        unique_abstracts = []
        seen_pmids = set()
        duplicate_count = 0

        for doc in abstracts:
            doc_id = str(doc["pmid"])
            if doc_id in existing_ids or doc_id in seen_pmids:
                duplicate_count += 1
                continue
            seen_pmids.add(doc_id)
            unique_abstracts.append(doc)

        logging.info(f"Filtered out {duplicate_count} duplicate documents (already in DB or batch)")

        if not unique_abstracts:
            logging.info("No new documents to add.")
            return 0

        # Process documents in batches
        total_docs = len(unique_abstracts)
        if total_docs == 0:
            logging.warning("No documents to process")
            return 0
            
        logging.info(f"Processing {total_docs} documents in batches of {batch_size}")
        added_count = 0
        
        for i in range(0, total_docs, batch_size):
            batch = unique_abstracts[i:i + batch_size]
            batch_ids = [str(doc['pmid']) for doc in batch]
            
            # Skip if document with same ID already exists
            try:
                existing = collection.get(ids=batch_ids)
                if existing and len(existing['ids']) > 0:
                    existing_ids = set(existing['ids'])
                    batch = [doc for doc, doc_id in zip(batch, batch_ids) if doc_id not in existing_ids]
                    if not batch:  # All documents in batch already exist
                        logging.info(f"Skipping batch {i//batch_size + 1} - all documents already exist")
                        continue
                    batch_ids = [str(doc['pmid']) for doc in batch]
            except Exception as e:
                logging.warning(f"Error checking for existing documents: {e}")
            
            if not batch:  # Skip if no new documents in this batch
                continue
                
            # Generate embeddings for the batch
            batch_texts = [f"{doc['title']} {doc['abstract']}" for doc in batch]
            batch_embeddings = model.encode(batch_texts, show_progress_bar=True)
            
            # Prepare metadata
            batch_metadata = [
                {
                    'title': doc['title'],
                    'abstract': doc['abstract'],
                    'topic': doc.get("topic", ""),
                    'source': 'pubmed',
                    'pmid': str(doc['pmid'])
                } for doc in batch
            ]
            
            # Add to collection
            try:
                collection.upsert(
                    ids=batch_ids,
                    embeddings=batch_embeddings.tolist(),
                    metadatas=batch_metadata,
                    documents=batch_texts
                )
                added_count += len(batch)
                logging.info(f"Added {len(batch)} documents to collection")
            except Exception as e:
                logging.error(f"Error adding batch to collection: {e}")
            
            logging.info(f"Processed batch {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size}")
        
        logging.info(f"Added {added_count} new documents to the collection")
        return added_count
        
    except Exception as e:
        logging.error(f"Error in update_chromadb: {str(e)}")
        logging.exception("Full traceback:")
        return 0

# Main scraper
def scrape_pubmed_topics(topics):
    all_data = {}
    for topic in tqdm(topics, desc="Topics"):
        pmids = fetch_pmids(topic)
        if not pmids:
            logging.warning(f"No PMIDs found for topic '{topic}', skipping.")
            continue
        abstracts = fetch_abstracts(pmids)
        all_data[topic] = abstracts
        logging.info(f"Retrieved {len(abstracts)} abstracts for topic: {topic}")
    return all_data

def load_latest_json():
    """Load the most recent JSON file with PubMed data from the parent directory."""
    try:
        # Look for JSON files in the parent directory
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        json_files = glob.glob(os.path.join(parent_dir, 'pubmed_abstracts_full_*.json'))
        
        if not json_files:
            logging.error("No JSON files found in the parent directory.")
            return None
        
        # Get the most recent file
        latest_file = max(json_files, key=os.path.getmtime)
        logging.info(f"Loading data from {latest_file}")
        
        # Read the file with explicit encoding
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, dict):
            logging.error(f"Unexpected data format in {latest_file}. Expected a dictionary.")
            return None
            
        # Convert the data to the format expected by update_chromadb
        all_abstracts = []
        for topic, abstracts in data.items():
            if not isinstance(abstracts, list):
                logging.warning(f"Skipping topic '{topic}': expected list of abstracts, got {type(abstracts)}")
                continue
                
            for abstract in abstracts:
                if not isinstance(abstract, dict):
                    logging.warning(f"Skipping abstract in topic '{topic}': expected dictionary, got {type(abstract)}")
                    continue
                    
                # Ensure required fields are present
                if 'pmid' not in abstract:
                    logging.warning("Skipping abstract: missing 'pmid' field")
                    continue
                    
                # Add topic to the abstract
                abstract['topic'] = topic
                all_abstracts.append(abstract)
        
        if not all_abstracts:
            logging.error("No valid abstracts found in the JSON file.")
            return None
            
        logging.info(f"Successfully loaded {len(all_abstracts)} abstracts from {len(data)} topics")
        return all_abstracts
        
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON file {latest_file}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error loading JSON file: {e}", exc_info=True)
        return None

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process PubMed data')
    parser.add_argument('--download', action='store_true', help='Download new data from PubMed')
    args = parser.parse_args()
    
    if args.download:
        # Full search terms with enhanced keywords for better retrieval
        topics = [  
            # Trauma and stress-related
            'posttraumatic stress symptom AND (PTSD OR trauma-focused therapy OR EMDR)',
            'complex PTSD AND (treatment OR therapy OR intervention)',
            'acute stress disorder AND (management OR early intervention)',
            # Substance use and addiction
            'alcohol-related disorders AND (treatment OR rehabilitation OR medication)',
            'substance abuse/dependence AND (CBT OR MAT OR relapse prevention)',
            'drug/alcohol-induced mental disorder AND (treatment OR management)',
            'addiction AND (behavioral therapy OR medication-assisted treatment)',
            # Eating disorders
            'anorexia nervosa AND (treatment OR family-based therapy OR refeeding)',
            'bulimia nervosa AND (CBT OR SSRIs OR nutritional counseling)',
            'binge eating disorder AND (treatment OR therapy OR medication)',
            # Movement and tic disorders
            'tics AND (behavioral therapy OR medication OR habit reversal)',
            'Tourette syndrome AND (treatment OR management OR therapy)',
            'chronic tic disorder AND (intervention OR treatment)'
        ]
        
        # Scrape and save data
        all_data = scrape_pubmed_topics(topics)
        
        # Save to JSON with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"pubmed_abstracts_full_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(all_data, f, indent=2)
        logging.info(f"Saved data to {output_file}")
        
        # Prepare data for ChromaDB
        all_abstracts = []
        for topic, abstracts in all_data.items():
            for abstract in abstracts:
                abstract['topic'] = topic
                all_abstracts.append(abstract)
    else:
        # Load existing data
        all_abstracts = load_latest_json()
        if not all_abstracts:
            logging.error("No data to process. Exiting.")
            exit(1)
    
    # Update ChromaDB with the data
    if all_abstracts:
        num_added = update_chromadb(all_abstracts)
        logging.info(f"Processed {len(all_abstracts)} documents. Added {num_added} new documents to ChromaDB")
    print(f"Using existing data from: {latest_file}")
    logging.info(f"Loading data from existing file: {latest_file}")
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Calculate total unique papers
    unique_pmids = set()
    for topic_data in results.values():
        unique_pmids.update(paper['pmid'] for paper in topic_data)
    total_papers = len(unique_pmids)
    
    print(f"\n=== PubMed Data Summary ===")
    print(f"Total search terms: {len(topics)}")
    print(f"Total unique papers found: {total_papers}")
    print(f"Data loaded from: {latest_file}")
    print("\nTo update the data, run this script with --download flag")
    
    # Update ChromaDB with existing abstracts
    print("\nUpdating ChromaDB collection...")
    all_papers = [paper for papers in results.values() for paper in papers]
    num_added = update_chromadb(all_papers)
    print(f"Added/updated {num_added} documents in ChromaDB")
