import logging
from Bio import Entrez
from tqdm import tqdm
import time
import json
import chromadb
from chromadb.utils import embedding_functions

# Configure Entrez
Entrez.email = "anamekawy@gmail.com"
Entrez.tool = "PubMedScraperTool"

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Get PMIDs for a given term
def fetch_pmids(term, retmax=4000):
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
def update_chromadb(abstracts, db_path="./pubmed_db", collection_name="pubmed_abstracts"):
    """Update existing ChromaDB collection with new abstracts."""
    import chromadb
    from chromadb.utils import embedding_functions
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=db_path)
    
    # Define the embedding function with the correct model
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    try:
        # Try to get existing collection
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
        existing_ids = set(collection.get()["ids"])
        logging.info(f"Found existing collection with {len(existing_ids)} documents")
    except Exception as e:
        logging.info("Creating new collection")
        collection = client.create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
        existing_ids = set()
    
    # Prepare new documents to add
    documents_to_add = []
    metadatas_to_add = []
    ids_to_add = []
    
    for doc in abstracts:
        doc_id = doc["pmid"]
        if doc_id not in existing_ids:
            # Add the abstract as the main document content
            documents_to_add.append(doc["abstract"])
            
            # Prepare metadata matching existing structure
            metadata = {
                "title": doc["title"],
                "abstract": doc["abstract"],
                "pmid": doc_id,
                "topic": doc.get("topic", "")  # Use the topic from the document or empty string
            }
            
            metadatas_to_add.append(metadata)
            ids_to_add.append(doc_id)
    
    # Add new documents in batches
    batch_size = 100
    num_added = 0
    
    for i in range(0, len(ids_to_add), batch_size):
        batch_ids = ids_to_add[i:i+batch_size]
        batch_docs = documents_to_add[i:i+batch_size]
        batch_metas = metadatas_to_add[i:i+batch_size]
        
        collection.add(
            documents=batch_docs,
            metadatas=batch_metas,
            ids=batch_ids
        )
        num_added += len(batch_ids)
        logging.info(f"Added batch {i//batch_size + 1} with {len(batch_ids)} new documents")
    
    logging.info(f"Added {num_added} new documents to the collection")
    return num_added

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

# Example usage
if __name__ == "__main__":
    import datetime
    
    # Full search terms
    topics = [
        # Pharmacological treatments
        "pharmacotherapy AND major depressive disorder",
        "pharmacotherapy AND bipolar disorder",
        "pharmacotherapy AND schizophrenia",
        "antidepressant treatment AND major depressive disorder",
        "mood stabilizers AND bipolar disorder",
        "antipsychotic treatment AND schizophrenia",
        
        # Psychotherapeutic approaches
        "psychotherapy AND major depressive disorder",
        "psychotherapy AND bipolar disorder",
        "psychotherapy AND schizophrenia",
        "cognitive behavioral therapy AND major depressive disorder",
        "cognitive behavioral therapy AND bipolar disorder",
        "cognitive behavioral therapy AND schizophrenia",
        "family-focused therapy AND bipolar disorder",
        "family-focused therapy AND schizophrenia",
        "psychoeducation AND bipolar disorder",
        "psychoeducation AND schizophrenia",
        
        # Neuromodulation techniques
        "transcranial magnetic stimulation AND major depressive disorder",
        "transcranial magnetic stimulation AND bipolar disorder",
        "transcranial magnetic stimulation AND schizophrenia",
        "electroconvulsive therapy AND major depressive disorder",
        "electroconvulsive therapy AND bipolar disorder",
        "electroconvulsive therapy AND schizophrenia",
        "deep brain stimulation AND major depressive disorder",
        "deep brain stimulation AND bipolar disorder",
        "deep brain stimulation AND schizophrenia",
        "vagus nerve stimulation AND major depressive disorder",
        "vagus nerve stimulation AND bipolar disorder",
        "vagus nerve stimulation AND schizophrenia",
        
        # Novel and emerging treatments
        "ketamine treatment AND major depressive disorder",
        "ketamine treatment AND bipolar disorder",
        "esketamine treatment AND major depressive disorder",
        "psilocybin therapy AND major depressive disorder",
        "psilocybin therapy AND bipolar disorder",
        "psychedelic-assisted therapy AND major depressive disorder",
        "adjunctive anti-inflammatory treatment AND major depressive disorder",
        "adjunctive anti-inflammatory treatment AND bipolar disorder",
        "adjunctive anti-inflammatory treatment AND schizophrenia",
        "glutamate modulators AND major depressive disorder",
        "glutamate modulators AND bipolar disorder"
    ]
    
    # Create timestamp for output files
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"pubmed_abstracts_full_{timestamp}.json"
    
    logging.info(f"Starting full PubMed search with {len(topics)} terms...")
    results = scrape_pubmed_topics(topics)
    
    # Save results to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Count total unique papers
    total_papers = sum(len(papers) for papers in results.values())
    
    print(f"\nFull search completed!")
    print(f"Total search terms processed: {len(topics)}")
    print(f"Total unique papers found: {total_papers}")
    print(f"Results saved to: {output_file}")
    
    # Update ChromaDB with new abstracts
    print("\nUpdating ChromaDB collection...")
    num_added = update_chromadb([doc for topic_docs in results.values() for doc in topic_docs])
    print(f"Added {num_added} new documents to ChromaDB")
