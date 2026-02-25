"""
Combine and Ingest Expanded Corpus

Combines:
1. pubmed_trd_new.jsonl (2,179 generic TRD papers)
2. pubmed_trd_extras.jsonl (184 gene-specific TRD papers)
3. pubmed_gene_depression.jsonl (NEW comprehensive gene papers)

Then ingests into ChromaDB for semantic retrieval.
"""

import json
import chromadb
from sentence_transformers import SentenceTransformer
from collections import Counter, defaultdict
from pathlib import Path

# Config
INPUT_FILES = [
    "pubmed_trd_new.jsonl",
    "pubmed_trd_extras.jsonl",
    "pubmed_gene_depression.jsonl",
    "pubmed_gene_aliases.jsonl"  # Alias search results
]

OUTPUT_JSON = "pubmed_final_corpus_v2.json"
CHROMA_DIR = "./chroma_expanded_corpus_v2"
COLLECTION_NAME = "pubmed_depression_expanded"

# Embedding model
MODEL_NAME = "all-MiniLM-L6-v2"


def load_and_combine_corpora(input_files):
    """Load all input files and deduplicate by PMID."""
    
    combined = {}
    source_counts = defaultdict(int)
    
    for filepath in input_files:
        if not Path(filepath).exists():
            print(f"Warning: {filepath} not found, skipping...")
            continue
        
        print(f"Loading {filepath}...")
        
        # Determine file format
        if filepath.endswith('.jsonl'):
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        pmid = str(rec.get('pmid', '')).strip()
                        
                        if pmid and pmid not in combined:
                            combined[pmid] = rec
                            source_counts[filepath] += 1
                    except json.JSONDecodeError:
                        continue
        
        elif filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                records = json.load(f)
                for rec in records:
                    pmid = str(rec.get('pmid', '')).strip()
                    if pmid and pmid not in combined:
                        combined[pmid] = rec
                        source_counts[filepath] += 1
    
    print(f"\nLoaded from sources:")
    for source, count in source_counts.items():
        print(f"  {source}: {count} papers")
    
    return list(combined.values())


def analyze_gene_coverage(corpus, genes):
    """Analyze how many papers mention each gene."""
    
    gene_coverage = {}
    
    for gene in genes:
        count = 0
        pmids = []
        
        for rec in corpus:
            full_text = (rec.get('title', '') + ' ' + rec.get('abstract', '')).lower()
            if gene.lower() in full_text:
                count += 1
                pmids.append(rec.get('pmid'))
        
        gene_coverage[gene] = {
            'count': count,
            'pmids': pmids[:10]  # Sample
        }
    
    return gene_coverage


def ingest_to_chromadb(corpus, chroma_dir, collection_name, model_name):
    """Ingest corpus into ChromaDB."""
    
    print(f"\nInitializing ChromaDB at {chroma_dir}...")
    print(f"Loading embedding model: {model_name}...")
    
    model = SentenceTransformer(model_name)
    client = chromadb.PersistentClient(path=chroma_dir)
    
    # Delete existing collection if exists
    try:
        client.delete_collection(name=collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except:
        pass
    
    collection = client.create_collection(name=collection_name)
    
    # Prepare data
    documents = []
    metadatas = []
    ids = []
    
    for rec in corpus:
        pmid = str(rec.get('pmid', '')).strip()
        title = rec.get('title', '')
        abstract = rec.get('abstract', '')
        
        if not pmid or not abstract:
            continue
        
        text = f"{title}\n\n{abstract}"
        
        documents.append(text)
        metadatas.append({
            'pmid': pmid,
            'source': 'pubmed',
            'gene_tag': rec.get('gene', 'general')  # If gene-specific
        })
        ids.append(pmid)
    
    print(f"\nEmbedding {len(documents)} documents...")
    
    embeddings = model.encode(
        documents,
        show_progress_bar=True,
        batch_size=256,
        convert_to_numpy=True
    )
    
    print(f"Adding to ChromaDB...")
    
    collection.add(
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"Saved {len(documents)} papers to ChromaDB")
    
    return collection


def test_retrieval(collection, test_queries):
    """Test retrieval quality with sample queries."""
    
    print(f"\n{'='*70}")
    print("TESTING RETRIEVAL")
    print("="*70)
    
    for query, expected_gene in test_queries:
        print(f"\nQuery: \"{query}\"")
        print(f"Expected gene: {expected_gene}")
        
        results = collection.query(
            query_texts=[query],
            n_results=5
        )
        
        if results and results['documents'] and results['documents'][0]:
            docs = results['documents'][0]
            metas = results['metadatas'][0]
            dists = results['distances'][0]
            
            gene_found = False
            for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), 1):
                title = doc.split('\n')[0][:80]
                gene_in_doc = expected_gene.lower() in doc.lower()
                
                if gene_in_doc:
                    gene_found = True
                
                print(f"  [{i}] PMID: {meta['pmid']}, Dist: {dist:.3f}, Has {expected_gene}: {'Yes' if gene_in_doc else 'No'}")
                print(f"      {title}...")
            
            status = "PASS" if gene_found else "FAIL"
            print(f"\n  {status}: Gene {expected_gene} {'found' if gene_found else 'NOT found'} in top 5")


def main():
    print("="*70)
    print("COMBINE AND INGEST EXPANDED CORPUS")
    print("="*70)
    
    # Load genes
    with open("features.json") as f:
        features = json.load(f)
    genes = [k for k, v in features.items() if v.get("type") == "gene"]
    
    # Combine corpora
    print(f"\nCombining corpora...")
    corpus = load_and_combine_corpora(INPUT_FILES)
    print(f"\nTotal unique papers: {len(corpus)}")
    
    # Save combined corpus
    print(f"\nSaving combined corpus to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)
    
    # Analyze gene coverage
    print(f"\n{'='*70}")
    print("GENE COVERAGE ANALYSIS")
    print("="*70)
    
    coverage = analyze_gene_coverage(corpus, genes)
    
    for gene in genes:
        count = coverage[gene]['count']
        pct = (count / len(corpus)) * 100
        status = "[OK]" if count > 20 else "[LOW]" if count > 5 else "[NONE]"
        print(f"{status} {gene:<10} {count:>4} papers ({pct:>5.2f}%)")
    
    # Ingest to ChromaDB
    print(f"\n{'='*70}")
    print("INGESTING TO CHROMADB")
    print("="*70)
    
    collection = ingest_to_chromadb(
        corpus,
        CHROMA_DIR,
        COLLECTION_NAME,
        MODEL_NAME
    )
    
    # Test retrieval
    test_queries = [
        ("MAOA gene treatment resistant depression", "MAOA"),
        ("DRD4 antidepressant response", "DRD4"),
        ("GPT2 depression", "GPT2"),
        ("KCND2 mood disorder", "KCND2")
    ]
    
    test_retrieval(collection, test_queries)
    
    print(f"\n{'='*70}")
    print("DONE!")
    print("="*70)
    print(f"\nExpanded corpus location: {CHROMA_DIR}")
    print(f"Collection name: {COLLECTION_NAME}")
    print(f"\nNext steps:")
    print(f"1. Update rag_pipeline.py to use new ChromaDB path")
    print(f"2. Re-run RAG pipeline with expanded corpus")
    print(f"3. Evaluate with all 16 genes")


if __name__ == "__main__":
    main()