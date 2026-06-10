"""
Fetch the gene-focused PubMed corpus for the TRD/MDD benchmark.

For each gene in features.json, this script searches PubMed for papers that
connect the gene to depression, antidepressants, mood disorders, treatment
response, or pharmacogenetics. The result is written as JSONL and later merged
with the broader TRD corpus by final_corpus.py.
"""

from collections import Counter
import json
import time

from pymed import PubMed


FEATURES_FILE = "features.json"
OUTPUT_FILE = "pubmed_gene_depression.jsonl"
MAX_RESULTS_PER_GENE = 200
EMAIL = "ahmedmekkawi97@gmail.com"

pubmed = PubMed(tool="RAG_Thesis_Tool", email=EMAIL)


def load_genes(features_file):
    """Load gene names from features.json."""
    with open(features_file, encoding="utf-8") as f:
        features = json.load(f)
    return [k for k, v in features.items() if v.get("type") == "gene"]


def fetch_papers_for_gene(gene, max_results=200):
    """
    Fetch depression and antidepressant-response papers for one gene.

    The query is intentionally broader than the final benchmark prompt. The
    downstream vector search narrows the evidence at retrieval time.
    """
    query = f"""
    ({gene}[Title/Abstract])
    AND (
        depression[Title/Abstract] OR
        depressive disorder[Title/Abstract] OR
        antidepressant[Title/Abstract] OR
        "mood disorder"[Title/Abstract] OR
        "treatment response"[Title/Abstract] OR
        "pharmacogenetics"[Title/Abstract]
    )
    NOT (
        cancer[Title/Abstract] OR
        carcinoma[Title/Abstract] OR
        tumor[Title/Abstract] OR
        oncology[Title/Abstract]
    )
    """

    print(f"\n[{gene}] Querying PubMed...")

    try:
        results = pubmed.query(query, max_results=max_results)

        papers = []
        for article in results:
            pmid_raw = article.pubmed_id

            if not pmid_raw:
                continue

            pmid = str(pmid_raw).strip().split()[0]

            if not pmid.isdigit():
                continue

            papers.append(
                {
                    "pmid": pmid,
                    "title": article.title or "",
                    "abstract": article.abstract or "",
                    "gene": gene,
                    "keywords": getattr(article, "keywords", None) or [],
                }
            )

        print(f"[{gene}] Retrieved {len(papers)} papers")
        return papers

    except Exception as exc:
        print(f"[{gene}] ERROR: {exc}")
        return []


def deduplicate_papers(papers):
    """Remove duplicate PMIDs, keeping the first occurrence."""
    seen = set()
    unique = []

    for paper in papers:
        pmid = paper["pmid"]
        if pmid not in seen:
            seen.add(pmid)
            unique.append(paper)

    return unique


def main():
    print("=" * 70)
    print("FETCHING GENE-DEPRESSION PAPERS")
    print("=" * 70)

    genes = load_genes(FEATURES_FILE)
    print(f"\nLoaded {len(genes)} genes: {', '.join(genes)}")

    all_papers = []

    for i, gene in enumerate(genes, 1):
        print(f"\n[{i}/{len(genes)}] Processing {gene}...")

        papers = fetch_papers_for_gene(gene, max_results=MAX_RESULTS_PER_GENE)
        all_papers.extend(papers)

        if i < len(genes):
            print("Rate limiting for PubMed (3 sec)...")
            time.sleep(3)

    print(f"\nTotal papers retrieved: {len(all_papers)}")
    unique_papers = deduplicate_papers(all_papers)
    print(f"After deduplication: {len(unique_papers)} unique papers")

    print(f"\nSaving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for paper in unique_papers:
            f.write(json.dumps(paper, ensure_ascii=False) + "\n")

    print("\nSUMMARY BY GENE")
    print("=" * 70)

    gene_counts = Counter(p["gene"] for p in unique_papers)
    for gene in genes:
        count = gene_counts[gene]
        status = "[OK]" if count > 20 else "[LOW]" if count > 5 else "[NONE]"
        print(f"{status} {gene:<10} {count:>4} papers")

    print("\nDone. Next: run final_corpus.py to combine and embed the corpus.")


if __name__ == "__main__":
    main()
