"""
top_cited_pmids.py

For each gene, find the most-cited PMID across all models and reps,
fetch its title from PubMed, and write a summary text file.

Usage:
    python top_cited_pmids.py
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List
from collections import Counter

from Bio import Entrez

# ============================================================================
# Config
# ============================================================================

RESULTS_DIR = Path(r"C:\Users\aemekkawi\Documents\GitHub\matching-system\primekg-rag\primekg-rag\rag_results")
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "evaluation_results" / "top_cited_pmid_per_gene.txt"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

Entrez.email = "your_email@example.com"


# ============================================================================
# Helpers
# ============================================================================

def extract_pmids_from_text(text: str) -> List[str]:
    if not text:
        return []
    seen, out = set(), []
    for h in re.findall(r"PMID[:\s]*(\d+)", text, flags=re.IGNORECASE):
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


def get_combined_output_text(parsed: dict) -> str:
    if not isinstance(parsed, dict):
        return ""
    parts = [
        parsed.get("answer", ""),
        parsed.get("clinical_relevance", ""),
    ]
    kms = parsed.get("key_mechanisms", [])
    if isinstance(kms, list):
        for km in kms:
            if isinstance(km, dict):
                parts.append(km.get("description", ""))
    return " ".join(p for p in parts if p)


def load_all_results(results_dir: Path) -> list:
    results = []
    for jf in sorted(results_dir.glob("rep_*/results_*.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning(f"Skipping {jf}: {e}")
            continue
        rep = int(jf.parent.name.replace("rep_", ""))
        for r in data:
            r["rep"] = rep
        results.extend(data)
    log.info(f"Total results: {len(results)}")
    return results


def fetch_pubmed_titles(pmids: List[str]) -> Dict[str, str]:
    pmids = [str(p) for p in pmids if str(p).strip()]
    if not pmids:
        return {}
    title_map = {}
    try:
        handle = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="xml", retmode="text")
        records = Entrez.read(handle)
        handle.close()
        for article in records.get("PubmedArticle", []):
            try:
                medline = article["MedlineCitation"]
                pmid = str(medline["PMID"])
                title = article["MedlineCitation"]["Article"].get("ArticleTitle", "")
                if isinstance(title, list):
                    title = " ".join(map(str, title))
                title_map[pmid] = str(title).strip()
            except Exception:
                continue
    except Exception as e:
        log.warning(f"Failed to fetch PubMed titles: {e}")
    return title_map


# ============================================================================
# Main
# ============================================================================

def main():
    if not RESULTS_DIR.exists():
        log.error(f"Results dir not found: {RESULTS_DIR}")
        return

    results = load_all_results(RESULTS_DIR)
    if not results:
        log.error("No results found.")
        return

    # Group by gene, count all PMID citations across all models and reps
    gene_counters: Dict[str, Counter] = {}
    for r in results:
        gene = r.get("gene", "")
        if not gene:
            continue
        parsed = r.get("parsed")
        if not isinstance(parsed, dict):
            continue
        combined = get_combined_output_text(parsed)
        cited = extract_pmids_from_text(combined)
        if gene not in gene_counters:
            gene_counters[gene] = Counter()
        for pmid in cited:
            gene_counters[gene][pmid] += 1

    # Get top PMID per gene
    top_pmids = {}
    for gene in sorted(gene_counters.keys()):
        if gene_counters[gene]:
            pmid, count = gene_counters[gene].most_common(1)[0]
            top_pmids[gene] = (pmid, count)

    # Fetch titles for all top PMIDs in one batch
    all_top_pmid_ids = [pmid for pmid, _ in top_pmids.values()]
    title_map = fetch_pubmed_titles(all_top_pmid_ids)

    # Write output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("Top Cited PMID per Gene\n")
        f.write("=" * 100 + "\n")
        f.write(f"(across {len(set(r.get('model','') for r in results))} models × 10 reps)\n\n")

        for gene in sorted(top_pmids.keys()):
            pmid, count = top_pmids[gene]
            title = title_map.get(pmid, "[TITLE NOT FOUND]")
            f.write(f"{gene:10s}  PMID:{pmid}  cited {count} times\n")
            f.write(f"            {title}\n\n")

    log.info(f"Saved -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()