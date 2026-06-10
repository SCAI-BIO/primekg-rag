"""
pmid_citation_analysis.py

Per-gene heatmap: model × PMID citation frequency.
Each cell = how many times (out of 10 reps) that model cited that PMID.

Also writes one TXT file per gene with:
PMID: paper title

Usage:
    python pmid_citation_analysis.py
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from Bio import Entrez

# ============================================================================
# Config
# ============================================================================

RESULTS_DIR = Path(r"C:\Users\aemekkawi\Documents\GitHub\matching-system\primekg-rag\primekg-rag\rag_results")
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "evaluation_results" / "heatmaps"
PMID_TXT_DIR = BASE_DIR / "evaluation_results" / "pmid_titles"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# put your email here for PubMed Entrez
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
            r["_source_file"] = str(jf)
        results.extend(data)
        log.info(f"Loaded {len(data)} from {jf}")
    log.info(f"Total results: {len(results)}")
    return results


def fetch_pubmed_titles(pmids: List[str]) -> Dict[str, str]:
    """
    Fetch PMID -> title from PubMed using Entrez efetch.
    """
    pmids = [str(p) for p in pmids if str(p).strip()]
    if not pmids:
        return {}

    title_map = {}

    try:
        handle = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="xml", retmode="text")
        records = Entrez.read(handle)
        handle.close()

        articles = records.get("PubmedArticle", [])
        for article in articles:
            try:
                medline = article["MedlineCitation"]
                pmid = str(medline["PMID"])
                article_data = medline["Article"]
                title = article_data.get("ArticleTitle", "")
                if isinstance(title, list):
                    title = " ".join(map(str, title))
                title_map[pmid] = str(title).strip()
            except Exception:
                continue

    except Exception as e:
        log.warning(f"Failed to fetch PubMed titles: {e}")

    return title_map


def sanitize_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]+', "_", name)


# ============================================================================
# TXT export: PMID -> Title per gene
# ============================================================================

def save_pmid_titles_per_gene(results: list, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    by_gene: Dict[str, list] = defaultdict(list)
    for r in results:
        gene = r.get("gene", "")
        if gene:
            by_gene[gene].append(r)

    print("Extracted genes from results:", sorted(by_gene.keys()))
    print("Number of extracted genes:", len(by_gene))

    for gene, gene_results in sorted(by_gene.items()):
        # collect all unique retrieved_pmids for this gene across rows
        pmids = []
        seen = set()
        for r in gene_results:
            for pmid in r.get("retrieved_pmids", []) or []:
                pmid = str(pmid)
                if pmid not in seen:
                    seen.add(pmid)
                    pmids.append(pmid)

        if not pmids:
            log.info(f"Skipping PMID title export for {gene}: no PMIDs found")
            continue

        title_map = fetch_pubmed_titles(pmids)

        out_path = output_dir / f"{sanitize_filename(gene)}_pmid_titles.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"Gene: {gene}\n")
            f.write("=" * 100 + "\n\n")
            for pmid in pmids:
                title = title_map.get(pmid, "[TITLE NOT FOUND]")
                f.write(f"{pmid}: {title}\n")

        log.info(f"Saved PMID titles for {gene} -> {out_path}")


# ============================================================================
# Heatmap
# ============================================================================

def plot_pmid_citation_heatmaps(results: list, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by gene
    by_gene: Dict[str, list] = {}
    for r in results:
        gene = r.get("gene", "")
        if gene:
            by_gene.setdefault(gene, []).append(r)

    for gene in sorted(by_gene.keys()):
        print(gene)

    for gene, gene_results in sorted(by_gene.items()):
        # collect all unique retrieved PMIDs for this gene across all rows
        pmids = []
        seen = set()
        for r in gene_results:
            for p in r.get("retrieved_pmids", []) or []:
                p = str(p)
                if p not in seen:
                    seen.add(p)
                    pmids.append(p)

        if not pmids:
            log.info(f"Skipping {gene}: no retrieved_pmids")
            continue

        models = sorted(set(r.get("model", "") for r in gene_results))

        freq = {m: {p: 0 for p in pmids} for m in models}

        for r in gene_results:
            model = r.get("model", "")
            parsed = r.get("parsed")
            if not isinstance(parsed, dict):
                continue
            combined = get_combined_output_text(parsed)
            cited = extract_pmids_from_text(combined)
            for pmid in cited:
                if pmid in freq[model]:
                    freq[model][pmid] += 1

        matrix = np.array([[freq[m][p] for p in pmids] for m in models])

        short_models = []
        for m in models:
            short = (m.split(":")[0]
                     .replace("deepseek-r1", "DS")
                     .replace("qwen3", "QW")
                     .replace("llama3.2", "LL3.2")
                     .replace("llama3.1", "LL3.1"))
            size = m.split(":")[1] if ":" in m else ""
            short_models.append(f"{short}:{size}")

        fig, ax = plt.subplots(figsize=(max(10, len(pmids) * 0.9), max(5, len(models) * 0.5)))
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=10)

        for i in range(len(models)):
            for j in range(len(pmids)):
                val = int(matrix[i, j])
                color = "white" if val > 5 else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")

        ax.set_xticks(range(len(pmids)))
        ax.set_xticklabels(pmids, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(short_models, fontsize=8)
        ax.set_xlabel("PMID")
        ax.set_ylabel("Model")
        ax.set_title(f"PMID Citation Frequency — {gene}\n(count out of 10 reps)", fontsize=11)

        fig.colorbar(im, ax=ax, label="Times cited (out of 10 reps)", shrink=0.8)
        fig.tight_layout()

        out = output_dir / f"heatmap_pmid_{sanitize_filename(gene)}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        log.info(f"Saved {out}")


# ============================================================================
# Main
# ============================================================================

def main():
    if not RESULTS_DIR.exists():
        log.error(f"Results dir not found: {RESULTS_DIR}")
        return

    print("RUNNING FILE:", Path(__file__).resolve())
    print("RESULTS_DIR:", RESULTS_DIR.resolve())

    results = load_all_results(RESULTS_DIR)
    if not results:
        log.error("No results found.")
        return

    genes = sorted({r.get("gene") for r in results if r.get("gene")})
    print("Unique genes count:", len(genes))
    print("Unique genes:", genes)

    plot_pmid_citation_heatmaps(results, OUTPUT_DIR)
    save_pmid_titles_per_gene(results, PMID_TXT_DIR)

    log.info("Done.")


if __name__ == "__main__":
    main()