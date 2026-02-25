import json
import logging
import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import chromadb


# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# Script assumed inside rag_results/
RESULTS_DIR = Path(r'C:\Users\aemekkawi\Documents\GitHub\matching-system\rag_results')
OUTPUT_DIR = BASE_DIR / "evaluation_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---- Load from environment ----
CHROMA_DIR = r"C:\Users\aemekkawi\Documents\GitHub\matching-system\chroma_data\pubmed_trd_mdd"
COLLECTION_NAME = 'pubmed_trd_mdd'
JUDGE_MODEL = "gpt-4o-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---- Validation ----
if not CHROMA_DIR:
    raise RuntimeError("CHROMA_DIR not set in .env")

if not COLLECTION_NAME:
    raise RuntimeError("COLLECTION_NAME not set in .env")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in .env")

client = OpenAI(api_key=OPENAI_API_KEY)
# =============================================================================
# PMID extraction helpers
# =============================================================================

PMID_PATTERN = re.compile(r"PMID[:\s]*(\d+)", flags=re.IGNORECASE)


def extract_pmids_from_text(text):
    if not text:
        return []
    return list(dict.fromkeys(PMID_PATTERN.findall(text)))


def normalize_pmids(xs):
    if not xs:
        return []
    found = []
    for x in xs:
        found.extend(re.findall(r"\d+", str(x)))
    return list(dict.fromkeys(found))


def safe_get(d, path, default=""):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# =============================================================================
# Load results
# =============================================================================

def load_all_results(results_dir):
    results = []

    for jf in sorted(results_dir.glob("rep_*/results_*.json")):
        data = json.loads(Path(jf).read_text(encoding="utf-8"))

        rep = int(jf.parent.name.replace("rep_", ""))

        for r in data:
            r["rep"] = rep

        results.extend(data)
        log.info(f"Loaded {len(data)} from {jf}")

    log.info(f"Total results: {len(results)}")
    return results


# =============================================================================
# Chroma connection + abstract fetch
# =============================================================================

def connect_chroma():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(name=COLLECTION_NAME)


def fetch_abstract(collection, pmid):
    """
    Tries common schema patterns:
      metadata pmid
      metadata PMID
      id lookup
    """

    try:
        res = collection.get(where={"pmid": pmid}, include=["documents"])
        if res["documents"]:
            return res["documents"][0]
    except:
        pass

    try:
        res = collection.get(where={"PMID": pmid}, include=["documents"])
        if res["documents"]:
            return res["documents"][0]
    except:
        pass

    try:
        res = collection.get(ids=[pmid], include=["documents"])
        if res["documents"]:
            return res["documents"][0]
    except:
        pass

    return None


# =============================================================================
# Judge — Binary support scoring
# =============================================================================

def judge_field_support(field_name, field_text, gene, abstracts):

    if not field_text or not abstracts:
        return 0

    query = (
        f"Is {gene} associated with antidepressant treatment response "
        f"or remission in major depressive disorder (MDD)?"
    )

    system = (
        "You are grading whether cited research papers support claims "
        "in a model output field.\n\n"
        f"Research Question:\n{query}\n\n"
        "Return STRICT JSON ONLY:\n"
        '{"score": 0 or 1}\n\n'
        "Score 1 if at least one abstract supports the field claims "
        "within this research scope.\n"
        "Score 0 otherwise."
    )

    user = {
        "gene": gene,
        "field_name": field_name,
        "field_text": field_text,
        "abstracts": abstracts,
    }

    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user)},
        ],
        temperature=0,
    )

    out = json.loads(resp.choices[0].message.content)
    return int(out.get("score", 0))


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(results, collection):

    rows = []

    for r in results:

        gene = r.get("gene", "")
        model = r.get("model", "")
        rep = r.get("rep", 0)

        parsed = r.get("parsed", {})

        answer_text = parsed.get("answer", "")
        clinical_text = parsed.get("clinical_relevance", "")

        # ---- cited pmids ----
        cited_answer = extract_pmids_from_text(answer_text)
        cited_struct = normalize_pmids(parsed.get("evidence_pmids", []))

        cited_pmids = list(dict.fromkeys(cited_answer + cited_struct))

        # ---- fetch abstracts ----
        abstracts = []

        for pmid in cited_pmids:
            abs_text = fetch_abstract(collection, pmid)
            if abs_text:
                abstracts.append({
                    "pmid": pmid,
                    "abstract": abs_text
                })

        # ---- judge fields ----
        if os.getenv("OPENAI_API_KEY") and abstracts:

            answer_support = judge_field_support(
                "answer",
                answer_text,
                gene,
                abstracts
            )

            clinical_support = judge_field_support(
                "clinical_relevance",
                clinical_text,
                gene,
                abstracts
            )

        else:
            answer_support = 0
            clinical_support = 0

        rows.append({
            "rep": rep,
            "gene": gene,
            "model": model,
            "num_cited": len(cited_pmids),
            "answer_support": answer_support,
            "clinical_relevance_support": clinical_support,
        })

    return rows


# =============================================================================
# Main
# =============================================================================

def main():

    if not RESULTS_DIR.exists():
        log.error("Results directory not found.")
        return

    if not os.getenv("OPENAI_API_KEY"):
        log.warning("OPENAI_API_KEY missing — support scoring skipped.")

    collection = connect_chroma()
    log.info("Connected to ChromaDB.")

    results = load_all_results(RESULTS_DIR)

    rows = evaluate(results, collection)

    df = pd.DataFrame(rows)

    out_csv = OUTPUT_DIR / "field_support_scores.csv"
    df.to_csv(out_csv, index=False)

    log.info(f"Saved → {out_csv}")


if __name__ == "__main__":
    main()