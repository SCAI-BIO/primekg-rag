import json
import time
import logging
import re
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests
import chromadb

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ============================================================================
# Config
# ============================================================================

SHAP_GENES = [
    "MAOA", "DRD4", "HTR2C", "ADRA1A", "ADRA2A",
    "OPRM1", "OPRK1", "KCND2", "KCND3", "KCNQ2",
    "KCNQ3", "CHRNA3", "CHRNB4", "CHRM1", "ABAT", "GPT2"
]

MODELS = [
    {"name": "qwen3:8b", "size": "8b"},
    {"name": "qwen3:14b", "size": "14b"},
]

PATHWAYS_CSV = "pathways.csv"
CHROMA_DIR = r"C:\Users\aemekkawi\Documents\GitHub\matching-system\pubmed_trd_mdd"
COLLECTION_NAME = "pubmed_trd_mdd"
OUTPUT_DIR = "./rag_results"

N_ABSTRACTS = 10
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT = 300
MAX_RETRIES = 2

# ---------------------------------------------------------------------------
# Question-as-query templates (exactly 3 per feature)
# ---------------------------------------------------------------------------
QUESTION_TEMPLATES = [
    # Q1: general association (broad recall)
    "What is the role or association of {feature} in major depressive disorder (MDD)?",

    # Q2: treatment resistance (core outcome)
    "What is the effect of {feature} on treatment-resistant depression or poor treatment response in major depressive disorder (MDD)?",

    # Q3: treatment response/remission (predictor framing)
    "Is {feature} associated with antidepressant treatment response or remission in major depressive disorder (MDD)?",
]


# ============================================================================
# Logging
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)


# ============================================================================
# Helpers
# ============================================================================

def build_feature_questions(feature: str):
    """Return exactly 3 natural-language questions for a feature (question-as-query)."""
    feature = (feature or "").strip()
    return [t.format(feature=feature) for t in QUESTION_TEMPLATES]


def gene_mentioned(gene: str, text: str) -> bool:
    """
    Hard filter: keep only docs that explicitly mention the gene.
    Uses word boundaries to reduce accidental matches.
    """
    if not gene or not text:
        return False
    pattern = re.compile(rf"\b{re.escape(gene)}\b", flags=re.IGNORECASE)
    return pattern.search(text) is not None


# ============================================================================
# Data loading
# ============================================================================

def load_pathways(csv_file):
    """Load gene pathways from neo4j export."""
    log.info(f"Loading pathways from {csv_file}")
    df = pd.read_csv(csv_file)

    pathways = {}
    for gene in SHAP_GENES:
        rows = df[df["gene"] == gene]
        paths = []
        for _, r in rows.iterrows():
            paths.append({
                "intermediate": r.get("intermediate_node", r.get("intermediate", "Unknown")),
                "type": r.get("intermediate_type", "Unknown"),
                "rel1": r.get("relationship_1", "related_to"),
                "rel2": r.get("relationship_2", "associated_with"),
            })
        pathways[gene] = paths

    total = sum(len(p) for p in pathways.values())
    nonempty = sum(1 for g in pathways if pathways[g])
    log.info(f"Loaded {total} pathways for {nonempty} genes")
    return pathways


def connect_chroma(path, collection_name):
    """Connect to chromadb collection."""
    log.info(f"Connecting to ChromaDB at {path}")
    client = chromadb.PersistentClient(path=path)
    coll = client.get_collection(name=collection_name)
    log.info(f"Collection '{collection_name}' has {coll.count():,} docs")
    return coll


# ============================================================================
# Retrieval (question-as-query + gene mention filter only)
# ============================================================================

def retrieve_abstracts(gene, collection, n=10):
    """
    Retrieval (minimal + clean):
    - 3 natural language questions per gene (question-as-query)
    - query Chroma for each question (top 100 candidates)
    - HARD FILTER: keep only docs that mention the gene
    - No scoring / no manual reranking; keep first-seen order
    """
    questions = build_feature_questions(gene)

    candidates = {}  # pmid -> {pmid,title,abstract}

    for q in questions:
        try:
            results = collection.query(
                query_texts=[q],
                n_results=100
            )

            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]

            if not docs:
                continue

            for i, doc in enumerate(docs):
                meta = metas[i] if i < len(metas) else {}
                pmid = meta.get("pmid", f"unk_{i}")

                if pmid in candidates:
                    continue

                full_text = doc or ""
                if not gene_mentioned(gene, full_text):
                    continue  # HARD FILTER

                parts = full_text.split("\n\n", 1)
                title = parts[0] if parts else ""
                abstract = parts[1] if len(parts) > 1 else full_text

                candidates[pmid] = {
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "gene_mentioned": True,
                }

        except Exception as e:
            log.warning(f"retrieval error for '{q}': {e}")
            continue

    out = list(candidates.values())[:n]
    log.info(f"  {gene}: {len(out)} abstracts (question-based + gene filter)")
    return out


# ============================================================================
# Prompt building
# ============================================================================

def format_pathways(gene, pathways):
    if not pathways:
        return "No pathways found."

    by_type = {}
    for p in pathways:
        t = p.get("type", "Unknown")
        by_type.setdefault(t, []).append(p)

    lines = []
    for t, ps in sorted(by_type.items()):
        lines.append(f"\n{t.upper()} ({len(ps)}):")
        for p in ps[:5]:
            lines.append(f"  {gene} -[{p['rel1']}]-> {p['intermediate']} -[{p['rel2']}]-> MDD")
        if len(ps) > 5:
            lines.append(f"  ... +{len(ps)-5} more")
    return "\n".join(lines)


def format_abstracts(abstracts):
    if not abstracts:
        return "No abstracts found."

    lines = []
    for i, a in enumerate(abstracts, 1):
        text = (a.get("abstract") or "")[:400]
        if len(a.get("abstract") or "") > 400:
            text = text.rsplit(" ", 1)[0] + "..."
        marker = "*" if a.get("gene_mentioned") else " "
        lines.append(f"[{i}]{marker} PMID:{a['pmid']}\n{a['title']}\n{text}\n")
    return "\n".join(lines)


def build_prompt(gene, pathways, abstracts):
    """Build analysis prompt (JSON only)."""
    path_text = format_pathways(gene, pathways)
    abs_text = format_abstracts(abstracts)
    pmids = [a["pmid"] for a in abstracts]

    qs = "\n".join([f"- {q}" for q in build_feature_questions(gene)])

    return f"""Analyze {gene} in treatment-resistant major depressive disorder (TR-MDD).

RETRIEVAL QUESTIONS:
{qs}

KNOWLEDGE GRAPH PATHWAYS:
{path_text}

LITERATURE (* = mentions {gene}):
{abs_text}

TASK:
1. Explain how {gene} contributes to depression (MDD)
2. Explain how it relates to treatment response/resistance
3. Cite PMIDs inline like [PMID:12345678]
4. Only cite from: {pmids}
5. Output JSON only (no extra text)

Respond with JSON only:
{{
  "gene": "{gene}",
  "answer": "150-300 word analysis with inline citations",
  "key_mechanisms": [{{"name": "...", "description": "...", "evidence_pmids": ["..."]}}],
  "kg_pathways_used": [{{"source": "{gene}", "intermediate": "...", "target": "MDD", "relevance": "..."}}],
  "citations": [{{"claim": "...", "evidence_pmid": "...", "quote": "..."}}],
  "evidence_pmids": ["all cited"],
  "confidence": "high|medium|low",
  "limitations": "...",
  "clinical_relevance": "..."
}}
JSON:"""


# ============================================================================
# Ollama
# ============================================================================

def query_ollama(prompt, model, timeout=OLLAMA_TIMEOUT):
    """Query ollama with retries."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            start = time.time()
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 2000}
                },
                timeout=timeout
            )
            elapsed = time.time() - start

            if resp.status_code == 200:
                return resp.json().get("response", ""), elapsed

            log.warning(f"ollama status {resp.status_code}")

        except requests.exceptions.Timeout:
            log.warning(f"ollama timeout (attempt {attempt+1})")
        except requests.exceptions.ConnectionError:
            log.error("can't connect to ollama")
            break
        except Exception as e:
            log.warning(f"ollama error: {e}")

        if attempt < MAX_RETRIES:
            time.sleep(3)

    return None, 0


def clean_json(text):
    """Extract json from llm response."""
    if not text:
        return ""
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return text.strip()

    js = text[start:end + 1]
    js = re.sub(r",\s*}", "}", js)
    js = re.sub(r",\s*]", "]", js)
    return js.strip()


# ============================================================================
# Pipeline
# ============================================================================

def process_gene(gene, pathways, collection, model_name):
    """Run RAG for one gene."""
    log.info(f"Processing {gene} with {model_name}")

    result = {
        "gene": gene,
        "model": model_name,
        "status": "failed",
        "num_pathways": len(pathways),
        "num_abstracts": 0,
        "retrieved_pmids": [],
        "cited_pmids": [],
        "hallucinated_pmids": [],
        "response_time": 0,
        "parsed": None,
        "error": None,
    }

    # retrieve
    abstracts = retrieve_abstracts(gene, collection, N_ABSTRACTS)
    result["num_abstracts"] = len(abstracts)
    result["retrieved_pmids"] = [a["pmid"] for a in abstracts]

    # prompt
    prompt = build_prompt(gene, pathways, abstracts)

    # query
    response, elapsed = query_ollama(prompt, model_name)
    result["response_time"] = elapsed

    if not response:
        result["error"] = "no response"
        return result

    result["raw_response"] = response

    # parse
    cleaned = clean_json(response)
    try:
        parsed = json.loads(cleaned)
        result["parsed"] = parsed
        result["status"] = "success"

        # check citations by PMID membership only (minimal)
        cited = parsed.get("evidence_pmids", []) or []
        result["cited_pmids"] = cited
        result["hallucinated_pmids"] = [p for p in cited if p not in result["retrieved_pmids"]]

        if result["hallucinated_pmids"]:
            result["status"] = "hallucination"
            log.warning(f"  hallucinated pmids: {result['hallucinated_pmids']}")
        else:
            log.info("  success")

    except json.JSONDecodeError as e:
        result["status"] = "json_error"
        result["error"] = str(e)
        log.error(f"  json error: {e}")

    return result


def run_model(model_cfg, pathways_by_gene, collection):
    """Run all genes for one model."""
    name = model_cfg["name"]

    log.info(f"\n{'=' * 50}")
    log.info(f"Model: {name}")
    log.info(f"{'=' * 50}")

    genes = [g for g in SHAP_GENES if pathways_by_gene.get(g)]
    results = []

    iterator = tqdm(genes, desc=name) if tqdm else genes
    for gene in iterator:
        r = process_gene(gene, pathways_by_gene[gene], collection, name)
        results.append(r)
        time.sleep(1)

    return results


def save_results(results, model_name):
    """Save results to json and csv."""
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = model_name.replace(":", "_")

    # json
    jpath = Path(OUTPUT_DIR) / f"results_{safe}_{ts}.json"
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    # csv summary
    cpath = Path(OUTPUT_DIR) / f"summary_{safe}_{ts}.csv"
    rows = []
    for r in results:
        cited = len(r.get("cited_pmids", []))
        hall = len(r.get("hallucinated_pmids", []))
        acc = (cited - hall) / cited if cited else 0
        rows.append({
            "gene": r["gene"],
            "status": r["status"],
            "time_s": round(r["response_time"], 1),
            "pathways": r["num_pathways"],
            "abstracts": r.get("num_abstracts", 0),
            "cited": cited,
            "hallucinated": hall,
            "accuracy": round(acc, 3),
        })
    pd.DataFrame(rows).to_csv(cpath, index=False)

    log.info(f"Saved: {jpath}")
    log.info(f"Saved: {cpath}")

    success = sum(1 for r in results if r["status"] == "success")
    log.info(f"\nResults: {success}/{len(results)} success")


def main():
    log.info("TR-MDD RAG Pipeline (minimal)")
    log.info(f"Models: {[m['name'] for m in MODELS]}")
    log.info(f"Genes: {len(SHAP_GENES)}")

    pathways = load_pathways(PATHWAYS_CSV)
    collection = connect_chroma(CHROMA_DIR, COLLECTION_NAME)

    for model in MODELS:
        results = run_model(model, pathways, collection)
        save_results(results, model["name"])

    log.info("\nDone.")


if __name__ == "__main__":
    main()
