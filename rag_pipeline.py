import json
import time
import logging
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional

import pandas as pd
import requests
import chromadb

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from schema import TRMDDAnalysisResponse, get_trmdd_schema

try:
    from pydantic import ValidationError
except ImportError:
    ValidationError = Exception


BASE_DIR = Path(__file__).resolve().parent

PATHWAYS_CSV = BASE_DIR / "pathways.csv"
CHROMA_DIR = BASE_DIR / "chroma_data" / "pubmed_trd_mdd"
OUTPUT_DIR = BASE_DIR / "rag_results"
COLLECTION_NAME = "pubmed_trd_mdd"


# ============================================================================
# Experiment Config
# ============================================================================

SHAP_GENES = ['MAOA', 'DRD4', 'HTR2C', 'ADRA1A', 'ADRA2A', 'OPRM1', 'OPRK1', 'KCND2', 'KCND3', 'KCNQ2', 'KCNQ3', 'CHRNA3', 'CHRNB4', 'CHRM1', 'ABAT', 'GPT2']


MODELS = [
    {"name": "qwen3:0.6b", "size": "0.6b", "family": "qwen3"},
    {"name": "qwen3:1.7b", "size": "1.7b", "family": "qwen3"},
    {"name": "qwen3:8b", "size": "8b", "family": "qwen3"},
    {"name": "qwen3:14b", "size": "14b", "family": "qwen3"},
    {"name": "qwen3:30b", "size": "30b", "family": "qwen3"},

    {"name": "deepseek-r1:1.5b", "size": "1.5b", "family": "deepseek-r1"},
    {"name": "deepseek-r1:8b", "size": "8b", "family": "deepseek-r1"},
    {"name": "deepseek-r1:14b", "size": "14b", "family": "deepseek-r1"},
    {"name": "deepseek-r1:32b", "size": "32b", "family": "deepseek-r1"},

    {"name": "llama3.2:1b", "size": "1b", "family": "llama3.2"},
    {"name": "llama3.2:3b", "size": "3b", "family": "llama3.2"},
    {"name": "llama3.1:8b", "size": "8b", "family": "llama3.1"},
]
# Retrieval
N_ABSTRACTS = 10

# Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT = 300
MAX_RETRIES = 2
TEMPERATURE = 0.1
NUM_PREDICT = 2000

# Repetitions
N_REPETITIONS = 100
TEST_MODE = False


# ============================================================================
# Logging
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ============================================================================
# Data Loading
# ============================================================================
def load_pathways(csv_path: Path) -> Dict[str, List[Dict]]:
    """
    Load gene-disease pathways from Neo4j CSV export.

    Expected CSV columns:
        gene, intermediate_node, intermediate_type, relationship_1, relationship_2

    Returns:
        {
          gene: [
            {
              "intermediate": "...",
              "type": "...",
              "rel1": "...",
              "rel2": "..."
            }
          ]
        }
    """

    log.info(f"Loading pathways from {csv_path}")

    if not csv_path.exists():
        log.error(f"Pathways CSV not found: {csv_path}")
        sys.exit(1)

    # Load CSV
    df = pd.read_csv(csv_path)

    # Remove duplicate rows
    df = df.drop_duplicates()

    pathways = {}

    for gene in SHAP_GENES:
        rows = df[df["gene"] == gene]
        paths = []

        for _, r in rows.iterrows():
            paths.append({
                "intermediate": str(r.get("intermediate_node", "Unknown")).strip(),
                "type": str(r.get("intermediate_type", "Unknown")).strip(),
                "rel1": str(r.get("relationship_1", "related_to")).strip(),
                "rel2": str(r.get("relationship_2", "associated_with")).strip(),
            })

        pathways[gene] = paths

    total = sum(len(v) for v in pathways.values())
    nonempty = sum(1 for v in pathways.values() if v)

    log.info(f"Loaded {total} pathways for {nonempty}/{len(SHAP_GENES)} genes")

    return pathways


def get_valid_intermediates(pathways: List[Dict]) -> set:
    """Set of valid intermediate node names from CSV for hallucination check."""
    return {p["intermediate"] for p in pathways}


def connect_chroma(chroma_dir: Path, collection_name: str):
    """Connect to ChromaDB persistent collection."""
    log.info(f"Connecting to ChromaDB at {chroma_dir}")
    if not chroma_dir.exists():
        log.error(f"ChromaDB directory not found: {chroma_dir}")
        sys.exit(1)

    client = chromadb.PersistentClient(path=str(chroma_dir))
    coll = client.get_collection(name=collection_name)
    log.info(f"Collection '{collection_name}' has {coll.count():,} documents")
    return coll


# ============================================================================
# Retrieval — Single Semantic Query
# ============================================================================

def retrieve_abstracts(gene: str, collection, n: int = N_ABSTRACTS) -> List[Dict]:
    """
    Single focused semantic query per gene — no RRF, no hard filter.

    Query: "Role of {gene} in treatment-resistant major depressive disorder
            and antidepressant response mechanisms"

    Returns top-N abstracts sorted by cosine similarity.
    """
    query = (
        f"Role of {gene} in treatment-resistant major depressive disorder "
        f"and antidepressant response mechanisms"
    )

    try:
        results = collection.query(query_texts=[query], n_results=n)

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        abstracts = []
        for i, doc in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            pmid = meta.get("pmid", f"unk_{i}")
            dist = distances[i] if i < len(distances) else None

            full_text = doc or ""
            parts = full_text.split("\n\n", 1)
            title = parts[0] if parts else ""
            abstract = parts[1] if len(parts) > 1 else full_text

            abstracts.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "distance": dist,
            })

        log.info(f"  {gene}: retrieved {len(abstracts)} abstracts")
        return abstracts

    except Exception as e:
        log.error(f"  {gene}: retrieval error — {e}")
        return []


# ============================================================================
# Prompt Building
# ============================================================================

def format_abstracts(abstracts: List[Dict]) -> str:
    """Format retrieved abstracts for the prompt."""
    if not abstracts:
        return "No literature retrieved."

    lines = []
    for i, a in enumerate(abstracts, 1):
        text = (a.get("abstract") or "").strip()
        lines.append(f"[{i}] PMID:{a['pmid']}\n{a['title']}\n{text}\n")

    return "\n".join(lines)


def format_pathways(gene: str, pathways: List[Dict]) -> Tuple[str, Dict[int, Dict]]:
    """
    Format pathways with numbered [T1], [T2], ... indices.

    Returns:
        path_text: formatted KG pathway text
        kg_index: {1: pathway_dict_for_T1, 2: pathway_dict_for_T2, ...}
    """
    if not pathways:
        return "No pathways available for this gene.", {}

    by_type: Dict[str, List[Tuple[int, Dict]]] = {}
    kg_index: Dict[int, Dict] = {}

    for i, p in enumerate(pathways, 1):
        kg_index[i] = p
        by_type.setdefault(p.get("type", "Unknown"), []).append((i, p))

    lines = [f"KNOWLEDGE GRAPH PATHWAYS FOR {gene}:"]

    for t in sorted(by_type):
        ps = by_type[t]
        label = "pathway" if len(ps) == 1 else "pathways"
        lines.append(f"\n{t.upper()} ({len(ps)} {label}):")

        for idx, p in ps:
            lines.append(
                f"  [T{idx}] {gene} -[{p['rel1']}]-> {p['intermediate']} "
                f"-[{p['rel2']}]-> MDD"
            )

    return "\n".join(lines), kg_index


def build_prompt(gene: str, pathways: List[Dict], abstracts: List[Dict]) -> Tuple[str, Dict[int, Dict]]:
    """
    Build the LLM prompt with inline numbered triple citations [T#]
    and standard inline PMID citations [PMID:########].

    Returns:
        (prompt_str, kg_index_dict)
    """
    path_text, kg_index = format_pathways(gene, pathways)
    abs_text = format_abstracts(abstracts)
    pmid_list = [a["pmid"] for a in abstracts]
    intermediates = [p["intermediate"] for p in pathways]
    triple_ids = [f"T{i}" for i in kg_index.keys()]

    prompt = f"""You are a biomedical researcher with expertise in pharmacogenomics, psychiatric genetics, and treatment-resistant depression. You synthesize evidence strictly from provided sources and never introduce claims without supporting citations.

Analyze the role of {gene} in Treatment-Resistant Major Depressive Disorder (TR-MDD). Major Depressive Disorder (MDD) affects a significant portion of the population, and approximately 30% of patients do not respond to standard antidepressant therapies, a condition referred to as Treatment-Resistant MDD (TR-MDD).

KNOWLEDGE GRAPH PATHWAYS:
{path_text}

RETRIEVED LITERATURE:
{abs_text}

AVAILABLE PMIDs: {pmid_list}
AVAILABLE TRIPLE IDS: {triple_ids}
VALID PATHWAY INTERMEDIATES: {intermediates}

INSTRUCTIONS:

EVIDENCE CONSTRAINT:
- Do NOT produce any statement, claim, or narrative that is not directly supported by the provided literature or knowledge graph pathways above.
- If the evidence is insufficient to make a claim, explicitly state that the evidence is lacking rather than speculating.
- Do not invent PMIDs.
- Do not invent triple identifiers.
- Use only the evidence shown above.

EVIDENCE SELECTION:
- First identify which pathway entries are relevant to the answer.
- Then identify which PMIDs are relevant to the answer.
- Then write the final response using ONLY those selected evidence items.

CONTENT RULES:
1. In "answer": Write a 150-350 word narrative synthesis of how {gene} contributes to MDD and treatment resistance. This is your main analysis. Cite PMIDs inline like [PMID:12345678]. Cite knowledge graph pathway entries inline like [T1], [T2], etc. Focus on connecting literature evidence with pathway evidence to build the clinical picture.
2. In "key_mechanisms": List ONLY the distinct biological mechanisms. Keep each mechanism focused on HOW it contributes to treatment resistance. Do NOT repeat the full narrative from "answer".
3. In "limitations": Be actionable — state what specific evidence is missing and what studies or experiments would be needed to resolve the gaps.
4. In "clinical_relevance": State concrete clinical implications for TR-MDD treatment decisions.

CITATION RULES:
5. Keep PMID citations exactly in this format: [PMID:12345678]
6. Cite pathway evidence inline as [T1], [T2], etc.
7. ONLY cite PMIDs from the AVAILABLE PMIDs list above.
8. ONLY cite triple IDs from the AVAILABLE TRIPLE IDS list above.
9. Every important factual claim in "answer", "key_mechanisms", and "clinical_relevance" must have at least one supporting citation.
10. If a claim is supported by both pathway evidence and literature evidence, cite both.

PATHWAY RULES:
11. Do NOT fabricate pathway nodes — only use intermediates from the VALID PATHWAY INTERMEDIATES list.
12. Keep kg_index as metadata only. The real evidence signal is the inline citation in the text.

OUTPUT:
13. Output ONLY valid JSON matching this exact structure:

{{
  "gene": "{gene}",
  "relevant_triples": ["T1", "T2"],
  "relevant_pmids": ["12345678"],
  "answer": "150-350 word narrative analysis with inline [T#] and [PMID:...] citations",
  "key_mechanisms": [
    {{
      "name": "short mechanism label",
      "description": "how this mechanism contributes to treatment resistance specifically with inline citations"
    }}
  ],
  "limitations": "what evidence is missing + what studies are needed to fill the gaps",
  "clinical_relevance": "concrete clinical implications for TR-MDD treatment with inline citations"
}}

JSON:"""

    return prompt, kg_index


# ============================================================================
# Ollama Query
# ============================================================================

def query_ollama(prompt: str, model: str, timeout: int = OLLAMA_TIMEOUT) -> Tuple[Optional[str], float]:
    """
    Query Ollama with retries.
    Uses structured output via format parameter (JSON schema).
    Returns (response_text | None, elapsed_seconds).
    """
    schema = get_trmdd_schema()
    last_elapsed = 0.0

    for attempt in range(MAX_RETRIES + 1):
        try:
            start = time.time()
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "format": schema,
                    "options": {
                        "temperature": TEMPERATURE,
                        "num_predict": NUM_PREDICT,
                    },
                },
                timeout=timeout,
            )
            elapsed = time.time() - start
            last_elapsed = elapsed

            if resp.status_code != 200:
                log.warning(
                    f"  Ollama HTTP {resp.status_code} (attempt {attempt + 1}) | "
                    f"body={resp.text[:1000]}"
                )
            else:
                try:
                    data = resp.json()
                except Exception as e:
                    log.warning(f"  Ollama JSON decode failed (attempt {attempt + 1}): {e}")
                    log.warning(f"  Raw body: {resp.text[:1000]}")
                    data = {}

                raw = data.get("response")
                if raw and str(raw).strip():
                    return str(raw), elapsed

                log.warning(
                    f"  Empty Ollama response field (attempt {attempt + 1}) | "
                    f"keys={list(data.keys()) if isinstance(data, dict) else type(data)} | "
                    f"body={resp.text[:1000]}"
                )

        except requests.exceptions.Timeout:
            log.warning(f"  Ollama timeout (attempt {attempt + 1})")
        except requests.exceptions.ConnectionError:
            log.error("  Cannot connect to Ollama — is it running?")
            break
        except Exception as e:
            log.warning(f"  Ollama error (attempt {attempt + 1}): {e}")

        if attempt < MAX_RETRIES:
            time.sleep(3)

    return None, last_elapsed


# ============================================================================
# Response Cleaning & Validation
# ============================================================================

def clean_json(text: str) -> str:
    """
    Clean LLM response for JSON parsing:
      1. Strip <think>...</think> blocks
      2. Strip markdown code fences
      3. Extract outermost {...}
      4. Fix trailing commas
    """
    if not text:
        return ""

    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
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


def parse_and_validate(raw_response: str) -> Dict[str, Any]:
    """
    Two-stage parsing:
      1. json.loads()
      2. TRMDDAnalysisResponse(**dict)

    Returns {"parsed": dict|None, "status": str, "error": str|None}
    """
    cleaned = clean_json(raw_response)

    try:
        parsed_dict = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return {"parsed": None, "status": "json_error", "error": f"JSONDecodeError: {e}"}

    try:
        validated = TRMDDAnalysisResponse(**parsed_dict)
        return {"parsed": validated.model_dump(), "status": "success", "error": None}
    except ValidationError as e:
        return {"parsed": parsed_dict, "status": "validation_error", "error": f"ValidationError: {e}"}
    except Exception as e:
        return {"parsed": parsed_dict, "status": "validation_error", "error": str(e)}


# ============================================================================
# Process Single Gene
# ============================================================================

def process_gene(
    gene: str,
    pathways: List[Dict],
    collection,
    model_name: str,
) -> Dict[str, Any]:
    """
    Full pipeline for one gene:
      1. Retrieve abstracts
      2. Build prompt
      3. Query LLM
      4. Parse & validate
    """
    log.info(f"Processing {gene} with {model_name}")

    result = {
        "gene": gene,
        "model": model_name,
        "status": "failed",
        "num_pathways_provided": len(pathways),
        "num_abstracts": 0,
        "num_retrieved_pmids": 0,
        "retrieved_pmids": [],
        "kg_index": {},
        "response_time": 0,
        "parsed": None,
        "raw_response": None,
        "error": None,
    }

    # 1. Retrieve
    abstracts = retrieve_abstracts(gene, collection, N_ABSTRACTS)
    retrieved_pmids = [a["pmid"] for a in abstracts]
    result["num_abstracts"] = len(abstracts)
    result["retrieved_pmids"] = retrieved_pmids
    result["num_retrieved_pmids"] = len(retrieved_pmids)

    if not abstracts:
        result["error"] = "no abstracts retrieved"
        return result

    # 2. Build prompt
    prompt, kg_index = build_prompt(gene, pathways, abstracts)
    result["kg_index"] = kg_index

    # 3. Query LLM
    response, elapsed = query_ollama(prompt, model_name)
    result["response_time"] = elapsed

    if not response:
        result["error"] = "no response from Ollama"
        return result

    result["raw_response"] = response

    # 4. Parse & validate
    pv = parse_and_validate(response)
    result["parsed"] = pv["parsed"]
    result["status"] = pv["status"]
    result["error"] = pv["error"]

    if result["status"] == "success":
        log.info(f"  OK  {gene}: status=success")
    else:
        log.warning(f"  FAIL {gene}: {result['status']} — {result['error']}")

    return result


# ============================================================================
# Run One Model Over All Genes
# ============================================================================

def run_model(
    model_cfg: Dict,
    pathways_by_gene: Dict[str, List[Dict]],
    collection,
    genes: List[str],
) -> List[Dict]:
    """Run the pipeline for every gene with one model."""
    name = model_cfg["name"]
    log.info(f"\n{'=' * 60}")
    log.info(f"Model: {name}  |  Genes: {len(genes)}")
    log.info(f"{'=' * 60}")

    results = []
    iterator = tqdm(genes, desc=name) if tqdm else genes

    for gene in iterator:
        r = process_gene(gene, pathways_by_gene.get(gene, []), collection, name)
        results.append(r)
        time.sleep(1)

    return results


# ============================================================================
# Save Results — rep_XX/ structure
# ============================================================================

CSV_COLUMNS = [
    "rep", "gene", "model", "status", "time_s",
    "num_pathways_provided", "num_abstracts", "num_retrieved_pmids",
]


def save_results(
    results: List[Dict],
    model_name: str,
    rep_idx: int,
) -> None:
    """
    Save to rep_XX/ folder:
      - results_{model}_{timestamp}.json
      - summary_{model}_{timestamp}.csv
    """
    rep_dir = OUTPUT_DIR / f"rep_{rep_idx:02d}"
    rep_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = model_name.replace(":", "_").replace("/", "_")

    json_path = rep_dir / f"results_{safe}_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"Saved JSON -> {json_path}")

    csv_path = rep_dir / f"summary_{safe}_{ts}.csv"
    rows = []
    for r in results:
        rows.append({
            "rep": rep_idx,
            "gene": r["gene"],
            "model": r["model"],
            "status": r["status"],
            "time_s": round(r["response_time"], 1),
            "num_pathways_provided": r["num_pathways_provided"],
            "num_abstracts": r["num_abstracts"],
            "num_retrieved_pmids": r.get("num_retrieved_pmids", 0),
        })

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    df.to_csv(csv_path, index=False)
    log.info(f"Saved CSV  -> {csv_path}")

    ok = sum(1 for r in results if r["status"] == "success")
    log.info(f"  {ok}/{len(results)} success")


# ============================================================================
# Main
# ============================================================================

def main():
    log.info("=" * 60)
    log.info("TR-MDD RAG Pipeline")
    log.info("=" * 60)
    log.info(f"BASE_DIR:    {BASE_DIR}")
    log.info(f"PATHWAYS:    {PATHWAYS_CSV}")
    log.info(f"CHROMA_DIR:  {CHROMA_DIR}")
    log.info(f"OUTPUT_DIR:  {OUTPUT_DIR}")

    if TEST_MODE:
        genes = ['MAOA', 'DRD4', 'HTR2C', 'ADRA1A', 'ADRA2A', 'OPRM1', 'OPRK1', 'KCND2', 'KCND3', 'KCNQ2', 'KCNQ3', 'CHRNA3', 'CHRNB4', 'CHRM1', 'ABAT', 'GPT2']
        n_reps = 1
        log.info("*** TEST MODE: MAOA only, 1 rep ***")
    else:
        genes = SHAP_GENES
        n_reps = N_REPETITIONS

    total_runs = len(genes) * len(MODELS) * n_reps
    log.info(f"Models:      {[m['name'] for m in MODELS]}")
    log.info(f"Genes:       {len(genes)}")
    log.info(f"Repetitions: {n_reps}")
    log.info(f"Total runs:  {total_runs}")

    pathways = load_pathways(PATHWAYS_CSV)
    collection = connect_chroma(CHROMA_DIR, COLLECTION_NAME)

    for rep in range(1, n_reps + 1):
        log.info(f"\n{'#' * 60}")
        log.info(f"REPETITION {rep}/{n_reps}")
        log.info(f"{'#' * 60}")

        rep_dir = OUTPUT_DIR / f"rep_{rep:02d}"

        for model_cfg in MODELS:
            safe = model_cfg["name"].replace(":", "_").replace("/", "_")
            existing = list(rep_dir.glob(f"results_{safe}_*.json")) if rep_dir.exists() else []

            if existing:
                log.info(f"SKIP {model_cfg['name']} rep {rep} — already exists: {existing[0].name}")
                continue

            results = run_model(model_cfg, pathways, collection, genes)
            save_results(results, model_cfg["name"], rep)

    log.info("\n" + "=" * 60)
    log.info("Pipeline complete.")
    log.info(f"Results in: {OUTPUT_DIR}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()