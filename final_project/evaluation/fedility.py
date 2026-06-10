import json
import time
import logging
import re
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Check your .env file.")

client = OpenAI(api_key=api_key)

# ============================================================================
# Config
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent

RESULTS_DIR = BASE_DIR / "rag_results"
OUTPUT_DIR = BASE_DIR / "kg_triplet_judgment_results"

OPENAI_MODEL = "gpt-4o-mini"

TEST_MODE = True
TEST_MAX_FILES = 1
TEST_MAX_ROWS_PER_FILE = 1

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
# OpenAI Client
# ============================================================================

client = OpenAI(api_key=api_key)
# ============================================================================
# Text Helpers
# ============================================================================

def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


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


def split_sentences(text: str) -> List[str]:
    text = normalize_ws(text)
    if not text:
        return []

    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9\[])', text)
    return [p.strip() for p in parts if p.strip()]


def extract_t_refs(text: str) -> List[str]:
    refs = re.findall(r"\bT(\d+)\b", text or "", flags=re.IGNORECASE)
    return [f"T{x}" for x in refs]


def strip_t_citations(text: str) -> str:
    text = re.sub(r"\[(?:\s*T\d+(?:\s*,\s*T\d+)*\s*)\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bT\d+\b", "", text, flags=re.IGNORECASE)
    return normalize_ws(text)


def find_claims_with_trefs(text: str) -> List[Dict[str, Any]]:
    rows = []

    for sent in split_sentences(text):
        refs = extract_t_refs(sent)
        if refs:
            rows.append({
                "sentence": sent,
                "clean_claim": strip_t_citations(sent),
                "t_refs": refs,
            })

    return rows

# ============================================================================
# KG Helpers
# ============================================================================

def triplet_from_kg_entry(entry: Dict[str, Any]) -> Dict[str, str]:
    return {
        "intermediate": str(entry.get("intermediate", entry.get("intermediate_node", "")) or ""),
        "type": str(entry.get("type", entry.get("intermediate_type", "")) or ""),
        "rel1": str(entry.get("rel1", entry.get("relationship_1", "")) or ""),
        "rel2": str(entry.get("rel2", entry.get("relationship_2", "")) or ""),
    }


def format_triplet(gene: str, triplet: Dict[str, str]) -> str:
    return (
        f"{gene} -[{triplet.get('rel1', '')}]-> "
        f"{triplet.get('intermediate', '')} -[{triplet.get('rel2', '')}]-> MDD"
    )

# ============================================================================
# Prompt
# ============================================================================

def build_judge_prompt(
    gene: str,
    t_ref: str,
    triplet: Dict[str, str],
    sentence: str,
    clean_claim: str,
) -> str:
    triplet_text = format_triplet(gene, triplet)

    sentence = (sentence or "")[:400]
    clean_claim = (clean_claim or "")[:300]

    return f"""You are evaluating ONE cited knowledge-graph triplet used by an LLM.

You must make TWO strict binary judgments.

JUDGMENT 1: faithful
Return 1 if the cited triplet is used faithfully with biomedical background to it.
Return 0 if the sentence hallucinates, overstates, misrepresents, or adds facts not actually present in the triplet.

JUDGMENT 2: supports_claim
Only if the triplet use is faithful, decide whether that triplet supports the claim in the sentence.
Return 1 only if the triplet gives real support to the claim.
Return 0 if the triplet is too weak, too indirect, irrelevant, or does not justify the claim.

Strict rules:
- Use ONLY the triplet below.
- Do NOT use outside biomedical knowledge.
- Do NOT infer missing facts.
- If the sentence adds mechanisms, causality, treatment effects, severity, modulation, or clinical conclusions not present in the triplet, faithful = 0.
- If faithful = 0, then supports_claim = 0.
- Be strict and conservative.

Return ONLY valid JSON in exactly this form:
{{
  "faithful": 0,
  "supports_claim": 0,
  "reason_faithful": "short reason",
  "reason_support": "short reason"
}}

INPUTS

Gene: {gene}
Cited triplet ID: {t_ref}
Triplet: {triplet_text}
Original sentence: {sentence}
Claim without citation marker: {clean_claim}
"""

# ============================================================================
# OpenAI Query
# ============================================================================

def query_openai(prompt: str) -> Tuple[Optional[str], float]:
    try:
        start = time.time()

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.3,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict evaluator. Always return valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )

        elapsed = time.time() - start
        content = response.choices[0].message.content

        if content and str(content).strip():
            return str(content), elapsed

        return None, elapsed

    except Exception as e:
        log.error(f"OpenAI error: {e}")
        return None, 0.0

# ============================================================================
# Parsing
# ============================================================================

def clean_json(text: str) -> str:
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


def parse_judge_response(raw_response: str) -> Dict[str, Any]:
    cleaned = clean_json(raw_response)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return {
            "parsed": None,
            "status": "json_error",
            "error": f"JSONDecodeError: {e}",
        }

    faithful = 1 if int(parsed.get("faithful", 0)) == 1 else 0
    supports_claim = 1 if int(parsed.get("supports_claim", 0)) == 1 else 0

    if faithful == 0:
        supports_claim = 0

    parsed["faithful"] = faithful
    parsed["supports_claim"] = supports_claim
    parsed["reason_faithful"] = str(parsed.get("reason_faithful", "")).strip()
    parsed["reason_support"] = str(parsed.get("reason_support", "")).strip()

    return {
        "parsed": parsed,
        "status": "success",
        "error": None,
    }

# ============================================================================
# Core Evaluation
# ============================================================================

def judge_one_triplet(
    gene: str,
    kg_index: Dict[str, Any],
    sentence: str,
    clean_claim: str,
    t_ref: str,
) -> Dict[str, Any]:
    t_num = t_ref.upper().replace("T", "")
    kg_entry = kg_index.get(t_num)

    result = {
        "t_ref": t_ref,
        "sentence": sentence,
        "clean_claim": clean_claim,
        "faithful": 0,
        "supports_claim": 0,
        "reason_faithful": "",
        "reason_support": "",
        "triplet": None,
        "judge_model": OPENAI_MODEL,
        "judge_status": "failed",
        "judge_error": None,
        "judge_time_s": 0.0,
        "raw_judge_response": None,
    }

    if not kg_entry:
        result["judge_status"] = "missing_t_ref"
        result["judge_error"] = f"{t_ref} not found in kg_index"
        result["reason_faithful"] = "Referenced triplet ID not found in kg_index."
        result["reason_support"] = "Cannot support claim because referenced triplet is missing."
        return result

    triplet = triplet_from_kg_entry(kg_entry)
    result["triplet"] = triplet

    prompt = build_judge_prompt(
        gene=gene,
        t_ref=t_ref,
        triplet=triplet,
        sentence=sentence,
        clean_claim=clean_claim,
    )

    log.info(f"JUDGING {gene} | {t_ref} | claim={clean_claim[:120]}")

    raw_response, elapsed = query_openai(prompt)
    result["judge_time_s"] = elapsed
    result["raw_judge_response"] = raw_response

    log.info(f"DONE {gene} | {t_ref} | time={elapsed:.2f}s")

    if not raw_response:
        result["judge_status"] = "no_response"
        result["judge_error"] = "no response from OpenAI"
        return result

    parsed = parse_judge_response(raw_response)
    result["judge_status"] = parsed["status"]
    result["judge_error"] = parsed["error"]

    if parsed["status"] != "success":
        return result

    judged = parsed["parsed"]
    result["faithful"] = judged.get("faithful", 0)
    result["supports_claim"] = judged.get("supports_claim", 0)
    result["reason_faithful"] = judged.get("reason_faithful", "")
    result["reason_support"] = judged.get("reason_support", "")

    return result


def process_result_row(row: Dict[str, Any]) -> Dict[str, Any]:
    gene = row.get("gene")
    model = row.get("model")
    parsed = row.get("parsed", {}) or {}
    kg_index = row.get("kg_index", {}) or {}

    combined_text = get_combined_output_text(parsed)
    claims = find_claims_with_trefs(combined_text)

    log.info(f"{gene} | found {len(claims)} claim sentences with T refs")

    out = {
        "gene": gene,
        "source_model": model,
        "judge_model": OPENAI_MODEL,
        "status": "success",
        "num_claim_sentences_with_trefs": len(claims),
        "num_triplet_checks": 0,
        "triplet_judgments": [],
    }

    for claim_block in claims:
        sentence = claim_block["sentence"]
        clean_claim = claim_block["clean_claim"]
        t_refs = claim_block["t_refs"]

        for t_ref in t_refs:
            judged = judge_one_triplet(
                gene=gene,
                kg_index=kg_index,
                sentence=sentence,
                clean_claim=clean_claim,
                t_ref=t_ref,
            )
            out["triplet_judgments"].append(judged)

    out["num_triplet_checks"] = len(out["triplet_judgments"])
    return out

# ============================================================================
# Saving
# ============================================================================

CSV_COLUMNS = [
    "file",
    "gene",
    "source_model",
    "judge_model",
    "t_ref",
    "faithful",
    "supports_claim",
    "judge_status",
    "judge_time_s",
    "sentence",
    "clean_claim",
    "reason_faithful",
    "reason_support",
]


def save_outputs(all_results: List[Dict], out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = out_dir / f"{prefix}_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"Saved JSON -> {json_path}")

    rows = []
    for block in all_results:
        file_name = block.get("file")
        for item in block.get("row_results", []):
            for j in item.get("triplet_judgments", []):
                rows.append({
                    "file": file_name,
                    "gene": item.get("gene"),
                    "source_model": item.get("source_model"),
                    "judge_model": item.get("judge_model"),
                    "t_ref": j.get("t_ref"),
                    "faithful": j.get("faithful"),
                    "supports_claim": j.get("supports_claim"),
                    "judge_status": j.get("judge_status"),
                    "judge_time_s": round(j.get("judge_time_s", 0.0), 2),
                    "sentence": j.get("sentence"),
                    "clean_claim": j.get("clean_claim"),
                    "reason_faithful": j.get("reason_faithful"),
                    "reason_support": j.get("reason_support"),
                })

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    csv_path = out_dir / f"{prefix}_{ts}.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"Saved CSV  -> {csv_path}")

# ============================================================================
# Main
# ============================================================================

def main():
    log.info("=" * 60)
    log.info("KG Triplet Faithfulness + Support Judge")
    log.info("=" * 60)
    log.info(f"RESULTS_DIR: {RESULTS_DIR}")
    log.info(f"OUTPUT_DIR:  {OUTPUT_DIR}")
    log.info(f"JUDGE_MODEL: {OPENAI_MODEL}")

    if not os.getenv("OPENAI_API_KEY"):
        log.error("OPENAI_API_KEY is not set.")
        sys.exit(1)

    if not RESULTS_DIR.exists():
        log.error(f"RESULTS_DIR not found: {RESULTS_DIR}")
        sys.exit(1)

    files = sorted(RESULTS_DIR.glob("rep_*/results_*.json"))

    if TEST_MODE:
        files = files[:TEST_MAX_FILES]
        log.info(f"TEST MODE: limiting to first {len(files)} file(s)")

    if not files:
        log.error("No results_*.json files found.")
        sys.exit(1)

    all_outputs = []

    for jf in files:
        log.info(f"Reading file: {jf}")

        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            log.error(f"Failed reading {jf}: {e}")
            continue

        row_results = []
        iterable = data[:TEST_MAX_ROWS_PER_FILE] if TEST_MODE else data

        for row in iterable:
            log.info(f"Processing gene={row.get('gene')} model={row.get('model')}")

            parsed = row.get("parsed")
            kg_index = row.get("kg_index")

            if not isinstance(parsed, dict) or not isinstance(kg_index, dict) or not kg_index:
                log.info("Skipping row: parsed or kg_index missing/empty")
                continue

            combined_text = get_combined_output_text(parsed)
            if not combined_text.strip():
                log.info("Skipping row: empty combined text")
                continue

            if not re.search(r"\bT\d+\b", combined_text, flags=re.IGNORECASE):
                log.info("Skipping row: no T# citations in combined text")
                continue

            judged = process_result_row(row)
            row_results.append(judged)

        block = {
            "file": str(jf),
            "row_results": row_results,
        }
        all_outputs.append(block)

    save_outputs(all_outputs, OUTPUT_DIR, "kg_triplet_judgments")

    total_checks = 0
    faithful_yes = 0
    support_yes = 0

    for block in all_outputs:
        for item in block.get("row_results", []):
            for j in item.get("triplet_judgments", []):
                total_checks += 1
                faithful_yes += int(j.get("faithful", 0))
                support_yes += int(j.get("supports_claim", 0))

    log.info("=" * 60)
    log.info(f"Total triplet checks: {total_checks}")
    log.info(f"Faithful = 1 count:   {faithful_yes}")
    log.info(f"Support = 1 count:    {support_yes}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()