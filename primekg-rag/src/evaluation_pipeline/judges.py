import json
import re
from typing import Dict

from .config import JUDGE_MODEL, client, log
from .debug import debug_write


FIDELITY_SYSTEM_PROMPT = """\
You are a citation fidelity evaluator for biomedical text.

You will receive:
1. The model's generated answer with inline citations like [PMID:12345678]
2. Literature evidence: a mapping of PMIDs to title and abstract

For each PMID citation, check whether the claim surrounding that citation is faithfully supported by the cited abstract.

Return ONLY valid JSON:
{
  "pmid_fidelity": {
    "total_checked": <int>,
    "faithful": ["12345678"],
    "unfaithful": [
      {"id": "34567890", "reason": "abstract discusses rats, model claimed human trial"}
    ]
  }
}
"""


def judge(system_prompt: str, user_content: str, debug_label: str = ""):
    if client is None:
        return {}

    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
        )

        raw = resp.choices[0].message.content

        if debug_label:
            debug_write(f"\n{'=' * 60}")
            debug_write(f"JUDGE RAW RESPONSE - {debug_label}")
            debug_write(f"{'=' * 60}")
            debug_write(raw)
            debug_write(f"{'=' * 60}\n")

        raw = re.sub(r"```json\s*", "", raw)
        raw = re.sub(r"```\s*", "", raw)

        return json.loads(raw.strip())

    except Exception as e:
        log.debug(f"Judge error: {e}")

        if debug_label:
            debug_write(f"JUDGE ERROR - {debug_label}: {e}")

        return {}


def judge_fidelity(parsed: dict, abstract_map: Dict[str, Dict]) -> tuple:
    lit_evidence = {}

    for pmid, info in (abstract_map or {}).items():
        lit_evidence[pmid] = {
            "title": info.get("title", ""),
            "abstract": (info.get("abstract") or "")[:500],
        }

    payload = {
        "model_output": parsed,
        "literature_evidence": lit_evidence,
    }

    user_payload = json.dumps(payload, indent=2, default=str)
    result = judge(FIDELITY_SYSTEM_PROMPT, user_payload, debug_label="fidelity")

    if not result:
        result = {
            "pmid_fidelity": {
                "total_checked": 0,
                "faithful": [],
                "unfaithful": [],
            }
        }

    return result, payload


def judge_validity(parsed: dict):
    if not isinstance(parsed, dict):
        return 0, ""

    claims_only = {
        "gene": parsed.get("gene", ""),
        "answer": parsed.get("answer", ""),
        "key_mechanisms": parsed.get("key_mechanisms", []),
        "clinical_relevance": parsed.get("clinical_relevance", ""),
    }

    result = judge(
        (
            "You are a biomedical expert. Evaluate whether the scientific claims below "
            "are valid based on established biomedical knowledge. Ignore citation formatting. "
            "Return JSON: {\"score\":0|1, \"reason\":\"<=20 words\"}"
        ),
        json.dumps(claims_only, default=str),
        debug_label="validity",
    )

    return int(result.get("score", 0)), str(result.get("reason", "")).strip()

