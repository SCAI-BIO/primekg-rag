import re


def extract_pmids_from_text(text: str):
    if not text:
        return []

    seen, out = set(), []
    for h in re.findall(r"PMID[:\s]*(\d+)", text, flags=re.IGNORECASE):
        if h not in seen:
            seen.add(h)
            out.append(h)

    return out


def dedupe_str_list(xs):
    seen, out = set(), []

    for x in xs or []:
        s = str(x).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)

    return out


def get_combined_output_text(parsed: dict) -> str:
    """
    Combine only fields used for PMID citation evaluation.

    IMPORTANT:
    limitations is intentionally ignored.
    """
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
            else:
                parts.append(str(km))

    return " ".join(str(p) for p in parts if p)

