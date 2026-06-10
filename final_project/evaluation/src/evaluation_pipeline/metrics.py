from .text_utils import dedupe_str_list, extract_pmids_from_text, get_combined_output_text


def eval_structure_sections(result: dict):
    p = result.get("parsed")

    if not isinstance(p, dict):
        return {
            "answer_present": 0,
            "key_mechanisms_present": 0,
            "limitations_present": 0,
            "clinical_relevance_present": 0,
        }

    return {
        "answer_present": 1 if p.get("answer") else 0,
        "key_mechanisms_present": 1 if p.get("key_mechanisms") else 0,
        "limitations_present": 1 if p.get("limitations") else 0,
        "clinical_relevance_present": 1 if p.get("clinical_relevance") else 0,
    }


def eval_citations(result: dict):
    p = result.get("parsed")

    if not isinstance(p, dict):
        return 0, 0, 0, 0.0, 0.0

    combined_text = get_combined_output_text(p)
    cited = extract_pmids_from_text(combined_text)
    retrieved = dedupe_str_list(result.get("retrieved_pmids", []))

    cited_n = len(cited)
    retrieved_n = len(retrieved)

    if cited_n == 0:
        return 0, 0, retrieved_n, 0.0, 0.0

    retrieved_set = set(retrieved)
    grounded_n = sum(1 for pmid in cited if pmid in retrieved_set)

    precision = grounded_n / cited_n if cited_n else 0.0
    recall = grounded_n / retrieved_n if retrieved_n else 0.0

    return cited_n, grounded_n, retrieved_n, round(precision, 3), round(recall, 3)

