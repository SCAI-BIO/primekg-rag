import json

from .chroma_access import retrieve_abstracts_by_pmids
from .config import DEBUG_GENE, DEBUG_MODELS, DEBUG_REP, ENABLE_LLM_JUDGE, client, log
from .judges import judge_fidelity, judge_validity
from .metrics import eval_citations, eval_structure_sections
from .text_utils import extract_pmids_from_text, get_combined_output_text


def evaluate(results, collection=None):
    evals = []
    evidence_store = []

    has_api_key = ENABLE_LLM_JUDGE and client is not None

    if DEBUG_GENE:
        results = [
            r for r in results
            if r.get("gene") == DEBUG_GENE
            and (DEBUG_REP is None or r.get("rep") == DEBUG_REP)
            and (not DEBUG_MODELS or r.get("model") in DEBUG_MODELS)
        ]

        log.info(
            f"DEBUG MODE: {len(results)} results "
            f"(gene={DEBUG_GENE} rep={DEBUG_REP} models={DEBUG_MODELS})"
        )

    for i, r in enumerate(results):
        gene = r.get("gene", "")
        model = r.get("model", "")

        struct = eval_structure_sections(r)
        cited_n, grounded_n, retrieved_n, pmid_precision, pmid_recall = eval_citations(r)

        parsed = r.get("parsed")

        fidelity = {
            "pmid_fidelity": {
                "total_checked": 0,
                "faithful": [],
                "unfaithful": [],
                "unchecked": [],
            }
        }

        evidence_payload = {}

        if isinstance(parsed, dict) and has_api_key:
            combined_text = get_combined_output_text(parsed)
            cited_pmid_ids = extract_pmids_from_text(combined_text)
            abstract_map = retrieve_abstracts_by_pmids(cited_pmid_ids, collection)

            unchecked_pmids = [
                str(p) for p in cited_pmid_ids
                if str(p) not in abstract_map
            ]

            fidelity, evidence_payload = judge_fidelity(parsed, abstract_map)
            fidelity["pmid_fidelity"]["unchecked"] = unchecked_pmids

            v_score, v_reason = judge_validity(parsed)

        else:
            v_score, v_reason = 0, ""

        struct_score = 1 if all(v == 1 for v in struct.values()) else 0

        pmid_fid = fidelity.get("pmid_fidelity", {})

        pmid_fid_checked = int(pmid_fid.get("total_checked", 0))
        pmid_fid_faithful_n = len(pmid_fid.get("faithful", []))
        pmid_fid_unfaithful = pmid_fid.get("unfaithful", [])
        pmid_fid_unchecked = pmid_fid.get("unchecked", [])

        pmid_fid_score = (
            pmid_fid_faithful_n / pmid_fid_checked
            if pmid_fid_checked > 0
            else 0.0
        )

        evals.append({
            "rep": r.get("rep", 0),
            "gene": gene,
            "model": model,
            "status": r.get("status", ""),
            "response_time": round(float(r.get("response_time", 0) or 0), 1),

            **struct,
            "structure_score": struct_score,

            "num_cited": cited_n,
            "num_grounded": grounded_n,
            "num_retrieved": retrieved_n,
            "pmid_precision": pmid_precision,
            "pmid_recall": pmid_recall,

            "pmid_fidelity_checked": pmid_fid_checked,
            "pmid_fidelity_faithful": pmid_fid_faithful_n,
            "pmid_fidelity_score": round(pmid_fid_score, 3),
            "pmid_fidelity_unfaithful": json.dumps(pmid_fid_unfaithful, default=str),
            "pmid_fidelity_unchecked": json.dumps(pmid_fid_unchecked, default=str),

            "validity_score": v_score,
            "validity_reason": v_reason,
        })

        evidence_store.append({
            "rep": r.get("rep", 0),
            "gene": gene,
            "model": model,
            "parsed_output": parsed,
            "literature_evidence": evidence_payload.get("literature_evidence", {}),
            "fidelity_judge_response": fidelity,
            "validity_judge_response": {
                "score": v_score,
                "reason": v_reason,
            },
        })

        if (i + 1) % 50 == 0:
            log.info(f"  Evaluated {i + 1}/{len(results)}")

    return evals, evidence_store

