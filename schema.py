import re
from typing import List

from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Helpers
# ============================================================================

def normalize_pmid(raw: str) -> str:
    """
    Extract digits-only PMID from various LLM formats:
      "12345678"        -> "12345678"
      "PMID:12345678"   -> "12345678"
      "PMID: 12345678"  -> "12345678"
      "[PMID:12345678]" -> "12345678"

    Raises ValueError if no digits found.
    """
    cleaned = re.sub(r"[^0-9]", "", str(raw))
    if not cleaned:
        raise ValueError(f"PMID must contain digits, got: '{raw}'")
    return cleaned


def normalize_triple_id(raw: str) -> str:
    """
    Normalize triple IDs to T# format:
      "T1"   -> "T1"
      "[T1]" -> "T1"
      " t2 " -> "T2"

    Raises ValueError if format is invalid.
    """
    s = str(raw).strip().upper()
    s = s.replace("[", "").replace("]", "")
    if not re.fullmatch(r"T\d+", s):
        raise ValueError(f"Triple ID must look like T1, got: '{raw}'")
    return s


# ============================================================================
# Sub-models
# ============================================================================

class Mechanism(BaseModel):
    """A biological mechanism linking the gene to TR-MDD."""

    name: str = Field(
        description="Short name of the mechanism"
    )
    description: str = Field(
        description="How this mechanism contributes to treatment-resistant depression, with inline citations where relevant"
    )


# ============================================================================
# Main response schema
# ============================================================================

class TRMDDAnalysisResponse(BaseModel):
    """
    Structured response for TR-MDD gene analysis.

    Core fields:
      - gene
      - answer
      - key_mechanisms
      - relevant_triples
      - relevant_pmids
      - limitations
      - clinical_relevance

    Main evidence grounding is evaluated from inline citations in:
      - answer
      - key_mechanisms[].description
      - clinical_relevance
    """

    gene: str = Field(
        description="Gene symbol being analyzed"
    )

    answer: str = Field(
        description="Comprehensive analysis (150-350 words) with inline citations such as [T1] and [PMID:12345678]"
    )

    key_mechanisms: List[Mechanism] = Field(
        description="Biological mechanisms linking this gene to TR-MDD",
        min_length=1,
    )

    relevant_triples: List[str] = Field(
        description="Triple identifiers selected as relevant evidence, e.g. ['T1', 'T3']",
        default_factory=list,
    )

    relevant_pmids: List[str] = Field(
        description="PMIDs selected as relevant evidence, digits only",
        default_factory=list,
    )

    limitations: str = Field(
        description="Gaps, inconsistencies, or evidence limitations"
    )

    clinical_relevance: str = Field(
        description="Clinical implications for TR-MDD treatment, with inline citations where relevant"
    )

    @field_validator("relevant_pmids")
    @classmethod
    def clean_relevant_pmids(cls, v: List[str]) -> List[str]:
        return [normalize_pmid(p) for p in v]

    @field_validator("relevant_triples")
    @classmethod
    def clean_relevant_triples(cls, v: List[str]) -> List[str]:
        return [normalize_triple_id(t) for t in v]


# ============================================================================
# Schema export
# ============================================================================

def get_trmdd_schema() -> dict:
    """Return JSON schema for structured output enforcement."""
    return TRMDDAnalysisResponse.model_json_schema()