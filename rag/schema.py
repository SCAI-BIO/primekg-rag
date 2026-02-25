from typing import List, Optional
from pydantic import BaseModel, Field



class Mechanism(BaseModel):
    """A specific biological mechanism linking the gene to TR-MDD."""
    name: str = Field(description="Short name of the mechanism")

    description: str = Field(
        description="How this mechanism contributes to treatment-resistant depression"
    )

    evidence_pmids: List[str] = Field(
        description="PMIDs supporting this mechanism",
        default_factory=list
    )


class PathwayUsage(BaseModel):
    """Knowledge Graph pathway documentation."""
    source: str = Field(description="Source gene")

    intermediate: str = Field(
        description="Intermediate node connecting gene to MDD"
    )

    target: str = Field(
        description="Target disease (MDD)"
    )

    relevance: str = Field(
        description="Why this pathway is relevant to TR-MDD"
    )


class Citation(BaseModel):
    """Claim-level citation grounding."""
    claim: str = Field(description="Specific claim being made")

    evidence_pmid: str = Field(
        description="PMID supporting this claim"
    )

    quote: Optional[str] = Field(
        description="Short quote from abstract",
        default=None
    )


class TRMDDAnalysisResponse(BaseModel):
    """
    Structured response for TR-MDD gene analysis.
    Supports automated grounding + evidence evaluation.
    """

    # -------------------------------------------------------------------------
    # Identification
    # -------------------------------------------------------------------------
    gene: str = Field(description="Gene symbol being analyzed")

    
    answer: str = Field(
        description="Comprehensive analysis (150–350 words) with inline PMID citations"
    )

    # -------------------------------------------------------------------------
    # Mechanistic evidence
    # -------------------------------------------------------------------------
    key_mechanisms: List[Mechanism] = Field(
        description="Mechanisms linking this gene to TR-MDD",
        min_items=1
    )

    kg_pathways_used: List[PathwayUsage] = Field(
        description="Knowledge graph pathways used",
        default_factory=list
    )

    
    citations: List[Citation] = Field(
        description="Specific claims with evidence",
        min_items=1
    )

    evidence_pmids: List[str] = Field(
        description="All PMIDs cited in the analysis",
        min_items=1
    )

   
    limitations: str = Field(
        description="Gaps, inconsistencies, or evidence limitations"
    )

    clinical_relevance: str = Field(
        description="Clinical implications for TR-MDD"
    )




def get_json_schema() -> dict:
    """Return JSON schema for Ollama structured output enforcement."""
    return TRMDDAnalysisResponse.model_json_schema()



if __name__ == "__main__":
    import json
    print(json.dumps(get_json_schema(), indent=2))
