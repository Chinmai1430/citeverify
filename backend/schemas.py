from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ClaimStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTORY = "CONTRADICTORY"
    POTENTIALLY_OUTDATED = "POTENTIALLY_OUTDATED"


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ExtractedClaim(BaseModel):
    """Output produced by Chiranth's parser module for each claim in a document."""
    claim_id: int
    claim_text: str
    inline_marker: Optional[str] = None
    raw_reference: Optional[str] = None


class RetrievedSource(BaseModel):
    """Output produced by Rithvik's retriever module for each referenced paper."""
    found: bool
    title: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    doi_url: Optional[str] = None
    is_retracted: bool = False
    source_passage: Optional[str] = None
    evidence_scope: str = "NONE"  # FULL_TEXT, ABSTRACT_ONLY, METADATA_ONLY, NONE
    mismatch_flags: List[str] = Field(default_factory=list)


class ClaimVerificationResult(BaseModel):
    """Final claim-by-claim verdict consumed by Dhanush's frontend UI."""
    claim_id: int
    claim_text: str
    matched_citation: Optional[str] = None
    evidence_passage: Optional[str] = None
    verdict: ClaimStatus
    explanation: str
    confidence: ConfidenceLevel
    uncertainty_note: str
    needs_human_review: bool = True
    source_url: Optional[str] = None
    mismatch_flags: List[str] = Field(default_factory=list)


class VerificationReport(BaseModel):
    """Top-level verification report returned by /api/verify."""
    document_title: str
    total_claims: int
    summary_counts: Dict[str, int]
    results: List[ClaimVerificationResult]
    limitations: List[str]