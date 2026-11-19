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


class RankedEvidenceItem(BaseModel):
    """Single ranked evidence passage from Gap 1 hybrid retrieval & NLI."""
    passage_id: str
    text: str
    bm25_score: float = 0.0
    semantic_score: float = 0.0
    hybrid_rrf_score: float = 0.0
    cross_encoder_score: float = 0.0
    logical_stance: str = "NEUTRAL"
    stance_confidence: str = "LOW"
    similarity_trap: bool = False
    stance_rationale: str = ""


class EvidenceRankingData(BaseModel):
    """Gap 1 hybrid BM25 + semantic RRF + cross-encoder + NLI output for a claim."""
    embedding_backend: str = "tfidf_fallback"
    nli_backend: str = "rule_based_nli"
    recommended_verdict: ClaimStatus = ClaimStatus.UNSUPPORTED
    similarity_trap_detected: bool = False
    top_passage_text: Optional[str] = None
    top_bm25_score: float = 0.0
    top_semantic_score: float = 0.0
    top_rrf_score: float = 0.0
    top_cross_encoder_score: float = 0.0
    top_logical_stance: str = "NEUTRAL"
    explanation: str = ""
    ranked_candidates: List[RankedEvidenceItem] = Field(default_factory=list)


class AtomicPropositionResult(BaseModel):
    """Single atomic proposition and its independent verification outcome (Gap 2)."""
    proposition_id: str
    proposition_text: str
    verdict: ClaimStatus
    confidence: ConfidenceLevel
    explanation: str
    preserved_qualifiers: List[str] = Field(default_factory=list)
    mismatch_flags: List[str] = Field(default_factory=list)


class ClaimDecompositionData(BaseModel):
    """Gap 2 atomic claim decomposition and aggregated proposition verdict."""
    is_compound: bool = False
    decomposition_method: str = "rule_based"
    overall_verdict: ClaimStatus = ClaimStatus.UNSUPPORTED
    overall_confidence: ConfidenceLevel = ConfidenceLevel.LOW
    summary_explanation: str = ""
    propositions: List[AtomicPropositionResult] = Field(default_factory=list)


class TemporalAssessmentData(BaseModel):
    """Gap 3 time-aware citation verification & evolving consensus tracking."""
    historical_claim_verdict: ClaimStatus
    replaces_historical_verdict: bool = False
    temporal_status: str = "INSUFFICIENT_TEMPORAL_DATA"
    cited_publication_year: Optional[int] = None
    citation_age_years: Optional[int] = None
    publication_stage: str = "UNKNOWN"
    has_newer_journal_version: bool = False
    has_correction: bool = False
    is_retracted: bool = False
    subsequent_citation_count: int = 0
    confirming_studies_count: int = 0
    challenging_studies_count: int = 0
    systematic_reviews_count: int = 0
    chronological_timeline: List[str] = Field(default_factory=list)
    temporal_summary: str = ""
    temporal_flags: List[str] = Field(default_factory=list)
    needs_temporal_review: bool = False


class UncertaintyCalibrationData(BaseModel):
    """Gap 4 uncertainty-aware calibration, reliability signals, and abstention policy."""
    action: str = "FLAG_FOR_REVIEW"  # PRESENT_VERDICT, FLAG_FOR_REVIEW, ABSTAIN, MARK_UNVERIFIABLE
    calibrated_confidence: ConfidenceLevel = ConfidenceLevel.LOW
    calibrated_probability: float = 0.0
    evidence_coverage_score: float = 0.0
    retrieval_confidence: float = 0.0
    passage_agreement_score: float = 0.5
    composite_reliability_score: float = 0.0
    has_conflicting_passages: bool = False
    abstained: bool = False
    user_message: str = ""
    uncertainty_flags: List[str] = Field(default_factory=list)


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
    # Gap 1: Hybrid Evidence Ranking & NLI
    evidence_ranking: Optional[EvidenceRankingData] = None
    # Gap 2: Atomic Claim Decomposition
    claim_decomposition: Optional[ClaimDecompositionData] = None
    # Gap 3: Time-Aware Citation Verification & Consensus Tracking
    temporal_assessment: Optional[TemporalAssessmentData] = None
    # Gap 4: Uncertainty-Aware Calibration & Abstention
    uncertainty_calibration: Optional[UncertaintyCalibrationData] = None


class VerificationReport(BaseModel):
    """Top-level verification report returned by /api/verify."""
    document_title: str
    total_claims: int
    summary_counts: Dict[str, int]
    results: List[ClaimVerificationResult]
    limitations: List[str]