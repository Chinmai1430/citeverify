import os
import re
import time
from pathlib import Path
from collections import Counter
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

try:
    from backend.schemas import (
        AtomicPropositionResult,
        ClaimDecompositionData,
        ClaimStatus,
        ClaimVerificationResult,
        ConfidenceLevel,
        EvidenceRankingData,
        ExtractedClaim,
        RankedEvidenceItem,
        RetrievedSource,
        TemporalAssessmentData,
        UncertaintyCalibrationData,
        VerificationReport,
    )
    from backend.services.claim_decomposer import (
        PropositionVerdict,
        combine_proposition_verdicts,
        decompose_claim,
    )
    from backend.services.evidence_ranker import (
        EvidenceCandidate,
        RankedEvidence,
        rank_and_verify_evidence,
    )
    from backend.services.temporal_tracker import (
        PublicationStage,
        TemporalCitationProfile,
        assess_temporal_validity,
        fetch_temporal_profile_live,
    )
    from backend.services.uncertainty_calibrator import (
        calibrate_verification_result,
    )
except ImportError:
    from schemas import (
        AtomicPropositionResult,
        ClaimDecompositionData,
        ClaimStatus,
        ClaimVerificationResult,
        ConfidenceLevel,
        EvidenceRankingData,
        ExtractedClaim,
        RankedEvidenceItem,
        RetrievedSource,
        TemporalAssessmentData,
        UncertaintyCalibrationData,
        VerificationReport,
    )
    from services.claim_decomposer import (
        PropositionVerdict,
        combine_proposition_verdicts,
        decompose_claim,
    )
    from services.evidence_ranker import (
        EvidenceCandidate,
        RankedEvidence,
        rank_and_verify_evidence,
    )
    from services.temporal_tracker import (
        PublicationStage,
        TemporalCitationProfile,
        assess_temporal_validity,
        fetch_temporal_profile_live,
    )
    from services.uncertainty_calibrator import (
        calibrate_verification_result,
    )

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
load_dotenv()

PRIMARY_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
FALLBACK_GEMINI_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.6-flash")
SECONDARY_FALLBACK_MODEL = os.getenv("GEMINI_SECONDARY_FALLBACK_MODEL", "gemini-3.8-flash")

RETRY_SCHEDULE = [
    (PRIMARY_GEMINI_MODEL, 1.5),
    (PRIMARY_GEMINI_MODEL, 2.0),
    (FALLBACK_GEMINI_MODEL, 1.5),
    (SECONDARY_FALLBACK_MODEL, 0.0),
]

_INLINE_DOI_RE = re.compile(r"(10\.\d{4,9}/[^\s\"'<>]+)", re.IGNORECASE)


class _LLMVerdictOutput(BaseModel):
    """Strict structured output schema enforced on Gemini responses."""

    verdict: ClaimStatus
    explanation: str = Field(
        description="Concise 1-2 sentence explanation comparing the claim to the source evidence."
    )
    confidence: ConfidenceLevel
    uncertainty_note: str = Field(
        description="Clear note on evidence limitations, scope gaps, or why human review is helpful."
    )
    needs_human_review: bool
    extra_mismatch_flags: List[str] = Field(default_factory=list)


def _format_citation(claim: ExtractedClaim, source: RetrievedSource) -> str:
    """Builds a clean readable citation label from retrieved metadata or raw reference."""
    if source.found and source.title:
        lead_author = source.authors[0] if source.authors else "Unknown Author"
        et_al = " et al." if len(source.authors) > 1 else ""
        year_part = f" ({source.year})" if source.year else ""
        return f"{lead_author}{et_al}{year_part} — {source.title}"
    return claim.raw_reference or claim.inline_marker or "Unmatched Citation"


def _extract_clean_doi(claim: ExtractedClaim, source: RetrievedSource) -> Optional[str]:
    """Extracts a bare DOI from either source.doi_url or claim.raw_reference."""
    for candidate in (source.doi_url, claim.raw_reference):
        if not candidate:
            continue
        match = _INLINE_DOI_RE.search(candidate)
        if match:
            return match.group(1).rstrip(".,;)]}")
    return None


def _split_passage_into_candidates(source: RetrievedSource) -> List[EvidenceCandidate]:
    """Splits a retrieved abstract/passage into sentence-level candidates for Gap 1 ranking."""
    raw_text = (source.source_passage or "").strip()
    if not raw_text:
        return []

    raw_sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+", raw_text) if s.strip()
    ]

    merged: List[str] = []
    buffer = ""
    for sent in raw_sentences:
        if buffer:
            sent = f"{buffer} {sent}"
            buffer = ""
        if len(sent) < 35:
            buffer = sent
        else:
            merged.append(sent)
    if buffer:
        if merged:
            merged[-1] = f"{merged[-1]} {buffer}"
        else:
            merged.append(buffer)

    if not merged:
        merged = [raw_text]

    return [
        EvidenceCandidate(
            passage_id=f"p_{idx}",
            text=sentence,
            source_title=source.title,
            source_url=source.doi_url,
        )
        for idx, sentence in enumerate(merged[:12], start=1)
    ]


def _run_gap1_evidence_ranking(
    claim: ExtractedClaim,
    source: RetrievedSource,
) -> Tuple[Optional[EvidenceRankingData], List[RankedEvidence]]:
    """Runs Gap 1 hybrid BM25 + semantic RRF + cross-encoder + NLI ranking on retrieved text."""
    candidates = _split_passage_into_candidates(source)
    if not candidates:
        return None, []

    try:
        summary = rank_and_verify_evidence(
            claim_text=claim.claim_text,
            candidates=candidates,
            top_k=3,
            use_embeddings=False,
            use_llm_nli=False,
        )
    except Exception as exc:
        print(f"Gap 1 evidence ranking skipped due to error: {exc}")
        return None, []

    top = summary.top_evidence
    ranked_items = [
        RankedEvidenceItem(
            passage_id=item.passage_id,
            text=item.text,
            bm25_score=item.bm25_score,
            semantic_score=item.semantic_score,
            hybrid_rrf_score=item.hybrid_rrf_score,
            cross_encoder_score=item.cross_encoder_score,
            logical_stance=item.logical_stance.value,
            stance_confidence=item.stance_confidence.value,
            similarity_trap=item.similarity_trap,
            stance_rationale=item.stance_rationale,
        )
        for item in summary.ranked_candidates
    ]

    ranking_data = EvidenceRankingData(
        embedding_backend=summary.embedding_backend,
        nli_backend=summary.nli_backend,
        recommended_verdict=summary.recommended_verdict,
        similarity_trap_detected=summary.similarity_trap_detected,
        top_passage_text=top.text if top else None,
        top_bm25_score=top.bm25_score if top else 0.0,
        top_semantic_score=top.semantic_score if top else 0.0,
        top_rrf_score=top.hybrid_rrf_score if top else 0.0,
        top_cross_encoder_score=top.cross_encoder_score if top else 0.0,
        top_logical_stance=top.logical_stance.value if top else "NEUTRAL",
        explanation=summary.explanation,
        ranked_candidates=ranked_items,
    )
    return ranking_data, summary.ranked_candidates


def _attach_gap2_decomposition(
    claim: ExtractedClaim,
    source: RetrievedSource,
    result: ClaimVerificationResult,
) -> ClaimVerificationResult:
    """Runs Gap 2 atomic claim decomposition and evaluates each proposition independently."""
    try:
        decomposed = decompose_claim(claim, use_llm=False)
        if not decomposed.is_compound:
            result.claim_decomposition = ClaimDecompositionData(
                is_compound=False,
                decomposition_method=decomposed.decomposition_method,
                overall_verdict=result.verdict,
                overall_confidence=result.confidence,
                summary_explanation="Single atomic claim; no multi-proposition split required.",
                propositions=[
                    AtomicPropositionResult(
                        proposition_id=prop.proposition_id,
                        proposition_text=prop.text,
                        verdict=result.verdict,
                        confidence=result.confidence,
                        explanation=result.explanation,
                        preserved_qualifiers=list(prop.preserved_qualifiers),
                        mismatch_flags=list(result.mismatch_flags),
                    )
                    for prop in decomposed.propositions
                ],
            )
            return result

        candidates = _split_passage_into_candidates(source)
        prop_verdicts: List[PropositionVerdict] = []
        qualifiers_map = {
            prop.proposition_id: list(prop.preserved_qualifiers)
            for prop in decomposed.propositions
        }

        for prop in decomposed.propositions:
            if source.is_retracted:
                prop_verdicts.append(
                    PropositionVerdict(
                        proposition_id=prop.proposition_id,
                        proposition_text=prop.text,
                        verdict=ClaimStatus.POTENTIALLY_OUTDATED,
                        confidence=ConfidenceLevel.HIGH,
                        explanation="Cited publication has been formally retracted.",
                        mismatch_flags=["RETRACTED_SOURCE"],
                    )
                )
            elif not source.found or not candidates:
                prop_verdicts.append(
                    PropositionVerdict(
                        proposition_id=prop.proposition_id,
                        proposition_text=prop.text,
                        verdict=ClaimStatus.UNSUPPORTED,
                        confidence=ConfidenceLevel.LOW,
                        explanation="No accessible source passage available to verify this proposition.",
                        mismatch_flags=["NO_SOURCE_PASSAGE"],
                    )
                )
            else:
                ev_summary = rank_and_verify_evidence(
                    claim_text=prop.text,
                    candidates=candidates,
                    top_k=2,
                    use_embeddings=False,
                    use_llm_nli=False,
                )
                top_ev = ev_summary.top_evidence
                prop_flags = list(top_ev.diagnostic_flags) if top_ev else []
                prop_verdicts.append(
                    PropositionVerdict(
                        proposition_id=prop.proposition_id,
                        proposition_text=prop.text,
                        verdict=ev_summary.recommended_verdict,
                        confidence=ev_summary.recommended_confidence,
                        explanation=top_ev.stance_rationale if top_ev else ev_summary.explanation,
                        mismatch_flags=prop_flags,
                    )
                )

        combined = combine_proposition_verdicts(claim, prop_verdicts)

        for flag in combined.mismatch_flags:
            if flag == "PARTIAL_CONTRADICTION" and flag not in result.mismatch_flags:
                result.mismatch_flags.append(flag)

        result.claim_decomposition = ClaimDecompositionData(
            is_compound=True,
            decomposition_method=decomposed.decomposition_method,
            overall_verdict=combined.overall_verdict,
            overall_confidence=combined.overall_confidence,
            summary_explanation=combined.summary_explanation,
            propositions=[
                AtomicPropositionResult(
                    proposition_id=pv.proposition_id,
                    proposition_text=pv.proposition_text,
                    verdict=pv.verdict,
                    confidence=pv.confidence,
                    explanation=pv.explanation,
                    preserved_qualifiers=qualifiers_map.get(pv.proposition_id, []),
                    mismatch_flags=list(pv.mismatch_flags),
                )
                for pv in combined.proposition_verdicts
            ],
        )
    except Exception as exc:
        print(f"Gap 2 claim decomposition skipped due to error: {exc}")
    return result


def _attach_gap3_temporal_tracking(
    claim: ExtractedClaim,
    source: RetrievedSource,
    result: ClaimVerificationResult,
) -> ClaimVerificationResult:
    """Runs Gap 3 time-aware citation & consensus tracking without altering historical verdict."""
    if not source.found:
        return result

    try:
        clean_doi = _extract_clean_doi(claim, source)
        skip_live = os.getenv("DISABLE_LIVE_TEMPORAL_FETCH", "0") == "1"

        if not skip_live and (clean_doi or source.title):
            profile = fetch_temporal_profile_live(
                doi=clean_doi,
                title=source.title,
                timeout=4.5,
                max_citing_works=6,
            )
        else:
            profile = TemporalCitationProfile(
                title=source.title or "Unknown Source",
                doi=clean_doi,
                publication_year=source.year,
                publication_stage=(
                    PublicationStage.RETRACTED
                    if source.is_retracted
                    else PublicationStage.UNKNOWN
                ),
                is_retracted=source.is_retracted,
            )

        # Ensure known retriever metadata is preserved even if live lookup times out
        if profile.publication_year is None and source.year is not None:
            profile.publication_year = source.year
        if source.is_retracted:
            profile.is_retracted = True
            profile.publication_stage = PublicationStage.RETRACTED

        assessment = assess_temporal_validity(
            profile=profile,
            historical_verdict=result.verdict,
        )

        result.temporal_assessment = TemporalAssessmentData(
            historical_claim_verdict=assessment.historical_claim_verdict,
            replaces_historical_verdict=assessment.replaces_historical_verdict,
            temporal_status=assessment.temporal_status.value,
            cited_publication_year=assessment.cited_publication_year,
            citation_age_years=assessment.citation_age_years,
            publication_stage=assessment.publication_stage.value,
            has_newer_journal_version=assessment.has_newer_journal_version,
            has_correction=assessment.has_correction,
            is_retracted=assessment.is_retracted,
            subsequent_citation_count=assessment.subsequent_citation_count,
            confirming_studies_count=assessment.confirming_studies_count,
            challenging_studies_count=assessment.challenging_studies_count,
            systematic_reviews_count=assessment.systematic_reviews_count,
            chronological_timeline=list(assessment.chronological_timeline[:6]),
            temporal_summary=assessment.temporal_summary,
            temporal_flags=list(assessment.temporal_flags),
            needs_temporal_review=assessment.needs_temporal_review,
        )
    except Exception as exc:
        print(f"Gap 3 temporal tracking skipped due to error: {exc}")
    return result


def _attach_gap4_uncertainty(
    result: ClaimVerificationResult,
    source: RetrievedSource,
    ranked_candidates: List[RankedEvidence],
) -> ClaimVerificationResult:
    """Runs Gap 4 uncertainty calibration & abstention policy on any ClaimVerificationResult."""
    try:
        decision = calibrate_verification_result(
            result,
            source_accessible=source.found,
            evidence_scope=source.evidence_scope,
            ranked_evidence=ranked_candidates,
        )
        result.uncertainty_calibration = UncertaintyCalibrationData(
            action=decision.action.value,
            calibrated_confidence=decision.calibrated_confidence,
            calibrated_probability=decision.calibrated_probability,
            evidence_coverage_score=decision.signals.evidence_coverage_score,
            retrieval_confidence=decision.signals.retrieval_confidence,
            passage_agreement_score=decision.signals.passage_agreement_score,
            composite_reliability_score=decision.signals.composite_reliability_score,
            has_conflicting_passages=decision.signals.has_conflicting_passages,
            abstained=decision.abstained,
            user_message=decision.user_message,
            uncertainty_flags=list(decision.uncertainty_flags),
        )
        if decision.needs_human_review:
            result.needs_human_review = True
    except Exception as exc:
        print(f"Gap 4 uncertainty calibration skipped due to error: {exc}")
    return result


def _verify_single_claim_core(
    claim: ExtractedClaim,
    source: RetrievedSource,
    evidence_ranking: Optional[EvidenceRankingData],
) -> ClaimVerificationResult:
    """Core verification logic producing the base ClaimVerificationResult."""
    citation_label = _format_citation(claim, source)
    flags = list(source.mismatch_flags)

    if (
        evidence_ranking
        and evidence_ranking.similarity_trap_detected
        and "SIMILARITY_TRAP_CONTRADICTION" not in flags
    ):
        flags.append("SIMILARITY_TRAP_CONTRADICTION")

    # Rule 1: Paper not found in open scholarly indexes - never fabricate verification
    if not source.found:
        if "Reference not found in open scholarly indexes" not in flags:
            flags.append("Reference not found in open scholarly indexes")
        return ClaimVerificationResult(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            matched_citation=citation_label,
            evidence_passage=None,
            verdict=ClaimStatus.UNSUPPORTED,
            explanation=(
                "No matching publication was located in open scholarly databases "
                "(OpenAlex, CrossRef, or arXiv) for this reference."
            ),
            confidence=ConfidenceLevel.LOW,
            uncertainty_note=(
                "Absence from open indexes does not prove fabrication; the source may be "
                "a book, regional report, or formatting mismatch. Manual lookup is required."
            ),
            needs_human_review=True,
            source_url=source.doi_url,
            mismatch_flags=flags,
            evidence_ranking=evidence_ranking,
        )

    # Rule 2: Retracted source - immediately flag as potentially outdated / invalid
    if source.is_retracted:
        if "Retracted publication" not in flags:
            flags.append("Retracted publication")
        return ClaimVerificationResult(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            matched_citation=citation_label,
            evidence_passage=source.source_passage,
            verdict=ClaimStatus.POTENTIALLY_OUTDATED,
            explanation=(
                "The cited publication has been formally retracted by its publisher and "
                "cannot be relied upon as valid supporting evidence."
            ),
            confidence=ConfidenceLevel.HIGH,
            uncertainty_note=(
                "Flagged directly from scholarly retraction metadata. Verify the official "
                "retraction notice via the source link."
            ),
            needs_human_review=True,
            source_url=source.doi_url,
            mismatch_flags=flags,
            evidence_ranking=evidence_ranking,
        )

    # Rule 3: Source found, but no abstract/full-text passage was retrievable (e.g. strict paywall)
    if not source.source_passage or not source.source_passage.strip():
        if "No source text available (paywalled or metadata-only)" not in flags:
            flags.append("No source text available (paywalled or metadata-only)")
        return ClaimVerificationResult(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            matched_citation=citation_label,
            evidence_passage=None,
            verdict=ClaimStatus.UNSUPPORTED,
            explanation=(
                "The cited paper exists in scholarly indexes, but neither its abstract nor "
                "full text was accessible via open APIs to verify this specific claim."
            ),
            confidence=ConfidenceLevel.LOW,
            uncertainty_note=(
                "Metadata-only check due to paywall or missing abstract. A human reviewer "
                "must inspect the original paper directly."
            ),
            needs_human_review=True,
            source_url=source.doi_url,
            mismatch_flags=flags,
            evidence_ranking=evidence_ranking,
        )

    # Rule 4: Compare claim against retrieved passage using Gemini Structured JSON
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return ClaimVerificationResult(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            matched_citation=citation_label,
            evidence_passage=source.source_passage,
            verdict=ClaimStatus.UNSUPPORTED,
            explanation="Verification skipped because GEMINI_API_KEY is not configured.",
            confidence=ConfidenceLevel.LOW,
            uncertainty_note="Configure GEMINI_API_KEY in environment variables to enable AI comparison.",
            needs_human_review=True,
            source_url=source.doi_url,
            mismatch_flags=flags,
            evidence_ranking=evidence_ranking,
        )

    prompt = f"""You are a rigorous scientific integrity auditor for CiteVerify.
Compare the DOCUMENT CLAIM against the RETRIEVED SOURCE PASSAGE strictly and objectively.

DOCUMENT CLAIM:
"{claim.claim_text}"

RETRIEVED PAPER METADATA:
- Title: {source.title}
- Authors: {", ".join(source.authors) if source.authors else "Unknown"}
- Year: {source.year or "Unknown"}
- Evidence Scope Available: {source.evidence_scope}

RETRIEVED SOURCE PASSAGE:
"{source.source_passage}"

CLASSIFICATION RULES:
- SUPPORTED: The passage directly and accurately backs the claim (including exact numbers, metrics, and scope).
- PARTIALLY_SUPPORTED: The passage backs part of the claim, or shows general improvement/trend, but exaggerates a metric, omits key scope limits (e.g., pilot study), or leaves details unconfirmed.
- CONTRADICTORY: The passage reports numbers, methods, or conclusions that directly conflict with the claim.
- UNSUPPORTED: The passage is unrelated or does not mention the claimed finding at all.
- POTENTIALLY_OUTDATED: The claim relies on dated findings or superseded scope where newer context is required.

HONEST UNCERTAINTY RULES:
- Never treat an automated verdict as final academic judgment.
- If Evidence Scope Available is ABSTRACT_ONLY, explicitly mention in uncertainty_note that full text/methods were not inspected and set needs_human_review=True unless the abstract states the exact metric verbatim.
- Flag any numerical distortion, scope exaggeration, or mismatched citation in extra_mismatch_flags.
"""

    client = genai.Client(api_key=api_key)
    llm_result: Optional[_LLMVerdictOutput] = None
    last_exc: Optional[Exception] = None
    blocked_models = set()

    for attempt_idx, (model_name, retry_delay) in enumerate(RETRY_SCHEDULE, start=1):
        if model_name in blocked_models:
            continue
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=_LLMVerdictOutput,
                ),
            )
            llm_result = _LLMVerdictOutput.model_validate_json(response.text)
            break
        except Exception as exc:
            last_exc = exc
            status_code = getattr(exc, "code", None)
            print(
                f"Gemini API error (attempt {attempt_idx}/{len(RETRY_SCHEDULE)}, "
                f"model={model_name}, code={status_code}): {exc}"
            )
            if status_code in (400, 403, 404):
                blocked_models.add(model_name)
                continue
            if retry_delay > 0:
                time.sleep(retry_delay)

    if llm_result is None:
        err_name = type(last_exc).__name__ if last_exc else "UnknownError"
        err_code = getattr(last_exc, "code", None)
        code_suffix = f" {err_code}" if err_code else ""
        return ClaimVerificationResult(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            matched_citation=citation_label,
            evidence_passage=source.source_passage,
            verdict=ClaimStatus.UNSUPPORTED,
            explanation="Automated evidence comparison encountered a temporary API error.",
            confidence=ConfidenceLevel.LOW,
            uncertainty_note=(
                f"Fallback triggered ({err_name}{code_suffix}). Manual review required."
            ),
            needs_human_review=True,
            source_url=source.doi_url,
            mismatch_flags=flags,
            evidence_ranking=evidence_ranking,
        )

    calibrated_confidence = llm_result.confidence
    needs_review = llm_result.needs_human_review
    uncertainty_note = llm_result.uncertainty_note

    if source.evidence_scope == "ABSTRACT_ONLY":
        if calibrated_confidence == ConfidenceLevel.HIGH and llm_result.verdict == ClaimStatus.SUPPORTED:
            calibrated_confidence = ConfidenceLevel.MEDIUM
        needs_review = True
        if "abstract" not in uncertainty_note.lower():
            uncertainty_note += " (Evaluated using abstract only; verify against full text.)"

    combined_flags = list(dict.fromkeys(flags + llm_result.extra_mismatch_flags))

    return ClaimVerificationResult(
        claim_id=claim.claim_id,
        claim_text=claim.claim_text,
        matched_citation=citation_label,
        evidence_passage=source.source_passage,
        verdict=llm_result.verdict,
        explanation=llm_result.explanation,
        confidence=calibrated_confidence,
        uncertainty_note=uncertainty_note,
        needs_human_review=needs_review,
        source_url=source.doi_url,
        mismatch_flags=combined_flags,
        evidence_ranking=evidence_ranking,
    )


def verify_single_claim(
    claim: ExtractedClaim,
    source: RetrievedSource,
) -> ClaimVerificationResult:
    """Compares one extracted claim against its retrieved source with Gaps 1, 2, 3 & 4."""
    evidence_ranking, ranked_candidates = _run_gap1_evidence_ranking(claim, source)
    result = _verify_single_claim_core(claim, source, evidence_ranking)
    result = _attach_gap2_decomposition(claim, source, result)
    result = _attach_gap3_temporal_tracking(claim, source, result)
    result = _attach_gap4_uncertainty(result, source, ranked_candidates)
    return result


def build_verification_report(
    document_title: str,
    claim_source_pairs: List[Tuple[ExtractedClaim, RetrievedSource]],
) -> VerificationReport:
    """Runs claim-by-claim verification and compiles the final structured VerificationReport."""
    results: List[ClaimVerificationResult] = []
    abstract_only_count = 0
    not_found_count = 0

    for claim, source in claim_source_pairs:
        if not source.found:
            not_found_count += 1
        elif source.evidence_scope == "ABSTRACT_ONLY":
            abstract_only_count += 1
        results.append(verify_single_claim(claim, source))

    counts = Counter(r.verdict.value for r in results)
    summary_counts = {status.value: counts.get(status.value, 0) for status in ClaimStatus}

    limitations = [
        "Automated verification is decision support, not a final academic judgment; human review is recommended."
    ]
    if abstract_only_count > 0:
        limitations.append(
            f"{abstract_only_count} source(s) were verified using abstract-only text due to paywall or open-access limits."
        )
    if not_found_count > 0:
        limitations.append(
            f"{not_found_count} reference(s) could not be matched in OpenAlex, CrossRef, or arXiv and require manual lookup."
        )

    return VerificationReport(
        document_title=document_title,
        total_claims=len(results),
        summary_counts=summary_counts,
        results=results,
        limitations=limitations,
    )