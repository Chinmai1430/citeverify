import os
import time
from collections import Counter
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

try:
    from backend.schemas import (
        ClaimStatus,
        ClaimVerificationResult,
        ConfidenceLevel,
        ExtractedClaim,
        RetrievedSource,
        VerificationReport,
    )
except ImportError:
    from schemas import (
        ClaimStatus,
        ClaimVerificationResult,
        ConfidenceLevel,
        ExtractedClaim,
        RetrievedSource,
        VerificationReport,
    )

load_dotenv()

PRIMARY_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
FALLBACK_GEMINI_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash")
RETRY_SCHEDULE = [
    (PRIMARY_GEMINI_MODEL, 1.0),
    (PRIMARY_GEMINI_MODEL, 2.0),
    (FALLBACK_GEMINI_MODEL, 0.0),
]


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


def verify_single_claim(
    claim: ExtractedClaim,
    source: RetrievedSource,
) -> ClaimVerificationResult:
    """Compares one extracted claim against its retrieved source and returns a calibrated verdict."""
    citation_label = _format_citation(claim, source)
    flags = list(source.mismatch_flags)

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

    for attempt_idx, (model_name, retry_delay) in enumerate(RETRY_SCHEDULE, start=1):
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
            print(
                f"Gemini API error (attempt {attempt_idx}/{len(RETRY_SCHEDULE)}, "
                f"model={model_name}): {exc}"
            )
            if retry_delay > 0:
                time.sleep(retry_delay)

    if llm_result is None:
        return ClaimVerificationResult(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            matched_citation=citation_label,
            evidence_passage=source.source_passage,
            verdict=ClaimStatus.UNSUPPORTED,
            explanation="Automated evidence comparison encountered a temporary API error.",
            confidence=ConfidenceLevel.LOW,
            uncertainty_note=(
                f"Fallback triggered ({type(last_exc).__name__ if last_exc else 'UnknownError'}). "
                "Manual review required."
            ),
            needs_human_review=True,
            source_url=source.doi_url,
            mismatch_flags=flags,
        )

    # Calibrate uncertainty honestly for paywalled / abstract-only evidence
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
    )


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