import os
import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import httpx
from pydantic import BaseModel, Field

try:
    from backend.schemas import ClaimStatus
except ImportError:
    from schemas import ClaimStatus

OPENALEX_WORKS_API = "https://api.openalex.org/works"
CROSSREF_WORKS_API = "https://api.crossref.org/works"

_DOI_PREFIX_RE = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", re.IGNORECASE)

_SYSTEMATIC_REVIEW_RE = re.compile(
    r"\b(?:systematic\s+review|cochrane\s+review|umbrella\s+review|scoping\s+review)\b",
    re.IGNORECASE,
)
_META_ANALYSIS_RE = re.compile(r"\bmeta[-\s]?analysis\b", re.IGNORECASE)
_REPLICATION_RE = re.compile(
    r"\b(?:replication|reproduc(?:e|ed|ing|ibility)|external\s+validation)\b",
    re.IGNORECASE,
)

_CHALLENGE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bno\s+significant\s+(?:benefit|effect|difference|improvement|reduction)\b", re.IGNORECASE),
    re.compile(r"\bfailed\s+to\s+(?:replicate|reproduce|confirm|demonstrate|show)\b", re.IGNORECASE),
    re.compile(r"\bdid\s+not\s+(?:confirm|replicate|improve|reduce|find\s+a\s+benefit)\b", re.IGNORECASE),
    re.compile(r"\b(?:contradicts?|refutes?|overturns?|challenges?|not\s+effective|ineffective|harmful)\b", re.IGNORECASE),
)

_INSUFFICIENT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\binsufficient\s+evidence\b", re.IGNORECASE),
    re.compile(r"\binconclusive\b", re.IGNORECASE),
    re.compile(r"\bno\s+clear\s+evidence\b", re.IGNORECASE),
    re.compile(r"\bvery\s+low[-\s]certainty\s+evidence\b", re.IGNORECASE),
    re.compile(r"\blittle\s+or\s+no\s+(?:evidence|effect|benefit)\b", re.IGNORECASE),
)

_CONFIRM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:confirms?|confirmed|replicated|consistent\s+with|corroborates?|supports?\s+prior)\b", re.IGNORECASE),
    re.compile(r"\bsignificant\s+(?:benefit|improvement|reduction|efficacy)\b", re.IGNORECASE),
)

_PREPRINT_VENUES: tuple[str, ...] = (
    "arxiv",
    "biorxiv",
    "medrxiv",
    "ssrn",
    "research square",
    "chemrxiv",
    "psyarxiv",
    "preprint",
)


class PublicationStage(StrEnum):
    """Lifecycle stage of the cited scholarly record."""

    PREPRINT = "PREPRINT"
    PEER_REVIEWED_JOURNAL = "PEER_REVIEWED_JOURNAL"
    CONFERENCE = "CONFERENCE"
    SYSTEMATIC_REVIEW = "SYSTEMATIC_REVIEW"
    RETRACTED = "RETRACTED"
    CORRECTED = "CORRECTED"
    UNKNOWN = "UNKNOWN"


class SubsequentStance(StrEnum):
    """Relationship of a newer citing study or review to the originally cited finding."""

    CONFIRMS = "CONFIRMS"
    CHALLENGES = "CHALLENGES"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NEUTRAL_CITATION = "NEUTRAL_CITATION"


class TemporalConsensusStatus(StrEnum):
    """Separate temporal validity classification that never overwrites the historical verdict."""

    CONSENSUS_ALIGNED = "CONSENSUS_ALIGNED"
    SUPERSEDED_OR_CHALLENGED = "SUPERSEDED_OR_CHALLENGED"
    RETRACTED = "RETRACTED"
    CORRECTED_OR_UPDATED = "CORRECTED_OR_UPDATED"
    STALE_NEEDS_UPDATE = "STALE_NEEDS_UPDATE"
    INSUFFICIENT_TEMPORAL_DATA = "INSUFFICIENT_TEMPORAL_DATA"


class VersionRecord(BaseModel):
    """Tracks preprint vs. published journal versions of the same work."""

    stage: PublicationStage
    doi_url: str | None = None
    publication_date: str | None = None
    venue: str | None = None
    is_latest: bool = False


class EditorialNotice(BaseModel):
    """Formal post-publication notice attached to a cited paper."""

    notice_type: str  # "retraction", "correction", "expression_of_concern"
    date: str | None = None
    detail: str
    notice_url: str | None = None


class SubsequentStudy(BaseModel):
    """Newer study, replication, or systematic review citing or evaluating the cited paper."""

    title: str
    year: int
    study_type: str = "primary_study"  # "primary_study", "replication", "systematic_review", "meta_analysis"
    stance: SubsequentStance = SubsequentStance.NEUTRAL_CITATION
    summary: str = ""
    doi_url: str | None = None
    citation_count: int = 0


class TemporalCitationProfile(BaseModel):
    """Structured temporal history collected for a single cited source."""

    title: str
    doi: str | None = None
    publication_year: int | None = None
    publication_date: str | None = None
    publication_stage: PublicationStage = PublicationStage.UNKNOWN
    is_retracted: bool = False
    cited_by_count: int = 0
    versions: list[VersionRecord] = Field(default_factory=list)
    editorial_notices: list[EditorialNotice] = Field(default_factory=list)
    subsequent_studies: list[SubsequentStudy] = Field(default_factory=list)


class TemporalAssessment(BaseModel):
    """Separate time-aware assessment alongside (never replacing) the historical verdict."""

    historical_claim_verdict: ClaimStatus
    replaces_historical_verdict: bool = False
    temporal_status: TemporalConsensusStatus
    cited_publication_year: int | None = None
    citation_age_years: int | None = None
    publication_stage: PublicationStage = PublicationStage.UNKNOWN
    has_newer_journal_version: bool = False
    has_correction: bool = False
    is_retracted: bool = False
    subsequent_citation_count: int = 0
    confirming_studies_count: int = 0
    challenging_studies_count: int = 0
    systematic_reviews_count: int = 0
    chronological_timeline: list[str] = Field(default_factory=list)
    temporal_summary: str
    temporal_flags: list[str] = Field(default_factory=list)
    needs_temporal_review: bool = False


# Inputs: Raw DOI string or DOI URL.
# Outputs: Normalized bare DOI (e.g., "10.1056/NEJMoa2021436") or None.
def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    cleaned = _DOI_PREFIX_RE.sub("", doi.strip()).strip().rstrip(".")
    return cleaned or None


# Inputs: Title and optional abstract text of a subsequent paper.
# Outputs: Study type label ("meta_analysis", "systematic_review", "replication", or "primary_study").
def detect_study_type(title: str, abstract: str = "") -> str:
    combined = f"{title} {abstract}".strip()
    if _META_ANALYSIS_RE.search(combined):
        return "meta_analysis"
    if _SYSTEMATIC_REVIEW_RE.search(combined):
        return "systematic_review"
    if _REPLICATION_RE.search(combined):
        return "replication"
    return "primary_study"


# Inputs: Text (title + abstract/snippet) of a newer citing paper.
# Outputs: SubsequentStance indicating whether it confirms, challenges, or finds insufficient evidence.
def classify_subsequent_stance(text: str) -> SubsequentStance:
    if not text or not text.strip():
        return SubsequentStance.NEUTRAL_CITATION

    if any(pattern.search(text) for pattern in _CHALLENGE_PATTERNS):
        return SubsequentStance.CHALLENGES
    if any(pattern.search(text) for pattern in _INSUFFICIENT_PATTERNS):
        return SubsequentStance.INSUFFICIENT_EVIDENCE
    if any(pattern.search(text) for pattern in _CONFIRM_PATTERNS):
        return SubsequentStance.CONFIRMS
    return SubsequentStance.NEUTRAL_CITATION


# Inputs: OpenAlex abstract_inverted_index dict.
# Outputs: Reconstructed plaintext abstract string.
def reconstruct_openalex_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    if not inverted_index:
        return ""
    positioned_words: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            positioned_words.append((pos, word))
    positioned_words.sort(key=lambda item: item[0])
    return " ".join(word for _, word in positioned_words).strip()


# Inputs: TemporalCitationProfile, original ClaimStatus verdict, optional reference_year and staleness cutoff.
# Outputs: Separate TemporalAssessment that preserves historical_claim_verdict untouched.
def assess_temporal_validity(
    profile: TemporalCitationProfile,
    historical_verdict: ClaimStatus,
    reference_year: int | None = None,
    staleness_threshold_years: int = 8,
) -> TemporalAssessment:
    current_year = reference_year if reference_year is not None else datetime.now(UTC).year
    pub_year = profile.publication_year
    if pub_year is None and profile.publication_date:
        year_match = re.match(r"((?:19|20)\d{2})", profile.publication_date.strip())
        if year_match:
            pub_year = int(year_match.group(1))

    age_years = max(0, current_year - pub_year) if pub_year is not None else None

    flags: list[str] = []
    notice_types = {notice.notice_type.lower().strip() for notice in profile.editorial_notices}
    is_retracted = (
        profile.is_retracted
        or profile.publication_stage == PublicationStage.RETRACTED
        or "retraction" in notice_types
    )
    has_correction = (
        profile.publication_stage == PublicationStage.CORRECTED
        or "correction" in notice_types
        or "erratum" in notice_types
    )
    has_concern = "expression_of_concern" in notice_types

    has_newer_journal_version = profile.publication_stage == PublicationStage.PREPRINT and any(
        v.stage == PublicationStage.PEER_REVIEWED_JOURNAL for v in profile.versions
    )

    if is_retracted:
        flags.append("RETRACTED_STUDY")
    if has_correction:
        flags.append("CORRECTION_ISSUED")
    if has_concern:
        flags.append("EXPRESSION_OF_CONCERN")
    if has_newer_journal_version:
        flags.append("PREPRINT_SUPERSEDED")
    elif profile.publication_stage == PublicationStage.PREPRINT:
        flags.append("PREPRINT_UNPUBLISHED")

    # Sort subsequent studies chronologically and keep those published in or after the cited study's year
    ordered_studies = sorted(profile.subsequent_studies, key=lambda s: s.year)
    newer_studies = [
        s for s in ordered_studies if pub_year is None or s.year >= pub_year
    ]

    confirming = [s for s in newer_studies if s.stance == SubsequentStance.CONFIRMS]
    challenging = [
        s for s in newer_studies
        if s.stance in {SubsequentStance.CHALLENGES, SubsequentStance.INSUFFICIENT_EVIDENCE}
    ]
    reviews = [
        s for s in newer_studies
        if s.study_type in {"systematic_review", "meta_analysis"}
    ]
    challenging_reviews = [
        s for s in reviews
        if s.stance in {SubsequentStance.CHALLENGES, SubsequentStance.INSUFFICIENT_EVIDENCE}
    ]
    confirming_reviews = [
        s for s in reviews if s.stance == SubsequentStance.CONFIRMS
    ]

    if challenging:
        flags.append("SUBSEQUENT_CHALLENGE")
    if challenging_reviews:
        flags.append("SYSTEMATIC_REVIEW_CONFLICT")

    # Build chronological timeline for transparent audit reporting
    timeline: list[str] = []
    if pub_year is not None:
        stage_label = profile.publication_stage.value.replace("_", " ").lower()
        timeline.append(f"{pub_year}: Cited study published ({stage_label}) — {profile.title}")
    for version in profile.versions:
        if version.stage != profile.publication_stage:
            v_date = version.publication_date or "undated"
            venue_txt = f" in {version.venue}" if version.venue else ""
            timeline.append(f"{v_date}: Alternate version ({version.stage.value}){venue_txt}")
    for notice in profile.editorial_notices:
        n_date = notice.date or "undated"
        timeline.append(f"{n_date}: Editorial notice [{notice.notice_type.upper()}] — {notice.detail}")
    for study in newer_studies:
        type_label = study.study_type.replace("_", " ")
        detail = f" ({study.summary})" if study.summary else ""
        timeline.append(
            f"{study.year}: Subsequent {type_label} [{study.stance.value}] — {study.title}{detail}"
        )

    # Determine separate TemporalConsensusStatus without altering historical_claim_verdict
    if is_retracted:
        status = TemporalConsensusStatus.RETRACTED
        needs_review = True
        summary = (
            "The cited study has been formally retracted after publication. "
            "Historical claim alignment is preserved for audit traceability, "
            "but the citation is no longer scientifically valid."
        )
    elif challenging_reviews or (
        len(challenging) > 0 and len(challenging) >= len(confirming)
    ):
        status = TemporalConsensusStatus.SUPERSEDED_OR_CHALLENGED
        needs_review = True
        latest_challenge_year = max(s.year for s in challenging)
        review_note = (
            f" including {len(challenging_reviews)} updated systematic review/meta-analysis"
            if challenging_reviews
            else ""
        )
        if historical_verdict == ClaimStatus.SUPPORTED:
            summary = (
                "The cited study supports the historical claim, but subsequent research has "
                f"challenged the finding ({len(challenging)} conflicting follow-up study/studies "
                f"through {latest_challenge_year}{review_note})."
            )
        else:
            summary = (
                f"Subsequent research through {latest_challenge_year} has challenged the cited study "
                f"({len(challenging)} conflicting follow-up study/studies{review_note})."
            )
    elif has_newer_journal_version or has_correction or has_concern:
        status = TemporalConsensusStatus.CORRECTED_OR_UPDATED
        needs_review = True
        reasons: list[str] = []
        if has_newer_journal_version:
            reasons.append("a newer peer-reviewed journal version supersedes the cited preprint")
        if has_correction:
            reasons.append("a formal post-publication correction/erratum was issued")
        if has_concern:
            reasons.append("an editorial expression of concern was published")
        summary = (
            "Post-publication updates exist for the cited work: "
            + "; ".join(reasons)
            + ". Verify the claim against the latest authoritative record."
        )
    elif confirming_reviews or len(confirming) > 0:
        status = TemporalConsensusStatus.CONSENSUS_ALIGNED
        needs_review = False
        latest_confirm_year = max(s.year for s in confirming)
        summary = (
            f"Subsequent literature through {latest_confirm_year} ({len(confirming)} confirming "
            f"study/studies, {len(confirming_reviews)} systematic review/meta-analysis) "
            "remains aligned with the cited study."
        )
    elif profile.publication_stage == PublicationStage.PREPRINT:
        status = TemporalConsensusStatus.STALE_NEEDS_UPDATE
        needs_review = True
        summary = (
            "The cited source is a preprint with no confirmed peer-reviewed journal version on record. "
            "Check whether a final peer-reviewed publication or updated findings exist."
        )
    elif age_years is not None and age_years >= staleness_threshold_years:
        flags.append("AGING_CITATION")
        status = TemporalConsensusStatus.STALE_NEEDS_UPDATE
        needs_review = True
        summary = (
            f"The cited study is {age_years} years old (published {pub_year}) with no tracked "
            "recent confirming systematic reviews. Consider checking for newer consensus evidence."
        )
    elif pub_year is None and not newer_studies and profile.cited_by_count == 0:
        status = TemporalConsensusStatus.INSUFFICIENT_TEMPORAL_DATA
        needs_review = True
        summary = "Insufficient publication date and subsequent citation metadata to assess temporal consensus."
    else:
        status = TemporalConsensusStatus.CONSENSUS_ALIGNED
        needs_review = False
        summary = (
            f"Cited study ({pub_year}) shows no retractions, corrections, or conflicting "
            "subsequent evidence in tracked literature."
        )

    return TemporalAssessment(
        historical_claim_verdict=historical_verdict,
        replaces_historical_verdict=False,
        temporal_status=status,
        cited_publication_year=pub_year,
        citation_age_years=age_years,
        publication_stage=profile.publication_stage,
        has_newer_journal_version=has_newer_journal_version,
        has_correction=has_correction,
        is_retracted=is_retracted,
        subsequent_citation_count=profile.cited_by_count,
        confirming_studies_count=len(confirming),
        challenging_studies_count=len(challenging),
        systematic_reviews_count=len(reviews),
        chronological_timeline=timeline,
        temporal_summary=summary,
        temporal_flags=flags,
        needs_temporal_review=needs_review,
    )


# Inputs: OpenAlex primary work JSON payload and optional list of OpenAlex citing-work payloads.
# Outputs: Populated TemporalCitationProfile extracted from OpenAlex metadata.
def build_profile_from_openalex_payload(
    work: dict[str, Any],
    citing_works: list[dict[str, Any]] | None = None,
    crossref_message: dict[str, Any] | None = None,
) -> TemporalCitationProfile:
    title = str(work.get("display_name") or work.get("title") or "Untitled Work").strip()
    doi = normalize_doi(work.get("doi"))
    pub_year = work.get("publication_year")
    pub_date = work.get("publication_date")
    is_retracted = bool(work.get("is_retracted", False))
    cited_by_count = int(work.get("cited_by_count") or 0)

    primary_loc = work.get("primary_location") or {}
    source_obj = primary_loc.get("source") or {}
    venue_name = str(source_obj.get("display_name") or "").strip()
    work_type = str(work.get("type") or "").lower()

    if is_retracted:
        stage = PublicationStage.RETRACTED
    elif work_type == "preprint" or any(p in venue_name.lower() for p in _PREPRINT_VENUES):
        stage = PublicationStage.PREPRINT
    elif detect_study_type(title) in {"systematic_review", "meta_analysis"}:
        stage = PublicationStage.SYSTEMATIC_REVIEW
    elif work_type in {"proceedings-article", "conference-paper"}:
        stage = PublicationStage.CONFERENCE
    elif work_type in {"article", "journal-article", "review"}:
        stage = PublicationStage.PEER_REVIEWED_JOURNAL
    else:
        stage = PublicationStage.UNKNOWN

    versions: list[VersionRecord] = []
    notices: list[EditorialNotice] = []

    if is_retracted:
        notices.append(
            EditorialNotice(
                notice_type="retraction",
                date=pub_date,
                detail="Marked as retracted in OpenAlex scholarly metadata.",
                notice_url=work.get("doi"),
            )
        )

    # Parse Crossref editorial notices and preprint -> journal relations when provided
    if crossref_message:
        for update in crossref_message.get("updated-by") or []:
            u_type = str(update.get("type") or "correction").lower()
            label = str(update.get("label") or u_type)
            updated_parts = ((update.get("updated") or {}).get("date-parts") or [[]])[0]
            u_date = "-".join(str(p) for p in updated_parts) if updated_parts else None
            u_doi = normalize_doi(update.get("DOI"))
            notices.append(
                EditorialNotice(
                    notice_type="retraction" if "retract" in u_type else (
                        "expression_of_concern" if "concern" in u_type else "correction"
                    ),
                    date=u_date,
                    detail=f"Crossref update notice: {label}",
                    notice_url=f"https://doi.org/{u_doi}" if u_doi else None,
                )
            )
            if "retract" in u_type:
                is_retracted = True

        relations = crossref_message.get("relation") or {}
        for rel_item in relations.get("is-preprint-of") or []:
            rel_doi = normalize_doi(rel_item.get("id"))
            versions.append(
                VersionRecord(
                    stage=PublicationStage.PEER_REVIEWED_JOURNAL,
                    doi_url=f"https://doi.org/{rel_doi}" if rel_doi else None,
                    venue="Peer-reviewed published version (Crossref relation)",
                    is_latest=True,
                )
            )

    # Parse subsequent citing studies and systematic reviews
    subsequent: list[SubsequentStudy] = []
    for citing in citing_works or []:
        c_title = str(citing.get("display_name") or citing.get("title") or "").strip()
        c_year = citing.get("publication_year")
        if not c_title or not isinstance(c_year, int):
            continue
        c_abstract = reconstruct_openalex_abstract(citing.get("abstract_inverted_index"))
        combined_text = f"{c_title}. {c_abstract}".strip()
        subsequent.append(
            SubsequentStudy(
                title=c_title,
                year=c_year,
                study_type=detect_study_type(c_title, c_abstract),
                stance=classify_subsequent_stance(combined_text),
                summary=c_abstract[:200].strip() if c_abstract else "",
                doi_url=citing.get("doi"),
                citation_count=int(citing.get("cited_by_count") or 0),
            )
        )

    return TemporalCitationProfile(
        title=title,
        doi=doi,
        publication_year=pub_year if isinstance(pub_year, int) else None,
        publication_date=str(pub_date) if pub_date else None,
        publication_stage=stage,
        is_retracted=is_retracted,
        cited_by_count=cited_by_count,
        versions=versions,
        editorial_notices=notices,
        subsequent_studies=subsequent,
    )


# Inputs: DOI or paper title, timeout seconds, and max citing works to inspect via OpenAlex + Crossref.
# Outputs: Populated TemporalCitationProfile, degrading gracefully to an empty profile on network failure.
def fetch_temporal_profile_live(
    doi: str | None = None,
    title: str | None = None,
    timeout: float = 10.0,
    max_citing_works: int = 10,
) -> TemporalCitationProfile:
    norm_doi = normalize_doi(doi)
    contact_email = os.getenv("CONTACT_EMAIL", "").strip()
    headers = {
        "User-Agent": (
            f"CiteVerify/1.0 (mailto:{contact_email})"
            if contact_email
            else "CiteVerify/1.0"
        )
    }

    work_payload: dict[str, Any] | None = None
    citing_payloads: list[dict[str, Any]] = []
    crossref_message: dict[str, Any] | None = None

    try:
        with httpx.Client(timeout=timeout, headers=headers, follow_redirects=True) as client:
            if norm_doi:
                oa_resp = client.get(f"{OPENALEX_WORKS_API}/https://doi.org/{norm_doi}")
                if oa_resp.status_code == 200:
                    work_payload = oa_resp.json()
            elif title:
                oa_resp = client.get(
                    OPENALEX_WORKS_API,
                    params={"search": title, "per-page": 1},
                )
                if oa_resp.status_code == 200:
                    results = oa_resp.json().get("results") or []
                    if results:
                        work_payload = results[0]

            if work_payload:
                effective_doi = norm_doi or normalize_doi(work_payload.get("doi"))
                cited_by_url = work_payload.get("cited_by_api_url")
                if cited_by_url and max_citing_works > 0:
                    citing_resp = client.get(
                        cited_by_url,
                        params={
                            "per-page": max_citing_works,
                            "sort": "publication_year:desc",
                        },
                    )
                    if citing_resp.status_code == 200:
                        citing_payloads = citing_resp.json().get("results") or []

                if effective_doi:
                    cr_resp = client.get(f"{CROSSREF_WORKS_API}/{effective_doi}")
                    if cr_resp.status_code == 200:
                        crossref_message = cr_resp.json().get("message")
    except (httpx.HTTPError, ValueError, KeyError, OSError):
        pass

    if not work_payload:
        return TemporalCitationProfile(
            title=title or norm_doi or "Unknown Source",
            doi=norm_doi,
        )

    return build_profile_from_openalex_payload(
        work_payload,
        citing_works=citing_payloads,
        crossref_message=crossref_message,
    )