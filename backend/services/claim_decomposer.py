import os
import re
from collections.abc import Callable

from pydantic import BaseModel, Field

try:
    from backend.schemas import (
        ClaimStatus,
        ClaimVerificationResult,
        ConfidenceLevel,
        ExtractedClaim,
    )
except ImportError:
    from schemas import (
        ClaimStatus,
        ClaimVerificationResult,
        ConfidenceLevel,
        ExtractedClaim,
    )

DEFAULT_DECOMPOSER_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# Trailing inline citation markers (e.g., [1], [1, 2], (Smith et al., 2020))
_TRAILING_CITATION_RE = re.compile(
    r"\s*((?:\[\d+(?:\s*[-,]\s*\d+)*\]|\([A-Z][A-Za-z\-]+(?:\s+et\s+al\.)?(?:\s*(?:&|and)\s*[A-Z][A-Za-z\-]+)?,\s*(?:19|20)\d{2}[a-z]?\))+)\s*[.!?]?$"
)

# Leading population, study-design, or conditional qualifiers (e.g., "In elderly patients, ...")
_LEADING_QUALIFIER_RE = re.compile(
    r"^((?:In|Among|For|During|Across|Within|When|After|Before|Under|Using|On)\s+[^,;]{3,90},)\s+(.+)$",
    re.IGNORECASE,
)

# Contrast / subordination splitters that introduce a second proposition
_SUBORDINATING_SPLIT_RE = re.compile(
    r",?\s+\b(while|whereas|although|even\s+though|though|yet)\b\s+",
    re.IGNORECASE,
)

# Semicolon + optional discourse connector splitters
_SEMICOLON_SPLIT_RE = re.compile(
    r"\s*;\s*(?:(?:however|moreover|furthermore|additionally|meanwhile|consequently|nevertheless),?\s+)?",
    re.IGNORECASE,
)

# Coordinating conjunctions only split when followed by a clause containing its own verb/gerund
_COORDINATING_SPLIT_RE = re.compile(
    r",?\s+\b(but|and)\b\s+(?=(?:also\s+|still\s+|simultaneously\s+|concurrently\s+)?)",
    re.IGNORECASE,
)

# Common academic verbs/auxiliaries used to locate subject boundaries and validate verb clauses
_FINITE_VERB_RE = re.compile(
    r"\b("
    r"is|are|was|were|be|been|has|have|had|can|could|may|might|should|will|would|"
    r"outperforms?|outperformed|improves?|improved|reduces?|reduced|increases?|increased|"
    r"decreases?|decreased|lowers?|lowered|raises?|raised|achieves?|achieved|"
    r"requires?|required|causes?|caused|shows?|showed|demonstrates?|demonstrated|"
    r"proves?|proved|indicates?|indicated|suggests?|suggested|maintains?|maintained|"
    r"preserves?|preserved|eliminates?|eliminated|accelerates?|accelerated|"
    r"doubles?|doubled|halves?|halved|cuts?|cut|drops?|dropped|rises?|rose|"
    r"reaches?|reached|exceeds?|exceeded|falls?|fell|leads?|led|produces?|produced|"
    r"uses?|used|provides?|provided|offers?|offered|yields?|yielded|fails?|failed|"
    r"lacks?|lacked|prevents?|prevented|triggers?|triggered|correlates?|correlated|"
    r"associates?|associated|replaces?|replaced|surpasses?|surpassed"
    r")\b",
    re.IGNORECASE,
)

_GERUND_MAP: dict[str, str] = {
    "requiring": "require",
    "reducing": "reduce",
    "improving": "improve",
    "increasing": "increase",
    "decreasing": "decrease",
    "lowering": "lower",
    "raising": "raise",
    "achieving": "achieve",
    "causing": "cause",
    "using": "use",
    "maintaining": "maintain",
    "preserving": "preserve",
    "providing": "provide",
    "producing": "produce",
    "eliminating": "eliminate",
    "accelerating": "accelerate",
    "outperforming": "outperform",
    "surpassing": "surpass",
    "cutting": "cut",
    "halving": "halve",
    "doubling": "double",
    "consuming": "consume",
    "saving": "save",
    "yielding": "yield",
    "offering": "offer",
    "preventing": "prevent",
    "triggering": "trigger",
    "showing": "show",
    "demonstrating": "demonstrate",
    "leading": "lead",
    "having": "have",
    "being": "be",
    "retaining": "retain",
    "incurring": "incur",
    "avoiding": "avoid",
}

_NUMBER_OR_QUALIFIER_RE = re.compile(
    r"\b\d+(?:\.\d+)?%?\b|\b(?:fewer|less|more|higher|lower|greater|faster|slower|only|significantly|statistically|prospective|retrospective)\b",
    re.IGNORECASE,
)


class AtomicProposition(BaseModel):
    """A single verifiable proposition extracted from a potentially compound claim."""

    proposition_id: str
    parent_claim_id: int
    text: str
    preserved_qualifiers: list[str] = Field(default_factory=list)
    inline_marker: str | None = None
    raw_reference: str | None = None


class DecomposedClaim(BaseModel):
    """Decomposition result mapping one ExtractedClaim to one or more atomic propositions."""

    parent_claim_id: int
    original_claim_text: str
    is_compound: bool
    decomposition_method: str  # "rule_based" or "llm"
    propositions: list[AtomicProposition]


class PropositionVerdict(BaseModel):
    """Verification outcome for a single atomic proposition."""

    proposition_id: str
    proposition_text: str
    verdict: ClaimStatus
    confidence: ConfidenceLevel
    explanation: str
    mismatch_flags: list[str] = Field(default_factory=list)


class CombinedClaimVerdict(BaseModel):
    """Aggregated verdict synthesized from all atomic propositions of a compound claim."""

    parent_claim_id: int
    original_claim_text: str
    overall_verdict: ClaimStatus
    overall_confidence: ConfidenceLevel
    needs_human_review: bool
    summary_explanation: str
    mismatch_flags: list[str] = Field(default_factory=list)
    proposition_verdicts: list[PropositionVerdict] = Field(default_factory=list)


class _LLMDecompositionSchema(BaseModel):
    """Strict structured output schema when Gemini-assisted decomposition is enabled."""

    propositions: list[str] = Field(
        description=(
            "List of self-contained atomic factual propositions extracted from the claim. "
            "Each proposition must preserve the original subject, numbers, populations, "
            "conditions, and qualifiers."
        )
    )


# Inputs: English gerund word (e.g., 'requiring') and whether the subject appears plural.
# Outputs: Finite present-tense verb form (e.g., 'require' or 'requires').
def _gerund_to_finite(gerund: str, subject_is_plural: bool) -> str:
    lower = gerund.lower()
    if lower in _GERUND_MAP:
        base = _GERUND_MAP[lower]
    elif lower.endswith("ing") and len(lower) > 5:
        stem = lower[:-3]
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        base = stem
    else:
        return gerund

    if subject_is_plural:
        return base
    if base == "have":
        return "has"
    if base == "be":
        return "is"
    if base.endswith(("s", "sh", "ch", "x", "z")):
        return f"{base}es"
    if base.endswith("y") and len(base) > 1 and base[-2] not in "aeiou":
        return f"{base[:-1]}ies"
    return f"{base}s"


# Inputs: Subject noun phrase string.
# Outputs: True if the subject appears grammatically plural.
def _is_plural_subject(subject: str) -> bool:
    tokens = re.findall(r"[A-Za-z]+", subject.lower())
    if not tokens:
        return False
    if "and" in tokens:
        return True
    head = tokens[-1]
    if head in {"data", "criteria", "people", "children", "men", "women", "patients", "models", "networks", "cnns"}:
        return True
    return head.endswith("s") and not head.endswith(("ss", "us", "is", "ics"))


# Inputs: A single clause string (without leading qualifier).
# Outputs: Extracted subject phrase preceding the first main verb, or None if not found.
def _extract_subject(clause: str) -> str | None:
    match = _FINITE_VERB_RE.search(clause)
    if not match:
        return None
    subject = clause[: match.start()].strip(" ,;:-")
    if not subject or len(subject.split()) > 10:
        return None
    return subject


# Inputs: Clause string, optional leading qualifier, optional inherited subject, and citation suffix.
# Outputs: Clean, grammatically complete standalone proposition sentence.
def _finalize_proposition(
    clause: str,
    qualifier: str | None,
    inherited_subject: str | None,
    citation_suffix: str | None,
) -> str:
    text = clause.strip(" ,;.")
    if not text:
        return ""

    # Strip leading discourse adverbs left over from conjunction splitting
    text = re.sub(
        r"^(?:also|still|simultaneously|concurrently|nevertheless|however)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    first_word = text.split()[0] if text.split() else ""
    first_word_clean = re.sub(r"[^A-Za-z]", "", first_word)

    # Case 1: Participial clause ("requiring fewer computational resources")
    if inherited_subject and first_word_clean.lower().endswith("ing") and (
        first_word_clean.lower() in _GERUND_MAP or len(first_word_clean) > 5
    ):
        plural = _is_plural_subject(inherited_subject)
        finite_verb = _gerund_to_finite(first_word_clean, subject_is_plural=plural)
        remainder = text[len(first_word) :].strip()
        text = f"{inherited_subject} {finite_verb} {remainder}".strip()

    # Case 2: Verb-headed predicate without subject ("lowered systolic blood pressure by 12 mmHg")
    elif inherited_subject and _FINITE_VERB_RE.match(text):
        text = f"{inherited_subject} {text}"

    # Case 3: Pronoun reference ("it also reduced...", "they require...")
    elif inherited_subject and re.match(r"^(?:it|they)\s+", text, flags=re.IGNORECASE):
        text = re.sub(r"^(?:it|they)\s+", f"{inherited_subject} ", text, count=1, flags=re.IGNORECASE)

    # Re-attach leading population/condition qualifier if missing from this clause
    if qualifier and not text.lower().startswith(qualifier.lower().rstrip(",")):
        if text and text[0].isupper() and not re.match(r"^(?:[A-Z]{2,}|CNN|GPU|ICU|AI|ML)\b", text):
            # Preserve proper nouns/acronyms; lowercase common noun phrases after qualifier comma
            first_token = text.split()[0]
            if first_token.lower() in {"the", "a", "an", "deep", "sparse", "transformer", "clinical", "increasing", "decreasing"}:
                text = text[0].lower() + text[1:]
        text = f"{qualifier} {text}"
    elif text and text[0].islower():
        text = text[0].upper() + text[1:]

    if citation_suffix and citation_suffix not in text:
        text = f"{text} {citation_suffix}"

    if not text.endswith((".", "!", "?")):
        text = f"{text}."
    return re.sub(r"\s+", " ", text).strip()


# Inputs: Raw claim text string.
# Outputs: True if the sentence contains multiple verifiable propositions.
def is_compound_claim(claim_text: str) -> bool:
    if not claim_text or len(claim_text.split()) < 8:
        return False
    if _SEMICOLON_SPLIT_RE.search(claim_text):
        return True
    if _SUBORDINATING_SPLIT_RE.search(claim_text):
        return True

    # Check coordinating 'and'/'but' only when the trailing span introduces its own verb/gerund
    for match in _COORDINATING_SPLIT_RE.finditer(claim_text):
        left = claim_text[: match.start()].strip()
        right = claim_text[match.end() :].strip()
        if len(left.split()) < 4 or len(right.split()) < 3:
            continue
        has_left_verb = bool(_FINITE_VERB_RE.search(left))
        has_right_verb = bool(
            _FINITE_VERB_RE.search(right)
            or re.match(r"^[A-Za-z]+ing\b", right)
        )
        if has_left_verb and has_right_verb:
            return True
    return False


# Inputs: Raw claim text string.
# Outputs: List of standalone atomic proposition strings preserving qualifiers and numbers.
def split_claim_rule_based(claim_text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", (claim_text or "").strip())
    if not cleaned:
        return []

    # Separate trailing citation marker so it can be attached cleanly to every proposition
    citation_suffix: str | None = None
    cite_match = _TRAILING_CITATION_RE.search(cleaned)
    working = cleaned
    if cite_match:
        citation_suffix = cite_match.group(1).strip()
        working = cleaned[: cite_match.start()].strip()
    working = working.rstrip(".!?").strip()

    # Extract shared leading condition/population qualifier if present
    qualifier: str | None = None
    qual_match = _LEADING_QUALIFIER_RE.match(working)
    if qual_match:
        qualifier = qual_match.group(1).strip()
        working = qual_match.group(2).strip()

    # Stage 1: Split on semicolons
    segments = [seg.strip() for seg in _SEMICOLON_SPLIT_RE.split(working) if seg.strip()]

    # Stage 2: Split on contrast/subordinating markers (while, whereas, although, yet)
    expanded: list[str] = []
    for seg in segments:
        parts = _SUBORDINATING_SPLIT_RE.split(seg)
        # re.split with 1 capturing group returns [clause0, marker1, clause1, ...]
        clauses = [parts[idx].strip() for idx in range(0, len(parts), 2) if parts[idx].strip()]
        expanded.extend(clauses)

    # Stage 3: Split on coordinating conjunctions (but / and) ONLY when both sides have verbs
    final_raw_clauses: list[str] = []
    for seg in expanded:
        split_done = False
        for match in _COORDINATING_SPLIT_RE.finditer(seg):
            left = seg[: match.start()].strip(" ,")
            right = seg[match.end() :].strip(" ,")
            if len(left.split()) < 4 or len(right.split()) < 3:
                continue
            if _FINITE_VERB_RE.search(left) and (
                _FINITE_VERB_RE.search(right) or re.match(r"^[A-Za-z]+ing\b", right)
            ):
                final_raw_clauses.extend([left, right])
                split_done = True
                break
        if not split_done:
            final_raw_clauses.append(seg)

    if len(final_raw_clauses) <= 1:
        single = _finalize_proposition(working, qualifier, None, citation_suffix)
        return [single] if single else []

    primary_subject = _extract_subject(final_raw_clauses[0])
    propositions: list[str] = []
    for idx, raw_clause in enumerate(final_raw_clauses):
        inherited = None if idx == 0 else primary_subject
        # If the clause already introduces its own distinct subject before a finite verb, don't override it
        if idx > 0 and _extract_subject(raw_clause) and not re.match(
            r"^(?:it|they|also|still)\b", raw_clause, flags=re.IGNORECASE
        ):
            inherited = None
        prop = _finalize_proposition(raw_clause, qualifier, inherited, citation_suffix)
        if prop and prop not in propositions:
            propositions.append(prop)

    return propositions


# Inputs: Raw claim text and optional Gemini model name.
# Outputs: List of atomic propositions from Gemini, or None if unavailable/unconfigured.
def _decompose_with_gemini(
    claim_text: str,
    model_name: str = DEFAULT_DECOMPOSER_MODEL,
) -> list[str] | None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    try:
        from google import genai
        from google.genai import errors as genai_errors
        from google.genai import types
    except ImportError:
        return None

    prompt = (
        "Decompose the following academic claim into minimal, standalone, verifiable atomic propositions.\n"
        "Rules:\n"
        "1. Preserve ALL numbers, percentages, populations, conditions, comparators, and qualifiers.\n"
        "2. Replace pronouns or dangling participles with the explicit subject from the sentence.\n"
        "3. Preserve any inline citation marker (e.g., [1]) on each proposition.\n"
        "4. If the claim already states a single atomic fact, return a 1-item list.\n\n"
        f"Claim: {claim_text}"
    )

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_LLMDecompositionSchema,
                temperature=0.0,
            ),
        )
        parsed = _LLMDecompositionSchema.model_validate_json(response.text)
        cleaned = [p.strip() for p in parsed.propositions if p and p.strip()]
        return cleaned if cleaned else None
    except (genai_errors.APIError, ValueError, RuntimeError, OSError):
        return None


# Inputs: ExtractedClaim object and optional use_llm flag (defaults to deterministic rule-based).
# Outputs: DecomposedClaim containing 1 or more AtomicProposition items with preserved qualifiers.
def decompose_claim(
    claim: ExtractedClaim,
    use_llm: bool = False,
) -> DecomposedClaim:
    method = "rule_based"
    prop_texts: list[str] | None = None

    if use_llm and is_compound_claim(claim.claim_text):
        prop_texts = _decompose_with_gemini(claim.claim_text)
        if prop_texts:
            method = "llm"

    if not prop_texts:
        prop_texts = split_claim_rule_based(claim.claim_text)

    if not prop_texts:
        prop_texts = [claim.claim_text.strip()]

    propositions: list[AtomicProposition] = []
    for idx, text in enumerate(prop_texts, start=1):
        qualifiers = sorted(set(_NUMBER_OR_QUALIFIER_RE.findall(text)))
        propositions.append(
            AtomicProposition(
                proposition_id=f"{claim.claim_id}.{idx}",
                parent_claim_id=claim.claim_id,
                text=text,
                preserved_qualifiers=qualifiers,
                inline_marker=claim.inline_marker,
                raw_reference=claim.raw_reference,
            )
        )

    return DecomposedClaim(
        parent_claim_id=claim.claim_id,
        original_claim_text=claim.claim_text,
        is_compound=len(propositions) > 1,
        decomposition_method=method,
        propositions=propositions,
    )


# Inputs: ConfidenceLevel list from proposition verdicts.
# Outputs: Conservative overall ConfidenceLevel (minimum confidence across propositions).
def _aggregate_confidence(levels: list[ConfidenceLevel]) -> ConfidenceLevel:
    rank = {
        ConfidenceLevel.LOW: 0,
        ConfidenceLevel.MEDIUM: 1,
        ConfidenceLevel.HIGH: 2,
    }
    if not levels:
        return ConfidenceLevel.LOW
    min_level = min(levels, key=lambda lvl: rank[lvl])
    return min_level


# Inputs: Parent ExtractedClaim and list of PropositionVerdict outcomes for its atomic propositions.
# Outputs: Single CombinedClaimVerdict preserving full per-proposition accountability.
def combine_proposition_verdicts(
    claim: ExtractedClaim,
    proposition_verdicts: list[PropositionVerdict],
) -> CombinedClaimVerdict:
    if not proposition_verdicts:
        return CombinedClaimVerdict(
            parent_claim_id=claim.claim_id,
            original_claim_text=claim.claim_text,
            overall_verdict=ClaimStatus.UNSUPPORTED,
            overall_confidence=ConfidenceLevel.LOW,
            needs_human_review=True,
            summary_explanation="No atomic propositions were available for verification.",
            mismatch_flags=["EMPTY_PROPOSITIONS"],
            proposition_verdicts=[],
        )

    verdicts = [pv.verdict for pv in proposition_verdicts]
    verdict_set = set(verdicts)
    flags: list[str] = []
    for pv in proposition_verdicts:
        for flag in pv.mismatch_flags:
            if flag not in flags:
                flags.append(flag)

    supported_n = verdicts.count(ClaimStatus.SUPPORTED)
    contradicted_n = verdicts.count(ClaimStatus.CONTRADICTORY)
    partial_n = verdicts.count(ClaimStatus.PARTIALLY_SUPPORTED)
    unsupported_n = verdicts.count(ClaimStatus.UNSUPPORTED)
    outdated_n = verdicts.count(ClaimStatus.POTENTIALLY_OUTDATED)
    total_n = len(verdicts)

    if outdated_n > 0:
        overall = ClaimStatus.POTENTIALLY_OUTDATED
        if "RETRACTED_OR_OUTDATED_PROPOSITION" not in flags:
            flags.append("RETRACTED_OR_OUTDATED_PROPOSITION")
    elif verdict_set == {ClaimStatus.SUPPORTED}:
        overall = ClaimStatus.SUPPORTED
    elif verdict_set == {ClaimStatus.CONTRADICTORY}:
        overall = ClaimStatus.CONTRADICTORY
    elif verdict_set == {ClaimStatus.UNSUPPORTED}:
        overall = ClaimStatus.UNSUPPORTED
    elif contradicted_n > 0 and (supported_n > 0 or partial_n > 0):
        # Classic Gap 2 case: one part is true, another part is actively contradicted
        overall = ClaimStatus.PARTIALLY_SUPPORTED
        if "PARTIAL_CONTRADICTION" not in flags:
            flags.append("PARTIAL_CONTRADICTION")
    elif contradicted_n > 0 and unsupported_n > 0:
        overall = ClaimStatus.CONTRADICTORY
        if "PARTIAL_CONTRADICTION" not in flags:
            flags.append("PARTIAL_CONTRADICTION")
    else:
        overall = ClaimStatus.PARTIALLY_SUPPORTED
        if unsupported_n > 0 and "INCOMPLETE_PROPOSITION_EVIDENCE" not in flags:
            flags.append("INCOMPLETE_PROPOSITION_EVIDENCE")

    overall_conf = _aggregate_confidence([pv.confidence for pv in proposition_verdicts])
    needs_review = overall != ClaimStatus.SUPPORTED or overall_conf != ConfidenceLevel.HIGH

    breakdown_parts = [
        f"[{pv.proposition_id}: {pv.verdict.value}] {pv.proposition_text}"
        for pv in proposition_verdicts
    ]
    summary = (
        f"Atomic verification across {total_n} proposition(s): "
        f"{supported_n} supported, {contradicted_n} contradictory, "
        f"{partial_n} partially supported, {unsupported_n} unsupported, {outdated_n} outdated. "
        + " | ".join(breakdown_parts)
    )

    return CombinedClaimVerdict(
        parent_claim_id=claim.claim_id,
        original_claim_text=claim.claim_text,
        overall_verdict=overall,
        overall_confidence=overall_conf,
        needs_human_review=needs_review,
        summary_explanation=summary,
        mismatch_flags=flags,
        proposition_verdicts=proposition_verdicts,
    )


# Inputs: ExtractedClaim, a callable verifying a single ExtractedClaim -> ClaimVerificationResult, and use_llm flag.
# Outputs: CombinedClaimVerdict after decomposing and verifying each atomic proposition separately.
def verify_decomposed_claim(
    claim: ExtractedClaim,
    verify_fn: Callable[[ExtractedClaim], ClaimVerificationResult],
    use_llm_decomposition: bool = False,
) -> CombinedClaimVerdict:
    decomposed = decompose_claim(claim, use_llm=use_llm_decomposition)
    prop_verdicts: list[PropositionVerdict] = []

    for idx, prop in enumerate(decomposed.propositions, start=1):
        sub_claim = ExtractedClaim(
            claim_id=idx,
            claim_text=prop.text,
            inline_marker=prop.inline_marker,
            raw_reference=prop.raw_reference,
        )
        res = verify_fn(sub_claim)
        prop_verdicts.append(
            PropositionVerdict(
                proposition_id=prop.proposition_id,
                proposition_text=prop.text,
                verdict=res.verdict,
                confidence=res.confidence,
                explanation=res.explanation,
                mismatch_flags=list(res.mismatch_flags),
            )
        )

    return combine_proposition_verdicts(claim, prop_verdicts)