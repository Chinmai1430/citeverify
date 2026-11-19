import math
import os
import re
from collections import Counter
from enum import StrEnum

from pydantic import BaseModel, Field

try:
    from backend.schemas import ClaimStatus, ConfidenceLevel
except ImportError:
    from schemas import ClaimStatus, ConfidenceLevel

DEFAULT_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
DEFAULT_NLI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
DEFAULT_EMBEDDING_DIM = 768

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
        "for", "from", "had", "has", "have", "in", "into", "is", "it", "its",
        "of", "on", "or", "our", "that", "the", "their", "these", "this",
        "those", "to", "was", "we", "were", "with", "observed", "found",
        "reported", "showed", "after", "before", "certain", "threshold",
        "study", "paper", "results", "data", "group", "compared",
    }
)

_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?%?|[A-Za-z][A-Za-z0-9\-]*")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?")

_NEGATION_RE = re.compile(
    r"\b(?:not|no|never|neither|nor|none|cannot|can't|didn't|doesn't|don't|failed\s+to|fails\s+to|without|lack(?:s|ed)?\s+of|no\s+significant)\b",
    re.IGNORECASE,
)

# Canonicalize directional/effect verbs so inflections match cleanly across claim and evidence
_CANONICAL_VERB_MAP: dict[str, str] = {
    "improve": "improve", "improves": "improve", "improved": "improve", "improving": "improve", "improvement": "improve",
    "increase": "increase", "increases": "increase", "increased": "increase", "increasing": "increase",
    "raise": "raise", "raises": "raise", "raised": "raise", "raising": "raise",
    "boost": "boost", "boosts": "boost", "boosted": "boost", "boosting": "boost",
    "enhance": "enhance", "enhances": "enhance", "enhanced": "enhance", "enhancing": "enhance",
    "higher": "higher", "greater": "greater", "more": "more", "better": "better",
    "outperform": "outperform", "outperforms": "outperform", "outperformed": "outperform", "outperforming": "outperform",
    "surpass": "surpass", "surpasses": "surpass", "surpassed": "surpass",
    "accelerate": "accelerate", "accelerates": "accelerate", "accelerated": "accelerate",
    "reduce": "reduce", "reduces": "reduce", "reduced": "reduce", "reducing": "reduce", "reduction": "reduce",
    "decrease": "decrease", "decreases": "decrease", "decreased": "decrease", "decreasing": "decrease",
    "lower": "lower", "lowers": "lower", "lowered": "lower", "lowering": "lower",
    "drop": "drop", "drops": "drop", "dropped": "drop", "dropping": "drop",
    "decline": "decline", "declines": "decline", "declined": "decline", "declining": "decline",
    "worsen": "worsen", "worsens": "worsen", "worsened": "worsen", "worsening": "worsen",
    "degrade": "degrade", "degrades": "degrade", "degraded": "degrade", "degrading": "degrade",
    "hurt": "hurt", "hurts": "hurt", "harm": "harm", "harms": "harm", "harmed": "harm",
    "fewer": "fewer", "less": "less", "worse": "worse", "slower": "slower",
    "underperform": "underperform", "underperforms": "underperform", "underperformed": "underperform",
    "prevent": "prevent", "prevents": "prevent", "prevented": "prevent", "preventing": "prevent",
    "cause": "cause", "causes": "cause", "caused": "cause", "causing": "cause",
    "require": "require", "requires": "require", "required": "require", "requiring": "require",
    "achieve": "achieve", "achieves": "achieve", "achieved": "achieve", "achieving": "achieve",
}

_POSITIVE_POLARITY: frozenset[str] = frozenset(
    {
        "improve", "increase", "raise", "boost", "enhance", "higher",
        "greater", "more", "better", "outperform", "surpass", "accelerate",
    }
)

_NEGATIVE_POLARITY: frozenset[str] = frozenset(
    {
        "reduce", "decrease", "lower", "drop", "decline", "worsen",
        "degrade", "hurt", "harm", "fewer", "less", "worse", "slower",
        "underperform",
    }
)


class LogicalStance(StrEnum):
    """Independent NLI relationship between a claim and an evidence passage."""

    ENTAILS = "ENTAILS"
    CONTRADICTS = "CONTRADICTS"
    NEUTRAL = "NEUTRAL"


class EvidenceCandidate(BaseModel):
    """Input evidence passage to be ranked and evaluated against a claim."""

    passage_id: str
    text: str
    source_title: str | None = None
    source_url: str | None = None


class RankedEvidence(BaseModel):
    """Single passage scored across topical, semantic, reranking, and logical-stance axes."""

    passage_id: str
    text: str
    source_title: str | None = None
    source_url: str | None = None
    bm25_score: float = 0.0
    semantic_score: float = 0.0
    hybrid_rrf_score: float = 0.0
    cross_encoder_score: float = 0.0
    logical_stance: LogicalStance = LogicalStance.NEUTRAL
    stance_confidence: ConfidenceLevel = ConfidenceLevel.LOW
    similarity_trap: bool = False
    stance_rationale: str = ""
    diagnostic_flags: list[str] = Field(default_factory=list)


class EvidenceVerificationSummary(BaseModel):
    """Output of the 5-stage pipeline separating retrieval relevance from logical support."""

    claim_text: str
    embedding_backend: str  # "gemini-embedding-001" or "tfidf_fallback"
    nli_backend: str  # "gemini_nli" or "rule_based_nli"
    top_evidence: RankedEvidence | None = None
    ranked_candidates: list[RankedEvidence] = Field(default_factory=list)
    recommended_verdict: ClaimStatus = ClaimStatus.UNSUPPORTED
    recommended_confidence: ConfidenceLevel = ConfidenceLevel.LOW
    similarity_trap_detected: bool = False
    explanation: str = ""


class _LLMNLIResponse(BaseModel):
    """Strict structured output schema for Gemini NLI classification."""

    stance: LogicalStance = Field(
        description="ENTAILS if evidence supports the claim, CONTRADICTS if it refutes or reverses it, NEUTRAL otherwise."
    )
    confidence: ConfidenceLevel
    rationale: str = Field(
        description="Concise explanation distinguishing topical overlap from actual logical entailment or contradiction."
    )


# Inputs: Raw text string.
# Outputs: List of lowercase normalized content tokens (excluding stopwords).
def _tokenize(text: str) -> list[str]:
    raw_tokens = _TOKEN_RE.findall((text or "").lower())
    normalized: list[str] = []
    for tok in raw_tokens:
        if tok in _STOPWORDS:
            continue
        canon = _CANONICAL_VERB_MAP.get(tok, tok)
        if canon not in _STOPWORDS:
            normalized.append(canon)
    return normalized


# Inputs: Numeric vector of floats.
# Outputs: L2-normalized unit vector (or original zero vector if norm is 0).
def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return vector
    return [v / norm for v in vector]


# Inputs: Two equal-length float vectors.
# Outputs: Cosine similarity score in [-1.0, 1.0].
def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or len(vec_a) != len(vec_b):
        return 0.0
    a = _l2_normalize(vec_a)
    b = _l2_normalize(vec_b)
    return sum(x * y for x, y in zip(a, b, strict=True))


# Inputs: Claim text and list of candidate passage texts.
# Outputs: List of BM25Okapi lexical relevance scores aligned with passages.
def score_bm25(
    claim_text: str,
    passages: list[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    if not passages:
        return []

    query_tokens = _tokenize(claim_text)
    if not query_tokens:
        return [0.0 for _ in passages]

    doc_tokens_list = [_tokenize(p) for p in passages]
    num_docs = len(doc_tokens_list)
    avgdl = sum(len(doc) for doc in doc_tokens_list) / max(num_docs, 1)
    if avgdl == 0.0:
        return [0.0 for _ in passages]

    doc_freqs: Counter[str] = Counter()
    for doc in doc_tokens_list:
        for term in set(doc):
            doc_freqs[term] += 1

    scores: list[float] = []
    for doc in doc_tokens_list:
        if not doc:
            scores.append(0.0)
            continue
        tf_counts = Counter(doc)
        doc_len = len(doc)
        score = 0.0
        for term in set(query_tokens):
            tf = tf_counts.get(term, 0)
            if tf == 0:
                continue
            df = doc_freqs.get(term, 0)
            idf = math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))
            denom = tf + k1 * (1.0 - b + b * (doc_len / avgdl))
            score += idf * ((tf * (k1 + 1.0)) / denom)
        scores.append(round(score, 6))
    return scores


# Inputs: Claim text and list of passage texts.
# Outputs: Offline TF-IDF cosine similarity scores in [0.0, 1.0].
def _score_tfidf_cosine(claim_text: str, passages: list[str]) -> list[float]:
    if not passages:
        return []

    all_docs = [_tokenize(claim_text), *(_tokenize(p) for p in passages)]
    vocab = sorted({tok for doc in all_docs for tok in doc})
    if not vocab:
        return [0.0 for _ in passages]

    num_docs = len(all_docs)
    df: Counter[str] = Counter()
    for doc in all_docs:
        for tok in set(doc):
            df[tok] += 1

    idf = {tok: math.log((num_docs + 1.0) / (df[tok] + 1.0)) + 1.0 for tok in vocab}

    def _vectorize(tokens: list[str]) -> list[float]:
        counts = Counter(tokens)
        return [counts.get(tok, 0) * idf[tok] for tok in vocab]

    claim_vec = _vectorize(all_docs[0])
    return [
        round(max(0.0, _cosine_similarity(claim_vec, _vectorize(doc))), 6)
        for doc in all_docs[1:]
    ]


# Inputs: Claim text, passage texts, optional embedding model name.
# Outputs: Tuple of (cosine similarity scores, backend label), using Gemini embeddings when available.
def score_embeddings(
    claim_text: str,
    passages: list[str],
    use_embeddings: bool = True,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> tuple[list[float], str]:
    if not passages:
        return [], "tfidf_fallback"

    api_key = os.getenv("GEMINI_API_KEY") if use_embeddings else None
    if api_key:
        try:
            from google import genai
            from google.genai import errors as genai_errors
            from google.genai import types

            client = genai.Client(api_key=api_key)
            claim_resp = client.models.embed_content(
                model=model_name,
                contents=[claim_text],
                config=types.EmbedContentConfig(
                    task_type="FACT_VERIFICATION",
                    output_dimensionality=DEFAULT_EMBEDDING_DIM,
                ),
            )
            doc_resp = client.models.embed_content(
                model=model_name,
                contents=passages,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=DEFAULT_EMBEDDING_DIM,
                ),
            )
            if claim_resp.embeddings and doc_resp.embeddings and len(doc_resp.embeddings) == len(passages):
                claim_vec = _l2_normalize(list(claim_resp.embeddings[0].values))
                scores = [
                    round(
                        max(0.0, _cosine_similarity(claim_vec, _l2_normalize(list(emb.values)))),
                        6,
                    )
                    for emb in doc_resp.embeddings
                ]
                return scores, model_name
        except (ImportError, genai_errors.APIError, ValueError, RuntimeError, OSError):
            pass

    return _score_tfidf_cosine(claim_text, passages), "tfidf_fallback"


# Inputs: Two score lists of equal length (BM25 and semantic scores) and RRF constant k.
# Outputs: Reciprocal Rank Fusion scores aligned with input candidate indices.
def reciprocal_rank_fusion(
    bm25_scores: list[float],
    semantic_scores: list[float],
    rrf_k: int = 60,
) -> list[float]:
    n = len(bm25_scores)
    if n == 0:
        return []

    bm25_order = sorted(range(n), key=lambda idx: bm25_scores[idx], reverse=True)
    sem_order = sorted(range(n), key=lambda idx: semantic_scores[idx], reverse=True)

    bm25_rank = {idx: rank for rank, idx in enumerate(bm25_order, start=1)}
    sem_rank = {idx: rank for rank, idx in enumerate(sem_order, start=1)}

    fused: list[float] = []
    for idx in range(n):
        score = (1.0 / (rrf_k + bm25_rank[idx])) + (1.0 / (rrf_k + sem_rank[idx]))
        fused.append(round(score, 6))
    return fused


# Inputs: Claim text, passage text, BM25 score, and semantic score.
# Outputs: Pairwise cross-encoder relevance score in [0.0, 1.0].
def _score_cross_encoder_pair(
    claim_text: str,
    passage_text: str,
    bm25_norm: float,
    semantic_score: float,
) -> float:
    claim_tokens = set(_tokenize(claim_text))
    passage_tokens = set(_tokenize(passage_text))
    if not claim_tokens or not passage_tokens:
        return 0.0

    # Strip directional words when measuring topical entity/concept coverage so contradictions still rank as relevant evidence
    non_polar_claim = {
        tok for tok in claim_tokens
        if tok not in _POSITIVE_POLARITY and tok not in _NEGATIVE_POLARITY
    }
    base_claim = non_polar_claim or claim_tokens
    concept_coverage = len(base_claim & passage_tokens) / len(base_claim)

    claim_nums = set(_NUMBER_RE.findall(claim_text))
    passage_nums = set(_NUMBER_RE.findall(passage_text))
    num_bonus = 0.0
    if claim_nums and passage_nums:
        # Sharing a numeric domain (even if values differ) signals direct empirical relevance
        num_bonus = 0.12 if (claim_nums & passage_nums) else 0.06

    combined = (0.45 * concept_coverage) + (0.35 * semantic_score) + (0.20 * bm25_norm) + num_bonus
    return round(min(1.0, max(0.0, combined)), 4)


# Inputs: Claim text and candidate evidence passage text.
# Outputs: Tuple of (LogicalStance, ConfidenceLevel, rationale, diagnostic_flags) using deterministic NLI rules.
def _classify_stance_rule_based(
    claim_text: str,
    passage_text: str,
) -> tuple[LogicalStance, ConfidenceLevel, str, list[str]]:
    claim_tokens = set(_tokenize(claim_text))
    passage_tokens = set(_tokenize(passage_text))
    flags: list[str] = []

    non_polar_claim = {
        tok for tok in claim_tokens
        if tok not in _POSITIVE_POLARITY and tok not in _NEGATIVE_POLARITY
    }
    base_claim = non_polar_claim or claim_tokens
    concept_overlap = (
        len(base_claim & passage_tokens) / len(base_claim) if base_claim else 0.0
    )

    if concept_overlap < 0.35:
        return (
            LogicalStance.NEUTRAL,
            ConfidenceLevel.LOW,
            "Passage does not share enough core claim concepts to confirm or refute the statement.",
            ["LOW_CONCEPT_OVERLAP"],
        )

    # 1. Negation flip check ("reduces risk" vs "did not reduce risk")
    claim_negated = bool(_NEGATION_RE.search(claim_text))
    passage_negated = bool(_NEGATION_RE.search(passage_text))
    if claim_negated != passage_negated and concept_overlap >= 0.5:
        flags.append("NEGATION_CONTRADICTION")
        return (
            LogicalStance.CONTRADICTS,
            ConfidenceLevel.HIGH,
            "Evidence reverses the polarity of the claim via explicit negation on the same subject.",
            flags,
        )

    # 2. Differential directional polarity check (ignores shared condition verbs like "increasing the learning rate")
    claim_polar = claim_tokens & (_POSITIVE_POLARITY | _NEGATIVE_POLARITY)
    passage_polar = passage_tokens & (_POSITIVE_POLARITY | _NEGATIVE_POLARITY)
    shared_polar = claim_polar & passage_polar

    unique_claim_polar = claim_polar - shared_polar
    unique_passage_polar = passage_polar - shared_polar

    claim_pos = unique_claim_polar & _POSITIVE_POLARITY
    claim_neg = unique_claim_polar & _NEGATIVE_POLARITY
    pass_pos = unique_passage_polar & _POSITIVE_POLARITY
    pass_neg = unique_passage_polar & _NEGATIVE_POLARITY

    claim_dir = (1 if claim_pos and not claim_neg else (-1 if claim_neg and not claim_pos else 0))
    pass_dir = (1 if pass_pos and not pass_neg else (-1 if pass_neg and not pass_pos else 0))

    # 3. Numeric comparison check (e.g., 62.4% vs 22.9%)
    claim_pcts = set(re.findall(r"\b\d+(?:\.\d+)?%", claim_text))
    passage_pcts = set(re.findall(r"\b\d+(?:\.\d+)?%", passage_text))
    numeric_conflict = bool(claim_pcts and passage_pcts and not (claim_pcts & passage_pcts))
    if numeric_conflict:
        flags.append("NUMERIC_CONTRADICTION")

    if claim_dir != 0 and pass_dir != 0 and claim_dir != pass_dir:
        flags.append("POLARITY_CONTRADICTION")
        return (
            LogicalStance.CONTRADICTS,
            ConfidenceLevel.HIGH,
            (
                f"Evidence states an opposing effect direction "
                f"(claim: {sorted(unique_claim_polar)} vs evidence: {sorted(unique_passage_polar)}) "
                "despite high topical similarity."
            ),
            flags,
        )

    if numeric_conflict:
        return (
            LogicalStance.CONTRADICTS,
            ConfidenceLevel.HIGH,
            (
                f"Evidence reports conflicting quantitative values "
                f"(claim: {sorted(claim_pcts)} vs evidence: {sorted(passage_pcts)})."
            ),
            flags,
        )

    # 4. Entailment verification: require concept alignment, matching numbers, and no missing directional claim
    claim_all_nums = set(_NUMBER_RE.findall(claim_text))
    passage_all_nums = set(_NUMBER_RE.findall(passage_text))
    all_claim_nums_matched = claim_all_nums.issubset(passage_all_nums) if claim_all_nums else True

    claim_has_unverified_direction = bool(unique_claim_polar) and not bool(shared_polar & (_POSITIVE_POLARITY | _NEGATIVE_POLARITY))
    # If claim asserts a directional effect (e.g., "improves", "reduces") that passage omits entirely, keep NEUTRAL
    if claim_polar and not passage_polar:
        claim_has_unverified_direction = True

    if concept_overlap >= 0.60 and all_claim_nums_matched and not claim_has_unverified_direction:
        return (
            LogicalStance.ENTAILS,
            ConfidenceLevel.HIGH if concept_overlap >= 0.75 else ConfidenceLevel.MEDIUM,
            "Evidence aligns with the claim's core entities, direction, and reported quantitative values.",
            flags,
        )

    flags.append("TOPICAL_ONLY_NO_ENTAILMENT")
    return (
        LogicalStance.NEUTRAL,
        ConfidenceLevel.MEDIUM,
        "Passage discusses related concepts but does not logically confirm or refute the claim's specific assertion.",
        flags,
    )


# Inputs: Claim text, passage text, optional Gemini model name.
# Outputs: Tuple of (LogicalStance, ConfidenceLevel, rationale) from Gemini, or None if unavailable.
def _classify_stance_with_gemini(
    claim_text: str,
    passage_text: str,
    model_name: str = DEFAULT_NLI_MODEL,
) -> tuple[LogicalStance, ConfidenceLevel, str] | None:
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
        "You are an academic Natural Language Inference (NLI) verifier.\n"
        "Classify the logical relationship between the CLAIM and the EVIDENCE PASSAGE.\n"
        "CRITICAL: Do NOT confuse topical or semantic similarity with logical support.\n"
        "- ENTAILS: The passage directly supports the claim's direction, scope, and numbers.\n"
        "- CONTRADICTS: The passage refutes, negates, reverses the direction, or reports conflicting metrics.\n"
        "- NEUTRAL: The passage discusses the same topic without proving or disproving the claim.\n\n"
        f"CLAIM: {claim_text}\n"
        f"EVIDENCE PASSAGE: {passage_text}"
    )

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_LLMNLIResponse,
                temperature=0.0,
            ),
        )
        parsed = _LLMNLIResponse.model_validate_json(response.text)
        return parsed.stance, parsed.confidence, parsed.rationale
    except (genai_errors.APIError, ValueError, RuntimeError, OSError):
        return None


# Inputs: Claim text, candidate evidence passages, retrieval/NLI configuration flags.
# Outputs: EvidenceVerificationSummary separating retrieval scores from independent logical stance.
def rank_and_verify_evidence(
    claim_text: str,
    candidates: list[EvidenceCandidate],
    top_k: int = 3,
    use_embeddings: bool = False,
    use_llm_nli: bool = False,
) -> EvidenceVerificationSummary:
    if not candidates:
        return EvidenceVerificationSummary(
            claim_text=claim_text,
            embedding_backend="tfidf_fallback",
            nli_backend="rule_based_nli",
            recommended_verdict=ClaimStatus.UNSUPPORTED,
            recommended_confidence=ConfidenceLevel.LOW,
            explanation="No candidate evidence passages were provided for ranking.",
        )

    passage_texts = [c.text for c in candidates]

    # Stage 1: BM25 Lexical Retrieval
    bm25_scores = score_bm25(claim_text, passage_texts)
    max_bm25 = max(bm25_scores) if bm25_scores else 0.0

    # Stage 2: Embedding-Based Semantic Retrieval (Gemini or TF-IDF fallback)
    semantic_scores, embedding_backend = score_embeddings(
        claim_text,
        passage_texts,
        use_embeddings=use_embeddings,
    )

    # Stage 3: Hybrid Candidate Selection via Reciprocal Rank Fusion (RRF)
    rrf_scores = reciprocal_rank_fusion(bm25_scores, semantic_scores)

    # Select top_k hybrid candidates for Cross-Encoder Reranking & Independent NLI
    candidate_indices = sorted(
        range(len(candidates)),
        key=lambda idx: (rrf_scores[idx], semantic_scores[idx], bm25_scores[idx]),
        reverse=True,
    )[: max(1, top_k)]

    nli_backend = "rule_based_nli"
    ranked: list[RankedEvidence] = []

    for idx in candidate_indices:
        cand = candidates[idx]
        bm25_val = bm25_scores[idx]
        bm25_norm = (bm25_val / max_bm25) if max_bm25 > 0.0 else 0.0
        sem_val = semantic_scores[idx]
        rrf_val = rrf_scores[idx]

        # Stage 4: Cross-Encoder Pairwise Relevance Reranking
        ce_score = _score_cross_encoder_pair(claim_text, cand.text, bm25_norm, sem_val)

        # Stage 5: Independent Logical Stance Classification (LLM NLI or Deterministic NLI)
        stance: LogicalStance
        confidence: ConfidenceLevel
        rationale: str
        flags: list[str] = []

        llm_result = (
            _classify_stance_with_gemini(claim_text, cand.text)
            if use_llm_nli
            else None
        )
        if llm_result is not None:
            stance, confidence, rationale = llm_result
            nli_backend = "gemini_nli"
        else:
            stance, confidence, rationale, flags = _classify_stance_rule_based(
                claim_text, cand.text
            )

        # Flag Similarity Trap: high topical/semantic relevance that actually contradicts the claim
        is_similarity_trap = stance == LogicalStance.CONTRADICTS and (
            ce_score >= 0.45 or sem_val >= 0.35
        )
        if is_similarity_trap and "SIMILARITY_TRAP_CONTRADICTION" not in flags:
            flags.append("SIMILARITY_TRAP_CONTRADICTION")

        ranked.append(
            RankedEvidence(
                passage_id=cand.passage_id,
                text=cand.text,
                source_title=cand.source_title,
                source_url=cand.source_url,
                bm25_score=bm25_val,
                semantic_score=sem_val,
                hybrid_rrf_score=rrf_val,
                cross_encoder_score=ce_score,
                logical_stance=stance,
                stance_confidence=confidence,
                similarity_trap=is_similarity_trap,
                stance_rationale=rationale,
                diagnostic_flags=flags,
            )
        )

    # Order reranked top_k candidates by cross-encoder relevance score
    ranked.sort(key=lambda item: (item.cross_encoder_score, item.hybrid_rrf_score), reverse=True)

    # Choose decisive logical evidence (prioritize highest-relevance CONTRADICTS or ENTAILS over NEUTRAL)
    decisive = next(
        (item for item in ranked if item.logical_stance != LogicalStance.NEUTRAL),
        ranked[0],
    )

    if decisive.logical_stance == LogicalStance.ENTAILS:
        rec_verdict = ClaimStatus.SUPPORTED
    elif decisive.logical_stance == LogicalStance.CONTRADICTS:
        rec_verdict = ClaimStatus.CONTRADICTORY
    else:
        rec_verdict = ClaimStatus.UNSUPPORTED

    trap_detected = any(item.similarity_trap for item in ranked)

    return EvidenceVerificationSummary(
        claim_text=claim_text,
        embedding_backend=embedding_backend,
        nli_backend=nli_backend,
        top_evidence=decisive,
        ranked_candidates=ranked,
        recommended_verdict=rec_verdict,
        recommended_confidence=decisive.stance_confidence,
        similarity_trap_detected=trap_detected,
        explanation=(
            f"Top evidence [{decisive.passage_id}] scored {decisive.cross_encoder_score:.2f} relevance "
            f"(BM25={decisive.bm25_score:.2f}, Semantic={decisive.semantic_score:.2f}) with independent "
            f"logical stance {decisive.logical_stance.value}: {decisive.stance_rationale}"
        ),
    )