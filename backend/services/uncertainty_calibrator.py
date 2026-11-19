import math
import re
from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, Field

try:
    from backend.schemas import (
        ClaimStatus,
        ClaimVerificationResult,
        ConfidenceLevel,
    )
    from backend.services.evidence_ranker import RankedEvidence
except ImportError:
    from schemas import (
        ClaimStatus,
        ClaimVerificationResult,
        ConfidenceLevel,
    )

    from services.evidence_ranker import RankedEvidence

_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?%?|[A-Za-z][A-Za-z0-9\-]*")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?")

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
        "for", "from", "had", "has", "have", "in", "into", "is", "it", "its",
        "of", "on", "or", "our", "that", "the", "their", "these", "this",
        "those", "to", "was", "we", "were", "with",
    }
)

_SCOPE_WEIGHTS: dict[str, float] = {
    "FULL_TEXT": 1.00,
    "ABSTRACT_ONLY": 0.88,
    "METADATA_ONLY": 0.30,
    "NONE": 0.00,
}

_CONFIDENCE_TO_RAW_PROB: dict[ConfidenceLevel, float] = {
    ConfidenceLevel.HIGH: 0.88,
    ConfidenceLevel.MEDIUM: 0.65,
    ConfidenceLevel.LOW: 0.38,
}

ABSTENTION_MESSAGE = "I could not find enough evidence to verify this claim."


class VerificationAction(StrEnum):
    """System behavior selected by the uncertainty-aware decision policy."""

    PRESENT_VERDICT = "PRESENT_VERDICT"
    FLAG_FOR_REVIEW = "FLAG_FOR_REVIEW"
    ABSTAIN = "ABSTAIN"
    MARK_UNVERIFIABLE = "MARK_UNVERIFIABLE"


class CalibrationSample(BaseModel):
    """Single labeled validation example used to fit and evaluate probability calibration."""

    raw_probability: float = Field(ge=0.0, le=1.0)
    is_correct: bool


class CalibrationMetrics(BaseModel):
    """Fitted temperature scaler parameters and before/after calibration diagnostics."""

    temperature: float
    ece_before: float
    ece_after: float
    brier_before: float
    brier_after: float
    nll_before: float
    nll_after: float
    sample_count: int


class UncertaintyThresholds(BaseModel):
    """Configurable thresholds governing automatic verdicts, human review, and abstention."""

    present_verdict_min_probability: float = 0.75
    present_verdict_min_coverage: float = 0.55
    present_verdict_min_retrieval: float = 0.55
    abstain_below_probability: float = 0.55
    abstain_below_coverage: float = 0.35
    abstain_below_retrieval: float = 0.35
    conflict_agreement_threshold: float = 0.65


class UncertaintySignals(BaseModel):
    """Decomposed reliability signals backing a verification decision."""

    source_accessible: bool
    evidence_scope: str
    evidence_coverage_score: float
    retrieval_confidence: float
    passage_agreement_score: float
    has_conflicting_passages: bool
    raw_classifier_probability: float
    calibrated_probability: float
    calibration_temperature: float
    composite_reliability_score: float


class CalibratedVerificationDecision(BaseModel):
    """Final uncertainty-aware output that can present, flag, abstain, or mark unverifiable."""

    claim_text: str
    raw_verdict: ClaimStatus
    effective_verdict: ClaimStatus | None = None
    action: VerificationAction
    calibrated_confidence: ConfidenceLevel
    calibrated_probability: float
    needs_human_review: bool
    abstained: bool
    user_message: str
    signals: UncertaintySignals
    uncertainty_flags: list[str] = Field(default_factory=list)


# Inputs: Probability value in [0, 1].
# Outputs: Numerically safe probability clipped away from exact 0 and 1.
def _clip_prob(prob: float, eps: float = 1e-6) -> float:
    return min(1.0 - eps, max(eps, float(prob)))


# Inputs: Probability in (0, 1).
# Outputs: Log-odds value ln(p / (1 - p)).
def _logit(prob: float) -> float:
    p = _clip_prob(prob)
    return math.log(p / (1.0 - p))


# Inputs: Real-valued logit z.
# Outputs: Sigmoid probability in (0, 1).
def _sigmoid(z: float) -> float:
    if z >= 0:
        exp_neg = math.exp(-z)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(z)
    return exp_pos / (1.0 + exp_pos)


# Inputs: Raw text string.
# Outputs: Set of normalized content tokens excluding stopwords.
def _content_tokens(text: str) -> set[str]:
    return {
        tok for tok in _TOKEN_RE.findall((text or "").lower())
        if tok not in _STOPWORDS
    }


# Inputs: Raw classifier/LLM probability and temperature T > 0.
# Outputs: Temperature-scaled calibrated probability in (0, 1).
def apply_temperature_scaling(raw_probability: float, temperature: float = 1.0) -> float:
    temp = max(0.05, float(temperature))
    scaled = _sigmoid(_logit(raw_probability) / temp)
    return round(scaled, 4)


# Inputs: Predicted probabilities and binary correctness labels.
# Outputs: Mean Negative Log-Likelihood (binary cross-entropy).
def compute_nll(probabilities: Sequence[float], labels: Sequence[bool]) -> float:
    if not probabilities:
        return 0.0
    total = 0.0
    for prob, label in zip(probabilities, labels, strict=True):
        p = _clip_prob(prob)
        total += -math.log(p) if label else -math.log(1.0 - p)
    return round(total / len(probabilities), 6)


# Inputs: Predicted probabilities and binary correctness labels.
# Outputs: Mean squared error (Brier score), lower is better.
def compute_brier_score(probabilities: Sequence[float], labels: Sequence[bool]) -> float:
    if not probabilities:
        return 0.0
    total = sum(
        (float(prob) - (1.0 if label else 0.0)) ** 2
        for prob, label in zip(probabilities, labels, strict=True)
    )
    return round(total / len(probabilities), 6)


# Inputs: Predicted probabilities, binary labels, and number of calibration bins.
# Outputs: Expected Calibration Error (ECE) in [0.0, 1.0], lower is better.
def compute_ece(
    probabilities: Sequence[float],
    labels: Sequence[bool],
    num_bins: int = 10,
) -> float:
    n = len(probabilities)
    if n == 0 or num_bins <= 0:
        return 0.0

    ece = 0.0
    for bin_idx in range(num_bins):
        low = bin_idx / num_bins
        high = (bin_idx + 1) / num_bins
        members = [
            (float(p), 1.0 if y else 0.0)
            for p, y in zip(probabilities, labels, strict=True)
            if (low <= float(p) <= high if bin_idx == num_bins - 1 else low <= float(p) < high)
        ]
        if not members:
            continue
        avg_conf = sum(p for p, _ in members) / len(members)
        avg_acc = sum(y for _, y in members) / len(members)
        ece += (len(members) / n) * abs(avg_conf - avg_acc)

    return round(ece, 6)


# Inputs: Labeled validation CalibrationSample list and temperature search bounds.
# Outputs: CalibrationMetrics with optimal temperature T minimizing validation NLL.
def fit_temperature_scaler(
    validation_samples: Sequence[CalibrationSample],
    min_temp: float = 0.25,
    max_temp: float = 6.0,
    grid_steps: int = 116,
) -> CalibrationMetrics:
    if not validation_samples:
        return CalibrationMetrics(
            temperature=1.0,
            ece_before=0.0,
            ece_after=0.0,
            brier_before=0.0,
            brier_after=0.0,
            nll_before=0.0,
            nll_after=0.0,
            sample_count=0,
        )

    raw_probs = [s.raw_probability for s in validation_samples]
    labels = [s.is_correct for s in validation_samples]

    def _loss(temp: float) -> float:
        cal_probs = [apply_temperature_scaling(p, temp) for p in raw_probs]
        return compute_nll(cal_probs, labels)

    # Coarse grid search over [min_temp, max_temp]
    step = (max_temp - min_temp) / max(1, grid_steps - 1)
    best_temp = 1.0
    best_loss = _loss(1.0)
    for i in range(grid_steps):
        candidate_t = min_temp + i * step
        loss = _loss(candidate_t)
        if loss < best_loss:
            best_loss = loss
            best_temp = candidate_t

    # Local golden-section refinement around best coarse candidate
    left = max(min_temp, best_temp - step)
    right = min(max_temp, best_temp + step)
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(25):
        mid1 = right - phi * (right - left)
        mid2 = left + phi * (right - left)
        if _loss(mid1) < _loss(mid2):
            right = mid2
        else:
            left = mid1
    optimal_temp = round((left + right) / 2.0, 4)

    cal_probs = [apply_temperature_scaling(p, optimal_temp) for p in raw_probs]

    return CalibrationMetrics(
        temperature=optimal_temp,
        ece_before=compute_ece(raw_probs, labels),
        ece_after=compute_ece(cal_probs, labels),
        brier_before=compute_brier_score(raw_probs, labels),
        brier_after=compute_brier_score(cal_probs, labels),
        nll_before=compute_nll(raw_probs, labels),
        nll_after=compute_nll(cal_probs, labels),
        sample_count=len(validation_samples),
    )


# Inputs: Claim text, list of retrieved evidence passages, and evidence scope label.
# Outputs: Evidence coverage score in [0.0, 1.0] combining concept & numeric coverage with scope weight.
def compute_evidence_coverage(
    claim_text: str,
    evidence_passages: Sequence[str],
    evidence_scope: str = "ABSTRACT_ONLY",
) -> float:
    scope_key = (evidence_scope or "NONE").upper().strip()
    scope_weight = _SCOPE_WEIGHTS.get(scope_key, 0.5)
    non_empty = [p.strip() for p in evidence_passages if p and p.strip()]

    if scope_weight == 0.0 or not non_empty:
        return 0.0

    claim_tokens = _content_tokens(claim_text)
    if not claim_tokens:
        return 0.0

    combined_evidence = " ".join(non_empty)
    evidence_tokens = _content_tokens(combined_evidence)
    concept_cov = len(claim_tokens & evidence_tokens) / len(claim_tokens)

    claim_nums = set(_NUMBER_RE.findall(claim_text))
    if claim_nums:
        evidence_nums = set(_NUMBER_RE.findall(combined_evidence))
        num_cov = len(claim_nums & evidence_nums) / len(claim_nums)
        raw_cov = (0.70 * concept_cov) + (0.30 * num_cov)
    else:
        raw_cov = concept_cov

    return round(min(1.0, max(0.0, raw_cov * scope_weight)), 4)


# Inputs: Whether source was found, evidence scope string, and optional relevance scores in [0, 1].
# Outputs: Retrieval confidence score in [0.0, 1.0].
def compute_retrieval_confidence(
    source_found: bool,
    evidence_scope: str = "ABSTRACT_ONLY",
    relevance_scores: Sequence[float] | None = None,
) -> float:
    if not source_found:
        return 0.0

    scope_key = (evidence_scope or "NONE").upper().strip()
    scope_weight = _SCOPE_WEIGHTS.get(scope_key, 0.5)
    if scope_weight == 0.0:
        return 0.0

    valid_scores = [min(1.0, max(0.0, float(s))) for s in (relevance_scores or [])]
    if valid_scores:
        top_rel = max(valid_scores)
        mean_rel = sum(valid_scores) / len(valid_scores)
        rel_signal = (0.75 * top_rel) + (0.25 * mean_rel)
        score = (0.60 * rel_signal) + (0.40 * scope_weight)
    else:
        score = 0.80 * scope_weight

    return round(min(1.0, max(0.0, score)), 4)


# Inputs: Sequence of passage stance strings (e.g., ENTAILS, CONTRADICTS, NEUTRAL).
# Outputs: Tuple of (agreement_score in [0.0, 1.0], has_conflicting_passages).
def compute_passage_agreement(stances: Sequence[str]) -> tuple[float, bool]:
    normalized = [str(s).upper().strip() for s in stances if s]
    if not normalized:
        return 0.5, False

    entails_n = sum(1 for s in normalized if s in {"ENTAILS", "SUPPORTED"})
    contradicts_n = sum(1 for s in normalized if s in {"CONTRADICTS", "CONTRADICTORY"})
    directional_n = entails_n + contradicts_n

    if directional_n == 0:
        # All passages are neutral; no conflict, but no directional consensus either
        return 0.5, False

    has_conflict = entails_n > 0 and contradicts_n > 0
    majority_ratio = max(entails_n, contradicts_n) / directional_n
    if has_conflict:
        # Penalize active disagreement so mixed evidence reliably triggers human review
        agreement = round(max(0.0, majority_ratio - 0.35), 4)
        return agreement, True

    # Penalize lightly if most retrieved passages are neutral noise
    signal_ratio = directional_n / len(normalized)
    agreement = round(min(1.0, 0.70 + (0.30 * signal_ratio)), 4)
    return agreement, False


# Inputs: ConfidenceLevel enum.
# Outputs: Default uncalibrated probability proxy when numeric classifier logits are unavailable.
def probability_from_confidence_level(level: ConfidenceLevel) -> float:
    return _CONFIDENCE_TO_RAW_PROB.get(level, 0.50)


# Inputs: Calibrated probability in [0, 1].
# Outputs: Calibrated ConfidenceLevel bucket.
def confidence_level_from_probability(calibrated_probability: float) -> ConfidenceLevel:
    if calibrated_probability >= 0.75:
        return ConfidenceLevel.HIGH
    if calibrated_probability >= 0.55:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


# Inputs: Claim text, raw verdict, retrieval/coverage/stance signals, and optional calibration parameters.
# Outputs: CalibratedVerificationDecision implementing the 4-way abstention & review policy.
def assess_uncertainty_and_decide(
    claim_text: str,
    raw_verdict: ClaimStatus,
    *,
    source_accessible: bool,
    evidence_scope: str = "ABSTRACT_ONLY",
    evidence_passages: Sequence[str] = (),
    relevance_scores: Sequence[float] | None = None,
    passage_stances: Sequence[str] = (),
    raw_classifier_probability: float | None = None,
    raw_confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM,
    calibration_temperature: float = 1.0,
    thresholds: UncertaintyThresholds | None = None,
) -> CalibratedVerificationDecision:
    cfg = thresholds or UncertaintyThresholds()
    scope_key = (evidence_scope or "NONE").upper().strip()

    coverage = compute_evidence_coverage(claim_text, evidence_passages, evidence_scope=scope_key)
    retrieval_conf = compute_retrieval_confidence(
        source_found=source_accessible,
        evidence_scope=scope_key,
        relevance_scores=relevance_scores,
    )
    agreement, has_conflict = compute_passage_agreement(passage_stances)

    raw_prob = (
        float(raw_classifier_probability)
        if raw_classifier_probability is not None
        else probability_from_confidence_level(raw_confidence)
    )
    raw_prob = round(min(1.0, max(0.0, raw_prob)), 4)
    calibrated_prob = apply_temperature_scaling(raw_prob, temperature=calibration_temperature)

    composite = round(
        (0.40 * calibrated_prob)
        + (0.30 * coverage)
        + (0.20 * retrieval_conf)
        + (0.10 * agreement),
        4,
    )

    signals = UncertaintySignals(
        source_accessible=source_accessible,
        evidence_scope=scope_key,
        evidence_coverage_score=coverage,
        retrieval_confidence=retrieval_conf,
        passage_agreement_score=agreement,
        has_conflicting_passages=has_conflict,
        raw_classifier_probability=raw_prob,
        calibrated_probability=calibrated_prob,
        calibration_temperature=round(float(calibration_temperature), 4),
        composite_reliability_score=composite,
    )

    flags: list[str] = []
    if not source_accessible or scope_key == "NONE":
        flags.append("NO_ACCESSIBLE_SOURCE")
        return CalibratedVerificationDecision(
            claim_text=claim_text,
            raw_verdict=raw_verdict,
            effective_verdict=None,
            action=VerificationAction.MARK_UNVERIFIABLE,
            calibrated_confidence=ConfidenceLevel.LOW,
            calibrated_probability=min(calibrated_prob, 0.25),
            needs_human_review=True,
            abstained=True,
            user_message=(
                "Unverifiable: no accessible source text was retrieved to evaluate this claim."
            ),
            signals=signals,
            uncertainty_flags=flags,
        )

    if has_conflict or (
        passage_stances and agreement < cfg.conflict_agreement_threshold and has_conflict
    ):
        flags.append("CONFLICTING_EVIDENCE_PASSAGES")
        return CalibratedVerificationDecision(
            claim_text=claim_text,
            raw_verdict=raw_verdict,
            effective_verdict=raw_verdict,
            action=VerificationAction.FLAG_FOR_REVIEW,
            calibrated_confidence=ConfidenceLevel.LOW
            if calibrated_prob < 0.55
            else ConfidenceLevel.MEDIUM,
            calibrated_probability=calibrated_prob,
            needs_human_review=True,
            abstained=False,
            user_message=(
                "Flagged for human review: retrieved evidence passages contain conflicting "
                "findings regarding this claim."
            ),
            signals=signals,
            uncertainty_flags=flags,
        )

    # Check weak-evidence abstention triggers (prevents confidently wrong verdicts)
    if scope_key == "METADATA_ONLY":
        flags.append("METADATA_ONLY_EVIDENCE")
    if coverage < cfg.abstain_below_coverage:
        flags.append("WEAK_EVIDENCE_COVERAGE")
    if retrieval_conf < cfg.abstain_below_retrieval:
        flags.append("LOW_RETRIEVAL_CONFIDENCE")
    if calibrated_prob < cfg.abstain_below_probability:
        flags.append("LOW_CALIBRATED_PROBABILITY")

    if flags:
        return CalibratedVerificationDecision(
            claim_text=claim_text,
            raw_verdict=raw_verdict,
            effective_verdict=None,
            action=VerificationAction.ABSTAIN,
            calibrated_confidence=ConfidenceLevel.LOW,
            calibrated_probability=calibrated_prob,
            needs_human_review=True,
            abstained=True,
            user_message=ABSTENTION_MESSAGE,
            signals=signals,
            uncertainty_flags=flags,
        )

    # Strong evidence and agreement -> present verdict directly
    meets_strong_bar = (
        calibrated_prob >= cfg.present_verdict_min_probability
        and coverage >= cfg.present_verdict_min_coverage
        and retrieval_conf >= cfg.present_verdict_min_retrieval
        and agreement >= cfg.conflict_agreement_threshold
    )
    cal_conf = confidence_level_from_probability(calibrated_prob)

    if meets_strong_bar:
        return CalibratedVerificationDecision(
            claim_text=claim_text,
            raw_verdict=raw_verdict,
            effective_verdict=raw_verdict,
            action=VerificationAction.PRESENT_VERDICT,
            calibrated_confidence=cal_conf,
            calibrated_probability=calibrated_prob,
            needs_human_review=False,
            abstained=False,
            user_message=(
                f"Verified ({raw_verdict.value}) with strong evidence coverage "
                f"({coverage:.0%}) and calibrated probability {calibrated_prob:.0%}."
            ),
            signals=signals,
            uncertainty_flags=[],
        )

    flags.append("BELOW_AUTO_VERDICT_THRESHOLD")
    return CalibratedVerificationDecision(
        claim_text=claim_text,
        raw_verdict=raw_verdict,
        effective_verdict=raw_verdict,
        action=VerificationAction.FLAG_FOR_REVIEW,
        calibrated_confidence=cal_conf,
        calibrated_probability=calibrated_prob,
        needs_human_review=True,
        abstained=False,
        user_message=(
            f"Preliminary verdict ({raw_verdict.value}) flagged for human review "
            f"(calibrated probability {calibrated_prob:.0%}, coverage {coverage:.0%})."
        ),
        signals=signals,
        uncertainty_flags=flags,
    )


# Inputs: ClaimVerificationResult plus optional RankedEvidence list and calibration parameters.
# Outputs: CalibratedVerificationDecision seamlessly wrapping existing verifier/ranker outputs.
def calibrate_verification_result(
    result: ClaimVerificationResult,
    *,
    source_accessible: bool | None = None,
    evidence_scope: str | None = None,
    ranked_evidence: Sequence[RankedEvidence] = (),
    raw_classifier_probability: float | None = None,
    calibration_temperature: float = 1.0,
    thresholds: UncertaintyThresholds | None = None,
) -> CalibratedVerificationDecision:
    if ranked_evidence:
        passages = [item.text for item in ranked_evidence]
        relevance_scores = [item.cross_encoder_score for item in ranked_evidence]
        stances = [item.logical_stance.value for item in ranked_evidence]
    else:
        passages = [result.evidence_passage] if result.evidence_passage else []
        relevance_scores = None
        if result.verdict == ClaimStatus.SUPPORTED:
            stances = ["ENTAILS"]
        elif result.verdict == ClaimStatus.CONTRADICTORY:
            stances = ["CONTRADICTS"]
        else:
            stances = []

    inferred_accessible = (
        source_accessible
        if source_accessible is not None
        else bool(passages or result.source_url)
    )
    if evidence_scope is not None:
        inferred_scope = evidence_scope
    elif not inferred_accessible:
        inferred_scope = "NONE"
    elif passages:
        inferred_scope = "ABSTRACT_ONLY"
    else:
        inferred_scope = "METADATA_ONLY"

    return assess_uncertainty_and_decide(
        claim_text=result.claim_text,
        raw_verdict=result.verdict,
        source_accessible=inferred_accessible,
        evidence_scope=inferred_scope,
        evidence_passages=passages,
        relevance_scores=relevance_scores,
        passage_stances=stances,
        raw_classifier_probability=raw_classifier_probability,
        raw_confidence=result.confidence,
        calibration_temperature=calibration_temperature,
        thresholds=thresholds,
    )