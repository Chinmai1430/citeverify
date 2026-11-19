import unittest

from backend.schemas import (
    ClaimStatus,
    ClaimVerificationResult,
    ConfidenceLevel,
)
from backend.services.evidence_ranker import LogicalStance, RankedEvidence
from backend.services.uncertainty_calibrator import (
    ABSTENTION_MESSAGE,
    CalibrationSample,
    VerificationAction,
    apply_temperature_scaling,
    assess_uncertainty_and_decide,
    calibrate_verification_result,
    compute_evidence_coverage,
    compute_passage_agreement,
    fit_temperature_scaler,
)


class TestUncertaintyCalibrator(unittest.TestCase):
    # Inputs: Strong full-text/abstract evidence, high relevance, unanimous ENTAILS, high calibrated probability.
    # Outputs: Verifies PRESENT_VERDICT with abstained=False and needs_human_review=False.
    def test_strong_evidence_and_agreement_presents_verdict(self):
        decision = assess_uncertainty_and_decide(
            claim_text="Deep learning achieved 97.5% sensitivity and 93.4% specificity on EyePACS-1.",
            raw_verdict=ClaimStatus.SUPPORTED,
            source_accessible=True,
            evidence_scope="FULL_TEXT",
            evidence_passages=[
                "On EyePACS-1, the deep learning algorithm achieved 97.5% sensitivity and 93.4% specificity.",
                "External validation confirmed 97.5% sensitivity and 93.4% specificity on EyePACS-1.",
            ],
            relevance_scores=[0.94, 0.89],
            passage_stances=["ENTAILS", "ENTAILS"],
            raw_classifier_probability=0.92,
            calibration_temperature=1.1,
        )

        self.assertEqual(decision.action, VerificationAction.PRESENT_VERDICT)
        self.assertEqual(decision.effective_verdict, ClaimStatus.SUPPORTED)
        self.assertEqual(decision.calibrated_confidence, ConfidenceLevel.HIGH)
        self.assertFalse(decision.abstained)
        self.assertFalse(decision.needs_human_review)
        self.assertEqual(decision.uncertainty_flags, [])

    # Inputs: Two high-relevance retrieved passages where one ENTAILS and one CONTRADICTS.
    # Outputs: Verifies FLAG_FOR_REVIEW with CONFLICTING_EVIDENCE_PASSAGES flag.
    def test_conflicting_passages_flagged_for_human_review(self):
        decision = assess_uncertainty_and_decide(
            claim_text="Treatment X reduces 28-day mortality in hospitalized ICU patients.",
            raw_verdict=ClaimStatus.PARTIALLY_SUPPORTED,
            source_accessible=True,
            evidence_scope="ABSTRACT_ONLY",
            evidence_passages=[
                "Treatment X reduced 28-day mortality in hospitalized ICU patients.",
                "In a multicenter ICU trial, Treatment X did not reduce 28-day mortality.",
            ],
            relevance_scores=[0.88, 0.86],
            passage_stances=["ENTAILS", "CONTRADICTS"],
            raw_classifier_probability=0.80,
        )

        self.assertEqual(decision.action, VerificationAction.FLAG_FOR_REVIEW)
        self.assertEqual(decision.effective_verdict, ClaimStatus.PARTIALLY_SUPPORTED)
        self.assertFalse(decision.abstained)
        self.assertTrue(decision.needs_human_review)
        self.assertTrue(decision.signals.has_conflicting_passages)
        self.assertIn("CONFLICTING_EVIDENCE_PASSAGES", decision.uncertainty_flags)

    # Inputs: Confident LLM raw verdict backed only by weak/off-topic or METADATA_ONLY evidence.
    # Outputs: Verifies system abstains with explicit abstention message and effective_verdict=None.
    def test_weak_evidence_triggers_abstention(self):
        decision = assess_uncertainty_and_decide(
            claim_text="Sparse attention reduced ICU inference latency by 41.2% across 12 hospitals.",
            raw_verdict=ClaimStatus.SUPPORTED,
            source_accessible=True,
            evidence_scope="METADATA_ONLY",
            evidence_passages=["A review of neural architectures in healthcare."],
            relevance_scores=[0.22],
            passage_stances=["NEUTRAL"],
            raw_classifier_probability=0.91,
        )

        self.assertEqual(decision.action, VerificationAction.ABSTAIN)
        self.assertIsNone(decision.effective_verdict)
        self.assertTrue(decision.abstained)
        self.assertTrue(decision.needs_human_review)
        self.assertEqual(decision.user_message, ABSTENTION_MESSAGE)
        self.assertIn("METADATA_ONLY_EVIDENCE", decision.uncertainty_flags)
        self.assertIn("WEAK_EVIDENCE_COVERAGE", decision.uncertainty_flags)

    # Inputs: Fabricated or paywalled citation with source_accessible=False and scope=NONE.
    # Outputs: Verifies MARK_UNVERIFIABLE with effective_verdict=None and NO_ACCESSIBLE_SOURCE flag.
    def test_no_accessible_source_marks_unverifiable(self):
        decision = assess_uncertainty_and_decide(
            claim_text="Kowalski & Verma (2024) reduced clinical misdiagnosis by 74.2%.",
            raw_verdict=ClaimStatus.UNSUPPORTED,
            source_accessible=False,
            evidence_scope="NONE",
            evidence_passages=[],
            raw_classifier_probability=0.85,
        )

        self.assertEqual(decision.action, VerificationAction.MARK_UNVERIFIABLE)
        self.assertIsNone(decision.effective_verdict)
        self.assertTrue(decision.abstained)
        self.assertTrue(decision.needs_human_review)
        self.assertIn("NO_ACCESSIBLE_SOURCE", decision.uncertainty_flags)

    # Inputs: Overconfident validation set (model predicts 0.95/0.90 but is only ~67% accurate).
    # Outputs: Verifies temperature scaler learns T > 1.0 and improves ECE, Brier score, and NLL.
    def test_validation_set_temperature_scaling_improves_calibration(self):
        validation_set = [
            CalibrationSample(raw_probability=0.95, is_correct=True),
            CalibrationSample(raw_probability=0.95, is_correct=True),
            CalibrationSample(raw_probability=0.95, is_correct=False),
            CalibrationSample(raw_probability=0.92, is_correct=True),
            CalibrationSample(raw_probability=0.92, is_correct=False),
            CalibrationSample(raw_probability=0.90, is_correct=True),
            CalibrationSample(raw_probability=0.90, is_correct=True),
            CalibrationSample(raw_probability=0.90, is_correct=False),
            CalibrationSample(raw_probability=0.85, is_correct=True),
            CalibrationSample(raw_probability=0.85, is_correct=False),
            CalibrationSample(raw_probability=0.80, is_correct=True),
            CalibrationSample(raw_probability=0.80, is_correct=False),
        ]

        metrics = fit_temperature_scaler(validation_set)

        self.assertEqual(metrics.sample_count, 12)
        self.assertGreater(metrics.temperature, 1.5)
        self.assertLess(metrics.ece_after, metrics.ece_before)
        self.assertLess(metrics.brier_after, metrics.brier_before)
        self.assertLess(metrics.nll_after, metrics.nll_before)

        softened = apply_temperature_scaling(0.95, temperature=metrics.temperature)
        self.assertLess(softened, 0.80)

    # Inputs: Moderate coverage and moderate calibrated probability between abstain and auto-verdict thresholds.
    # Outputs: Verifies FLAG_FOR_REVIEW with BELOW_AUTO_VERDICT_THRESHOLD.
    def test_moderate_confidence_flags_for_review_instead_of_auto_verdict(self):
        decision = assess_uncertainty_and_decide(
            claim_text="Dexamethasone reduced 28-day mortality to 22.9% in ventilated patients.",
            raw_verdict=ClaimStatus.SUPPORTED,
            source_accessible=True,
            evidence_scope="ABSTRACT_ONLY",
            evidence_passages=[
                "Dexamethasone reduced 28-day mortality to 22.9% in ventilated respiratory patients."
            ],
            relevance_scores=[0.78],
            passage_stances=["ENTAILS"],
            raw_classifier_probability=0.66,
            calibration_temperature=1.0,
        )

        self.assertEqual(decision.action, VerificationAction.FLAG_FOR_REVIEW)
        self.assertEqual(decision.effective_verdict, ClaimStatus.SUPPORTED)
        self.assertFalse(decision.abstained)
        self.assertTrue(decision.needs_human_review)
        self.assertIn("BELOW_AUTO_VERDICT_THRESHOLD", decision.uncertainty_flags)

    # Inputs: ClaimVerificationResult paired with RankedEvidence passages (one ENTAILS, one CONTRADICTS).
    # Outputs: Verifies calibrate_verification_result adapter seamlessly converts schemas + ranker outputs.
    def test_calibrate_verification_result_adapter_with_ranked_evidence(self):
        verdict_result = ClaimVerificationResult(
            claim_id=1,
            claim_text="Increasing the learning rate improves model accuracy.",
            verdict=ClaimStatus.CONTRADICTORY,
            explanation="Passages disagree across regimes.",
            confidence=ConfidenceLevel.HIGH,
            uncertainty_note="Needs review.",
            needs_human_review=False,
            source_url="https://doi.org/10.1000/lr-study",
        )
        ranked = [
            RankedEvidence(
                passage_id="p1",
                text="Increasing the learning rate reduced model accuracy after a threshold.",
                cross_encoder_score=0.90,
                logical_stance=LogicalStance.CONTRADICTS,
            ),
            RankedEvidence(
                passage_id="p2",
                text="Increasing the learning rate improved accuracy during early warmup epochs.",
                cross_encoder_score=0.82,
                logical_stance=LogicalStance.ENTAILS,
            ),
        ]

        decision = calibrate_verification_result(
            verdict_result,
            ranked_evidence=ranked,
            evidence_scope="ABSTRACT_ONLY",
            calibration_temperature=1.2,
        )

        self.assertEqual(decision.action, VerificationAction.FLAG_FOR_REVIEW)
        self.assertTrue(decision.signals.has_conflicting_passages)
        self.assertGreater(
            compute_evidence_coverage(verdict_result.claim_text, [r.text for r in ranked]),
            0.5,
        )
        agreement, conflict = compute_passage_agreement(["ENTAILS", "CONTRADICTS"])
        self.assertTrue(conflict)
        self.assertLess(agreement, 0.5)


if __name__ == "__main__":
    unittest.main()