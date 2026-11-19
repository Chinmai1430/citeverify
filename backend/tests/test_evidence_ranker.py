import unittest

from backend.schemas import ClaimStatus
from backend.services.evidence_ranker import (
    EvidenceCandidate,
    LogicalStance,
    rank_and_verify_evidence,
    reciprocal_rank_fusion,
    score_bm25,
)


class TestEvidenceRanker(unittest.TestCase):
    # Inputs: Classic Gap 1 learning-rate contradiction alongside topical neutral & distractor passages.
    # Outputs: Verifies high-similarity contradiction is retrieved #1, classified CONTRADICTS, and flagged as similarity_trap.
    def test_learning_rate_similarity_trap_contradiction(self):
        claim = "Increasing the learning rate improves model accuracy."
        candidates = [
            EvidenceCandidate(
                passage_id="p_contradict",
                text="We observed that increasing the learning rate reduced model accuracy after a certain threshold.",
                source_title="Optimization Thresholds in Deep Networks",
            ),
            EvidenceCandidate(
                passage_id="p_neutral",
                text="Learning rate schedules are a widely used hyperparameter choice in deep learning architectures.",
                source_title="Hyperparameter Tuning Survey",
            ),
            EvidenceCandidate(
                passage_id="p_irrelevant",
                text="Convolutional filters extract spatial edges from chest X-ray radiographs.",
                source_title="Medical Imaging Baselines",
            ),
        ]

        summary = rank_and_verify_evidence(
            claim,
            candidates,
            top_k=3,
            use_embeddings=False,
            use_llm_nli=False,
        )

        self.assertIsNotNone(summary.top_evidence)
        self.assertEqual(summary.top_evidence.passage_id, "p_contradict")
        self.assertEqual(summary.top_evidence.logical_stance, LogicalStance.CONTRADICTS)
        self.assertTrue(summary.top_evidence.similarity_trap)
        self.assertTrue(summary.similarity_trap_detected)
        self.assertEqual(summary.recommended_verdict, ClaimStatus.CONTRADICTORY)
        self.assertIn("POLARITY_CONTRADICTION", summary.top_evidence.diagnostic_flags)
        self.assertIn("SIMILARITY_TRAP_CONTRADICTION", summary.top_evidence.diagnostic_flags)

    # Inputs: Quantitative EyePACS-1 claim with supporting passage, topical neutral passage, and distractor.
    # Outputs: Verifies supporting passage ranks #1 with ENTAILS and no similarity trap.
    def test_supporting_evidence_distinguished_from_neutral(self):
        claim = "Deep learning achieved 97.5% sensitivity on EyePACS-1."
        candidates = [
            EvidenceCandidate(
                passage_id="p_distractor",
                text="Randomized trials evaluate steroid therapy in hospitalized respiratory cohorts.",
            ),
            EvidenceCandidate(
                passage_id="p_neutral",
                text="Diabetic retinopathy screening programs frequently review EyePACS-1 fundus photographs.",
            ),
            EvidenceCandidate(
                passage_id="p_support",
                text="On EyePACS-1, the deep learning algorithm achieved 97.5% sensitivity and 93.4% specificity.",
            ),
        ]

        summary = rank_and_verify_evidence(claim, candidates, top_k=3)

        self.assertIsNotNone(summary.top_evidence)
        self.assertEqual(summary.top_evidence.passage_id, "p_support")
        self.assertEqual(summary.top_evidence.logical_stance, LogicalStance.ENTAILS)
        self.assertFalse(summary.similarity_trap_detected)
        self.assertEqual(summary.recommended_verdict, ClaimStatus.SUPPORTED)

    # Inputs: Claim and evidence identical except for explicit negation ("did not reduce").
    # Outputs: Verifies negation flip is caught as CONTRADICTS and similarity_trap.
    def test_negation_flip_caught_as_contradiction(self):
        claim = "Vitamin D supplementation reduces winter fracture risk in older adults."
        candidates = [
            EvidenceCandidate(
                passage_id="p_negated",
                text="Vitamin D supplementation did not reduce winter fracture risk in older adults.",
            )
        ]

        summary = rank_and_verify_evidence(claim, candidates, top_k=1)

        self.assertEqual(summary.top_evidence.logical_stance, LogicalStance.CONTRADICTS)
        self.assertEqual(summary.recommended_verdict, ClaimStatus.CONTRADICTORY)
        self.assertTrue(summary.similarity_trap_detected)
        self.assertIn("NEGATION_CONTRADICTION", summary.top_evidence.diagnostic_flags)

    # Inputs: Claim reversing mortality direction and percentage vs RECOVERY trial evidence.
    # Outputs: Verifies both polarity and numeric contradictions are flagged.
    def test_numeric_and_direction_contradiction(self):
        claim = "Dexamethasone increased 28-day mortality to 62.4% in ventilated patients."
        candidates = [
            EvidenceCandidate(
                passage_id="p_recovery",
                text="In ventilated patients, dexamethasone reduced 28-day mortality to 22.9% compared with 25.7% in usual care.",
            )
        ]

        summary = rank_and_verify_evidence(claim, candidates, top_k=1)

        self.assertEqual(summary.top_evidence.logical_stance, LogicalStance.CONTRADICTS)
        self.assertEqual(summary.recommended_verdict, ClaimStatus.CONTRADICTORY)
        self.assertIn("NUMERIC_CONTRADICTION", summary.top_evidence.diagnostic_flags)
        self.assertIn("POLARITY_CONTRADICTION", summary.top_evidence.diagnostic_flags)

    # Inputs: Claim with quantitative assertion paired only with a topically related background passage.
    # Outputs: Verifies topical overlap alone yields NEUTRAL / UNSUPPORTED rather than false SUPPORTED.
    def test_topical_overlap_without_entailment_stays_neutral(self):
        claim = "Sparse clinical transformers reduce ICU mortality by 15%."
        candidates = [
            EvidenceCandidate(
                passage_id="p_background",
                text="Sparse clinical transformers are neural sequence architectures applied to ICU time-series records.",
            )
        ]

        summary = rank_and_verify_evidence(claim, candidates, top_k=1)

        self.assertEqual(summary.top_evidence.logical_stance, LogicalStance.NEUTRAL)
        self.assertEqual(summary.recommended_verdict, ClaimStatus.UNSUPPORTED)
        self.assertFalse(summary.similarity_trap_detected)

    # Inputs: BM25 and RRF helper inputs with mixed lexical/semantic rankings.
    # Outputs: Verifies BM25 scores relevant terms higher and RRF promotes consensus candidates.
    def test_bm25_and_hybrid_rrf_scoring(self):
        passages = [
            "Attention mechanisms achieve 28.4 BLEU on WMT 2014 English-to-German translation.",
            "Classical statistical phrase tables were used in older translation systems.",
        ]
        bm25 = score_bm25("Transformers achieve 28.4 BLEU on WMT 2014 translation.", passages)
        self.assertGreater(bm25[0], bm25[1])

        rrf = reciprocal_rank_fusion([2.5, 0.4, 1.0], [0.8, 0.9, 0.2], rrf_k=60)
        self.assertEqual(len(rrf), 3)
        self.assertGreater(rrf[0], rrf[2])


if __name__ == "__main__":
    unittest.main()