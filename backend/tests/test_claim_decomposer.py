import unittest

from backend.schemas import (
    ClaimStatus,
    ClaimVerificationResult,
    ConfidenceLevel,
    ExtractedClaim,
)
from backend.services.claim_decomposer import (
    PropositionVerdict,
    combine_proposition_verdicts,
    decompose_claim,
    is_compound_claim,
    verify_decomposed_claim,
)


class TestClaimDecomposer(unittest.TestCase):
    # Inputs: Compound claim comparing Transformer accuracy vs computational efficiency.
    # Outputs: Verifies 2 standalone atomic propositions preserving subject and qualifiers.
    def test_transformer_vs_cnn_compound_decomposition(self):
        claim = ExtractedClaim(
            claim_id=1,
            claim_text=(
                "Transformer models outperform CNNs in image classification "
                "while requiring fewer computational resources [1]."
            ),
            inline_marker="[1]",
            raw_reference="[1] Dosovitskiy et al. (2021). ViT. ICLR.",
        )

        self.assertTrue(is_compound_claim(claim.claim_text))
        decomposed = decompose_claim(claim, use_llm=False)

        self.assertTrue(decomposed.is_compound)
        self.assertEqual(len(decomposed.propositions), 2)
        self.assertIn("Transformer models outperform CNNs in image classification", decomposed.propositions[0].text)
        self.assertIn("[1]", decomposed.propositions[0].text)
        self.assertIn("Transformer models require fewer computational resources", decomposed.propositions[1].text)
        self.assertIn("fewer", decomposed.propositions[1].preserved_qualifiers)
        self.assertIn("[1]", decomposed.propositions[1].text)

    # Inputs: Clinical compound claim with leading population qualifier and two numerical metrics.
    # Outputs: Verifies both atomic propositions inherit the population qualifier, subject, and numbers.
    def test_preserves_population_qualifier_and_numbers(self):
        claim = ExtractedClaim(
            claim_id=2,
            claim_text=(
                "In elderly diabetic patients, Drug X reduced hospital readmissions by 28% "
                "and lowered systolic blood pressure by 12 mmHg [2]."
            ),
            inline_marker="[2]",
            raw_reference="[2] Patel et al. (2024). Clinical Trial. NEJM.",
        )

        decomposed = decompose_claim(claim, use_llm=False)

        self.assertTrue(decomposed.is_compound)
        self.assertEqual(len(decomposed.propositions), 2)
        self.assertIn("In elderly diabetic patients,", decomposed.propositions[0].text)
        self.assertIn("28%", decomposed.propositions[0].text)
        self.assertIn("In elderly diabetic patients,", decomposed.propositions[1].text)
        self.assertIn("Drug X lowered systolic blood pressure by 12 mmHg", decomposed.propositions[1].text)

    # Inputs: Single atomic claim with a coordinate noun phrase ("sensitivity and specificity").
    # Outputs: Verifies it is NOT falsely split into broken fragments.
    def test_simple_atomic_claim_not_falsely_split(self):
        claim = ExtractedClaim(
            claim_id=3,
            claim_text="Deep learning achieved 97.5% sensitivity and 93.4% specificity on EyePACS-1 [1].",
            inline_marker="[1]",
            raw_reference="[1] Gulshan et al. (2016). JAMA.",
        )

        self.assertFalse(is_compound_claim(claim.claim_text))
        decomposed = decompose_claim(claim, use_llm=False)

        self.assertFalse(decomposed.is_compound)
        self.assertEqual(len(decomposed.propositions), 1)
        self.assertIn("97.5% sensitivity and 93.4% specificity", decomposed.propositions[0].text)

    # Inputs: Semicolon compound claim where Clause 2 introduces its own distinct subject.
    # Outputs: Verifies Clause 2 keeps its own subject instead of inheriting Clause 1's subject.
    def test_semicolon_distinct_subjects(self):
        claim = ExtractedClaim(
            claim_id=4,
            claim_text=(
                "Deep learning reached 97.5% sensitivity on EyePACS-1; "
                "however, specificity dropped to 81.2% in rural clinics [3]."
            ),
            inline_marker="[3]",
            raw_reference="[3] Validation Study (2025).",
        )

        decomposed = decompose_claim(claim, use_llm=False)

        self.assertEqual(len(decomposed.propositions), 2)
        self.assertIn("Deep learning reached 97.5% sensitivity", decomposed.propositions[0].text)
        self.assertTrue(decomposed.propositions[1].text.startswith("Specificity dropped to 81.2%"))

    # Inputs: Compound claim where Proposition 1 is SUPPORTED and Proposition 2 is CONTRADICTORY.
    # Outputs: Verifies end-to-end atomic verification combines them into PARTIALLY_SUPPORTED + flag.
    def test_verify_and_combine_mixed_support_and_contradiction(self):
        claim = ExtractedClaim(
            claim_id=5,
            claim_text=(
                "Transformer models outperform CNNs in image classification "
                "while requiring fewer computational resources [1]."
            ),
            inline_marker="[1]",
            raw_reference="[1] Dosovitskiy et al. (2021). ViT. ICLR.",
        )

        def _mock_verify(sub_claim: ExtractedClaim) -> ClaimVerificationResult:
            if "outperform CNNs" in sub_claim.claim_text:
                return ClaimVerificationResult(
                    claim_id=sub_claim.claim_id,
                    claim_text=sub_claim.claim_text,
                    verdict=ClaimStatus.SUPPORTED,
                    explanation="ViT outperformed ResNet baselines when pretrained at scale.",
                    confidence=ConfidenceLevel.HIGH,
                    uncertainty_note="Confirmed in abstract.",
                    needs_human_review=False,
                )
            return ClaimVerificationResult(
                claim_id=sub_claim.claim_id,
                claim_text=sub_claim.claim_text,
                verdict=ClaimStatus.CONTRADICTORY,
                explanation="Transformers required substantially more compute and data than CNNs.",
                confidence=ConfidenceLevel.HIGH,
                uncertainty_note="Direct compute contradiction.",
                needs_human_review=True,
                mismatch_flags=["COMPUTE_CONTRADICTION"],
            )

        combined = verify_decomposed_claim(claim, verify_fn=_mock_verify, use_llm_decomposition=False)

        self.assertEqual(combined.overall_verdict, ClaimStatus.PARTIALLY_SUPPORTED)
        self.assertTrue(combined.needs_human_review)
        self.assertIn("PARTIAL_CONTRADICTION", combined.mismatch_flags)
        self.assertIn("COMPUTE_CONTRADICTION", combined.mismatch_flags)
        self.assertEqual(len(combined.proposition_verdicts), 2)
        self.assertEqual(combined.proposition_verdicts[0].verdict, ClaimStatus.SUPPORTED)
        self.assertEqual(combined.proposition_verdicts[1].verdict, ClaimStatus.CONTRADICTORY)

    # Inputs: Proposition verdicts where one proposition cites a retracted/outdated study.
    # Outputs: Verifies POTENTIALLY_OUTDATED takes precedence in verdict aggregation.
    def test_combine_outdated_precedence(self):
        claim = ExtractedClaim(claim_id=6, claim_text="Claim A while Claim B.")
        combined = combine_proposition_verdicts(
            claim,
            [
                PropositionVerdict(
                    proposition_id="6.1",
                    proposition_text="Claim A.",
                    verdict=ClaimStatus.SUPPORTED,
                    confidence=ConfidenceLevel.HIGH,
                    explanation="Matches text.",
                ),
                PropositionVerdict(
                    proposition_id="6.2",
                    proposition_text="Claim B.",
                    verdict=ClaimStatus.POTENTIALLY_OUTDATED,
                    confidence=ConfidenceLevel.HIGH,
                    explanation="Source retracted.",
                ),
            ],
        )
        self.assertEqual(combined.overall_verdict, ClaimStatus.POTENTIALLY_OUTDATED)
        self.assertIn("RETRACTED_OR_OUTDATED_PROPOSITION", combined.mismatch_flags)


if __name__ == "__main__":
    unittest.main()