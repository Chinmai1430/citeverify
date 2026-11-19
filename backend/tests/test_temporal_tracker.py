import unittest

from backend.schemas import ClaimStatus
from backend.services.temporal_tracker import (
    EditorialNotice,
    PublicationStage,
    SubsequentStance,
    SubsequentStudy,
    TemporalCitationProfile,
    TemporalConsensusStatus,
    VersionRecord,
    assess_temporal_validity,
    build_profile_from_openalex_payload,
    classify_subsequent_stance,
    detect_study_type,
)


class TestTemporalTracker(unittest.TestCase):
    # Inputs: Classic 2012 -> 2018 -> 2023 evolving consensus timeline with historical verdict SUPPORTED.
    # Outputs: Verifies separate SUPERSEDED_OR_CHALLENGED assessment without overwriting historical verdict.
    def test_classic_2012_2018_2023_challenged_consensus(self):
        profile = TemporalCitationProfile(
            title="Study A: Early Randomized Trial of Treatment X",
            doi="10.1000/study-a-2012",
            publication_year=2012,
            publication_stage=PublicationStage.PEER_REVIEWED_JOURNAL,
            cited_by_count=410,
            subsequent_studies=[
                SubsequentStudy(
                    title="Study B: Multicenter Replication Trial of Treatment X",
                    year=2018,
                    study_type="replication",
                    stance=SubsequentStance.CHALLENGES,
                    summary="Reported no significant benefit over placebo.",
                ),
                SubsequentStudy(
                    title="Updated Systematic Review and Meta-Analysis of Treatment X",
                    year=2023,
                    study_type="systematic_review",
                    stance=SubsequentStance.INSUFFICIENT_EVIDENCE,
                    summary="Large systematic review found insufficient evidence of efficacy.",
                ),
            ],
        )

        assessment = assess_temporal_validity(
            profile,
            historical_verdict=ClaimStatus.SUPPORTED,
            reference_year=2026,
        )

        self.assertEqual(assessment.historical_claim_verdict, ClaimStatus.SUPPORTED)
        self.assertFalse(assessment.replaces_historical_verdict)
        self.assertEqual(assessment.temporal_status, TemporalConsensusStatus.SUPERSEDED_OR_CHALLENGED)
        self.assertEqual(assessment.citation_age_years, 14)
        self.assertEqual(assessment.challenging_studies_count, 2)
        self.assertEqual(assessment.systematic_reviews_count, 1)
        self.assertTrue(assessment.needs_temporal_review)
        self.assertIn(
            "The cited study supports the historical claim, but subsequent research has challenged the finding",
            assessment.temporal_summary,
        )
        self.assertIn("SUBSEQUENT_CHALLENGE", assessment.temporal_flags)
        self.assertIn("SYSTEMATIC_REVIEW_CONFLICT", assessment.temporal_flags)
        self.assertEqual(len(assessment.chronological_timeline), 3)

    # Inputs: Cited arXiv/medRxiv preprint that now has a newer peer-reviewed journal version.
    # Outputs: Verifies CORRECTED_OR_UPDATED status and PREPRINT_SUPERSEDED flag.
    def test_preprint_superseded_by_journal_version(self):
        profile = TemporalCitationProfile(
            title="Sparse Clinical Forecasting Preprint",
            publication_year=2023,
            publication_stage=PublicationStage.PREPRINT,
            versions=[
                VersionRecord(
                    stage=PublicationStage.PEER_REVIEWED_JOURNAL,
                    doi_url="https://doi.org/10.1038/s41591-024-00001-x",
                    publication_date="2024-05-12",
                    venue="Nature Medicine",
                    is_latest=True,
                )
            ],
        )

        assessment = assess_temporal_validity(
            profile,
            historical_verdict=ClaimStatus.SUPPORTED,
            reference_year=2026,
        )

        self.assertEqual(assessment.temporal_status, TemporalConsensusStatus.CORRECTED_OR_UPDATED)
        self.assertTrue(assessment.has_newer_journal_version)
        self.assertIn("PREPRINT_SUPERSEDED", assessment.temporal_flags)
        self.assertTrue(assessment.needs_temporal_review)

    # Inputs: Cited studies with (a) formal retraction and (b) formal erratum/correction notice.
    # Outputs: Verifies RETRACTED and CORRECTED_OR_UPDATED classifications while keeping historical verdict.
    def test_retraction_and_correction_notices(self):
        retracted_profile = TemporalCitationProfile(
            title="Retracted Registry Analysis",
            publication_year=2020,
            publication_stage=PublicationStage.PEER_REVIEWED_JOURNAL,
            is_retracted=True,
            editorial_notices=[
                EditorialNotice(
                    notice_type="retraction",
                    date="2020-06-04",
                    detail="Retracted due to unverified primary dataset.",
                )
            ],
        )
        retracted_eval = assess_temporal_validity(
            retracted_profile,
            historical_verdict=ClaimStatus.SUPPORTED,
            reference_year=2026,
        )
        self.assertEqual(retracted_eval.temporal_status, TemporalConsensusStatus.RETRACTED)
        self.assertTrue(retracted_eval.is_retracted)
        self.assertIn("RETRACTED_STUDY", retracted_eval.temporal_flags)
        self.assertEqual(retracted_eval.historical_claim_verdict, ClaimStatus.SUPPORTED)

        corrected_profile = TemporalCitationProfile(
            title="Clinical Biomarker Study",
            publication_year=2022,
            publication_stage=PublicationStage.PEER_REVIEWED_JOURNAL,
            editorial_notices=[
                EditorialNotice(
                    notice_type="correction",
                    date="2023-02-10",
                    detail="Table 2 confidence intervals corrected.",
                )
            ],
        )
        corrected_eval = assess_temporal_validity(
            corrected_profile,
            historical_verdict=ClaimStatus.SUPPORTED,
            reference_year=2026,
        )
        self.assertEqual(corrected_eval.temporal_status, TemporalConsensusStatus.CORRECTED_OR_UPDATED)
        self.assertTrue(corrected_eval.has_correction)
        self.assertIn("CORRECTION_ISSUED", corrected_eval.temporal_flags)

    # Inputs: Older study confirmed by a newer 2024 systematic review.
    # Outputs: Verifies CONSENSUS_ALIGNED with needs_temporal_review=False.
    def test_consensus_aligned_with_recent_confirming_review(self):
        profile = TemporalCitationProfile(
            title="Deep Learning for Diabetic Retinopathy",
            publication_year=2016,
            publication_stage=PublicationStage.PEER_REVIEWED_JOURNAL,
            cited_by_count=5200,
            subsequent_studies=[
                SubsequentStudy(
                    title="2024 Systematic Review of Retinopathy Screening AI",
                    year=2024,
                    study_type="systematic_review",
                    stance=SubsequentStance.CONFIRMS,
                    summary="Confirmed high sensitivity across multi-ethnic clinical cohorts.",
                )
            ],
        )

        assessment = assess_temporal_validity(
            profile,
            historical_verdict=ClaimStatus.SUPPORTED,
            reference_year=2026,
        )

        self.assertEqual(assessment.temporal_status, TemporalConsensusStatus.CONSENSUS_ALIGNED)
        self.assertFalse(assessment.needs_temporal_review)
        self.assertEqual(assessment.confirming_studies_count, 1)

    # Inputs: Aging 2010 citation in 2026 with no tracked confirming reviews.
    # Outputs: Verifies STALE_NEEDS_UPDATE and AGING_CITATION flag.
    def test_stale_aging_citation_without_followups(self):
        profile = TemporalCitationProfile(
            title="Legacy Genomics Heuristic",
            publication_year=2010,
            publication_stage=PublicationStage.PEER_REVIEWED_JOURNAL,
            cited_by_count=45,
        )

        assessment = assess_temporal_validity(
            profile,
            historical_verdict=ClaimStatus.SUPPORTED,
            reference_year=2026,
            staleness_threshold_years=8,
        )

        self.assertEqual(assessment.temporal_status, TemporalConsensusStatus.STALE_NEEDS_UPDATE)
        self.assertIn("AGING_CITATION", assessment.temporal_flags)
        self.assertTrue(assessment.needs_temporal_review)

    # Inputs: Sample abstracts/titles for subsequent citing studies.
    # Outputs: Verifies stance classification and study-type detection helpers.
    def test_subsequent_stance_and_study_type_classifiers(self):
        self.assertEqual(
            detect_study_type("A Cochrane Systematic Review of Steroids in ICU"),
            "systematic_review",
        )
        self.assertEqual(
            detect_study_type("Updated Meta-Analysis of Treatment Efficacy"),
            "meta_analysis",
        )
        self.assertEqual(
            classify_subsequent_stance("Trial B reported no significant benefit over standard care."),
            SubsequentStance.CHALLENGES,
        )
        self.assertEqual(
            classify_subsequent_stance("The systematic review found insufficient evidence to recommend use."),
            SubsequentStance.INSUFFICIENT_EVIDENCE,
        )
        self.assertEqual(
            classify_subsequent_stance("Multicenter validation confirmed significant benefit in mortality."),
            SubsequentStance.CONFIRMS,
        )

    # Inputs: Simulated OpenAlex work, citing-work inverted abstract, and Crossref update/preprint payload.
    # Outputs: Verifies offline API payload parser builds a complete TemporalCitationProfile.
    def test_build_profile_from_openalex_and_crossref_payloads(self):
        work = {
            "display_name": "Early Treatment Trial",
            "doi": "https://doi.org/10.1101/2021.01.01.000001",
            "publication_year": 2021,
            "publication_date": "2021-01-05",
            "type": "preprint",
            "is_retracted": False,
            "cited_by_count": 32,
            "primary_location": {"source": {"display_name": "medRxiv"}},
        }
        citing_works = [
            {
                "display_name": "Systematic Review of Early Treatment",
                "publication_year": 2024,
                "doi": "https://doi.org/10.1000/rev2024",
                "cited_by_count": 18,
                "abstract_inverted_index": {
                    "This": [0],
                    "review": [1],
                    "found": [2],
                    "insufficient": [3],
                    "evidence": [4],
                    "for": [5],
                    "efficacy.": [6],
                },
            }
        ]
        crossref_msg = {
            "updated-by": [
                {
                    "type": "correction",
                    "label": "Author Correction",
                    "DOI": "10.1000/corr2022",
                    "updated": {"date-parts": [[2022, 4, 15]]},
                }
            ],
            "relation": {
                "is-preprint-of": [{"id": "10.1056/NEJMoa2022001"}],
            },
        }

        profile = build_profile_from_openalex_payload(
            work,
            citing_works=citing_works,
            crossref_message=crossref_msg,
        )

        self.assertEqual(profile.publication_stage, PublicationStage.PREPRINT)
        self.assertEqual(profile.doi, "10.1101/2021.01.01.000001")
        self.assertEqual(len(profile.versions), 1)
        self.assertEqual(len(profile.editorial_notices), 1)
        self.assertEqual(len(profile.subsequent_studies), 1)
        self.assertEqual(profile.subsequent_studies[0].study_type, "systematic_review")
        self.assertEqual(profile.subsequent_studies[0].stance, SubsequentStance.INSUFFICIENT_EVIDENCE)


if __name__ == "__main__":
    unittest.main()