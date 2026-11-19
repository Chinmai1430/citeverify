import unittest
from datetime import UTC, datetime

import pymupdf

from backend.schemas import (
    ClaimStatus,
    ClaimVerificationResult,
    ConfidenceLevel,
    VerificationReport,
)
from backend.services.report_exporter import (
    build_export_filename,
    export_report_to_markdown,
    export_report_to_pdf_bytes,
)


def _sample_report(claim_count: int = 5) -> VerificationReport:
    base_results = [
        ClaimVerificationResult(
            claim_id=1,
            claim_text="Deep learning achieved 97.5% sensitivity on EyePACS-1 [1].",
            matched_citation="Gulshan et al. (2016) — JAMA",
            evidence_passage="Sensitivity was 97.5% and specificity was 93.4% on EyePACS-1.",
            verdict=ClaimStatus.SUPPORTED,
            explanation="Metrics match the published JAMA abstract verbatim.",
            confidence=ConfidenceLevel.HIGH,
            uncertainty_note="Verified against full abstract.",
            needs_human_review=False,
            source_url="https://doi.org/10.1001/jama.2016.17216",
            mismatch_flags=[],
        ),
        ClaimVerificationResult(
            claim_id=2,
            claim_text="Prospective trials across 30 countries replaced human radiologists [2].",
            matched_citation="McKinney et al. (2020) — Nature",
            evidence_passage="Evaluated on retrospective datasets from the UK and USA.",
            verdict=ClaimStatus.PARTIALLY_SUPPORTED,
            explanation="Real study exists, but scope is exaggerated beyond UK/US retrospective data.",
            confidence=ConfidenceLevel.MEDIUM,
            uncertainty_note="Abstract-only comparison.",
            needs_human_review=True,
            source_url="https://doi.org/10.1038/s41586-019-1799-6",
            mismatch_flags=["SCOPE_EXAGGERATION"],
        ),
        ClaimVerificationResult(
            claim_id=3,
            claim_text="Dexamethasone increased 28-day mortality to 62.4% [3].",
            matched_citation="Horby et al. (2021) — NEJM",
            evidence_passage="28-day mortality was 22.9% in the dexamethasone group vs 25.7% usual care.",
            verdict=ClaimStatus.CONTRADICTORY,
            explanation="Claim reverses both direction and mortality percentage.",
            confidence=ConfidenceLevel.HIGH,
            uncertainty_note="Direct quantitative contradiction found.",
            needs_human_review=True,
            source_url="https://doi.org/10.1056/NEJMoa2021436",
            mismatch_flags=["NUMERIC_MISMATCH"],
        ),
        ClaimVerificationResult(
            claim_id=4,
            claim_text="Autonomous transformer cut misdiagnosis by 74.2% [4].",
            matched_citation=None,
            evidence_passage=None,
            verdict=ClaimStatus.UNSUPPORTED,
            explanation="No matching publication found in OpenAlex, Crossref, or Europe PMC.",
            confidence=ConfidenceLevel.LOW,
            uncertainty_note="Reference could not be retrieved from scholarly databases.",
            needs_human_review=True,
            source_url=None,
            mismatch_flags=["SOURCE_NOT_FOUND"],
        ),
        ClaimVerificationResult(
            claim_id=5,
            claim_text="Hydroxychloroquine registry analysis reported lower survival [5].",
            matched_citation="Mehra et al. (2020) — The Lancet",
            evidence_passage="This article has been retracted.",
            verdict=ClaimStatus.POTENTIALLY_OUTDATED,
            explanation="Cited Lancet study was formally retracted in June 2020.",
            confidence=ConfidenceLevel.HIGH,
            uncertainty_note="Retraction notice flagged via metadata.",
            needs_human_review=True,
            source_url="https://doi.org/10.1016/S0140-6736(20)31180-6",
            mismatch_flags=["RETRACTED_SOURCE"],
        ),
    ]

    results: list[ClaimVerificationResult] = []
    for idx in range(claim_count):
        template = base_results[idx % len(base_results)]
        results.append(template.model_copy(update={"claim_id": idx + 1}))

    summary_counts: dict[str, int] = {}
    for res in results:
        key = res.verdict.value
        summary_counts[key] = summary_counts.get(key, 0) + 1

    return VerificationReport(
        document_title="Rapid Clinical Evidence Review: AI & Therapeutics (2026)",
        total_claims=len(results),
        summary_counts=summary_counts,
        results=results,
        limitations=[
            "Automated verification relied on abstracts where full-text PDFs were paywalled.",
            "Human domain expert review is advised before clinical policy changes.",
        ],
    )


class TestReportExporter(unittest.TestCase):
    # Inputs: Titles with punctuation, spaces, and empty strings across md/pdf extensions.
    # Outputs: Verifies clean, URL-safe download filenames.
    def test_build_export_filename(self):
        self.assertEqual(
            build_export_filename("Rapid Clinical Review: AI (2026)!", "pdf"),
            "citeverify_audit_rapid_clinical_review_ai_2026.pdf",
        )
        self.assertEqual(
            build_export_filename("   ", ".MD"),
            "citeverify_audit_untitled_document.md",
        )

    # Inputs: 5-claim VerificationReport covering all 5 ClaimStatus values.
    # Outputs: Verifies Markdown contains executive table, claim cards, URLs, flags, and limitations.
    def test_export_report_to_markdown(self):
        report = _sample_report(5)
        fixed_time = datetime(2026, 11, 19, 12, 30, tzinfo=UTC)
        md = export_report_to_markdown(report, generated_at=fixed_time)

        self.assertIn("# CiteVerify Citation & Claim Audit Report", md)
        self.assertIn("2026-11-19 12:30 UTC", md)
        self.assertIn("Flagged for Human Review:** 4 / 5", md)
        self.assertIn("| **Supported** (`SUPPORTED`) | 1 | 20.0% |", md)
        self.assertIn("### Claim 3 — Contradictory", md)
        self.assertIn("`RETRACTED_SOURCE`", md)
        self.assertIn("https://doi.org/10.1056/NEJMoa2021436", md)
        self.assertIn("## Verification Scope & Limitations", md)

    # Inputs: 5-claim report and a 12-claim stress report to force multi-page PDF pagination.
    # Outputs: Verifies valid PDF bytes, readable extracted text, and multiple paginated pages.
    def test_export_report_to_pdf_bytes_and_pagination(self):
        fixed_time = datetime(2026, 11, 19, 12, 30, tzinfo=UTC)
        small_pdf = export_report_to_pdf_bytes(_sample_report(5), generated_at=fixed_time)
        self.assertTrue(small_pdf.startswith(b"%PDF-"))

        with pymupdf.open(stream=small_pdf, filetype="pdf") as doc:
            extracted = "\n".join(page.get_text("text") for page in doc)
            self.assertIn("CiteVerify", extracted)
            self.assertIn("Executive Verdict Summary", extracted)
            self.assertIn("RETRACTED_SOURCE", extracted)
            self.assertIn("Verification Scope & Limitations", extracted)
            self.assertIn(f"Page 1 of {len(doc)}", extracted)

        large_pdf = export_report_to_pdf_bytes(_sample_report(12), generated_at=fixed_time)
        with pymupdf.open(stream=large_pdf, filetype="pdf") as doc:
            self.assertGreaterEqual(len(doc), 3)
            last_page_text = doc[-1].get_text("text")
            self.assertIn("Claim 12", "\n".join(p.get_text("text") for p in doc))
            self.assertIn(f"Page {len(doc)} of {len(doc)}", last_page_text)


if __name__ == "__main__":
    unittest.main()