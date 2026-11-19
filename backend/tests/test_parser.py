import unittest
from pathlib import Path

import pymupdf

from backend.schemas import ExtractedClaim
from backend.services.parser import (
    clean_extracted_text,
    extract_claims_from_pdf,
    extract_claims_from_text,
    extract_document_title,
    extract_text_from_pdf,
    split_body_and_references,
)


class TestDocumentParser(unittest.TestCase):
    # Inputs: An in-memory PDF containing a hyphenated word and a References section.
    # Outputs: Verifies PDF extraction, hyphen rejoining, and body/reference splitting work.
    def test_pdf_extraction_cleaning_and_split(self):
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text(
            (72, 72),
            "Neural nets reach 95% accu-\nracy [1].\n\nReferences\n[1] Smith, 2024.",
        )
        pdf_bytes = doc.tobytes()
        doc.close()

        raw_text = extract_text_from_pdf(pdf_bytes)
        clean_text = clean_extracted_text(raw_text)
        body, refs = split_body_and_references(clean_text)

        self.assertIn("accuracy [1].", body)
        self.assertIn("[1] Smith, 2024.", refs)

    # Inputs: An in-memory PDF with numeric citations [1] and [1, 2], plus an uncited statistic.
    # Outputs: Verifies ExtractedClaim objects map markers to exact references and filter noise.
    def test_numeric_citations_from_pdf(self):
        sample_pdf_text = (
            "Impact of Sleep on Memory Consolidation\n\n"
            "Introduction\n"
            "Does sleep matter for students?\n"
            "Deep sleep improves memory recall by 35.5% in controlled trials [1].\n"
            "Recent reviews confirm consistent gains across age groups [1, 2].\n"
            "Overall hospital readmissions fell by 18% during the two-year pilot.\n\n"
            "References\n"
            "[1] Walker, M. (2019). Sleep and Memory Consolidation. Nature Reviews.\n"
            "[2] Patel, R. (2022). Longitudinal Cognitive Outcomes. Lancet Neurology.\n"
        )

        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), sample_pdf_text)
        pdf_bytes = doc.tobytes()
        doc.close()

        claims = extract_claims_from_pdf(pdf_bytes)

        self.assertEqual(len(claims), 3)
        self.assertIsInstance(claims[0], ExtractedClaim)

        # Claim 1: Single numeric citation [1]
        self.assertEqual(claims[0].claim_id, 1)
        self.assertEqual(claims[0].inline_marker, "[1]")
        self.assertIn("Walker, M. (2019)", claims[0].raw_reference)

        # Claim 2: Multi numeric citation [1, 2]
        self.assertEqual(claims[1].claim_id, 2)
        self.assertEqual(claims[1].inline_marker, "[1, 2]")
        self.assertIn("Walker, M. (2019)", claims[1].raw_reference)
        self.assertIn("Patel, R. (2022)", claims[1].raw_reference)

        # Claim 3: Uncited quantitative claim (still extracted for verification)
        self.assertEqual(claims[2].claim_id, 3)
        self.assertIsNone(claims[2].inline_marker)
        self.assertIsNone(claims[2].raw_reference)

    # Inputs: Raw pasted academic text using APA/Harvard (Author, Year) and narrative citations.
    # Outputs: Verifies parenthetical and narrative author-year citations match their references.
    def test_author_year_citations_from_text(self):
        sample_text = (
            "Climate Models and Urban Heat\n\n"
            "Urban tree canopy reduces peak summer temperatures by 2.4 degrees (Smith et al., 2020).\n"
            "Furthermore, Doe (2023) demonstrated that reflective roofs lower cooling energy demand.\n\n"
            "References\n"
            "Smith, J., Lee, K., & Rao, P. (2020). Urban Forestry and Heat Mitigation. Science.\n\n"
            "Doe, A. (2023). Cool Roof Performance in Dense Cities. Energy Journal.\n"
        )

        self.assertEqual(extract_document_title(sample_text), "Climate Models and Urban Heat")
        claims = extract_claims_from_text(sample_text)

        self.assertEqual(len(claims), 2)
        self.assertEqual(claims[0].inline_marker, "(Smith et al., 2020)")
        self.assertIn("Urban Forestry and Heat Mitigation", claims[0].raw_reference)

        self.assertEqual(claims[1].inline_marker, "Doe (2023)")
        self.assertIn("Cool Roof Performance", claims[1].raw_reference)

    # Inputs: The 2-page demo_paper.pdf stored in backend/tests/sample_papers/.
    # Outputs: Verifies all 5 planted claims and their [1]-[5] references extract cleanly.
    def test_demo_paper_pdf_extraction(self):
        demo_pdf = Path(__file__).parent / "sample_papers" / "demo_paper.pdf"
        self.assertTrue(demo_pdf.exists(), f"Missing demo PDF at {demo_pdf}")

        claims = extract_claims_from_pdf(demo_pdf)

        self.assertEqual(len(claims), 5)
        self.assertEqual([c.claim_id for c in claims], [1, 2, 3, 4, 5])
        self.assertEqual(
            [c.inline_marker for c in claims],
            ["[1]", "[2]", "[3]", "[4]", "[5]"],
        )

        expected_pairs = [
            ("97.5% sensitivity", "Gulshan"),
            ("30 countries", "McKinney"),
            ("62.4%", "Horby"),
            ("74.2%", "Kowalski"),
            ("hydroxychloroquine", "Mehra"),
        ]
        for claim, (claim_snippet, ref_author) in zip(claims, expected_pairs, strict=True):
            self.assertIn(claim_snippet, claim.claim_text)
            self.assertIsNotNone(claim.raw_reference)
            self.assertIn(ref_author, claim.raw_reference)

    # Inputs: Synthetic 2-column IEEE/arXiv PDF with full-width title, cross-column claim, and Column 2 references.
    # Outputs: Verifies reading order reads left column completely before right column without interleaving.
    def test_two_column_ieee_arxiv_pdf_layout(self):
        doc = pymupdf.open()
        page = doc.new_page(width=612, height=792)

        # Full-width IEEE title banner
        page.insert_textbox(
            pymupdf.Rect(50, 40, 562, 95),
            "Sparse Attention Architectures for Clinical Time-Series Forecasting",
            fontsize=14,
            fontname="hebo",
        )

        # Left column (x=50..290): Section I + Claim 1 + first half of Claim 2 wrapping to right column
        page.insert_textbox(
            pymupdf.Rect(50, 120, 290, 145),
            "I. INTRODUCTION",
            fontsize=11,
            fontname="hebo",
        )
        page.insert_textbox(
            pymupdf.Rect(50, 150, 290, 270),
            "Transformer architectures achieve 28.4 BLEU on the WMT 2014 English-to-German translation benchmark [1].",
            fontsize=10,
        )
        page.insert_textbox(
            pymupdf.Rect(50, 290, 290, 380),
            "In multi-hospital ICU telemetry benchmarks, sparse clinical encoders reduced inference latency by 41.2%",
            fontsize=10,
        )

        # Right column (x=320..562): Continuation of Claim 2 + Claim 3 + References
        page.insert_textbox(
            pymupdf.Rect(320, 120, 562, 195),
            "while maintaining equivalent mortality prediction accuracy across five cohorts [2].",
            fontsize=10,
        )
        page.insert_textbox(
            pymupdf.Rect(320, 210, 562, 235),
            "II. RESULTS",
            fontsize=11,
            fontname="hebo",
        )
        page.insert_textbox(
            pymupdf.Rect(320, 240, 562, 330),
            "Ablation trials confirmed that quantized weights lowered GPU memory consumption by 52% [3].",
            fontsize=10,
        )
        page.insert_textbox(
            pymupdf.Rect(320, 355, 562, 380),
            "REFERENCES",
            fontsize=11,
            fontname="hebo",
        )
        page.insert_textbox(
            pymupdf.Rect(320, 385, 562, 580),
            (
                "[1] Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS.\n"
                "[2] Chen, L., & Park, H. (2023). Sparse Clinical Encoders. IEEE JBHI.\n"
                "[3] Gomez, R. (2024). Quantized Clinical Transformers. arXiv:2401.01234."
            ),
            fontsize=9.5,
        )

        pdf_bytes = doc.tobytes()
        doc.close()

        claims = extract_claims_from_pdf(pdf_bytes)

        self.assertEqual(len(claims), 3)
        self.assertEqual([c.inline_marker for c in claims], ["[1]", "[2]", "[3]"])
        self.assertIn("28.4 BLEU", claims[0].claim_text)
        self.assertIn("Vaswani", claims[0].raw_reference)
        # Claim 2 must seamlessly rejoin across the column break
        self.assertIn("41.2%", claims[1].claim_text)
        self.assertIn("five cohorts [2].", claims[1].claim_text)
        self.assertIn("Chen, L.", claims[1].raw_reference)
        self.assertIn("52%", claims[2].claim_text)
        self.assertIn("Gomez", claims[2].raw_reference)

    # Inputs: Messy copy-pasted text with ligatures, soft hyphens, broken wraps, page numbers, and split [1] refs.
    # Outputs: Verifies normalization and claim-reference mapping recover cleanly from broken pasted input.
    def test_broken_copy_pasted_text_edge_cases(self):
        messy_text = (
            "   Clinical\u00a0AI   Audit\r\n\r\n"
            "1. Introduction\r\n"
            "Our   \ufb01nal   classi\u00ad\ufb01er   achieved   92.8%   sensi- \r\n"
            "  tivity   in   prospective   sepsis   screening\u200b [1].\r\n"
            "Page 2 of 9\r\n"
            "External   validation   showed   a   31.5%   reduc-\r\n"
            "tion   in   false   alarms   across   urban   clinics [2].\r\n"
            "  - 3 -  \r\n\r\n"
            "References\r\n"
            "[1]\r\n"
            "Nair, P., & Lin, S. (2023). Prospective Sepsis Classi\ufb01ers. Lancet Digital Health.\r\n"
            "[2]   Okafor, D. (2024). Alarm Fatigue\r\n"
            "Reduction in Emergency Medicine. BMJ Open.\r\n"
        )

        claims = extract_claims_from_text(messy_text)

        self.assertEqual(len(claims), 2)
        self.assertIn("final classifier achieved 92.8% sensitivity", claims[0].claim_text)
        self.assertNotIn("Page 2 of 9", claims[0].claim_text)
        self.assertEqual(claims[0].inline_marker, "[1]")
        self.assertIn("Nair, P.", claims[0].raw_reference)
        self.assertIn("Classifiers", claims[0].raw_reference)

        self.assertIn("31.5% reduction in false alarms", claims[1].claim_text)
        self.assertEqual(claims[1].inline_marker, "[2]")
        self.assertIn("Okafor, D.", claims[1].raw_reference)


if __name__ == "__main__":
    unittest.main()