import unittest
import fitz
from backend.services.parser import (
    clean_extracted_text,
    extract_text_from_pdf,
    split_body_and_references,
)


class TestDocumentParser(unittest.TestCase):
    # Inputs: An in-memory PDF containing a hyphenated word and a References section.
    # Outputs: Verifies PDF extraction, hyphen rejoining, and body/reference splitting work.
    def test_pdf_extraction_cleaning_and_split(self):
        doc = fitz.open()
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


if __name__ == "__main__":
    unittest.main()