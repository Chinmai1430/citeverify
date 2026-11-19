import re
from pathlib import Path
from typing import Tuple, Union
import fitz  # PyMuPDF


# Inputs: pdf_source as raw PDF bytes (from FastAPI UploadFile) or a local file path.
# Outputs: Full plain-text string extracted page-by-page from the PDF document.
def extract_text_from_pdf(pdf_source: Union[bytes, str, Path]) -> str:
    if isinstance(pdf_source, bytes):
        if not pdf_source:
            raise ValueError("Uploaded PDF bytes are empty.")
        doc_ctx = fitz.open(stream=pdf_source, filetype="pdf")
    else:
        file_path = Path(pdf_source)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        doc_ctx = fitz.open(file_path)

    pages_text = []
    with doc_ctx as doc:
        for page in doc:
            page_text = page.get_text("text")
            if page_text:
                pages_text.append(page_text)

    combined_text = "\n".join(pages_text).strip()
    if not combined_text:
        raise ValueError("No readable text could be extracted from the PDF.")
    return combined_text


# Inputs: raw_text string directly from a PDF or user-pasted academic document.
# Outputs: Cleaned string with normalized newlines, rejoined hyphenated words, and trimmed spacing.
def clean_extracted_text(raw_text: str) -> str:
    if not raw_text:
        return ""

    # Normalize Windows/Mac line endings to standard newlines
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    # Rejoin words split by PDF line-end hyphens (e.g., "verifi-\ncation" -> "verification")
    text = re.sub(r"([A-Za-z])-\n([A-Za-z])", r"\1\2", text)

    # Collapse multiple spaces/tabs on the same line without destroying line breaks
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse 3 or more blank lines into a clean double newline
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# Inputs: Cleaned full document text containing main content and an optional references section.
# Outputs: Tuple of (body_text, references_text), where references_text is empty if none is found.
def split_body_and_references(text: str) -> Tuple[str, str]:
    if not text:
        return "", ""

    # Match standalone academic reference headings like "References", "7. REFERENCES", or "Works Cited"
    heading_pattern = re.compile(
        r"(?im)^\s*(?:\d+\.?\s+)?(references|bibliography|works\s+cited|literature\s+cited)\s*:?\s*$"
    )

    matches = list(heading_pattern.finditer(text))
    if not matches:
        return text.strip(), ""

    # Use the last match so a Table of Contents entry does not cut the paper early
    last_match = matches[-1]
    body_text = text[: last_match.start()].strip()
    references_text = text[last_match.end() :].strip()

    return body_text, references_text