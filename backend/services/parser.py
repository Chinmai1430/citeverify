import re
from pathlib import Path

import pymupdf

# Support running from repo root (backend.schemas) or inside backend/ (schemas)
try:
    from backend.schemas import ExtractedClaim
    from backend.services.citation_matcher import (
        detect_inline_citations,
        match_marker_to_reference,
        parse_reference_list,
    )
except ImportError:
    from schemas import ExtractedClaim

    from services.citation_matcher import (
        detect_inline_citations,
        match_marker_to_reference,
        parse_reference_list,
    )


# Common academic abbreviations whose trailing dot should not split a sentence
_ABBREVIATIONS = (
    "et al.",
    "e.g.",
    "i.e.",
    "fig.",
    "figs.",
    "eq.",
    "dr.",
    "prof.",
    "vs.",
    "vol.",
    "no.",
    "pp.",
    "etc.",
    "ref.",
    "sec.",
)

# Standalone section heading lines that should not merge into body sentences
_SECTION_HEADING_PATTERN = re.compile(
    r"^(?:"
    r"\d+(?:\.\d+)*\.?\s+[A-Z][^\n.!?]{1,90}"
    r"|(?:abstract|introduction|background|methods?|methodology|"
    r"materials?\s+and\s+methods?|results?|discussion|conclusions?|"
    r"limitations?|acknowledgements?)\s*:?"
    r")$",
    re.IGNORECASE,
)

# Phrases that signal factual findings, metrics, or conclusions needing evidence
_CLAIM_SIGNAL_PATTERN = re.compile(
    r"(?i)\b("
    r"shows?|showed|shown|demonstrates?|demonstrated|proves?|proved|proven|"
    r"indicates?|indicated|reveals?|revealed|suggests?|suggested|confirms?|confirmed|"
    r"reduces?|reduced|reduction|increases?|increased|improves?|improved|improvement|"
    r"outperforms?|outperformed|achieves?|achieved|causes?|caused|correlates?|correlated|"
    r"significantly|statistically|concludes?|concluded|found\s+that|results?\s+show|"
    r"evidence|according\s+to|associated\s+with|higher\s+than|lower\s+than|more\s+likely"
    r")\b"
)

# Numbers/statistics that typically represent verifiable empirical claims
_QUANTITATIVE_PATTERN = re.compile(
    r"(?i)(?:\b\d+(?:\.\d+)?\s*%|\bp\s*[<>=]\s*0?\.\d+|\b\d+(?:\.\d+)?\s*(?:x|fold|times|million|billion|thousand|mg|kg|km|hours?|days?|years?)\b)"
)


# Inputs: pdf_source as raw PDF bytes (from FastAPI UploadFile) or a local file path.
# Outputs: Full plain-text string extracted block-by-block with preserved paragraph boundaries.
def extract_text_from_pdf(pdf_source: bytes | str | Path) -> str:
    if isinstance(pdf_source, bytes):
        if not pdf_source:
            raise ValueError("Uploaded PDF bytes are empty.")
        doc_ctx = pymupdf.open(stream=pdf_source, filetype="pdf")
    else:
        file_path = Path(pdf_source)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        doc_ctx = pymupdf.open(file_path)

    pages_text: list[str] = []
    with doc_ctx as doc:
        for page in doc:
            # Extract text blocks in visual reading order so headings stay separate from body text
            blocks = sorted(
                (b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()),
                key=lambda b: (round(b[1], 1), round(b[0], 1)),
            )
            if blocks:
                page_text = "\n\n".join(b[4].strip() for b in blocks)
                pages_text.append(page_text)

    combined_text = "\n\n".join(pages_text).strip()
    if not combined_text:
        raise ValueError("No readable text could be extracted from the PDF.")
    return combined_text


# Inputs: raw_text string directly from a PDF or user-pasted academic document.
# Outputs: Cleaned string with normalized newlines, rejoined hyphenated words, and trimmed spacing.
def clean_extracted_text(raw_text: str) -> str:
    if not raw_text:
        return ""

    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"([A-Za-z])-\n([A-Za-z])", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# Inputs: Cleaned full document text containing main content and an optional references section.
# Outputs: Tuple of (body_text, references_text), where references_text is empty if none is found.
def split_body_and_references(text: str) -> tuple[str, str]:
    if not text:
        return "", ""

    heading_pattern = re.compile(
        r"(?im)^\s*(?:\d+\.?\s+)?(references|bibliography|works\s+cited|literature\s+cited)\s*:?\s*$"
    )

    matches = list(heading_pattern.finditer(text))
    if not matches:
        return text.strip(), ""

    last_match = matches[-1]
    body_text = text[: last_match.start()].strip()
    references_text = text[last_match.end() :].strip()
    return body_text, references_text


# Inputs: Cleaned document text.
# Outputs: Best-guess document title string derived from the opening non-empty line.
def extract_document_title(text: str) -> str:
    for line in text.splitlines():
        candidate = line.strip()
        if len(candidate) >= 5 and not candidate.lower().startswith(("abstract", "http")):
            return candidate[:150]
    return "Untitled Document"


# Inputs: Main document body text (without the references section).
# Outputs: List of individual cleaned sentences with headings removed and decimals kept intact.
def split_into_sentences(body_text: str) -> list[str]:
    if not body_text or not body_text.strip():
        return []

    sentences: list[str] = []
    paragraphs = re.split(r"\n\s*\n", body_text.strip())

    for paragraph in paragraphs:
        # Strip leading standalone section headings (e.g., "Introduction" or "2. Methods")
        lines = [ln.strip() for ln in paragraph.splitlines() if ln.strip()]
        while lines and _SECTION_HEADING_PATTERN.fullmatch(lines[0]):
            lines.pop(0)
        if not lines:
            continue

        # Join PDF line wraps inside the same paragraph into a single clean line
        merged = " ".join(lines).strip()

        # Skip standalone non-sentence blocks (banners, titles, author bylines, headings)
        if not re.search(r"[.!?][\"')\]]*$", merged):
            continue

        # Temporarily shield dots inside common academic abbreviations
        protected = merged
        for abbr in _ABBREVIATIONS:
            safe_token = abbr.replace(".", "<DOT>")
            protected = re.sub(re.escape(abbr), safe_token, protected, flags=re.IGNORECASE)

        # Temporarily shield decimal numbers (e.g., 95.4%)
        protected = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", protected)

        # Split after sentence-ending punctuation followed by whitespace and a likely new sentence
        raw_parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", protected)
        for part in raw_parts:
            restored = part.replace("<DOT>", ".").strip()
            if restored:
                sentences.append(restored)

    return sentences


# Inputs: A single sentence string and a boolean indicating whether an inline citation was found.
# Outputs: True if the sentence is a factual claim, statistic, or conclusion needing verification.
def is_verifiable_claim(sentence: str, has_citation: bool) -> bool:
    cleaned = sentence.strip()
    words = cleaned.split()

    # Ignore very short fragments, section headings, or questions
    if len(cleaned) < 25 or len(words) < 5 or cleaned.endswith("?"):
        return False

    # Any full sentence carrying an inline citation is always worth verifying
    if has_citation:
        return True

    # Otherwise keep sentences that state quantitative findings or empirical conclusions
    return bool(
        _QUANTITATIVE_PATTERN.search(cleaned)
        or _CLAIM_SIGNAL_PATTERN.search(cleaned)
    )


# Inputs: Raw text from a pasted academic document or extracted PDF.
# Outputs: Ordered list of ExtractedClaim objects mapped to inline markers and matched references.
def extract_claims_from_text(raw_text: str) -> list[ExtractedClaim]:
    cleaned_text = clean_extracted_text(raw_text)
    if not cleaned_text:
        return []

    body_text, references_text = split_body_and_references(cleaned_text)
    reference_map = parse_reference_list(references_text)
    sentences = split_into_sentences(body_text)

    extracted_claims: list[ExtractedClaim] = []
    claim_counter = 1

    for sentence in sentences:
        markers = detect_inline_citations(sentence)
        if not is_verifiable_claim(sentence, has_citation=bool(markers)):
            continue

        inline_marker = "; ".join(markers) if markers else None
        matched_refs: list[str] = []
        for marker in markers:
            matched = match_marker_to_reference(marker, reference_map)
            if matched and matched not in matched_refs:
                matched_refs.append(matched)

        raw_reference = " | ".join(matched_refs) if matched_refs else None

        extracted_claims.append(
            ExtractedClaim(
                claim_id=claim_counter,
                claim_text=sentence,
                inline_marker=inline_marker,
                raw_reference=raw_reference,
            )
        )
        claim_counter += 1

    return extracted_claims


# Inputs: pdf_source as raw PDF bytes (from FastAPI UploadFile) or a local file path.
# Outputs: Ordered list of ExtractedClaim objects extracted and mapped from the PDF document.
def extract_claims_from_pdf(pdf_source: bytes | str | Path) -> list[ExtractedClaim]:
    raw_text = extract_text_from_pdf(pdf_source)
    return extract_claims_from_text(raw_text)