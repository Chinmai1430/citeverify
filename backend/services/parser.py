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

# Standalone section heading lines (numeric, Roman numeral IEEE, or standard academic)
_SECTION_HEADING_PATTERN = re.compile(
    r"^(?:"
    r"(?:\d+(?:\.\d+)*|[IVX]+)\.?\s+[A-Z][^\n.!?]{1,90}"
    r"|(?:abstract|introduction|background|methods?|methodology|"
    r"materials?\s+and\s+methods?|results?|discussion|conclusions?|"
    r"limitations?|acknowledgements?)\s*:?"
    r")$",
    re.IGNORECASE,
)

# Standalone page numbers or footer markers common in copy-pasted PDFs
_PAGE_NOISE_PATTERN = re.compile(
    r"(?im)^\s*(?:page\s+\d+(?:\s+of\s+\d+)?|[-–—]?\s*\d{1,3}\s*[-–—]?)\s*$"
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

_LIGATURE_MAP = str.maketrans(
    {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\u00a0": " ",
        "\u200b": "",
        "\ufeff": "",
        "\u00ad": "",
    }
)


# Inputs: Raw PyMuPDF text blocks from a single page and the page width in points.
# Outputs: Text blocks sorted in true reading order (supports both 1-column and 2-column IEEE/arXiv pages).
def _sort_page_blocks_reading_order(
    blocks: list[tuple], page_width: float
) -> list[tuple]:
    text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]
    if len(text_blocks) <= 1 or page_width <= 0:
        return text_blocks

    mid_x = page_width / 2.0
    margin = 12.0

    full_width: list[tuple] = []
    narrow: list[tuple] = []
    for block in text_blocks:
        x0, x1 = block[0], block[2]
        spans_mid = x0 < (mid_x - margin) and x1 > (mid_x + margin)
        is_wide = (x1 - x0) > (page_width * 0.58)
        if spans_mid or is_wide:
            full_width.append(block)
        else:
            narrow.append(block)

    # Only apply 2-column ordering when narrow blocks populate both left and right halves
    has_left = any(((b[0] + b[2]) / 2.0) < mid_x for b in narrow)
    has_right = any(((b[0] + b[2]) / 2.0) >= mid_x for b in narrow)
    if not (has_left and has_right):
        return sorted(text_blocks, key=lambda b: (round(b[1], 1), round(b[0], 1)))

    full_width.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))
    boundaries = [float("-inf")] + [b[1] for b in full_width] + [float("inf")]

    ordered: list[tuple] = []
    for idx in range(len(boundaries) - 1):
        y_start, y_end = boundaries[idx], boundaries[idx + 1]
        band = [b for b in narrow if y_start <= ((b[1] + b[3]) / 2.0) < y_end]

        left_col = sorted(
            (b for b in band if ((b[0] + b[2]) / 2.0) < mid_x),
            key=lambda b: (round(b[1], 1), round(b[0], 1)),
        )
        right_col = sorted(
            (b for b in band if ((b[0] + b[2]) / 2.0) >= mid_x),
            key=lambda b: (round(b[1], 1), round(b[0], 1)),
        )
        ordered.extend(left_col)
        ordered.extend(right_col)

        if idx < len(full_width):
            ordered.append(full_width[idx])

    return ordered


# Inputs: pdf_source as raw PDF bytes (from FastAPI UploadFile) or a local file path.
# Outputs: Full plain-text string extracted in reading order with preserved paragraph boundaries.
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
            blocks = _sort_page_blocks_reading_order(
                page.get_text("blocks"), page.rect.width
            )
            if blocks:
                page_text = "\n\n".join(b[4].strip() for b in blocks)
                pages_text.append(page_text)

    combined_text = "\n\n".join(pages_text).strip()
    if not combined_text:
        raise ValueError("No readable text could be extracted from the PDF.")
    return combined_text


# Inputs: raw_text string directly from a PDF or user-pasted academic document.
# Outputs: Cleaned string with ligatures fixed, page noise removed, and hyphenated wraps rejoined.
def clean_extracted_text(raw_text: str) -> str:
    if not raw_text:
        return ""

    text = raw_text.translate(_LIGATURE_MAP)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _PAGE_NOISE_PATTERN.sub("", text)
    text = re.sub(r"([A-Za-z])-\s*\n\s*([A-Za-z])", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# Inputs: Cleaned full document text containing main content and an optional references section.
# Outputs: Tuple of (body_text, references_text), where references_text is empty if none is found.
def split_body_and_references(text: str) -> tuple[str, str]:
    if not text:
        return "", ""

    heading_pattern = re.compile(
        r"(?im)^\s*(?:(?:\d+|[IVX]+)\.?\s+)?(references|bibliography|works\s+cited|literature\s+cited)\s*:?\s*$"
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
# Outputs: List of cleaned sentences with headings stripped and cross-column splits rejoined.
def split_into_sentences(body_text: str) -> list[str]:
    if not body_text or not body_text.strip():
        return []

    raw_paragraphs = re.split(r"\n\s*\n", body_text.strip())
    merged_paragraphs: list[str] = []

    for paragraph in raw_paragraphs:
        lines = [ln.strip() for ln in paragraph.splitlines() if ln.strip()]
        while lines and _SECTION_HEADING_PATTERN.fullmatch(lines[0]):
            lines.pop(0)
        if not lines:
            continue

        merged = " ".join(lines).strip()

        # Heal sentences split across 2-column or page breaks (previous block unfinished, next continues)
        if (
            merged_paragraphs
            and not re.search(r"[.!?][\"')\]]*$", merged_paragraphs[-1])
            and re.match(r"^[a-z(\[]", merged)
        ):
            prev = merged_paragraphs[-1]
            if prev.endswith("-") and re.match(r"^[a-z]", merged):
                merged_paragraphs[-1] = prev[:-1] + merged
            else:
                merged_paragraphs[-1] = f"{prev} {merged}"
        else:
            merged_paragraphs.append(merged)

    sentences: list[str] = []
    for merged in merged_paragraphs:
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