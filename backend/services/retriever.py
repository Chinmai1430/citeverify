import re
from typing import Dict, List, Optional

try:
    from backend.schemas import ExtractedClaim, RetrievedSource
except ImportError:
    from schemas import ExtractedClaim, RetrievedSource

OPENALEX_API_URL = "https://api.openalex.org/works"
CROSSREF_API_URL = "https://api.crossref.org/works"
ARXIV_API_URL = "https://export.arxiv.org/api/query"
DEFAULT_TIMEOUT = 10.0
REQUEST_HEADERS = {
    "User-Agent": "CiteVerify/1.0 (mailto:citeverify@sapthahacks.edu)"
}


# Input: raw citation string (or None) extracted from the uploaded paper.
# Output: normalized DOI string (e.g., "10.1038/...") or None if absent.
def extract_doi(raw_reference: Optional[str]) -> Optional[str]:
    if not raw_reference:
        return None
    match = re.search(r"(10\.\d{4,9}/[^\s\"'<>]+)", raw_reference, re.IGNORECASE)
    if not match:
        return None
    return match.group(1).rstrip(".,;)]}")


# Input: raw citation string (or None) that may contain an arXiv identifier or URL.
# Output: cleaned arXiv ID (e.g., "1706.03762") or None if not found.
def extract_arxiv_id(raw_reference: Optional[str]) -> Optional[str]:
    if not raw_reference:
        return None
    pattern = r"(?:arXiv[:\s]*|arxiv\.org/(?:abs|pdf)/)(\d{4}\.\d{4,5}(?:v\d+)?)"
    match = re.search(pattern, raw_reference, re.IGNORECASE)
    return match.group(1) if match else None


# Input: raw citation string (or None) containing a publication year.
# Output: four-digit integer year (1900-2099) or None if no year is detected.
def extract_year(raw_reference: Optional[str]) -> Optional[int]:
    if not raw_reference:
        return None
    match = re.search(r"\b(19\d{2}|20\d{2})\b", raw_reference)
    return int(match.group(1)) if match else None


# Input: raw citation text and optional fallback claim text when reference is empty.
# Output: whitespace-cleaned search query suitable for title/author API lookups.
def clean_search_query(raw_reference: Optional[str], fallback_text: str = "") -> str:
    text = raw_reference or fallback_text
    if not text:
        return ""
    cleaned = re.sub(r"https?://\S+", " ", text)
    cleaned = re.sub(r"doi:\s*\S+", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"arXiv:\s*\S+", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\[\d+\]\s*", "", cleaned.strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:250]


# Input: OpenAlex abstract_inverted_index mapping each word to its token positions.
# Output: reconstructed plain-text abstract string, or None if unavailable.
def reconstruct_openalex_abstract(
    abstract_inverted_index: Optional[Dict[str, List[int]]]
) -> Optional[str]:
    if not abstract_inverted_index:
        return None
    max_pos = max(
        (pos for positions in abstract_inverted_index.values() for pos in positions),
        default=-1,
    )
    if max_pos < 0:
        return None
    words = [""] * (max_pos + 1)
    for word, positions in abstract_inverted_index.items():
        for pos in positions:
            if 0 <= pos <= max_pos:
                words[pos] = word
    abstract = " ".join(w for w in words if w).strip()
    return abstract or None