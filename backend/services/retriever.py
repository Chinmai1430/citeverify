import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

import httpx

try:
    from backend.schemas import ExtractedClaim, RetrievedSource
except ImportError:
    from schemas import ExtractedClaim, RetrievedSource

OPENALEX_API_URL = "https://api.openalex.org/works"
CROSSREF_API_URL = "https://api.crossref.org/works"
ARXIV_API_URL = "https://export.arxiv.org/api/query"
EUROPE_PMC_API_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
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


# Input: text string (or None) that may contain XML/HTML tags from scholarly APIs.
# Output: clean plain-text string with tags removed and whitespace normalized.
def strip_xml_tags(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    cleaned = re.sub(r"<[^>]+>", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or None


# Input: active httpx client, optional DOI, and optional title/author search query.
# Output: normalized paper metadata dictionary from OpenAlex, or None if not found.
def fetch_from_openalex(
    client: httpx.Client,
    doi: Optional[str] = None,
    query: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    try:
        work = None
        if doi:
            resp = client.get(f"{OPENALEX_API_URL}/https://doi.org/{doi}")
            if resp.status_code == 200:
                work = resp.json()
        if not work and query:
            resp = client.get(
                OPENALEX_API_URL,
                params={"search": query, "per-page": 1},
            )
            if resp.status_code == 200:
                results = resp.json().get("results") or []
                work = results[0] if results else None
        if not work:
            return None

        authors = [
            a["author"]["display_name"]
            for a in work.get("authorships", [])
            if a.get("author") and a["author"].get("display_name")
        ]
        oa_info = work.get("open_access") or {}
        raw_doi = work.get("doi")
        return {
            "provider": "openalex",
            "title": work.get("display_name") or work.get("title"),
            "authors": authors,
            "year": work.get("publication_year"),
            "doi": raw_doi.replace("https://doi.org/", "") if raw_doi else None,
            "url": raw_doi or oa_info.get("oa_url") or work.get("id"),
            "abstract": reconstruct_openalex_abstract(
                work.get("abstract_inverted_index")
            ),
            "is_retracted": bool(work.get("is_retracted", False)),
            "is_open_access": bool(oa_info.get("is_oa", False)),
        }
    except (httpx.HTTPError, ValueError):
        return None


# Input: active httpx client, optional DOI, and optional title/author search query.
# Output: normalized paper metadata dictionary from CrossRef, or None if not found.
def fetch_from_crossref(
    client: httpx.Client,
    doi: Optional[str] = None,
    query: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    try:
        item = None
        if doi:
            resp = client.get(f"{CROSSREF_API_URL}/{doi}")
            if resp.status_code == 200:
                item = resp.json().get("message")
        if not item and query:
            resp = client.get(
                CROSSREF_API_URL,
                params={"query.bibliographic": query, "rows": 1},
            )
            if resp.status_code == 200:
                items = resp.json().get("message", {}).get("items") or []
                item = items[0] if items else None
        if not item:
            return None

        titles = item.get("title") or []
        title = titles[0] if titles else None
        authors = []
        for author in item.get("author", []):
            full_name = f"{author.get('given', '')} {author.get('family', '')}".strip()
            if full_name:
                authors.append(full_name)

        year = None
        for date_key in ("published", "published-print", "published-online", "issued"):
            parts = item.get(date_key, {}).get("date-parts")
            if parts and parts[0] and parts[0][0]:
                year = int(parts[0][0])
                break

        updates = (item.get("update-to") or []) + (item.get("updated-by") or [])
        is_retracted = any(
            "retract" in str(u.get("type", "")).lower()
            or "retract" in str(u.get("label", "")).lower()
            for u in updates
        )
        if title and "retracted" in title.lower():
            is_retracted = True

        item_doi = item.get("DOI")
        return {
            "provider": "crossref",
            "title": title,
            "authors": authors,
            "year": year,
            "doi": item_doi,
            "url": f"https://doi.org/{item_doi}" if item_doi else item.get("URL"),
            "abstract": strip_xml_tags(item.get("abstract")),
            "is_retracted": is_retracted,
            "is_open_access": False,
        }
    except (httpx.HTTPError, ValueError):
        return None


# Input: active httpx client, optional arXiv ID, and optional title/author query.
# Output: normalized preprint metadata dictionary from arXiv, or None if not found.
def fetch_from_arxiv(
    client: httpx.Client,
    arxiv_id: Optional[str] = None,
    query: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    try:
        if arxiv_id:
            params = {"id_list": arxiv_id, "max_results": 1}
        elif query:
            params = {"search_query": f"all:{query}", "max_results": 1}
        else:
            return None

        resp = client.get(ARXIV_API_URL, params=params)
        if resp.status_code != 200:
            return None

        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }
        root = ET.fromstring(resp.text)
        entry = root.find("atom:entry", ns)
        if entry is None:
            return None

        title_el = entry.find("atom:title", ns)
        title = re.sub(r"\s+", " ", title_el.text).strip() if title_el is not None and title_el.text else None
        if not title or title.lower() == "error":
            return None

        summary_el = entry.find("atom:summary", ns)
        abstract = (
            re.sub(r"\s+", " ", summary_el.text).strip()
            if summary_el is not None and summary_el.text
            else None
        )
        authors = [
            name_el.text.strip()
            for name_el in entry.findall("atom:author/atom:name", ns)
            if name_el.text
        ]
        pub_el = entry.find("atom:published", ns)
        year = int(pub_el.text[:4]) if pub_el is not None and pub_el.text and pub_el.text[:4].isdigit() else None
        id_el = entry.find("atom:id", ns)
        doi_el = entry.find("arxiv:doi", ns)
        doi_val = doi_el.text.strip() if doi_el is not None and doi_el.text else None

        return {
            "provider": "arxiv",
            "title": title,
            "authors": authors,
            "year": year,
            "doi": doi_val,
            "url": id_el.text.strip() if id_el is not None and id_el.text else None,
            "abstract": abstract,
            "is_retracted": "withdrawn" in (abstract or "").lower(),
            "is_open_access": True,
        }
    except (httpx.HTTPError, ET.ParseError, ValueError):
        return None


# Input: active httpx client, optional DOI, and optional title/author search query.
# Output: normalized paper metadata from Europe PMC, or None if not found.
def fetch_from_europe_pmc(
    client: httpx.Client,
    doi: Optional[str] = None,
    query: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    try:
        search_expr = f'DOI:"{doi}"' if doi else query
        if not search_expr:
            return None

        resp = client.get(
            EUROPE_PMC_API_URL,
            params={"query": search_expr, "format": "json", "resultType": "core", "pageSize": 1},
        )
        if resp.status_code != 200:
            return None

        results = resp.json().get("resultList", {}).get("result") or []
        if not results:
            return None
        item = results[0]

        author_entries = item.get("authorList", {}).get("author") or []
        authors = [a["fullName"] for a in author_entries if a.get("fullName")]
        pub_year = item.get("pubYear")
        year = int(pub_year) if pub_year and str(pub_year).isdigit() else None
        pub_types = [t.lower() for t in item.get("pubTypeList", {}).get("pubType", [])]
        is_retracted = any("retract" in t for t in pub_types)
        item_doi = item.get("doi")

        return {
            "provider": "europe_pmc",
            "title": strip_xml_tags(item.get("title")),
            "authors": authors,
            "year": year,
            "doi": item_doi,
            "url": f"https://doi.org/{item_doi}" if item_doi else None,
            "abstract": strip_xml_tags(item.get("abstractText")),
            "is_retracted": is_retracted,
            "is_open_access": item.get("isOpenAccess") == "Y",
        }
    except (httpx.HTTPError, ValueError):
        return None