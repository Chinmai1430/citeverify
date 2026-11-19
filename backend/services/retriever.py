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

STOP_WORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "their",
    "have", "been", "were", "also", "using", "based", "study", "paper",
    "results", "show", "which", "about", "more", "than", "between",
}


# Input: any text string (or None) from a claim, title, or reference.
# Output: set of lowercase meaningful keywords (4+ letters, excluding stop words).
def extract_meaningful_tokens(text: Optional[str]) -> set:
    if not text:
        return set()
    words = re.findall(r"[a-z]{4,}", text.lower())
    return {w for w in words if w not in STOP_WORDS}


# Input: primary paper dict and an optional secondary paper dict from another API.
# Output: merged dict filling missing fields and combining retraction/open-access flags.
def merge_paper_records(
    primary: Dict[str, Any], secondary: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    if not secondary:
        return primary
    merged = dict(primary)
    for field in ("title", "year", "doi", "url", "abstract"):
        if not merged.get(field) and secondary.get(field):
            merged[field] = secondary[field]
    if not merged.get("authors") and secondary.get("authors"):
        merged["authors"] = secondary["authors"]
    merged["is_retracted"] = bool(
        merged.get("is_retracted") or secondary.get("is_retracted")
    )
    merged["is_open_access"] = bool(
        merged.get("is_open_access") or secondary.get("is_open_access")
    )
    return merged


# Input: original ExtractedClaim, retrieved paper dict, cited DOI, and cited year.
# Output: list of clear warning flags for mismatches, retractions, or paywall limits.
def detect_reference_mismatches(
    claim: ExtractedClaim,
    paper: Dict[str, Any],
    cited_doi: Optional[str],
    cited_year: Optional[int],
) -> List[str]:
    flags: List[str] = []

    if paper.get("is_retracted"):
        flags.append("RETRACTED_SOURCE: original publication is retracted or withdrawn")

    paper_doi = paper.get("doi")
    if cited_doi and paper_doi and cited_doi.lower() != paper_doi.lower():
        flags.append("DOI_MISMATCH: cited DOI does not match retrieved publication")

    paper_year = paper.get("year")
    if cited_year and paper_year and cited_year != paper_year:
        flags.append(
            f"YEAR_MISMATCH: cited {cited_year}, original published in {paper_year}"
        )

    if claim.inline_marker and paper.get("authors"):
        marker_words = {
            w.lower()
            for w in re.findall(r"[A-Za-z]{3,}", claim.inline_marker)
            if w.lower() not in {"and", "see", "ref", "fig", "tab"}
        }
        author_surnames = {
            re.sub(r"[^a-z]", "", name.split()[-1].lower())
            for name in paper["authors"]
            if name.strip()
        }
        if marker_words and author_surnames and not (marker_words & author_surnames):
            flags.append(
                "AUTHOR_MISMATCH: inline citation author not found in source authors"
            )

    clean_ref = clean_search_query(claim.raw_reference)
    ref_tokens = extract_meaningful_tokens(clean_ref)
    title_tokens = extract_meaningful_tokens(paper.get("title"))
    if len(ref_tokens) >= 3 and len(title_tokens) >= 2:
        overlap_ratio = len(title_tokens & ref_tokens) / len(title_tokens)
        if overlap_ratio < 0.35:
            flags.append(
                "TITLE_MISMATCH: cited reference text does not match retrieved title"
            )

    claim_tokens = extract_meaningful_tokens(claim.claim_text)
    source_text = f"{paper.get('title') or ''} {paper.get('abstract') or ''}"
    source_tokens = extract_meaningful_tokens(source_text)
    if len(claim_tokens) >= 3 and len(source_tokens) >= 3:
        if not (claim_tokens & source_tokens):
            flags.append(
                "POTENTIALLY_UNRELATED_SOURCE: claim terms absent from source title/abstract"
            )

    if paper.get("abstract"):
        if not paper.get("is_open_access"):
            flags.append("PAYWALLED_ABSTRACT_ONLY_NEEDS_HUMAN_REVIEW")
    else:
        flags.append("METADATA_ONLY_NO_ABSTRACT_NEEDS_HUMAN_REVIEW")

    return flags


# Input: ExtractedClaim containing claim text, inline citation, and raw reference.
# Output: RetrievedSource schema with metadata, abstract, scope, and mismatch flags.
def retrieve_source(claim: ExtractedClaim) -> RetrievedSource:
    if not claim.raw_reference or not claim.raw_reference.strip():
        return RetrievedSource(
            found=False,
            evidence_scope="NONE",
            mismatch_flags=["MISSING_REFERENCE: no citation provided for this claim"],
        )

    cited_doi = extract_doi(claim.raw_reference)
    cited_arxiv_id = extract_arxiv_id(claim.raw_reference)
    cited_year = extract_year(claim.raw_reference)
    search_query = clean_search_query(claim.raw_reference, fallback_text=claim.claim_text)

    paper: Optional[Dict[str, Any]] = None

    with httpx.Client(
        headers=REQUEST_HEADERS, timeout=DEFAULT_TIMEOUT, follow_redirects=True
    ) as client:
        if cited_arxiv_id:
            paper = fetch_from_arxiv(client, arxiv_id=cited_arxiv_id)

        if not paper and cited_doi:
            paper = fetch_from_openalex(client, doi=cited_doi)
            crossref_hit = fetch_from_crossref(client, doi=cited_doi)
            if paper and crossref_hit:
                paper = merge_paper_records(paper, crossref_hit)
            elif not paper:
                paper = crossref_hit
            if not paper or not paper.get("abstract"):
                epmc_hit = fetch_from_europe_pmc(client, doi=cited_doi)
                paper = merge_paper_records(paper, epmc_hit) if paper else epmc_hit

        if not paper and search_query:
            for fetcher in (
                fetch_from_openalex,
                fetch_from_crossref,
                fetch_from_arxiv,
                fetch_from_europe_pmc,
            ):
                paper = fetcher(client, query=search_query)
                if paper:
                    break

        if paper and not paper.get("abstract") and paper.get("doi"):
            enrichment = fetch_from_europe_pmc(client, doi=paper["doi"])
            paper = merge_paper_records(paper, enrichment)

    if not paper:
        return RetrievedSource(
            found=False,
            evidence_scope="NONE",
            mismatch_flags=["UNVERIFIABLE_REFERENCE: source not found in scholarly APIs"],
        )

    evidence_scope = "ABSTRACT_ONLY" if paper.get("abstract") else "METADATA_ONLY"
    mismatch_flags = detect_reference_mismatches(claim, paper, cited_doi, cited_year)

    return RetrievedSource(
        found=True,
        title=paper.get("title"),
        authors=paper.get("authors") or [],
        year=paper.get("year"),
        doi_url=paper.get("url"),
        is_retracted=bool(paper.get("is_retracted", False)),
        source_passage=paper.get("abstract"),
        evidence_scope=evidence_scope,
        mismatch_flags=mismatch_flags,
    )


# Input: list of ExtractedClaim objects produced by the document parser.
# Output: list of corresponding RetrievedSource objects in the same order.
def retrieve_sources_batch(claims: List[ExtractedClaim]) -> List[RetrievedSource]:
    return [retrieve_source(claim) for claim in claims]