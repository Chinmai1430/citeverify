import re

# Common non-author nouns that can precede a year in titles/headers (e.g., "Review (2026)")
_NON_AUTHOR_WORDS = {
    "abstract",
    "analysis",
    "annual",
    "appendix",
    "audit",
    "benchmark",
    "bulletin",
    "chapter",
    "conference",
    "corpus",
    "data",
    "dataset",
    "edition",
    "figure",
    "guideline",
    "guidelines",
    "handbook",
    "issue",
    "journal",
    "manual",
    "paper",
    "protocol",
    "release",
    "report",
    "review",
    "section",
    "series",
    "standard",
    "study",
    "summary",
    "survey",
    "symposium",
    "system",
    "table",
    "trial",
    "update",
    "version",
    "volume",
    "workshop",
    "year",
}


# Inputs: A single reference entry string (e.g., "[1] Smith, J. (2024). Title...").
# Outputs: Optional normalized lookup key in "surname_year" format (e.g., "smith_2024").
def _extract_author_year_key(ref_entry: str) -> str | None:
    cleaned = re.sub(r"^\s*(?:\[\d+\]|\d+[\.\)])\s*", "", ref_entry).strip()
    year_match = re.search(r"\b((?:19|20)\d{2}[a-z]?)\b", cleaned)
    if not year_match:
        return None

    # Prefer the surname before the first comma (APA/Harvard style: "Smith, J.")
    surname_match = re.match(r"^([A-Z][A-Za-z'\-]+)\s*,", cleaned)
    if not surname_match:
        # Fallback for "J. Smith" or "John Smith" styles before the year
        before_year = cleaned[: year_match.start()]
        tokens = re.findall(r"[A-Z][A-Za-z'\-]{1,}", before_year)
        if not tokens:
            return None
        surname = tokens[0]
    else:
        surname = surname_match.group(1)

    return f"{surname.lower()}_{year_match.group(1).lower()}"


# Inputs: Raw text of the document's References / Bibliography section.
# Outputs: Dictionary mapping citation keys (e.g., "1" or "smith_2024") to raw reference strings.
def parse_reference_list(references_text: str) -> dict[str, str]:
    if not references_text or not references_text.strip():
        return {}

    lines = [line.strip() for line in references_text.splitlines() if line.strip()]
    numbered_start = re.compile(r"^(?:\[(\d+)\]|(\d+)[\.\)])\s+(.+)$")

    entries: list[tuple[str | None, str]] = []
    has_numbered_refs = any(numbered_start.match(line) for line in lines)

    if has_numbered_refs:
        current_num: str | None = None
        current_parts: list[str] = []
        for line in lines:
            match = numbered_start.match(line)
            if match:
                if current_parts:
                    entries.append((current_num, " ".join(current_parts).strip()))
                current_num = match.group(1) or match.group(2)
                current_parts = [line]
            elif current_parts:
                current_parts.append(line)
        if current_parts:
            entries.append((current_num, " ".join(current_parts).strip()))
    else:
        # Unnumbered references (APA/Harvard): split on blank lines or new "Surname, ..." lines
        author_start = re.compile(r"^[A-Z][A-Za-z'\-]+,\s+[A-Z]")
        raw_blocks = re.split(r"\n\s*\n", references_text.strip())
        for block in raw_blocks:
            block_lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
            current_parts = []
            for ln in block_lines:
                if author_start.match(ln) and current_parts:
                    entries.append((None, " ".join(current_parts).strip()))
                    current_parts = [ln]
                else:
                    current_parts.append(ln)
            if current_parts:
                entries.append((None, " ".join(current_parts).strip()))

    reference_map: dict[str, str] = {}
    for ref_num, full_ref in entries:
        if ref_num:
            reference_map[ref_num] = full_ref
        author_year_key = _extract_author_year_key(full_ref)
        if author_year_key and author_year_key not in reference_map:
            reference_map[author_year_key] = full_ref

    return reference_map


# Inputs: A single sentence or claim string from the document body.
# Outputs: Ordered, deduplicated list of inline citation markers found in the sentence.
def detect_inline_citations(sentence: str) -> list[str]:
    if not sentence:
        return []

    patterns = [
        # Numeric bracket markers: [1], [1, 2], [1-3], [1, 3-5]
        r"\[\s*\d+(?:\s*[-–]\s*\d+)?(?:\s*,\s*\d+(?:\s*[-–]\s*\d+)?)*\s*\]",
        # Parenthetical author-year markers: (Smith, 2024), (Smith et al., 2020; Doe, 2021)
        r"\([^()]*\b(?:19|20)\d{2}[a-z]?\b[^()]*\)",
        # Narrative author-year markers: Smith (2024), Smith et al. (2020), Smith and Jones (2021)
        r"\b[A-Z][A-Za-z'\-]+(?:\s+(?:et\s+al\.|(?:and|&)\s+[A-Z][A-Za-z'\-]+))?\s*\(\s*(?:19|20)\d{2}[a-z]?\s*\)",
    ]

    found: list[tuple[int, str]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, sentence):
            marker = match.group(0).strip()
            # Skip parenthetical matches that are strictly a year with no author inside
            if marker.startswith("(") and re.fullmatch(
                r"\(\s*(?:19|20)\d{2}[a-z]?\s*\)", marker
            ):
                continue
            # Skip false-positive narrative matches like "Review (2026)" or "Report (2024)"
            if not marker.startswith(("[", "(")):
                leading_word = marker.split()[0].lower()
                if leading_word in _NON_AUTHOR_WORDS:
                    continue
            found.append((match.start(), marker))

    # Sort by position in sentence and deduplicate while preserving order
    found.sort(key=lambda item: item[0])
    unique_markers: list[str] = []
    for _, marker in found:
        if marker not in unique_markers:
            unique_markers.append(marker)

    return unique_markers


# Inputs: An inline citation marker string and the parsed reference_map dictionary.
# Outputs: Matched raw reference string (joined with " | " if multiple), or None if unmatched.
def match_marker_to_reference(
    inline_marker: str, reference_map: dict[str, str]
) -> str | None:
    if not inline_marker or not reference_map:
        return None

    matched_refs: list[str] = []

    # Case 1: Numeric bracket citation like [1], [1, 3], or [2-4]
    if inline_marker.startswith("[") and inline_marker.endswith("]"):
        inner = inline_marker[1:-1]
        for part in inner.split(","):
            part = part.strip()
            range_match = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)", part)
            if range_match:
                start_num, end_num = int(range_match.group(1)), int(range_match.group(2))
                for num in range(start_num, min(end_num, start_num + 20) + 1):
                    ref = reference_map.get(str(num))
                    if ref and ref not in matched_refs:
                        matched_refs.append(ref)
            elif part.isdigit():
                ref = reference_map.get(part)
                if ref and ref not in matched_refs:
                    matched_refs.append(ref)

        return " | ".join(matched_refs) if matched_refs else None

    # Case 2: Author-Year citation like (Smith et al., 2020) or Smith (2024)
    segments = re.split(r";", inline_marker)
    for segment in segments:
        year_match = re.search(r"\b((?:19|20)\d{2}[a-z]?)\b", segment)
        if not year_match:
            continue
        year = year_match.group(1).lower()

        # Extract candidate author surname(s) before the year
        before_year = segment[: year_match.start()]
        before_year = re.sub(
            r"(?i)\b(see|e\.g\.|cf\.|also|et\s+al\.?|and)\b", " ", before_year
        )
        author_tokens = re.findall(r"[A-Z][A-Za-z'\-]+", before_year)
        if not author_tokens:
            continue

        surname = author_tokens[0].lower()
        exact_key = f"{surname}_{year}"
        ref = reference_map.get(exact_key)

        # Fallback: scan reference values for both the surname and the year
        if not ref:
            for candidate_ref in set(reference_map.values()):
                if surname in candidate_ref.lower() and year in candidate_ref.lower():
                    ref = candidate_ref
                    break

        if ref and ref not in matched_refs:
            matched_refs.append(ref)

    return " | ".join(matched_refs) if matched_refs else None