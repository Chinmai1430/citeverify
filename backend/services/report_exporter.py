import re
from datetime import UTC, datetime
from pathlib import Path

import pymupdf

# Support running from repo root (backend.schemas) or inside backend/ (schemas)
try:
    from backend.schemas import ClaimStatus, VerificationReport
except ImportError:
    from schemas import ClaimStatus, VerificationReport


_STATUS_ORDER = (
    ClaimStatus.SUPPORTED,
    ClaimStatus.PARTIALLY_SUPPORTED,
    ClaimStatus.CONTRADICTORY,
    ClaimStatus.UNSUPPORTED,
    ClaimStatus.POTENTIALLY_OUTDATED,
)

_STATUS_LABELS: dict[str, str] = {
    ClaimStatus.SUPPORTED.value: "Supported",
    ClaimStatus.PARTIALLY_SUPPORTED.value: "Partially Supported",
    ClaimStatus.CONTRADICTORY.value: "Contradictory",
    ClaimStatus.UNSUPPORTED.value: "Unsupported / Missing",
    ClaimStatus.POTENTIALLY_OUTDATED.value: "Potentially Outdated / Retracted",
}

_STATUS_COLORS: dict[str, tuple[float, float, float]] = {
    ClaimStatus.SUPPORTED.value: (0.09, 0.48, 0.24),
    ClaimStatus.PARTIALLY_SUPPORTED.value: (0.72, 0.45, 0.04),
    ClaimStatus.CONTRADICTORY.value: (0.74, 0.12, 0.12),
    ClaimStatus.UNSUPPORTED.value: (0.35, 0.39, 0.45),
    ClaimStatus.POTENTIALLY_OUTDATED.value: (0.55, 0.16, 0.52),
}

_PDF_CHAR_REPLACEMENTS = str.maketrans(
    {
        "\u2014": "--",
        "\u2013": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": "-",
        "\u2026": "...",
        "\u00a0": " ",
    }
)


# Inputs: Arbitrary text string that may include smart quotes or non-Latin-1 Unicode characters.
# Outputs: Latin-1 safe string suitable for PyMuPDF built-in PDF fonts.
def _sanitize_pdf_text(text: str) -> str:
    normalized = text.translate(_PDF_CHAR_REPLACEMENTS)
    return normalized.encode("latin-1", errors="replace").decode("latin-1")


# Inputs: VerificationReport instance.
# Outputs: Normalized dictionary mapping every ClaimStatus value string to its integer count.
def _normalized_summary_counts(report: VerificationReport) -> dict[str, int]:
    counts: dict[str, int] = {status.value: 0 for status in _STATUS_ORDER}
    if report.summary_counts:
        for key, value in report.summary_counts.items():
            counts[str(key)] = int(value)
    else:
        for result in report.results:
            key = result.verdict.value
            counts[key] = counts.get(key, 0) + 1
    return counts


# Inputs: Document title and desired file extension (e.g., "md" or "pdf").
# Outputs: Safe lowercase filename string for HTTP Content-Disposition download headers.
def build_export_filename(document_title: str, extension: str) -> str:
    ext = extension.lstrip(".").lower() or "md"
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", (document_title or "").strip()).strip("_").lower()
    slug = slug[:60] if slug else "untitled_document"
    return f"citeverify_audit_{slug}.{ext}"


# Inputs: VerificationReport schema object and optional UTC timestamp override.
# Outputs: Complete, well-structured Markdown audit report string.
def export_report_to_markdown(
    report: VerificationReport,
    generated_at: datetime | None = None,
) -> str:
    timestamp = (generated_at or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M UTC")
    counts = _normalized_summary_counts(report)
    total = report.total_claims or len(report.results)
    review_needed = sum(1 for item in report.results if item.needs_human_review)

    lines: list[str] = [
        "# CiteVerify Citation & Claim Audit Report",
        "",
        f"**Document Title:** {report.document_title or 'Untitled Document'}  ",
        f"**Generated:** {timestamp}  ",
        f"**Total Claims Audited:** {total}  ",
        f"**Flagged for Human Review:** {review_needed} / {total}",
        "",
        "---",
        "",
        "## Executive Verdict Summary",
        "",
        "| Verdict | Count | Share |",
        "| --- | ---: | ---: |",
    ]

    for status in _STATUS_ORDER:
        count = counts.get(status.value, 0)
        share = f"{(count / total * 100):.1f}%" if total > 0 else "0.0%"
        label = _STATUS_LABELS[status.value]
        lines.append(f"| **{label}** (`{status.value}`) | {count} | {share} |")

    lines.extend(["", "---", "", "## Claim-by-Claim Verification Details", ""])

    if not report.results:
        lines.append("_No verifiable claims were found in this document._")
        lines.append("")
    else:
        for item in report.results:
            verdict_val = item.verdict.value
            verdict_label = _STATUS_LABELS.get(verdict_val, verdict_val)
            review_label = "Yes" if item.needs_human_review else "No"

            lines.append(f"### Claim {item.claim_id} — {verdict_label}")
            lines.append("")
            lines.append(
                f"- **Verdict:** `{verdict_val}` | **Confidence:** `{item.confidence.value}` | "
                f"**Needs Human Review:** {review_label}"
            )
            lines.append(f"- **Claim:** {item.claim_text}")
            lines.append(
                f"- **Matched Citation:** {item.matched_citation or 'None matched'}"
            )
            if item.source_url:
                lines.append(f"- **Source URL:** {item.source_url}")
            if item.mismatch_flags:
                flags_formatted = ", ".join(f"`{flag}`" for flag in item.mismatch_flags)
                lines.append(f"- **Mismatch Flags:** {flags_formatted}")
            lines.append(f"- **Explanation:** {item.explanation}")
            lines.append(f"- **Uncertainty Note:** {item.uncertainty_note}")

            if item.evidence_passage:
                lines.append("- **Evidence Passage:**")
                lines.append("")
                for passage_line in item.evidence_passage.strip().splitlines():
                    lines.append(f"  > {passage_line.strip()}")

            lines.append("")

    if report.limitations:
        lines.extend(["---", "", "## Verification Scope & Limitations", ""])
        for note in report.limitations:
            lines.append(f"- {note}")
        lines.append("")

    lines.extend(
        [
            "---",
            "_Generated automatically by CiteVerify. Final scholarly or clinical decisions require human review._",
            "",
        ]
    )
    return "\n".join(lines)


class _PDFReportBuilder:
    """Internal helper managing multi-page flow layout for PyMuPDF audit reports."""

    def __init__(self) -> None:
        self.doc = pymupdf.open()
        self.page_width = 612.0
        self.page_height = 792.0
        self.left_margin = 52.0
        self.right_margin = 560.0
        self.top_margin = 52.0
        self.bottom_limit = 742.0
        self.page = self.doc.new_page(width=self.page_width, height=self.page_height)
        self.y = self.top_margin

    # Inputs: Minimum vertical space needed in points.
    # Outputs: None (creates a fresh page if remaining height is below min_space).
    def ensure_space(self, min_space: float) -> None:
        if self.y + min_space > self.bottom_limit:
            self._new_page()

    def _new_page(self) -> None:
        self.page = self.doc.new_page(width=self.page_width, height=self.page_height)
        self.y = self.top_margin

    # Inputs: Text string, font style, color, indent, and trailing gap.
    # Outputs: None (draws text with automatic pagination and long-block splitting).
    def add_text(
        self,
        text: str,
        fontsize: float = 9.5,
        fontname: str = "helv",
        color: tuple[float, float, float] = (0.12, 0.14, 0.18),
        indent: float = 0.0,
        gap: float = 6.0,
    ) -> None:
        safe_text = _sanitize_pdf_text(text).strip()
        if not safe_text:
            return

        x0 = self.left_margin + indent
        chunks = self._split_oversized_text(safe_text, max_chars=1800)

        for chunk in chunks:
            rect = pymupdf.Rect(x0, self.y, self.right_margin, self.bottom_limit)
            unused = self.page.insert_textbox(
                rect,
                chunk,
                fontsize=fontsize,
                fontname=fontname,
                color=color,
                align=pymupdf.TEXT_ALIGN_LEFT,
            )
            if unused < 0:
                self._new_page()
                rect = pymupdf.Rect(x0, self.y, self.right_margin, self.bottom_limit)
                unused = self.page.insert_textbox(
                    rect,
                    chunk,
                    fontsize=fontsize,
                    fontname=fontname,
                    color=color,
                    align=pymupdf.TEXT_ALIGN_LEFT,
                )
            if unused >= 0:
                self.y = (self.bottom_limit - unused) + gap
            else:
                self.y = self.bottom_limit

    # Inputs: Horizontal rule color and trailing gap.
    # Outputs: None (draws a subtle divider line across the content width).
    def add_divider(
        self,
        color: tuple[float, float, float] = (0.82, 0.85, 0.89),
        gap: float = 10.0,
    ) -> None:
        self.ensure_space(16.0)
        self.page.draw_line(
            pymupdf.Point(self.left_margin, self.y),
            pymupdf.Point(self.right_margin, self.y),
            color=color,
            width=0.75,
        )
        self.y += gap

    # Inputs: Text string and character threshold.
    # Outputs: List of word-boundary chunks safe to render within a single PDF page.
    @staticmethod
    def _split_oversized_text(text: str, max_chars: int) -> list[str]:
        if len(text) <= max_chars:
            return [text]
        words = text.split()
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for word in words:
            if current and current_len + len(word) + 1 > max_chars:
                chunks.append(" ".join(current))
                current = [word]
                current_len = len(word)
            else:
                current.append(word)
                current_len += len(word) + 1
        if current:
            chunks.append(" ".join(current))
        return chunks

    # Inputs: None.
    # Outputs: Finalized multi-page PDF bytes with footers stamped on every page.
    def finalize_bytes(self) -> bytes:
        total_pages = len(self.doc)
        for index, page in enumerate(self.doc, start=1):
            footer_rect = pymupdf.Rect(
                self.left_margin, 756.0, self.right_margin, 776.0
            )
            page.insert_textbox(
                footer_rect,
                f"CiteVerify Citation Audit Report   |   Page {index} of {total_pages}",
                fontsize=8.0,
                fontname="helv",
                color=(0.45, 0.49, 0.55),
                align=pymupdf.TEXT_ALIGN_RIGHT,
            )
        pdf_bytes = self.doc.tobytes()
        self.doc.close()
        return pdf_bytes


# Inputs: VerificationReport schema object and optional UTC timestamp override.
# Outputs: Downloadable multi-page PDF audit report as raw bytes.
def export_report_to_pdf_bytes(
    report: VerificationReport,
    generated_at: datetime | None = None,
) -> bytes:
    timestamp = (generated_at or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M UTC")
    counts = _normalized_summary_counts(report)
    total = report.total_claims or len(report.results)
    review_needed = sum(1 for item in report.results if item.needs_human_review)

    builder = _PDFReportBuilder()

    # Header banner on Page 1
    banner_rect = pymupdf.Rect(0, 0, builder.page_width, 64.0)
    builder.page.draw_rect(banner_rect, color=None, fill=(0.09, 0.16, 0.29))
    builder.page.insert_textbox(
        pymupdf.Rect(builder.left_margin, 18.0, builder.right_margin, 52.0),
        "CiteVerify  |  Citation & Claim Verification Audit Report",
        fontsize=14.0,
        fontname="hebo",
        color=(1.0, 1.0, 1.0),
    )
    builder.y = 80.0

    # Document metadata block
    builder.add_text(
        report.document_title or "Untitled Document",
        fontsize=14.0,
        fontname="hebo",
        color=(0.08, 0.12, 0.20),
        gap=4.0,
    )
    builder.add_text(
        f"Generated: {timestamp}   |   Total Claims: {total}   |   Flagged for Human Review: {review_needed}/{total}",
        fontsize=9.0,
        fontname="heit",
        color=(0.32, 0.36, 0.42),
        gap=10.0,
    )
    builder.add_divider(gap=12.0)

    # Executive Summary section
    builder.add_text(
        "Executive Verdict Summary",
        fontsize=11.5,
        fontname="hebo",
        color=(0.09, 0.16, 0.29),
        gap=6.0,
    )
    for status in _STATUS_ORDER:
        count = counts.get(status.value, 0)
        share = f"{(count / total * 100):.1f}%" if total > 0 else "0.0%"
        label = _STATUS_LABELS[status.value]
        color = _STATUS_COLORS.get(status.value, (0.2, 0.2, 0.2))
        builder.add_text(
            f"- {label} ({status.value}): {count} ({share})",
            fontsize=9.5,
            fontname="hebo" if count > 0 else "helv",
            color=color if count > 0 else (0.4, 0.44, 0.49),
            indent=8.0,
            gap=3.5,
        )

    builder.y += 6.0
    builder.add_divider(gap=12.0)

    # Claim-by-claim section
    builder.add_text(
        "Claim-by-Claim Verification Details",
        fontsize=11.5,
        fontname="hebo",
        color=(0.09, 0.16, 0.29),
        gap=8.0,
    )

    if not report.results:
        builder.add_text("No verifiable claims were found in this document.", fontsize=9.5)
    else:
        for item in report.results:
            builder.ensure_space(125.0)
            verdict_val = item.verdict.value
            verdict_label = _STATUS_LABELS.get(verdict_val, verdict_val)
            verdict_color = _STATUS_COLORS.get(verdict_val, (0.2, 0.2, 0.2))
            review_flag = "Human Review Required" if item.needs_human_review else "Auto-Cleared"

            builder.add_text(
                f"Claim {item.claim_id}  |  {verdict_label.upper()}  ({item.confidence.value} Confidence  -  {review_flag})",
                fontsize=10.0,
                fontname="hebo",
                color=verdict_color,
                gap=4.0,
            )
            builder.add_text(
                f"Claim: {item.claim_text}",
                fontsize=9.5,
                fontname="helv",
                indent=8.0,
                gap=3.5,
            )
            builder.add_text(
                f"Matched Citation: {item.matched_citation or 'None matched'}",
                fontsize=9.0,
                fontname="heit",
                color=(0.25, 0.29, 0.35),
                indent=8.0,
                gap=3.5,
            )
            if item.source_url:
                builder.add_text(
                    f"Source URL: {item.source_url}",
                    fontsize=8.5,
                    fontname="helv",
                    color=(0.11, 0.34, 0.64),
                    indent=8.0,
                    gap=3.5,
                )
            if item.mismatch_flags:
                builder.add_text(
                    f"Mismatch Flags: {', '.join(item.mismatch_flags)}",
                    fontsize=8.5,
                    fontname="hebo",
                    color=(0.74, 0.12, 0.12),
                    indent=8.0,
                    gap=3.5,
                )
            if item.evidence_passage:
                builder.add_text(
                    f"Evidence Excerpt: \"{item.evidence_passage.strip()}\"",
                    fontsize=8.8,
                    fontname="heit",
                    color=(0.24, 0.27, 0.32),
                    indent=14.0,
                    gap=3.5,
                )
            builder.add_text(
                f"Explanation: {item.explanation}",
                fontsize=9.2,
                fontname="helv",
                indent=8.0,
                gap=3.5,
            )
            builder.add_text(
                f"Uncertainty Note: {item.uncertainty_note}",
                fontsize=8.8,
                fontname="heit",
                color=(0.35, 0.39, 0.45),
                indent=8.0,
                gap=8.0,
            )
            builder.add_divider(color=(0.90, 0.91, 0.94), gap=8.0)

    # Limitations section
    if report.limitations:
        builder.ensure_space(70.0)
        builder.add_text(
            "Verification Scope & Limitations",
            fontsize=11.0,
            fontname="hebo",
            color=(0.09, 0.16, 0.29),
            gap=5.0,
        )
        for note in report.limitations:
            builder.add_text(
                f"- {note}",
                fontsize=9.0,
                fontname="helv",
                color=(0.28, 0.32, 0.38),
                indent=8.0,
                gap=3.5,
            )

    return builder.finalize_bytes()


# Inputs: VerificationReport object and destination Path/str for the Markdown file.
# Outputs: Saved file Path.
def save_report_markdown(
    report: VerificationReport, output_path: str | Path
) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(export_report_to_markdown(report), encoding="utf-8")
    return target


# Inputs: VerificationReport object and destination Path/str for the PDF file.
# Outputs: Saved file Path.
def save_report_pdf(
    report: VerificationReport, output_path: str | Path
) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(export_report_to_pdf_bytes(report))
    return target