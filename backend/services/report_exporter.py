import re
from datetime import UTC, datetime
from pathlib import Path

import pymupdf

# Support running from repo root (backend.schemas) or inside backend/ (schemas)
try:
    from backend.schemas import ClaimStatus, ClaimVerificationResult, VerificationReport
except ImportError:
    from schemas import ClaimStatus, ClaimVerificationResult, VerificationReport


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

_TEMPORAL_STATUS_LABELS: dict[str, str] = {
    "CONSENSUS_ALIGNED": "Consensus Aligned",
    "SUPERSEDED_OR_CHALLENGED": "Superseded / Challenged",
    "RETRACTED": "Retracted",
    "CORRECTED_OR_UPDATED": "Corrected / Updated",
    "STALE_NEEDS_UPDATE": "Aging / Check Updates",
    "INSUFFICIENT_TEMPORAL_DATA": "Limited Temporal Data",
}

_CALIBRATION_ACTION_LABELS: dict[str, str] = {
    "PRESENT_VERDICT": "Auto-Verified",
    "FLAG_FOR_REVIEW": "Flag for Review",
    "ABSTAIN": "Abstained",
    "MARK_UNVERIFIABLE": "Unverifiable",
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
        "\u2192": "->",
    }
)


# Inputs: Arbitrary text string that may include smart quotes or non-Latin-1 Unicode characters.
# Outputs: Latin-1 safe string suitable for PyMuPDF built-in PDF fonts.
def _sanitize_pdf_text(text: str) -> str:
    normalized = text.translate(_PDF_CHAR_REPLACEMENTS)
    return normalized.encode("latin-1", errors="replace").decode("latin-1")


# Inputs: Float value in [0.0, 1.0].
# Outputs: Whole-percent formatted string (e.g., "88%").
def _fmt_pct(value: float) -> str:
    clamped = min(1.0, max(0.0, float(value)))
    return f"{round(clamped * 100)}%"


# Inputs: Enum-style snake_case string (e.g., "PEER_REVIEWED_JOURNAL").
# Outputs: Human-readable title-cased label (e.g., "Peer Reviewed Journal").
def _fmt_stage(stage: str) -> str:
    return " ".join(part.capitalize() for part in (stage or "UNKNOWN").split("_"))


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


# Inputs: VerificationReport instance.
# Outputs: Dict of document-level Gap 1-4 roll-up counts, or None if no claim has Gap 1-4 data.
def _compute_gap_rollup(report: VerificationReport) -> dict[str, int] | None:
    has_any = False
    similarity_traps = 0
    compound_claims = 0
    temporal_alerts = 0
    abstentions = 0

    for item in report.results:
        if (
            item.evidence_ranking
            or item.claim_decomposition
            or item.temporal_assessment
            or item.uncertainty_calibration
        ):
            has_any = True
        if item.evidence_ranking and item.evidence_ranking.similarity_trap_detected:
            similarity_traps += 1
        if item.claim_decomposition and item.claim_decomposition.is_compound:
            compound_claims += 1
        if item.temporal_assessment and item.temporal_assessment.needs_temporal_review:
            temporal_alerts += 1
        if item.uncertainty_calibration and item.uncertainty_calibration.abstained:
            abstentions += 1

    if not has_any:
        return None

    return {
        "similarity_traps": similarity_traps,
        "compound_claims": compound_claims,
        "temporal_alerts": temporal_alerts,
        "abstentions": abstentions,
    }


# Inputs: ClaimVerificationResult instance.
# Outputs: True if any Gap 1-4 diagnostic object is populated on this claim.
def _has_gap_diagnostics(item: ClaimVerificationResult) -> bool:
    return any(
        (
            item.evidence_ranking,
            item.claim_decomposition,
            item.temporal_assessment,
            item.uncertainty_calibration,
        )
    )


# Inputs: Document title and desired file extension (e.g., "md" or "pdf").
# Outputs: Safe lowercase filename string for HTTP Content-Disposition download headers.
def build_export_filename(document_title: str, extension: str) -> str:
    ext = extension.lstrip(".").lower() or "md"
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", (document_title or "").strip()).strip("_").lower()
    slug = slug[:60] if slug else "untitled_document"
    return f"citeverify_audit_{slug}.{ext}"


# Inputs: VerificationReport schema object and optional UTC timestamp override.
# Outputs: Complete, well-structured Markdown audit report string including Gaps 1-4 diagnostics.
def export_report_to_markdown(
    report: VerificationReport,
    generated_at: datetime | None = None,
) -> str:
    timestamp = (generated_at or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M UTC")
    counts = _normalized_summary_counts(report)
    total = report.total_claims or len(report.results)
    review_needed = sum(1 for item in report.results if item.needs_human_review)
    gap_rollup = _compute_gap_rollup(report)

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

    if gap_rollup is not None:
        lines.extend(
            [
                "",
                "### Deep Audit Diagnostics Roll-Up (Gaps 1–4)",
                "",
                "| Module | Signal | Count |",
                "| --- | --- | ---: |",
                f"| **Gap 1 · Hybrid NLI Ranking** | Similarity Traps Flagged | {gap_rollup['similarity_traps']} |",
                f"| **Gap 2 · Atomic Decomposition** | Compound Claims Split | {gap_rollup['compound_claims']} |",
                f"| **Gap 3 · Temporal Consensus** | Temporal Review Alerts | {gap_rollup['temporal_alerts']} |",
                f"| **Gap 4 · Uncertainty Calibration** | Abstentions / Unverifiable | {gap_rollup['abstentions']} |",
            ]
        )

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

            # Deep Audit Diagnostics (Gaps 1-4)
            if _has_gap_diagnostics(item):
                lines.append("#### Deep Audit Diagnostics (Gaps 1–4)")
                lines.append("")

                g1 = item.evidence_ranking
                if g1:
                    trap_label = "YES (Similarity Trap Flagged)" if g1.similarity_trap_detected else "No"
                    lines.append(
                        f"- **Gap 1 (Hybrid NLI Ranking):** Stance `{g1.top_logical_stance}` | "
                        f"Cross-Encoder `{g1.top_cross_encoder_score:.2f}` | "
                        f"BM25 `{g1.top_bm25_score:.2f}` | Semantic `{g1.top_semantic_score:.2f}` | "
                        f"RRF `{g1.top_rrf_score:.4f}` | Similarity Trap: **{trap_label}** "
                        f"({len(g1.ranked_candidates)} passage(s) ranked)"
                    )

                g2 = item.claim_decomposition
                if g2:
                    split_label = (
                        f"Compound ({len(g2.propositions)} propositions)"
                        if g2.is_compound
                        else "Atomic (1 proposition)"
                    )
                    lines.append(
                        f"- **Gap 2 (Atomic Decomposition):** {split_label} | "
                        f"Aggregated Verdict: `{g2.overall_verdict.value}` ({g2.overall_confidence.value})"
                    )
                    if g2.is_compound and g2.propositions:
                        for prop in g2.propositions:
                            lines.append(
                                f"  - `[{prop.proposition_id}]` **{prop.verdict.value}** "
                                f"({prop.confidence.value}): {prop.proposition_text}"
                            )

                g3 = item.temporal_assessment
                if g3:
                    t_label = _TEMPORAL_STATUS_LABELS.get(g3.temporal_status, g3.temporal_status)
                    year_txt = str(g3.cited_publication_year) if g3.cited_publication_year else "Year N/A"
                    lines.append(
                        f"- **Gap 3 (Temporal Consensus):** **{t_label}** (`{g3.temporal_status}`) | "
                        f"{year_txt} ({_fmt_stage(g3.publication_stage)}) | "
                        f"Citations: `{g3.subsequent_citation_count:,}` | "
                        f"Confirming: `+{g3.confirming_studies_count}` / Challenging: `-{g3.challenging_studies_count}` | "
                        f"Systematic Reviews: `{g3.systematic_reviews_count}`"
                    )
                    if g3.temporal_summary:
                        lines.append(f"  - *Summary:* {g3.temporal_summary}")
                    if g3.chronological_timeline:
                        for tl_entry in g3.chronological_timeline[:3]:
                            lines.append(f"  - *Timeline:* {tl_entry}")

                g4 = item.uncertainty_calibration
                if g4:
                    a_label = _CALIBRATION_ACTION_LABELS.get(g4.action, g4.action)
                    lines.append(
                        f"- **Gap 4 (Uncertainty Calibration):** Policy **{a_label}** (`{g4.action}`) | "
                        f"Calibrated Prob `{_fmt_pct(g4.calibrated_probability)}` | "
                        f"Coverage `{_fmt_pct(g4.evidence_coverage_score)}` | "
                        f"Retrieval `{_fmt_pct(g4.retrieval_confidence)}` | "
                        f"Agreement `{_fmt_pct(g4.passage_agreement_score)}` | "
                        f"Composite `{_fmt_pct(g4.composite_reliability_score)}`"
                    )
                    if g4.user_message:
                        lines.append(f"  - *Policy Note:* {g4.user_message}")
                    if g4.uncertainty_flags:
                        u_flags = ", ".join(f"`{f}`" for f in g4.uncertainty_flags)
                        lines.append(f"  - *Calibration Flags:* {u_flags}")

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
# Outputs: Downloadable multi-page PDF audit report as raw bytes including Gaps 1-4 diagnostics.
def export_report_to_pdf_bytes(
    report: VerificationReport,
    generated_at: datetime | None = None,
) -> bytes:
    timestamp = (generated_at or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M UTC")
    counts = _normalized_summary_counts(report)
    total = report.total_claims or len(report.results)
    review_needed = sum(1 for item in report.results if item.needs_human_review)
    gap_rollup = _compute_gap_rollup(report)

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

    if gap_rollup is not None:
        builder.y += 4.0
        builder.add_text(
            "Deep Audit Diagnostics Roll-Up (Gaps 1-4)",
            fontsize=10.0,
            fontname="hebo",
            color=(0.15, 0.22, 0.42),
            gap=4.0,
        )
        builder.add_text(
            (
                f"Gap 1 Similarity Traps: {gap_rollup['similarity_traps']}   |   "
                f"Gap 2 Compound Claims: {gap_rollup['compound_claims']}   |   "
                f"Gap 3 Temporal Alerts: {gap_rollup['temporal_alerts']}   |   "
                f"Gap 4 Abstentions: {gap_rollup['abstentions']}"
            ),
            fontsize=9.0,
            fontname="hebo",
            color=(0.20, 0.25, 0.36),
            indent=8.0,
            gap=4.0,
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
            builder.ensure_space(130.0)
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
                gap=4.5,
            )

            # Deep Audit Diagnostics (Gaps 1-4)
            if _has_gap_diagnostics(item):
                builder.ensure_space(55.0)
                builder.add_text(
                    "Deep Audit Diagnostics (Gaps 1-4):",
                    fontsize=8.8,
                    fontname="hebo",
                    color=(0.15, 0.22, 0.42),
                    indent=8.0,
                    gap=2.5,
                )

                g1 = item.evidence_ranking
                if g1:
                    trap_txt = "YES - Similarity Trap Flagged" if g1.similarity_trap_detected else "No"
                    builder.add_text(
                        (
                            f"- Gap 1 (Hybrid NLI): Stance {g1.top_logical_stance} | "
                            f"CE {g1.top_cross_encoder_score:.2f}, BM25 {g1.top_bm25_score:.2f}, "
                            f"Sem {g1.top_semantic_score:.2f}, RRF {g1.top_rrf_score:.4f} | "
                            f"Similarity Trap: {trap_txt} ({len(g1.ranked_candidates)} ranked)"
                        ),
                        fontsize=8.4,
                        fontname="helv",
                        color=(0.74, 0.12, 0.12) if g1.similarity_trap_detected else (0.20, 0.25, 0.32),
                        indent=14.0,
                        gap=2.5,
                    )

                g2 = item.claim_decomposition
                if g2:
                    split_txt = (
                        f"Compound ({len(g2.propositions)} parts)"
                        if g2.is_compound
                        else "Atomic (1 part)"
                    )
                    builder.add_text(
                        (
                            f"- Gap 2 (Atomic Split): {split_txt} | "
                            f"Aggregated Verdict: {g2.overall_verdict.value} ({g2.overall_confidence.value})"
                        ),
                        fontsize=8.4,
                        fontname="helv",
                        color=(0.20, 0.25, 0.32),
                        indent=14.0,
                        gap=2.5,
                    )
                    if g2.is_compound and g2.propositions:
                        for prop in g2.propositions:
                            builder.add_text(
                                f"  [{prop.proposition_id}: {prop.verdict.value}] {prop.proposition_text}",
                                fontsize=8.2,
                                fontname="heit",
                                color=(0.28, 0.32, 0.38),
                                indent=20.0,
                                gap=2.0,
                            )

                g3 = item.temporal_assessment
                if g3:
                    t_label = _TEMPORAL_STATUS_LABELS.get(g3.temporal_status, g3.temporal_status)
                    year_txt = str(g3.cited_publication_year) if g3.cited_publication_year else "Year N/A"
                    builder.add_text(
                        (
                            f"- Gap 3 (Temporal Consensus): {t_label} | {year_txt} ({_fmt_stage(g3.publication_stage)}) | "
                            f"{g3.subsequent_citation_count:,} cites | "
                            f"+{g3.confirming_studies_count} confirming / -{g3.challenging_studies_count} challenging | "
                            f"{g3.systematic_reviews_count} review(s)"
                        ),
                        fontsize=8.4,
                        fontname="helv",
                        color=(0.66, 0.35, 0.04) if g3.needs_temporal_review else (0.20, 0.25, 0.32),
                        indent=14.0,
                        gap=2.5,
                    )
                    if g3.temporal_summary:
                        builder.add_text(
                            f"  Summary: {g3.temporal_summary}",
                            fontsize=8.2,
                            fontname="heit",
                            color=(0.30, 0.34, 0.40),
                            indent=20.0,
                            gap=2.0,
                        )
                    for tl_entry in g3.chronological_timeline[:2]:
                        builder.add_text(
                            f"  Timeline: {tl_entry}",
                            fontsize=8.0,
                            fontname="helv",
                            color=(0.34, 0.38, 0.44),
                            indent=20.0,
                            gap=2.0,
                        )

                g4 = item.uncertainty_calibration
                if g4:
                    a_label = _CALIBRATION_ACTION_LABELS.get(g4.action, g4.action)
                    builder.add_text(
                        (
                            f"- Gap 4 (Uncertainty Calibration): Policy {a_label} | "
                            f"Calibrated Prob {_fmt_pct(g4.calibrated_probability)}, "
                            f"Coverage {_fmt_pct(g4.evidence_coverage_score)}, "
                            f"Retrieval {_fmt_pct(g4.retrieval_confidence)}, "
                            f"Agreement {_fmt_pct(g4.passage_agreement_score)}, "
                            f"Composite {_fmt_pct(g4.composite_reliability_score)}"
                        ),
                        fontsize=8.4,
                        fontname="helv",
                        color=(0.55, 0.16, 0.52) if g4.abstained else (0.20, 0.25, 0.32),
                        indent=14.0,
                        gap=2.5,
                    )
                    if g4.user_message:
                        builder.add_text(
                            f"  Policy Note: {g4.user_message}",
                            fontsize=8.2,
                            fontname="heit",
                            color=(0.30, 0.34, 0.40),
                            indent=20.0,
                            gap=2.5,
                        )

            builder.y += 4.0
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