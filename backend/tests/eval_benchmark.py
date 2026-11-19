import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Ensure both repo root and backend/ are importable (matches how retriever/verifier import schemas)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_DIR = _REPO_ROOT / "backend"
for _path in (str(_REPO_ROOT), str(_BACKEND_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from backend.schemas import (
    ClaimStatus,
    ClaimVerificationResult,
    ConfidenceLevel,
    ExtractedClaim,
    VerificationReport,
)
from backend.services.parser import (
    extract_claims_from_pdf,
    extract_document_title,
    extract_text_from_pdf,
)
from backend.services.report_exporter import (
    build_export_filename,
    save_report_markdown,
    save_report_pdf,
)

DEFAULT_DEMO_PDF = Path(__file__).parent / "sample_papers" / "demo_paper.pdf"


@dataclass(frozen=True)
class BenchmarkCase:
    claim_id: int
    inline_marker: str
    scenario: str
    expected_verdict: ClaimStatus


DEMO_GROUND_TRUTH: list[BenchmarkCase] = [
    BenchmarkCase(
        claim_id=1,
        inline_marker="[1]",
        scenario="Gulshan (2016) - Accurate metric",
        expected_verdict=ClaimStatus.SUPPORTED,
    ),
    BenchmarkCase(
        claim_id=2,
        inline_marker="[2]",
        scenario="McKinney (2020) - Scope exaggeration",
        expected_verdict=ClaimStatus.PARTIALLY_SUPPORTED,
    ),
    BenchmarkCase(
        claim_id=3,
        inline_marker="[3]",
        scenario="Horby (2021) - Wrong mortality metric",
        expected_verdict=ClaimStatus.CONTRADICTORY,
    ),
    BenchmarkCase(
        claim_id=4,
        inline_marker="[4]",
        scenario="Kowalski (2024) - Fabricated citation",
        expected_verdict=ClaimStatus.UNSUPPORTED,
    ),
    BenchmarkCase(
        claim_id=5,
        inline_marker="[5]",
        scenario="Mehra (2020) - Retracted Lancet study",
        expected_verdict=ClaimStatus.POTENTIALLY_OUTDATED,
    ),
]


# Inputs: Path to demo_paper.pdf.
# Outputs: Tuple of (VerificationReport, stage_timings_seconds) from the full live pipeline.
def run_live_pipeline(pdf_path: Path) -> tuple[VerificationReport, dict[str, float]]:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    try:
        from backend.services.retriever import retrieve_sources_batch
        from backend.services.verifier import build_verification_report
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Missing backend dependency ({exc.name}). Run "
            "`pip install -r backend/requirements.txt` or use `--offline`."
        ) from exc

    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    raw_text = extract_text_from_pdf(pdf_path)
    document_title = extract_document_title(raw_text)
    claims = extract_claims_from_pdf(pdf_path)
    timings["parse"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    sources = retrieve_sources_batch(claims)
    timings["retrieve"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    claim_source_pairs = list(zip(claims, sources, strict=True))
    report = build_verification_report(document_title, claim_source_pairs)
    timings["verify"] = time.perf_counter() - t2
    timings["total"] = timings["parse"] + timings["retrieve"] + timings["verify"]

    return report, timings


# Inputs: Path to demo_paper.pdf.
# Outputs: Tuple of (VerificationReport, stage_timings_seconds) using real parsing + mock verdicts.
def run_offline_pipeline(pdf_path: Path) -> tuple[VerificationReport, dict[str, float]]:
    t0 = time.perf_counter()
    raw_text = extract_text_from_pdf(pdf_path)
    document_title = extract_document_title(raw_text)
    claims: list[ExtractedClaim] = extract_claims_from_pdf(pdf_path)
    parse_elapsed = time.perf_counter() - t0

    mock_details: dict[int, tuple[ClaimStatus, ConfidenceLevel, str, list[str], bool]] = {
        1: (
            ClaimStatus.SUPPORTED,
            ConfidenceLevel.HIGH,
            "Simulated match: 97.5% sensitivity / 93.4% specificity matches Gulshan et al. (2016).",
            [],
            False,
        ),
        2: (
            ClaimStatus.PARTIALLY_SUPPORTED,
            ConfidenceLevel.MEDIUM,
            "Simulated match: Study exists, but 30-country prospective trial scope is exaggerated.",
            ["SCOPE_EXAGGERATION"],
            True,
        ),
        3: (
            ClaimStatus.CONTRADICTORY,
            ConfidenceLevel.HIGH,
            "Simulated match: RECOVERY trial reduced mortality to 22.9%, contradicting 62.4%.",
            ["NUMERIC_MISMATCH"],
            True,
        ),
        4: (
            ClaimStatus.UNSUPPORTED,
            ConfidenceLevel.LOW,
            "Simulated match: Fabricated citation not found in scholarly databases.",
            ["SOURCE_NOT_FOUND"],
            True,
        ),
        5: (
            ClaimStatus.POTENTIALLY_OUTDATED,
            ConfidenceLevel.HIGH,
            "Simulated match: Mehra et al. (2020) was formally retracted by The Lancet.",
            ["RETRACTED_SOURCE"],
            True,
        ),
    }

    results: list[ClaimVerificationResult] = []
    summary_counts: dict[str, int] = {}
    for claim in claims:
        verdict, confidence, explanation, flags, review = mock_details.get(
            claim.claim_id,
            (
                ClaimStatus.UNSUPPORTED,
                ConfidenceLevel.LOW,
                "Simulated fallback verdict.",
                [],
                True,
            ),
        )
        summary_counts[verdict.value] = summary_counts.get(verdict.value, 0) + 1
        results.append(
            ClaimVerificationResult(
                claim_id=claim.claim_id,
                claim_text=claim.claim_text,
                matched_citation=claim.raw_reference,
                evidence_passage=None,
                verdict=verdict,
                explanation=explanation,
                confidence=confidence,
                uncertainty_note="Offline smoke mode (retrieval and LLM verification simulated).",
                needs_human_review=review,
                source_url=None,
                mismatch_flags=flags,
            )
        )

    report = VerificationReport(
        document_title=document_title,
        total_claims=len(results),
        summary_counts=summary_counts,
        results=results,
        limitations=[
            "Offline benchmark run: PDF parsing is live; retrieval and LLM verdicts are simulated."
        ],
    )
    timings = {"parse": parse_elapsed, "retrieve": 0.0, "verify": 0.0, "total": parse_elapsed}
    return report, timings


# Inputs: VerificationReport from pipeline and ground-truth BenchmarkCase list.
# Outputs: Dictionary containing per-claim evaluation rows, exact matches, and accuracy percent.
def evaluate_verdicts(
    report: VerificationReport,
    ground_truth: list[BenchmarkCase],
) -> dict:
    result_by_id = {item.claim_id: item for item in report.results}
    rows: list[dict] = []
    passed = 0

    for case in ground_truth:
        actual = result_by_id.get(case.claim_id)
        actual_verdict = actual.verdict.value if actual else "MISSING"
        confidence = actual.confidence.value if actual else "N/A"
        is_match = actual_verdict == case.expected_verdict.value
        if is_match:
            passed += 1

        rows.append(
            {
                "claim_id": case.claim_id,
                "marker": case.inline_marker,
                "scenario": case.scenario,
                "expected": case.expected_verdict.value,
                "actual": actual_verdict,
                "confidence": confidence,
                "matched": is_match,
            }
        )

    total = len(ground_truth)
    accuracy = (passed / total * 100.0) if total else 0.0
    return {
        "rows": rows,
        "passed": passed,
        "total": total,
        "accuracy_pct": accuracy,
        "extracted_count": report.total_claims,
    }


# Inputs: Evaluation summary dict, VerificationReport, stage timings, and mode label.
# Outputs: Formatted ASCII benchmark scorecard string suitable for terminal/judge demos.
def format_benchmark_summary(
    evaluation: dict,
    report: VerificationReport,
    timings: dict[str, float],
    mode_label: str,
) -> str:
    divider = "=" * 104
    thin_divider = "-" * 104

    lines: list[str] = [
        divider,
        f"  CITEVERIFY END-TO-END VERDICT ACCURACY BENCHMARK  [{mode_label}]",
        divider,
        f"  Document : {report.document_title}",
        f"  Claims   : {evaluation['extracted_count']} extracted  |  {evaluation['total']} ground-truth cases",
        (
            f"  Timings  : Parse={timings['parse']:.2f}s  |  "
            f"Retrieve={timings['retrieve']:.2f}s  |  "
            f"Verify={timings['verify']:.2f}s  |  Total={timings['total']:.2f}s"
        ),
        thin_divider,
        f"  {'ID':<3} {'Ref':<4} {'Scenario':<42} {'Expected':<21} {'Actual':<21} {'Result'}",
        thin_divider,
    ]

    for row in evaluation["rows"]:
        status = "PASS" if row["matched"] else "FAIL"
        scenario_short = row["scenario"][:41]
        lines.append(
            f"  {row['claim_id']:<3} {row['marker']:<4} {scenario_short:<42} "
            f"{row['expected']:<21} {row['actual']:<21} [{status}]"
        )

    lines.extend(
        [
            thin_divider,
            (
                f"  VERDICT ACCURACY: {evaluation['passed']} / {evaluation['total']} "
                f"({evaluation['accuracy_pct']:.1f}%)"
            ),
            divider,
        ]
    )
    return "\n".join(lines)


def main() -> int:
    arg_parser = argparse.ArgumentParser(
        description="Run CiteVerify end-to-end benchmark on demo_paper.pdf."
    )
    arg_parser.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_DEMO_PDF,
        help="Path to the input PDF (defaults to backend/tests/sample_papers/demo_paper.pdf).",
    )
    arg_parser.add_argument(
        "--offline",
        action="store_true",
        help="Run real PDF parser with simulated retrieval/verification (no API keys required).",
    )
    arg_parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Optional directory to save exported Markdown and PDF audit reports.",
    )
    args = arg_parser.parse_args()

    if not args.pdf.exists():
        print(f"Error: PDF not found at {args.pdf}", file=sys.stderr)
        return 1

    try:
        if args.offline:
            report, timings = run_offline_pipeline(args.pdf)
            mode_label = "OFFLINE SMOKE TEST"
        else:
            report, timings = run_live_pipeline(args.pdf)
            mode_label = "LIVE PIPELINE"
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"Benchmark execution failed: {exc}", file=sys.stderr)
        return 1

    evaluation = evaluate_verdicts(report, DEMO_GROUND_TRUTH)
    print(format_benchmark_summary(evaluation, report, timings, mode_label))

    if args.export_dir is not None:
        md_path = save_report_markdown(
            report,
            args.export_dir / build_export_filename(report.document_title, "md"),
        )
        pdf_path = save_report_pdf(
            report,
            args.export_dir / build_export_filename(report.document_title, "pdf"),
        )
        print(f"\nExported Markdown report : {md_path}")
        print(f"Exported PDF report      : {pdf_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())