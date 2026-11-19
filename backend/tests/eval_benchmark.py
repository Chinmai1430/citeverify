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
from backend.services.claim_decomposer import (
    decompose_claim,
    verify_decomposed_claim,
)
from backend.services.evidence_ranker import (
    EvidenceCandidate,
    LogicalStance,
    rank_and_verify_evidence,
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
    expects_source_found: bool = True


DEMO_GROUND_TRUTH: list[BenchmarkCase] = [
    BenchmarkCase(
        claim_id=1,
        inline_marker="[1]",
        scenario="Gulshan (2016) - Accurate metric",
        expected_verdict=ClaimStatus.SUPPORTED,
        expects_source_found=True,
    ),
    BenchmarkCase(
        claim_id=2,
        inline_marker="[2]",
        scenario="McKinney (2020) - Scope exaggeration",
        expected_verdict=ClaimStatus.PARTIALLY_SUPPORTED,
        expects_source_found=True,
    ),
    BenchmarkCase(
        claim_id=3,
        inline_marker="[3]",
        scenario="Horby (2021) - Wrong mortality metric",
        expected_verdict=ClaimStatus.CONTRADICTORY,
        expects_source_found=True,
    ),
    BenchmarkCase(
        claim_id=4,
        inline_marker="[4]",
        scenario="Kowalski (2024) - Fabricated citation",
        expected_verdict=ClaimStatus.UNSUPPORTED,
        expects_source_found=False,
    ),
    BenchmarkCase(
        claim_id=5,
        inline_marker="[5]",
        scenario="Mehra (2020) - Retracted Lancet study",
        expected_verdict=ClaimStatus.POTENTIALLY_OUTDATED,
        expects_source_found=True,
    ),
]


# Inputs: Path to demo_paper.pdf.
# Outputs: Tuple of (VerificationReport, stage_timings_seconds, per_claim_retrieval_meta) from the live pipeline.
def run_live_pipeline(
    pdf_path: Path,
) -> tuple[VerificationReport, dict[str, float], dict[int, dict[str, bool | str]]]:
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

    retrieval_meta: dict[int, dict[str, bool | str]] = {}
    for claim, src in zip(claims, sources, strict=True):
        found = bool(getattr(src, "found", False))
        passage = getattr(src, "source_passage", None)
        scope = str(getattr(src, "evidence_scope", "NONE"))
        retrieval_meta[claim.claim_id] = {
            "found": found,
            "has_passage": bool(passage and str(passage).strip()),
            "evidence_scope": scope,
        }

    t2 = time.perf_counter()
    claim_source_pairs = list(zip(claims, sources, strict=True))
    report = build_verification_report(document_title, claim_source_pairs)
    timings["verify"] = time.perf_counter() - t2
    timings["total"] = timings["parse"] + timings["retrieve"] + timings["verify"]

    return report, timings, retrieval_meta


# Inputs: Path to demo_paper.pdf.
# Outputs: Tuple of (VerificationReport, stage_timings_seconds, per_claim_retrieval_meta) using real parsing + mocks.
def run_offline_pipeline(
    pdf_path: Path,
) -> tuple[VerificationReport, dict[str, float], dict[int, dict[str, bool | str]]]:
    t0 = time.perf_counter()
    raw_text = extract_text_from_pdf(pdf_path)
    document_title = extract_document_title(raw_text)
    claims: list[ExtractedClaim] = extract_claims_from_pdf(pdf_path)
    parse_elapsed = time.perf_counter() - t0

    mock_details: dict[int, tuple[ClaimStatus, ConfidenceLevel, str, list[str], bool, bool, bool]] = {
        1: (
            ClaimStatus.SUPPORTED,
            ConfidenceLevel.HIGH,
            "Simulated match: 97.5% sensitivity / 93.4% specificity matches Gulshan et al. (2016).",
            [],
            False,
            True,
            True,
        ),
        2: (
            ClaimStatus.PARTIALLY_SUPPORTED,
            ConfidenceLevel.MEDIUM,
            "Simulated match: Study exists, but 30-country prospective trial scope is exaggerated.",
            ["SCOPE_EXAGGERATION"],
            True,
            True,
            True,
        ),
        3: (
            ClaimStatus.CONTRADICTORY,
            ConfidenceLevel.HIGH,
            "Simulated match: RECOVERY trial reduced mortality to 22.9%, contradicting 62.4%.",
            ["NUMERIC_MISMATCH"],
            True,
            True,
            True,
        ),
        4: (
            ClaimStatus.UNSUPPORTED,
            ConfidenceLevel.LOW,
            "Simulated match: Fabricated citation not found in scholarly databases.",
            ["SOURCE_NOT_FOUND"],
            True,
            False,
            False,
        ),
        5: (
            ClaimStatus.POTENTIALLY_OUTDATED,
            ConfidenceLevel.HIGH,
            "Simulated match: Mehra et al. (2020) was formally retracted by The Lancet.",
            ["RETRACTED_SOURCE"],
            True,
            True,
            True,
        ),
    }

    results: list[ClaimVerificationResult] = []
    retrieval_meta: dict[int, dict[str, bool | str]] = {}
    summary_counts: dict[str, int] = {}

    for claim in claims:
        verdict, confidence, explanation, flags, review, found, has_passage = mock_details.get(
            claim.claim_id,
            (
                ClaimStatus.UNSUPPORTED,
                ConfidenceLevel.LOW,
                "Simulated fallback verdict.",
                [],
                True,
                False,
                False,
            ),
        )
        retrieval_meta[claim.claim_id] = {
            "found": found,
            "has_passage": has_passage,
            "evidence_scope": "ABSTRACT_ONLY" if has_passage else "NONE",
        }
        summary_counts[verdict.value] = summary_counts.get(verdict.value, 0) + 1
        results.append(
            ClaimVerificationResult(
                claim_id=claim.claim_id,
                claim_text=claim.claim_text,
                matched_citation=claim.raw_reference,
                evidence_passage="Simulated abstract evidence." if has_passage else None,
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
    return report, timings, retrieval_meta


# Inputs: VerificationReport, ground-truth BenchmarkCase list, and optional per-claim retrieval metadata.
# Outputs: Dictionary separating Evidence Retrieval Recall (source & passage) from Verdict Accuracy.
def evaluate_verdicts(
    report: VerificationReport,
    ground_truth: list[BenchmarkCase],
    retrieval_meta: dict[int, dict[str, bool | str]] | None = None,
) -> dict:
    result_by_id = {item.claim_id: item for item in report.results}
    meta_map = retrieval_meta or {}
    rows: list[dict] = []

    verdict_passed = 0
    real_sources_total = 0
    sources_found_count = 0
    passages_retrieved_count = 0
    fabricated_rejected_count = 0
    fabricated_total = 0

    for case in ground_truth:
        actual = result_by_id.get(case.claim_id)
        actual_verdict = actual.verdict.value if actual else "MISSING"
        confidence = actual.confidence.value if actual else "N/A"
        is_match = actual_verdict == case.expected_verdict.value
        if is_match:
            verdict_passed += 1

        meta = meta_map.get(case.claim_id, {})
        found = bool(meta.get("found", bool(actual and (actual.source_url or actual.evidence_passage))))
        has_passage = bool(meta.get("has_passage", bool(actual and actual.evidence_passage)))

        if case.expects_source_found:
            real_sources_total += 1
            if found:
                sources_found_count += 1
            if has_passage:
                passages_retrieved_count += 1
            retrieval_label = (
                "PASSAGE" if has_passage else ("META_ONLY" if found else "MISSED")
            )
        else:
            fabricated_total += 1
            if not found:
                fabricated_rejected_count += 1
                retrieval_label = "NONE (OK)"
            else:
                retrieval_label = "FALSE_HIT"

        rows.append(
            {
                "claim_id": case.claim_id,
                "marker": case.inline_marker,
                "scenario": case.scenario,
                "retrieval": retrieval_label,
                "expected": case.expected_verdict.value,
                "actual": actual_verdict,
                "confidence": confidence,
                "matched": is_match,
            }
        )

    total = len(ground_truth)
    verdict_accuracy = (verdict_passed / total * 100.0) if total else 0.0
    source_recall = (sources_found_count / real_sources_total * 100.0) if real_sources_total else 0.0
    passage_recall = (
        (passages_retrieved_count / real_sources_total * 100.0) if real_sources_total else 0.0
    )

    return {
        "rows": rows,
        "passed": verdict_passed,
        "total": total,
        "accuracy_pct": verdict_accuracy,
        "sources_found": sources_found_count,
        "passages_retrieved": passages_retrieved_count,
        "real_sources_total": real_sources_total,
        "source_recall_pct": source_recall,
        "passage_recall_pct": passage_recall,
        "fabricated_rejected": fabricated_rejected_count,
        "fabricated_total": fabricated_total,
        "extracted_count": report.total_claims,
    }


# Inputs: Optional flags to enable live Gemini embeddings/NLI during the Gap 1 suite.
# Outputs: Dictionary measuring Gap 1 Evidence Retrieval Recall@1, Logical Stance Accuracy, and Trap Detection.
def run_gap1_evidence_benchmark(
    use_embeddings: bool = False,
    use_llm_nli: bool = False,
) -> dict:
    cases = [
        {
            "name": "Learning rate polarity trap",
            "claim": "Increasing the learning rate improves model accuracy.",
            "candidates": [
                EvidenceCandidate(
                    passage_id="p_contradict",
                    text="We observed that increasing the learning rate reduced model accuracy after a certain threshold.",
                ),
                EvidenceCandidate(
                    passage_id="p_neutral",
                    text="Learning rate schedules are a common hyperparameter choice in neural network training.",
                ),
                EvidenceCandidate(
                    passage_id="p_distractor",
                    text="Convolutional filters extract spatial edges from chest X-ray radiographs.",
                ),
            ],
            "expected_passage": "p_contradict",
            "expected_stance": LogicalStance.CONTRADICTS,
            "expect_trap": True,
        },
        {
            "name": "EyePACS-1 direct entailment",
            "claim": "Deep learning achieved 97.5% sensitivity on EyePACS-1.",
            "candidates": [
                EvidenceCandidate(
                    passage_id="p_distractor",
                    text="Randomized trials evaluate steroid therapy in hospitalized respiratory cohorts.",
                ),
                EvidenceCandidate(
                    passage_id="p_neutral",
                    text="Diabetic retinopathy screening programs frequently review EyePACS-1 fundus photographs.",
                ),
                EvidenceCandidate(
                    passage_id="p_support",
                    text="On EyePACS-1, the deep learning algorithm achieved 97.5% sensitivity and 93.4% specificity.",
                ),
            ],
            "expected_passage": "p_support",
            "expected_stance": LogicalStance.ENTAILS,
            "expect_trap": False,
        },
        {
            "name": "Explicit negation flip trap",
            "claim": "Vitamin D supplementation reduces winter fracture risk in older adults.",
            "candidates": [
                EvidenceCandidate(
                    passage_id="p_negated",
                    text="Vitamin D supplementation did not reduce winter fracture risk in older adults.",
                ),
                EvidenceCandidate(
                    passage_id="p_bg",
                    text="Vitamin D metabolism is studied extensively across geriatric nutrition cohorts.",
                ),
            ],
            "expected_passage": "p_negated",
            "expected_stance": LogicalStance.CONTRADICTS,
            "expect_trap": True,
        },
        {
            "name": "Topical overlap only (Neutral)",
            "claim": "Sparse clinical transformers reduce ICU mortality by 15%.",
            "candidates": [
                EvidenceCandidate(
                    passage_id="p_bg",
                    text="Sparse clinical transformers are neural sequence architectures applied to ICU time-series records.",
                ),
                EvidenceCandidate(
                    passage_id="p_distractor",
                    text="Magnetic resonance imaging uses radiofrequency pulses to image soft tissues.",
                ),
            ],
            "expected_passage": "p_bg",
            "expected_stance": LogicalStance.NEUTRAL,
            "expect_trap": False,
        },
    ]

    rows: list[dict] = []
    recall_hits = 0
    stance_hits = 0
    traps_expected = 0
    traps_caught = 0
    embedding_backend = "tfidf_fallback"
    nli_backend = "rule_based_nli"

    for case in cases:
        summary = rank_and_verify_evidence(
            case["claim"],
            case["candidates"],
            top_k=3,
            use_embeddings=use_embeddings,
            use_llm_nli=use_llm_nli,
        )
        embedding_backend = summary.embedding_backend
        nli_backend = summary.nli_backend

        top = summary.top_evidence
        top_id = top.passage_id if top else "NONE"
        actual_stance = top.logical_stance if top else LogicalStance.NEUTRAL

        recall_ok = top_id == case["expected_passage"]
        stance_ok = actual_stance == case["expected_stance"]
        if recall_ok:
            recall_hits += 1
        if stance_ok:
            stance_hits += 1
        if case["expect_trap"]:
            traps_expected += 1
            if summary.similarity_trap_detected:
                traps_caught += 1

        rows.append(
            {
                "name": case["name"],
                "recall_ok": recall_ok,
                "expected_stance": case["expected_stance"].value,
                "actual_stance": actual_stance.value,
                "trap_flagged": summary.similarity_trap_detected,
                "passed": recall_ok and stance_ok and (summary.similarity_trap_detected == case["expect_trap"]),
            }
        )

    total = len(cases)
    return {
        "rows": rows,
        "recall_hits": recall_hits,
        "stance_hits": stance_hits,
        "traps_caught": traps_caught,
        "traps_expected": traps_expected,
        "total": total,
        "recall_pct": (recall_hits / total * 100.0) if total else 0.0,
        "stance_pct": (stance_hits / total * 100.0) if total else 0.0,
        "embedding_backend": embedding_backend,
        "nli_backend": nli_backend,
    }


# Inputs: Optional flag to allow LLM-assisted decomposition.
# Outputs: Dictionary measuring Gap 2 Atomic Decomposition & Combined Verdict Accuracy.
def run_gap2_decomposition_benchmark(use_llm_decomposition: bool = False) -> dict:
    cases = [
        {
            "name": "Transformer vs CNN (mixed verdict)",
            "claim": ExtractedClaim(
                claim_id=1,
                claim_text=(
                    "Transformer models outperform CNNs in image classification "
                    "while requiring fewer computational resources [1]."
                ),
                inline_marker="[1]",
            ),
            "expected_props": 2,
            "expected_verdict": ClaimStatus.PARTIALLY_SUPPORTED,
            "mock_fn": lambda sub: ClaimVerificationResult(
                claim_id=sub.claim_id,
                claim_text=sub.claim_text,
                verdict=(
                    ClaimStatus.SUPPORTED
                    if "outperform CNNs" in sub.claim_text
                    else ClaimStatus.CONTRADICTORY
                ),
                explanation="Mock atomic proposition check.",
                confidence=ConfidenceLevel.HIGH,
                uncertainty_note="Benchmark check.",
                needs_human_review="outperform CNNs" not in sub.claim_text,
            ),
        },
        {
            "name": "Population + dual metrics (both supported)",
            "claim": ExtractedClaim(
                claim_id=2,
                claim_text=(
                    "In elderly diabetic patients, Drug X reduced hospital readmissions by 28% "
                    "and lowered systolic blood pressure by 12 mmHg [2]."
                ),
                inline_marker="[2]",
            ),
            "expected_props": 2,
            "expected_verdict": ClaimStatus.SUPPORTED,
            "mock_fn": lambda sub: ClaimVerificationResult(
                claim_id=sub.claim_id,
                claim_text=sub.claim_text,
                verdict=ClaimStatus.SUPPORTED,
                explanation="Confirmed metric in cohort.",
                confidence=ConfidenceLevel.HIGH,
                uncertainty_note="Benchmark check.",
                needs_human_review=False,
            ),
        },
        {
            "name": "Atomic sentence guard (no false split)",
            "claim": ExtractedClaim(
                claim_id=3,
                claim_text="Deep learning achieved 97.5% sensitivity and 93.4% specificity on EyePACS-1 [1].",
                inline_marker="[1]",
            ),
            "expected_props": 1,
            "expected_verdict": ClaimStatus.SUPPORTED,
            "mock_fn": lambda sub: ClaimVerificationResult(
                claim_id=sub.claim_id,
                claim_text=sub.claim_text,
                verdict=ClaimStatus.SUPPORTED,
                explanation="Confirmed metrics.",
                confidence=ConfidenceLevel.HIGH,
                uncertainty_note="Benchmark check.",
                needs_human_review=False,
            ),
        },
    ]

    rows: list[dict] = []
    passed = 0
    for case in cases:
        decomposed = decompose_claim(case["claim"], use_llm=use_llm_decomposition)
        combined = verify_decomposed_claim(
            case["claim"],
            verify_fn=case["mock_fn"],
            use_llm_decomposition=use_llm_decomposition,
        )
        ok = (
            len(decomposed.propositions) == case["expected_props"]
            and combined.overall_verdict == case["expected_verdict"]
        )
        if ok:
            passed += 1
        rows.append(
            {
                "name": case["name"],
                "expected_props": case["expected_props"],
                "actual_props": len(decomposed.propositions),
                "expected_verdict": case["expected_verdict"].value,
                "actual_verdict": combined.overall_verdict.value,
                "passed": ok,
            }
        )

    total = len(cases)
    return {
        "rows": rows,
        "passed": passed,
        "total": total,
        "accuracy_pct": (passed / total * 100.0) if total else 0.0,
    }


# Inputs: Evaluation dict, VerificationReport, timings, mode label, and optional Gap 1 / Gap 2 summaries.
# Outputs: Formatted ASCII benchmark scorecard separating Evidence Retrieval Recall and Verdict Accuracy.
def format_benchmark_summary(
    evaluation: dict,
    report: VerificationReport,
    timings: dict[str, float],
    mode_label: str,
    gap1_eval: dict | None = None,
    gap2_eval: dict | None = None,
) -> str:
    divider = "=" * 112
    thin_divider = "-" * 112

    lines: list[str] = [
        divider,
        f"  CITEVERIFY DECOUPLED RETRIEVAL & VERDICT BENCHMARK  [{mode_label}]",
        divider,
        f"  Document : {report.document_title}",
        f"  Claims   : {evaluation['extracted_count']} extracted  |  {evaluation['total']} ground-truth cases",
        (
            f"  Timings  : Parse={timings['parse']:.2f}s  |  "
            f"Retrieve={timings['retrieve']:.2f}s  |  "
            f"Verify={timings['verify']:.2f}s  |  Total={timings['total']:.2f}s"
        ),
        thin_divider,
        f"  {'ID':<3} {'Ref':<4} {'Scenario':<38} {'Evidence':<10} {'Expected':<21} {'Actual':<21} {'Result'}",
        thin_divider,
    ]

    for row in evaluation["rows"]:
        status = "PASS" if row["matched"] else "FAIL"
        scenario_short = row["scenario"][:37]
        lines.append(
            f"  {row['claim_id']:<3} {row['marker']:<4} {scenario_short:<38} "
            f"{row['retrieval']:<10} {row['expected']:<21} {row['actual']:<21} [{status}]"
        )

    lines.extend(
        [
            thin_divider,
            (
                f"  SOURCE RETRIEVAL RECALL  : {evaluation['sources_found']} / {evaluation['real_sources_total']} "
                f"({evaluation['source_recall_pct']:.1f}%)   |   "
                f"PASSAGE RECALL: {evaluation['passages_retrieved']} / {evaluation['real_sources_total']} "
                f"({evaluation['passage_recall_pct']:.1f}%)"
            ),
            (
                f"  END-TO-END VERDICT ACC   : {evaluation['passed']} / {evaluation['total']} "
                f"({evaluation['accuracy_pct']:.1f}%)   |   "
                f"FABRICATED REJECTED: {evaluation['fabricated_rejected']} / {evaluation['fabricated_total']}"
            ),
            divider,
        ]
    )

    if gap1_eval is not None:
        lines.extend(
            [
                "",
                divider,
                (
                    f"  GAP 1: HYBRID EVIDENCE RETRIEVAL vs. LOGICAL STANCE  "
                    f"[Embed={gap1_eval['embedding_backend']} | NLI={gap1_eval['nli_backend']}]"
                ),
                thin_divider,
                f"  {'Case':<34} {'Recall@1':<10} {'Expected':<14} {'Actual':<14} {'SimTrap':<10} {'Result'}",
                thin_divider,
            ]
        )
        for row in gap1_eval["rows"]:
            lines.append(
                f"  {row['name'][:33]:<34} {'HIT' if row['recall_ok'] else 'MISS':<10} "
                f"{row['expected_stance']:<14} {row['actual_stance']:<14} "
                f"{'YES' if row['trap_flagged'] else 'NO':<10} [{'PASS' if row['passed'] else 'FAIL'}]"
            )
        lines.extend(
            [
                thin_divider,
                (
                    f"  RETRIEVAL RECALL@1: {gap1_eval['recall_hits']}/{gap1_eval['total']} ({gap1_eval['recall_pct']:.1f}%)  |  "
                    f"STANCE ACCURACY: {gap1_eval['stance_hits']}/{gap1_eval['total']} ({gap1_eval['stance_pct']:.1f}%)  |  "
                    f"SIMILARITY TRAPS CAUGHT: {gap1_eval['traps_caught']}/{gap1_eval['traps_expected']}"
                ),
                divider,
            ]
        )

    if gap2_eval is not None:
        lines.extend(
            [
                "",
                divider,
                "  GAP 2: ATOMIC CLAIM DECOMPOSITION & VERDICT AGGREGATION",
                thin_divider,
                f"  {'Case':<44} {'Props':<10} {'Expected':<21} {'Actual':<21} {'Result'}",
                thin_divider,
            ]
        )
        for row in gap2_eval["rows"]:
            prop_ratio = f"{row['actual_props']}/{row['expected_props']}"
            lines.append(
                f"  {row['name'][:43]:<44} {prop_ratio:<10} "
                f"{row['expected_verdict']:<21} {row['actual_verdict']:<21} "
                f"[{'PASS' if row['passed'] else 'FAIL'}]"
            )
        lines.extend(
            [
                thin_divider,
                (
                    f"  ATOMIC DECOMPOSITION & AGGREGATION ACCURACY: "
                    f"{gap2_eval['passed']} / {gap2_eval['total']} ({gap2_eval['accuracy_pct']:.1f}%)"
                ),
                divider,
            ]
        )

    return "\n".join(lines)


def main() -> int:
    arg_parser = argparse.ArgumentParser(
        description="Run CiteVerify decoupled retrieval/verdict benchmark and Gap 1/Gap 2 evaluation."
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
        "--live-gap-ai",
        action="store_true",
        help="Enable live Gemini embeddings and Gemini NLI during the Gap 1/Gap 2 benchmark sections.",
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
            report, timings, retrieval_meta = run_offline_pipeline(args.pdf)
            mode_label = "OFFLINE SMOKE TEST"
        else:
            report, timings, retrieval_meta = run_live_pipeline(args.pdf)
            mode_label = "LIVE PIPELINE"
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"Benchmark execution failed: {exc}", file=sys.stderr)
        return 1

    evaluation = evaluate_verdicts(report, DEMO_GROUND_TRUTH, retrieval_meta=retrieval_meta)
    use_live_gap = args.live_gap_ai and not args.offline
    gap1_eval = run_gap1_evidence_benchmark(
        use_embeddings=use_live_gap,
        use_llm_nli=use_live_gap,
    )
    gap2_eval = run_gap2_decomposition_benchmark(use_llm_decomposition=use_live_gap)

    print(
        format_benchmark_summary(
            evaluation,
            report,
            timings,
            mode_label,
            gap1_eval=gap1_eval,
            gap2_eval=gap2_eval,
        )
    )

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