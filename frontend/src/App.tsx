import React, {
  useState,
  useEffect,
  useMemo,
  useRef,
  useCallback,
} from "react";
import type { User as SupabaseUser } from "@supabase/supabase-js";
import {
  ShieldCheck,
  FileText,
  Upload,
  Search,
  ExternalLink,
  AlertTriangle,
  CheckCircle2,
  HelpCircle,
  XCircle,
  Clock,
  RefreshCw,
  Download,
  RotateCcw,
  AlertCircle,
  BookOpen,
  Loader2,
  X,
  Link2,
  Eye,
  EyeOff,
  FileUp,
  Sparkles,
  Check,
  Mail,
  Lock,
  ArrowRight,
  LogOut,
  FileSearch,
  ClipboardList,
  History,
  Settings,
  Menu,
  Trash2,
  Activity,
  Code2,
  Layers,
  ChevronRight,
  ChevronDown,
  GitBranch,
  Gauge,
  Plus,
  Server,
  Filter,
  Cloud,
  Database,
  User as UserIcon,
} from "lucide-react";
import { supabase, isSupabaseConfigured } from "./lib/supabase";

// ============================================================================
// 1. Strict Backend Contract Types (Mirrors backend/schemas.py Exactly)
// ============================================================================

export type ClaimStatus =
  | "SUPPORTED"
  | "PARTIALLY_SUPPORTED"
  | "UNSUPPORTED"
  | "CONTRADICTORY"
  | "POTENTIALLY_OUTDATED";

export type ConfidenceLevel = "HIGH" | "MEDIUM" | "LOW";

export interface RankedEvidenceItem {
  passage_id: string;
  text: string;
  bm25_score: number;
  semantic_score: number;
  hybrid_rrf_score: number;
  cross_encoder_score: number;
  logical_stance: string;
  stance_confidence: string;
  similarity_trap: boolean;
  stance_rationale: string;
}

export interface EvidenceRankingData {
  embedding_backend: string;
  nli_backend: string;
  recommended_verdict: ClaimStatus;
  similarity_trap_detected: boolean;
  top_passage_text?: string | null;
  top_bm25_score: number;
  top_semantic_score: number;
  top_rrf_score: number;
  top_cross_encoder_score: number;
  top_logical_stance: string;
  explanation: string;
  ranked_candidates: RankedEvidenceItem[];
}

export interface AtomicPropositionResult {
  proposition_id: string;
  proposition_text: string;
  verdict: ClaimStatus;
  confidence: ConfidenceLevel;
  explanation: string;
  preserved_qualifiers: string[];
  mismatch_flags: string[];
}

export interface ClaimDecompositionData {
  is_compound: boolean;
  decomposition_method: string;
  overall_verdict: ClaimStatus;
  overall_confidence: ConfidenceLevel;
  summary_explanation: string;
  propositions: AtomicPropositionResult[];
}

export interface TemporalAssessmentData {
  historical_claim_verdict: ClaimStatus;
  replaces_historical_verdict: boolean;
  temporal_status: string;
  cited_publication_year?: number | null;
  citation_age_years?: number | null;
  publication_stage: string;
  has_newer_journal_version: boolean;
  has_correction: boolean;
  is_retracted: boolean;
  subsequent_citation_count: number;
  confirming_studies_count: number;
  challenging_studies_count: number;
  systematic_reviews_count: number;
  chronological_timeline: string[];
  temporal_summary: string;
  temporal_flags: string[];
  needs_temporal_review: boolean;
}

export interface UncertaintyCalibrationData {
  action: string;
  calibrated_confidence: ConfidenceLevel;
  calibrated_probability: number;
  evidence_coverage_score: number;
  retrieval_confidence: number;
  passage_agreement_score: number;
  composite_reliability_score: number;
  has_conflicting_passages: boolean;
  abstained: boolean;
  user_message: string;
  uncertainty_flags: string[];
}

export interface ClaimVerificationResult {
  claim_id: number;
  claim_text: string;
  matched_citation?: string | null;
  evidence_passage?: string | null;
  verdict: ClaimStatus;
  explanation: string;
  confidence: ConfidenceLevel;
  uncertainty_note: string;
  needs_human_review: boolean;
  source_url?: string | null;
  mismatch_flags: string[];
  // Gaps 1-4 (Optional for backwards compatibility with older Cloud History)
  evidence_ranking?: EvidenceRankingData | null;
  claim_decomposition?: ClaimDecompositionData | null;
  temporal_assessment?: TemporalAssessmentData | null;
  uncertainty_calibration?: UncertaintyCalibrationData | null;
}

export interface VerificationReport {
  document_title: string;
  total_claims: number;
  summary_counts: Record<string, number>;
  results: ClaimVerificationResult[];
  limitations: string[];
}

// ============================================================================
// 2. Frontend Application State Types
// ============================================================================

type HealthState = "checking" | "online" | "offline";
type InputMode = "pdf" | "text";
type VerdictFilter = "ALL" | ClaimStatus;
type NavView = "verify" | "report" | "history" | "settings";
type AuthMode = "signin" | "signup";
type GapTab = "gap1" | "gap2" | "gap3" | "gap4";

interface AuthSession {
  userId: string | null;
  email: string;
  displayName: string;
  provider: "email" | "google" | "guest";
  isCloudAuth: boolean;
}

interface SessionHistoryEntry {
  id: string;
  verifiedAt: string;
  inputMethod: InputMode;
  report: VerificationReport;
  syncedToCloud: boolean;
}

interface HealthCheckDiagnostics {
  status: HealthState;
  httpCode: number | null;
  latencyMs: number | null;
  checkedAt: string | null;
  payload: unknown;
  errorMessage: string | null;
}

interface SupabaseReportRow {
  id: string;
  document_title: string;
  input_method: string;
  total_claims: number;
  report_json: unknown;
  created_at: string;
}

// ============================================================================
// 3. Semantic Design Tokens & Constants (Zero Hardcoded Verdicts)
// ============================================================================

const DEFAULT_API_URL = "https://citeverify-api.onrender.com";
const HISTORY_STORAGE_KEY = "citeverify_session_history";

const DEMO_SAMPLE_TITLE = "Clinical AI & COVID-19 Citation Integrity Benchmark";

const DEMO_SAMPLE_TEXT = `In an evaluation of retinal fundus photographs from adults with diabetes, a deep learning algorithm achieved 97.5% sensitivity and 93.4% specificity for detecting referable diabetic retinopathy on the EyePACS-1 validation dataset [1].

In hospitalized COVID-19 patients, dexamethasone reduced 28-day mortality among patients receiving invasive mechanical ventilation, while increasing overall 28-day mortality to 62.4% [2].

A multinational hospital registry analysis reported that hydroxychloroquine regimens were associated with decreased in-hospital survival and higher ventricular arrhythmia rates in COVID-19 patients [3].

An autonomous clinical transformer model reduced emergency department misdiagnosis rates by 74.2% across forty European hospitals in a 2024 multi-center audit [4].

References
[1] Gulshan V, Peng L, Coram M, et al. Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus Photographs. JAMA. 2016;316(22):2402-2410. doi:10.1001/jama.2016.17216
[2] Horby P, Lim WS, Emberson JR, et al. Dexamethasone in Hospitalized Patients with Covid-19. N Engl J Med. 2021;384(8):693-704. doi:10.1056/NEJMoa2021436
[3] Mehra MR, Desai SS, Ruschitzka F, Patel AN. RETRACTED: Hydroxychloroquine or chloroquine with or without a macrolide for treatment of COVID-19: a multinational registry analysis. The Lancet. 2020. doi:10.1016/S0140-6736(20)31180-6
[4] Vance K, Sterling R. Autonomous Clinical Transformers for Emergency Triage. European Journal of Clinical AI. 2024;12(4):101-118.`;

const ALL_STATUSES: ClaimStatus[] = [
  "SUPPORTED",
  "PARTIALLY_SUPPORTED",
  "UNSUPPORTED",
  "CONTRADICTORY",
  "POTENTIALLY_OUTDATED",
];

const VERIFICATION_PIPELINE_STEPS = [
  {
    id: 0,
    title: "Parsing & Decomposing",
    detail: "Extracting claims, citations, and atomic sub-propositions (Gap 2)",
  },
  {
    id: 1,
    title: "Retrieving & Ranking",
    detail: "Resolving DOIs + BM25/Semantic RRF & NLI reranking (Gap 1)",
  },
  {
    id: 2,
    title: "Temporal Consensus",
    detail: "Tracking retractions, versions, and citing studies (Gap 3)",
  },
  {
    id: 3,
    title: "Calibrating Uncertainty",
    detail: "Scoring coverage, agreement, and abstention policy (Gap 4)",
  },
] as const;

interface StatusMeta {
  label: string;
  shortLabel: string;
  badgeClass: string;
  dotClass: string;
  barClass: string;
  borderAccent: string;
  cardGlow: string;
  icon: React.ComponentType<{ className?: string }>;
}

const STATUS_CONFIG: Record<ClaimStatus, StatusMeta> = {
  SUPPORTED: {
    label: "Supported",
    shortLabel: "Supported",
    badgeClass: "border-emerald-500/30 bg-emerald-500/12 text-emerald-300",
    dotClass: "bg-emerald-400",
    barClass: "bg-emerald-400",
    borderAccent: "border-l-emerald-400/80",
    cardGlow: "hover:border-emerald-500/30",
    icon: CheckCircle2,
  },
  PARTIALLY_SUPPORTED: {
    label: "Partially Supported",
    shortLabel: "Partial",
    badgeClass: "border-amber-500/30 bg-amber-500/12 text-amber-300",
    dotClass: "bg-amber-400",
    barClass: "bg-amber-400",
    borderAccent: "border-l-amber-400/80",
    cardGlow: "hover:border-amber-500/30",
    icon: HelpCircle,
  },
  UNSUPPORTED: {
    label: "Unsupported",
    shortLabel: "Unsupported",
    badgeClass: "border-slate-500/35 bg-slate-500/12 text-slate-300",
    dotClass: "bg-slate-400",
    barClass: "bg-slate-400",
    borderAccent: "border-l-slate-400/75",
    cardGlow: "hover:border-slate-400/30",
    icon: AlertCircle,
  },
  CONTRADICTORY: {
    label: "Contradictory",
    shortLabel: "Contradictory",
    badgeClass: "border-rose-500/35 bg-rose-500/12 text-rose-300",
    dotClass: "bg-rose-400",
    barClass: "bg-rose-400",
    borderAccent: "border-l-rose-400/80",
    cardGlow: "hover:border-rose-500/30",
    icon: XCircle,
  },
  POTENTIALLY_OUTDATED: {
    label: "Potentially Outdated",
    shortLabel: "Outdated",
    badgeClass: "border-orange-500/35 bg-orange-500/12 text-orange-300",
    dotClass: "bg-orange-400",
    barClass: "bg-gradient-to-r from-violet-400 to-orange-400",
    borderAccent: "border-l-orange-400/80",
    cardGlow: "hover:border-orange-500/30",
    icon: Clock,
  },
};

const CONFIDENCE_CONFIG: Record<
  ConfidenceLevel,
  { label: string; className: string }
> = {
  HIGH: {
    label: "High Confidence",
    className: "border-cyan-400/30 bg-cyan-400/10 text-cyan-200",
  },
  MEDIUM: {
    label: "Medium Confidence",
    className: "border-violet-400/30 bg-violet-400/12 text-violet-200",
  },
  LOW: {
    label: "Low Confidence",
    className: "border-slate-500/30 bg-slate-500/10 text-slate-300",
  },
};

const TEMPORAL_STATUS_META: Record<
  string,
  { label: string; badgeClass: string; isAlert: boolean }
> = {
  CONSENSUS_ALIGNED: {
    label: "Consensus Aligned",
    badgeClass: "border-emerald-500/30 bg-emerald-500/12 text-emerald-300",
    isAlert: false,
  },
  SUPERSEDED_OR_CHALLENGED: {
    label: "Superseded / Challenged",
    badgeClass: "border-rose-500/35 bg-rose-500/12 text-rose-300",
    isAlert: true,
  },
  RETRACTED: {
    label: "Retracted",
    badgeClass: "border-rose-500/40 bg-rose-500/15 text-rose-200",
    isAlert: true,
  },
  CORRECTED_OR_UPDATED: {
    label: "Corrected / Updated",
    badgeClass: "border-amber-500/35 bg-amber-500/12 text-amber-200",
    isAlert: true,
  },
  STALE_NEEDS_UPDATE: {
    label: "Aging · Check Updates",
    badgeClass: "border-orange-500/35 bg-orange-500/12 text-orange-200",
    isAlert: true,
  },
  INSUFFICIENT_TEMPORAL_DATA: {
    label: "Limited Temporal Data",
    badgeClass: "border-slate-500/35 bg-slate-500/12 text-slate-300",
    isAlert: false,
  },
};

const CALIBRATION_ACTION_META: Record<
  string,
  { label: string; badgeClass: string }
> = {
  PRESENT_VERDICT: {
    label: "Auto-Verified",
    badgeClass: "border-emerald-500/30 bg-emerald-500/12 text-emerald-300",
  },
  FLAG_FOR_REVIEW: {
    label: "Flag for Review",
    badgeClass: "border-amber-500/35 bg-amber-500/12 text-amber-200",
  },
  ABSTAIN: {
    label: "Abstained",
    badgeClass: "border-fuchsia-500/35 bg-fuchsia-500/12 text-fuchsia-200",
  },
  MARK_UNVERIFIABLE: {
    label: "Unverifiable",
    badgeClass: "border-slate-500/35 bg-slate-500/12 text-slate-300",
  },
};

// ============================================================================
// 4. Environment, Normalization, Auth & Storage Helpers
// ============================================================================

function resolveEnvApiUrl(): string {
  try {
    const envObj = (
      import.meta as unknown as { env?: Record<string, string | undefined> }
    )?.env;
    const raw = envObj?.VITE_API_URL;
    if (typeof raw === "string" && raw.trim().length > 0) {
      return raw.trim().replace(/\/+$/, "");
    }
  } catch {
    // Fallback cleanly when import.meta.env is unavailable
  }
  return DEFAULT_API_URL;
}

function asVerdict(val: unknown): ClaimStatus {
  const s = String(val ?? "UNSUPPORTED").toUpperCase();
  return (ALL_STATUSES as string[]).includes(s)
    ? (s as ClaimStatus)
    : "UNSUPPORTED";
}

function asConfidence(val: unknown): ConfidenceLevel {
  const s = String(val ?? "LOW").toUpperCase();
  return s === "HIGH" || s === "MEDIUM" || s === "LOW" ? s : "LOW";
}

function asNum(val: unknown, fallback = 0): number {
  const n = Number(val);
  return Number.isFinite(n) ? n : fallback;
}

function asStrArray(val: unknown): string[] {
  return Array.isArray(val)
    ? val.filter((item): item is string => typeof item === "string")
    : [];
}

function normalizeEvidenceRanking(raw: unknown): EvidenceRankingData | null {
  if (!raw || typeof raw !== "object") return null;
  const o = raw as Record<string, unknown>;
  const rawCands = Array.isArray(o.ranked_candidates)
    ? o.ranked_candidates
    : [];
  const ranked_candidates: RankedEvidenceItem[] = rawCands
    .filter((c): c is Record<string, unknown> =>
      Boolean(c && typeof c === "object"),
    )
    .map((c, i) => ({
      passage_id:
        typeof c.passage_id === "string" ? c.passage_id : `p_${i + 1}`,
      text: typeof c.text === "string" ? c.text : "",
      bm25_score: asNum(c.bm25_score),
      semantic_score: asNum(c.semantic_score),
      hybrid_rrf_score: asNum(c.hybrid_rrf_score),
      cross_encoder_score: asNum(c.cross_encoder_score),
      logical_stance:
        typeof c.logical_stance === "string" ? c.logical_stance : "NEUTRAL",
      stance_confidence:
        typeof c.stance_confidence === "string" ? c.stance_confidence : "LOW",
      similarity_trap: Boolean(c.similarity_trap),
      stance_rationale:
        typeof c.stance_rationale === "string" ? c.stance_rationale : "",
    }));

  return {
    embedding_backend:
      typeof o.embedding_backend === "string"
        ? o.embedding_backend
        : "tfidf_fallback",
    nli_backend:
      typeof o.nli_backend === "string" ? o.nli_backend : "rule_based_nli",
    recommended_verdict: asVerdict(o.recommended_verdict),
    similarity_trap_detected: Boolean(o.similarity_trap_detected),
    top_passage_text:
      typeof o.top_passage_text === "string" ? o.top_passage_text : null,
    top_bm25_score: asNum(o.top_bm25_score),
    top_semantic_score: asNum(o.top_semantic_score),
    top_rrf_score: asNum(o.top_rrf_score),
    top_cross_encoder_score: asNum(o.top_cross_encoder_score),
    top_logical_stance:
      typeof o.top_logical_stance === "string"
        ? o.top_logical_stance
        : "NEUTRAL",
    explanation: typeof o.explanation === "string" ? o.explanation : "",
    ranked_candidates,
  };
}

function normalizeClaimDecomposition(
  raw: unknown,
): ClaimDecompositionData | null {
  if (!raw || typeof raw !== "object") return null;
  const o = raw as Record<string, unknown>;
  const rawProps = Array.isArray(o.propositions) ? o.propositions : [];
  const propositions: AtomicPropositionResult[] = rawProps
    .filter((p): p is Record<string, unknown> =>
      Boolean(p && typeof p === "object"),
    )
    .map((p, i) => ({
      proposition_id:
        typeof p.proposition_id === "string" ? p.proposition_id : `1.${i + 1}`,
      proposition_text:
        typeof p.proposition_text === "string" ? p.proposition_text : "",
      verdict: asVerdict(p.verdict),
      confidence: asConfidence(p.confidence),
      explanation: typeof p.explanation === "string" ? p.explanation : "",
      preserved_qualifiers: asStrArray(p.preserved_qualifiers),
      mismatch_flags: asStrArray(p.mismatch_flags),
    }));

  return {
    is_compound: Boolean(o.is_compound),
    decomposition_method:
      typeof o.decomposition_method === "string"
        ? o.decomposition_method
        : "rule_based",
    overall_verdict: asVerdict(o.overall_verdict),
    overall_confidence: asConfidence(o.overall_confidence),
    summary_explanation:
      typeof o.summary_explanation === "string" ? o.summary_explanation : "",
    propositions,
  };
}

function normalizeTemporalAssessment(
  raw: unknown,
): TemporalAssessmentData | null {
  if (!raw || typeof raw !== "object") return null;
  const o = raw as Record<string, unknown>;
  return {
    historical_claim_verdict: asVerdict(o.historical_claim_verdict),
    replaces_historical_verdict: Boolean(o.replaces_historical_verdict),
    temporal_status:
      typeof o.temporal_status === "string"
        ? o.temporal_status
        : "INSUFFICIENT_TEMPORAL_DATA",
    cited_publication_year:
      typeof o.cited_publication_year === "number"
        ? o.cited_publication_year
        : null,
    citation_age_years:
      typeof o.citation_age_years === "number" ? o.citation_age_years : null,
    publication_stage:
      typeof o.publication_stage === "string" ? o.publication_stage : "UNKNOWN",
    has_newer_journal_version: Boolean(o.has_newer_journal_version),
    has_correction: Boolean(o.has_correction),
    is_retracted: Boolean(o.is_retracted),
    subsequent_citation_count: asNum(o.subsequent_citation_count),
    confirming_studies_count: asNum(o.confirming_studies_count),
    challenging_studies_count: asNum(o.challenging_studies_count),
    systematic_reviews_count: asNum(o.systematic_reviews_count),
    chronological_timeline: asStrArray(o.chronological_timeline),
    temporal_summary:
      typeof o.temporal_summary === "string" ? o.temporal_summary : "",
    temporal_flags: asStrArray(o.temporal_flags),
    needs_temporal_review: Boolean(o.needs_temporal_review),
  };
}

function normalizeUncertaintyCalibration(
  raw: unknown,
): UncertaintyCalibrationData | null {
  if (!raw || typeof raw !== "object") return null;
  const o = raw as Record<string, unknown>;
  return {
    action: typeof o.action === "string" ? o.action : "FLAG_FOR_REVIEW",
    calibrated_confidence: asConfidence(o.calibrated_confidence),
    calibrated_probability: asNum(o.calibrated_probability),
    evidence_coverage_score: asNum(o.evidence_coverage_score),
    retrieval_confidence: asNum(o.retrieval_confidence),
    passage_agreement_score: asNum(o.passage_agreement_score, 0.5),
    composite_reliability_score: asNum(o.composite_reliability_score),
    has_conflicting_passages: Boolean(o.has_conflicting_passages),
    abstained: Boolean(o.abstained),
    user_message: typeof o.user_message === "string" ? o.user_message : "",
    uncertainty_flags: asStrArray(o.uncertainty_flags),
  };
}

/**
 * Normalizes a live /api/verify JSON payload strictly against VerificationReport (including Gaps 1-4).
 */
function normalizeReport(raw: unknown): VerificationReport {
  const data = (raw && typeof raw === "object" ? raw : {}) as Record<
    string,
    unknown
  >;
  const rawResults = Array.isArray(data.results) ? data.results : [];

  const results: ClaimVerificationResult[] = rawResults.map((entry, idx) => {
    const r = (entry && typeof entry === "object" ? entry : {}) as Record<
      string,
      unknown
    >;

    return {
      claim_id: typeof r.claim_id === "number" ? r.claim_id : idx + 1,
      claim_text: typeof r.claim_text === "string" ? r.claim_text : "",
      matched_citation:
        typeof r.matched_citation === "string" ? r.matched_citation : null,
      evidence_passage:
        typeof r.evidence_passage === "string" ? r.evidence_passage : null,
      verdict: asVerdict(r.verdict),
      explanation: typeof r.explanation === "string" ? r.explanation : "",
      confidence: asConfidence(r.confidence),
      uncertainty_note:
        typeof r.uncertainty_note === "string" ? r.uncertainty_note : "",
      needs_human_review:
        typeof r.needs_human_review === "boolean" ? r.needs_human_review : true,
      source_url: typeof r.source_url === "string" ? r.source_url : null,
      mismatch_flags: asStrArray(r.mismatch_flags),
      evidence_ranking: normalizeEvidenceRanking(r.evidence_ranking),
      claim_decomposition: normalizeClaimDecomposition(r.claim_decomposition),
      temporal_assessment: normalizeTemporalAssessment(r.temporal_assessment),
      uncertainty_calibration: normalizeUncertaintyCalibration(
        r.uncertainty_calibration,
      ),
    };
  });

  const rawSummary =
    data.summary_counts && typeof data.summary_counts === "object"
      ? (data.summary_counts as Record<string, unknown>)
      : {};

  const summary_counts: Record<string, number> = {};
  for (const [key, val] of Object.entries(rawSummary)) {
    const num = Number(val);
    if (!Number.isNaN(num)) {
      summary_counts[key] = num;
    }
  }

  return {
    document_title:
      typeof data.document_title === "string" &&
      data.document_title.trim().length > 0
        ? data.document_title
        : "Untitled Research Document",
    total_claims:
      typeof data.total_claims === "number"
        ? data.total_claims
        : results.length,
    summary_counts,
    results,
    limitations: asStrArray(data.limitations),
  };
}

function computeStatusCounts(
  report: VerificationReport,
): Record<ClaimStatus, number> {
  const counts: Record<ClaimStatus, number> = {
    SUPPORTED: 0,
    PARTIALLY_SUPPORTED: 0,
    UNSUPPORTED: 0,
    CONTRADICTORY: 0,
    POTENTIALLY_OUTDATED: 0,
  };

  const hasBackendSummary = Object.keys(report.summary_counts).length > 0;
  if (hasBackendSummary) {
    for (const status of ALL_STATUSES) {
      counts[status] = Number(report.summary_counts[status] ?? 0);
    }
  } else {
    for (const item of report.results) {
      if (item.verdict in counts) {
        counts[item.verdict] += 1;
      }
    }
  }
  return counts;
}

function mapSupabaseUser(user: SupabaseUser): AuthSession {
  const email = user.email || "researcher@citeverify.app";
  const metadata = (user.user_metadata || {}) as Record<string, unknown>;
  const fullName =
    typeof metadata.full_name === "string" &&
    metadata.full_name.trim().length > 0
      ? metadata.full_name.trim()
      : typeof metadata.name === "string" && metadata.name.trim().length > 0
        ? metadata.name.trim()
        : deriveDisplayName(email);

  const rawProvider = String(
    user.app_metadata?.provider || "email",
  ).toLowerCase();

  return {
    userId: user.id,
    email,
    displayName: fullName,
    provider: rawProvider === "google" ? "google" : "email",
    isCloudAuth: true,
  };
}

function loadLocalSessionHistory(): SessionHistoryEntry[] {
  try {
    const raw = window.localStorage.getItem(HISTORY_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter((item): item is SessionHistoryEntry =>
        Boolean(
          item &&
          typeof item.id === "string" &&
          item.report &&
          Array.isArray(item.report.results),
        ),
      )
      .map((item) => ({
        ...item,
        report: normalizeReport(item.report),
        syncedToCloud: Boolean(item.syncedToCloud),
      }));
  } catch {
    return [];
  }
}

function saveLocalSessionHistory(entries: SessionHistoryEntry[]): void {
  try {
    window.localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(entries));
  } catch {
    // Ignore storage quota or private browsing restrictions
  }
}

function deriveDisplayName(email: string): string {
  const localPart = email.split("@")[0] || "Researcher";
  return localPart
    .replace(/[._-]+/g, " ")
    .replace(/\b\w/g, (ch) => ch.toUpperCase())
    .trim();
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function formatTimestamp(iso: string): string {
  try {
    return new Date(iso).toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

function formatPercent(val: number): string {
  return `${Math.round(Math.min(1, Math.max(0, val)) * 100)}%`;
}

function formatStageLabel(stage: string): string {
  return stage
    .toLowerCase()
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

// ============================================================================
// 5. Built-In Icon Branding & Official Google "G" SVG (Zero External Images)
// ============================================================================

const BrandBadge: React.FC<{ size?: "sm" | "md" | "lg" }> = ({
  size = "md",
}) => {
  const dimensions =
    size === "lg"
      ? "h-14 w-14 rounded-2xl"
      : size === "sm"
        ? "h-8 w-8 rounded-lg"
        : "h-9 w-9 rounded-xl";
  const iconSize =
    size === "lg" ? "h-7 w-7" : size === "sm" ? "h-4 w-4" : "h-5 w-5";

  return (
    <div
      className={`relative inline-flex shrink-0 items-center justify-center ${dimensions} bg-gradient-to-br from-violet-500 via-indigo-500 to-fuchsia-500 p-[1px] shadow-[0_0_28px_-6px_rgba(139,92,246,0.75)]`}
    >
      <div
        className={`flex h-full w-full items-center justify-center ${dimensions} bg-gradient-to-br from-[#1b1238] via-[#130f2b] to-[#0d1428] text-violet-200`}
      >
        <ShieldCheck
          className={`${iconSize} text-cyan-300 drop-shadow-[0_0_8px_rgba(34,211,238,0.5)]`}
        />
      </div>
    </div>
  );
};

const GoogleIcon: React.FC<{ className?: string }> = ({
  className = "h-4 w-4",
}) => (
  <svg viewBox="0 0 24 24" aria-hidden="true" className={className}>
    <path
      fill="#4285F4"
      d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
    />
    <path
      fill="#34A853"
      d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
    />
    <path
      fill="#FBBC05"
      d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
    />
    <path
      fill="#EA4335"
      d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
    />
  </svg>
);

// ============================================================================
// 6. Aurora Violet Login & Sign-Up Screen (Supabase Auth + Instant Guest Demo)
// ============================================================================

interface AuroraLoginScreenProps {
  health: HealthCheckDiagnostics;
  onRefreshHealth: () => void;
  onLoginSuccess: (session: AuthSession) => void;
}

const AuroraLoginScreen: React.FC<AuroraLoginScreenProps> = ({
  health,
  onRefreshHealth,
  onLoginSuccess,
}) => {
  const [authMode, setAuthMode] = useState<AuthMode>("signin");
  const [fullName, setFullName] = useState<string>("");
  const [email, setEmail] = useState<string>("");
  const [password, setPassword] = useState<string>("");
  const [showPassword, setShowPassword] = useState<boolean>(false);
  const [formNotice, setFormNotice] = useState<{
    type: "error" | "info" | "success";
    text: string;
  } | null>(null);
  const [submittingVia, setSubmittingVia] = useState<"email" | "google" | null>(
    null,
  );

  const handleEmailAuth = async (e: React.FormEvent) => {
    e.preventDefault();
    setFormNotice(null);

    const cleanedEmail = email.trim();
    if (!cleanedEmail || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(cleanedEmail)) {
      setFormNotice({
        type: "error",
        text: "Please enter a valid email address.",
      });
      return;
    }

    if (password.length < 6) {
      setFormNotice({
        type: "error",
        text: "Please enter a password with at least 6 characters.",
      });
      return;
    }

    setSubmittingVia("email");

    if (!supabase || !isSupabaseConfigured) {
      window.setTimeout(() => {
        setSubmittingVia(null);
        onLoginSuccess({
          userId: null,
          email: cleanedEmail,
          displayName: fullName.trim() || deriveDisplayName(cleanedEmail),
          provider: "email",
          isCloudAuth: false,
        });
      }, 350);
      return;
    }

    try {
      if (authMode === "signup") {
        const displayName = fullName.trim() || deriveDisplayName(cleanedEmail);
        const { data, error } = await supabase.auth.signUp({
          email: cleanedEmail,
          password,
          options: {
            data: { full_name: displayName },
            emailRedirectTo: window.location.origin,
          },
        });

        if (error) {
          throw error;
        }

        if (data.session?.user) {
          onLoginSuccess(mapSupabaseUser(data.session.user));
          return;
        }

        setFormNotice({
          type: "success",
          text: "Account created! Check your email inbox to confirm your address, then sign in.",
        });
        setAuthMode("signin");
      } else {
        const { data, error } = await supabase.auth.signInWithPassword({
          email: cleanedEmail,
          password,
        });

        if (error) {
          throw error;
        }

        if (data.user) {
          onLoginSuccess(mapSupabaseUser(data.user));
        }
      }
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : "Authentication failed. Please check your credentials and try again.";
      setFormNotice({ type: "error", text: message });
    } finally {
      setSubmittingVia(null);
    }
  };

  const handleGoogleSignIn = async () => {
    setFormNotice(null);
    setSubmittingVia("google");

    if (!supabase || !isSupabaseConfigured) {
      window.setTimeout(() => {
        setSubmittingVia(null);
        onLoginSuccess({
          userId: null,
          email: "google.account@citeverify.app",
          displayName: "Google Researcher",
          provider: "google",
          isCloudAuth: false,
        });
      }, 350);
      return;
    }

    try {
      const { error } = await supabase.auth.signInWithOAuth({
        provider: "google",
        options: {
          redirectTo: window.location.origin,
        },
      });
      if (error) {
        throw error;
      }
    } catch (err) {
      const rawMsg = err instanceof Error ? err.message : "";
      const friendlyMsg = rawMsg
        .toLowerCase()
        .includes("provider is not enabled")
        ? "Google OAuth is not enabled in Supabase yet. Sign in or create an account with Email & Password above, or continue as Guest below."
        : rawMsg || "Unable to start Google sign-in right now.";
      setFormNotice({ type: "error", text: friendlyMsg });
      setSubmittingVia(null);
    }
  };

  const handleGuestDemo = () => {
    setFormNotice(null);
    onLoginSuccess({
      userId: null,
      email: "guest@citeverify.demo",
      displayName: "Guest Reviewer",
      provider: "guest",
      isCloudAuth: false,
    });
  };

  const healthLabel =
    health.status === "online"
      ? `API Online${health.latencyMs !== null ? ` · ${health.latencyMs}ms` : ""}`
      : health.status === "checking"
        ? "Checking API..."
        : "API Waking / Offline";

  return (
    <div className="relative flex min-h-screen w-full flex-col justify-between overflow-hidden bg-[#070711] text-slate-100 selection:bg-violet-500/30 selection:text-violet-100">
      <div
        aria-hidden="true"
        className="pointer-events-none fixed inset-0 overflow-hidden"
      >
        <div className="absolute -top-48 left-1/2 h-[560px] w-[980px] -translate-x-1/2 rounded-full bg-[radial-gradient(ellipse_at_center,_rgba(139,92,246,0.28)_0%,_rgba(99,102,241,0.16)_38%,_transparent_72%)] blur-3xl" />
        <div className="absolute -right-28 top-12 h-[420px] w-[480px] rounded-full bg-[radial-gradient(circle,_rgba(34,211,238,0.14)_0%,_rgba(168,85,247,0.1)_45%,_transparent_72%)] blur-3xl" />
        <div className="absolute -bottom-36 -left-24 h-[460px] w-[540px] rounded-full bg-[radial-gradient(circle,_rgba(217,70,239,0.14)_0%,_rgba(99,102,241,0.14)_45%,_transparent_72%)] blur-3xl" />
        <div
          className="absolute inset-0 opacity-[0.035]"
          style={{
            backgroundImage:
              "linear-gradient(to right, rgba(255,255,255,0.6) 1px, transparent 1px), linear-gradient(to bottom, rgba(255,255,255,0.6) 1px, transparent 1px)",
            backgroundSize: "52px 52px",
            maskImage:
              "radial-gradient(ellipse at 50% 35%, black 15%, transparent 72%)",
            WebkitMaskImage:
              "radial-gradient(ellipse at 50% 35%, black 15%, transparent 72%)",
          }}
        />
      </div>

      <header className="relative z-10 mx-auto flex w-full max-w-6xl items-center justify-between px-5 pt-6 sm:px-8">
        <div className="flex items-center gap-3">
          <BrandBadge size="md" />
          <div>
            <span className="block text-base font-semibold tracking-tight text-white leading-none">
              CiteVerify
            </span>
            <span className="mt-0.5 block text-[11px] tracking-wide text-slate-400">
              Citation Intelligence
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {isSupabaseConfigured && (
            <span className="hidden items-center gap-1.5 rounded-full border border-emerald-500/25 bg-emerald-500/10 px-3 py-1.5 text-xs font-medium text-emerald-300 sm:inline-flex">
              <Cloud className="h-3.5 w-3.5" />
              Cloud Auth Ready
            </span>
          )}
          <button
            type="button"
            onClick={onRefreshHealth}
            title="Ping live backend /health endpoint"
            className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-[#0d0c1d]/75 px-3.5 py-1.5 text-xs font-medium text-slate-300 backdrop-blur-xl transition-colors hover:border-violet-400/40 hover:text-white"
          >
            <span className="relative flex h-2 w-2">
              {health.status === "online" && (
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-60" />
              )}
              <span
                className={`relative inline-flex h-2 w-2 rounded-full ${
                  health.status === "online"
                    ? "bg-emerald-400"
                    : health.status === "checking"
                      ? "bg-amber-400"
                      : "bg-rose-400"
                }`}
              />
            </span>
            <span>{healthLabel}</span>
          </button>
        </div>
      </header>

      <main className="relative z-10 flex flex-1 items-center justify-center px-4 py-10">
        <div className="w-full max-w-md">
          <div className="relative rounded-3xl border border-white/10 bg-[#0d0c1d]/80 p-7 sm:p-9 shadow-[0_28px_90px_-18px_rgba(0,0,0,0.9),0_0_65px_-22px_rgba(139,92,246,0.35)] ring-1 ring-violet-400/15 backdrop-blur-2xl">
            <div
              aria-hidden="true"
              className="pointer-events-none absolute inset-x-10 top-0 h-px bg-gradient-to-r from-transparent via-violet-400/70 to-transparent"
            />

            <div className="mb-6 text-center">
              <div className="mb-4 flex justify-center">
                <BrandBadge size="lg" />
              </div>
              <h1 className="text-2xl font-semibold tracking-tight text-white">
                {authMode === "signin"
                  ? "Sign in to CiteVerify"
                  : "Create your account"}
              </h1>
              <p className="mt-1.5 text-sm text-slate-400">
                {authMode === "signin"
                  ? "Access your scientific claim & citation verification workbench"
                  : "Sync verified manuscripts and audit reports in the cloud"}
              </p>
            </div>

            <div className="mb-5 grid grid-cols-2 rounded-xl border border-white/10 bg-black/35 p-1">
              <button
                type="button"
                onClick={() => {
                  setAuthMode("signin");
                  setFormNotice(null);
                }}
                className={`rounded-lg py-2 text-xs font-medium transition-all ${
                  authMode === "signin"
                    ? "bg-gradient-to-r from-violet-600/90 to-indigo-600/90 text-white shadow-sm"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                Sign In
              </button>
              <button
                type="button"
                onClick={() => {
                  setAuthMode("signup");
                  setFormNotice(null);
                }}
                className={`rounded-lg py-2 text-xs font-medium transition-all ${
                  authMode === "signup"
                    ? "bg-gradient-to-r from-violet-600/90 to-indigo-600/90 text-white shadow-sm"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                Create Account
              </button>
            </div>

            {formNotice && (
              <div
                role="alert"
                className={`mb-5 flex items-start gap-2.5 rounded-xl border px-3.5 py-2.5 text-xs leading-relaxed ${
                  formNotice.type === "error"
                    ? "border-rose-500/35 bg-rose-500/10 text-rose-200"
                    : formNotice.type === "success"
                      ? "border-emerald-500/35 bg-emerald-500/10 text-emerald-200"
                      : "border-violet-400/35 bg-violet-500/10 text-violet-200"
                }`}
              >
                {formNotice.type === "success" ? (
                  <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-400" />
                ) : (
                  <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                )}
                <span>{formNotice.text}</span>
              </div>
            )}

            <form onSubmit={handleEmailAuth} className="space-y-4" noValidate>
              {authMode === "signup" && (
                <div className="space-y-1.5">
                  <label
                    htmlFor="signup-name"
                    className="block text-xs font-medium tracking-wide text-slate-300"
                  >
                    Full name <span className="text-slate-500">(optional)</span>
                  </label>
                  <div className="group relative">
                    <UserIcon className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500 transition-colors group-focus-within:text-violet-400" />
                    <input
                      id="signup-name"
                      type="text"
                      autoComplete="name"
                      value={fullName}
                      onChange={(e) => setFullName(e.target.value)}
                      placeholder="Dr. Ada Lovelace"
                      className="w-full rounded-xl border border-white/10 bg-white/[0.04] py-2.5 pl-10 pr-4 text-sm text-white placeholder:text-slate-500 transition-all focus:border-violet-400/75 focus:bg-white/[0.06] focus:outline-none focus:ring-4 focus:ring-violet-500/15"
                    />
                  </div>
                </div>
              )}

              <div className="space-y-1.5">
                <label
                  htmlFor="login-email"
                  className="block text-xs font-medium tracking-wide text-slate-300"
                >
                  Email address
                </label>
                <div className="group relative">
                  <Mail className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500 transition-colors group-focus-within:text-violet-400" />
                  <input
                    id="login-email"
                    type="email"
                    autoComplete="email"
                    value={email}
                    onChange={(e) => {
                      setEmail(e.target.value);
                      if (formNotice) setFormNotice(null);
                    }}
                    placeholder="researcher@university.edu"
                    className="w-full rounded-xl border border-white/10 bg-white/[0.04] py-2.5 pl-10 pr-4 text-sm text-white placeholder:text-slate-500 transition-all focus:border-violet-400/75 focus:bg-white/[0.06] focus:outline-none focus:ring-4 focus:ring-violet-500/15"
                  />
                </div>
              </div>

              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <label
                    htmlFor="login-password"
                    className="block text-xs font-medium tracking-wide text-slate-300"
                  >
                    Password
                  </label>
                  {authMode === "signin" && (
                    <button
                      type="button"
                      onClick={() =>
                        setFormNotice({
                          type: "info",
                          text: isSupabaseConfigured
                            ? 'New here? Switch to "Create Account" above, or click "Continue as Guest" below for instant demo access.'
                            : "Enter any valid email and 6+ character password, or click Continue as Guest below.",
                        })
                      }
                      className="text-xs font-medium text-violet-300/90 transition-colors hover:text-violet-200"
                    >
                      Need an account?
                    </button>
                  )}
                </div>
                <div className="group relative">
                  <Lock className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500 transition-colors group-focus-within:text-violet-400" />
                  <input
                    id="login-password"
                    type={showPassword ? "text" : "password"}
                    autoComplete={
                      authMode === "signin"
                        ? "current-password"
                        : "new-password"
                    }
                    value={password}
                    onChange={(e) => {
                      setPassword(e.target.value);
                      if (formNotice) setFormNotice(null);
                    }}
                    placeholder="••••••••"
                    className="w-full rounded-xl border border-white/10 bg-white/[0.04] py-2.5 pl-10 pr-11 text-sm text-white placeholder:text-slate-500 transition-all focus:border-violet-400/75 focus:bg-white/[0.06] focus:outline-none focus:ring-4 focus:ring-violet-500/15"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword((prev) => !prev)}
                    aria-label={
                      showPassword ? "Hide password" : "Show password"
                    }
                    className="absolute right-3 top-1/2 -translate-y-1/2 rounded-lg p-1 text-slate-500 transition-colors hover:text-slate-200"
                  >
                    {showPassword ? (
                      <EyeOff className="h-4 w-4" />
                    ) : (
                      <Eye className="h-4 w-4" />
                    )}
                  </button>
                </div>
              </div>

              <button
                type="submit"
                disabled={submittingVia !== null}
                className="group relative mt-2 flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-violet-600 via-indigo-600 to-fuchsia-600 px-4 py-2.5 text-sm font-medium text-white shadow-[0_12px_32px_-10px_rgba(139,92,246,0.8)] transition-all hover:from-violet-500 hover:via-indigo-500 hover:to-fuchsia-500 hover:shadow-[0_14px_38px_-8px_rgba(168,85,247,0.9)] active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-65"
              >
                {submittingVia === "email" ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    <span>
                      {authMode === "signin"
                        ? "Signing in..."
                        : "Creating account..."}
                    </span>
                  </>
                ) : (
                  <>
                    <span>
                      {authMode === "signin" ? "Sign In" : "Create Account"}
                    </span>
                    <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
                  </>
                )}
              </button>
            </form>

            <div className="relative my-6 flex items-center justify-center">
              <div className="h-px flex-1 bg-gradient-to-r from-transparent via-white/12 to-white/12" />
              <span className="px-3 text-[11px] font-medium uppercase tracking-widest text-slate-500">
                or continue with
              </span>
              <div className="h-px flex-1 bg-gradient-to-l from-transparent via-white/12 to-white/12" />
            </div>

            <button
              type="button"
              onClick={handleGoogleSignIn}
              disabled={submittingVia !== null}
              className="flex w-full items-center justify-center gap-3 rounded-xl border border-white/12 bg-white/[0.04] px-4 py-2.5 text-sm font-medium text-slate-100 transition-all hover:border-violet-400/45 hover:bg-white/[0.08] active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-65"
            >
              {submittingVia === "google" ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin text-violet-300" />
                  <span>Connecting Google...</span>
                </>
              ) : (
                <>
                  <GoogleIcon className="h-4 w-4" />
                  <span>Continue with Google</span>
                </>
              )}
            </button>

            {/* Instant Guest Demo Bypass for Evaluators & Reviewers */}
            <button
              type="button"
              onClick={handleGuestDemo}
              disabled={submittingVia !== null}
              className="mt-3 flex w-full items-center justify-center gap-2 rounded-xl border border-cyan-400/30 bg-cyan-500/10 px-4 py-2.5 text-sm font-medium text-cyan-200 transition-all hover:border-cyan-300/60 hover:bg-cyan-500/20 hover:text-white active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-65"
            >
              <Sparkles className="h-4 w-4 text-cyan-300" />
              <span>Continue as Guest (Instant Demo)</span>
            </button>
            <p className="mt-2 text-center text-[11px] text-slate-500">
              No account needed — test live PDF/text verification immediately
            </p>
          </div>
        </div>
      </main>

      <footer className="relative z-10 pb-6 text-center text-xs text-slate-500">
        CiteVerify · Scientific Claim & Citation Verification Platform
      </footer>
    </div>
  );
};

// ============================================================================
// 7. Claim Verification Card (Report Viewer + Interactive Gaps 1-4 Inspector)
// ============================================================================

const MetricBar: React.FC<{
  label: string;
  value: number;
  colorClass: string;
}> = ({ label, value, colorClass }) => {
  const clamped = Math.min(1, Math.max(0, value));
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-[11px]">
        <span className="text-slate-400">{label}</span>
        <span className="font-mono font-semibold text-slate-200">
          {formatPercent(clamped)}
        </span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/[0.07]">
        <div
          style={{ width: `${Math.round(clamped * 100)}%` }}
          className={`h-full rounded-full transition-all ${colorClass}`}
        />
      </div>
    </div>
  );
};

const ClaimVerificationCard: React.FC<{ item: ClaimVerificationResult }> = ({
  item,
}) => {
  const statusMeta = STATUS_CONFIG[item.verdict] ?? STATUS_CONFIG.UNSUPPORTED;
  const confidenceMeta =
    CONFIDENCE_CONFIG[item.confidence] ?? CONFIDENCE_CONFIG.LOW;
  const StatusIcon = statusMeta.icon;

  const [openGapTab, setOpenGapTab] = useState<GapTab | null>(null);

  const g1 = item.evidence_ranking ?? null;
  const g2 = item.claim_decomposition ?? null;
  const g3 = item.temporal_assessment ?? null;
  const g4 = item.uncertainty_calibration ?? null;
  const hasAnyGapData = Boolean(g1 || g2 || g3 || g4);

  const temporalMeta = g3
    ? (TEMPORAL_STATUS_META[g3.temporal_status] ??
      TEMPORAL_STATUS_META.INSUFFICIENT_TEMPORAL_DATA)
    : null;
  const actionMeta = g4
    ? (CALIBRATION_ACTION_META[g4.action] ??
      CALIBRATION_ACTION_META.FLAG_FOR_REVIEW)
    : null;

  const toggleGapTab = (tab: GapTab) => {
    setOpenGapTab((prev) => (prev === tab ? null : tab));
  };

  return (
    <article
      className={`rounded-2xl border border-white/10 border-l-2 ${statusMeta.borderAccent} ${statusMeta.cardGlow} bg-[#0d0c1d]/80 p-5 sm:p-6 backdrop-blur-xl transition-all`}
    >
      {/* Card Metadata Header */}
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2.5">
        <div className="flex flex-wrap items-center gap-2">
          <span className="inline-flex items-center rounded-md border border-white/10 bg-white/[0.05] px-2 py-0.5 font-mono text-xs text-slate-300">
            #{item.claim_id}
          </span>
          <span
            className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium ${statusMeta.badgeClass}`}
          >
            <StatusIcon className="h-3.5 w-3.5" />
            {statusMeta.label}
          </span>
          <span
            className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium ${confidenceMeta.className}`}
          >
            {confidenceMeta.label}
          </span>

          {/* Live Gap 4 Calibrated Probability & Policy Badge */}
          {g4 && actionMeta && (
            <span
              className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-medium ${actionMeta.badgeClass}`}
            >
              <Gauge className="h-3 w-3" />
              {formatPercent(g4.calibrated_probability)} · {actionMeta.label}
            </span>
          )}

          {/* Live Gap 1 Similarity Trap Badge */}
          {g1?.similarity_trap_detected && (
            <span className="inline-flex items-center gap-1 rounded-full border border-rose-500/40 bg-rose-500/15 px-2.5 py-0.5 text-xs font-medium text-rose-200">
              <AlertTriangle className="h-3 w-3" />
              Similarity Trap
            </span>
          )}

          {/* Live Gap 2 Compound Claim Badge */}
          {g2?.is_compound && (
            <span className="inline-flex items-center gap-1 rounded-full border border-violet-400/35 bg-violet-500/15 px-2.5 py-0.5 text-xs font-medium text-violet-200">
              <GitBranch className="h-3 w-3" />
              Compound ({g2.propositions.length} Parts)
            </span>
          )}

          {/* Live Gap 3 Temporal Consensus Badge */}
          {g3 && temporalMeta && (
            <span
              className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-medium ${temporalMeta.badgeClass}`}
            >
              <Clock className="h-3 w-3" />
              {temporalMeta.label}
            </span>
          )}
        </div>

        {item.needs_human_review && (
          <span className="inline-flex items-center gap-1.5 rounded-full border border-amber-500/35 bg-amber-500/12 px-2.5 py-0.5 text-xs font-medium text-amber-300">
            <AlertTriangle className="h-3.5 w-3.5" />
            Needs Human Review
          </span>
        )}
      </div>

      {/* Prominent Claim Text */}
      <h3 className="mb-4 text-[15px] font-medium leading-relaxed text-white sm:text-base">
        {item.claim_text}
      </h3>

      {/* Gap 4 Abstention / Unverifiable Callout Banner */}
      {g4?.abstained && g4.user_message && (
        <div className="mb-4 flex items-start gap-2.5 rounded-xl border border-fuchsia-500/30 bg-fuchsia-500/[0.08] px-3.5 py-2.5 text-xs leading-relaxed text-fuchsia-100">
          <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-fuchsia-300" />
          <div>
            <span className="font-semibold text-fuchsia-200">
              Uncertainty Guard ({actionMeta?.label ?? g4.action}):{" "}
            </span>
            {g4.user_message}
          </div>
        </div>
      )}

      {/* Clean 2-Column Detail Grid */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {/* Left Column: Explanation, Uncertainty Note, Mismatch Flags */}
        <div className="flex flex-col justify-between space-y-4 rounded-xl border border-white/[0.08] bg-black/25 p-4">
          <div className="space-y-3">
            <div>
              <span className="mb-1.5 block text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                Verification Explanation
              </span>
              <p className="text-sm leading-relaxed text-slate-300">
                {item.explanation || (
                  <span className="italic text-slate-500">
                    No explanation provided.
                  </span>
                )}
              </p>
            </div>

            {item.uncertainty_note &&
              item.uncertainty_note.trim().length > 0 && (
                <div className="flex items-start gap-2.5 rounded-lg border border-amber-500/20 bg-amber-500/[0.06] px-3 py-2.5 text-xs leading-relaxed text-slate-300">
                  <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-400" />
                  <div>
                    <span className="font-semibold text-amber-200">
                      Uncertainty Note:{" "}
                    </span>
                    {item.uncertainty_note}
                  </div>
                </div>
              )}
          </div>

          {item.mismatch_flags.length > 0 && (
            <div className="border-t border-white/[0.07] pt-3">
              <span className="mb-2 block text-[11px] font-semibold uppercase tracking-wider text-rose-400/90">
                Mismatch Flags
              </span>
              <div className="flex flex-wrap gap-1.5">
                {item.mismatch_flags.map((flag, idx) => (
                  <span
                    key={`${item.claim_id}-flag-${idx}`}
                    className="inline-flex items-center rounded-md border border-rose-500/30 bg-rose-500/12 px-2 py-0.5 text-xs font-medium text-rose-200"
                  >
                    {flag}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Matched Citation, Evidence Passage, Source URL */}
        <div className="flex flex-col justify-between space-y-4 rounded-xl border border-white/[0.08] bg-black/25 p-4">
          <div className="space-y-3.5">
            <div>
              <span className="mb-1.5 inline-flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                <BookOpen className="h-3.5 w-3.5 text-violet-400" />
                Matched Citation
              </span>
              <p className="text-sm leading-relaxed text-slate-200">
                {item.matched_citation ? (
                  item.matched_citation
                ) : (
                  <span className="italic text-slate-500">
                    No matching citation resolved for this claim.
                  </span>
                )}
              </p>
            </div>

            <div>
              <span className="mb-1.5 inline-flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                <FileText className="h-3.5 w-3.5 text-cyan-400" />
                Evidence Passage
              </span>
              {item.evidence_passage ? (
                <blockquote className="rounded-lg border border-white/[0.06] border-l-2 border-l-violet-400/70 bg-white/[0.02] p-3 text-xs italic leading-relaxed text-slate-300 sm:text-sm">
                  “{item.evidence_passage}”
                </blockquote>
              ) : (
                <p className="text-xs italic text-slate-500 sm:text-sm">
                  No direct source passage returned for comparison.
                </p>
              )}
            </div>
          </div>

          {item.source_url && (
            <div className="border-t border-white/[0.07] pt-3">
              <a
                href={item.source_url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1.5 rounded-lg border border-violet-400/30 bg-violet-500/12 px-3 py-1.5 text-xs font-medium text-violet-200 transition-colors hover:border-violet-400/60 hover:bg-violet-500/20 hover:text-white"
              >
                Open Referenced Source
                <ExternalLink className="h-3.5 w-3.5" />
              </a>
            </div>
          )}
        </div>
      </div>

      {/* ================================================================== */}
      {/* Interactive Deep Verification Diagnostics (Gaps 1 - 4)             */}
      {/* ================================================================== */}
      {hasAnyGapData && (
        <div className="mt-4 border-t border-white/[0.08] pt-4">
          <div className="mb-2.5 flex items-center justify-between">
            <span className="inline-flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-violet-300/90">
              <Sparkles className="h-3.5 w-3.5 text-cyan-300" />
              Deep Audit Diagnostics (Gaps 1–4)
            </span>
            <span className="text-[11px] text-slate-500">
              Click any module to inspect details
            </span>
          </div>

          {/* 4-Tile Clickable Summary Strip */}
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 xl:grid-cols-4">
            {/* Gap 1 Tile: Hybrid NLI Ranking */}
            <button
              type="button"
              disabled={!g1}
              onClick={() => toggleGapTab("gap1")}
              className={`flex flex-col justify-between rounded-xl border p-3 text-left transition-all ${
                openGapTab === "gap1"
                  ? "border-cyan-400/55 bg-cyan-500/12 ring-1 ring-cyan-400/25"
                  : g1
                    ? "border-white/[0.08] bg-white/[0.02] hover:border-cyan-400/35 hover:bg-white/[0.04]"
                    : "cursor-not-allowed border-white/[0.05] bg-black/15 opacity-50"
              }`}
            >
              <div className="flex w-full items-center justify-between gap-2">
                <span className="text-[10px] font-semibold uppercase tracking-wider text-cyan-300">
                  Gap 1 · Hybrid NLI
                </span>
                <ChevronDown
                  className={`h-3.5 w-3.5 text-slate-400 transition-transform ${
                    openGapTab === "gap1" ? "rotate-180 text-cyan-300" : ""
                  }`}
                />
              </div>
              <div className="mt-1.5">
                {g1 ? (
                  <>
                    <p className="text-xs font-semibold text-white">
                      {g1.top_logical_stance}{" "}
                      <span className="font-mono font-normal text-slate-400">
                        · CE {g1.top_cross_encoder_score.toFixed(2)}
                      </span>
                    </p>
                    <p className="mt-0.5 truncate text-[11px] text-slate-400">
                      {g1.similarity_trap_detected
                        ? "Similarity trap flagged"
                        : `${g1.ranked_candidates.length} passage(s) ranked`}
                    </p>
                  </>
                ) : (
                  <p className="text-xs text-slate-500">No passage text</p>
                )}
              </div>
            </button>

            {/* Gap 2 Tile: Atomic Decomposition */}
            <button
              type="button"
              disabled={!g2}
              onClick={() => toggleGapTab("gap2")}
              className={`flex flex-col justify-between rounded-xl border p-3 text-left transition-all ${
                openGapTab === "gap2"
                  ? "border-violet-400/55 bg-violet-500/15 ring-1 ring-violet-400/25"
                  : g2
                    ? "border-white/[0.08] bg-white/[0.02] hover:border-violet-400/35 hover:bg-white/[0.04]"
                    : "cursor-not-allowed border-white/[0.05] bg-black/15 opacity-50"
              }`}
            >
              <div className="flex w-full items-center justify-between gap-2">
                <span className="text-[10px] font-semibold uppercase tracking-wider text-violet-300">
                  Gap 2 · Atomic Split
                </span>
                <ChevronDown
                  className={`h-3.5 w-3.5 text-slate-400 transition-transform ${
                    openGapTab === "gap2" ? "rotate-180 text-violet-300" : ""
                  }`}
                />
              </div>
              <div className="mt-1.5">
                {g2 ? (
                  <>
                    <p className="text-xs font-semibold text-white">
                      {g2.is_compound
                        ? `Compound (${g2.propositions.length} parts)`
                        : "Atomic (1 part)"}
                    </p>
                    <p className="mt-0.5 truncate text-[11px] text-slate-400">
                      Overall:{" "}
                      {STATUS_CONFIG[g2.overall_verdict]?.shortLabel ??
                        g2.overall_verdict}
                    </p>
                  </>
                ) : (
                  <p className="text-xs text-slate-500">Not evaluated</p>
                )}
              </div>
            </button>

            {/* Gap 3 Tile: Temporal Consensus */}
            <button
              type="button"
              disabled={!g3}
              onClick={() => toggleGapTab("gap3")}
              className={`flex flex-col justify-between rounded-xl border p-3 text-left transition-all ${
                openGapTab === "gap3"
                  ? "border-amber-400/55 bg-amber-500/12 ring-1 ring-amber-400/25"
                  : g3
                    ? "border-white/[0.08] bg-white/[0.02] hover:border-amber-400/35 hover:bg-white/[0.04]"
                    : "cursor-not-allowed border-white/[0.05] bg-black/15 opacity-50"
              }`}
            >
              <div className="flex w-full items-center justify-between gap-2">
                <span className="text-[10px] font-semibold uppercase tracking-wider text-amber-300">
                  Gap 3 · Temporal
                </span>
                <ChevronDown
                  className={`h-3.5 w-3.5 text-slate-400 transition-transform ${
                    openGapTab === "gap3" ? "rotate-180 text-amber-300" : ""
                  }`}
                />
              </div>
              <div className="mt-1.5">
                {g3 && temporalMeta ? (
                  <>
                    <p className="truncate text-xs font-semibold text-white">
                      {temporalMeta.label}
                    </p>
                    <p className="mt-0.5 truncate text-[11px] text-slate-400">
                      {g3.cited_publication_year ?? "Year N/A"} ·{" "}
                      {g3.subsequent_citation_count.toLocaleString()} cites
                    </p>
                  </>
                ) : (
                  <p className="text-xs text-slate-500">Source not indexed</p>
                )}
              </div>
            </button>

            {/* Gap 4 Tile: Uncertainty & Calibration */}
            <button
              type="button"
              disabled={!g4}
              onClick={() => toggleGapTab("gap4")}
              className={`flex flex-col justify-between rounded-xl border p-3 text-left transition-all ${
                openGapTab === "gap4"
                  ? "border-emerald-400/55 bg-emerald-500/12 ring-1 ring-emerald-400/25"
                  : g4
                    ? "border-white/[0.08] bg-white/[0.02] hover:border-emerald-400/35 hover:bg-white/[0.04]"
                    : "cursor-not-allowed border-white/[0.05] bg-black/15 opacity-50"
              }`}
            >
              <div className="flex w-full items-center justify-between gap-2">
                <span className="text-[10px] font-semibold uppercase tracking-wider text-emerald-300">
                  Gap 4 · Calibration
                </span>
                <ChevronDown
                  className={`h-3.5 w-3.5 text-slate-400 transition-transform ${
                    openGapTab === "gap4" ? "rotate-180 text-emerald-300" : ""
                  }`}
                />
              </div>
              <div className="mt-1.5">
                {g4 && actionMeta ? (
                  <>
                    <p className="text-xs font-semibold text-white">
                      {formatPercent(g4.calibrated_probability)} Prob{" "}
                      <span className="font-normal text-slate-400">
                        · {actionMeta.label}
                      </span>
                    </p>
                    <p className="mt-0.5 truncate text-[11px] text-slate-400">
                      Coverage {formatPercent(g4.evidence_coverage_score)} · Rel{" "}
                      {formatPercent(g4.composite_reliability_score)}
                    </p>
                  </>
                ) : (
                  <p className="text-xs text-slate-500">Not calibrated</p>
                )}
              </div>
            </button>
          </div>

          {/* Expanded Panel: Gap 1 (Hybrid Evidence Ranking & NLI) */}
          {openGapTab === "gap1" && g1 && (
            <div className="mt-3 space-y-3 rounded-xl border border-cyan-400/25 bg-black/35 p-4">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <p className="text-xs font-semibold text-cyan-200">
                  Hybrid BM25 + Semantic RRF + Cross-Encoder + NLI Ranking
                </p>
                <span className="font-mono text-[11px] text-slate-400">
                  {g1.embedding_backend} · {g1.nli_backend}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                <div className="rounded-lg border border-white/[0.07] bg-white/[0.02] p-2.5">
                  <span className="block text-[10px] uppercase text-slate-400">
                    Top BM25
                  </span>
                  <span className="font-mono text-sm font-semibold text-white">
                    {g1.top_bm25_score.toFixed(3)}
                  </span>
                </div>
                <div className="rounded-lg border border-white/[0.07] bg-white/[0.02] p-2.5">
                  <span className="block text-[10px] uppercase text-slate-400">
                    Semantic Sim
                  </span>
                  <span className="font-mono text-sm font-semibold text-white">
                    {g1.top_semantic_score.toFixed(3)}
                  </span>
                </div>
                <div className="rounded-lg border border-white/[0.07] bg-white/[0.02] p-2.5">
                  <span className="block text-[10px] uppercase text-slate-400">
                    Hybrid RRF
                  </span>
                  <span className="font-mono text-sm font-semibold text-white">
                    {g1.top_rrf_score.toFixed(4)}
                  </span>
                </div>
                <div className="rounded-lg border border-white/[0.07] bg-white/[0.02] p-2.5">
                  <span className="block text-[10px] uppercase text-slate-400">
                    Cross-Encoder
                  </span>
                  <span className="font-mono text-sm font-semibold text-cyan-200">
                    {g1.top_cross_encoder_score.toFixed(3)}
                  </span>
                </div>
              </div>

              {g1.ranked_candidates.length > 0 && (
                <div className="space-y-2">
                  <span className="block text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                    Top Ranked Passages
                  </span>
                  {g1.ranked_candidates.map((cand) => (
                    <div
                      key={`${item.claim_id}-${cand.passage_id}`}
                      className={`rounded-lg border p-3 text-xs ${
                        cand.similarity_trap
                          ? "border-rose-500/35 bg-rose-500/[0.07]"
                          : "border-white/[0.07] bg-white/[0.02]"
                      }`}
                    >
                      <div className="mb-1.5 flex flex-wrap items-center justify-between gap-2">
                        <div className="flex flex-wrap items-center gap-1.5">
                          <span className="rounded bg-white/[0.06] px-1.5 py-0.5 font-mono text-[10px] text-slate-300">
                            {cand.passage_id}
                          </span>
                          <span
                            className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${
                              cand.logical_stance === "ENTAILS"
                                ? "bg-emerald-500/15 text-emerald-300"
                                : cand.logical_stance === "CONTRADICTS"
                                  ? "bg-rose-500/15 text-rose-300"
                                  : "bg-slate-500/20 text-slate-300"
                            }`}
                          >
                            {cand.logical_stance}
                          </span>
                          {cand.similarity_trap && (
                            <span className="rounded-full bg-rose-500/25 px-2 py-0.5 text-[10px] font-semibold text-rose-200">
                              Similarity Trap
                            </span>
                          )}
                        </div>
                        <span className="font-mono text-[10px] text-slate-400">
                          CE {cand.cross_encoder_score.toFixed(2)} · Sem{" "}
                          {cand.semantic_score.toFixed(2)} · BM25{" "}
                          {cand.bm25_score.toFixed(2)}
                        </span>
                      </div>
                      <p className="italic leading-relaxed text-slate-200">
                        “{cand.text}”
                      </p>
                      {cand.stance_rationale && (
                        <p className="mt-1 text-[11px] text-slate-400">
                          {cand.stance_rationale}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Expanded Panel: Gap 2 (Atomic Claim Decomposition) */}
          {openGapTab === "gap2" && g2 && (
            <div className="mt-3 space-y-3 rounded-xl border border-violet-400/25 bg-black/35 p-4">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <p className="text-xs font-semibold text-violet-200">
                  Atomic Claim Decomposition ({g2.propositions.length}{" "}
                  {g2.propositions.length === 1
                    ? "Proposition"
                    : "Propositions"}
                  )
                </p>
                <span className="font-mono text-[11px] text-slate-400">
                  Method: {g2.decomposition_method}
                </span>
              </div>

              {g2.summary_explanation && (
                <p className="text-xs leading-relaxed text-slate-300">
                  {g2.summary_explanation}
                </p>
              )}

              <div className="space-y-2">
                {g2.propositions.map((prop) => {
                  const pMeta =
                    STATUS_CONFIG[prop.verdict] ?? STATUS_CONFIG.UNSUPPORTED;
                  return (
                    <div
                      key={`${item.claim_id}-prop-${prop.proposition_id}`}
                      className="rounded-lg border border-white/[0.08] bg-white/[0.02] p-3 text-xs"
                    >
                      <div className="mb-1.5 flex flex-wrap items-center gap-2">
                        <span className="rounded bg-white/[0.06] px-1.5 py-0.5 font-mono text-[10px] text-slate-300">
                          {prop.proposition_id}
                        </span>
                        <span
                          className={`rounded-full border px-2 py-0.5 text-[10px] font-medium ${pMeta.badgeClass}`}
                        >
                          {pMeta.label}
                        </span>
                        <span className="text-[10px] text-slate-400">
                          Confidence: {prop.confidence}
                        </span>
                      </div>
                      <p className="font-medium leading-relaxed text-white">
                        {prop.proposition_text}
                      </p>
                      {prop.explanation && (
                        <p className="mt-1 text-[11px] text-slate-400">
                          {prop.explanation}
                        </p>
                      )}
                      {prop.preserved_qualifiers.length > 0 && (
                        <div className="mt-2 flex flex-wrap gap-1">
                          {prop.preserved_qualifiers.map((q, qIdx) => (
                            <span
                              key={`${prop.proposition_id}-q-${qIdx}`}
                              className="rounded border border-violet-400/25 bg-violet-500/10 px-1.5 py-0.5 font-mono text-[10px] text-violet-200"
                            >
                              {q}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Expanded Panel: Gap 3 (Time-Aware Consensus Tracking) */}
          {openGapTab === "gap3" && g3 && temporalMeta && (
            <div className="mt-3 space-y-3 rounded-xl border border-amber-400/25 bg-black/35 p-4">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <p className="text-xs font-semibold text-amber-200">
                  Time-Aware Citation & Consensus Tracking (Preserves Historical
                  Verdict)
                </p>
                <span
                  className={`rounded-full border px-2 py-0.5 text-[10px] font-medium ${temporalMeta.badgeClass}`}
                >
                  {temporalMeta.label}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                <div className="rounded-lg border border-white/[0.07] bg-white/[0.02] p-2.5">
                  <span className="block text-[10px] uppercase text-slate-400">
                    Stage & Year
                  </span>
                  <span className="text-xs font-semibold text-white">
                    {g3.cited_publication_year ?? "N/A"} ·{" "}
                    {formatStageLabel(g3.publication_stage)}
                  </span>
                </div>
                <div className="rounded-lg border border-white/[0.07] bg-white/[0.02] p-2.5">
                  <span className="block text-[10px] uppercase text-slate-400">
                    Total Citations
                  </span>
                  <span className="font-mono text-sm font-semibold text-white">
                    {g3.subsequent_citation_count.toLocaleString()}
                  </span>
                </div>
                <div className="rounded-lg border border-white/[0.07] bg-white/[0.02] p-2.5">
                  <span className="block text-[10px] uppercase text-slate-400">
                    Confirming / Challenging
                  </span>
                  <span className="font-mono text-sm font-semibold text-white">
                    <span className="text-emerald-300">
                      +{g3.confirming_studies_count}
                    </span>{" "}
                    /{" "}
                    <span className="text-rose-300">
                      -{g3.challenging_studies_count}
                    </span>
                  </span>
                </div>
                <div className="rounded-lg border border-white/[0.07] bg-white/[0.02] p-2.5">
                  <span className="block text-[10px] uppercase text-slate-400">
                    Systematic Reviews
                  </span>
                  <span className="font-mono text-sm font-semibold text-cyan-200">
                    {g3.systematic_reviews_count}
                  </span>
                </div>
              </div>

              {g3.temporal_summary && (
                <p className="text-xs leading-relaxed text-slate-300">
                  {g3.temporal_summary}
                </p>
              )}

              {g3.temporal_flags.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {g3.temporal_flags.map((flag, idx) => (
                    <span
                      key={`${item.claim_id}-tflag-${idx}`}
                      className="rounded-md border border-amber-500/30 bg-amber-500/10 px-2 py-0.5 font-mono text-[10px] text-amber-200"
                    >
                      {flag}
                    </span>
                  ))}
                </div>
              )}

              {g3.chronological_timeline.length > 0 && (
                <div>
                  <span className="mb-1.5 block text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                    Chronological Citation Timeline
                  </span>
                  <ul className="space-y-1.5 border-l border-amber-400/30 pl-3 text-xs text-slate-300">
                    {g3.chronological_timeline.map((entry, idx) => (
                      <li
                        key={`${item.claim_id}-tl-${idx}`}
                        className="leading-relaxed"
                      >
                        {entry}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {/* Expanded Panel: Gap 4 (Uncertainty Calibration & Abstention) */}
          {openGapTab === "gap4" && g4 && actionMeta && (
            <div className="mt-3 space-y-3 rounded-xl border border-emerald-400/25 bg-black/35 p-4">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <p className="text-xs font-semibold text-emerald-200">
                  Uncertainty-Aware Calibration & Abstention Policy
                </p>
                <span
                  className={`rounded-full border px-2 py-0.5 text-[10px] font-medium ${actionMeta.badgeClass}`}
                >
                  Policy: {actionMeta.label}
                </span>
              </div>

              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                <MetricBar
                  label="Calibrated Probability"
                  value={g4.calibrated_probability}
                  colorClass="bg-gradient-to-r from-violet-400 to-cyan-400"
                />
                <MetricBar
                  label="Evidence Coverage"
                  value={g4.evidence_coverage_score}
                  colorClass="bg-emerald-400"
                />
                <MetricBar
                  label="Retrieval Confidence"
                  value={g4.retrieval_confidence}
                  colorClass="bg-cyan-400"
                />
                <MetricBar
                  label="Passage Agreement"
                  value={g4.passage_agreement_score}
                  colorClass={
                    g4.has_conflicting_passages
                      ? "bg-rose-400"
                      : "bg-indigo-400"
                  }
                />
              </div>

              <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-white/[0.07] bg-white/[0.02] px-3 py-2 text-xs">
                <span className="text-slate-300">{g4.user_message}</span>
                <span className="font-mono text-[11px] font-semibold text-emerald-300">
                  Composite: {formatPercent(g4.composite_reliability_score)}
                </span>
              </div>

              {g4.uncertainty_flags.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {g4.uncertainty_flags.map((flag, idx) => (
                    <span
                      key={`${item.claim_id}-uflag-${idx}`}
                      className="rounded-md border border-fuchsia-500/30 bg-fuchsia-500/10 px-2 py-0.5 font-mono text-[10px] text-fuchsia-200"
                    >
                      {flag}
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </article>
  );
};

// ============================================================================
// 8. Main Application Component (Supabase Auth + Cloud History + Workbench)
// ============================================================================

export default function App() {
  const [session, setSession] = useState<AuthSession | null>(null);
  const [authLoading, setAuthLoading] = useState<boolean>(isSupabaseConfigured);

  const [activeView, setActiveView] = useState<NavView>("verify");
  const [mobileMenuOpen, setMobileMenuOpen] = useState<boolean>(false);

  const [apiBaseUrl, setApiBaseUrl] = useState<string>(resolveEnvApiUrl);
  const [apiUrlInput, setApiUrlInput] = useState<string>(resolveEnvApiUrl);
  const [settingsSavedNotice, setSettingsSavedNotice] =
    useState<boolean>(false);
  const [health, setHealth] = useState<HealthCheckDiagnostics>({
    status: "checking",
    httpCode: null,
    latencyMs: null,
    checkedAt: null,
    payload: null,
    errorMessage: null,
  });

  const [inputMode, setInputMode] = useState<InputMode>("pdf");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [documentTitle, setDocumentTitle] = useState<string>("");
  const [rawText, setRawText] = useState<string>("");
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [isVerifying, setIsVerifying] = useState<boolean>(false);
  const [activePipelineStep, setActivePipelineStep] = useState<number>(0);
  const [showColdStartNotice, setShowColdStartNotice] =
    useState<boolean>(false);
  const [verifyError, setVerifyError] = useState<string | null>(null);
  const [exportingFormat, setExportingFormat] = useState<
    "pdf" | "markdown" | null
  >(null);
  const [exportError, setExportError] = useState<string | null>(null);

  const [activeReport, setActiveReport] = useState<VerificationReport | null>(
    null,
  );
  const [verdictFilter, setVerdictFilter] = useState<VerdictFilter>("ALL");
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [onlyHumanReview, setOnlyHumanReview] = useState<boolean>(false);

  const [sessionHistory, setSessionHistory] = useState<SessionHistoryEntry[]>(
    () => (isSupabaseConfigured ? [] : loadLocalSessionHistory()),
  );
  const [historyLoading, setHistoryLoading] = useState<boolean>(false);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [confirmClearHistory, setConfirmClearHistory] =
    useState<boolean>(false);

  // --------------------------------------------------------------------------
  // Supabase Auth Session Bootstrap & Listener
  // --------------------------------------------------------------------------
  useEffect(() => {
    if (!supabase || !isSupabaseConfigured) {
      setAuthLoading(false);
      return;
    }

    let isMounted = true;

    supabase.auth
      .getSession()
      .then(({ data }) => {
        if (!isMounted) return;
        if (data.session?.user) {
          setSession(mapSupabaseUser(data.session.user));
        } else {
          setSession((prev) => (prev && !prev.isCloudAuth ? prev : null));
        }
      })
      .catch(() => {
        if (isMounted) {
          setSession((prev) => (prev && !prev.isCloudAuth ? prev : null));
        }
      })
      .finally(() => {
        if (isMounted) setAuthLoading(false);
      });

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, nextSession) => {
      if (!isMounted) return;
      if (nextSession?.user) {
        setSession(mapSupabaseUser(nextSession.user));
      } else {
        setSession((prev) => {
          if (prev && !prev.isCloudAuth) {
            return prev;
          }
          setActiveReport(null);
          setSessionHistory([]);
          return null;
        });
      }
      setAuthLoading(false);
    });

    return () => {
      isMounted = false;
      subscription.unsubscribe();
    };
  }, []);

  // --------------------------------------------------------------------------
  // Load Cloud Report History from Supabase Postgres
  // --------------------------------------------------------------------------
  const fetchCloudHistory = useCallback(async (userId: string) => {
    if (!supabase || !isSupabaseConfigured) return;
    setHistoryLoading(true);
    setHistoryError(null);

    try {
      const { data, error } = await supabase
        .from("verification_reports")
        .select(
          "id, document_title, input_method, total_claims, report_json, created_at",
        )
        .eq("user_id", userId)
        .order("created_at", { ascending: false })
        .limit(50);

      if (error) {
        throw error;
      }

      const rows = (data || []) as SupabaseReportRow[];
      const mapped: SessionHistoryEntry[] = rows.map((row) => ({
        id: row.id,
        verifiedAt: row.created_at,
        inputMethod: row.input_method === "text" ? "text" : "pdf",
        report: normalizeReport(row.report_json),
        syncedToCloud: true,
      }));

      setSessionHistory(mapped);
    } catch (err) {
      setHistoryError(
        err instanceof Error
          ? `Could not load cloud history: ${err.message}`
          : "Could not load cloud history from Supabase.",
      );
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  useEffect(() => {
    if (session?.userId && session.isCloudAuth && isSupabaseConfigured) {
      fetchCloudHistory(session.userId);
    }
  }, [session?.userId, session?.isCloudAuth, fetchCloudHistory]);

  // --------------------------------------------------------------------------
  // Live GET /health Diagnostics
  // --------------------------------------------------------------------------
  const runHealthCheck = useCallback(async (targetUrl: string) => {
    setHealth((prev) => ({ ...prev, status: "checking", errorMessage: null }));
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), 14000);
    const startTime = performance.now();

    try {
      const response = await fetch(`${targetUrl}/health`, {
        method: "GET",
        signal: controller.signal,
      });
      const latencyMs = Math.round(performance.now() - startTime);
      const jsonPayload = await response.json().catch(() => null);

      if (!response.ok) {
        setHealth({
          status: "offline",
          httpCode: response.status,
          latencyMs,
          checkedAt: new Date().toISOString(),
          payload: jsonPayload,
          errorMessage: `Backend responded with HTTP ${response.status}.`,
        });
        return;
      }

      setHealth({
        status: "online",
        httpCode: response.status,
        latencyMs,
        checkedAt: new Date().toISOString(),
        payload: jsonPayload,
        errorMessage: null,
      });
    } catch (err) {
      const latencyMs = Math.round(performance.now() - startTime);
      setHealth({
        status: "offline",
        httpCode: null,
        latencyMs,
        checkedAt: new Date().toISOString(),
        payload: null,
        errorMessage:
          err instanceof Error && err.name === "AbortError"
            ? "Health check timed out after 14s (Render free tier may be waking up)."
            : "Unable to reach backend /health endpoint. Check the API URL or server status.",
      });
    } finally {
      window.clearTimeout(timeoutId);
    }
  }, []);

  useEffect(() => {
    runHealthCheck(apiBaseUrl);
  }, [apiBaseUrl, runHealthCheck]);

  useEffect(() => {
    if (!isVerifying) {
      setActivePipelineStep(0);
      setShowColdStartNotice(false);
      return;
    }

    const stepInterval = window.setInterval(() => {
      setActivePipelineStep((prev) =>
        prev < VERIFICATION_PIPELINE_STEPS.length - 1 ? prev + 1 : prev,
      );
    }, 2400);

    const coldStartTimer = window.setTimeout(() => {
      setShowColdStartNotice(true);
    }, 4500);

    return () => {
      window.clearInterval(stepInterval);
      window.clearTimeout(coldStartTimer);
    };
  }, [isVerifying]);

  const textMetrics = useMemo(() => {
    const trimmed = rawText.trim();
    return {
      words: trimmed ? trimmed.split(/\s+/).length : 0,
      chars: rawText.length,
    };
  }, [rawText]);

  const reportCounts = useMemo(() => {
    if (!activeReport) return null;
    return computeStatusCounts(activeReport);
  }, [activeReport]);

  // Document-level Gaps 1-4 diagnostics roll-up
  const gapSummary = useMemo(() => {
    if (!activeReport) return null;
    let hasDiagnostics = false;
    let similarityTraps = 0;
    let compoundClaims = 0;
    let temporalAlerts = 0;
    let abstainedClaims = 0;

    for (const r of activeReport.results) {
      if (
        r.evidence_ranking ||
        r.claim_decomposition ||
        r.temporal_assessment ||
        r.uncertainty_calibration
      ) {
        hasDiagnostics = true;
      }
      if (r.evidence_ranking?.similarity_trap_detected) similarityTraps += 1;
      if (r.claim_decomposition?.is_compound) compoundClaims += 1;
      if (r.temporal_assessment?.needs_temporal_review) temporalAlerts += 1;
      if (r.uncertainty_calibration?.abstained) abstainedClaims += 1;
    }

    if (!hasDiagnostics) return null;
    return {
      similarityTraps,
      compoundClaims,
      temporalAlerts,
      abstainedClaims,
    };
  }, [activeReport]);

  const filteredClaims = useMemo(() => {
    if (!activeReport) return [];
    const query = searchQuery.trim().toLowerCase();

    return activeReport.results.filter((item) => {
      if (verdictFilter !== "ALL" && item.verdict !== verdictFilter)
        return false;
      if (onlyHumanReview && !item.needs_human_review) return false;
      if (query) {
        const haystack = [
          String(item.claim_id),
          item.claim_text,
          item.matched_citation ?? "",
          item.evidence_passage ?? "",
          item.explanation,
          item.uncertainty_note,
          ...item.mismatch_flags,
          item.temporal_assessment?.temporal_status ?? "",
          item.uncertainty_calibration?.action ?? "",
          ...(item.claim_decomposition?.propositions.map(
            (p) => p.proposition_text,
          ) ?? []),
        ]
          .join(" ")
          .toLowerCase();
        return haystack.includes(query);
      }
      return true;
    });
  }, [activeReport, verdictFilter, onlyHumanReview, searchQuery]);

  // --------------------------------------------------------------------------
  // Interaction Handlers (Live FastAPI + Supabase Cloud Persistence)
  // --------------------------------------------------------------------------
  const handleFilePicked = (file: File | null) => {
    setVerifyError(null);
    if (!file) {
      setSelectedFile(null);
      return;
    }

    const isPdf =
      file.type === "application/pdf" ||
      file.name.toLowerCase().endsWith(".pdf");
    if (!isPdf) {
      setVerifyError(
        "Please upload a valid PDF document (.pdf) or switch to Paste Text mode.",
      );
      return;
    }

    setSelectedFile(file);
    if (!documentTitle.trim()) {
      setDocumentTitle(file.name.replace(/\.pdf$/i, ""));
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files?.[0] ?? null;
    handleFilePicked(droppedFile);
  };

  const handleLoadDemoSample = () => {
    setVerifyError(null);
    setExportError(null);
    setSelectedFile(null);
    setInputMode("text");
    setDocumentTitle(DEMO_SAMPLE_TITLE);
    setRawText(DEMO_SAMPLE_TEXT);
  };

  const handleVerifySubmit = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    setVerifyError(null);
    setExportError(null);

    if (inputMode === "pdf" && !selectedFile) {
      setVerifyError("Please select a PDF research paper to verify.");
      return;
    }

    if (inputMode === "text" && rawText.trim().length < 20) {
      setVerifyError(
        "Please paste at least 20 characters of scientific text or claims with citations.",
      );
      return;
    }

    setIsVerifying(true);
    setActivePipelineStep(0);

    try {
      let response: Response;

      if (inputMode === "pdf" && selectedFile) {
        const formData = new FormData();
        formData.append("file", selectedFile);
        if (documentTitle.trim()) {
          formData.append("document_title", documentTitle.trim());
        }
        response = await fetch(`${apiBaseUrl}/api/verify`, {
          method: "POST",
          body: formData,
        });
      } else {
        response = await fetch(`${apiBaseUrl}/api/verify`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: rawText.trim(),
            document_title: documentTitle.trim() || undefined,
          }),
        });
      }

      if (!response.ok) {
        const errBody = await response.json().catch(() => null);
        const detail =
          errBody && typeof errBody.detail === "string" ? errBody.detail : null;

        if (response.status === 404) {
          throw new Error(
            `POST ${apiBaseUrl}/api/verify returned 404 Not Found. Your live backend is healthy, and this UI will work automatically as soon as the /api/verify route is deployed.`,
          );
        }

        throw new Error(
          detail
            ? `Verification failed (HTTP ${response.status}): ${detail}`
            : `Backend responded with HTTP ${response.status}. Check /docs on your backend API.`,
        );
      }

      const rawJson = await response.json();
      const normalized = normalizeReport(rawJson);

      setActiveReport(normalized);
      setVerdictFilter("ALL");
      setSearchQuery("");
      setOnlyHumanReview(false);
      setExportError(null);

      const tempId = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      const nowIso = new Date().toISOString();

      if (supabase && isSupabaseConfigured && session?.userId) {
        const { data: inserted, error: insertError } = await supabase
          .from("verification_reports")
          .insert({
            user_id: session.userId,
            document_title: normalized.document_title,
            input_method: inputMode,
            total_claims: normalized.total_claims,
            report_json: normalized,
          })
          .select("id, created_at")
          .single();

        if (insertError) {
          setHistoryError(
            `Report verified, but cloud save failed: ${insertError.message}`,
          );
          const fallbackEntry: SessionHistoryEntry = {
            id: tempId,
            verifiedAt: nowIso,
            inputMethod: inputMode,
            report: normalized,
            syncedToCloud: false,
          };
          setSessionHistory((prev) => [fallbackEntry, ...prev].slice(0, 50));
        } else {
          setHistoryError(null);
          const cloudEntry: SessionHistoryEntry = {
            id: inserted?.id || tempId,
            verifiedAt: inserted?.created_at || nowIso,
            inputMethod: inputMode,
            report: normalized,
            syncedToCloud: true,
          };
          setSessionHistory((prev) => [cloudEntry, ...prev].slice(0, 50));
        }
      } else {
        const localEntry: SessionHistoryEntry = {
          id: tempId,
          verifiedAt: nowIso,
          inputMethod: inputMode,
          report: normalized,
          syncedToCloud: false,
        };
        setSessionHistory((prev) => {
          const updated = [localEntry, ...prev].slice(0, 25);
          saveLocalSessionHistory(updated);
          return updated;
        });
      }

      setActiveView("report");
    } catch (err) {
      setVerifyError(
        err instanceof Error
          ? err.message
          : "Unexpected network error while contacting the verification API.",
      );
    } finally {
      setIsVerifying(false);
    }
  };

  const handleExportJson = () => {
    if (!activeReport) return;
    const blob = new Blob([JSON.stringify(activeReport, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    const safeTitle = activeReport.document_title
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-|-$/g, "");
    anchor.href = url;
    anchor.download = `${safeTitle || "citeverify-report"}.json`;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
  };

  const handleExportReport = async (format: "pdf" | "markdown") => {
    if (!activeReport || exportingFormat) return;
    setExportingFormat(format);
    setExportError(null);

    try {
      const response = await fetch(
        `${apiBaseUrl}/api/export?format=${format}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(activeReport),
        },
      );

      if (!response.ok) {
        let message = `Export failed with status ${response.status}.`;
        try {
          const errBody = await response.json();
          if (typeof errBody?.detail === "string") {
            message = errBody.detail;
          }
        } catch {
          // Ignore non-JSON error responses
        }
        throw new Error(message);
      }

      const blob = await response.blob();
      const disposition = response.headers.get("Content-Disposition") || "";
      const filenameMatch = disposition.match(/filename="?([^"]+)"?/i);
      const ext = format === "pdf" ? "pdf" : "md";
      const safeTitle = activeReport.document_title
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "_")
        .replace(/^_|_$/g, "");
      const filename =
        filenameMatch?.[1] ||
        `citeverify_audit_${safeTitle || "report"}.${ext}`;

      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = filename;
      document.body.appendChild(anchor);
      anchor.click();
      document.body.removeChild(anchor);
      URL.revokeObjectURL(url);
    } catch (err) {
      setExportError(
        err instanceof Error
          ? err.message
          : "Failed to export report. Please try again.",
      );
    } finally {
      setExportingFormat(null);
    }
  };

  const handleSaveApiUrl = (e: React.FormEvent) => {
    e.preventDefault();
    const cleaned = apiUrlInput.trim().replace(/\/+$/, "");
    if (!cleaned || !/^https?:\/\//i.test(cleaned)) {
      return;
    }
    setApiBaseUrl(cleaned);
    setApiUrlInput(cleaned);
    setSettingsSavedNotice(true);
    window.setTimeout(() => setSettingsSavedNotice(false), 2600);
  };

  const handleResetApiUrl = () => {
    const envUrl = resolveEnvApiUrl();
    setApiBaseUrl(envUrl);
    setApiUrlInput(envUrl);
    setSettingsSavedNotice(true);
    window.setTimeout(() => setSettingsSavedNotice(false), 2600);
  };

  const handleOpenHistoryEntry = (entry: SessionHistoryEntry) => {
    setActiveReport(entry.report);
    setVerdictFilter("ALL");
    setSearchQuery("");
    setOnlyHumanReview(false);
    setExportError(null);
    setActiveView("report");
  };

  const handleDeleteHistoryEntry = async (entry: SessionHistoryEntry) => {
    setHistoryError(null);
    setSessionHistory((prev) => {
      const updated = prev.filter((item) => item.id !== entry.id);
      if (!session?.isCloudAuth) {
        saveLocalSessionHistory(updated);
      }
      return updated;
    });

    if (
      supabase &&
      isSupabaseConfigured &&
      session?.userId &&
      entry.syncedToCloud
    ) {
      const { error } = await supabase
        .from("verification_reports")
        .delete()
        .eq("id", entry.id)
        .eq("user_id", session.userId);

      if (error) {
        setHistoryError(`Failed to delete cloud report: ${error.message}`);
        fetchCloudHistory(session.userId);
      }
    }
  };

  const handleClearAllHistory = async () => {
    setHistoryError(null);
    setConfirmClearHistory(false);
    setSessionHistory([]);

    if (supabase && isSupabaseConfigured && session?.userId) {
      const { error } = await supabase
        .from("verification_reports")
        .delete()
        .eq("user_id", session.userId);

      if (error) {
        setHistoryError(`Failed to clear cloud history: ${error.message}`);
        fetchCloudHistory(session.userId);
      }
    } else {
      saveLocalSessionHistory([]);
    }
  };

  const handleSignOut = async () => {
    setMobileMenuOpen(false);
    if (supabase && isSupabaseConfigured && session?.isCloudAuth) {
      await supabase.auth.signOut().catch(() => null);
    }
    setSession(null);
    setActiveReport(null);
    setSessionHistory(isSupabaseConfigured ? [] : loadLocalSessionHistory());
  };

  if (authLoading) {
    return (
      <div className="flex min-h-screen w-full flex-col items-center justify-center bg-[#070711] px-4 text-slate-100">
        <BrandBadge size="lg" />
        <div className="mt-5 flex items-center gap-2.5 text-sm text-slate-300">
          <Loader2 className="h-4 w-4 animate-spin text-violet-400" />
          <span>Restoring your CiteVerify workspace...</span>
        </div>
      </div>
    );
  }

  if (!session) {
    return (
      <AuroraLoginScreen
        health={health}
        onRefreshHealth={() => runHealthCheck(apiBaseUrl)}
        onLoginSuccess={(nextSession) => {
          setSession(nextSession);
          if (!nextSession.isCloudAuth) {
            setSessionHistory(loadLocalSessionHistory());
          }
          setActiveView("verify");
        }}
      />
    );
  }

  const isCloudMode = Boolean(
    session.isCloudAuth && session.userId && isSupabaseConfigured,
  );

  const navItems: Array<{
    id: NavView;
    label: string;
    icon: React.ComponentType<{ className?: string }>;
    badge?: string | number;
  }> = [
    { id: "verify", label: "Verify Paper", icon: FileSearch },
    {
      id: "report",
      label: "Active Report",
      icon: ClipboardList,
      badge: activeReport ? activeReport.total_claims : undefined,
    },
    {
      id: "history",
      label: isCloudMode ? "Cloud History" : "Session History",
      icon: History,
      badge: sessionHistory.length > 0 ? sessionHistory.length : undefined,
    },
    { id: "settings", label: "API & Settings", icon: Settings },
  ];

  const viewHeaderMeta: Record<NavView, { title: string; subtitle: string }> = {
    verify: {
      title: "Verify Scientific Paper",
      subtitle:
        "Upload a PDF manuscript or paste claim passages to cross-check citations",
    },
    report: {
      title: "Verification Report",
      subtitle:
        "Claim-by-claim verdicts, hybrid NLI ranking, atomic split, temporal consensus & calibration",
    },
    history: {
      title: isCloudMode ? "Cloud Verification History" : "Session History",
      subtitle: isCloudMode
        ? "Your verified audit reports synced securely in Supabase Postgres"
        : "Reports verified during this browser session (stored locally)",
    },
    settings: {
      title: "API & Settings",
      subtitle:
        "Live backend diagnostics, Supabase cloud status, and schema contract",
    },
  };

  const userInitial = (session.displayName || session.email || "R")
    .charAt(0)
    .toUpperCase();

  const renderSidebarContent = () => (
    <div className="flex h-full flex-col justify-between">
      <div>
        <div className="flex items-center justify-between px-5 pt-6 pb-5">
          <div className="flex items-center gap-3">
            <BrandBadge size="md" />
            <div>
              <span className="block text-sm font-semibold tracking-tight text-white">
                CiteVerify
              </span>
              <span className="block text-[11px] text-slate-400">
                Research Workbench
              </span>
            </div>
          </div>
          <button
            type="button"
            onClick={() => setMobileMenuOpen(false)}
            aria-label="Close navigation menu"
            className="rounded-lg p-1.5 text-slate-400 hover:bg-white/5 hover:text-white lg:hidden"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="px-4 pb-4">
          <button
            type="button"
            onClick={() => {
              setActiveView("verify");
              setVerifyError(null);
              setMobileMenuOpen(false);
            }}
            className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-violet-600 via-indigo-600 to-fuchsia-600 px-3.5 py-2.5 text-xs font-semibold text-white shadow-[0_10px_28px_-10px_rgba(139,92,246,0.75)] transition-all hover:from-violet-500 hover:via-indigo-500 hover:to-fuchsia-500"
          >
            <Plus className="h-4 w-4" />
            New Verification
          </button>
        </div>

        <nav aria-label="Workspace navigation" className="space-y-1 px-3">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeView === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => {
                  setActiveView(item.id);
                  setMobileMenuOpen(false);
                }}
                className={`group flex w-full items-center justify-between rounded-xl px-3.5 py-2.5 text-sm font-medium transition-all ${
                  isActive
                    ? "border border-violet-400/30 bg-violet-500/15 text-white shadow-[inset_0_1px_0_rgba(255,255,255,0.06)]"
                    : "border border-transparent text-slate-400 hover:bg-white/[0.04] hover:text-slate-100"
                }`}
              >
                <span className="flex items-center gap-3">
                  <Icon
                    className={`h-4 w-4 transition-colors ${
                      isActive
                        ? "text-violet-300"
                        : "text-slate-500 group-hover:text-slate-300"
                    }`}
                  />
                  {item.label}
                </span>

                {item.badge !== undefined ? (
                  <span
                    className={`rounded-full px-2 py-0.5 font-mono text-[11px] ${
                      isActive
                        ? "bg-violet-400/25 text-violet-100"
                        : "bg-white/[0.06] text-slate-400"
                    }`}
                  >
                    {item.badge}
                  </span>
                ) : item.id === "settings" ? (
                  <span
                    className={`h-2 w-2 rounded-full ${
                      health.status === "online"
                        ? "bg-emerald-400"
                        : health.status === "checking"
                          ? "bg-amber-400"
                          : "bg-rose-400"
                    }`}
                  />
                ) : null}
              </button>
            );
          })}
        </nav>
      </div>

      <div className="border-t border-white/[0.08] p-4">
        <div className="rounded-2xl border border-white/[0.08] bg-white/[0.03] p-3">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border border-violet-400/30 bg-gradient-to-br from-violet-500/30 to-cyan-500/30 text-sm font-semibold text-violet-100">
              {userInitial}
            </div>
            <div className="min-w-0 flex-1">
              <p className="truncate text-xs font-semibold text-white">
                {session.displayName}
              </p>
              <p className="truncate text-[11px] text-slate-400">
                {session.email}
              </p>
            </div>
          </div>
          {isCloudMode ? (
            <div className="mt-2.5 flex items-center gap-1.5 text-[11px] text-emerald-300/90">
              <Cloud className="h-3 w-3 shrink-0" />
              <span>Supabase Cloud Sync Active</span>
            </div>
          ) : (
            <div className="mt-2.5 flex items-center gap-1.5 text-[11px] text-cyan-300/90">
              <Sparkles className="h-3 w-3 shrink-0" />
              <span>Guest Demo Session (Local)</span>
            </div>
          )}
          <button
            type="button"
            onClick={handleSignOut}
            className="mt-3 flex w-full items-center justify-center gap-2 rounded-lg border border-white/10 bg-white/[0.03] px-3 py-1.5 text-xs font-medium text-slate-300 transition-colors hover:border-rose-400/35 hover:bg-rose-500/10 hover:text-rose-200"
          >
            <LogOut className="h-3.5 w-3.5" />
            {session.provider === "guest" ? "Exit Guest Mode" : "Sign out"}
          </button>
        </div>
      </div>
    </div>
  );

  return (
    <div className="relative flex min-h-screen w-full bg-[#070711] text-slate-100 selection:bg-violet-500/30 selection:text-violet-100">
      <div
        aria-hidden="true"
        className="pointer-events-none fixed inset-0 overflow-hidden"
      >
        <div className="absolute -top-40 left-1/3 h-[420px] w-[720px] rounded-full bg-[radial-gradient(ellipse_at_center,_rgba(139,92,246,0.16)_0%,_rgba(99,102,241,0.08)_45%,_transparent_72%)] blur-3xl" />
        <div className="absolute bottom-0 right-0 h-[380px] w-[480px] rounded-full bg-[radial-gradient(circle,_rgba(34,211,238,0.09)_0%,_rgba(168,85,247,0.06)_45%,_transparent_72%)] blur-3xl" />
      </div>

      <aside className="sticky top-0 z-20 hidden h-screen w-64 shrink-0 border-r border-white/[0.08] bg-[#0a0918]/85 backdrop-blur-2xl lg:block">
        {renderSidebarContent()}
      </aside>

      {mobileMenuOpen && (
        <div className="fixed inset-0 z-50 flex lg:hidden">
          <div
            className="fixed inset-0 bg-black/70 backdrop-blur-sm"
            onClick={() => setMobileMenuOpen(false)}
          />
          <aside className="relative z-10 flex h-full w-72 max-w-[82vw] flex-col border-r border-white/10 bg-[#0b0a1b] shadow-2xl">
            {renderSidebarContent()}
          </aside>
        </div>
      )}

      <div className="relative z-10 flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-20 flex items-center justify-between gap-4 border-b border-white/[0.08] bg-[#070711]/80 px-4 py-3.5 backdrop-blur-xl sm:px-8">
          <div className="flex min-w-0 items-center gap-3">
            <button
              type="button"
              onClick={() => setMobileMenuOpen(true)}
              aria-label="Open navigation menu"
              className="rounded-lg border border-white/10 bg-white/[0.04] p-2 text-slate-300 hover:text-white lg:hidden"
            >
              <Menu className="h-4 w-4" />
            </button>
            <div className="min-w-0">
              <h1 className="truncate text-base font-semibold tracking-tight text-white sm:text-lg">
                {viewHeaderMeta[activeView].title}
              </h1>
              <p className="hidden truncate text-xs text-slate-400 sm:block">
                {viewHeaderMeta[activeView].subtitle}
              </p>
            </div>
          </div>

          <div className="flex shrink-0 items-center gap-2.5">
            <button
              type="button"
              onClick={() => runHealthCheck(apiBaseUrl)}
              title="Click to re-check live backend /health"
              className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-[#0d0c1d]/90 px-3 py-1.5 text-xs font-medium text-slate-300 transition-colors hover:border-violet-400/40 hover:text-white"
            >
              <span className="relative flex h-2 w-2">
                {health.status === "online" && (
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-60" />
                )}
                <span
                  className={`relative inline-flex h-2 w-2 rounded-full ${
                    health.status === "online"
                      ? "bg-emerald-400"
                      : health.status === "checking"
                        ? "bg-amber-400"
                        : "bg-rose-400"
                  }`}
                />
              </span>
              <span className="hidden sm:inline">
                {health.status === "online"
                  ? `API Healthy${health.latencyMs !== null ? ` · ${health.latencyMs}ms` : ""}`
                  : health.status === "checking"
                    ? "Pinging API..."
                    : "API Offline"}
              </span>
              <RefreshCw
                className={`h-3 w-3 text-slate-400 ${
                  health.status === "checking" ? "animate-spin" : ""
                }`}
              />
            </button>

            <a
              href={`${apiBaseUrl}/docs`}
              target="_blank"
              rel="noopener noreferrer"
              className="hidden items-center gap-1.5 rounded-full border border-violet-400/25 bg-violet-500/10 px-3 py-1.5 text-xs font-medium text-violet-200 transition-colors hover:border-violet-400/50 hover:bg-violet-500/20 md:inline-flex"
            >
              API Docs
              <ExternalLink className="h-3 w-3" />
            </a>
          </div>
        </header>

        <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-6 sm:px-8 sm:py-8">
          {/* ================================================================ */}
          {/* VIEW 1: VERIFY PAPER WORKBENCH                                   */}
          {/* ================================================================ */}
          {activeView === "verify" && (
            <div className="space-y-6">
              <section className="relative overflow-hidden rounded-3xl border border-white/10 bg-[#0d0c1d]/80 p-5 sm:p-8 shadow-[0_24px_80px_-24px_rgba(0,0,0,0.85)] backdrop-blur-2xl">
                <div
                  aria-hidden="true"
                  className="pointer-events-none absolute inset-x-12 top-0 h-px bg-gradient-to-r from-transparent via-violet-400/60 to-transparent"
                />

                <div className="mb-6 flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
                  <div>
                    <div className="inline-flex items-center gap-1.5 rounded-full border border-violet-400/25 bg-violet-500/10 px-2.5 py-0.5 text-[11px] font-medium text-violet-200">
                      <Sparkles className="h-3 w-3 text-cyan-300" />
                      Live Verification Engine · Gaps 1–4 Active
                    </div>
                    <h2 className="mt-2 text-xl font-semibold tracking-tight text-white">
                      Submit a Manuscript for Citation Audit
                    </h2>
                  </div>

                  <div className="flex flex-wrap items-center gap-2.5">
                    <button
                      type="button"
                      onClick={handleLoadDemoSample}
                      disabled={isVerifying}
                      className="inline-flex items-center gap-1.5 rounded-xl border border-cyan-400/35 bg-cyan-500/12 px-3.5 py-2 text-xs font-medium text-cyan-200 transition-all hover:border-cyan-300/60 hover:bg-cyan-500/20 hover:text-white disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      <Sparkles className="h-3.5 w-3.5 text-cyan-300" />
                      Load 4-Gap Demo Sample
                    </button>

                    <div className="inline-flex rounded-xl border border-white/10 bg-black/35 p-1">
                      <button
                        type="button"
                        onClick={() => {
                          setInputMode("pdf");
                          setVerifyError(null);
                        }}
                        className={`inline-flex items-center gap-2 rounded-lg px-3.5 py-2 text-xs font-medium transition-all ${
                          inputMode === "pdf"
                            ? "bg-gradient-to-r from-violet-600/90 to-indigo-600/90 text-white shadow-sm"
                            : "text-slate-400 hover:text-slate-200"
                        }`}
                      >
                        <Upload className="h-3.5 w-3.5" />
                        Upload PDF
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          setInputMode("text");
                          setVerifyError(null);
                        }}
                        className={`inline-flex items-center gap-2 rounded-lg px-3.5 py-2 text-xs font-medium transition-all ${
                          inputMode === "text"
                            ? "bg-gradient-to-r from-violet-600/90 to-indigo-600/90 text-white shadow-sm"
                            : "text-slate-400 hover:text-slate-200"
                        }`}
                      >
                        <FileText className="h-3.5 w-3.5" />
                        Paste Text
                      </button>
                    </div>
                  </div>
                </div>

                <form onSubmit={handleVerifySubmit} className="space-y-5">
                  <div>
                    <label
                      htmlFor="doc-title"
                      className="mb-1.5 block text-xs font-medium text-slate-300"
                    >
                      Document Title{" "}
                      <span className="text-slate-500">(optional)</span>
                    </label>
                    <input
                      id="doc-title"
                      type="text"
                      value={documentTitle}
                      onChange={(e) => setDocumentTitle(e.target.value)}
                      placeholder="e.g., Attention Mechanisms in Biomedical Literature Review"
                      disabled={isVerifying}
                      className="w-full rounded-xl border border-white/10 bg-white/[0.03] px-4 py-2.5 text-sm text-white placeholder:text-slate-500 transition-all focus:border-violet-400/70 focus:bg-white/[0.05] focus:outline-none focus:ring-4 focus:ring-violet-500/15 disabled:opacity-60"
                    />
                  </div>

                  {inputMode === "pdf" ? (
                    <div>
                      <input
                        ref={fileInputRef}
                        type="file"
                        accept=".pdf,application/pdf"
                        onChange={(e) =>
                          handleFilePicked(e.target.files?.[0] ?? null)
                        }
                        className="hidden"
                      />

                      {!selectedFile ? (
                        <div
                          role="button"
                          tabIndex={0}
                          onClick={() =>
                            !isVerifying && fileInputRef.current?.click()
                          }
                          onKeyDown={(e) => {
                            if (
                              (e.key === "Enter" || e.key === " ") &&
                              !isVerifying
                            ) {
                              e.preventDefault();
                              fileInputRef.current?.click();
                            }
                          }}
                          onDragOver={(e) => {
                            e.preventDefault();
                            if (!isVerifying) setIsDragging(true);
                          }}
                          onDragLeave={() => setIsDragging(false)}
                          onDrop={handleDrop}
                          className={`group flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 py-12 text-center transition-all ${
                            isDragging
                              ? "border-violet-400 bg-violet-500/15"
                              : "border-white/15 bg-white/[0.02] hover:border-violet-400/50 hover:bg-violet-500/[0.05]"
                          }`}
                        >
                          <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl border border-violet-400/30 bg-violet-500/15 text-violet-300 transition-transform group-hover:scale-105">
                            <FileUp className="h-6 w-6" />
                          </div>
                          <p className="text-sm font-medium text-white">
                            Drag and drop your PDF paper here, or{" "}
                            <span className="text-violet-300 underline decoration-violet-400/50 underline-offset-4">
                              browse files
                            </span>
                          </p>
                          <p className="mt-1.5 text-xs text-slate-400">
                            Supports standard academic PDFs with inline
                            citations and reference lists
                          </p>
                        </div>
                      ) : (
                        <div className="flex flex-wrap items-center justify-between gap-4 rounded-2xl border border-violet-400/35 bg-violet-500/10 p-4">
                          <div className="flex min-w-0 items-center gap-3.5">
                            <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-violet-500/20 text-violet-200">
                              <FileText className="h-5 w-5" />
                            </div>
                            <div className="min-w-0">
                              <p className="truncate text-sm font-medium text-white">
                                {selectedFile.name}
                              </p>
                              <p className="text-xs text-slate-400">
                                PDF Document ·{" "}
                                {formatFileSize(selectedFile.size)}
                              </p>
                            </div>
                          </div>

                          <div className="flex items-center gap-2">
                            <button
                              type="button"
                              disabled={isVerifying}
                              onClick={() => fileInputRef.current?.click()}
                              className="rounded-lg border border-white/15 bg-white/[0.05] px-3 py-1.5 text-xs font-medium text-slate-200 transition-colors hover:bg-white/10 disabled:opacity-50"
                            >
                              Replace
                            </button>
                            <button
                              type="button"
                              disabled={isVerifying}
                              onClick={() => setSelectedFile(null)}
                              aria-label="Remove selected file"
                              className="rounded-lg border border-rose-500/30 bg-rose-500/10 p-1.5 text-rose-300 transition-colors hover:bg-rose-500/20 disabled:opacity-50"
                            >
                              <X className="h-4 w-4" />
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div>
                      <div className="mb-1.5 flex items-center justify-between">
                        <label
                          htmlFor="raw-paper-text"
                          className="text-xs font-medium text-slate-300"
                        >
                          Scientific Claims & Reference List
                        </label>
                        <div className="flex items-center gap-3 text-[11px] text-slate-400">
                          <span>{textMetrics.words} words</span>
                          <span>{textMetrics.chars} chars</span>
                          {rawText.length > 0 && !isVerifying && (
                            <button
                              type="button"
                              onClick={() => setRawText("")}
                              className="text-violet-300 hover:text-violet-200"
                            >
                              Clear
                            </button>
                          )}
                        </div>
                      </div>
                      <textarea
                        id="raw-paper-text"
                        rows={8}
                        value={rawText}
                        onChange={(e) => {
                          setRawText(e.target.value);
                          if (verifyError) setVerifyError(null);
                        }}
                        disabled={isVerifying}
                        placeholder="Paste paragraphs containing inline citations (e.g., [1] or Smith et al., 2022) along with their corresponding references..."
                        className="w-full resize-y rounded-2xl border border-white/10 bg-white/[0.03] p-4 text-sm leading-relaxed text-white placeholder:text-slate-500 transition-all focus:border-violet-400/70 focus:bg-white/[0.05] focus:outline-none focus:ring-4 focus:ring-violet-500/15 disabled:opacity-60"
                      />
                    </div>
                  )}

                  {verifyError && (
                    <div
                      role="alert"
                      className="flex items-start gap-3 rounded-xl border border-rose-500/35 bg-rose-500/10 p-4 text-xs leading-relaxed text-rose-200 sm:text-sm"
                    >
                      <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-rose-400" />
                      <div className="flex-1">
                        <p className="font-medium text-rose-100">
                          Verification Request Notice
                        </p>
                        <p className="mt-0.5 text-rose-200/90">{verifyError}</p>
                      </div>
                      <button
                        type="button"
                        onClick={() => setVerifyError(null)}
                        className="rounded-lg p-1 text-rose-300 hover:bg-rose-500/15"
                      >
                        <X className="h-4 w-4" />
                      </button>
                    </div>
                  )}

                  {isVerifying && (
                    <div className="rounded-2xl border border-violet-400/25 bg-violet-500/[0.07] p-5">
                      <div className="mb-4 flex items-center justify-between">
                        <div className="flex items-center gap-2.5">
                          <Loader2 className="h-4 w-4 animate-spin text-violet-300" />
                          <span className="text-sm font-medium text-white">
                            Running CiteVerify Pipeline...
                          </span>
                        </div>
                        <span className="font-mono text-xs text-violet-300">
                          Step {activePipelineStep + 1} /{" "}
                          {VERIFICATION_PIPELINE_STEPS.length}
                        </span>
                      </div>

                      <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2 lg:grid-cols-4">
                        {VERIFICATION_PIPELINE_STEPS.map((step) => {
                          const isDone = step.id < activePipelineStep;
                          const isCurrent = step.id === activePipelineStep;
                          return (
                            <div
                              key={step.id}
                              className={`rounded-xl border p-3 transition-all ${
                                isCurrent
                                  ? "border-violet-400/50 bg-violet-500/15"
                                  : isDone
                                    ? "border-emerald-500/30 bg-emerald-500/10"
                                    : "border-white/[0.06] bg-black/20 opacity-55"
                              }`}
                            >
                              <div className="flex items-center gap-2">
                                {isDone ? (
                                  <Check className="h-3.5 w-3.5 text-emerald-400" />
                                ) : isCurrent ? (
                                  <Loader2 className="h-3.5 w-3.5 animate-spin text-violet-300" />
                                ) : (
                                  <span className="h-2 w-2 rounded-full bg-slate-600" />
                                )}
                                <span className="text-xs font-semibold text-white">
                                  {step.title}
                                </span>
                              </div>
                              <p className="mt-1 text-[11px] leading-relaxed text-slate-400">
                                {step.detail}
                              </p>
                            </div>
                          );
                        })}
                      </div>

                      {showColdStartNotice && (
                        <p className="mt-3.5 flex items-center gap-2 text-xs text-amber-200/90">
                          <Clock className="h-3.5 w-3.5 shrink-0 text-amber-400" />
                          Render free-tier servers can take 15–40 seconds to
                          wake up on the first request. Waiting for live
                          response...
                        </p>
                      )}
                    </div>
                  )}

                  <div className="flex flex-col-reverse items-stretch justify-between gap-3 border-t border-white/[0.08] pt-5 sm:flex-row sm:items-center">
                    <div className="flex items-center gap-2 text-xs text-slate-400">
                      <Server className="h-3.5 w-3.5 text-violet-400" />
                      <span className="truncate font-mono">
                        {apiBaseUrl}/api/verify
                      </span>
                    </div>

                    <button
                      type="submit"
                      disabled={isVerifying}
                      className="inline-flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-violet-600 via-indigo-600 to-fuchsia-600 px-6 py-2.5 text-sm font-semibold text-white shadow-[0_12px_32px_-10px_rgba(139,92,246,0.85)] transition-all hover:from-violet-500 hover:via-indigo-500 hover:to-fuchsia-500 active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      {isVerifying ? (
                        <>
                          <Loader2 className="h-4 w-4 animate-spin" />
                          Verifying Citations...
                        </>
                      ) : (
                        <>
                          <ShieldCheck className="h-4 w-4" />
                          Verify Citations
                        </>
                      )}
                    </button>
                  </div>
                </form>
              </section>

              {/* 4-Gap Engine Capability Strip */}
              <section className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
                <div className="rounded-2xl border border-white/[0.08] bg-[#0d0c1d]/65 p-5 backdrop-blur-xl">
                  <div className="mb-3 inline-flex h-9 w-9 items-center justify-center rounded-xl border border-cyan-400/25 bg-cyan-500/12 text-cyan-300">
                    <Layers className="h-4 w-4" />
                  </div>
                  <h3 className="text-sm font-semibold text-white">
                    Gap 1 · Hybrid NLI Ranking
                  </h3>
                  <p className="mt-1.5 text-xs leading-relaxed text-slate-400">
                    Combines BM25, semantic RRF, cross-encoder scoring, and NLI
                    stance checks to catch similarity-trap contradictions.
                  </p>
                </div>

                <div className="rounded-2xl border border-white/[0.08] bg-[#0d0c1d]/65 p-5 backdrop-blur-xl">
                  <div className="mb-3 inline-flex h-9 w-9 items-center justify-center rounded-xl border border-violet-400/25 bg-violet-500/12 text-violet-300">
                    <GitBranch className="h-4 w-4" />
                  </div>
                  <h3 className="text-sm font-semibold text-white">
                    Gap 2 · Atomic Decomposition
                  </h3>
                  <p className="mt-1.5 text-xs leading-relaxed text-slate-400">
                    Splits compound claims into standalone propositions and
                    audits each half independently to expose partial
                    contradictions.
                  </p>
                </div>

                <div className="rounded-2xl border border-white/[0.08] bg-[#0d0c1d]/65 p-5 backdrop-blur-xl">
                  <div className="mb-3 inline-flex h-9 w-9 items-center justify-center rounded-xl border border-amber-400/25 bg-amber-500/12 text-amber-300">
                    <Clock className="h-4 w-4" />
                  </div>
                  <h3 className="text-sm font-semibold text-white">
                    Gap 3 · Temporal Consensus
                  </h3>
                  <p className="mt-1.5 text-xs leading-relaxed text-slate-400">
                    Tracks retractions, corrections, preprints, and newer citing
                    reviews without overwriting historical alignment.
                  </p>
                </div>

                <div className="rounded-2xl border border-white/[0.08] bg-[#0d0c1d]/65 p-5 backdrop-blur-xl">
                  <div className="mb-3 inline-flex h-9 w-9 items-center justify-center rounded-xl border border-emerald-400/25 bg-emerald-500/12 text-emerald-300">
                    <Gauge className="h-4 w-4" />
                  </div>
                  <h3 className="text-sm font-semibold text-white">
                    Gap 4 · Uncertainty & Abstention
                  </h3>
                  <p className="mt-1.5 text-xs leading-relaxed text-slate-400">
                    Calibrates probability, evidence coverage, and agreement to
                    abstain or flag for review when evidence is weak.
                  </p>
                </div>
              </section>
            </div>
          )}

          {/* ================================================================ */}
          {/* VIEW 2: ACTIVE VERIFICATION REPORT VIEWER                        */}
          {/* ================================================================ */}
          {activeView === "report" && (
            <div className="space-y-6">
              {!activeReport ? (
                <div className="flex flex-col items-center justify-center rounded-3xl border border-white/10 bg-[#0d0c1d]/75 px-6 py-16 text-center backdrop-blur-xl">
                  <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl border border-violet-400/25 bg-violet-500/12 text-violet-300">
                    <ClipboardList className="h-6 w-6" />
                  </div>
                  <h2 className="text-lg font-semibold text-white">
                    No Active Verification Report
                  </h2>
                  <p className="mt-1.5 max-w-md text-sm text-slate-400">
                    Submit a PDF manuscript or paste scientific text from the
                    Verify Paper workspace to inspect claim-by-claim verdicts
                    here.
                  </p>
                  <button
                    type="button"
                    onClick={() => setActiveView("verify")}
                    className="mt-6 inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-violet-600 via-indigo-600 to-fuchsia-600 px-5 py-2.5 text-sm font-medium text-white shadow-[0_10px_28px_-10px_rgba(139,92,246,0.8)] transition-all hover:from-violet-500 hover:via-indigo-500 hover:to-fuchsia-500"
                  >
                    Go to Verify Paper
                    <ChevronRight className="h-4 w-4" />
                  </button>
                </div>
              ) : (
                <>
                  <section className="rounded-3xl border border-white/10 bg-[#0d0c1d]/80 p-5 sm:p-7 backdrop-blur-2xl">
                    <div className="flex flex-col justify-between gap-4 md:flex-row md:items-start">
                      <div>
                        <span className="inline-flex items-center gap-1.5 rounded-full border border-cyan-400/25 bg-cyan-500/10 px-2.5 py-0.5 text-[11px] font-medium text-cyan-200">
                          Verified Document Report
                        </span>
                        <h2 className="mt-2 text-xl font-semibold tracking-tight text-white sm:text-2xl">
                          {activeReport.document_title}
                        </h2>
                        <p className="mt-1 text-xs text-slate-400">
                          {activeReport.total_claims} total claims evaluated ·
                          Showing {filteredClaims.length} matching current
                          filters
                        </p>
                      </div>

                      <div className="flex flex-col items-end gap-2">
                        <div className="flex flex-wrap items-center gap-2.5">
                          <button
                            type="button"
                            onClick={() => handleExportReport("pdf")}
                            disabled={exportingFormat !== null}
                            className="inline-flex items-center gap-2 rounded-xl border border-cyan-400/30 bg-cyan-500/10 px-3.5 py-2 text-xs font-medium text-cyan-200 transition-colors hover:border-cyan-300/60 hover:bg-cyan-500/20 disabled:cursor-not-allowed disabled:opacity-50"
                          >
                            {exportingFormat === "pdf" ? (
                              <Loader2 className="h-3.5 w-3.5 animate-spin" />
                            ) : (
                              <FileText className="h-3.5 w-3.5" />
                            )}
                            {exportingFormat === "pdf"
                              ? "Exporting PDF..."
                              : "Export PDF"}
                          </button>
                          <button
                            type="button"
                            onClick={() => handleExportReport("markdown")}
                            disabled={exportingFormat !== null}
                            className="inline-flex items-center gap-2 rounded-xl border border-white/12 bg-white/[0.04] px-3.5 py-2 text-xs font-medium text-slate-200 transition-colors hover:border-violet-400/45 hover:bg-white/[0.08] disabled:cursor-not-allowed disabled:opacity-50"
                          >
                            {exportingFormat === "markdown" ? (
                              <Loader2 className="h-3.5 w-3.5 animate-spin" />
                            ) : (
                              <Download className="h-3.5 w-3.5" />
                            )}
                            {exportingFormat === "markdown"
                              ? "Exporting MD..."
                              : "Export MD"}
                          </button>
                          <button
                            type="button"
                            onClick={handleExportJson}
                            className="inline-flex items-center gap-2 rounded-xl border border-white/12 bg-white/[0.04] px-3.5 py-2 text-xs font-medium text-slate-200 transition-colors hover:border-violet-400/45 hover:bg-white/[0.08]"
                          >
                            <Download className="h-3.5 w-3.5" />
                            Export JSON
                          </button>
                          <button
                            type="button"
                            onClick={() => setActiveView("verify")}
                            className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-violet-600 to-indigo-600 px-3.5 py-2 text-xs font-medium text-white transition-all hover:from-violet-500 hover:to-indigo-500"
                          >
                            <Plus className="h-3.5 w-3.5" />
                            Verify Another
                          </button>
                        </div>
                        {exportError && (
                          <p className="text-xs text-rose-400">{exportError}</p>
                        )}
                      </div>
                    </div>

                    {reportCounts && (
                      <>
                        <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
                          {ALL_STATUSES.map((status) => {
                            const meta = STATUS_CONFIG[status];
                            const Icon = meta.icon;
                            const count = reportCounts[status];
                            const isSelected = verdictFilter === status;

                            return (
                              <button
                                key={status}
                                type="button"
                                onClick={() =>
                                  setVerdictFilter((prev) =>
                                    prev === status ? "ALL" : status,
                                  )
                                }
                                className={`flex flex-col items-start rounded-2xl border p-3.5 text-left transition-all ${
                                  isSelected
                                    ? "border-violet-400/60 bg-violet-500/15 ring-2 ring-violet-500/20"
                                    : "border-white/[0.08] bg-white/[0.02] hover:border-white/20 hover:bg-white/[0.04]"
                                }`}
                              >
                                <div className="flex w-full items-center justify-between">
                                  <span
                                    className={`h-2 w-2 rounded-full ${meta.dotClass}`}
                                  />
                                  <Icon className="h-4 w-4 text-slate-400" />
                                </div>
                                <span className="mt-2.5 text-2xl font-bold tracking-tight text-white">
                                  {count}
                                </span>
                                <span className="mt-0.5 text-[11px] font-medium text-slate-400">
                                  {meta.shortLabel}
                                </span>
                              </button>
                            );
                          })}
                        </div>

                        {activeReport.total_claims > 0 && (
                          <div className="mt-5">
                            <div className="flex h-2 w-full overflow-hidden rounded-full bg-white/[0.06]">
                              {ALL_STATUSES.map((status) => {
                                const count = reportCounts[status];
                                if (count <= 0) return null;
                                const pct =
                                  (count / activeReport.total_claims) * 100;
                                return (
                                  <div
                                    key={`bar-${status}`}
                                    style={{ width: `${pct}%` }}
                                    title={`${STATUS_CONFIG[status].label}: ${count}`}
                                    className={`${STATUS_CONFIG[status].barClass} transition-all`}
                                  />
                                );
                              })}
                            </div>
                          </div>
                        )}
                      </>
                    )}

                    {/* Document-Level Gaps 1-4 Signal Roll-Up */}
                    {gapSummary && (
                      <div className="mt-5 grid grid-cols-2 gap-2.5 border-t border-white/[0.08] pt-4 sm:grid-cols-4">
                        <div className="rounded-xl border border-cyan-400/20 bg-cyan-500/[0.05] px-3 py-2">
                          <span className="block text-[10px] font-semibold uppercase tracking-wider text-cyan-300">
                            Gap 1 · Similarity Traps
                          </span>
                          <span className="mt-0.5 block font-mono text-base font-bold text-white">
                            {gapSummary.similarityTraps}
                          </span>
                        </div>
                        <div className="rounded-xl border border-violet-400/20 bg-violet-500/[0.05] px-3 py-2">
                          <span className="block text-[10px] font-semibold uppercase tracking-wider text-violet-300">
                            Gap 2 · Compound Claims
                          </span>
                          <span className="mt-0.5 block font-mono text-base font-bold text-white">
                            {gapSummary.compoundClaims}
                          </span>
                        </div>
                        <div className="rounded-xl border border-amber-400/20 bg-amber-500/[0.05] px-3 py-2">
                          <span className="block text-[10px] font-semibold uppercase tracking-wider text-amber-300">
                            Gap 3 · Temporal Alerts
                          </span>
                          <span className="mt-0.5 block font-mono text-base font-bold text-white">
                            {gapSummary.temporalAlerts}
                          </span>
                        </div>
                        <div className="rounded-xl border border-fuchsia-400/20 bg-fuchsia-500/[0.05] px-3 py-2">
                          <span className="block text-[10px] font-semibold uppercase tracking-wider text-fuchsia-300">
                            Gap 4 · Abstentions
                          </span>
                          <span className="mt-0.5 block font-mono text-base font-bold text-white">
                            {gapSummary.abstainedClaims}
                          </span>
                        </div>
                      </div>
                    )}
                  </section>

                  {activeReport.limitations.length > 0 && (
                    <section className="rounded-2xl border border-amber-500/25 bg-amber-500/[0.07] p-4 sm:p-5">
                      <div className="flex items-start gap-3">
                        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-400" />
                        <div>
                          <h3 className="text-xs font-semibold uppercase tracking-wider text-amber-200">
                            Verification Limitations
                          </h3>
                          <ul className="mt-2 list-disc space-y-1 pl-4 text-xs leading-relaxed text-slate-300">
                            {activeReport.limitations.map((lim, idx) => (
                              <li key={`lim-${idx}`}>{lim}</li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    </section>
                  )}

                  <section className="flex flex-col gap-3 rounded-2xl border border-white/[0.08] bg-[#0d0c1d]/75 p-4 backdrop-blur-xl md:flex-row md:items-center md:justify-between">
                    <div className="relative flex-1">
                      <Search className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
                      <input
                        type="text"
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        placeholder="Search claims, citations, passages, temporal status, or flags..."
                        className="w-full rounded-xl border border-white/10 bg-white/[0.03] py-2 pl-10 pr-4 text-sm text-white placeholder:text-slate-500 focus:border-violet-400/65 focus:outline-none focus:ring-4 focus:ring-violet-500/15"
                      />
                    </div>

                    <div className="flex flex-wrap items-center gap-2">
                      <button
                        type="button"
                        onClick={() => setOnlyHumanReview((prev) => !prev)}
                        className={`inline-flex items-center gap-1.5 rounded-xl border px-3 py-2 text-xs font-medium transition-all ${
                          onlyHumanReview
                            ? "border-amber-400/50 bg-amber-500/15 text-amber-200"
                            : "border-white/10 bg-white/[0.03] text-slate-300 hover:border-white/20"
                        }`}
                      >
                        <Filter className="h-3.5 w-3.5" />
                        Needs Human Review
                      </button>

                      {(verdictFilter !== "ALL" ||
                        searchQuery.trim() ||
                        onlyHumanReview) && (
                        <button
                          type="button"
                          onClick={() => {
                            setVerdictFilter("ALL");
                            setSearchQuery("");
                            setOnlyHumanReview(false);
                          }}
                          className="inline-flex items-center gap-1.5 rounded-xl border border-white/10 bg-white/[0.03] px-3 py-2 text-xs font-medium text-slate-300 transition-colors hover:border-violet-400/40 hover:text-white"
                        >
                          <RotateCcw className="h-3.5 w-3.5" />
                          Reset
                        </button>
                      )}
                    </div>
                  </section>

                  {filteredClaims.length === 0 ? (
                    <div className="rounded-2xl border border-white/[0.08] bg-[#0d0c1d]/70 px-6 py-12 text-center">
                      <p className="text-sm font-medium text-slate-300">
                        No claims match your current search or filter criteria.
                      </p>
                      <button
                        type="button"
                        onClick={() => {
                          setVerdictFilter("ALL");
                          setSearchQuery("");
                          setOnlyHumanReview(false);
                        }}
                        className="mt-3 text-xs font-medium text-violet-300 hover:text-violet-200"
                      >
                        Clear all filters
                      </button>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      {filteredClaims.map((claim) => (
                        <ClaimVerificationCard
                          key={claim.claim_id}
                          item={claim}
                        />
                      ))}
                    </div>
                  )}
                </>
              )}
            </div>
          )}

          {/* ================================================================ */}
          {/* VIEW 3: CLOUD & SESSION HISTORY                                  */}
          {/* ================================================================ */}
          {activeView === "history" && (
            <div className="space-y-5">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex flex-wrap items-center gap-2.5">
                  <p className="text-xs text-slate-400">
                    {sessionHistory.length} verified{" "}
                    {sessionHistory.length === 1 ? "report" : "reports"}{" "}
                    {isCloudMode
                      ? "saved in Supabase Cloud"
                      : "saved in this browser"}
                  </p>
                  {isCloudMode && (
                    <span className="inline-flex items-center gap-1 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2.5 py-0.5 text-[11px] font-medium text-emerald-300">
                      <Cloud className="h-3 w-3" />
                      Synced
                    </span>
                  )}
                </div>

                <div className="flex items-center gap-2">
                  {isCloudMode && session.userId && (
                    <button
                      type="button"
                      onClick={() =>
                        fetchCloudHistory(session.userId as string)
                      }
                      disabled={historyLoading}
                      className="inline-flex items-center gap-1.5 rounded-xl border border-white/10 bg-white/[0.04] px-3 py-1.5 text-xs font-medium text-slate-200 transition-colors hover:border-violet-400/40 disabled:opacity-50"
                    >
                      <RefreshCw
                        className={`h-3.5 w-3.5 ${historyLoading ? "animate-spin" : ""}`}
                      />
                      Refresh
                    </button>
                  )}

                  {sessionHistory.length > 0 && (
                    <>
                      {!confirmClearHistory ? (
                        <button
                          type="button"
                          onClick={() => setConfirmClearHistory(true)}
                          className="inline-flex items-center gap-1.5 rounded-xl border border-rose-500/30 bg-rose-500/10 px-3 py-1.5 text-xs font-medium text-rose-200 transition-colors hover:bg-rose-500/20"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                          Clear History
                        </button>
                      ) : (
                        <div className="inline-flex items-center gap-2 rounded-xl border border-rose-500/40 bg-rose-500/15 px-3 py-1.5 text-xs">
                          <span className="text-rose-100">Clear all?</span>
                          <button
                            type="button"
                            onClick={handleClearAllHistory}
                            className="font-semibold text-rose-200 underline underline-offset-2 hover:text-white"
                          >
                            Confirm
                          </button>
                          <button
                            type="button"
                            onClick={() => setConfirmClearHistory(false)}
                            className="text-slate-300 hover:text-white"
                          >
                            Cancel
                          </button>
                        </div>
                      )}
                    </>
                  )}
                </div>
              </div>

              {historyError && (
                <div
                  role="alert"
                  className="flex items-start gap-2.5 rounded-xl border border-rose-500/35 bg-rose-500/10 px-4 py-3 text-xs text-rose-200"
                >
                  <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-rose-400" />
                  <span>{historyError}</span>
                </div>
              )}

              {historyLoading && sessionHistory.length === 0 ? (
                <div className="flex flex-col items-center justify-center rounded-3xl border border-white/10 bg-[#0d0c1d]/75 px-6 py-16 text-center backdrop-blur-xl">
                  <Loader2 className="h-6 w-6 animate-spin text-violet-400" />
                  <p className="mt-3 text-sm text-slate-300">
                    Loading your verified reports from Supabase...
                  </p>
                </div>
              ) : sessionHistory.length === 0 ? (
                <div className="flex flex-col items-center justify-center rounded-3xl border border-white/10 bg-[#0d0c1d]/75 px-6 py-16 text-center backdrop-blur-xl">
                  <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl border border-violet-400/25 bg-violet-500/12 text-violet-300">
                    <History className="h-6 w-6" />
                  </div>
                  <h2 className="text-lg font-semibold text-white">
                    No Verification History Yet
                  </h2>
                  <p className="mt-1.5 max-w-md text-sm text-slate-400">
                    {isCloudMode
                      ? "Every manuscript you verify is automatically saved to your Supabase account so you can re-open or export it from any device."
                      : "Every real verification report returned by your backend will automatically be archived here so you can re-open or export it anytime."}
                  </p>
                  <button
                    type="button"
                    onClick={() => setActiveView("verify")}
                    className="mt-6 inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-violet-600 via-indigo-600 to-fuchsia-600 px-5 py-2.5 text-sm font-medium text-white shadow-[0_10px_28px_-10px_rgba(139,92,246,0.8)]"
                  >
                    Start First Verification
                    <ChevronRight className="h-4 w-4" />
                  </button>
                </div>
              ) : (
                <div className="space-y-3">
                  {sessionHistory.map((entry) => {
                    const counts = computeStatusCounts(entry.report);
                    return (
                      <div
                        key={entry.id}
                        className="flex flex-col justify-between gap-4 rounded-2xl border border-white/10 bg-[#0d0c1d]/80 p-5 backdrop-blur-xl transition-colors hover:border-violet-400/35 sm:flex-row sm:items-center"
                      >
                        <div className="min-w-0 space-y-2">
                          <div className="flex flex-wrap items-center gap-2">
                            <span className="rounded-md border border-violet-400/25 bg-violet-500/12 px-2 py-0.5 text-[11px] font-medium uppercase text-violet-200">
                              {entry.inputMethod}
                            </span>
                            {entry.syncedToCloud && (
                              <span className="inline-flex items-center gap-1 rounded-md border border-emerald-500/25 bg-emerald-500/10 px-2 py-0.5 text-[11px] font-medium text-emerald-300">
                                <Cloud className="h-3 w-3" />
                                Cloud
                              </span>
                            )}
                            <span className="text-xs text-slate-400">
                              {formatTimestamp(entry.verifiedAt)}
                            </span>
                            <span className="text-xs text-slate-500">·</span>
                            <span className="text-xs font-medium text-slate-300">
                              {entry.report.total_claims} claims
                            </span>
                          </div>
                          <h3 className="truncate text-base font-semibold text-white">
                            {entry.report.document_title}
                          </h3>
                          <div className="flex flex-wrap gap-1.5">
                            {ALL_STATUSES.map((st) =>
                              counts[st] > 0 ? (
                                <span
                                  key={`${entry.id}-${st}`}
                                  className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium ${STATUS_CONFIG[st].badgeClass}`}
                                >
                                  {STATUS_CONFIG[st].shortLabel}: {counts[st]}
                                </span>
                              ) : null,
                            )}
                          </div>
                        </div>

                        <div className="flex shrink-0 items-center gap-2">
                          <button
                            type="button"
                            onClick={() => handleOpenHistoryEntry(entry)}
                            className="inline-flex items-center gap-1.5 rounded-xl border border-violet-400/35 bg-violet-500/15 px-3.5 py-2 text-xs font-medium text-violet-100 transition-colors hover:bg-violet-500/25"
                          >
                            Open Report
                            <ChevronRight className="h-3.5 w-3.5" />
                          </button>
                          <button
                            type="button"
                            onClick={() => handleDeleteHistoryEntry(entry)}
                            aria-label="Delete history entry"
                            className="rounded-xl border border-white/10 bg-white/[0.03] p-2 text-slate-400 transition-colors hover:border-rose-500/35 hover:bg-rose-500/10 hover:text-rose-300"
                          >
                            <Trash2 className="h-4 w-4" />
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {/* ================================================================ */}
          {/* VIEW 4: API & SETTINGS                                           */}
          {/* ================================================================ */}
          {activeView === "settings" && (
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              <section className="rounded-3xl border border-white/10 bg-[#0d0c1d]/80 p-6 backdrop-blur-2xl">
                <div className="mb-5 flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-violet-400/30 bg-violet-500/12 text-violet-300">
                    <Link2 className="h-5 w-5" />
                  </div>
                  <div>
                    <h2 className="text-base font-semibold text-white">
                      Backend API Endpoint
                    </h2>
                    <p className="text-xs text-slate-400">
                      Defaults automatically to{" "}
                      <code className="text-violet-300">VITE_API_URL</code>
                    </p>
                  </div>
                </div>

                <form onSubmit={handleSaveApiUrl} className="space-y-4">
                  <div>
                    <label
                      htmlFor="api-base-url"
                      className="mb-1.5 block text-xs font-medium text-slate-300"
                    >
                      Base API URL
                    </label>
                    <input
                      id="api-base-url"
                      type="url"
                      value={apiUrlInput}
                      onChange={(e) => setApiUrlInput(e.target.value)}
                      placeholder="https://citeverify-api.onrender.com"
                      className="w-full rounded-xl border border-white/10 bg-white/[0.04] px-3.5 py-2.5 font-mono text-xs text-white placeholder:text-slate-500 focus:border-violet-400/70 focus:outline-none focus:ring-4 focus:ring-violet-500/15"
                    />
                  </div>

                  <div className="flex flex-wrap items-center gap-2.5">
                    <button
                      type="submit"
                      className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-violet-600 to-indigo-600 px-4 py-2 text-xs font-semibold text-white transition-all hover:from-violet-500 hover:to-indigo-500"
                    >
                      <Check className="h-3.5 w-3.5" />
                      Apply Endpoint
                    </button>
                    <button
                      type="button"
                      onClick={handleResetApiUrl}
                      className="inline-flex items-center gap-1.5 rounded-xl border border-white/10 bg-white/[0.04] px-3.5 py-2 text-xs font-medium text-slate-300 hover:bg-white/[0.08]"
                    >
                      <RotateCcw className="h-3.5 w-3.5" />
                      Reset Default
                    </button>
                    <a
                      href={`${apiBaseUrl}/docs`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="ml-auto inline-flex items-center gap-1.5 text-xs font-medium text-violet-300 hover:text-violet-200"
                    >
                      Open /docs
                      <ExternalLink className="h-3.5 w-3.5" />
                    </a>
                  </div>

                  {settingsSavedNotice && (
                    <p className="inline-flex items-center gap-1.5 text-xs text-emerald-300">
                      <CheckCircle2 className="h-3.5 w-3.5" />
                      Endpoint updated and re-checked live.
                    </p>
                  )}
                </form>
              </section>

              <section className="rounded-3xl border border-white/10 bg-[#0d0c1d]/80 p-6 backdrop-blur-2xl">
                <div className="mb-5 flex items-center justify-between gap-3">
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-cyan-400/30 bg-cyan-500/12 text-cyan-300">
                      <Activity className="h-5 w-5" />
                    </div>
                    <div>
                      <h2 className="text-base font-semibold text-white">
                        Live /health Diagnostics
                      </h2>
                      <p className="text-xs text-slate-400">
                        Real-time ping against FastAPI backend
                      </p>
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => runHealthCheck(apiBaseUrl)}
                    className="inline-flex items-center gap-1.5 rounded-xl border border-white/10 bg-white/[0.04] px-3 py-1.5 text-xs font-medium text-slate-200 hover:border-violet-400/40"
                  >
                    <RefreshCw
                      className={`h-3.5 w-3.5 ${health.status === "checking" ? "animate-spin" : ""}`}
                    />
                    Re-test
                  </button>
                </div>

                <dl className="grid grid-cols-3 gap-3 text-xs">
                  <div className="rounded-xl border border-white/[0.07] bg-black/25 p-3">
                    <dt className="text-slate-400">Status</dt>
                    <dd className="mt-1 font-semibold capitalize text-white">
                      {health.status}
                    </dd>
                  </div>
                  <div className="rounded-xl border border-white/[0.07] bg-black/25 p-3">
                    <dt className="text-slate-400">HTTP Code</dt>
                    <dd className="mt-1 font-mono font-semibold text-white">
                      {health.httpCode ?? "—"}
                    </dd>
                  </div>
                  <div className="rounded-xl border border-white/[0.07] bg-black/25 p-3">
                    <dt className="text-slate-400">Latency</dt>
                    <dd className="mt-1 font-mono font-semibold text-white">
                      {health.latencyMs !== null
                        ? `${health.latencyMs} ms`
                        : "—"}
                    </dd>
                  </div>
                </dl>

                {health.errorMessage && (
                  <p className="mt-3 rounded-xl border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-xs text-rose-200">
                    {health.errorMessage}
                  </p>
                )}

                <div className="mt-4">
                  <span className="mb-1.5 block text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                    Response Payload
                  </span>
                  <pre className="max-h-36 overflow-auto rounded-xl border border-white/[0.07] bg-black/40 p-3 font-mono text-xs text-cyan-200/90">
                    {health.payload
                      ? JSON.stringify(health.payload, null, 2)
                      : "// No JSON payload received yet"}
                  </pre>
                </div>
              </section>

              <section className="rounded-3xl border border-white/10 bg-[#0d0c1d]/80 p-6 backdrop-blur-2xl lg:col-span-2">
                <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-emerald-400/25 bg-emerald-500/12 text-emerald-300">
                      <Database className="h-5 w-5" />
                    </div>
                    <div>
                      <h2 className="text-base font-semibold text-white">
                        Supabase Auth & Cloud History
                      </h2>
                      <p className="text-xs text-slate-400">
                        Row-Level-Security isolated Postgres table{" "}
                        <code className="text-emerald-300">
                          public.verification_reports
                        </code>
                      </p>
                    </div>
                  </div>
                  <span
                    className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-medium ${
                      isCloudMode
                        ? "border-emerald-500/30 bg-emerald-500/12 text-emerald-300"
                        : "border-amber-500/30 bg-amber-500/12 text-amber-300"
                    }`}
                  >
                    <Cloud className="h-3.5 w-3.5" />
                    {isCloudMode
                      ? "Connected & Syncing"
                      : "Local Fallback Mode"}
                  </span>
                </div>

                <dl className="grid grid-cols-1 gap-3 text-xs sm:grid-cols-3">
                  <div className="rounded-xl border border-white/[0.07] bg-black/25 p-3">
                    <dt className="text-slate-400">Signed-In Account</dt>
                    <dd className="mt-1 truncate font-medium text-white">
                      {session.email}
                    </dd>
                  </div>
                  <div className="rounded-xl border border-white/[0.07] bg-black/25 p-3">
                    <dt className="text-slate-400">Auth Provider</dt>
                    <dd className="mt-1 font-medium capitalize text-white">
                      {isCloudMode
                        ? `Supabase (${session.provider})`
                        : session.provider === "guest"
                          ? "Guest Demo (Local)"
                          : "Local Session"}
                    </dd>
                  </div>
                  <div className="rounded-xl border border-white/[0.07] bg-black/25 p-3">
                    <dt className="text-slate-400">
                      {isCloudMode ? "Stored Cloud Reports" : "Saved Reports"}
                    </dt>
                    <dd className="mt-1 font-mono font-semibold text-white">
                      {sessionHistory.length}
                    </dd>
                  </div>
                </dl>
              </section>

              <section className="rounded-3xl border border-white/10 bg-[#0d0c1d]/80 p-6 backdrop-blur-2xl lg:col-span-2">
                <div className="mb-4 flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-emerald-400/25 bg-emerald-500/12 text-emerald-300">
                    <Code2 className="h-5 w-5" />
                  </div>
                  <div>
                    <h2 className="text-base font-semibold text-white">
                      Expected Response Contract (backend/schemas.py)
                    </h2>
                    <p className="text-xs text-slate-400">
                      Frontend types mirror VerificationReport and
                      ClaimVerificationResult including Gaps 1–4
                    </p>
                  </div>
                </div>

                <pre className="overflow-x-auto rounded-2xl border border-white/[0.07] bg-black/40 p-4 font-mono text-xs leading-relaxed text-slate-300">
                  {`POST ${apiBaseUrl}/api/verify  ->  VerificationReport
{
  "document_title": string,
  "total_claims": number,
  "summary_counts": Record<ClaimStatus, number>,
  "results": ClaimVerificationResult[] // + evidence_ranking, claim_decomposition, temporal_assessment, uncertainty_calibration
  "limitations": string[]
}`}
                </pre>
              </section>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
