import React, { useState, useEffect, useMemo, useRef, useCallback } from 'react';
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
  Plus,
  Server,
  Filter,
} from 'lucide-react';

// ============================================================================
// 1. Strict Backend Contract Types (Mirrors backend/schemas.py Exactly)
// ============================================================================

export type ClaimStatus =
  | 'SUPPORTED'
  | 'PARTIALLY_SUPPORTED'
  | 'UNSUPPORTED'
  | 'CONTRADICTORY'
  | 'POTENTIALLY_OUTDATED';

export type ConfidenceLevel = 'HIGH' | 'MEDIUM' | 'LOW';

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

type HealthState = 'checking' | 'online' | 'offline';
type InputMode = 'pdf' | 'text';
type VerdictFilter = 'ALL' | ClaimStatus;
type NavView = 'verify' | 'report' | 'history' | 'settings';

interface AuthSession {
  email: string;
  displayName: string;
  provider: 'email' | 'google';
}

interface SessionHistoryEntry {
  id: string;
  verifiedAt: string;
  inputMethod: InputMode;
  report: VerificationReport;
}

interface HealthCheckDiagnostics {
  status: HealthState;
  httpCode: number | null;
  latencyMs: number | null;
  checkedAt: string | null;
  payload: unknown;
  errorMessage: string | null;
}

// ============================================================================
// 3. Semantic Design Tokens & Constants (Zero Hardcoded Data)
// ============================================================================

const DEFAULT_API_URL = 'https://citeverify-api.onrender.com';
const HISTORY_STORAGE_KEY = 'citeverify_session_history';

const ALL_STATUSES: ClaimStatus[] = [
  'SUPPORTED',
  'PARTIALLY_SUPPORTED',
  'UNSUPPORTED',
  'CONTRADICTORY',
  'POTENTIALLY_OUTDATED',
];

const VERIFICATION_PIPELINE_STEPS = [
  {
    id: 0,
    title: 'Parsing Claims',
    detail: 'Extracting assertions, inline markers, and bibliography entries',
  },
  {
    id: 1,
    title: 'Retrieving Sources',
    detail: 'Resolving DOIs, abstracts, and metadata from academic indices',
  },
  {
    id: 2,
    title: 'Cross-Checking Evidence',
    detail: 'Evaluating semantic alignment, retractions, and temporal validity',
  },
  {
    id: 3,
    title: 'Compiling Report',
    detail: 'Assigning verdicts, confidence scores, and human-review flags',
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
    label: 'Supported',
    shortLabel: 'Supported',
    badgeClass: 'border-emerald-500/30 bg-emerald-500/12 text-emerald-300',
    dotClass: 'bg-emerald-400',
    barClass: 'bg-emerald-400',
    borderAccent: 'border-l-emerald-400/80',
    cardGlow: 'hover:border-emerald-500/30',
    icon: CheckCircle2,
  },
  PARTIALLY_SUPPORTED: {
    label: 'Partially Supported',
    shortLabel: 'Partial',
    badgeClass: 'border-amber-500/30 bg-amber-500/12 text-amber-300',
    dotClass: 'bg-amber-400',
    barClass: 'bg-amber-400',
    borderAccent: 'border-l-amber-400/80',
    cardGlow: 'hover:border-amber-500/30',
    icon: HelpCircle,
  },
  UNSUPPORTED: {
    label: 'Unsupported',
    shortLabel: 'Unsupported',
    badgeClass: 'border-slate-500/35 bg-slate-500/12 text-slate-300',
    dotClass: 'bg-slate-400',
    barClass: 'bg-slate-400',
    borderAccent: 'border-l-slate-400/75',
    cardGlow: 'hover:border-slate-400/30',
    icon: AlertCircle,
  },
  CONTRADICTORY: {
    label: 'Contradictory',
    shortLabel: 'Contradictory',
    badgeClass: 'border-rose-500/35 bg-rose-500/12 text-rose-300',
    dotClass: 'bg-rose-400',
    barClass: 'bg-rose-400',
    borderAccent: 'border-l-rose-400/80',
    cardGlow: 'hover:border-rose-500/30',
    icon: XCircle,
  },
  POTENTIALLY_OUTDATED: {
    label: 'Potentially Outdated',
    shortLabel: 'Outdated',
    badgeClass: 'border-orange-500/35 bg-orange-500/12 text-orange-300',
    dotClass: 'bg-orange-400',
    barClass: 'bg-gradient-to-r from-violet-400 to-orange-400',
    borderAccent: 'border-l-orange-400/80',
    cardGlow: 'hover:border-orange-500/30',
    icon: Clock,
  },
};

const CONFIDENCE_CONFIG: Record<ConfidenceLevel, { label: string; className: string }> = {
  HIGH: {
    label: 'High Confidence',
    className: 'border-cyan-400/30 bg-cyan-400/10 text-cyan-200',
  },
  MEDIUM: {
    label: 'Medium Confidence',
    className: 'border-violet-400/30 bg-violet-400/12 text-violet-200',
  },
  LOW: {
    label: 'Low Confidence',
    className: 'border-slate-500/30 bg-slate-500/10 text-slate-300',
  },
};

// ============================================================================
// 4. Environment, Normalization & Storage Helpers
// ============================================================================

/**
 * Safely resolves VITE_API_URL in both Vite runtime and standalone preview environments.
 */
function resolveEnvApiUrl(): string {
  try {
    const envObj = (import.meta as unknown as { env?: Record<string, string | undefined> })?.env;
    const raw = envObj?.VITE_API_URL;
    if (typeof raw === 'string' && raw.trim().length > 0) {
      return raw.trim().replace(/\/+$/, '');
    }
  } catch {
    // Fallback cleanly when import.meta.env is unavailable
  }
  return DEFAULT_API_URL;
}

/**
 * Normalizes a live /api/verify JSON payload strictly against VerificationReport.
 */
function normalizeReport(raw: unknown): VerificationReport {
  const data = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>;
  const rawResults = Array.isArray(data.results) ? data.results : [];

  const results: ClaimVerificationResult[] = rawResults.map((entry, idx) => {
    const r = (entry && typeof entry === 'object' ? entry : {}) as Record<string, unknown>;
    const rawVerdict = String(r.verdict ?? 'UNSUPPORTED').toUpperCase();
    const verdict: ClaimStatus = (ALL_STATUSES as string[]).includes(rawVerdict)
      ? (rawVerdict as ClaimStatus)
      : 'UNSUPPORTED';

    const rawConfidence = String(r.confidence ?? 'LOW').toUpperCase();
    const confidence: ConfidenceLevel =
      rawConfidence === 'HIGH' || rawConfidence === 'MEDIUM' || rawConfidence === 'LOW'
        ? rawConfidence
        : 'LOW';

    return {
      claim_id: typeof r.claim_id === 'number' ? r.claim_id : idx + 1,
      claim_text: typeof r.claim_text === 'string' ? r.claim_text : '',
      matched_citation: typeof r.matched_citation === 'string' ? r.matched_citation : null,
      evidence_passage: typeof r.evidence_passage === 'string' ? r.evidence_passage : null,
      verdict,
      explanation: typeof r.explanation === 'string' ? r.explanation : '',
      confidence,
      uncertainty_note: typeof r.uncertainty_note === 'string' ? r.uncertainty_note : '',
      needs_human_review:
        typeof r.needs_human_review === 'boolean' ? r.needs_human_review : true,
      source_url: typeof r.source_url === 'string' ? r.source_url : null,
      mismatch_flags: Array.isArray(r.mismatch_flags)
        ? r.mismatch_flags.filter((f): f is string => typeof f === 'string')
        : [],
    };
  });

  const rawSummary =
    data.summary_counts && typeof data.summary_counts === 'object'
      ? (data.summary_counts as Record<string, unknown>)
      : {};

  const summary_counts: Record<string, number> = {};
  for (const [key, val] of Object.entries(rawSummary)) {
    const num = Number(val);
    if (!Number.isNaN(num)) {
      summary_counts[key] = num;
    }
  }

  const limitations = Array.isArray(data.limitations)
    ? data.limitations.filter((l): l is string => typeof l === 'string')
    : [];

  return {
    document_title:
      typeof data.document_title === 'string' && data.document_title.trim().length > 0
        ? data.document_title
        : 'Untitled Research Document',
    total_claims: typeof data.total_claims === 'number' ? data.total_claims : results.length,
    summary_counts,
    results,
    limitations,
  };
}

/**
 * Computes normalized counts across all 5 ClaimStatus buckets from a report.
 */
function computeStatusCounts(report: VerificationReport): Record<ClaimStatus, number> {
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

/**
 * Safely reads persisted session history from localStorage (starts empty).
 */
function loadSessionHistory(): SessionHistoryEntry[] {
  try {
    const raw = window.localStorage.getItem(HISTORY_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (item): item is SessionHistoryEntry =>
        Boolean(item && typeof item.id === 'string' && item.report && Array.isArray(item.report.results))
    );
  } catch {
    return [];
  }
}

function saveSessionHistory(entries: SessionHistoryEntry[]): void {
  try {
    window.localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(entries));
  } catch {
    // Ignore storage quota or private browsing restrictions
  }
}

function deriveDisplayName(email: string): string {
  const localPart = email.split('@')[0] || 'Researcher';
  return localPart
    .replace(/[._-]+/g, ' ')
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
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return iso;
  }
}

// ============================================================================
// 5. Built-In Icon Branding & Official Google "G" SVG (Zero External Images)
// ============================================================================

const BrandBadge: React.FC<{ size?: 'sm' | 'md' | 'lg' }> = ({ size = 'md' }) => {
  const dimensions =
    size === 'lg'
      ? 'h-14 w-14 rounded-2xl'
      : size === 'sm'
      ? 'h-8 w-8 rounded-lg'
      : 'h-9 w-9 rounded-xl';
  const iconSize =
    size === 'lg' ? 'h-7 w-7' : size === 'sm' ? 'h-4 w-4' : 'h-5 w-5';

  return (
    <div
      className={`relative inline-flex shrink-0 items-center justify-center ${dimensions} bg-gradient-to-br from-violet-500 via-indigo-500 to-fuchsia-500 p-[1px] shadow-[0_0_28px_-6px_rgba(139,92,246,0.75)]`}
    >
      <div
        className={`flex h-full w-full items-center justify-center ${dimensions} bg-gradient-to-br from-[#1b1238] via-[#130f2b] to-[#0d1428] text-violet-200`}
      >
        <ShieldCheck className={`${iconSize} text-cyan-300 drop-shadow-[0_0_8px_rgba(34,211,238,0.5)]`} />
      </div>
    </div>
  );
};

const GoogleIcon: React.FC<{ className?: string }> = ({ className = 'h-4 w-4' }) => (
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
// 6. Aurora Violet Login Screen (Entry Gate)
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
  const [email, setEmail] = useState<string>('');
  const [password, setPassword] = useState<string>('');
  const [showPassword, setShowPassword] = useState<boolean>(false);
  const [formNotice, setFormNotice] = useState<{ type: 'error' | 'info'; text: string } | null>(
    null
  );
  const [submittingVia, setSubmittingVia] = useState<'email' | 'google' | null>(null);

  const handleEmailSignIn = (e: React.FormEvent) => {
    e.preventDefault();
    setFormNotice(null);

    const cleanedEmail = email.trim();
    if (!cleanedEmail || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(cleanedEmail)) {
      setFormNotice({ type: 'error', text: 'Please enter a valid email address.' });
      return;
    }

    if (password.length < 6) {
      setFormNotice({
        type: 'error',
        text: 'Please enter a password with at least 6 characters.',
      });
      return;
    }

    setSubmittingVia('email');
    window.setTimeout(() => {
      setSubmittingVia(null);
      onLoginSuccess({
        email: cleanedEmail,
        displayName: deriveDisplayName(cleanedEmail),
        provider: 'email',
      });
    }, 450);
  };

  const handleGoogleSignIn = () => {
    setFormNotice(null);
    setSubmittingVia('google');
    window.setTimeout(() => {
      setSubmittingVia(null);
      onLoginSuccess({
        email: 'google.account@citeverify.app',
        displayName: 'Google Researcher',
        provider: 'google',
      });
    }, 450);
  };

  const healthLabel =
    health.status === 'online'
      ? `API Online${health.latencyMs !== null ? ` · ${health.latencyMs}ms` : ''}`
      : health.status === 'checking'
      ? 'Checking API...'
      : 'API Waking / Offline';

  return (
    <div className="relative flex min-h-screen w-full flex-col justify-between overflow-hidden bg-[#070711] text-slate-100 selection:bg-violet-500/30 selection:text-violet-100">
      {/* Ambient Aurora Violet, Indigo & Cyan Radial Glows */}
      <div aria-hidden="true" className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute -top-48 left-1/2 h-[560px] w-[980px] -translate-x-1/2 rounded-full bg-[radial-gradient(ellipse_at_center,_rgba(139,92,246,0.28)_0%,_rgba(99,102,241,0.16)_38%,_transparent_72%)] blur-3xl" />
        <div className="absolute -right-28 top-12 h-[420px] w-[480px] rounded-full bg-[radial-gradient(circle,_rgba(34,211,238,0.14)_0%,_rgba(168,85,247,0.1)_45%,_transparent_72%)] blur-3xl" />
        <div className="absolute -bottom-36 -left-24 h-[460px] w-[540px] rounded-full bg-[radial-gradient(circle,_rgba(217,70,239,0.14)_0%,_rgba(99,102,241,0.14)_45%,_transparent_72%)] blur-3xl" />
        <div
          className="absolute inset-0 opacity-[0.035]"
          style={{
            backgroundImage:
              'linear-gradient(to right, rgba(255,255,255,0.6) 1px, transparent 1px), linear-gradient(to bottom, rgba(255,255,255,0.6) 1px, transparent 1px)',
            backgroundSize: '52px 52px',
            maskImage: 'radial-gradient(ellipse at 50% 35%, black 15%, transparent 72%)',
            WebkitMaskImage: 'radial-gradient(ellipse at 50% 35%, black 15%, transparent 72%)',
          }}
        />
      </div>

      {/* Minimal Top Bar */}
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

        <button
          type="button"
          onClick={onRefreshHealth}
          title="Ping live backend /health endpoint"
          className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-[#0d0c1d]/75 px-3.5 py-1.5 text-xs font-medium text-slate-300 backdrop-blur-xl transition-colors hover:border-violet-400/40 hover:text-white"
        >
          <span className="relative flex h-2 w-2">
            {health.status === 'online' && (
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-60" />
            )}
            <span
              className={`relative inline-flex h-2 w-2 rounded-full ${
                health.status === 'online'
                  ? 'bg-emerald-400'
                  : health.status === 'checking'
                  ? 'bg-amber-400'
                  : 'bg-rose-400'
              }`}
            />
          </span>
          <span>{healthLabel}</span>
        </button>
      </header>

      {/* Centered Frosted-Glass Login Card */}
      <main className="relative z-10 flex flex-1 items-center justify-center px-4 py-10">
        <div className="w-full max-w-md">
          <div className="relative rounded-3xl border border-white/10 bg-[#0d0c1d]/80 p-7 sm:p-9 shadow-[0_28px_90px_-18px_rgba(0,0,0,0.9),0_0_65px_-22px_rgba(139,92,246,0.35)] ring-1 ring-violet-400/15 backdrop-blur-2xl">
            {/* Specular Aurora Hairline */}
            <div
              aria-hidden="true"
              className="pointer-events-none absolute inset-x-10 top-0 h-px bg-gradient-to-r from-transparent via-violet-400/70 to-transparent"
            />

            {/* Header */}
            <div className="mb-7 text-center">
              <div className="mb-4 flex justify-center">
                <BrandBadge size="lg" />
              </div>
              <h1 className="text-2xl font-semibold tracking-tight text-white">
                Sign in to CiteVerify
              </h1>
              <p className="mt-1.5 text-sm text-slate-400">
                Access your scientific claim &amp; citation verification workbench
              </p>
            </div>

            {/* Inline Validation / Notice */}
            {formNotice && (
              <div
                role="alert"
                className={`mb-5 flex items-start gap-2.5 rounded-xl border px-3.5 py-2.5 text-xs leading-relaxed ${
                  formNotice.type === 'error'
                    ? 'border-rose-500/35 bg-rose-500/10 text-rose-200'
                    : 'border-violet-400/35 bg-violet-500/10 text-violet-200'
                }`}
              >
                <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                <span>{formNotice.text}</span>
              </div>
            )}

            {/* Email & Password Form */}
            <form onSubmit={handleEmailSignIn} className="space-y-4" noValidate>
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
                  <button
                    type="button"
                    onClick={() =>
                      setFormNotice({
                        type: 'info',
                        text: 'Enter any valid email and 6+ character password to enter the workspace.',
                      })
                    }
                    className="text-xs font-medium text-violet-300/90 transition-colors hover:text-violet-200"
                  >
                    Forgot password?
                  </button>
                </div>
                <div className="group relative">
                  <Lock className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500 transition-colors group-focus-within:text-violet-400" />
                  <input
                    id="login-password"
                    type={showPassword ? 'text' : 'password'}
                    autoComplete="current-password"
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
                    aria-label={showPassword ? 'Hide password' : 'Show password'}
                    className="absolute right-3 top-1/2 -translate-y-1/2 rounded-lg p-1 text-slate-500 transition-colors hover:text-slate-200"
                  >
                    {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </button>
                </div>
              </div>

              {/* Primary Aurora Violet Sign In CTA */}
              <button
                type="submit"
                disabled={submittingVia !== null}
                className="group relative mt-2 flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-violet-600 via-indigo-600 to-fuchsia-600 px-4 py-2.5 text-sm font-medium text-white shadow-[0_12px_32px_-10px_rgba(139,92,246,0.8)] transition-all hover:from-violet-500 hover:via-indigo-500 hover:to-fuchsia-500 hover:shadow-[0_14px_38px_-8px_rgba(168,85,247,0.9)] active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-65"
              >
                {submittingVia === 'email' ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    <span>Signing in...</span>
                  </>
                ) : (
                  <>
                    <span>Sign In</span>
                    <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
                  </>
                )}
              </button>
            </form>

            {/* Clean Divider */}
            <div className="relative my-6 flex items-center justify-center">
              <div className="h-px flex-1 bg-gradient-to-r from-transparent via-white/12 to-white/12" />
              <span className="px-3 text-[11px] font-medium uppercase tracking-widest text-slate-500">
                or continue with
              </span>
              <div className="h-px flex-1 bg-gradient-to-l from-transparent via-white/12 to-white/12" />
            </div>

            {/* Continue with Google Button */}
            <button
              type="button"
              onClick={handleGoogleSignIn}
              disabled={submittingVia !== null}
              className="flex w-full items-center justify-center gap-3 rounded-xl border border-white/12 bg-white/[0.04] px-4 py-2.5 text-sm font-medium text-slate-100 transition-all hover:border-violet-400/45 hover:bg-white/[0.08] active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-65"
            >
              {submittingVia === 'google' ? (
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
          </div>
        </div>
      </main>

      <footer className="relative z-10 pb-6 text-center text-xs text-slate-500">
        CiteVerify &middot; Scientific Claim &amp; Citation Verification Platform
      </footer>
    </div>
  );
};

// ============================================================================
// 7. Claim Verification Card (Report Viewer)
// ============================================================================

const ClaimVerificationCard: React.FC<{ item: ClaimVerificationResult }> = ({ item }) => {
  const statusMeta = STATUS_CONFIG[item.verdict] ?? STATUS_CONFIG.UNSUPPORTED;
  const confidenceMeta = CONFIDENCE_CONFIG[item.confidence] ?? CONFIDENCE_CONFIG.LOW;
  const StatusIcon = statusMeta.icon;

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
        </div>

        {item.needs_human_review && (
          <span className="inline-flex items-center gap-1.5 rounded-full border border-amber-500/35 bg-amber-500/12 px-2.5 py-0.5 text-xs font-medium text-amber-300">
            <AlertTriangle className="h-3.5 w-3.5" />
            Needs Human Review
          </span>
        )}
      </div>

      {/* Prominent Claim Text */}
      <h3 className="mb-5 text-[15px] font-medium leading-relaxed text-white sm:text-base">
        {item.claim_text}
      </h3>

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
                  <span className="italic text-slate-500">No explanation provided.</span>
                )}
              </p>
            </div>

            {item.uncertainty_note && item.uncertainty_note.trim().length > 0 && (
              <div className="flex items-start gap-2.5 rounded-lg border border-amber-500/20 bg-amber-500/[0.06] px-3 py-2.5 text-xs leading-relaxed text-slate-300">
                <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-400" />
                <div>
                  <span className="font-semibold text-amber-200">Uncertainty Note: </span>
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
                  &ldquo;{item.evidence_passage}&rdquo;
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
    </article>
  );
};

// ============================================================================
// 8. Main Application Component (Login Gate + Complete Post-Login SaaS App)
// ============================================================================

export default function App() {
  // Authentication session state (starts on Aurora Violet login gate)
  const [session, setSession] = useState<AuthSession | null>(null);

  // Navigation & mobile sidebar drawer
  const [activeView, setActiveView] = useState<NavView>('verify');
  const [mobileMenuOpen, setMobileMenuOpen] = useState<boolean>(false);

  // Live API configuration & diagnostics
  const [apiBaseUrl, setApiBaseUrl] = useState<string>(resolveEnvApiUrl);
  const [apiUrlInput, setApiUrlInput] = useState<string>(resolveEnvApiUrl);
  const [settingsSavedNotice, setSettingsSavedNotice] = useState<boolean>(false);
  const [health, setHealth] = useState<HealthCheckDiagnostics>({
    status: 'checking',
    httpCode: null,
    latencyMs: null,
    checkedAt: null,
    payload: null,
    errorMessage: null,
  });

  // View 1: Verify Paper input state
  const [inputMode, setInputMode] = useState<InputMode>('pdf');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [documentTitle, setDocumentTitle] = useState<string>('');
  const [rawText, setRawText] = useState<string>('');
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Verification execution state
  const [isVerifying, setIsVerifying] = useState<boolean>(false);
  const [activePipelineStep, setActivePipelineStep] = useState<number>(0);
  const [showColdStartNotice, setShowColdStartNotice] = useState<boolean>(false);
  const [verifyError, setVerifyError] = useState<string | null>(null);

  // View 2: Active report & exploration filters
  const [activeReport, setActiveReport] = useState<VerificationReport | null>(null);
  const [verdictFilter, setVerdictFilter] = useState<VerdictFilter>('ALL');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [onlyHumanReview, setOnlyHumanReview] = useState<boolean>(false);

  // View 3: Session History (starts cleanly empty; only real verified reports stored)
  const [sessionHistory, setSessionHistory] = useState<SessionHistoryEntry[]>(loadSessionHistory);
  const [confirmClearHistory, setConfirmClearHistory] = useState<boolean>(false);

  // --------------------------------------------------------------------------
  // Live GET /health Diagnostics
  // --------------------------------------------------------------------------
  const runHealthCheck = useCallback(async (targetUrl: string) => {
    setHealth((prev) => ({ ...prev, status: 'checking', errorMessage: null }));
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), 14000);
    const startTime = performance.now();

    try {
      const response = await fetch(`${targetUrl}/health`, {
        method: 'GET',
        signal: controller.signal,
      });
      const latencyMs = Math.round(performance.now() - startTime);
      const jsonPayload = await response.json().catch(() => null);

      if (!response.ok) {
        setHealth({
          status: 'offline',
          httpCode: response.status,
          latencyMs,
          checkedAt: new Date().toISOString(),
          payload: jsonPayload,
          errorMessage: `Backend responded with HTTP ${response.status}.`,
        });
        return;
      }

      setHealth({
        status: 'online',
        httpCode: response.status,
        latencyMs,
        checkedAt: new Date().toISOString(),
        payload: jsonPayload,
        errorMessage: null,
      });
    } catch (err) {
      const latencyMs = Math.round(performance.now() - startTime);
      setHealth({
        status: 'offline',
        httpCode: null,
        latencyMs,
        checkedAt: new Date().toISOString(),
        payload: null,
        errorMessage:
          err instanceof Error && err.name === 'AbortError'
            ? 'Health check timed out after 14s (Render free tier may be waking up).'
            : 'Unable to reach backend /health endpoint. Check the API URL or server status.',
      });
    } finally {
      window.clearTimeout(timeoutId);
    }
  }, []);

  useEffect(() => {
    runHealthCheck(apiBaseUrl);
  }, [apiBaseUrl, runHealthCheck]);

  // Multi-step progress ticker + Render cold-start notice while verifying
  useEffect(() => {
    if (!isVerifying) {
      setActivePipelineStep(0);
      setShowColdStartNotice(false);
      return;
    }

    const stepInterval = window.setInterval(() => {
      setActivePipelineStep((prev) =>
        prev < VERIFICATION_PIPELINE_STEPS.length - 1 ? prev + 1 : prev
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

  // Derived text metrics
  const textMetrics = useMemo(() => {
    const trimmed = rawText.trim();
    return {
      words: trimmed ? trimmed.split(/\s+/).length : 0,
      chars: rawText.length,
    };
  }, [rawText]);

  // Derived report counts & filtered results
  const reportCounts = useMemo(() => {
    if (!activeReport) return null;
    return computeStatusCounts(activeReport);
  }, [activeReport]);

  const filteredClaims = useMemo(() => {
    if (!activeReport) return [];
    const query = searchQuery.trim().toLowerCase();

    return activeReport.results.filter((item) => {
      if (verdictFilter !== 'ALL' && item.verdict !== verdictFilter) return false;
      if (onlyHumanReview && !item.needs_human_review) return false;
      if (query) {
        const haystack = [
          String(item.claim_id),
          item.claim_text,
          item.matched_citation ?? '',
          item.evidence_passage ?? '',
          item.explanation,
          item.uncertainty_note,
          ...item.mismatch_flags,
        ]
          .join(' ')
          .toLowerCase();
        return haystack.includes(query);
      }
      return true;
    });
  }, [activeReport, verdictFilter, onlyHumanReview, searchQuery]);

  // --------------------------------------------------------------------------
  // Gate: Render Aurora Violet Login Screen if not signed in
  // --------------------------------------------------------------------------
  // Interaction Handlers (Zero Hardcoded Data, Live Backend Contract)
  // --------------------------------------------------------------------------
  const handleFilePicked = (file: File | null) => {
    setVerifyError(null);
    if (!file) {
      setSelectedFile(null);
      return;
    }

    const isPdf =
      file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf');
    if (!isPdf) {
      setVerifyError('Please upload a valid PDF document (.pdf) or switch to Paste Text mode.');
      return;
    }

    setSelectedFile(file);
    if (!documentTitle.trim()) {
      setDocumentTitle(file.name.replace(/\.pdf$/i, ''));
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files?.[0] ?? null;
    handleFilePicked(droppedFile);
  };

  const handleVerifySubmit = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    setVerifyError(null);

    if (inputMode === 'pdf' && !selectedFile) {
      setVerifyError('Please select a PDF research paper to verify.');
      return;
    }

    if (inputMode === 'text' && rawText.trim().length < 20) {
      setVerifyError(
        'Please paste at least 20 characters of scientific text or claims with citations.'
      );
      return;
    }

    setIsVerifying(true);
    setActivePipelineStep(0);

    try {
      let response: Response;

      if (inputMode === 'pdf' && selectedFile) {
        const formData = new FormData();
        formData.append('file', selectedFile);
        if (documentTitle.trim()) {
          formData.append('document_title', documentTitle.trim());
        }
        response = await fetch(`${apiBaseUrl}/api/verify`, {
          method: 'POST',
          body: formData,
        });
      } else {
        response = await fetch(`${apiBaseUrl}/api/verify`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text: rawText.trim(),
            document_title: documentTitle.trim() || undefined,
          }),
        });
      }

      if (!response.ok) {
        const errBody = await response.json().catch(() => null);
        const detail =
          errBody && typeof errBody.detail === 'string'
            ? errBody.detail
            : null;

        if (response.status === 404) {
          throw new Error(
            `POST ${apiBaseUrl}/api/verify returned 404 Not Found. Your live backend is healthy, and this UI will work automatically as soon as the /api/verify route is deployed.`
          );
        }

        throw new Error(
          detail
            ? `Verification failed (HTTP ${response.status}): ${detail}`
            : `Backend responded with HTTP ${response.status}. Check /docs on your backend API.`
        );
      }

      const rawJson = await response.json();
      const normalized = normalizeReport(rawJson);

      setActiveReport(normalized);
      setVerdictFilter('ALL');
      setSearchQuery('');
      setOnlyHumanReview(false);

      const newEntry: SessionHistoryEntry = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        verifiedAt: new Date().toISOString(),
        inputMethod: inputMode,
        report: normalized,
      };

      setSessionHistory((prev) => {
        const updated = [newEntry, ...prev].slice(0, 25);
        saveSessionHistory(updated);
        return updated;
      });

      setActiveView('report');
    } catch (err) {
      setVerifyError(
        err instanceof Error
          ? err.message
          : 'Unexpected network error while contacting the verification API.'
      );
    } finally {
      setIsVerifying(false);
    }
  };

  const handleExportJson = () => {
    if (!activeReport) return;
    const blob = new Blob([JSON.stringify(activeReport, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    const safeTitle = activeReport.document_title
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-|-$/g, '');
    anchor.href = url;
    anchor.download = `${safeTitle || 'citeverify-report'}.json`;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
  };

  const handleSaveApiUrl = (e: React.FormEvent) => {
    e.preventDefault();
    const cleaned = apiUrlInput.trim().replace(/\/+$/, '');
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
    setVerdictFilter('ALL');
    setSearchQuery('');
    setOnlyHumanReview(false);
    setActiveView('report');
  };

  const handleDeleteHistoryEntry = (id: string) => {
    setSessionHistory((prev) => {
      const updated = prev.filter((item) => item.id !== id);
      saveSessionHistory(updated);
      return updated;
    });
  };

  const handleClearAllHistory = () => {
    setSessionHistory([]);
    saveSessionHistory([]);
    setConfirmClearHistory(false);
  };

  const handleSignOut = () => {
    setSession(null);
    setMobileMenuOpen(false);
  };

  // --------------------------------------------------------------------------
  // Gate: Render Aurora Violet Login Screen if not signed in
  // --------------------------------------------------------------------------
  if (!session) {
    return (
      <AuroraLoginScreen
        health={health}
        onRefreshHealth={() => runHealthCheck(apiBaseUrl)}
        onLoginSuccess={(nextSession) => {
          setSession(nextSession);
          setActiveView('verify');
        }}
      />
    );
  }

  // --------------------------------------------------------------------------
  // Post-Login Navigation Config
  // --------------------------------------------------------------------------
  const navItems: Array<{
    id: NavView;
    label: string;
    icon: React.ComponentType<{ className?: string }>;
    badge?: string | number;
  }> = [
    { id: 'verify', label: 'Verify Paper', icon: FileSearch },
    {
      id: 'report',
      label: 'Active Report',
      icon: ClipboardList,
      badge: activeReport ? activeReport.total_claims : undefined,
    },
    {
      id: 'history',
      label: 'Session History',
      icon: History,
      badge: sessionHistory.length > 0 ? sessionHistory.length : undefined,
    },
    { id: 'settings', label: 'API & Settings', icon: Settings },
  ];

  const viewHeaderMeta: Record<NavView, { title: string; subtitle: string }> = {
    verify: {
      title: 'Verify Scientific Paper',
      subtitle: 'Upload a PDF manuscript or paste claim passages to cross-check citations',
    },
    report: {
      title: 'Verification Report',
      subtitle: 'Claim-by-claim verdict inspection, source passages, and human-review flags',
    },
    history: {
      title: 'Session History',
      subtitle: 'Reports verified during this browser session (stored locally)',
    },
    settings: {
      title: 'API & Settings',
      subtitle: 'Live backend diagnostics, endpoint configuration, and schema contract',
    },
  };

  const userInitial = (session.displayName || session.email || 'R').charAt(0).toUpperCase();

  // Reusable Sidebar Content (Desktop & Mobile Drawer)
  const renderSidebarContent = () => (
    <div className="flex h-full flex-col justify-between">
      <div>
        {/* Brand Header */}
        <div className="flex items-center justify-between px-5 pt-6 pb-5">
          <div className="flex items-center gap-3">
            <BrandBadge size="md" />
            <div>
              <span className="block text-sm font-semibold tracking-tight text-white">
                CiteVerify
              </span>
              <span className="block text-[11px] text-slate-400">Research Workbench</span>
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

        {/* Quick Action CTA */}
        <div className="px-4 pb-4">
          <button
            type="button"
            onClick={() => {
              setActiveView('verify');
              setVerifyError(null);
              setMobileMenuOpen(false);
            }}
            className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-violet-600 via-indigo-600 to-fuchsia-600 px-3.5 py-2.5 text-xs font-semibold text-white shadow-[0_10px_28px_-10px_rgba(139,92,246,0.75)] transition-all hover:from-violet-500 hover:via-indigo-500 hover:to-fuchsia-500"
          >
            <Plus className="h-4 w-4" />
            New Verification
          </button>
        </div>

        {/* Primary Navigation Links */}
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
                    ? 'border border-violet-400/30 bg-violet-500/15 text-white shadow-[inset_0_1px_0_rgba(255,255,255,0.06)]'
                    : 'border border-transparent text-slate-400 hover:bg-white/[0.04] hover:text-slate-100'
                }`}
              >
                <span className="flex items-center gap-3">
                  <Icon
                    className={`h-4 w-4 transition-colors ${
                      isActive ? 'text-violet-300' : 'text-slate-500 group-hover:text-slate-300'
                    }`}
                  />
                  {item.label}
                </span>

                {item.badge !== undefined ? (
                  <span
                    className={`rounded-full px-2 py-0.5 font-mono text-[11px] ${
                      isActive
                        ? 'bg-violet-400/25 text-violet-100'
                        : 'bg-white/[0.06] text-slate-400'
                    }`}
                  >
                    {item.badge}
                  </span>
                ) : item.id === 'settings' ? (
                  <span
                    className={`h-2 w-2 rounded-full ${
                      health.status === 'online'
                        ? 'bg-emerald-400'
                        : health.status === 'checking'
                        ? 'bg-amber-400'
                        : 'bg-rose-400'
                    }`}
                  />
                ) : null}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Signed-In Profile Card */}
      <div className="border-t border-white/[0.08] p-4">
        <div className="rounded-2xl border border-white/[0.08] bg-white/[0.03] p-3">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-violet-500/30 to-cyan-500/30 border border-violet-400/30 text-sm font-semibold text-violet-100">
              {userInitial}
            </div>
            <div className="min-w-0 flex-1">
              <p className="truncate text-xs font-semibold text-white">{session.displayName}</p>
              <p className="truncate text-[11px] text-slate-400">{session.email}</p>
            </div>
          </div>
          <button
            type="button"
            onClick={handleSignOut}
            className="mt-3 flex w-full items-center justify-center gap-2 rounded-lg border border-white/10 bg-white/[0.03] px-3 py-1.5 text-xs font-medium text-slate-300 transition-colors hover:border-rose-400/35 hover:bg-rose-500/10 hover:text-rose-200"
          >
            <LogOut className="h-3.5 w-3.5" />
            Sign out
          </button>
        </div>
      </div>
    </div>
  );

  // --------------------------------------------------------------------------
  // Main Post-Login SaaS Shell
  // --------------------------------------------------------------------------
  return (
    <div className="relative flex min-h-screen w-full bg-[#070711] text-slate-100 selection:bg-violet-500/30 selection:text-violet-100">
      {/* Subtle Ambient Aurora Backdrop */}
      <div aria-hidden="true" className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute -top-40 left-1/3 h-[420px] w-[720px] rounded-full bg-[radial-gradient(ellipse_at_center,_rgba(139,92,246,0.16)_0%,_rgba(99,102,241,0.08)_45%,_transparent_72%)] blur-3xl" />
        <div className="absolute bottom-0 right-0 h-[380px] w-[480px] rounded-full bg-[radial-gradient(circle,_rgba(34,211,238,0.09)_0%,_rgba(168,85,247,0.06)_45%,_transparent_72%)] blur-3xl" />
      </div>

      {/* Desktop Left Sidebar */}
      <aside className="sticky top-0 z-20 hidden h-screen w-64 shrink-0 border-r border-white/[0.08] bg-[#0a0918]/85 backdrop-blur-2xl lg:block">
        {renderSidebarContent()}
      </aside>

      {/* Mobile Slide-Over Sidebar */}
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

      {/* Right Main Column */}
      <div className="relative z-10 flex min-w-0 flex-1 flex-col">
        {/* Sticky Top Header Bar */}
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
            {/* Live API Health Indicator Pill */}
            <button
              type="button"
              onClick={() => runHealthCheck(apiBaseUrl)}
              title="Click to re-check live backend /health"
              className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-[#0d0c1d]/90 px-3 py-1.5 text-xs font-medium text-slate-300 transition-colors hover:border-violet-400/40 hover:text-white"
            >
              <span className="relative flex h-2 w-2">
                {health.status === 'online' && (
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-60" />
                )}
                <span
                  className={`relative inline-flex h-2 w-2 rounded-full ${
                    health.status === 'online'
                      ? 'bg-emerald-400'
                      : health.status === 'checking'
                      ? 'bg-amber-400'
                      : 'bg-rose-400'
                  }`}
                />
              </span>
              <span className="hidden sm:inline">
                {health.status === 'online'
                  ? `API Healthy${health.latencyMs !== null ? ` · ${health.latencyMs}ms` : ''}`
                  : health.status === 'checking'
                  ? 'Pinging API...'
                  : 'API Offline'}
              </span>
              <RefreshCw
                className={`h-3 w-3 text-slate-400 ${
                  health.status === 'checking' ? 'animate-spin' : ''
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

        {/* Scrollable Content Viewport */}
        <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-6 sm:px-8 sm:py-8">
          {/* ================================================================ */}
          {/* VIEW 1: VERIFY PAPER WORKBENCH                                   */}
          {/* ================================================================ */}
          {activeView === 'verify' && (
            <div className="space-y-6">
              {/* Primary Verification Card */}
              <section className="relative overflow-hidden rounded-3xl border border-white/10 bg-[#0d0c1d]/80 p-5 sm:p-8 shadow-[0_24px_80px_-24px_rgba(0,0,0,0.85)] backdrop-blur-2xl">
                <div
                  aria-hidden="true"
                  className="pointer-events-none absolute inset-x-12 top-0 h-px bg-gradient-to-r from-transparent via-violet-400/60 to-transparent"
                />

                {/* Mode Switcher + Title Row */}
                <div className="mb-6 flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
                  <div>
                    <div className="inline-flex items-center gap-1.5 rounded-full border border-violet-400/25 bg-violet-500/10 px-2.5 py-0.5 text-[11px] font-medium text-violet-200">
                      <Sparkles className="h-3 w-3 text-cyan-300" />
                      Live Verification Engine
                    </div>
                    <h2 className="mt-2 text-xl font-semibold tracking-tight text-white">
                      Submit a Manuscript for Citation Audit
                    </h2>
                  </div>

                  {/* Segmented Mode Control */}
                  <div className="inline-flex rounded-xl border border-white/10 bg-black/35 p-1">
                    <button
                      type="button"
                      onClick={() => {
                        setInputMode('pdf');
                        setVerifyError(null);
                      }}
                      className={`inline-flex items-center gap-2 rounded-lg px-3.5 py-2 text-xs font-medium transition-all ${
                        inputMode === 'pdf'
                          ? 'bg-gradient-to-r from-violet-600/90 to-indigo-600/90 text-white shadow-sm'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      <Upload className="h-3.5 w-3.5" />
                      Upload PDF
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setInputMode('text');
                        setVerifyError(null);
                      }}
                      className={`inline-flex items-center gap-2 rounded-lg px-3.5 py-2 text-xs font-medium transition-all ${
                        inputMode === 'text'
                          ? 'bg-gradient-to-r from-violet-600/90 to-indigo-600/90 text-white shadow-sm'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      <FileText className="h-3.5 w-3.5" />
                      Paste Text
                    </button>
                  </div>
                </div>

                <form onSubmit={handleVerifySubmit} className="space-y-5">
                  {/* Optional Document Title */}
                  <div>
                    <label
                      htmlFor="doc-title"
                      className="mb-1.5 block text-xs font-medium text-slate-300"
                    >
                      Document Title <span className="text-slate-500">(optional)</span>
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

                  {/* PDF Upload Mode */}
                  {inputMode === 'pdf' ? (
                    <div>
                      <input
                        ref={fileInputRef}
                        type="file"
                        accept=".pdf,application/pdf"
                        onChange={(e) => handleFilePicked(e.target.files?.[0] ?? null)}
                        className="hidden"
                      />

                      {!selectedFile ? (
                        <div
                          role="button"
                          tabIndex={0}
                          onClick={() => !isVerifying && fileInputRef.current?.click()}
                          onKeyDown={(e) => {
                            if ((e.key === 'Enter' || e.key === ' ') && !isVerifying) {
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
                              ? 'border-violet-400 bg-violet-500/15'
                              : 'border-white/15 bg-white/[0.02] hover:border-violet-400/50 hover:bg-violet-500/[0.05]'
                          }`}
                        >
                          <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl border border-violet-400/30 bg-violet-500/15 text-violet-300 transition-transform group-hover:scale-105">
                            <FileUp className="h-6 w-6" />
                          </div>
                          <p className="text-sm font-medium text-white">
                            Drag and drop your PDF paper here, or{' '}
                            <span className="text-violet-300 underline decoration-violet-400/50 underline-offset-4">
                              browse files
                            </span>
                          </p>
                          <p className="mt-1.5 text-xs text-slate-400">
                            Supports standard academic PDFs with inline citations and reference lists
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
                                PDF Document &middot; {formatFileSize(selectedFile.size)}
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
                    /* Paste Text Mode */
                    <div>
                      <div className="mb-1.5 flex items-center justify-between">
                        <label
                          htmlFor="raw-paper-text"
                          className="text-xs font-medium text-slate-300"
                        >
                          Scientific Claims &amp; Reference List
                        </label>
                        <div className="flex items-center gap-3 text-[11px] text-slate-400">
                          <span>{textMetrics.words} words</span>
                          <span>{textMetrics.chars} chars</span>
                          {rawText.length > 0 && !isVerifying && (
                            <button
                              type="button"
                              onClick={() => setRawText('')}
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

                  {/* Error Banner */}
                  {verifyError && (
                    <div
                      role="alert"
                      className="flex items-start gap-3 rounded-xl border border-rose-500/35 bg-rose-500/10 p-4 text-xs leading-relaxed text-rose-200 sm:text-sm"
                    >
                      <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-rose-400" />
                      <div className="flex-1">
                        <p className="font-medium text-rose-100">Verification Request Notice</p>
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

                  {/* Live Pipeline Progress Indicator */}
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
                          Step {activePipelineStep + 1} / {VERIFICATION_PIPELINE_STEPS.length}
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
                                  ? 'border-violet-400/50 bg-violet-500/15'
                                  : isDone
                                  ? 'border-emerald-500/30 bg-emerald-500/10'
                                  : 'border-white/[0.06] bg-black/20 opacity-55'
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
                          Render free-tier servers can take 15–40 seconds to wake up on the first
                          request. Waiting for live response...
                        </p>
                      )}
                    </div>
                  )}

                  {/* Action Footer */}
                  <div className="flex flex-col-reverse items-stretch justify-between gap-3 border-t border-white/[0.08] pt-5 sm:flex-row sm:items-center">
                    <div className="flex items-center gap-2 text-xs text-slate-400">
                      <Server className="h-3.5 w-3.5 text-violet-400" />
                      <span className="truncate font-mono">{apiBaseUrl}/api/verify</span>
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

              {/* Architecture Overview Strip (Pure System Info, Zero Mock Data) */}
              <section className="grid grid-cols-1 gap-4 md:grid-cols-3">
                <div className="rounded-2xl border border-white/[0.08] bg-[#0d0c1d]/65 p-5 backdrop-blur-xl">
                  <div className="mb-3 inline-flex h-9 w-9 items-center justify-center rounded-xl border border-violet-400/25 bg-violet-500/12 text-violet-300">
                    <Layers className="h-4 w-4" />
                  </div>
                  <h3 className="text-sm font-semibold text-white">1. Claim &amp; Marker Parsing</h3>
                  <p className="mt-1.5 text-xs leading-relaxed text-slate-400">
                    Extracts individual scientific claims, inline markers, and bibliography strings
                    from uploaded PDFs or pasted manuscript text.
                  </p>
                </div>

                <div className="rounded-2xl border border-white/[0.08] bg-[#0d0c1d]/65 p-5 backdrop-blur-xl">
                  <div className="mb-3 inline-flex h-9 w-9 items-center justify-center rounded-xl border border-cyan-400/25 bg-cyan-500/12 text-cyan-300">
                    <BookOpen className="h-4 w-4" />
                  </div>
                  <h3 className="text-sm font-semibold text-white">2. Source Evidence Retrieval</h3>
                  <p className="mt-1.5 text-xs leading-relaxed text-slate-400">
                    Resolves cited papers, extracts matching evidence passages, and detects
                    retractions or metadata mismatches automatically.
                  </p>
                </div>

                <div className="rounded-2xl border border-white/[0.08] bg-[#0d0c1d]/65 p-5 backdrop-blur-xl">
                  <div className="mb-3 inline-flex h-9 w-9 items-center justify-center rounded-xl border border-emerald-400/25 bg-emerald-500/12 text-emerald-300">
                    <ShieldCheck className="h-4 w-4" />
                  </div>
                  <h3 className="text-sm font-semibold text-white">3. 5-Tier Verdict &amp; Review</h3>
                  <p className="mt-1.5 text-xs leading-relaxed text-slate-400">
                    Scores every claim across Supported, Partially Supported, Unsupported,
                    Contradictory, or Potentially Outdated with confidence &amp; human-review flags.
                  </p>
                </div>
              </section>
            </div>
          )}

          {/* ================================================================ */}
          {/* VIEW 2: ACTIVE VERIFICATION REPORT VIEWER                        */}
          {/* ================================================================ */}
          {activeView === 'report' && (
            <div className="space-y-6">
              {!activeReport ? (
                <div className="flex flex-col items-center justify-center rounded-3xl border border-white/10 bg-[#0d0c1d]/75 px-6 py-16 text-center backdrop-blur-xl">
                  <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl border border-violet-400/25 bg-violet-500/12 text-violet-300">
                    <ClipboardList className="h-6 w-6" />
                  </div>
                  <h2 className="text-lg font-semibold text-white">No Active Verification Report</h2>
                  <p className="mt-1.5 max-w-md text-sm text-slate-400">
                    Submit a PDF manuscript or paste scientific text from the Verify Paper workspace
                    to inspect claim-by-claim verdicts here.
                  </p>
                  <button
                    type="button"
                    onClick={() => setActiveView('verify')}
                    className="mt-6 inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-violet-600 via-indigo-600 to-fuchsia-600 px-5 py-2.5 text-sm font-medium text-white shadow-[0_10px_28px_-10px_rgba(139,92,246,0.8)] transition-all hover:from-violet-500 hover:via-indigo-500 hover:to-fuchsia-500"
                  >
                    Go to Verify Paper
                    <ChevronRight className="h-4 w-4" />
                  </button>
                </div>
              ) : (
                <>
                  {/* Report Top Summary Banner */}
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
                          {activeReport.total_claims} total claims evaluated &middot; Showing{' '}
                          {filteredClaims.length} matching current filters
                        </p>
                      </div>

                      <div className="flex flex-wrap items-center gap-2.5">
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
                          onClick={() => setActiveView('verify')}
                          className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-violet-600 to-indigo-600 px-3.5 py-2 text-xs font-medium text-white transition-all hover:from-violet-500 hover:to-indigo-500"
                        >
                          <Plus className="h-3.5 w-3.5" />
                          Verify Another
                        </button>
                      </div>
                    </div>

                    {/* Dynamic 5-Verdict Summary Cards */}
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
                                  setVerdictFilter((prev) => (prev === status ? 'ALL' : status))
                                }
                                className={`flex flex-col items-start rounded-2xl border p-3.5 text-left transition-all ${
                                  isSelected
                                    ? 'border-violet-400/60 bg-violet-500/15 ring-2 ring-violet-500/20'
                                    : 'border-white/[0.08] bg-white/[0.02] hover:border-white/20 hover:bg-white/[0.04]'
                                }`}
                              >
                                <div className="flex w-full items-center justify-between">
                                  <span className={`h-2 w-2 rounded-full ${meta.dotClass}`} />
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

                        {/* Proportional Verdict Distribution Bar */}
                        {activeReport.total_claims > 0 && (
                          <div className="mt-5">
                            <div className="flex h-2 w-full overflow-hidden rounded-full bg-white/[0.06]">
                              {ALL_STATUSES.map((status) => {
                                const count = reportCounts[status];
                                if (count <= 0) return null;
                                const pct = (count / activeReport.total_claims) * 100;
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
                  </section>

                  {/* Limitations Callout (Only if returned by backend) */}
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

                  {/* Search & Filter Toolbar */}
                  <section className="flex flex-col gap-3 rounded-2xl border border-white/[0.08] bg-[#0d0c1d]/75 p-4 backdrop-blur-xl md:flex-row md:items-center md:justify-between">
                    <div className="relative flex-1">
                      <Search className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
                      <input
                        type="text"
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        placeholder="Search claims, citations, passages, or mismatch flags..."
                        className="w-full rounded-xl border border-white/10 bg-white/[0.03] py-2 pl-10 pr-4 text-sm text-white placeholder:text-slate-500 focus:border-violet-400/65 focus:outline-none focus:ring-4 focus:ring-violet-500/15"
                      />
                    </div>

                    <div className="flex flex-wrap items-center gap-2">
                      <button
                        type="button"
                        onClick={() => setOnlyHumanReview((prev) => !prev)}
                        className={`inline-flex items-center gap-1.5 rounded-xl border px-3 py-2 text-xs font-medium transition-all ${
                          onlyHumanReview
                            ? 'border-amber-400/50 bg-amber-500/15 text-amber-200'
                            : 'border-white/10 bg-white/[0.03] text-slate-300 hover:border-white/20'
                        }`}
                      >
                        <Filter className="h-3.5 w-3.5" />
                        Needs Human Review
                      </button>

                      {(verdictFilter !== 'ALL' || searchQuery.trim() || onlyHumanReview) && (
                        <button
                          type="button"
                          onClick={() => {
                            setVerdictFilter('ALL');
                            setSearchQuery('');
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

                  {/* Claim Cards Feed */}
                  {filteredClaims.length === 0 ? (
                    <div className="rounded-2xl border border-white/[0.08] bg-[#0d0c1d]/70 px-6 py-12 text-center">
                      <p className="text-sm font-medium text-slate-300">
                        No claims match your current search or filter criteria.
                      </p>
                      <button
                        type="button"
                        onClick={() => {
                          setVerdictFilter('ALL');
                          setSearchQuery('');
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
                        <ClaimVerificationCard key={claim.claim_id} item={claim} />
                      ))}
                    </div>
                  )}
                </>
              )}
            </div>
          )}

          {/* ================================================================ */}
          {/* VIEW 3: SESSION HISTORY                                          */}
          {/* ================================================================ */}
          {activeView === 'history' && (
            <div className="space-y-5">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <p className="text-xs text-slate-400">
                  {sessionHistory.length} verified{' '}
                  {sessionHistory.length === 1 ? 'report' : 'reports'} saved in this browser
                </p>

                {sessionHistory.length > 0 && (
                  <div className="flex items-center gap-2">
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
                  </div>
                )}
              </div>

              {sessionHistory.length === 0 ? (
                <div className="flex flex-col items-center justify-center rounded-3xl border border-white/10 bg-[#0d0c1d]/75 px-6 py-16 text-center backdrop-blur-xl">
                  <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl border border-violet-400/25 bg-violet-500/12 text-violet-300">
                    <History className="h-6 w-6" />
                  </div>
                  <h2 className="text-lg font-semibold text-white">No Verification History Yet</h2>
                  <p className="mt-1.5 max-w-md text-sm text-slate-400">
                    Every real verification report returned by your backend will automatically be
                    archived here so you can re-open or export it anytime.
                  </p>
                  <button
                    type="button"
                    onClick={() => setActiveView('verify')}
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
                            <span className="text-xs text-slate-400">
                              {formatTimestamp(entry.verifiedAt)}
                            </span>
                            <span className="text-xs text-slate-500">&middot;</span>
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
                              ) : null
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
                            onClick={() => handleDeleteHistoryEntry(entry.id)}
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
          {activeView === 'settings' && (
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              {/* Endpoint Configuration Card */}
              <section className="rounded-3xl border border-white/10 bg-[#0d0c1d]/80 p-6 backdrop-blur-2xl">
                <div className="mb-5 flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-violet-400/30 bg-violet-500/12 text-violet-300">
                    <Link2 className="h-5 w-5" />
                  </div>
                  <div>
                    <h2 className="text-base font-semibold text-white">Backend API Endpoint</h2>
                    <p className="text-xs text-slate-400">
                      Defaults automatically to <code className="text-violet-300">VITE_API_URL</code>
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

              {/* Live Health Diagnostics Card */}
              <section className="rounded-3xl border border-white/10 bg-[#0d0c1d]/80 p-6 backdrop-blur-2xl">
                <div className="mb-5 flex items-center justify-between gap-3">
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-cyan-400/30 bg-cyan-500/12 text-cyan-300">
                      <Activity className="h-5 w-5" />
                    </div>
                    <div>
                      <h2 className="text-base font-semibold text-white">Live /health Diagnostics</h2>
                      <p className="text-xs text-slate-400">Real-time ping against FastAPI backend</p>
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => runHealthCheck(apiBaseUrl)}
                    className="inline-flex items-center gap-1.5 rounded-xl border border-white/10 bg-white/[0.04] px-3 py-1.5 text-xs font-medium text-slate-200 hover:border-violet-400/40"
                  >
                    <RefreshCw
                      className={`h-3.5 w-3.5 ${health.status === 'checking' ? 'animate-spin' : ''}`}
                    />
                    Re-test
                  </button>
                </div>

                <dl className="grid grid-cols-3 gap-3 text-xs">
                  <div className="rounded-xl border border-white/[0.07] bg-black/25 p-3">
                    <dt className="text-slate-400">Status</dt>
                    <dd className="mt-1 font-semibold capitalize text-white">{health.status}</dd>
                  </div>
                  <div className="rounded-xl border border-white/[0.07] bg-black/25 p-3">
                    <dt className="text-slate-400">HTTP Code</dt>
                    <dd className="mt-1 font-mono font-semibold text-white">
                      {health.httpCode ?? '—'}
                    </dd>
                  </div>
                  <div className="rounded-xl border border-white/[0.07] bg-black/25 p-3">
                    <dt className="text-slate-400">Latency</dt>
                    <dd className="mt-1 font-mono font-semibold text-white">
                      {health.latencyMs !== null ? `${health.latencyMs} ms` : '—'}
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
                      : '// No JSON payload received yet'}
                  </pre>
                </div>
              </section>

              {/* Backend Schema Contract Card */}
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
                      Frontend types mirror VerificationReport and ClaimVerificationResult strictly
                    </p>
                  </div>
                </div>

                <pre className="overflow-x-auto rounded-2xl border border-white/[0.07] bg-black/40 p-4 font-mono text-xs leading-relaxed text-slate-300">
{`POST ${apiBaseUrl}/api/verify  ->  VerificationReport
{
  "document_title": string,
  "total_claims": number,
  "summary_counts": Record<ClaimStatus, number>,
  "results": ClaimVerificationResult[],
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