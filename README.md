# CiteVerify — Automated Scientific Claim & Citation Verification Platform

**CiteVerify** audits scientific manuscripts end-to-end by extracting factual claims and inline citations from academic PDFs or raw text, retrieving canonical source publications across five open scholarly APIs, running four research-grade verification engines (hybrid NLI ranking, atomic claim decomposition, time-aware consensus tracking, and uncertainty-aware calibration), and delivering interactive web diagnostics alongside downloadable **PDF / Markdown / JSON** audit reports and **Supabase Cloud History**.

### Live Production Deployments
* **Live Web Workbench (Vercel):** [https://citeverify-liard.vercel.app](https://citeverify-liard.vercel.app)
* **Backend API & Swagger Docs (Render):** [https://citeverify-api.onrender.com/docs](https://citeverify-api.onrender.com/docs)
* **API Health Endpoint:** [https://citeverify-api.onrender.com/health](https://citeverify-api.onrender.com/health)

> **Cold-Start Note:** The backend runs on Render's free tier (512 MB RAM) and may take ~30–50 seconds to wake up on the very first request after an idle period. Check the top-right **API Healthy** badge in the workbench before uploading.

---

## Core Features

* **Dual Input Modes & 2-Column Academic PDF Parsing:** Upload full academic PDFs (including multi-column IEEE/arXiv layouts and broken copy-paste artifacts) or paste manuscript excerpts with inline citations and reference lists.
* **5-API Scholarly Source Retrieval & Legal Open-Access Discovery:** Resolves DOIs, arXiv IDs, titles, authors, publication years, abstracts, and retraction notices across **OpenAlex**, **CrossRef**, **arXiv**, **Europe PMC**, and **Unpaywall**—automatically upgrading paywalled journal links to legal open-access repository copies whenever available.
* **5-Tier Calibrated Verdicts:** Evaluates every claim strictly against retrieved source evidence using structured JSON output from Gemini (`gemini-3.5-flash-lite` with automatic retry and `gemini-3.6-flash` / `gemini-3.8-flash` failover):
  * `SUPPORTED` — Source evidence directly confirms the claim and metrics.
  * `PARTIALLY_SUPPORTED` — Source aligns broadly but omits scope limits, conflates metrics, or exaggerates results.
  * `CONTRADICTORY` — Source numbers, methodology, or conclusions conflict directly with the claim.
  * `UNSUPPORTED` — Source is missing, strictly paywalled without accessible text, or unrelated.
  * `POTENTIALLY_OUTDATED` — Source has been formally retracted or superseded.
* **Four Research-Grade Verification Gaps (Live in `/api/verify`, UI & Exports):** Every audited claim is enriched with hybrid NLI passage ranking and similarity-trap detection (**Gap 1**), atomic compound-claim decomposition (**Gap 2**), live OpenAlex/CrossRef temporal consensus timelines (**Gap 3**), and calibrated probability + 4-state abstention gating (**Gap 4**)—engineered in pure Python for 512 MB Render RAM safety with zero heavy local PyTorch weights.
* **Supabase Authentication & Cloud History:** Supports **Email/Password** and **Google OAuth** sign-in via Supabase Auth, persisting every verified report automatically to a Row-Level-Security (RLS) isolated PostgreSQL table (`public.verification_reports`) with seamless local-storage fallback when unconfigured.
* **One-Click Audit Report Exports (Including Gaps 1–4):** Download multi-page **PDF**, **Markdown (`.md`)**, and raw **JSON** verification reports directly from the workbench via `POST /api/export`, complete with Executive Verdict Summaries, Document-Level Gaps 1–4 Roll-Ups, and per-claim Deep Audit Diagnostics.
* **Interactive Research Workbench:** Filter claims by verdict or human-review status, search across passages/temporal statuses/flags, expand interactive 4-tile **Deep Audit Diagnostics** panels on each claim card, inspect live `/health` latency diagnostics, and reopen past audits from **Cloud History**.

---

## Four Research-Grade Verification Gaps

CiteVerify addresses the four core failure modes of automated scientific fact-checking—all wired live into `POST /api/verify`, rendered interactively in `App.tsx`, and embedded in PDF/Markdown exports:

| Gap | Service Module | Core Techniques & Guardrails | Live Output & Policy |
| :--- | :--- | :--- | :--- |
| **Gap 1: Hybrid NLI Evidence Ranking** | `backend/services/evidence_ranker.py` | 5-stage pipeline: BM25Okapi (`k1=1.5, b=0.75`), asymmetric semantic similarity (`gemini-embedding-001` or offline TF-IDF cosine), Reciprocal Rank Fusion (`k=60`), non-polar pairwise cross-encoder reranking, and independent NLI stance classification (`ENTAILS` / `CONTRADICTS` / `NEUTRAL`). | Decouples topical overlap from logical support; flags high-relevance contradictions with `SIMILARITY_TRAP_CONTRADICTION`. |
| **Gap 2: Atomic Claim Decomposition** | `backend/services/claim_decomposer.py` | Compound clause detection (semicolons, contrast connectors, multi-verb clauses), atomic noun-phrase protection, subject inheritance, gerund-to-finite conjugation, qualifier/citation propagation, optional Gemini JSON split, and deterministic verdict combiner. | Splits compound claims into standalone propositions and exposes mixed true/false statements via `PARTIAL_CONTRADICTION`. |
| **Gap 3: Time-Aware Consensus Tracking** | `backend/services/temporal_tracker.py` | Live OpenAlex citing-works lookup (`cited_by_api_url` with merged query params) + CrossRef editorial updates (`updated-by`, `relation.is-preprint-of`), study-type detection (`meta_analysis`, `systematic_review`, `replication`), and chronological timeline builder. | Outputs a 6-state `TemporalAssessment` (`CONSENSUS_ALIGNED`, `SUPERSEDED_OR_CHALLENGED`, `RETRACTED`, `CORRECTED_OR_UPDATED`, `STALE_NEEDS_UPDATE`, `INSUFFICIENT_TEMPORAL_DATA`) while strictly preserving historical verdict (`replaces_historical_verdict=False`). |
| **Gap 4: Uncertainty Calibration & Abstention** | `backend/services/uncertainty_calibrator.py` | Decomposed evidence coverage (concept + numeric overlap × scope weight), retrieval confidence, passage agreement/conflict penalty (`-0.35`), validation-set temperature scaling (`fit_temperature_scaler` minimizing NLL/ECE/Brier), and composite reliability scoring. | Enforces 4-state policy: `PRESENT_VERDICT`, `FLAG_FOR_REVIEW`, `ABSTAIN` (*"I could not find enough evidence to verify this claim."*), or `MARK_UNVERIFIABLE`. |

---

## System Architecture

```text
┌─────────────────────────────┐      Supabase Auth (Email + Google OAuth)      ┌──────────────────────────────┐
│  React + TypeScript + Vite  │ ◀────────────────────────────────────────────▶ │  Supabase Auth & PostgreSQL  │
│  Tailwind CSS v4 (Vercel)   │      RLS Cloud History (verification_reports)  │  Cloud Persistence & Sync    │
└──────────────┬──────────────┘                                                └──────────────────────────────┘
               │ HTTPS (Multipart PDF / JSON) <-> VerificationReport + Gaps 1-4 / PDF / MD Blob
               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   FastAPI Backend — Render (backend/main.py)                                │
└──────────────────────────────────────────────────────┬──────────────────────────────────────────────────────┘
                                                       ▼
┌────────────────────────────────────┐   ┌──────────────────────────────────────────┐   ┌─────────────────────────────────────┐
│ 1. Parser & Citation Matcher       │──▶│ 2. 5-API Scholarly Retriever             │──▶│ 3. Gap 1: Hybrid Evidence Ranker    │
│ PyMuPDF 2-column reading order &   │   │ OpenAlex · CrossRef · arXiv              │   │ BM25 + Semantic RRF + Cross-Encoder │
│ inline marker -> bibliography map  │   │ Europe PMC · Unpaywall OA + mismatches   │   │ + NLI & Similarity-Trap Guard       │
└────────────────────────────────────┘   └──────────────────────────────────────────┘   └──────────────────┬──────────────────┘
                                                                                                           ▼
┌────────────────────────────────────┐   ┌──────────────────────────────────────────┐   ┌─────────────────────────────────────┐
│ 6. Gap 4: Uncertainty Calibrator   │◀──│ 5. Gap 3: Temporal Consensus Tracker     │◀──│ 4. Gemini Verifier + Gap 2 Splitter │
│ Coverage, retrieval, agreement,    │   │ OpenAlex citing works & CrossRef notices │   │ Structured JSON + retry failover +  │
│ temperature scaling & abstention   │   │ Separate timeline (preserves history)    │   │ atomic proposition decomposition    │
└─────────────────┬──────────────────┘   └──────────────────────────────────────────┘   └─────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 7. Report Exporter (backend/services/report_exporter.py) — Multi-Page PDF & Markdown with Gaps 1-4 Roll-Up  │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Live Production Validation

### 1. Clinical Benchmark (`demo_paper.pdf`)
Running the built-in 5-claim clinical manuscript (`demo_paper.pdf`) against production CiteVerify validates core verdicts, retraction tracking, and Gaps 1–4 calibration:

| Claim | Cited Source | Verdict | Confidence | Key Audit Finding & Deep Diagnostics |
| :--- | :--- | :--- | :--- | :--- |
| **#1** | Gulshan et al. (2016) | **Supported** | Medium | Verbatim match for 97.5% sensitivity & 93.4% specificity on EyePACS-1; Gap 3 tracks `9,150` citations (`Aging · Check Updates`); flagged for human review (`ABSTRACT_ONLY`). |
| **#2** | McKinney et al. (2020) | **Contradictory** | High | Claimed 30-country prospective RCTs replacing radiologists; abstract shows retrospective UK/US datasets; Gap 4 flags low lexical coverage (`25%`). |
| **#3** | Horby et al. (2020) | **Contradictory** | High | Claimed dexamethasone raised 28-day mortality to 62.4%; source reports 22.9% vs. 25.7% usual care (`YEAR_MISMATCH` + `numerical_distortion`; `12,441` citations). |
| **#4** | Fabricated 2024 clinical transformer | **Unsupported** | Low | Flagged `TITLE_MISMATCH` & `METADATA_ONLY`; Gap 4 abstains (`0%` coverage) without fabricating support. |
| **#5** | Mehra et al. (2020) | **Potentially Outdated** | High | Detected `RETRACTED_SOURCE` directly from retraction metadata; Gap 3 marks `Retracted` (`1,304` citations). |

### 2. Real-World Multi-Page Paywall Stress Test
Audited against a 4-page, 18-claim scholarly report (*Journal Paywalls and the Citation Verification Gap*), production CiteVerify:
* Upgraded paywalled citations via **Unpaywall / Europe PMC** to legal open-access PDFs (**Jinek 2012** `PMC6286148`, **Greenberg 2009** `BMJ`, **Larivière 2015** `PLOS ONE`, **Day 2020** `BMC`) while escalating strictly paywalled **Evans (1990)** (`JAMA`) for human review.
* Populated real 2025–2026 citing studies and systematic reviews in **Gap 3** timelines (e.g., 2025 quotation-inaccuracy meta-analysis citing Evans 1990; 2026 citation-accuracy research citing Jergas 2015).
* Marked uncited narrative/table sentences as `Unverifiable` (`MISSING_REFERENCE`, `0%` coverage) via **Gap 4**, and exported a 12-page PDF audit report containing full Gaps 1–4 diagnostics.

---

## API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Root status check confirming the API is live. |
| `GET` | `/health` | Lightweight health check for Render monitoring and UI latency diagnostics. |
| `POST` | `/api/parse` | Extracts verifiable `ExtractedClaim[]` from an uploaded PDF (`file`) or form text (`text`). |
| `POST` | `/api/verify` | Full end-to-end pipeline (`multipart/form-data` PDF upload or `application/json` text) returning `VerificationReport` with Gaps 1–4 diagnostics. |
| `POST` | `/api/verify-claim` | Verifies a single `(claim, source)` pair with Gaps 1–4 enrichment. |
| `POST` | `/api/verify-batch` | Verifies a batch of `(claim, source)` pairs for a document. |
| `POST` | `/api/export?format=pdf\|markdown` | Renders a `VerificationReport` (including Gaps 1–4 roll-up & claim diagnostics) into a downloadable PDF or Markdown file (`Content-Disposition` attachment). |

---

## Project Structure

```text
citeverify/
├── backend/
│   ├── main.py                           # FastAPI application, CORS & API routes
│   ├── schemas.py                        # Shared Pydantic contracts (core + optional Gaps 1-4 schemas)
│   ├── requirements.txt                  # Pure-Python production dependencies (512 MB RAM safe)
│   ├── services/
│   │   ├── parser.py                     # PyMuPDF 2-column extraction & claim parsing
│   │   ├── citation_matcher.py           # Inline marker to bibliography matching
│   │   ├── retriever.py                  # OpenAlex, CrossRef, arXiv, Europe PMC & Unpaywall retrieval
│   │   ├── evidence_ranker.py            # Gap 1: BM25 + Semantic RRF + Cross-Encoder + NLI ranking
│   │   ├── claim_decomposer.py           # Gap 2: Atomic claim decomposition & verdict aggregation
│   │   ├── temporal_tracker.py           # Gap 3: Time-aware citation & OpenAlex/CrossRef consensus tracking
│   │   ├── uncertainty_calibrator.py     # Gap 4: Coverage, temperature scaling & 4-state abstention policy
│   │   ├── verifier.py                   # End-to-end Gemini verifier + live Gaps 1-4 orchestration
│   │   └── report_exporter.py            # Multi-page PDF & Markdown exporter (with Gaps 1-4 diagnostics)
│   └── tests/
│       ├── test_parser.py                # 6 unit tests: 2-column PDF & citation matching
│       ├── test_report_exporter.py       # 3 unit tests: Markdown & paginated PDF exports
│       ├── test_evidence_ranker.py       # 6 unit tests: Gap 1 hybrid ranking & similarity traps
│       ├── test_claim_decomposer.py      # 6 unit tests: Gap 2 compound splitting & verdict combining
│       ├── test_temporal_tracker.py      # 7 unit tests: Gap 3 consensus timelines & editorial notices
│       ├── test_uncertainty_calibrator.py# 7 unit tests: Gap 4 temperature scaling & abstention policy
│       └── eval_benchmark.py             # Decoupled offline & live evaluation benchmark CLI
├── frontend/
│   ├── src/
│   │   ├── App.tsx                       # SaaS workbench, Supabase Auth, Cloud History & Gaps 1-4 UI
│   │   ├── lib/
│   │   │   └── supabase.ts               # Supabase client bootstrap & graceful fallback config
│   │   └── index.css                     # Tailwind CSS v4 entry
│   ├── vite.config.ts                    # Vite + Tailwind v4 build config
│   └── package.json
└── README.md
```

---

## Automated Test Suite (35 Unit Tests)

All 35 unit tests run offline in under 0.2 seconds using Python's built-in `unittest` runner (zero external API keys required):

| Test Suite File | Tests | Verified Capabilities |
| :--- | :---: | :--- |
| `backend/tests/test_parser.py` | 6 | PDF cleaning, 2-column IEEE/arXiv sorting, copy-paste artifacts, numeric & author-year matching, `demo_paper.pdf` regression. |
| `backend/tests/test_report_exporter.py` | 3 | URL-safe export filenames, Markdown audit summaries, multi-page PDF pagination (`Page X of Y`). |
| `backend/tests/test_evidence_ranker.py` | 6 | Gap 1 BM25/RRF fusion, similarity-trap contradiction detection, negation flips, neutral overlap guards. |
| `backend/tests/test_claim_decomposer.py` | 6 | Gap 2 compound splitting, qualifier/number retention, atomic guard, mixed-verdict aggregation. |
| `backend/tests/test_temporal_tracker.py` | 7 | Gap 3 `2012 -> 2018 -> 2023` challenged consensus, superseded preprints, retractions/corrections, OpenAlex/CrossRef parsing. |
| `backend/tests/test_uncertainty_calibrator.py` | 7 | Gap 4 4-way abstention/review policy, validation-set temperature scaling (ECE/Brier/NLL), verifier adapter. |

---

## Local Development Setup

### Prerequisites
* **Python 3.11+** (required for `datetime.UTC` and `enum.StrEnum`)
* **Node.js 18+**
* A Google AI Studio `GEMINI_API_KEY`
* *(Optional)* A **Supabase** project URL & anon key for Cloud Auth and PostgreSQL Cloud History

### 1. Clone the Repository
```bash
git clone [https://github.com/Chinmai1430/citeverify.git](https://github.com/Chinmai1430/citeverify.git)
cd citeverify
```

### 2. Run the Backend (FastAPI)
```bash
cd backend
python -m venv .venv

# Windows (Git Bash)
source .venv/Scripts/activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

Create `backend/.env`:
```env
GEMINI_API_KEY=your_google_ai_studio_api_key_here
# Optional overrides:
# GEMINI_MODEL=gemini-3.5-flash-lite
# UNPAYWALL_EMAIL=your_email@example.com
# CONTACT_EMAIL=your_email@example.com
# DISABLE_LIVE_TEMPORAL_FETCH=0
```

Start the API server:
```bash
uvicorn main:app --reload --port 8000
```
API docs will be live at `http://127.0.0.1:8000/docs`.

### 3. Run the Frontend (React + Vite)
In a second terminal from the repository root:
```bash
cd frontend
npm install
```

Create `frontend/.env.local`:
```env
VITE_API_URL=[http://127.0.0.1:8000](http://127.0.0.1:8000)
# Optional — enables Supabase Email/Google OAuth & Postgres Cloud History:
# VITE_SUPABASE_URL=[https://your-project-ref.supabase.co](https://your-project-ref.supabase.co)
# VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
```

Start the development server:
```bash
npm run dev
```

### 4. Run Unit Tests & Evaluation Benchmark
From the repository root with `backend/.venv` activated:
```bash
# Run all 35 unit tests offline
python -m unittest discover -s backend/tests -p "test_*.py" -v

# Run deterministic offline benchmark smoke check (Gap 1 & Gap 2 scorecards)
python -m backend.tests.eval_benchmark --offline

# Run live end-to-end benchmark (with optional live Gemini Gap AI & report exports)
python -m backend.tests.eval_benchmark --live-gap-ai --export-dir ./sample_exports
```

---

## Tech Stack

* **Frontend:** React, TypeScript, Vite, Tailwind CSS v4, Lucide React, `@supabase/supabase-js`
* **Backend:** Python 3.11+, FastAPI, Pydantic, PyMuPDF (`pymupdf`), HTTPX (`httpx`), Google GenAI SDK (`google-genai`), `python-dotenv`
* **Auth & Database:** Supabase Auth (Email/Password + Google OAuth), Supabase PostgreSQL with Row-Level Security (`verification_reports`)
* **Scholarly Data Sources:** OpenAlex API, CrossRef API, arXiv API, Europe PMC API, Unpaywall API
* **Deployment & CI:** Vercel (Frontend), Render (Backend), GitHub auto-deploy on `main`

---

## Team Contributions

| Member | Owned Modules & Responsibilities |
| :--- | :--- |
| **Chinmai B** | FastAPI backend orchestration (`main.py`), shared Pydantic schemas (`schemas.py`), Gemini structured verification engine & retry failover (`verifier.py`), live end-to-end integration of Gaps 1–4 across backend/UI/exports, Supabase Auth & Google OAuth + Cloud History wiring, and Render/Vercel production deployments. |
| **Chiranth** | 2-column IEEE/arXiv PDF reading-order & edge-case parsing (`parser.py`, `citation_matcher.py`), standalone Markdown & PDF report exporter (`report_exporter.py`), all four research-grade Gap services (`evidence_ranker.py`, `claim_decomposer.py`, `temporal_tracker.py`, `uncertainty_calibrator.py`), 35 automated unit tests, and decoupled evaluation benchmark CLI (`eval_benchmark.py`). |
| **D P Rithvik Kumar** | Multi-API scholarly source retrieval across OpenAlex, CrossRef, arXiv, Europe PMC, and Unpaywall (`retriever.py`) with legal open-access link discovery, retraction detection, and DOI/title/author/year mismatch flagging. |
| **Dhanush** | React + TypeScript + Tailwind CSS v4 verification workbench UI (`App.tsx`), Aurora Violet login screen, claim verification cards, dynamic search & filters, history view, and live `/health` diagnostics. |

---

## Disclaimer
CiteVerify is an automated decision-support tool designed to accelerate citation auditing—not a replacement for formal peer review or final academic judgment. Claims verified from abstract-only, paywalled, or metadata-only sources are explicitly flagged for human review or abstained under Gap 4's uncertainty policy.