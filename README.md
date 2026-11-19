# CiteVerify — Automated Scientific Claim & Citation Verification Platform

**CiteVerify** audits scientific manuscripts end-to-end by extracting factual claims and inline citations from academic PDFs or raw text, retrieving the original referenced publications across open scholarly databases, and verifying whether the cited evidence actually supports the claim—complete with calibrated confidence scores, mismatch flags, retraction detection, and downloadable **PDF / Markdown / JSON** audit reports.

### Live Production Deployments
* **Live Web Workbench (Vercel):** [https://citeverify-liard.vercel.app](https://citeverify-liard.vercel.app)
* **Backend API & Swagger Docs (Render):** [https://citeverify-api.onrender.com/docs](https://citeverify-api.onrender.com/docs)
* **API Health Endpoint:** [https://citeverify-api.onrender.com/health](https://citeverify-api.onrender.com/health)

> **Cold-Start Note:** The backend runs on Render's free tier and may take ~30–50 seconds to wake up on the very first request after an idle period. Check the top-right **API Healthy** badge in the workbench before uploading.

---

## Core Features

* **Dual Input Modes & 2-Column Academic PDF Parsing:** Upload full academic PDFs (including multi-column IEEE/arXiv layouts and broken copy-paste formatting) or paste manuscript excerpts with inline citations and reference lists.
* **Multi-API Scholarly Source Retrieval:** Resolves DOIs, titles, authors, publication years, abstracts, and retraction metadata across **OpenAlex**, **CrossRef**, and **arXiv**.
* **5-Tier Calibrated Verdicts:** Evaluates every claim strictly against retrieved source evidence using structured JSON output from Gemini (`gemini-3.5-flash-lite` with automatic retry and `gemini-3.6-flash` / `gemini-3.8-flash` failover):
  * `SUPPORTED` — Source evidence directly confirms the claim and metrics.
  * `PARTIALLY_SUPPORTED` — Source aligns broadly but omits scope limits or exaggerates results.
  * `CONTRADICTORY` — Source numbers, methodology, or conclusions conflict directly with the claim.
  * `UNSUPPORTED` — Source is missing, paywalled without accessible text, or unrelated.
  * `POTENTIALLY_OUTDATED` — Source has been formally retracted or superseded.
* **Honest Uncertainty & Human-Review Guardrails:** Automatically caps confidence and flags `Needs Human Review` when only abstracts are accessible (paywalled articles), when metadata mismatches occur (`TITLE_MISMATCH`, `YEAR_MISMATCH`), or when a cited paper is formally retracted (`RETRACTED_SOURCE`).
* **One-Click Audit Report Exports:** Download structured **PDF**, **Markdown (`.md`)**, and raw **JSON** verification reports directly from the workbench via `POST /api/export`.
* **Interactive Research Workbench:** Filter claims by verdict or human-review status, search across passages and mismatch flags, inspect live `/health` latency diagnostics, and revisit past audits in local **Session History**.

---

## System Architecture

~~~text
┌─────────────────────────────┐         HTTPS / JSON / Multipart        ┌──────────────────────────────────┐
│  React + TypeScript + Vite  │ ──────────────────────────────────────▶ │     FastAPI Backend (Render)     │
│  Tailwind CSS v4 (Vercel)   │ ◀────────────────────────────────────── │          backend/main.py         │
└─────────────────────────────┘   VerificationReport / PDF / MD Blob    └────────────────┬─────────────────┘
                                                                                         │
                       ┌─────────────────────────────────────────────────────────────────┘
                       ▼
┌────────────────────────────────────────┐     ┌────────────────────────────────────────┐
│  1. Parser & Citation Matcher          │ ──▶ │  2. Multi-API Scholarly Retriever      │
│  PyMuPDF block-level 2-column reading  │     │  OpenAlex · CrossRef · arXiv           │
│  & inline citation -> reference map    │     │  Retraction, year & title mismatching  │
└────────────────────────────────────────┘     └────────────────────┬───────────────────┘
                                                                    ▼
┌────────────────────────────────────────┐     ┌────────────────────────────────────────┐
│  4. Audit Report Exporter              │ ◀── │  3. Structured Gemini Verifier         │
│  PyMuPDF PDF generator & Markdown      │     │  Deterministic guardrails + Gemini     │
│  builder (POST /api/export)            │     │  structured JSON + retry failover      │
└────────────────────────────────────────┘     └────────────────────────────────────────┘
~~~

---

## Live Production Benchmark (`demo_paper.pdf`)

Running the built-in 5-claim demonstration manuscript (`demo_paper.pdf`) against production CiteVerify validates all core audit behaviors:

| Claim | Cited Source | Verdict | Confidence | Key Audit Finding |
| :--- | :--- | :--- | :--- | :--- |
| **#1** | Gulshan et al. (2016) | **Supported** | Medium | Verbatim match for 97.5% sensitivity and 93.4% specificity on EyePACS-1; flagged for human review due to `ABSTRACT_ONLY` scope. |
| **#2** | McKinney et al. (2020) | **Contradictory** | High | Claimed 30-country prospective RCTs replacing radiologists; retrieved abstract shows a retrospective UK/US dataset evaluation paving the way for clinical trials. |
| **#3** | Horby et al. (2020) | **Contradictory** | High | Claimed dexamethasone increased 28-day mortality to 62.4%; source reports a reduction to 22.9% vs. 25.7% usual care (plus `YEAR_MISMATCH`). |
| **#4** | Fabricated 2024 clinical transformer | **Unsupported** | Low | Flagged `TITLE_MISMATCH` and escalated to manual review without fabricating support. |
| **#5** | Mehra et al. (2020) | **Potentially Outdated** | High | Detected `RETRACTED_SOURCE` directly from publisher retraction metadata. |

---

## API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Root status check confirming the API is live. |
| `GET` | `/health` | Lightweight health check for deployment monitoring and UI diagnostics. |
| `POST` | `/api/parse` | Extracts verifiable `ExtractedClaim[]` from an uploaded PDF (`file`) or form text (`text`). |
| `POST` | `/api/verify` | Full end-to-end pipeline (`multipart/form-data` PDF upload or `application/json` text) returning `VerificationReport`. |
| `POST` | `/api/verify-claim` | Verifies a single `(claim, source)` pair. |
| `POST` | `/api/verify-batch` | Verifies a batch of `(claim, source)` pairs for a document. |
| `POST` | `/api/export?format=pdf\|markdown` | Renders a `VerificationReport` into a downloadable PDF or Markdown audit file (`Content-Disposition` attachment). |

---

## Project Structure

~~~text
citeverify/
├── backend/
│   ├── main.py                      # FastAPI application, CORS & API routes
│   ├── schemas.py                   # Shared Pydantic data contracts
│   ├── requirements.txt             # Python dependencies
│   ├── services/
│   │   ├── parser.py                # PyMuPDF 2-column extraction & claim parsing
│   │   ├── citation_matcher.py      # Inline marker to bibliography matching
│   │   ├── retriever.py             # OpenAlex, CrossRef & arXiv retrieval + mismatch checks
│   │   ├── verifier.py              # Guardrails + Gemini structured verification + retry logic
│   │   └── report_exporter.py       # Standalone Markdown & PDF audit report generator
│   └── tests/
│       ├── test_parser.py           # 2-column & edge-case parser unit tests
│       ├── test_report_exporter.py  # Markdown & PDF exporter unit tests
│       └── eval_benchmark.py        # Live & offline judge evaluation benchmark CLI
├── frontend/
│   ├── src/
│   │   ├── App.tsx                  # Full SaaS research workbench & report explorer
│   │   └── index.css                # Tailwind CSS v4 entry
│   ├── vite.config.ts               # Vite + Tailwind v4 build config
│   └── package.json
└── README.md
~~~

---

## Local Development Setup

### Prerequisites
* Python 3.10+
* Node.js 18+
* A Google AI Studio `GEMINI_API_KEY`

### 1. Clone the Repository
~~~bash
git clone https://github.com/Chinmai1430/citeverify.git
cd citeverify
~~~

### 2. Run the Backend (FastAPI)
~~~bash
cd backend
python -m venv .venv

# Windows (Git Bash)
source .venv/Scripts/activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
~~~

Create `backend/.env`:
~~~env
GEMINI_API_KEY=your_google_ai_studio_api_key_here
~~~

Start the API server:
~~~bash
uvicorn main:app --reload --port 8000
~~~
API docs will be live at `http://127.0.0.1:8000/docs`.

### 3. Run the Frontend (React + Vite)
In a second terminal from the repository root:
~~~bash
cd frontend
npm install
~~~

Create `frontend/.env.local`:
~~~env
VITE_API_URL=http://127.0.0.1:8000
~~~

Start the development server:
~~~bash
npm run dev
~~~

### 4. Run Tests & Evaluation Benchmark
From the repository root with your backend virtual environment activated:
~~~bash
# Run unit tests
pytest backend/tests

# Run offline benchmark smoke check
python -m backend.tests.eval_benchmark --offline

# Run live end-to-end benchmark and export sample reports
python -m backend.tests.eval_benchmark --export-dir ./sample_exports
~~~

---

## Tech Stack

* **Frontend:** React, TypeScript, Vite, Tailwind CSS v4, Lucide React
* **Backend:** Python 3, FastAPI, Pydantic, PyMuPDF (`pymupdf`), Google GenAI SDK (`google-genai`), `python-dotenv`
* **Scholarly Data Sources:** OpenAlex API, CrossRef API, arXiv API
* **Deployment & CI:** Vercel (Frontend), Render (Backend), GitHub auto-deploy on `main`

---

## Team Contributions

| Member | Owned Modules & Responsibilities |
| :--- | :--- |
| **Chinmai B** | FastAPI backend orchestration (`main.py`), shared Pydantic schemas (`schemas.py`), Gemini structured verification engine & retry failover (`verifier.py`), `/api/export` wiring, and Render/Vercel cloud deployments. |
| **Chiranth** | 2-column IEEE/arXiv PDF reading-order & edge-case parsing (`parser.py`, `citation_matcher.py`), standalone PDF/Markdown report exporter (`report_exporter.py`), unit test suites, and evaluation benchmark CLI (`eval_benchmark.py`). |
| **D P Rithvik Kumar** | Multi-API scholarly source retrieval across OpenAlex, CrossRef, and arXiv (`retriever.py`) with retraction detection and title/year mismatch flagging. |
| **Dhanush** | React + TypeScript + Tailwind CSS v4 verification workbench UI (`App.tsx`), login gate, claim cards, dynamic filters, session history, and live `/health` diagnostics. |

---

## Disclaimer
CiteVerify is an automated decision-support tool designed to accelerate citation auditing—not a replacement for formal peer review or final academic judgment. Claims verified from abstract-only or paywalled metadata are explicitly flagged for human review.