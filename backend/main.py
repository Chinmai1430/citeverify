from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from backend.schemas import (
        ClaimVerificationResult,
        ExtractedClaim,
        RetrievedSource,
        VerificationReport,
    )
    from backend.services.parser import (
        extract_claims_from_pdf,
        extract_claims_from_text,
        extract_document_title,
        extract_text_from_pdf,
    )
    from backend.services.retriever import retrieve_sources_batch
    from backend.services.verifier import (
        build_verification_report,
        verify_single_claim,
    )
except ImportError:
    from schemas import (
        ClaimVerificationResult,
        ExtractedClaim,
        RetrievedSource,
        VerificationReport,
    )
    from services.parser import (
        extract_claims_from_pdf,
        extract_claims_from_text,
        extract_document_title,
        extract_text_from_pdf,
    )
    from services.retriever import retrieve_sources_batch
    from services.verifier import (
        build_verification_report,
        verify_single_claim,
    )

MAX_CLAIMS_PER_REQUEST = 25

app = FastAPI(
    title="CiteVerify API",
    description="Scientific Claim and Citation Verification System",
    version="1.0.0",
)

# Allow frontend (local Vite and Vercel) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SingleVerifyRequest(BaseModel):
    """Input payload to verify one claim against one retrieved source."""
    claim: ExtractedClaim
    source: RetrievedSource


class BatchVerifyRequest(BaseModel):
    """Input payload to verify a list of claim-source pairs for a document."""
    document_title: str = "Untitled Academic Document"
    items: List[SingleVerifyRequest]


def _run_full_pipeline(
    *,
    pdf_bytes: Optional[bytes] = None,
    pdf_filename: Optional[str] = None,
    raw_text: Optional[str] = None,
    document_title: Optional[str] = None,
) -> VerificationReport:
    """Runs parser -> retriever -> verifier end-to-end and returns a VerificationReport."""
    resolved_title = (document_title or "").strip()

    if pdf_bytes is not None:
        claims = extract_claims_from_pdf(pdf_bytes)
        if not resolved_title:
            try:
                extracted_text = extract_text_from_pdf(pdf_bytes)
                resolved_title = extract_document_title(extracted_text).strip()
            except Exception:
                resolved_title = ""
        if not resolved_title and pdf_filename:
            resolved_title = pdf_filename.rsplit(".", 1)[0]
    elif raw_text is not None:
        claims = extract_claims_from_text(raw_text)
        if not resolved_title:
            resolved_title = extract_document_title(raw_text).strip()
    else:
        raise ValueError("Either a PDF file or text input is required.")

    if not resolved_title:
        resolved_title = "Untitled Academic Document"

    if not claims:
        raise HTTPException(
            status_code=400,
            detail=(
                "No verifiable scientific claims were detected in the submitted "
                "document. Include factual statements with inline citations and "
                "a References section."
            ),
        )

    truncated_count = 0
    if len(claims) > MAX_CLAIMS_PER_REQUEST:
        truncated_count = len(claims) - MAX_CLAIMS_PER_REQUEST
        claims = claims[:MAX_CLAIMS_PER_REQUEST]

    sources = retrieve_sources_batch(claims)
    pairs = list(zip(claims, sources))
    report = build_verification_report(resolved_title, pairs)

    if truncated_count > 0:
        report.limitations.append(
            f"Document contained more than {MAX_CLAIMS_PER_REQUEST} claims; "
            f"verified the first {MAX_CLAIMS_PER_REQUEST} claims "
            f"({truncated_count} additional claims omitted for response latency)."
        )

    return report


@app.get("/")
def root():
    """Root endpoint confirming the CiteVerify backend is live."""
    return {"message": "Hello CiteVerify! API is running.", "status": "ok"}


@app.get("/health")
def health_check():
    """Health check endpoint used by Render and deployment monitors."""
    return {"status": "healthy"}


@app.post("/api/parse", response_model=List[ExtractedClaim])
async def parse_document_endpoint(
    file: Optional[UploadFile] = File(default=None),
    text: Optional[str] = Form(default=None),
):
    """Extracts verifiable claims and matched citations from an uploaded PDF or raw text."""
    if file is not None:
        pdf_bytes = await file.read()
        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="Uploaded PDF file is empty.")
        return await run_in_threadpool(extract_claims_from_pdf, pdf_bytes)

    if text and text.strip():
        return await run_in_threadpool(extract_claims_from_text, text.strip())

    raise HTTPException(
        status_code=400,
        detail="Provide either a PDF 'file' or non-empty 'text' field.",
    )


@app.post("/api/verify", response_model=VerificationReport)
async def verify_document_endpoint(request: Request):
    """
    Full end-to-end verification endpoint consumed by the CiteVerify UI.
    Supports both:
    - multipart/form-data: 'file' (PDF) and optional 'document_title' (or 'text')
    - application/json: {"text": "...", "document_title": "..."}
    """
    content_type = (request.headers.get("content-type") or "").lower()

    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON request body.")

        raw_text = (body.get("text") or "").strip() if isinstance(body, dict) else ""
        document_title = (
            body.get("document_title") if isinstance(body, dict) else None
        )

        if not raw_text:
            raise HTTPException(
                status_code=400,
                detail="Please provide non-empty 'text' to verify.",
            )

        return await run_in_threadpool(
            _run_full_pipeline,
            raw_text=raw_text,
            document_title=document_title,
        )

    # Default: multipart/form-data (PDF upload or form-encoded text)
    try:
        form = await request.form()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Expected multipart/form-data (PDF upload) or application/json.",
        )

    uploaded = form.get("file")
    document_title_val = form.get("document_title")
    document_title = str(document_title_val) if document_title_val else None

    if uploaded is not None and hasattr(uploaded, "read"):
        filename = getattr(uploaded, "filename", "") or ""
        if filename and not filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files (.pdf) are supported for file upload.",
            )
        pdf_bytes = await uploaded.read()
        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="Uploaded PDF file is empty.")

        return await run_in_threadpool(
            _run_full_pipeline,
            pdf_bytes=pdf_bytes,
            pdf_filename=filename,
            document_title=document_title,
        )

    raw_text_val = form.get("text")
    raw_text = str(raw_text_val).strip() if raw_text_val else ""
    if raw_text:
        return await run_in_threadpool(
            _run_full_pipeline,
            raw_text=raw_text,
            document_title=document_title,
        )

    raise HTTPException(
        status_code=400,
        detail="No PDF 'file' or 'text' payload was provided.",
    )


@app.post("/api/verify-claim", response_model=ClaimVerificationResult)
def verify_claim_endpoint(payload: SingleVerifyRequest):
    """Verifies a single claim against its retrieved source passage."""
    return verify_single_claim(payload.claim, payload.source)


@app.post("/api/verify-batch", response_model=VerificationReport)
def verify_batch_endpoint(payload: BatchVerifyRequest):
    """Verifies multiple claims and returns a full claim-by-claim VerificationReport."""
    pairs = [(item.claim, item.source) for item in payload.items]
    return build_verification_report(payload.document_title, pairs)