from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from schemas import (
    ClaimVerificationResult,
    ExtractedClaim,
    RetrievedSource,
    VerificationReport,
)
from services.verifier import build_verification_report, verify_single_claim

app = FastAPI(
    title="CiteVerify API",
    description="Scientific Claim and Citation Verification System",
    version="0.2.0",
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


@app.get("/")
def root():
    """Root endpoint confirming the CiteVerify backend is live."""
    return {"message": "Hello CiteVerify! API is running.", "status": "ok"}


@app.get("/health")
def health_check():
    """Health check endpoint used by Render and deployment monitors."""
    return {"status": "healthy"}


@app.post("/api/verify-claim", response_model=ClaimVerificationResult)
def verify_claim_endpoint(payload: SingleVerifyRequest):
    """Verifies a single claim against its retrieved source passage."""
    return verify_single_claim(payload.claim, payload.source)


@app.post("/api/verify-batch", response_model=VerificationReport)
def verify_batch_endpoint(payload: BatchVerifyRequest):
    """Verifies multiple claims and returns a full claim-by-claim VerificationReport."""
    pairs = [(item.claim, item.source) for item in payload.items]
    return build_verification_report(payload.document_title, pairs)