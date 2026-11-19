from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="CiteVerify API",
    description="Scientific Claim and Citation Verification System",
    version="0.1.0",
)

# Allow frontend (local Vite and Vercel) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    """Root endpoint confirming the CiteVerify backend is live."""
    return {"message": "Hello CiteVerify! API is running.", "status": "ok"}


@app.get("/health")
def health_check():
    """Health check endpoint used by Render and deployment monitors."""
    return {"status": "healthy"}