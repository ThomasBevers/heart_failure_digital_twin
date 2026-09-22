"""Heart Failure Digital Twin API.

Run with:
    uvicorn api.main:app --reload --app-dir backend

Interactive docs at /docs (Swagger) and /redoc once running.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.heart_twin.echonet_integration import DEFAULT_EF_CHECKPOINT, DEFAULT_VOLUME_ESTIMATOR

from .routers import echo, patients, simulate
from .schemas import HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("heart_twin.api")

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

app = FastAPI(
    title="Heart Failure Digital Twin API",
    description=(
        "Patient-specific, closed-loop cardiovascular simulation, plus optional "
        "EchoNet-Dynamic-based EF/EDV/ESV inference from an uploaded echo video. "
        "Research prototype only -- not clinically validated."
    ),
    version="1.0.0",
)

# Permissive by default for local frontend development; set ALLOWED_ORIGINS to a
# comma-separated list (e.g. "http://localhost:5500,https://your-frontend.example")
# in production rather than relying on the "*" default.
_allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _allowed_origins == "*" else [origin.strip() for origin in _allowed_origins.split(",")],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Never leak a raw traceback to the client; log it server-side instead."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


app.include_router(patients.router)
app.include_router(simulate.router)
app.include_router(echo.router)


@app.get("/api/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    """Reports which optional capabilities are actually configured, not just that the server is up."""
    vendor_echonet = BACKEND_DIR / "vendor" / "echonet_dynamic" / "echonet"
    return HealthResponse(
        status="ok",
        echonet_dynamic_available=vendor_echonet.is_dir(),
        ef_checkpoint_available=DEFAULT_EF_CHECKPOINT.exists(),
        volume_estimator_available=DEFAULT_VOLUME_ESTIMATOR.exists(),
    )


@app.get("/api", tags=["health"])
def api_info() -> dict:
    return {
        "name": app.title,
        "version": app.version,
        "docs": "/docs",
        "health": "/api/health",
    }


# If a built frontend exists (frontend/index.html), serve it from the same
# process so the app can be run as a single server in dev. If it doesn't
# exist yet, the API still works fine on its own -- this mount is skipped.
if (FRONTEND_DIR / "index.html").is_file():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
    logger.info("Serving frontend from %s", FRONTEND_DIR)
else:
    logger.info("No frontend/index.html found -- running API-only (docs at /docs).")
