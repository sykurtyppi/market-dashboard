"""FastAPI application for the Market Risk Dashboard.

Phase 0 surface: /api/overview, /api/freshness, /api/health.
Reads cached data from SQLite; never calls collectors live per request.
"""
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.deps import get_db, get_health
from api.overview_service import build_overview
from api.schemas import FreshnessDetail, HealthResponse, OverviewResponse

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Market Risk Dashboard API",
    version="0.1.0",
    description="HTTP API over the market-risk backend. Phase 0.",
)

# Frontend dev + deployed origins. Overridable via FRONTEND_ORIGINS (comma-sep).
_origins = os.getenv(
    "FRONTEND_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health():
    return get_health().get_health_summary()


@app.get("/api/freshness", response_model=FreshnessDetail)
def freshness():
    snap = get_db().get_latest_snapshot(include_age=True) or {}
    return {
        "as_of": snap.get("date"),
        "status": snap.get("_status", "unknown"),
        "age": snap.get("_age_string", "unknown"),
        "age_hours": snap.get("_age_hours"),
        "is_fresh": bool(snap.get("_is_fresh", False)),
    }


@app.get("/api/overview", response_model=OverviewResponse)
def overview():
    return build_overview()
