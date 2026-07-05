"""FastAPI application for the Market Risk Dashboard.

Surface: /api/overview, /api/volatility, /api/breadth, /api/freshness,
/api/health (GET, cached SQLite) and /api/refresh (POST, background update).
Read endpoints never call collectors live per request.
"""
import logging
import os
import threading

from fastapi import BackgroundTasks, FastAPI, Header, Response, status
from fastapi.middleware.cors import CORSMiddleware

from api.deps import get_db, get_health
from api.overview_service import build_overview
from api.pages_service import (
    build_breadth,
    build_credit_liquidity,
    build_repo,
    build_treasury_stress,
    build_volatility,
)
from api.macro_service import build_cross_asset, build_fed_watch
from api.sectors_service import build_sectors
from api.schemas import (
    BreadthResponse,
    CreditLiquidityResponse,
    CrossAssetResponse,
    FedWatchResponse,
    FreshnessDetail,
    HealthResponse,
    OverviewResponse,
    RefreshResponse,
    RefreshStatus,
    RepoResponse,
    SectorsResponse,
    TreasuryResponse,
    VolatilityResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Market Risk Dashboard API",
    version="0.2.0",
    description="HTTP API over the market-risk backend.",
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

# Serialize data refreshes: only one background update runs at a time.
_refresh_lock = threading.Lock()


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


@app.get("/api/volatility", response_model=VolatilityResponse)
def volatility():
    return build_volatility()


@app.get("/api/breadth", response_model=BreadthResponse)
def breadth():
    return build_breadth()


@app.get("/api/sectors", response_model=SectorsResponse)
def sectors():
    return build_sectors()


@app.get("/api/credit-liquidity", response_model=CreditLiquidityResponse)
def credit_liquidity():
    return build_credit_liquidity()


@app.get("/api/treasury-stress", response_model=TreasuryResponse)
def treasury_stress():
    return build_treasury_stress()


@app.get("/api/repo", response_model=RepoResponse)
def repo():
    return build_repo()


@app.get("/api/fed-watch", response_model=FedWatchResponse)
def fed_watch():
    return build_fed_watch()


@app.get("/api/cross-asset", response_model=CrossAssetResponse)
def cross_asset():
    return build_cross_asset()


def _run_refresh():
    """Run a full data update, releasing the lock when done."""
    try:
        from scheduler.daily_update import MarketDataUpdater
        MarketDataUpdater().run_full_update()
    except Exception as exc:  # noqa: BLE001 - log and move on; lock still releases
        logger.error("Data refresh failed: %s", exc)
    finally:
        _refresh_lock.release()


@app.post("/api/refresh", response_model=RefreshResponse)
def refresh(
    background_tasks: BackgroundTasks,
    response: Response,
    x_api_token: str | None = Header(default=None),
):
    """Kick off a full data refresh in the background.

    Live and slow (hits FRED/Yahoo/CBOE), so it runs as a background task and
    returns immediately; poll /api/refresh/status for completion. If
    MARKET_API_TOKEN is set, the caller must supply it via the X-API-Token header.
    """
    expected = os.getenv("MARKET_API_TOKEN")
    if expected and x_api_token != expected:
        response.status_code = status.HTTP_401_UNAUTHORIZED
        return {"status": "unauthorized", "detail": "Invalid or missing X-API-Token"}

    if not _refresh_lock.acquire(blocking=False):
        return {"status": "already_running", "detail": "A refresh is already in progress"}

    background_tasks.add_task(_run_refresh)
    return {"status": "started", "detail": "Refresh started; poll /api/refresh/status for completion"}


@app.get("/api/refresh/status", response_model=RefreshStatus)
def refresh_status():
    return {"running": _refresh_lock.locked()}
