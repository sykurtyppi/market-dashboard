"""Smoke tests for the FastAPI layer (api/main.py).

Uses the DB as-is (read-only endpoints); asserts the endpoints respond with the
expected shape. No network — the overview endpoint reads cached SQLite data.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_freshness_returns_expected_shape(client):
    r = client.get("/api/freshness")
    assert r.status_code == 200
    body = r.json()
    for key in ("status", "age", "is_fresh"):
        assert key in body
    assert isinstance(body["is_fresh"], bool)


def test_overview_returns_expected_shape(client):
    r = client.get("/api/overview")
    assert r.status_code == 200
    body = r.json()
    assert set(["freshness", "regime", "metrics", "charts", "detail"]).issubset(body)
    assert isinstance(body["metrics"], list)
    assert isinstance(body["regime"]["components"], list)
    assert "vrp_history" in body["charts"]
    assert "credit_spreads" in body["charts"]


def test_overview_metrics_have_valid_states(client):
    r = client.get("/api/overview")
    body = r.json()
    valid = {"good", "warn", "crit", "neutral"}
    for m in body["metrics"]:
        assert m["state"] in valid
        assert "label" in m and "source" in m


def test_overview_chart_series_are_bounded(client):
    # Downsampling caps each series at 180 points (+ the forced last point).
    r = client.get("/api/overview")
    body = r.json()
    assert len(body["charts"]["vrp_history"]) <= 181
    assert len(body["charts"]["credit_spreads"]["hy"]) <= 181


def test_health_endpoint_responds(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert "overall_status" in r.json()


# --- Phase 1 pages ---

def test_volatility_returns_expected_shape(client):
    r = client.get("/api/volatility")
    assert r.status_code == 200
    body = r.json()
    assert {"regime_note", "metrics", "charts"}.issubset(body)
    for key in ("vrp_history", "vix", "realized_vol"):
        assert key in body["charts"]
    valid = {"good", "warn", "crit", "neutral"}
    assert all(m["state"] in valid for m in body["metrics"])


def test_breadth_returns_expected_shape(client):
    r = client.get("/api/breadth")
    assert r.status_code == 200
    body = r.json()
    assert "metrics" in body
    for key in ("ad_line", "mcclellan", "breadth_pct"):
        assert key in body["charts"]


# --- refresh (mock the updater so no live calls / DB writes happen) ---

def test_refresh_starts_and_reports_status(client):
    # TestClient runs the background task synchronously after the response, so
    # patch the updater to a no-op to avoid a real data pull.
    with patch("scheduler.daily_update.MarketDataUpdater") as MockUpdater:
        MockUpdater.return_value.run_full_update.return_value = None
        r = client.post("/api/refresh")
        assert r.status_code == 200
        assert r.json()["status"] in ("started", "already_running")
        MockUpdater.return_value.run_full_update.assert_called_once()

    status = client.get("/api/refresh/status")
    assert status.status_code == 200
    assert isinstance(status.json()["running"], bool)


def test_refresh_rejects_bad_token_when_configured(client, monkeypatch):
    monkeypatch.setenv("MARKET_API_TOKEN", "secret")
    r = client.post("/api/refresh")  # no X-API-Token header
    assert r.json()["status"] == "unauthorized"
