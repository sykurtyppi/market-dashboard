"""Smoke tests for the FastAPI layer (api/main.py).

Uses the DB as-is (read-only endpoints); asserts the endpoints respond with the
expected shape. No network — the overview endpoint reads cached SQLite data.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

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
