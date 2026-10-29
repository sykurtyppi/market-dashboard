"""Durable latest-good snapshots for slow dashboard datasets."""
from datetime import datetime, timedelta, timezone
import threading
from types import SimpleNamespace
from unittest.mock import patch

from database.db_manager import DatabaseManager


def test_successful_snapshot_round_trips(tmp_path):
    db = DatabaseManager(str(tmp_path / "market.db"))
    payload = {"as_of": "2026-10-28T14:00:00Z", "sectors": [{"ticker": "XLK"}]}

    db.save_api_snapshot("sectors", payload, as_of=payload["as_of"])

    stored = db.get_api_snapshot("sectors")
    assert stored["dataset"] == "sectors"
    assert stored["payload"] == payload
    assert stored["as_of"] == payload["as_of"]
    assert stored["refresh_status"] == "ok"
    assert stored["last_error"] is None


def test_failed_refresh_preserves_last_good_payload(tmp_path):
    from api.snapshot_service import refresh_snapshot

    db = DatabaseManager(str(tmp_path / "market.db"))
    original = {"as_of": "2026-10-28T14:00:00Z", "sectors": [{"ticker": "XLK"}]}
    db.save_api_snapshot("sectors", original, as_of=original["as_of"])

    with patch("api.snapshot_service._collectors", {"sectors": (lambda: {"sectors": []}, lambda p: bool(p["sectors"]))}):
        result = refresh_snapshot("sectors", db=db)

    stored = db.get_api_snapshot("sectors")
    assert result["status"] == "failed"
    assert stored["payload"] == original
    assert stored["refresh_status"] == "failed"
    assert stored["last_error"]


def test_failed_first_refresh_exposes_failed_status_without_error_detail(tmp_path):
    from api.snapshot_service import refresh_snapshot, snapshot_response

    db = DatabaseManager(str(tmp_path / "market.db"))
    with patch("api.snapshot_service._collectors", {"sectors": (lambda: {"sectors": []}, lambda p: bool(p["sectors"]))}):
        result = refresh_snapshot("sectors", db=db)

    body = snapshot_response(
        db,
        "sectors",
        {"as_of": None, "sectors": [], "warnings": ["Snapshot unavailable."]},
        stale_after_seconds=60,
    )
    assert result["status"] == "failed"
    assert body["refresh_status"] == "failed"
    assert body["stale"] is True
    assert "collector returned" not in str(body)


def test_overlapping_refresh_for_same_dataset_is_rejected(tmp_path):
    from api.snapshot_service import refresh_snapshot

    db = DatabaseManager(str(tmp_path / "market.db"))
    entered = threading.Event()
    release = threading.Event()
    calls = 0

    def collect():
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(timeout=2)
        return {"as_of": "2026-10-28T14:00:00Z", "sectors": [{"ticker": "XLK"}]}

    results = []
    with patch("api.snapshot_service._collectors", {"sectors": (collect, lambda p: bool(p["sectors"]))}):
        worker = threading.Thread(target=lambda: results.append(refresh_snapshot("sectors", db=db)))
        worker.start()
        assert entered.wait(timeout=2)
        overlapping = refresh_snapshot("sectors", db=db)
        release.set()
        worker.join(timeout=2)

    assert overlapping == {"dataset": "sectors", "status": "already_running"}
    assert results[0]["status"] == "ok"
    assert calls == 1


def test_sectors_api_builder_reads_snapshot_without_calling_collectors():
    from api import sectors_service

    payload = {
        "as_of": "2026-10-28T14:00:00Z",
        "sectors": [{"ticker": "XLK", "name": "Technology", "category": "Cyclical", "change_pct": 1.2, "price": 180.5, "state": "good"}],
        "rotation": {"signal": "Risk-On", "state": "good", "interpretation": "Cyclicals leading", "leading_sectors": []},
        "vix_term": [],
        "vix_structure": None,
        "vix_source": "Yahoo Finance · Cboe volatility index closes",
        "warnings": [],
    }
    snapshot = {
        "dataset": "sectors",
        "payload": payload,
        "as_of": payload["as_of"],
        "refreshed_at": datetime.now(timezone.utc).isoformat(),
        "refresh_status": "ok",
        "last_error": None,
    }
    db = SimpleNamespace(get_api_snapshot=lambda dataset: snapshot)

    with patch.object(sectors_service, "get_db", return_value=db), \
         patch("data_collectors.sector_collector.SectorCollector", side_effect=AssertionError("request-time provider call")):
        body = sectors_service.build_sectors()

    assert body["sectors"][0]["ticker"] == "XLK"
    assert body["stale"] is False
    assert body["refresh_status"] == "ok"
    assert body["data_age_seconds"] >= 0


def test_options_flow_api_builder_serves_stale_good_snapshot_after_failure():
    from api import flows_service

    refreshed_at = (datetime.now(timezone.utc) - timedelta(hours=8)).isoformat()
    payload = {
        "as_of": "2026-10-28T13:00:00Z",
        "etfs": [{"ticker": "SPY", "price": 744.0, "expiry": "2026-10-30", "dte": 2, "put_call_ratio": 0.94, "call_volume": 900000, "put_volume": 850000, "sentiment": "BULLISH", "state": "good"}],
        "warnings": [],
    }
    snapshot = {
        "dataset": "options_flow",
        "payload": payload,
        "as_of": payload["as_of"],
        "refreshed_at": refreshed_at,
        "refresh_status": "failed",
        "last_error": "provider timeout",
    }
    db = SimpleNamespace(get_api_snapshot=lambda dataset: snapshot)

    with patch.object(flows_service, "get_db", return_value=db), \
         patch("data_collectors.options_flow_collector.OptionsFlowCollector", side_effect=AssertionError("request-time provider call")):
        body = flows_service.build_options_flow()

    assert body["etfs"][0]["ticker"] == "SPY"
    assert body["stale"] is True
    assert body["refresh_status"] == "failed"
    assert "provider timeout" not in str(body)


def test_missing_snapshot_returns_fast_explicit_empty_state():
    from api import sectors_service

    db = SimpleNamespace(get_api_snapshot=lambda dataset: None)
    with patch.object(sectors_service, "get_db", return_value=db), \
         patch("data_collectors.sector_collector.SectorCollector", side_effect=AssertionError("request-time provider call")):
        body = sectors_service.build_sectors()

    assert body["sectors"] == []
    assert body["stale"] is True
    assert body["refresh_status"] == "missing"
    assert any("snapshot" in warning.lower() for warning in body["warnings"])


def test_snapshot_refresh_route_fails_closed_without_server_token(monkeypatch):
    from fastapi.testclient import TestClient
    from api.main import app

    monkeypatch.delenv("MARKET_API_TOKEN", raising=False)
    response = TestClient(app).post("/api/snapshots/refresh/sectors")

    assert response.status_code == 503
    assert response.json()["detail"] == "Snapshot refresh authentication is not configured"


def test_snapshot_refresh_route_rejects_invalid_token(monkeypatch):
    from fastapi.testclient import TestClient
    from api.main import app

    monkeypatch.setenv("MARKET_API_TOKEN", "configured-test-token")
    response = TestClient(app).post(
        "/api/snapshots/refresh/sectors",
        headers={"X-API-Token": "wrong-test-token"},
    )

    assert response.status_code == 401


def test_snapshot_refresh_route_reports_overlap_as_conflict(monkeypatch):
    from fastapi.testclient import TestClient
    from api.main import app

    monkeypatch.setenv("MARKET_API_TOKEN", "configured-test-token")
    with patch(
        "api.snapshot_service.refresh_snapshot",
        return_value={"dataset": "sectors", "status": "already_running"},
    ):
        response = TestClient(app).post(
            "/api/snapshots/refresh/sectors",
            headers={"X-API-Token": "configured-test-token"},
        )

    assert response.status_code == 409
    assert response.json()["detail"] == "Snapshot refresh already running"
