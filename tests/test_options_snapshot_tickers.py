# Options snapshot ticker validation regression.
"""Reject duplicate or unexpected ETF tickers; preserve the last good snapshot.

The options_flow validator used to accept any three rows, so a snapshot with
duplicate or unexpected tickers overwrote a good one. A valid snapshot must
have exactly one SPY, QQQ and IWM row in any order.
"""
from unittest.mock import call, patch

from api import snapshot_service as svc
from api.snapshot_service import refresh_snapshot
from database.db_manager import DatabaseManager

OPTION_TICKERS = ("SPY", "QQQ", "IWM")


def _payload(tickers, as_of):
    return {"as_of": as_of, "etfs": [{"ticker": t} for t in tickers]}


def test_duplicate_ticker_refresh_is_rejected(tmp_path, monkeypatch):
    db = DatabaseManager(str(tmp_path / "snapshot.sqlite"))
    original = _payload(OPTION_TICKERS, "first")
    db.save_api_snapshot("options_flow", original, as_of=original["as_of"])

    # Preserve the real validator; stub only the collector to a duplicate row.
    _, validator = svc._collectors["options_flow"]
    monkeypatch.setitem(
        svc._collectors,
        "options_flow",
        (lambda: _payload(["SPY", "QQQ", "QQQ"], "second"), validator),
    )

    result = refresh_snapshot("options_flow", db=db)
    stored = db.get_api_snapshot("options_flow")
    assert result["status"] == "failed"
    assert stored["payload"] == original


def test_valid_refresh_accepts_any_order(tmp_path, monkeypatch):
    db = DatabaseManager(str(tmp_path / "snapshot.sqlite"))
    payload = _payload(("IWM", "SPY", "QQQ"), "valid")
    _, validator = svc._collectors["options_flow"]
    monkeypatch.setitem(
        svc._collectors,
        "options_flow",
        (lambda: payload, validator),
    )
    result = refresh_snapshot("options_flow", db=db)
    assert result["status"] == "ok"
    assert db.get_api_snapshot("options_flow")["payload"] == payload


def test_sectors_validator_is_unchanged(tmp_path, monkeypatch):
    db = DatabaseManager(str(tmp_path / "snapshot.sqlite"))
    sectors = {"as_of": "valid", "sectors": [{"ticker": "XLK"}]}
    _, sectors_validator = svc._collectors["sectors"]
    monkeypatch.setitem(
        svc._collectors,
        "sectors",
        (lambda: sectors, sectors_validator),
    )
    assert refresh_snapshot("sectors", db=db)["status"] == "ok"
    assert db.get_api_snapshot("sectors")["payload"] == sectors
