"""Offline regressions for refresh outcomes and observation provenance."""
import sqlite3
from datetime import datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from api import overview_service, pages_service
from database.db_manager import DatabaseManager
from tests.test_scheduler_outcome import WorkflowDB, workflow_updater


@pytest.fixture(autouse=True)
def stub_unrelated_briefing(monkeypatch):
    monkeypatch.setattr("api.daily_briefing_service.generate_latest_briefing",
                        lambda db: {"posture": "test", "change": {"current_score": None}})


def test_existing_schema_adds_provenance_without_backfilling_or_erasing_values(tmp_path):
    path = str(tmp_path / "old-schema.db")
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE daily_snapshots (id INTEGER PRIMARY KEY, date DATE UNIQUE, "
                     "credit_spread_ig REAL, treasury_10y REAL)")
        conn.execute("INSERT INTO daily_snapshots (date, credit_spread_ig, treasury_10y) "
                     "VALUES ('2026-11-13', 5.2, 4.2)")
    # Repeated opens must not invent provenance or modify either raw value.
    for _ in range(2):
        db = DatabaseManager(path)
        snapshot = db.get_latest_snapshot()
        assert snapshot is not None
        assert snapshot["credit_spread_ig"] is None
        assert snapshot["credit_spread_ig_source"] is None
        assert snapshot["treasury_10y_source"] is None
        with sqlite3.connect(path) as conn:
            assert conn.execute("SELECT credit_spread_ig, treasury_10y FROM daily_snapshots").fetchall() == [(5.2, 4.2)]


@pytest.mark.parametrize("fred", [None, SimpleNamespace(get_all_indicators=lambda: {}),
    SimpleNamespace(get_all_indicators=lambda: {"treasury_10y": pd.DataFrame()})])
def test_missing_or_empty_fred_is_a_failed_expected_phase(fred):
    db = WorkflowDB()
    updater = workflow_updater(fred=fred, db=db)
    result = updater.run_full_update()
    assert "fred_indicators" in result
    assert "fred_indicators" in result.attempted_phases
    assert db.snapshots, "independent work must still complete"


@pytest.mark.parametrize("fred_value,yahoo_value,expected,source", [
    (4.1, 4.2, 4.1, "FRED (DGS10)"),
    (None, 4.2, 4.2, "Yahoo Finance (^TNX)"),
    (None, None, None, "Unavailable"),
])
def test_treasury_source_survives_refresh_persistence_and_api(
    tmp_path, monkeypatch, fred_value, yahoo_value, expected, source
):
    fred = None if fred_value is None else SimpleNamespace(get_all_indicators=lambda: {
        "treasury_10y": pd.DataFrame({"date": [pd.Timestamp("2026-11-13")], "DGS10": [fred_value]})
    })
    db = WorkflowDB()
    workflow_updater(fred=fred, yahoo=SimpleNamespace(
        get_all_data=lambda: {"treasury_10y": yahoo_value}), db=db).run_full_update()
    real_db = DatabaseManager(str(tmp_path / "treasury.db"))
    assert real_db.save_daily_snapshot(db.snapshots[0])[0]
    monkeypatch.setattr(overview_service, "get_db", lambda: real_db)
    metric = next(m for m in overview_service.build_overview()["metrics"] if m["key"] == "ten_y")
    assert metric["value"] == expected
    assert metric["source"] == source


def test_legacy_treasury_is_not_relabelled_fred(tmp_path, monkeypatch):
    db = DatabaseManager(str(tmp_path / "legacy.db"))
    assert db.save_daily_snapshot({"date": "2026-11-13", "treasury_10y": 4.2})[0]
    monkeypatch.setattr(overview_service, "get_db", lambda: db)
    metric = next(m for m in overview_service.build_overview()["metrics"] if m["key"] == "ten_y")
    assert metric["value"] == 4.2
    assert metric["source"] == "Unknown (legacy snapshot)"


def test_ig_history_only_exposes_verified_oas_without_deleting_raw_rows(tmp_path, monkeypatch):
    db = DatabaseManager(str(tmp_path / "ig.db"))
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    yield_date = (now - timedelta(days=2)).strftime("%Y-%m-%d")
    unknown_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    rows = [("credit_spread_ig", "BAMLC0A0CMEY", yield_date, 5.2),
            ("credit_spread_ig", None, unknown_date, 4.9),
            ("credit_spread_ig", "BAMLC0A0CM", today, 0.85)]
    with sqlite3.connect(db.db_path) as conn:
        conn.executemany("INSERT INTO indicators (indicator_name, series_id, date, value) VALUES (?,?,?,?)", rows)
    assert db.get_indicator_history("credit_spread_ig", days=20000)["value"].tolist() == [0.85]
    # Both public credit charts and the IG headline must use the safe reader.
    for service, build in [(overview_service, overview_service.build_overview),
                           (pages_service, pages_service.build_credit_liquidity)]:
        monkeypatch.setattr(service, "get_db", lambda: db)
        body = build()
        assert [p["value"] for p in body["charts"]["credit_spreads"]["ig"]] == [0.85]
        if service is pages_service:
            assert next(m for m in body["metrics"] if m["key"] == "ig")["value"] == 0.85
    with sqlite3.connect(db.db_path) as conn:
        assert conn.execute("SELECT series_id, value FROM indicators ORDER BY date").fetchall() == [
            ("BAMLC0A0CMEY", 5.2), (None, 4.9), ("BAMLC0A0CM", 0.85)]


@pytest.mark.parametrize("source,expected", [(None, None), ("FRED (BAMLC0A0CMEY)", None),
    ("FRED (BAMLC0A0CM)", 0.85)])
def test_ig_snapshot_readers_require_oas_provenance_and_preserve_storage(tmp_path, source, expected):
    db = DatabaseManager(str(tmp_path / "snapshot.db"))
    snapshot = {"date": "2026-11-13", "credit_spread_ig": 0.85,
                "credit_spread_ig_source": source}
    assert db.save_daily_snapshot(snapshot)[0]
    assert db.get_latest_snapshot()["credit_spread_ig"] == expected
    assert db.get_recent_snapshots()[0]["credit_spread_ig"] == expected
    with sqlite3.connect(db.db_path) as conn:
        assert conn.execute("SELECT credit_spread_ig FROM daily_snapshots").fetchone() == (0.85,)


def test_refresh_marks_actual_ig_series_and_reader_rejects_yield(tmp_path):
    for series, expected in [("BAMLC0A0CM", 0.85), ("BAMLC0A0CMEY", None)]:
        db = WorkflowDB()
        fred = SimpleNamespace(get_all_indicators=lambda: {"credit_spread_ig": pd.DataFrame({
            "date": [pd.Timestamp("2026-11-13")], series: [0.85]})})
        workflow_updater(fred=fred, db=db).run_full_update()
        assert db.snapshots[0]["credit_spread_ig_source"] == f"FRED ({series})"
        real_db = DatabaseManager(str(tmp_path / f"{series}.db"))
        assert real_db.save_daily_snapshot(db.snapshots[0])[0]
        assert real_db.get_latest_snapshot()["credit_spread_ig"] == expected
