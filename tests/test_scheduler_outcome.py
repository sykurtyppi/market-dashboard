"""The refresh must say when a phase failed.

run_full_update used to print UPDATE COMPLETE unconditionally and ignore the
return values of its own writes. Phases now record failures via
_phase_failed(); _finish() reports them and run_full_update returns them.
"""
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db_manager import DatabaseManager  # noqa: E402
from api import overview_service  # noqa: E402
from scheduler.daily_update import MarketDataUpdater  # noqa: E402


class WorkflowDB:
    def __init__(self, *, fred_saved=True, breadth_saved=True, vrp_saved=(True, None)):
        self.fred_saved = fred_saved
        self.breadth_saved = breadth_saved
        self.vrp_saved = vrp_saved
        self.snapshots = []

    def save_indicators_batch(self, data):
        return self.fred_saved

    def save_daily_snapshot(self, snapshot):
        self.snapshots.append(snapshot)
        return True, SimpleNamespace(errors=[])

    def save_breadth_data(self, data):
        return self.breadth_saved

    def save_indicator(self, *args, **kwargs):
        return True

    def save_vrp_data(self, data):
        return self.vrp_saved

    def save_benchmark_prices(self, symbol, prices):
        return None


def workflow_updater(*, fred=None, fear_greed=None, yahoo=None, db=None):
    u = MarketDataUpdater.__new__(MarketDataUpdater)
    u.failed_phases = []
    u.fred = fred
    u.fear_greed = fear_greed or SimpleNamespace(get_fear_greed_score=lambda: {"score": 50})
    u.cboe = SimpleNamespace(get_all_data=lambda: {})
    u.yahoo = yahoo or SimpleNamespace(get_all_data=lambda: {})
    u.breadth = SimpleNamespace(get_breadth_history=lambda days=90: pd.DataFrame())
    u.fed_bs = None
    u.move = SimpleNamespace(get_full_snapshot=lambda: None)
    u.repo = None
    u.liquidity = None
    u.db = db or WorkflowDB()
    u.left_strategy = SimpleNamespace(calculate_signal=lambda frame: {"signal": "INSUFFICIENT_DATA"})
    u.vrp_analyzer = SimpleNamespace(get_complete_analysis=lambda vix: {"error": "not used"})
    return u


def liquidity_frame():
    # Units as the collector delivers them: TGA in $ millions, RRP in $ billions.
    return pd.DataFrame({
        "date": [pd.Timestamp("2026-09-24"), pd.Timestamp("2026-09-25")],
        "rrp_on": [0.4, 0.5],
        "tga": [900_000.0, 977_000.0],
        "sofr": [3.88, 3.88],
    })


@pytest.fixture
def updater(tmp_path):
    """An updater with only the attributes the liquidity phase touches —
    __init__ would construct every live collector."""
    u = MarketDataUpdater.__new__(MarketDataUpdater)
    u.failed_phases = []
    u.db = DatabaseManager(str(tmp_path / "sched.db"))
    u.liquidity = SimpleNamespace(get_all_liquidity=lambda lookback_days=365: liquidity_frame())
    return u


def test_phase_failed_records_and_logs(updater, caplog):
    with caplog.at_level(logging.ERROR):
        updater._phase_failed("repo_market", RuntimeError("boom"))
    assert updater.failed_phases == ["repo_market"]
    assert "repo_market failed: boom" in caplog.text


def test_finish_reports_failures_and_returns_them(updater, caplog):
    updater.failed_phases = ["fed_balance_sheet", "liquidity_history"]
    with caplog.at_level(logging.INFO):
        result = updater._finish()
    assert result == ["fed_balance_sheet", "liquidity_history"]
    assert "UPDATE COMPLETE WITH FAILURES: fed_balance_sheet, liquidity_history" in caplog.text
    assert any(r.levelno == logging.ERROR for r in caplog.records)


def test_finish_is_clean_when_nothing_failed(updater, caplog):
    with caplog.at_level(logging.INFO):
        assert updater._finish() == []
    assert "UPDATE COMPLETE" in caplog.text
    assert "WITH FAILURES" not in caplog.text


def test_liquidity_phase_succeeds_and_writes(updater):
    updater._update_liquidity_history()
    assert updater.failed_phases == []
    with sqlite3.connect(updater.db.db_path) as con:
        assert con.execute("SELECT COUNT(*) FROM liquidity_history").fetchone() == (2,)


def test_liquidity_phase_records_a_failed_write_and_logs_no_success(updater, caplog):
    updater.db.save_liquidity_history = lambda df: False
    with caplog.at_level(logging.INFO):
        updater._update_liquidity_history()
    assert updater.failed_phases == ["liquidity_history"]
    # The old code logged the latest TGA / net liquidity as if the save worked.
    assert "Net Liquidity" not in caplog.text
    assert "TGA: $" not in caplog.text


def test_liquidity_phase_records_an_empty_collector(updater):
    updater.liquidity = SimpleNamespace(get_all_liquidity=lambda lookback_days=365: pd.DataFrame())
    updater._update_liquidity_history()
    assert updater.failed_phases == ["liquidity_history"]


def test_liquidity_phase_records_a_crash_without_raising(updater):
    updater.liquidity = SimpleNamespace(get_all_liquidity=lambda lookback_days=365: (_ for _ in ()).throw(RuntimeError("feed down")))
    updater._update_liquidity_history()   # must not raise: phases are isolated
    assert updater.failed_phases == ["liquidity_history"]


def test_liquidity_phase_disabled_by_configuration_is_not_a_failure(updater, caplog):
    # No FRED key means the collector was never built. That is configuration,
    # not a failed phase — same treatment as fed_balance_sheet and repo_market.
    updater.liquidity = None
    with caplog.at_level(logging.WARNING):
        updater._update_liquidity_history()
    assert updater.failed_phases == []
    assert "not available" in caplog.text


def test_fred_save_false_marks_phase_failed_but_snapshot_still_runs():
    frame = pd.DataFrame({"date": ["2026-10-26"], "DGS10": [4.1]})
    db = WorkflowDB(fred_saved=False)
    updater = workflow_updater(
        fred=SimpleNamespace(get_all_indicators=lambda: {"treasury_10y": frame}),
        db=db,
    )

    failed = updater.run_full_update()

    assert "fred_indicators" in failed
    assert db.snapshots, "independent snapshot phase must still run"


def test_repo_save_false_marks_phase_failed_without_aborting_snapshot(caplog):
    repo = pd.DataFrame({
        "date": [pd.Timestamp("2026-10-26")],
        "sofr": [3.88],
        "iorb": [3.90],
        "rrp_on": [82.5],
        "sofr_iorb_spread": [-2.0],
    })
    db = WorkflowDB()
    db.save_repo_data = lambda data: False
    updater = workflow_updater(db=db)
    updater.repo = SimpleNamespace(get_repo_history=lambda days_back=730: repo)

    with caplog.at_level(logging.INFO):
        failed = updater.run_full_update()

    assert "repo_market" in failed
    assert db.snapshots, "independent snapshot phase must still run"
    assert "Repo data saved to database" not in caplog.text


def test_fred_save_exception_marks_phase_failed_but_snapshot_still_runs():
    frame = pd.DataFrame({"date": ["2026-10-26"], "DGS10": [4.1]})
    db = WorkflowDB()
    db.save_indicators_batch = lambda data: (_ for _ in ()).throw(RuntimeError("disk full"))
    updater = workflow_updater(
        fred=SimpleNamespace(get_all_indicators=lambda: {"treasury_10y": frame}),
        db=db,
    )

    failed = updater.run_full_update()

    assert "fred_indicators" in failed
    assert db.snapshots, "independent snapshot phase must still run"


def test_optional_collector_exception_does_not_abort_unrelated_phases():
    yahoo_calls = []
    db = WorkflowDB()
    updater = workflow_updater(
        fear_greed=SimpleNamespace(
            get_fear_greed_score=lambda: (_ for _ in ()).throw(RuntimeError("offline"))
        ),
        yahoo=SimpleNamespace(
            get_all_data=lambda: yahoo_calls.append(True) or {"hy_spread_proxy": 4.2}
        ),
        db=db,
    )

    failed = updater.run_full_update()

    assert "fear_greed" in failed
    assert yahoo_calls == [True]
    assert db.snapshots[0]["credit_spread_hy"] == 4.2
    assert db.snapshots[0]["credit_spread_hy_source"] == (
        "Yahoo Finance (HYG yield − 10Y Treasury proxy)"
    )


def test_refresh_uses_one_new_york_market_date_across_utc_midnight():
    benchmark_cutoffs = []
    db = WorkflowDB()
    updater = workflow_updater(
        yahoo=SimpleNamespace(
            get_all_data=lambda: {},
            get_benchmark_prices=lambda *, completed_before: benchmark_cutoffs.append(
                completed_before
            ) or [("2026-10-28", 100.0)],
        ),
        db=db,
    )
    updater.run_full_update(
        run_at=datetime(2026, 10, 30, 0, 30, tzinfo=timezone.utc)
    )

    assert benchmark_cutoffs == [datetime(2026, 10, 29).date()]
    assert db.snapshots[0]["date"] == "2026-10-29"


def test_fred_credit_observation_keeps_its_series_provenance():
    credit = pd.DataFrame({
        "date": [pd.Timestamp("2026-10-26")],
        "BAMLH0A0HYM2": [3.0],
    })
    db = WorkflowDB()
    updater = workflow_updater(
        fred=SimpleNamespace(get_all_indicators=lambda: {"credit_spread_hy": credit}),
        yahoo=SimpleNamespace(get_all_data=lambda: {"hy_spread_proxy": 4.2}),
        db=db,
    )

    updater.run_full_update()

    assert db.snapshots[0]["credit_spread_hy"] == 3.0
    assert db.snapshots[0]["credit_spread_hy_source"] == "FRED (BAMLH0A0HYM2)"


@pytest.mark.parametrize(
    ("fred", "yahoo", "expected_value", "expected_source"),
    [
        (
            SimpleNamespace(get_all_indicators=lambda: {
                "credit_spread_hy": pd.DataFrame({
                    "date": [pd.Timestamp("2026-10-26")],
                    "BAMLH0A0HYM2": [3.0],
                })
            }),
            SimpleNamespace(get_all_data=lambda: {"hy_spread_proxy": 4.2}),
            3.0,
            "FRED (BAMLH0A0HYM2)",
        ),
        (
            None,
            SimpleNamespace(get_all_data=lambda: {"hy_spread_proxy": 4.2}),
            4.2,
            "Yahoo Finance (HYG yield − 10Y Treasury proxy)",
        ),
    ],
)
def test_credit_provenance_survives_selection_persistence_and_api(
    tmp_path, monkeypatch, fred, yahoo, expected_value, expected_source
):
    workflow_db = WorkflowDB()
    updater = workflow_updater(fred=fred, yahoo=yahoo, db=workflow_db)
    updater.run_full_update()

    selected = workflow_db.snapshots[0]
    assert selected["credit_spread_hy"] == expected_value
    assert selected["credit_spread_hy_source"] == expected_source

    real_db = DatabaseManager(str(tmp_path / "provenance.db"))
    saved, validation = real_db.save_daily_snapshot(selected)
    assert saved, validation.errors

    monkeypatch.setattr(overview_service, "get_db", lambda: real_db)
    body = overview_service.build_overview()
    credit = next(metric for metric in body["metrics"] if metric["key"] == "hy")
    assert credit["value"] == expected_value
    assert credit["source"] == expected_source


def test_tuple_save_failures_degrade_their_refresh_phases():
    breadth = pd.DataFrame({"date": ["2026-10-26"], "breadth_pct": [55.0]})
    db = WorkflowDB(
        breadth_saved=(0, ["database error"]),
        vrp_saved=(False, SimpleNamespace(errors=["invalid"])),
    )
    updater = workflow_updater(
        yahoo=SimpleNamespace(get_all_data=lambda: {"vix": 13.2}),
        db=db,
    )
    updater.breadth = SimpleNamespace(get_breadth_history=lambda days=90: breadth)
    updater.vrp_analyzer = SimpleNamespace(get_complete_analysis=lambda **kwargs: {
        "vrp": 1.2, "realized_vol": 12.0, "vix": 13.2,
        "regime": "Normal", "vix_range": "12-20", "expected_6m_return": 4.0,
    })

    failed = updater.run_full_update()

    assert "market_breadth" in failed
    assert "vrp" in failed


def test_breadth_save_exception_degrades_phase_and_continues():
    breadth = pd.DataFrame({"date": ["2026-10-26"], "breadth_pct": [55.0]})
    db = WorkflowDB()
    db.save_breadth_data = lambda data: (_ for _ in ()).throw(RuntimeError("disk full"))
    updater = workflow_updater(db=db)
    updater.breadth = SimpleNamespace(get_breadth_history=lambda days=90: breadth)

    failed = updater.run_full_update()

    assert "market_breadth" in failed
    assert db.snapshots, "independent snapshot phase must still run"


def test_snapshot_save_exception_degrades_phase_instead_of_aborting():
    db = WorkflowDB()
    db.save_daily_snapshot = lambda data: (_ for _ in ()).throw(RuntimeError("disk full"))
    updater = workflow_updater(db=db)

    failed = updater.run_full_update()

    assert "daily_snapshot" in failed
