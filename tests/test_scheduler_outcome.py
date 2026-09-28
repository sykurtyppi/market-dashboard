"""The refresh must say when a phase failed.

run_full_update used to print UPDATE COMPLETE unconditionally and ignore the
return values of its own writes. Phases now record failures via
_phase_failed(); _finish() reports them and run_full_update returns them.
"""
import logging
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db_manager import DatabaseManager  # noqa: E402
from scheduler.daily_update import MarketDataUpdater  # noqa: E402


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
