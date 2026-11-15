"""Persist actual component dates, not synthetic daily balance observations."""
import logging
import sqlite3
from types import SimpleNamespace

import pandas as pd
import pytest

from database.db_manager import DatabaseManager
from data_collectors.liquidity_collector import LiquidityCollector
from scheduler.daily_update import MarketDataUpdater


@pytest.mark.parametrize("public_csv", [False, True])
def test_tga_keeps_only_actual_weekly_observations(public_csv):
    collector = LiquidityCollector.__new__(LiquidityCollector)
    collector._use_public_csv = public_csv
    collector._fetch_series = lambda *args, **kwargs: pd.DataFrame({
        "date": pd.to_datetime(["2026-11-04", "2026-11-11"]),
        "value": [900000.0, 910000.0],
    })
    result = collector.get_tga("2026-11-01")
    assert result["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-11-04", "2026-11-11"]


def test_pipeline_derives_net_only_on_actual_shared_dates(tmp_path, caplog):
    db = DatabaseManager(str(tmp_path / "dates.db"))
    with sqlite3.connect(db.db_path) as conn:
        conn.executemany(
            "INSERT INTO fed_balance_sheet (date,total_assets) VALUES (?,?)",
            [("2026-11-04", 6700000.0), ("2026-11-11", 6750000.0)],
        )
    updater = MarketDataUpdater.__new__(MarketDataUpdater)
    updater.db = db
    updater.liquidity = SimpleNamespace(get_all_liquidity=lambda **kw: pd.DataFrame({
        "date": pd.to_datetime(["2026-11-04", "2026-11-05", "2026-11-11", "2026-11-13"]),
        "rrp_on": [1.0, 0.5, None, 0.325],
        "tga": [900000.0, None, 910000.0, None],
    }))
    updater._phase_failed = lambda *args: pytest.fail(str(args))
    with caplog.at_level(logging.INFO):
        updater._update_liquidity_history()
    with sqlite3.connect(db.db_path) as conn:
        rows = {row[0]: row[1:] for row in conn.execute(
            "SELECT date,fed_balance_sheet,net_liquidity FROM liquidity_history ORDER BY date"
        )}
    assert rows["2026-11-04"] == (6700.0, 5799.0)
    assert rows["2026-11-05"] == (None, None)
    assert rows["2026-11-11"] == (6750.0, None)
    assert rows["2026-11-13"] == (None, None)
    assert "Net Liquidity: $" not in caplog.text
    assert "net liquidity unavailable" in caplog.text.lower()
