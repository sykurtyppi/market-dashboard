"""Unit tests for database/health_check.py.

Runs against a throwaway SQLite file with the minimal schema the checks read,
using dates relative to today so freshness math needs no clock pinning:
a row dated N days ago is between N*24 and N*24+24 hours old.
"""
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.health_check import HealthCheckSystem, HealthStatus  # noqa: E402

SNAPSHOT_COLS = ("vix_spot", "credit_spread_hy", "treasury_10y", "fear_greed_score", "put_call_ratio")
LIQUIDITY_COLS = ("rrp_on", "tga", "net_liquidity", "sofr", "fed_balance_sheet")


def days_ago(n: int) -> str:
    return (datetime.now() - timedelta(days=n)).strftime("%Y-%m-%d")


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "health.db"
    with sqlite3.connect(path) as con:
        con.execute(f"CREATE TABLE daily_snapshots (date TEXT, {', '.join(c + ' REAL' for c in SNAPSHOT_COLS)})")
        con.execute(f"CREATE TABLE liquidity_history (date TEXT, {', '.join(c + ' REAL' for c in LIQUIDITY_COLS)})")
    return str(path)


def insert(db_path: str, table: str, **row) -> None:
    cols = ", ".join(row)
    marks = ", ".join("?" for _ in row)
    with sqlite3.connect(db_path) as con:
        con.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", tuple(row.values()))


def full_snapshot(db_path: str, date: str) -> None:
    insert(db_path, "daily_snapshots", date=date, **{c: 1.0 for c in SNAPSHOT_COLS})


class TestLiquiditySources:
    def test_liquidity_checks_read_liquidity_history(self, db):
        # Regression: these used to query the indicators table, which never
        # holds liquidity series, and reported "No data available" on a
        # healthy system.
        insert(db, "liquidity_history", date=days_ago(0), rrp_on=0.5, tga=977.0, net_liquidity=5770.0)
        checks = HealthCheckSystem(db).get_all_health_checks()
        for key in ("liquidity_rrp", "liquidity_tga", "liquidity_net"):
            assert checks[key].status is HealthStatus.HEALTHY, checks[key]
            assert checks[key].last_update.strftime("%Y-%m-%d") == days_ago(0)

    def test_uses_latest_non_null_observation_per_column(self, db):
        # liquidity_history carries staggered nulls: today's row may have RRP
        # only, while TGA and net liquidity last landed two days ago.
        insert(db, "liquidity_history", date=days_ago(2), rrp_on=0.4, tga=900.0, net_liquidity=5800.0)
        insert(db, "liquidity_history", date=days_ago(0), rrp_on=0.5, tga=None, net_liquidity=None)
        checks = HealthCheckSystem(db).get_all_health_checks()
        assert checks["liquidity_rrp"].last_update.strftime("%Y-%m-%d") == days_ago(0)
        assert checks["liquidity_tga"].last_update.strftime("%Y-%m-%d") == days_ago(2)
        assert checks["liquidity_net"].last_update.strftime("%Y-%m-%d") == days_ago(2)
        assert all(checks[k].status is HealthStatus.HEALTHY for k in ("liquidity_tga", "liquidity_net"))

    def test_no_liquidity_rows_is_unknown(self, db):
        checks = HealthCheckSystem(db).get_all_health_checks()
        assert checks["liquidity_rrp"].status is HealthStatus.UNKNOWN
        assert checks["liquidity_rrp"].message == "No data available"


class TestLiquidityThresholds:
    """Thresholds follow publication lag: RRP 96h, TGA / net liquidity 120h.
    One row per case — the check reads the latest observation, so cases must
    not share a database."""

    @pytest.mark.parametrize("days, expected", [
        (3, HealthStatus.HEALTHY),   # Monday reading Friday's print
        (4, HealthStatus.STALE),
        (8, HealthStatus.DEGRADED),
    ])
    def test_rrp(self, db, days, expected):
        insert(db, "liquidity_history", date=days_ago(days), rrp_on=0.5)
        res = HealthCheckSystem(db).check_data_source("Fed RRP", "rrp_on", table="liquidity_history")
        assert res.status is expected

    @pytest.mark.parametrize("days, expected", [
        (4, HealthStatus.HEALTHY),   # T+2 across a weekend
        (5, HealthStatus.STALE),
        (10, HealthStatus.DEGRADED),
    ])
    def test_tga(self, db, days, expected):
        insert(db, "liquidity_history", date=days_ago(days), tga=900.0)
        res = HealthCheckSystem(db).check_data_source("TGA Balance", "tga", table="liquidity_history")
        assert res.status is expected

    def test_net_liquidity_ten_days_old_is_degraded(self, db):
        insert(db, "liquidity_history", date=days_ago(10), net_liquidity=5800.0)
        res = HealthCheckSystem(db).check_data_source("Net Liquidity", "net_liquidity", table="liquidity_history")
        assert res.status is HealthStatus.DEGRADED


class TestNameValidation:
    def test_rejects_unknown_table(self, db):
        res = HealthCheckSystem(db).check_data_source("X", "rrp_on", table="indicators")
        assert res.status is HealthStatus.UNKNOWN
        assert "Invalid table" in res.message

    def test_rejects_column_not_allowed_for_table(self, db):
        # rrp_on is valid for liquidity_history, not for daily_snapshots.
        res = HealthCheckSystem(db).check_data_source("X", "rrp_on")
        assert res.status is HealthStatus.UNKNOWN
        assert "Invalid column" in res.message

    def test_rejects_injection_shaped_column(self, db):
        res = HealthCheckSystem(db).check_data_source("X", "date; DROP TABLE daily_snapshots", table="liquidity_history")
        assert res.status is HealthStatus.UNKNOWN


class TestSnapshotPathUnchanged:
    def test_fresh_snapshot_is_healthy(self, db):
        full_snapshot(db, days_ago(0))
        res = HealthCheckSystem(db).check_data_source("VIX", "vix_spot")
        assert res.status is HealthStatus.HEALTHY
        assert res.message.startswith("Data current")

    def test_three_day_old_snapshot_is_degraded(self, db):
        full_snapshot(db, days_ago(3))
        assert HealthCheckSystem(db).check_data_source("VIX", "vix_spot").status is HealthStatus.DEGRADED


def test_overall_is_healthy_when_every_source_is_present(db):
    # Regression: with the liquidity checks stuck on UNKNOWN, a fully healthy
    # system reported overall_status "unknown".
    full_snapshot(db, days_ago(0))
    insert(db, "liquidity_history", date=days_ago(0), rrp_on=0.5, tga=977.0, net_liquidity=5770.0)
    h = HealthCheckSystem(db)
    assert h.get_overall_health() is HealthStatus.HEALTHY
    summary = h.get_health_summary()
    assert summary["overall_status"] == "healthy"
    assert summary["summary"]["unknown"] == 0
    assert summary["total_sources"] == 9


def test_missing_liquidity_table_is_unknown_not_down(tmp_path):
    # A fresh database has daily_snapshots but no liquidity_history yet. That
    # must read like "no data" — not take the whole health tile to DOWN.
    path = tmp_path / "fresh.db"
    with sqlite3.connect(path):
        pass
    with sqlite3.connect(path) as con:
        con.execute(f"CREATE TABLE daily_snapshots (date TEXT, {', '.join(c + ' REAL' for c in SNAPSHOT_COLS)})")
    h = HealthCheckSystem(str(path))
    checks = h.get_all_health_checks()
    assert checks["liquidity_rrp"].status is HealthStatus.UNKNOWN
    assert "not present" in checks["liquidity_rrp"].message
    assert h.get_overall_health() is not HealthStatus.DOWN


def test_check_indicator_is_gone():
    # It read the indicators table by names nothing writes; keeping it around
    # invites the same bug back.
    assert not hasattr(HealthCheckSystem, "check_indicator")
