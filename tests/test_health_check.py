"""Unit tests for database/health_check.py.

Runs against a throwaway SQLite file with the minimal schema the checks read,
using dates relative to today so freshness math needs no clock pinning:
a row dated N days ago is between N*24 and N*24+24 hours old.
"""
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.health_check import HealthCheckSystem, HealthStatus, business_days_behind  # noqa: E402

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


def on(db_path: str, today: date) -> HealthCheckSystem:
    """A health checker whose calendar is pinned to `today`."""
    h = HealthCheckSystem(db_path)
    h._today = lambda: today
    return h


FRI = date(2026, 9, 25)  # a Friday
SAT, SUN, MON, TUE, WED, THU = (FRI + timedelta(days=i) for i in range(1, 7))


class TestBusinessDaysBehind:
    @pytest.mark.parametrize("last, today, expected", [
        (FRI, FRI, 0),
        (FRI, SAT, 0),                 # weekend counts for nothing
        (FRI, SUN, 0),
        (FRI, MON, 1),                 # Monday morning, before the refresh
        (FRI, TUE, 2),
        (FRI, WED, 3),
        (FRI - timedelta(days=1), FRI, 1),   # Thursday -> Friday
        (date(2026, 9, 21), date(2026, 9, 28), 5),   # Mon -> next Mon
        (SAT, MON, 1),                 # data dated on a weekend still counts Monday
        (MON, FRI, 0),                 # data from the future: never behind
    ])
    def test_counts_weekdays_after_last_through_today(self, last, today, expected):
        assert business_days_behind(last, today) == expected


class TestWeekendFalseAlarms:
    """The bug: Friday's snapshot read 'very stale' all weekend and every Monday
    morning under a 24-calendar-hour threshold."""

    def _vix(self, db, today):
        full_snapshot(db, FRI.isoformat())
        return on(db, today).check_data_source("VIX", "vix_spot")

    @pytest.mark.parametrize("today", [SAT, SUN, MON])
    def test_fridays_snapshot_is_healthy_through_monday_morning(self, db, today):
        assert self._vix(db, today).status is HealthStatus.HEALTHY

    def test_two_business_days_behind_is_stale(self, db):
        res = self._vix(db, TUE)
        assert res.status is HealthStatus.STALE
        assert res.message == "Data stale (2 business days behind, last: 2026-09-25)"

    def test_three_business_days_behind_is_degraded(self, db):
        res = self._vix(db, WED)
        assert res.status is HealthStatus.DEGRADED
        assert res.message.startswith("Data very stale (3 business days behind")

    def test_overall_health_is_healthy_on_a_saturday(self, db):
        full_snapshot(db, FRI.isoformat())
        insert(db, "liquidity_history", date=FRI.isoformat(), rrp_on=0.5, tga=977.0, net_liquidity=5770.0)
        assert on(db, SAT).get_overall_health() is HealthStatus.HEALTHY

    def test_calendar_age_is_still_reported(self, db):
        # Display keeps the true hours; only the status judgement changed.
        assert self._vix(db, SUN).age_hours > 24


class TestLiquidityGrace:
    """Grace follows publication lag plus one weekday for holidays."""

    @pytest.mark.parametrize("today, expected", [
        (MON, HealthStatus.HEALTHY),   # 1 behind: Friday's print is the latest
        (TUE, HealthStatus.HEALTHY),   # 2 behind: a Monday holiday
        (WED, HealthStatus.STALE),     # 3
        (THU, HealthStatus.DEGRADED),  # 4
    ])
    def test_rrp(self, db, today, expected):
        insert(db, "liquidity_history", date=FRI.isoformat(), rrp_on=0.5)
        assert on(db, today).check_data_source("Fed RRP", "rrp_on", table="liquidity_history").status is expected

    @pytest.mark.parametrize("today, expected", [
        (SUN, HealthStatus.HEALTHY),   # Wednesday's TGA is the latest on Sunday (T+2)
        (MON, HealthStatus.HEALTHY),   # 3 behind: T+2 plus a holiday
        (TUE, HealthStatus.STALE),     # 4
        (WED, HealthStatus.DEGRADED),  # 5
    ])
    def test_tga_and_net_liquidity(self, db, today, expected):
        wed = date(2026, 9, 23).isoformat()
        insert(db, "liquidity_history", date=wed, tga=900.0, net_liquidity=5800.0)
        h = on(db, today)
        assert h.check_data_source("TGA Balance", "tga", table="liquidity_history").status is expected
        assert h.check_data_source("Net Liquidity", "net_liquidity", table="liquidity_history").status is expected


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


class TestSnapshotPath:
    def test_fresh_snapshot_is_healthy(self, db):
        full_snapshot(db, days_ago(0))
        res = HealthCheckSystem(db).check_data_source("VIX", "vix_spot")
        assert res.status is HealthStatus.HEALTHY
        assert res.message.startswith("Data current")

    def test_week_old_snapshot_is_degraded_whatever_the_weekday(self, db):
        full_snapshot(db, days_ago(7))
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
