"""liquidity_history must exist on a fresh database.

The refresh writes this table via save_liquidity_history, but DatabaseManager
never created it — the DDL only ever existed in one hand-built database. On a
fresh DB the INSERT failed inside a broad except, the refresh still reported
success, and every liquidity row was silently dropped.
"""
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db_manager import DatabaseManager  # noqa: E402
from database.health_check import HealthCheckSystem, HealthStatus  # noqa: E402

EXPECTED_COLUMNS = {
    "id", "date", "rrp_on", "tga", "fed_balance_sheet", "net_liquidity",
    "sofr", "sofr_spread", "treasury_10y", "created_at",
}


@pytest.fixture
def db(tmp_path):
    return DatabaseManager(str(tmp_path / "fresh.db"))


def frame(*rows):
    """Rows shaped like the scheduler's liquidity DataFrame."""
    return pd.DataFrame(
        [{"date": pd.Timestamp(d), "rrp_on": r, "tga": t, "sofr": s, "fed_bs": f} for d, r, t, s, f in rows]
    )


def test_fresh_database_has_liquidity_history_with_expected_columns(db):
    with sqlite3.connect(db.db_path) as con:
        assert con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='liquidity_history'"
        ).fetchone()
        cols = {row[1] for row in con.execute("PRAGMA table_info(liquidity_history)")}
    assert cols == EXPECTED_COLUMNS


def test_date_is_unique_so_insert_or_replace_can_dedupe(db):
    with sqlite3.connect(db.db_path) as con:
        con.execute("INSERT INTO liquidity_history (date, rrp_on) VALUES ('2026-09-27', 0.5)")
        with pytest.raises(sqlite3.IntegrityError):
            con.execute("INSERT INTO liquidity_history (date, rrp_on) VALUES ('2026-09-27', 0.6)")


def test_save_liquidity_history_persists_on_a_fresh_database(db):
    db.save_liquidity_history(frame(("2026-09-23", 0.432, 883.335, 3.64, 6740.619)))
    with sqlite3.connect(db.db_path) as con:
        row = con.execute(
            "SELECT date, rrp_on, tga, fed_balance_sheet, sofr, net_liquidity FROM liquidity_history"
        ).fetchone()
    assert row[:5] == ("2026-09-23", 0.432, 883.335, 6740.619, 3.64)
    assert row[5] == pytest.approx(6740.619 - 883.335 - 0.432)  # Fed BS - TGA - RRP


def test_saving_the_same_date_twice_keeps_one_row_with_the_latest_values(db):
    db.save_liquidity_history(frame(("2026-09-24", 0.5, 900.0, 3.88, 6748.0)))
    db.save_liquidity_history(frame(("2026-09-24", 4.7, 977.0, 3.88, 6748.0)))
    with sqlite3.connect(db.db_path) as con:
        rows = con.execute("SELECT rrp_on, tga FROM liquidity_history WHERE date='2026-09-24'").fetchall()
    assert rows == [(4.7, 977.0)]


def test_init_is_idempotent_and_keeps_existing_rows(tmp_path):
    path = str(tmp_path / "again.db")
    DatabaseManager(path).save_liquidity_history(frame(("2026-09-25", 4.736, 977.0, 3.88, 6748.0)))
    DatabaseManager(path)  # second init must not recreate or clear the table
    with sqlite3.connect(path) as con:
        assert con.execute("SELECT COUNT(*) FROM liquidity_history").fetchone() == (1,)


def test_health_check_on_fresh_db_reports_no_data_rather_than_missing_table(db):
    checks = HealthCheckSystem(db.db_path).get_all_health_checks()
    assert checks["liquidity_rrp"].status is HealthStatus.UNKNOWN
    assert checks["liquidity_rrp"].message == "No data available"
