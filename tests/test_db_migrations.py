"""
Regression tests for the indicators-table integrity migration.

Older databases created the `indicators` table without
UNIQUE(indicator_name, date), so INSERT OR REPLACE had no conflict target and
appended a new row on every refresh — accumulating duplicates and letting stale
values shadow corrected ones. DatabaseManager now deduplicates and adds the
unique index on init. These tests lock that in.
"""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from database.db_manager import DatabaseManager

# Legacy schema: no UNIQUE(indicator_name, date), matching pre-migration DBs.
_LEGACY_INDICATORS = """
    CREATE TABLE indicators (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        indicator_name TEXT NOT NULL,
        series_id TEXT,
        date DATE NOT NULL,
        value REAL NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
"""

_LEGACY_REPO_MARKET = """
    CREATE TABLE repo_market (
        date TEXT PRIMARY KEY,
        sofr REAL,
        gc_repo REAL,
        rrp_on REAL,
        triparty_repo REAL,
        sofr_z_score REAL,
        stress_level TEXT,
        updated_at TEXT
    )
"""

_RRP_BILLIONS_MIGRATION = "repo_market_rrp_on_billions_v1"


def _make_legacy_db(path, rows):
    with sqlite3.connect(path) as conn:
        conn.execute(_LEGACY_INDICATORS)
        conn.executemany(
            "INSERT INTO indicators (indicator_name, series_id, date, value) VALUES (?,?,?,?)",
            rows,
        )
        conn.commit()


def _has_unique_index(path):
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_indicators_name_date'"
        ).fetchone()
    return bool(row)


def _make_old_contract_repo_db(path, rows):
    with sqlite3.connect(path) as conn:
        conn.execute(_LEGACY_REPO_MARKET)
        conn.executemany(
            "INSERT INTO repo_market (date, rrp_on) VALUES (?, ?)",
            rows,
        )
        conn.commit()


def _repo_values(path):
    with sqlite3.connect(path) as conn:
        return conn.execute(
            "SELECT date, rrp_on FROM repo_market ORDER BY date"
        ).fetchall()


def _has_migration(path, name):
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            "SELECT 1 FROM schema_migrations WHERE name = ?",
            (name,),
        ).fetchone()
    return bool(row)


def test_migration_dedupes_keeping_latest_value(tmp_path):
    db_path = str(tmp_path / "legacy.db")
    # Same (name, date) inserted twice — the corrected OAS row is inserted last
    # (higher id) and must win over the stale yield row.
    _make_legacy_db(db_path, [
        ("credit_spread_ig", "BAMLC0A0CMEY", "2026-07-02", 5.2),   # stale (lower id)
        ("credit_spread_ig", "BAMLC0A0CM", "2026-07-02", 0.75),    # corrected (higher id)
        ("credit_spread_hy", "BAMLH0A0HYM2", "2026-07-02", 2.75),
    ])

    DatabaseManager(db_path=db_path)  # __init__ runs the migration

    with sqlite3.connect(db_path) as conn:
        ig = conn.execute(
            "SELECT value FROM indicators WHERE indicator_name='credit_spread_ig' AND date='2026-07-02'"
        ).fetchall()
    assert ig == [(0.75,)], "duplicate collapsed to the most recent (corrected) value"
    assert _has_unique_index(db_path)


def test_migration_enables_insert_or_replace(tmp_path):
    db_path = str(tmp_path / "legacy.db")
    _make_legacy_db(db_path, [
        ("credit_spread_ig", "BAMLC0A0CM", "2026-07-02", 0.75),
    ])
    db = DatabaseManager(db_path=db_path)

    # After the migration the unique index exists, so a re-save replaces rather
    # than appends.
    db.save_indicator("credit_spread_ig", __import__("datetime").datetime(2026, 7, 2), 0.80)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT value FROM indicators WHERE indicator_name='credit_spread_ig' AND date='2026-07-02'"
        ).fetchall()
    assert rows == [(0.80,)], "re-save should replace the single row, not duplicate it"


def test_migration_is_idempotent_on_fresh_db(tmp_path):
    # A fresh DB is created with the constraint already; running init twice must
    # not error or drop data.
    db_path = str(tmp_path / "fresh.db")
    DatabaseManager(db_path=db_path)
    DatabaseManager(db_path=db_path)  # second init re-runs migrations
    assert _has_unique_index(db_path)


def test_daily_snapshot_credit_source_column_is_added_and_round_trips(tmp_path):
    db_path = str(tmp_path / "legacy-snapshot.db")
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE daily_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE UNIQUE NOT NULL,
                credit_spread_hy REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute(
            "INSERT INTO daily_snapshots (date, credit_spread_hy) VALUES (?, ?)",
            ("2026-07-02", 2.75),
        )

    db = DatabaseManager(db_path=db_path)
    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(daily_snapshots)")}
        old_row = conn.execute(
            "SELECT credit_spread_hy, credit_spread_hy_source FROM daily_snapshots"
        ).fetchone()

    assert "credit_spread_hy_source" in columns
    assert old_row == (2.75, None)

    fresh_db = DatabaseManager(db_path=str(tmp_path / "fresh-snapshot.db"))
    saved, validation = fresh_db.save_daily_snapshot({
        "date": "2026-07-03",
        "credit_spread_hy": 3.1,
        "credit_spread_hy_source": "FRED (BAMLH0A0HYM2)",
    })
    assert saved, validation.errors
    latest = fresh_db.get_latest_snapshot()
    assert latest is not None
    assert latest["credit_spread_hy_source"] == "FRED (BAMLH0A0HYM2)"


def test_migration_converts_existing_repo_rrp_to_billions_exactly_once(tmp_path):
    db_path = str(tmp_path / "old-contract.db")
    _make_old_contract_repo_db(
        db_path,
        [
            ("2026-10-14", 0.00003),
            ("2026-10-15", 0.105993),
            ("2026-10-16", None),
        ],
    )

    DatabaseManager(db_path=db_path)

    converted = _repo_values(db_path)
    assert converted == [
        ("2026-10-14", 0.00003 * 1000),
        ("2026-10-15", 0.105993 * 1000),
        ("2026-10-16", None),
    ]
    assert _has_migration(db_path, _RRP_BILLIONS_MIGRATION)

    DatabaseManager(db_path=db_path)

    assert _repo_values(db_path) == converted


def test_fresh_db_marks_repo_unit_migration_before_future_rows(tmp_path):
    db_path = str(tmp_path / "fresh-repo.db")
    DatabaseManager(db_path=db_path)
    assert _has_migration(db_path, _RRP_BILLIONS_MIGRATION)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO repo_market (date, rrp_on) VALUES (?, ?)",
            ("2026-10-27", 82.5),
        )
        conn.commit()

    DatabaseManager(db_path=db_path)

    assert _repo_values(db_path) == [("2026-10-27", 82.5)]

