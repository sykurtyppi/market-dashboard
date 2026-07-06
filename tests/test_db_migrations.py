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
