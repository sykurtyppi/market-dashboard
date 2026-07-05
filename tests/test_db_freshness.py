"""
Regression tests for DatabaseManager.get_latest_snapshot freshness calc.

Guards the bug where data age was computed from the `date` field (midnight of
the calendar day) instead of the actual write timestamp (`created_at`). A
snapshot written seconds ago but labelled with an older `date` must read as
fresh, not stale.
"""

import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from database.db_manager import DatabaseManager


def _insert_snapshot(db_path, date_str, created_at):
    """Insert a minimal snapshot row with a controlled created_at timestamp."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO daily_snapshots "
            "(date, treasury_10y, created_at) VALUES (?, ?, ?)",
            (date_str, 4.5, created_at.strftime("%Y-%m-%d %H:%M:%S")),
        )


@pytest.fixture
def db(tmp_path):
    return DatabaseManager(db_path=str(tmp_path / "test.db"))


def test_recent_write_is_fresh_even_when_date_is_old(db):
    # Arrange - data labelled 5 days ago but written 2 minutes ago
    now = datetime.now()
    _insert_snapshot(db.db_path, (now - timedelta(days=5)).strftime("%Y-%m-%d"),
                     created_at=now - timedelta(minutes=2))

    # Act
    snap = db.get_latest_snapshot(include_age=True)

    # Assert - freshness follows the write time, not the calendar date
    assert snap["_is_fresh"] is True
    assert snap["_status"] == "fresh"


def test_old_write_is_stale_even_when_date_is_today(db):
    # Arrange - date is today but it was written 30 hours ago
    now = datetime.now()
    _insert_snapshot(db.db_path, now.strftime("%Y-%m-%d"),
                     created_at=now - timedelta(hours=30))

    # Act
    snap = db.get_latest_snapshot(include_age=True)

    # Assert - not falsely reported fresh
    assert snap["_is_fresh"] is False
    assert snap["_status"] == "very_stale"


def test_age_hours_reflects_created_at(db):
    # Arrange
    now = datetime.now()
    _insert_snapshot(db.db_path, now.strftime("%Y-%m-%d"),
                     created_at=now - timedelta(hours=3))

    # Act
    snap = db.get_latest_snapshot(include_age=True)

    # Assert - ~3h old (allow slack for test runtime)
    assert 2.5 <= snap["_age_hours"] <= 3.5
