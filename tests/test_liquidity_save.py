"""save_liquidity_history must report failure instead of swallowing it.

A catch-all except logged every error — including "no such table" — and
returned None, so the refresh went on to log net liquidity and report
UPDATE COMPLETE while every row was dropped. Now: True on write, False on a
database failure (logged), and programming errors propagate.
"""
import logging
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db_manager import DatabaseManager  # noqa: E402


def frame(*rows):
    return pd.DataFrame(
        [{"date": pd.Timestamp(d), "rrp_on": r, "tga": t, "sofr": s, "fed_bs": f} for d, r, t, s, f in rows]
    )


@pytest.fixture
def db(tmp_path):
    return DatabaseManager(str(tmp_path / "liq.db"))


def test_successful_write_returns_true(db):
    assert db.save_liquidity_history(frame(("2026-09-25", 0.5, 977.0, 3.88, 6748.0))) is True


def test_empty_frame_returns_false_and_warns(db, caplog):
    with caplog.at_level(logging.WARNING):
        assert db.save_liquidity_history(pd.DataFrame()) is False
    assert "Empty liquidity DataFrame" in caplog.text


def test_database_failure_returns_false_and_logs_the_cause(db, caplog):
    with sqlite3.connect(db.db_path) as con:
        con.execute("DROP TABLE liquidity_history")
    with caplog.at_level(logging.ERROR):
        assert db.save_liquidity_history(frame(("2026-09-25", 0.5, 977.0, 3.88, 6748.0))) is False
    assert "Error saving liquidity history" in caplog.text
    assert "no such table" in caplog.text


def test_malformed_frame_is_a_bug_and_propagates(db):
    # A frame without a date column is a caller bug, not an operational
    # condition to log and move past.
    with pytest.raises(KeyError):
        db.save_liquidity_history(pd.DataFrame([{"rrp_on": 0.5, "tga": 977.0}]))
