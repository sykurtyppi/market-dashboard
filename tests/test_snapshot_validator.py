"""The supplementary put/call columns must survive validation and reach the DB.

Regression for a silent data loss: daily_snapshots gained cboe_equity_pc,
spy_put_call, spy_put_oi and spy_call_oi, the refresh populated them, and
save_daily_snapshot wrote validated.get(...) — but validate_daily_snapshot
never copied them into result.data, so every row stored NULL.
"""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db_manager import DatabaseManager  # noqa: E402
from utils.data_validator import DataValidator  # noqa: E402

SUPPLEMENTARY = {"cboe_equity_pc": 0.71, "spy_put_call": 1.13, "spy_put_oi": 1_250_000, "spy_call_oi": 980_000}
BASE = {"date": "2026-09-27", "vix_spot": 15.2, "put_call_ratio": 1.02}


def test_supplementary_put_call_fields_pass_through_validation():
    result = DataValidator().validate_daily_snapshot({**BASE, **SUPPLEMENTARY})
    assert result.is_valid, result.errors
    for key, value in SUPPLEMENTARY.items():
        assert result.data[key] == value


def test_supplementary_fields_are_persisted(tmp_path):
    db = DatabaseManager(str(tmp_path / "snap.db"))
    ok, _ = db.save_daily_snapshot({**BASE, **SUPPLEMENTARY})
    assert ok
    with sqlite3.connect(db.db_path) as con:
        row = con.execute(
            "SELECT cboe_equity_pc, spy_put_call, spy_put_oi, spy_call_oi FROM daily_snapshots"
        ).fetchone()
    assert row == (0.71, 1.13, 1_250_000, 980_000)


def test_absent_supplementary_fields_are_simply_omitted():
    result = DataValidator().validate_daily_snapshot(BASE)
    assert result.is_valid
    assert not any(k in result.data for k in SUPPLEMENTARY)
    assert result.warnings == []


def test_bad_supplementary_value_is_dropped_with_warning_not_rejected():
    # A garbage SPY OI must not take the day's VIX and credit data down with it.
    snap = {**BASE, **SUPPLEMENTARY, "spy_put_oi": -5, "spy_put_call": 99.0}
    result = DataValidator().validate_daily_snapshot(snap)
    assert result.is_valid, result.errors
    assert "spy_put_oi" not in result.data
    assert "spy_put_call" not in result.data
    assert result.data["cboe_equity_pc"] == 0.71
    assert any("spy_put_oi" in w and "dropped" in w for w in result.warnings)
    assert any("spy_put_call" in w for w in result.warnings)


def test_legacy_put_call_ratio_out_of_range_still_rejects():
    # Unchanged behaviour for the primary field.
    result = DataValidator().validate_daily_snapshot({**BASE, "put_call_ratio": 99.0})
    assert not result.is_valid
    assert any("put_call_ratio" in e for e in result.errors)


def test_spy_put_call_above_broad_market_ceiling_is_kept():
    # SPY's own P/C spikes well past the CBOE equity ratio's 3.0 ceiling on
    # stress days; those are the readings worth keeping.
    result = DataValidator().validate_daily_snapshot({**BASE, **SUPPLEMENTARY, "spy_put_call": 4.5})
    assert result.is_valid
    assert result.data["spy_put_call"] == 4.5
    assert result.warnings == []


def test_cboe_equity_pc_keeps_the_broad_market_ceiling():
    result = DataValidator().validate_daily_snapshot({**BASE, **SUPPLEMENTARY, "cboe_equity_pc": 4.5})
    assert result.is_valid
    assert "cboe_equity_pc" not in result.data
    assert any("cboe_equity_pc" in w for w in result.warnings)
