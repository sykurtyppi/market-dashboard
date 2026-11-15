"""Net liquidity requires all three nonmissing balance-sheet components."""
import sqlite3

import pandas as pd
import pytest

from database.db_manager import DatabaseManager


@pytest.mark.parametrize("missing", ["fed_bs", "tga", "rrp_on"])
def test_missing_component_keeps_net_liquidity_unavailable(tmp_path, missing):
    db = DatabaseManager(str(tmp_path / "net.db"))
    values = {"date": "2026-11-13", "fed_bs": 6700.0, "tga": 900.0, "rrp_on": 0.325}
    values[missing] = None
    assert db.save_liquidity_history(pd.DataFrame([values]))
    with sqlite3.connect(db.db_path) as conn:
        row = conn.execute("SELECT net_liquidity FROM liquidity_history").fetchone()
    assert row == (None,)


@pytest.mark.parametrize("rrp", [0.0, 0.325])
def test_complete_components_compute_actual_net_level(tmp_path, rrp):
    db = DatabaseManager(str(tmp_path / "net.db"))
    assert db.save_liquidity_history(pd.DataFrame([{
        "date": "2026-11-13", "fed_bs": 6700.0, "tga": 900.0, "rrp_on": rrp,
    }]))
    with sqlite3.connect(db.db_path) as conn:
        value = conn.execute("SELECT net_liquidity FROM liquidity_history").fetchone()[0]
    assert value == pytest.approx(6700.0 - 900.0 - rrp)
