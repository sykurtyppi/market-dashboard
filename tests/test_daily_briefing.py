"""Deterministic Daily Risk Briefing rules and persistence."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.daily_briefing_service import build_briefing
from database.db_manager import DatabaseManager


def snapshot(date: str, **overrides):
    base = {
        "date": date,
        "vix_spot": 18.0,
        "credit_spread_hy": 3.0,
        "fear_greed_score": 45.0,
        "market_breadth": 0.62,
        "skew": 130.0,
        "cboe_equity_pc": 0.9,
        "_status": "fresh",
        "_is_fresh": True,
    }
    base.update(overrides)
    return base


def test_briefing_assigns_transparent_posture_and_ranks_material_drivers():
    previous = snapshot("2026-10-28", vix_spot=16.0, market_breadth=0.72)
    current = snapshot("2026-10-29", vix_spot=30.0, market_breadth=0.42)

    body = build_briefing(
        current,
        previous,
        current_vrp={"vrp": 4.0},
        previous_vrp={"vrp": 7.0},
        current_move=125.0,
        previous_move=90.0,
    )

    assert body["posture"] in {"normal", "cautious", "defensive", "stress"}
    assert body["rules"]["posture_thresholds"] == {
        "normal": "score < 40",
        "cautious": "40 <= score < 60",
        "defensive": "60 <= score < 75",
        "stress": "score >= 75",
    }
    assert body["change"]["current_score"] > body["change"]["previous_score"]
    assert [driver["key"] for driver in body["drivers"][:3]] == ["breadth", "vix", "move"]
    assert all(driver["evidence_href"].startswith("/") for driver in body["drivers"])
    assert body["confirming"]
    assert body["why_it_matters"]


def test_unchanged_components_are_neutral_not_drivers():
    body = build_briefing(
        snapshot("2026-10-29"),
        snapshot("2026-10-28"),
        current_vrp={"vrp": 5.0},
        previous_vrp={"vrp": 5.0},
        current_move=90.0,
        previous_move=90.0,
    )
    assert body["drivers"] == []
    assert body["confirming"] == []
    assert body["conflicting"] == []


def test_first_briefing_states_that_no_comparison_baseline_exists():
    body = build_briefing(
        snapshot("2026-10-29"),
        None,
        current_vrp={"vrp": 5.0},
        current_move=90.0,
    )

    assert body["change"]["previous_score"] is None
    assert body["why_it_matters"] == "No prior briefing is available for comparison."


def test_missing_current_model_data_does_not_emit_market_regime_alert():
    body = build_briefing(
        {"date": "2026-10-29", "_status": "missing", "_is_fresh": False},
        snapshot("2026-10-28"),
        previous_vrp={"vrp": 5.0},
        previous_move=90.0,
    )
    trigger_types = {item["type"] for item in body["alert"]["triggers"]}
    assert body["posture"] == "unknown"
    assert body["change"]["regime_changed"] is False
    assert "posture_change" not in trigger_types
    assert "unusual_score_move" not in trigger_types
    assert "data_quality" in trigger_types


def test_partial_input_loss_does_not_emit_false_market_alert():
    previous = snapshot("2026-10-28", credit_spread_hy=8.0)
    current = snapshot("2026-10-29")
    current.pop("credit_spread_hy")

    body = build_briefing(
        current,
        previous,
        current_vrp={"vrp": 5.0},
        previous_vrp={"vrp": 5.0},
        current_move=90.0,
        previous_move=90.0,
    )

    trigger_types = {item["type"] for item in body["alert"]["triggers"]}
    assert body["change"]["inputs_comparable"] is False
    assert body["change"]["regime_changed"] is False
    assert "posture_change" not in trigger_types
    assert "unusual_score_move" not in trigger_types
    assert "data_quality" in trigger_types


def test_driver_attribution_includes_input_availability_changes():
    previous = snapshot("2026-10-28")
    previous.pop("cboe_equity_pc")
    current = snapshot("2026-10-29")

    body = build_briefing(
        current,
        previous,
        current_vrp={"vrp": 5.0},
        previous_vrp={"vrp": 5.0},
        current_move=90.0,
        previous_move=90.0,
    )

    put_call = next(driver for driver in body["all_driver_changes"] if driver["key"] == "put_call")
    assert put_call["availability_change"] == "added"
    attributed = sum(driver["risk_delta"] for driver in body["all_driver_changes"])
    assert round(attributed + body["change"]["attribution_residual"], 1) == body["change"]["score_delta"]


def test_alerts_only_for_material_triggers():
    stable = build_briefing(
        snapshot("2026-10-29"),
        snapshot("2026-10-28"),
        current_vrp={"vrp": 5.0},
        previous_vrp={"vrp": 5.0},
        current_move=90.0,
        previous_move=90.0,
    )
    assert stable["alert"] == {"should_alert": False, "triggers": []}

    changed = build_briefing(
        snapshot("2026-10-29", vix_spot=55.0, market_breadth=0.15, _status="stale", _is_fresh=False),
        snapshot("2026-10-28", vix_spot=16.0, market_breadth=0.75),
        current_vrp={"vrp": -8.0},
        previous_vrp={"vrp": 8.0},
        current_move=165.0,
        previous_move=80.0,
    )
    trigger_types = {item["type"] for item in changed["alert"]["triggers"]}
    assert changed["alert"]["should_alert"] is True
    assert {"posture_change", "unusual_score_move", "data_quality"}.issubset(trigger_types)


def test_briefing_persistence_is_idempotent_and_outcomes_backfill(tmp_path):
    db = DatabaseManager(str(tmp_path / "briefing.db"))
    payload = {
        "as_of": "2026-10-20",
        "posture": "cautious",
        "change": {"current_score": 52.0},
        "outcomes": {"benchmark": "SPY", "1d": None, "5d": None, "20d": None},
    }

    db.save_daily_briefing(payload)
    db.save_daily_briefing({**payload, "posture": "defensive"})
    latest = db.get_latest_daily_briefing()
    assert latest["posture"] == "defensive"
    assert len(db.get_daily_briefing_history()) == 1

    prices = [
        ("2026-10-20", 100.0),
        ("2026-10-21", 101.0),
        ("2026-10-22", 102.0),
        ("2026-10-23", 103.0),
        ("2026-10-26", 104.0),
        ("2026-10-27", 105.0),
    ]
    db.save_benchmark_prices("SPY", prices)
    updated = db.backfill_daily_briefing_outcomes("SPY")
    assert updated == 1
    latest = db.get_latest_daily_briefing()
    assert latest["outcomes"]["1d"] == pytest.approx(1.0)
    assert latest["outcomes"]["5d"] == pytest.approx(5.0)
    assert latest["outcomes"]["20d"] is None
    assert latest["outcomes"]["anchor_date"] == "2026-10-20"
    assert latest["outcomes"]["anchor_close"] == pytest.approx(100.0)

    # Once accountability starts, later corrections cannot silently move the anchor.
    db.save_benchmark_prices("SPY", [("2026-10-20", 110.0)])
    db.backfill_daily_briefing_outcomes("SPY")
    latest = db.get_latest_daily_briefing()
    assert latest["outcomes"]["anchor_close"] == pytest.approx(100.0)
    assert latest["outcomes"]["1d"] == pytest.approx(1.0)

    db.save_daily_briefing({**payload, "posture": "stress"})
    regenerated = db.get_latest_daily_briefing()
    assert regenerated["posture"] == "stress"
    assert regenerated["outcomes"]["anchor_date"] == "2026-10-20"
    assert regenerated["outcomes"]["anchor_close"] == pytest.approx(100.0)
    assert regenerated["outcomes"]["1d"] == pytest.approx(1.0)


def test_outcomes_wait_for_exact_briefing_date_anchor(tmp_path):
    db = DatabaseManager(str(tmp_path / "briefing.db"))
    db.save_daily_briefing({
        "as_of": "2026-10-20",
        "posture": "normal",
        "outcomes": {"benchmark": "SPY", "1d": None, "5d": None, "20d": None},
    })
    db.save_benchmark_prices("SPY", [
        ("2026-10-19", 99.0),
        ("2026-10-21", 101.0),
    ])

    assert db.backfill_daily_briefing_outcomes("SPY") == 1
    outcomes = db.get_latest_daily_briefing()["outcomes"]
    assert outcomes.get("anchor_date") is None
    assert outcomes["1d"] is None
    assert outcomes["anchor_status"] == "no_session"


def test_intraday_candidate_never_becomes_immutable_anchor(tmp_path, monkeypatch):
    from datetime import date
    from types import SimpleNamespace

    import pandas as pd
    import data_collectors.yahoo_collector as yahoo_module
    from data_collectors.yahoo_collector import YahooCollector

    db = DatabaseManager(str(tmp_path / "briefing.db"))
    db.save_daily_briefing({
        "as_of": "2026-10-29",
        "posture": "normal",
        "outcomes": {"benchmark": "SPY", "1d": None, "5d": None, "20d": None},
    })
    session = pd.Timestamp("2026-10-29", tz="America/New_York")
    ticker = SimpleNamespace(
        history=lambda period="3mo": pd.DataFrame({"Close": [100.0]}, index=[session])
    )
    monkeypatch.setattr(yahoo_module.yf, "Ticker", lambda symbol: ticker)
    collector = YahooCollector()

    db.save_benchmark_prices(
        "SPY", collector.get_benchmark_prices("SPY", completed_before=date(2026, 10, 29))
    )
    assert db.backfill_daily_briefing_outcomes("SPY") == 0
    assert db.get_latest_daily_briefing()["outcomes"].get("anchor_close") is None

    ticker.history = lambda period="3mo": pd.DataFrame({"Close": [105.0]}, index=[session])
    db.save_benchmark_prices(
        "SPY", collector.get_benchmark_prices("SPY", completed_before=date(2026, 10, 30))
    )
    assert db.backfill_daily_briefing_outcomes("SPY") == 1
    outcomes = db.get_latest_daily_briefing()["outcomes"]
    assert outcomes["anchor_status"] == "finalized"
    assert outcomes["anchor_close"] == pytest.approx(105.0)
