"""Smoke tests for the FastAPI layer (api/main.py).

Uses the DB as-is (read-only endpoints); asserts the endpoints respond with the
expected shape. No network — the overview endpoint reads cached SQLite data.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_freshness_returns_expected_shape(client):
    r = client.get("/api/freshness")
    assert r.status_code == 200
    body = r.json()
    for key in ("status", "age", "is_fresh"):
        assert key in body
    assert isinstance(body["is_fresh"], bool)


def test_overview_returns_expected_shape(client):
    r = client.get("/api/overview")
    assert r.status_code == 200
    body = r.json()
    assert set(["freshness", "regime", "metrics", "charts", "detail"]).issubset(body)
    assert isinstance(body["metrics"], list)
    assert isinstance(body["regime"]["components"], list)
    assert "vrp_history" in body["charts"]
    assert "credit_spreads" in body["charts"]


def test_overview_metrics_have_valid_states(client):
    r = client.get("/api/overview")
    body = r.json()
    valid = {"good", "warn", "crit", "neutral"}
    for m in body["metrics"]:
        assert m["state"] in valid
        assert "label" in m and "source" in m


def test_overview_chart_series_are_bounded(client):
    # Downsampling caps each series at 180 points (+ the forced last point).
    r = client.get("/api/overview")
    body = r.json()
    assert len(body["charts"]["vrp_history"]) <= 181
    assert len(body["charts"]["credit_spreads"]["hy"]) <= 181


def test_health_endpoint_responds(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert "overall_status" in r.json()


# --- Phase 1 pages ---

def test_volatility_returns_expected_shape(client):
    r = client.get("/api/volatility")
    assert r.status_code == 200
    body = r.json()
    assert {"regime_note", "metrics", "charts"}.issubset(body)
    for key in ("vrp_history", "vix", "realized_vol"):
        assert key in body["charts"]
    valid = {"good", "warn", "crit", "neutral"}
    assert all(m["state"] in valid for m in body["metrics"])


def test_breadth_returns_expected_shape(client):
    r = client.get("/api/breadth")
    assert r.status_code == 200
    body = r.json()
    assert "metrics" in body
    for key in ("ad_line", "mcclellan", "breadth_pct"):
        assert key in body["charts"]


# --- refresh (mock the updater so no live calls / DB writes happen) ---

def test_refresh_starts_and_reports_status(client):
    # TestClient runs the background task synchronously after the response, so
    # patch the updater to a no-op to avoid a real data pull.
    with patch("scheduler.daily_update.MarketDataUpdater") as MockUpdater:
        MockUpdater.return_value.run_full_update.return_value = None
        r = client.post("/api/refresh")
        assert r.status_code == 200
        assert r.json()["status"] in ("started", "already_running")
        MockUpdater.return_value.run_full_update.assert_called_once()

    status = client.get("/api/refresh/status")
    assert status.status_code == 200
    assert isinstance(status.json()["running"], bool)


def test_refresh_rejects_bad_token_when_configured(client, monkeypatch):
    monkeypatch.setenv("MARKET_API_TOKEN", "secret")
    r = client.post("/api/refresh")  # no X-API-Token header
    assert r.status_code == 401
    assert r.json()["status"] == "unauthorized"


def test_refresh_accepts_correct_token(client, monkeypatch):
    monkeypatch.setenv("MARKET_API_TOKEN", "secret")
    with patch("scheduler.daily_update.MarketDataUpdater") as MockUpdater:
        MockUpdater.return_value.run_full_update.return_value = None
        r = client.post("/api/refresh", headers={"X-API-Token": "secret"})
    assert r.status_code == 200
    assert r.json()["status"] in ("started", "already_running")


# --- Phase 2 pages ---

def test_credit_liquidity_returns_expected_shape(client):
    r = client.get("/api/credit-liquidity")
    assert r.status_code == 200
    body = r.json()
    assert "metrics" in body and "notes" in body
    for key in ("credit_spreads", "fed_assets", "qt_cumulative"):
        assert key in body["charts"]
    valid = {"good", "warn", "crit", "neutral"}
    assert all(m["state"] in valid for m in body["metrics"])


def test_sectors_returns_expected_shape(client):
    # Sectors is the one live endpoint — mock the collectors so no network is hit,
    # and clear the TTL cache so the mock is actually exercised.
    import api.sectors_service as svc
    svc._cache.clear()

    with patch("data_collectors.sector_collector.SectorCollector") as MockSC, \
         patch("data_collectors.cboe_collector.CBOECollector") as MockCBOE:
        MockSC.return_value.get_sector_performance.return_value = {
            "XLK": {"name": "Technology", "price": 180.5, "change_pct": 1.2, "category": "Cyclical"},
            "XLU": {"name": "Utilities", "price": 80.1, "change_pct": -0.4, "category": "Defensive"},
        }
        MockSC.return_value.get_rotation_signal.return_value = {
            "signal": "Risk-On", "interpretation": "Cyclicals leading",
            "leading_sectors": ["Technology (+1.2%)"],
        }
        MockCBOE.return_value.get_vix9d.return_value = 12.0
        MockCBOE.return_value.get_vix.return_value = 15.0
        MockCBOE.return_value.get_vix3m.return_value = 18.0
        r = client.get("/api/sectors")

    svc._cache.clear()
    assert r.status_code == 200
    body = r.json()
    assert len(body["sectors"]) == 2
    assert body["sectors"][0]["ticker"] == "XLK"  # sorted desc by change
    assert body["vix_structure"] == "Contango"
    assert body["rotation"]["state"] == "good"


def test_sectors_empty_data_warns_and_is_not_cached(client):
    # When sector data fails, the endpoint still returns 200 but with a warning
    # and an empty sector list — and it must NOT be cached (so recovery is fast).
    import api.sectors_service as svc
    svc._cache.clear()

    with patch("data_collectors.sector_collector.SectorCollector") as MockSC, \
         patch("data_collectors.cboe_collector.CBOECollector") as MockCBOE:
        MockSC.return_value.get_sector_performance.return_value = {}  # Yahoo failed
        MockSC.return_value.get_rotation_signal.return_value = {}
        MockCBOE.return_value.get_vix9d.return_value = 12.0
        MockCBOE.return_value.get_vix.return_value = 15.0
        MockCBOE.return_value.get_vix3m.return_value = 18.0
        r = client.get("/api/sectors")
        body = r.json()

        assert r.status_code == 200
        assert body["sectors"] == []
        assert any("unavailable" in w.lower() for w in body["warnings"])
        # VIX still served even when sectors fail
        assert len(body["vix_term"]) == 3
        # The failed fetch was not cached
        assert "sectors" not in svc._cache

    svc._cache.clear()


# --- Phase 3 pages: Treasury Stress & Repo ---

def test_treasury_returns_expected_shape(client):
    r = client.get("/api/treasury-stress")
    assert r.status_code == 200
    body = r.json()
    assert "regime_note" in body and "metrics" in body
    assert "move_history" in body["charts"]
    valid = {"good", "warn", "crit", "neutral"}
    assert all(m["state"] in valid for m in body["metrics"])


def test_repo_returns_expected_shape(client):
    r = client.get("/api/repo")
    assert r.status_code == 200
    body = r.json()
    assert "metrics" in body
    for key in ("sofr_history", "rrp_history"):
        assert key in body["charts"]


def test_treasury_uses_most_recent_row_not_oldest(client):
    # Regression: get_move_history returns newest-first (DESC). The builder must
    # report the most-recent MOVE, and the chart must run chronologically.
    from database.db_manager import DatabaseManager
    hist = DatabaseManager().get_move_history(days=365)
    if hist is None or hist.empty:
        return  # nothing to assert against
    newest_date = str(hist["date"].max())[:10]

    body = client.get("/api/treasury-stress").json()
    assert str(body["as_of"])[:10] == newest_date
    chart = body["charts"]["move_history"]
    if len(chart) >= 2:
        assert chart[0]["date"] <= chart[-1]["date"]  # ascending


def test_repo_uses_most_recent_row_not_oldest(client):
    # get_repo_history is newest-first (DESC); the builder must report the
    # most-recent SOFR and an elevated regime must not show a neutral state.
    from database.db_manager import DatabaseManager
    hist = DatabaseManager().get_repo_history(days=365)
    if hist is None or hist.empty:
        return
    newest_date = str(hist["date"].max())[:10]

    body = client.get("/api/repo").json()
    assert str(body["as_of"])[:10] == newest_date
    # The regime state must reflect the regime, not default to neutral.
    valid = {"good", "warn", "crit", "neutral"}
    assert body["state"] in valid
    if body["regime"] in ("ELEVATED", "TIGHTENING"):
        assert body["state"] == "warn"
    chart = body["charts"]["sofr_history"]
    if len(chart) >= 2:
        assert chart[0]["date"] <= chart[-1]["date"]


def test_credit_fed_assets_uses_most_recent_row_not_oldest(client):
    # get_fed_balance_sheet_history is newest-first (DESC); the fed-assets chart
    # must run chronologically and current value comes from the newest row.
    body = client.get("/api/credit-liquidity").json()
    chart = body["charts"]["fed_assets"]
    if len(chart) >= 2:
        assert chart[0]["date"] <= chart[-1]["date"]


# --- Phase 4 pages (macro, live — collectors mocked) ---

def test_fed_watch_returns_expected_shape(client):
    import api.macro_service as svc
    svc._cache.clear()
    with patch("data_collectors.fed_watch_collector.FedWatchCollector") as MockFW:
        MockFW.return_value.get_fed_watch_summary.return_value = {
            "current_rate": "3.50% - 3.75%", "current_rate_mid": 3.625, "effr": 3.63,
            "rate_source": "FRED", "rate_as_of": "2026-07-05",
            "next_meeting": {"date_str": "Jul 29, 2026", "days_until": 23},
            "most_likely": "No Change", "most_likely_prob": 98.0, "market_bias": "Neutral",
            "probabilities": {"Cut 25bp": 0.0, "No Change": 98.0, "Hike 25bp": 2.0},
            "implied_rate": 3.63, "terminal_rate": 3.5,
        }
        r = client.get("/api/fed-watch")
    svc._cache.clear()
    assert r.status_code == 200
    body = r.json()
    assert body["current_rate"] == "3.50% - 3.75%"
    assert body["next_meeting"]["days_until"] == 23
    assert len(body["probabilities"]) == 3
    assert body["most_likely"]["outcome"] == "No Change"


def test_fed_watch_empty_data_warns_and_not_cached(client):
    import api.macro_service as svc
    svc._cache.clear()
    with patch("data_collectors.fed_watch_collector.FedWatchCollector") as MockFW:
        MockFW.return_value.get_fed_watch_summary.return_value = {}
        r = client.get("/api/fed-watch")
        body = r.json()
        assert r.status_code == 200
        assert any("unavailable" in w.lower() for w in body["warnings"])
        assert "fed_watch" not in svc._cache
    svc._cache.clear()


def test_cross_asset_returns_expected_shape(client):
    import api.macro_service as svc
    svc._cache.clear()
    with patch("data_collectors.cross_asset_collector.CrossAssetCollector") as MockCA:
        MockCA.return_value.get_regime_signal.return_value = {
            "regime": "Risk-On", "color": "#4CAF50", "description": "Cyclicals bid", "confidence": 70,
        }
        MockCA.return_value.get_asset_performance_summary.return_value = {
            "SPY": {"name": "S&P 500", "price": 744.0, "change_pct": 1.0},
            "VIX": {"name": "VIX", "price": 15.0, "change_pct": -3.0},
        }
        MockCA.return_value.get_key_correlations.return_value = [
            {"pair": "Stock-Bond", "correlation": 0.42, "strength": "Moderate", "interpretation": "..."},
        ]
        r = client.get("/api/cross-asset")
    svc._cache.clear()
    assert r.status_code == 200
    body = r.json()
    assert body["regime"]["state"] == "good"  # Risk-On -> good
    assert len(body["assets"]) == 2
    assert body["assets"][0]["state"] == "good"  # SPY +1%
    assert len(body["correlations"]) == 1


def test_fed_watch_fallback_data_warns_marks_degraded_and_not_cached(client):
    # Partial fallback: current_rate exists but the collector is on fallback
    # data (rate_source=fallback, implied_rate None). Must warn, mark degraded,
    # and not be cached (so it retries when futures data returns).
    import api.macro_service as svc
    svc._cache.clear()
    with patch("data_collectors.fed_watch_collector.FedWatchCollector") as MockFW:
        MockFW.return_value.get_fed_watch_summary.return_value = {
            "current_rate": "3.50% - 3.75%", "current_rate_mid": 3.625,
            "rate_source": "fallback", "implied_rate": None,
            "next_meeting": {"date_str": "Jul 29, 2026", "days_until": 23},
            "most_likely": "No Change", "most_likely_prob": 60.0,
            "probabilities": {"No Change": 60.0, "Cut 25bp": 15.0, "Hike 25bp": 15.0},
        }
        r = client.get("/api/fed-watch")
        body = r.json()
        assert r.status_code == 200
        assert body["degraded"] is True
        assert any("fallback" in w.lower() for w in body["warnings"])
        assert "fed_watch" not in svc._cache  # degraded response not cached
    svc._cache.clear()


# --- Phase 5 pages (positioning & flows, live — collectors mocked) ---

def test_cot_returns_expected_shape(client):
    import api.flows_service as svc
    svc._cache.clear()
    with patch("data_collectors.cot_collector.COTCollector") as MockCOT:
        MockCOT.return_value.get_positioning_summary.return_value = {
            "timestamp": "2026-07-05T00:00:00",
            "positions": {
                "ES": {"name": "S&P 500 E-mini", "category": "equity", "date": "2026-06-23",
                       "spec_net": -35448, "spec_net_change": 158530, "comm_net": -87130,
                       "open_interest": 1980254},
            },
        }
        r = client.get("/api/cot")
    svc._cache.clear()
    assert r.status_code == 200
    body = r.json()
    assert len(body["positions"]) == 1
    assert body["positions"][0]["symbol"] == "ES"
    assert body["positions"][0]["spec_net"] == -35448


def test_cot_empty_data_warns_and_not_cached(client):
    import api.flows_service as svc
    svc._cache.clear()
    with patch("data_collectors.cot_collector.COTCollector") as MockCOT:
        MockCOT.return_value.get_positioning_summary.return_value = {}
        r = client.get("/api/cot")
        body = r.json()
        assert r.status_code == 200
        assert body["positions"] == []
        assert any("unavailable" in w.lower() for w in body["warnings"])
        assert "cot" not in svc._cache
    svc._cache.clear()


def _options_summary(**overrides):
    base = {
        "timestamp": "2026-07-05T00:00:00",
        "spy": {"ticker": "SPY", "current_price": 744.0, "expiry": "2026-07-06", "dte": 1,
                "total_call_volume": 900000, "total_put_volume": 850000,
                "put_call_ratio": 0.94, "sentiment": "BULLISH", "status": "ok"},
        "qqq": {"ticker": "QQQ", "current_price": 712.0, "dte": 1, "put_call_ratio": 0.93,
                "sentiment": "NEUTRAL", "status": "ok"},
        "iwm": {"ticker": "IWM", "current_price": 297.0, "dte": 1, "put_call_ratio": 0.80,
                "sentiment": "NEUTRAL", "status": "ok"},
    }
    base.update(overrides)
    return base


def test_options_flow_returns_expected_shape(client):
    import api.flows_service as svc
    svc._cache.clear()
    with patch("data_collectors.options_flow_collector.OptionsFlowCollector") as MockOF:
        MockOF.return_value.get_market_options_summary.return_value = _options_summary()
        r = client.get("/api/options-flow")
    svc._cache.clear()
    assert r.status_code == 200
    body = r.json()
    assert len(body["etfs"]) == 3  # all present
    assert body["etfs"][0]["ticker"] == "SPY"
    assert body["etfs"][0]["state"] == "good"  # BULLISH -> good
    assert body["warnings"] == []  # complete fetch, no warning


def test_options_flow_partial_data_warns_and_not_cached(client):
    # Some ETFs fail (the important SPY/QQQ), one succeeds. The user must be told
    # which are missing, and a partial result must not be cached.
    import api.flows_service as svc
    svc._cache.clear()
    with patch("data_collectors.options_flow_collector.OptionsFlowCollector") as MockOF:
        MockOF.return_value.get_market_options_summary.return_value = _options_summary(
            spy={"status": "error"}, qqq={"status": "error"},
        )
        r = client.get("/api/options-flow")
        body = r.json()
        assert r.status_code == 200
        assert [e["ticker"] for e in body["etfs"]] == ["IWM"]
        assert any("partially unavailable" in w.lower() for w in body["warnings"])
        assert "SPY" in body["warnings"][0] and "QQQ" in body["warnings"][0]
        assert "options_flow" not in svc._cache  # partial not cached
    svc._cache.clear()


def test_options_flow_empty_data_warns_and_not_cached(client):
    import api.flows_service as svc
    svc._cache.clear()
    with patch("data_collectors.options_flow_collector.OptionsFlowCollector") as MockOF:
        MockOF.return_value.get_market_options_summary.return_value = {}
        r = client.get("/api/options-flow")
        body = r.json()
        assert r.status_code == 200
        assert body["etfs"] == []
        assert any("unavailable" in w.lower() for w in body["warnings"])
        assert "options_flow" not in svc._cache
    svc._cache.clear()


# --- Phase 6 pages (Institutional Flow, Economic Calendar — collectors mocked) ---

def test_institutional_returns_expected_shape(client):
    import api.flows_service as svc
    svc._cache.clear()
    with patch("data_collectors.dark_pool_collector.DarkPoolCollector") as MockDP, \
         patch("data_collectors.insider_trading_collector.InsiderTradingCollector") as MockIns, \
         patch("data_collectors.treasury_auction_collector.TreasuryAuctionCollector") as MockAu:
        # Realistic payloads: state comes from the machine-readable `signal`,
        # not the descriptive `sentiment` prose.
        MockDP.return_value.get_dark_pool_summary.return_value = {
            "avg_dark_pool_pct": 38.0, "etf_avg_pct": 40.0, "stock_avg_pct": 36.0,
            "signal": "NORMAL", "sentiment": "Normal Activity",
            "interpretation": "...", "last_updated": "2026-07-05"}
        MockIns.return_value.get_insider_summary.return_value = {
            "total_transactions": 30, "buy_count": 5, "sell_count": 25,
            "buy_sell_ratio": 0.2, "signal": "BEARISH",
            "sentiment": "More selling than buying - Some profit taking", "period_days": 30}
        MockAu.return_value.get_auction_summary.return_value = {
            "avg_bid_to_cover": 2.48, "avg_indirect_pct": 66.9, "avg_direct_pct": 18.8,
            "auction_count": 3, "weak_auctions": 0, "strong_auctions": 1, "health": "Strong Demand"}
        r = client.get("/api/institutional")
    svc._cache.clear()
    assert r.status_code == 200
    body = r.json()
    assert body["warnings"] == []
    assert body["dark_pool"]["avg_pct"] == 38.0
    assert body["dark_pool"]["state"] == "neutral"  # NORMAL signal
    # BEARISH signal despite prose sentiment -> crit (the bug this guards)
    assert body["insider"]["state"] == "crit"
    assert body["auctions"]["state"] == "good"  # Strong -> good


def test_institutional_signal_state_mapping():
    # The mapper must classify the real collector signal vocabulary, including
    # insider "CAUTIOUS" -> warn (not neutral).
    from api.flows_service import _signal_state
    assert _signal_state("BEARISH") == "crit"
    assert _signal_state("BULLISH") == "good"
    assert _signal_state("CAUTIOUS") == "warn"
    assert _signal_state("ELEVATED") == "warn"
    assert _signal_state("NORMAL") == "neutral"
    assert _signal_state("NEUTRAL") == "neutral"
    # Falls back to prose sentiment only when no signal is present.
    assert _signal_state(None, "BULLISH") == "good"


def test_fomc_2026_dates_match_official_calendar():
    # Regression: 2026 FOMC decision days must match the official Fed calendar
    # (Oct 27-28 and Dec 8-9, not the earlier wrong Nov 4 / Dec 16).
    from datetime import datetime
    from data_collectors.economic_calendar_collector import FOMC_DATES_2026
    assert datetime(2026, 9, 16) in FOMC_DATES_2026
    assert datetime(2026, 10, 28) in FOMC_DATES_2026
    assert datetime(2026, 12, 9) in FOMC_DATES_2026
    assert datetime(2026, 11, 4) not in FOMC_DATES_2026
    assert datetime(2026, 12, 16) not in FOMC_DATES_2026


def test_institutional_partial_warns_and_not_cached(client):
    import api.flows_service as svc
    svc._cache.clear()
    with patch("data_collectors.dark_pool_collector.DarkPoolCollector") as MockDP, \
         patch("data_collectors.insider_trading_collector.InsiderTradingCollector") as MockIns, \
         patch("data_collectors.treasury_auction_collector.TreasuryAuctionCollector") as MockAu:
        MockDP.return_value.get_dark_pool_summary.return_value = {"avg_dark_pool_pct": 38.0}
        MockIns.return_value.get_insider_summary.return_value = {}  # missing
        MockAu.return_value.get_auction_summary.return_value = {}   # missing
        r = client.get("/api/institutional")
        body = r.json()
        assert r.status_code == 200
        assert body["insider"] is None and body["auctions"] is None
        assert any("partially unavailable" in w.lower() for w in body["warnings"])
        assert "institutional" not in svc._cache
    svc._cache.clear()


def test_economic_calendar_returns_expected_shape(client):
    import api.macro_service as svc
    svc._cache.clear()
    with patch("data_collectors.economic_calendar_collector.EconomicCalendarCollector") as MockEC:
        MockEC.return_value.get_upcoming_events.return_value = [
            {"name": "CPI (Inflation)", "date": "2026-07-13", "importance": "high",
             "category": "Inflation", "actual": 333.9, "forecast": None, "previous": 332.4,
             "yoy_change": 4.27, "unit": "% YoY"},
        ]
        r = client.get("/api/economic-calendar")
    svc._cache.clear()
    assert r.status_code == 200
    body = r.json()
    assert len(body["events"]) == 1
    assert body["events"][0]["name"] == "CPI (Inflation)"
    assert body["events"][0]["importance"] == "high"


def test_economic_calendar_empty_warns_and_not_cached(client):
    import api.macro_service as svc
    svc._cache.clear()
    with patch("data_collectors.economic_calendar_collector.EconomicCalendarCollector") as MockEC:
        MockEC.return_value.get_upcoming_events.return_value = []
        r = client.get("/api/economic-calendar")
        body = r.json()
        assert r.status_code == 200
        assert body["events"] == []
        assert any("unavailable" in w.lower() for w in body["warnings"])
        assert "economic_calendar" not in svc._cache
    svc._cache.clear()


# --- Final pages: Sentiment, LEFT, CTA (collectors mocked) ---

def test_sentiment_returns_expected_shape(client):
    import api.signals_service as svc
    svc._cache.clear()
    with patch("data_collectors.fear_greed_collector.FearGreedCollector") as MockFG:
        MockFG.return_value.get_fear_greed_score.return_value = {
            "score": 34.0, "rating": "Fear", "timestamp": "2026-07-06T00:00:00"}
        r = client.get("/api/sentiment")
    svc._cache.clear()
    assert r.status_code == 200
    body = r.json()
    assert body["fear_greed"]["score"] == 34.0
    assert body["fear_greed"]["state"] == "warn"  # fear -> warn


def test_left_returns_expected_shape(client):
    import api.signals_service as svc
    import pandas as pd
    svc._cache.clear()
    hyg = pd.DataFrame({"date": pd.bdate_range("2023-01-02", periods=5),
                        "BAMLH0A0HYM2": [3.0, 2.9, 2.8, 2.75, 2.75]})
    with patch("data_collectors.fred_collector.FREDCollector") as MockFRED, \
         patch("processors.left_strategy.LEFTStrategy") as MockLEFT:
        MockFRED.return_value.get_series.return_value = hyg
        MockLEFT.return_value.calculate_signal.return_value = {
            "signal": "BUY", "strength": 60.0, "current_spread": 2.75,
            "ema_330": 2.95, "pct_from_ema": -6.8, "date": "2026-07-02"}
        MockLEFT.return_value.get_historical_signals.return_value = pd.DataFrame(
            {"date": pd.bdate_range("2023-01-02", periods=3), "spread": [3.0, 2.9, 2.8],
             "ema_330": [3.1, 3.05, 3.0]})
        r = client.get("/api/left")
    svc._cache.clear()
    assert r.status_code == 200
    body = r.json()
    assert body["signal"] == "BUY"
    assert body["state"] == "good"  # BUY -> good
    assert len(body["charts"]["spread"]) == 3


def test_cta_returns_expected_shape(client):
    import api.signals_service as svc
    import pandas as pd
    from types import SimpleNamespace
    svc._cache.clear()
    result = SimpleNamespace(
        latest_state={"SPY": "LONG", "GLD": "SHORT", "EEM": "FLAT"},
        latest_exposure=pd.Series({"SPY": 0.8, "GLD": -0.5, "EEM": 0.0}),
    )
    with patch("data_collectors.cta_collector_cloud.CTACollectorCloud") as MockCTA:
        MockCTA.return_value.get_cta_analysis.return_value = result
        r = client.get("/api/cta")
    svc._cache.clear()
    assert r.status_code == 200
    body = r.json()
    assert body["long_count"] == 1 and body["short_count"] == 1 and body["flat_count"] == 1
    spy = next(p for p in body["positions"] if p["symbol"] == "SPY")
    assert spy["state"] == "good" and spy["exposure"] == 0.8


def test_cta_empty_warns_and_not_cached(client):
    import api.signals_service as svc
    from types import SimpleNamespace
    svc._cache.clear()
    with patch("data_collectors.cta_collector_cloud.CTACollectorCloud") as MockCTA:
        MockCTA.return_value.get_cta_analysis.return_value = SimpleNamespace(latest_state={}, latest_exposure=None)
        r = client.get("/api/cta")
        body = r.json()
        assert body["positions"] == []
        assert any("unavailable" in w.lower() for w in body["warnings"])
        assert "cta" not in svc._cache
    svc._cache.clear()
