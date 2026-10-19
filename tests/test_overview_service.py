"""Overview page assembly (api/overview_service.py)."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.overview_service import _credit_state, _put_call_source, build_overview  # noqa: E402

SNAPSHOT = {
    "date": "2026-10-19", "vix_spot": 18.6, "vix9d": 20.12, "vix_contango": 10.22,
    "credit_spread_hy": 3.23, "credit_spread_ig": 0.84, "fear_greed_score": 19.94,
    "market_breadth": 0.59, "put_call_ratio": 1.2589, "cboe_equity_pc": None, "spy_put_call": 1.2589,
    "vvix": 109.9, "skew": 168.59, "vrp": 8.4, "treasury_10y": 5.1, "left_signal": "NEUTRAL",
    "_status": "fresh", "_age_string": "1h ago", "_is_fresh": True,
}


def fake_db(move):
    return SimpleNamespace(
        get_latest_snapshot=lambda include_age=False: dict(SNAPSHOT),
        get_latest_vrp=lambda: {"vrp": 8.4},
        get_latest_move=lambda: move,
        get_vrp_history=lambda days=180: pd.DataFrame(),
        get_indicator_history=lambda name, days=365: pd.DataFrame(),
    )


class TestCreditBands:
    @pytest.mark.parametrize("hy, state", [(None, "neutral"), (2.6, "good"), (3.23, "good"), (3.99, "good"),
                                           (4.0, "warn"), (5.49, "warn"), (5.5, "crit"), (10.9, "crit")])
    def test_bands_follow_the_series_distribution(self, hy, state):
        assert _credit_state(hy) == state

    def test_credit_and_liquidity_page_uses_the_same_bands(self):
        # The same 3.23% read "Supportive" on the Overview and "Elevated" on
        # Credit & Liquidity; one function now serves both.
        from api import pages_service
        assert pages_service._credit_state is _credit_state
        assert not hasattr(pages_service, "_hy_state")


class TestPutCallSource:
    def test_names_the_series_actually_stored(self):
        assert _put_call_source({"cboe_equity_pc": 0.7, "spy_put_call": 1.2}) == "CBOE equity P/C (PCCE)"
        assert _put_call_source({"cboe_equity_pc": None, "spy_put_call": 1.2}) == "SPY volume proxy"
        assert _put_call_source({}) == "Best available"


class TestBuildOverview:
    def test_headline_uses_move_and_percent_credit(self):
        with patch("api.overview_service.get_db", return_value=fake_db({"move": 101.1})):
            body = build_overview()
        regime = body["regime"]
        assert regime["composite_risk"] == 32            # was 26 with credit stuck at 0 and no MOVE
        assert regime["breakdown"]["missing"] == []
        assert regime["breakdown"]["components"]["credit"] == pytest.approx(24.6)
        assert regime["breakdown"]["put_call_source"] == "SPY volume proxy"
        credit = next(c for c in regime["components"] if c["key"] == "credit")
        assert credit["state"] == "good" and credit["value"] == "No stress"

    def test_credit_cell_names_stress_not_watch_when_critical(self):
        with patch("api.overview_service.get_db", return_value=fake_db({"move": 101.1})):
            with patch.dict(SNAPSHOT, {"credit_spread_hy": 6.2}):
                body = build_overview()
        credit = next(c for c in body["regime"]["components"] if c["key"] == "credit")
        assert (credit["state"], credit["value"]) == ("crit", "Stress")

    def test_missing_move_is_declared_not_hidden(self):
        with patch("api.overview_service.get_db", return_value=fake_db({})):
            body = build_overview()
        assert body["regime"]["breakdown"]["missing"] == ["move"]
        assert body["regime"]["composite_risk"] is not None

    def test_detail_rows_are_judged_not_hardcoded(self):
        with patch("api.overview_service.get_db", return_value=fake_db({"move": 101.1})):
            detail = {r["indicator"]: r for r in build_overview()["detail"]}
        assert detail["VIX9D"]["state"] == "warn"                      # 20.12 is not "good" by the VIX bands
        assert detail["Put / Call Ratio"]["source"] == "SPY volume proxy"
