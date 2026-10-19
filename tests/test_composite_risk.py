"""Composite risk score (processors/composite_risk.py).

The 2026-10-19 snapshot is the worked example. The page showed 26; the same
inputs score 31.5 once credit is read as percent, MOVE is supplied, and SKEW
is windowed for the levels it actually prints.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from processors.composite_risk import (  # noqa: E402
    COMPONENTS, PUT_CALL_WINDOW, calculate_composite_risk_score, normalize, put_call_input,
)

LIVE = {
    "vix_spot": 18.6, "credit_spread_hy": 3.23, "fear_greed_score": 19.94,
    "market_breadth": 0.59, "skew": 168.59,
    "put_call_ratio": 1.2589, "cboe_equity_pc": None, "spy_put_call": 1.2589,
}
VRP = {"vrp": 8.4}
MOVE = 101.1


def old_score(snapshot, vrp):
    """The replaced implementation's arithmetic, kept here as the record of
    why the page said 26: basis-point credit formula on a percent input,
    SKEW on 110–150, one put/call window, and no MOVE."""
    c = {
        "vix": min(100, max(0, (snapshot["vix_spot"] - 10) / 50 * 100)),
        "credit": min(100, max(0, (snapshot["credit_spread_hy"] - 200) / 500 * 100)),
        "sentiment": snapshot["fear_greed_score"],
        "vrp": min(100, max(0, (10 - vrp["vrp"]) / 20 * 100)),
        "breadth": (1 - snapshot["market_breadth"]) * 100,
        "skew": min(100, max(0, (snapshot["skew"] - 110) / 40 * 100)),
        "put_call": min(100, max(0, (1.5 - snapshot["put_call_ratio"]) / 1.0 * 100)),
    }
    w = {"vix": .15, "credit": .20, "sentiment": .15, "vrp": .10, "breadth": .15, "skew": .10, "put_call": .05}
    return sum(c[k] * w[k] for k in c) / sum(w.values()), c


class TestCreditUnits:
    def test_scored_from_percent(self):
        assert normalize("credit", 3.23) == pytest.approx(24.6)
        assert normalize("credit", 2.0) == 0.0
        assert normalize("credit", 7.0) == 100.0
        assert normalize("credit", 10.9) == 100.0   # the 2020 peak saturates, not overflows

    def test_the_old_formula_zeroed_every_real_spread(self):
        _, old = old_score(LIVE, VRP)
        assert old["credit"] == 0.0
        assert normalize("credit", 7.0) > 0


class TestWindows:
    def test_skew_window_for_the_post_2020_regime(self):
        # Locally SKEW prints 137–169 (median ~146). On 110–150 a typical day
        # scored ~89 and a third of days saturated; on 120–170 it does not.
        assert normalize("skew", 120.0) == 0.0
        assert normalize("skew", 145.5) == pytest.approx(51.0)
        assert normalize("skew", 170.0) == 100.0

    def test_put_call_uses_one_window_for_every_source(self):
        # A per-source SPY window was tried and reverted: only three SPY
        # readings are stored, and the app's own copy calls >1.0 defensive.
        assert PUT_CALL_WINDOW == (0.5, 1.5)
        assert normalize("put_call", 0.6) == pytest.approx(90.0)
        assert normalize("put_call", 1.2589) == pytest.approx(24.1, abs=0.05)
        assert normalize("put_call", 1.5) == 0.0

    def test_put_call_input_prefers_the_official_series(self):
        assert put_call_input({"cboe_equity_pc": 0.7, "spy_put_call": 1.2, "put_call_ratio": 1.2}) == (0.7, "CBOE equity P/C (PCCE)")
        assert put_call_input({"cboe_equity_pc": None, "spy_put_call": 1.2, "put_call_ratio": 1.2}) == (1.2, "SPY volume proxy")
        assert put_call_input({"put_call_ratio": 0.9}) == (0.9, "Best available")


class TestWorkedExample:
    def test_corrected_score(self):
        r = calculate_composite_risk_score(LIVE, VRP, move_index=MOVE)
        # 17.2·.15 + 24.6·.20 + 19.94·.15 + 8·.10 + 41·.15 + 31.1·.10 + 97.18·.10 + 24.11·.05 = 31.47
        assert r["score"] == pytest.approx(31.5, abs=0.05)
        assert r["components"]["credit"] == pytest.approx(24.6)
        assert r["components"]["move"] == pytest.approx(31.1)
        assert r["put_call_source"] == "SPY volume proxy"
        assert r["missing"] == [] and r["component_count"] == 8
        assert sum(r["weights"].values()) == pytest.approx(1.0)
        assert r["interpretation"] == "LOW RISK"

    def test_the_published_26_is_the_old_arithmetic(self):
        old, _ = old_score(LIVE, VRP)
        assert old == pytest.approx(26.4, abs=0.05)
        assert calculate_composite_risk_score(LIVE, VRP, move_index=MOVE)["score"] > old

    def test_missing_move_is_reported_and_weights_renormalize(self):
        r = calculate_composite_risk_score(LIVE, VRP)
        assert r["missing"] == ["move"]
        assert sum(r["weights"].values()) == pytest.approx(0.90)
        assert r["score"] == pytest.approx(31.5, abs=0.05)   # (31.47 − 3.11) / 0.9


class TestShape:
    def test_empty_snapshot_scores_none(self):
        r = calculate_composite_risk_score({})
        assert r["score"] is None and r["missing"] == list(COMPONENTS) and r["put_call_source"] is None

    def test_single_input_scores_its_own_normalized_value(self):
        assert calculate_composite_risk_score({"vix_spot": 35.0})["score"] == pytest.approx(50.0)

    def test_unusable_values_are_ignored_not_fatal(self):
        r = calculate_composite_risk_score({"vix_spot": float("nan"), "skew": "abc", "credit_spread_hy": 3.0, "market_breadth": True})
        assert list(r["components"]) == ["credit"]

    def test_score_is_not_built_from_rounded_components(self):
        # 0.33 → 33.33… normalized; rounding first would give 33.3·0.15 ≠ 33.33·0.15
        r = calculate_composite_risk_score({"vix_spot": 26.6667})
        assert r["score"] == pytest.approx(33.3, abs=0.05)

    @pytest.mark.parametrize("vix, band", [(10, "VERY BULLISH"), (22.5, "LOW RISK"), (30, "NEUTRAL"), (40, "ELEVATED"), (47.5, "HIGH RISK")])
    def test_bands(self, vix, band):
        assert calculate_composite_risk_score({"vix_spot": vix})["interpretation"] == band

    def test_contrarian_inputs_lower_the_score(self):
        fear = calculate_composite_risk_score({"fear_greed_score": 10})["score"]
        greed = calculate_composite_risk_score({"fear_greed_score": 90})["score"]
        assert fear < greed
        puts = calculate_composite_risk_score({"put_call_ratio": 1.4})["score"]
        calls = calculate_composite_risk_score({"put_call_ratio": 0.6})["score"]
        assert puts < calls

    def test_legacy_ui_import_is_the_same_function(self):
        from dashboard.ui_helpers import calculate_composite_risk_score as legacy
        assert legacy is calculate_composite_risk_score
