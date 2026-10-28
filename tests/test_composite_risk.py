"""Composite risk score (processors/composite_risk.py).

The 2026-10-19 snapshot is the worked example. The page showed 26; the same
calibrated inputs score 32.2 once credit is read as percent, MOVE is supplied,
SKEW uses its post-2020 distribution, and an uncalibrated proxy is excluded.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from processors.composite_risk import (  # noqa: E402
    COMPONENTS, PUT_CALL_WINDOW, SKEW_WINDOW,
    calculate_composite_risk_score, normalize, put_call_input,
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
    def test_skew_window_is_anchored_to_post_2020_distribution(self):
        # Yahoo ^SKEW daily history from 2020-01-02 through 2026-10-27 has
        # 1,674 readings: p5=118.74, median=140.20, p95=159.96. Rounded
        # p5/p95 anchors keep the component interpretable without fitting it
        # to the handful of observations stored in the app database.
        assert SKEW_WINDOW == (120.0, 160.0)
        assert normalize("skew", 120.0) == 0.0
        assert normalize("skew", 140.0) == pytest.approx(50.0)
        assert normalize("skew", 160.0) == 100.0

    def test_put_call_window_matches_documented_cboe_equity_thresholds(self):
        # The model configuration defines 0.6 as extreme greed/complacency and
        # 1.2 as extreme fear/hedging for CBOE's equity-only ratio.
        assert PUT_CALL_WINDOW == (0.6, 1.2)
        assert normalize("put_call", 0.6) == 100.0
        assert normalize("put_call", 0.9) == pytest.approx(50.0)
        assert normalize("put_call", 1.2) == 0.0

    def test_put_call_input_prefers_the_official_series(self):
        assert put_call_input({"cboe_equity_pc": 0.7, "spy_put_call": 1.2, "put_call_ratio": 1.2}) == (0.7, "CBOE equity P/C (PCCE)")
        assert put_call_input({"cboe_equity_pc": None, "spy_put_call": 1.2, "put_call_ratio": 1.2}) == (None, "SPY volume proxy (uncalibrated)")
        assert put_call_input({"put_call_ratio": 0.9}) == (None, "Unverified put/call source")

    @pytest.mark.parametrize(
        "snapshot, source",
        [
            ({"spy_put_call": 1.1}, "SPY volume proxy (uncalibrated)"),
            ({"put_call_ratio": 0.9}, "Unverified put/call source"),
        ],
    )
    def test_uncalibrated_source_survives_when_no_component_is_scoreable(self, snapshot, source):
        result = calculate_composite_risk_score(snapshot)
        assert result["score"] is None
        assert result["put_call_source"] == source


class TestWorkedExample:
    def test_corrected_score(self):
        r = calculate_composite_risk_score(LIVE, VRP, move_index=MOVE)
        # The SPY proxy is displayed but not scored against CBOE PCCE bounds.
        # Remaining numerator 30.55 / available weight .95 = 32.16.
        assert r["score"] == pytest.approx(32.2, abs=0.05)
        assert r["components"]["credit"] == pytest.approx(24.6)
        assert r["components"]["move"] == pytest.approx(31.1)
        assert r["put_call_source"] == "SPY volume proxy (uncalibrated)"
        assert r["missing"] == ["put_call"] and r["component_count"] == 7
        assert sum(r["weights"].values()) == pytest.approx(0.95)
        assert r["interpretation"] == "LOW RISK"

    def test_the_published_26_is_the_old_arithmetic(self):
        old, _ = old_score(LIVE, VRP)
        assert old == pytest.approx(26.4, abs=0.05)
        assert calculate_composite_risk_score(LIVE, VRP, move_index=MOVE)["score"] > old

    def test_missing_move_is_reported_and_weights_renormalize(self):
        r = calculate_composite_risk_score(LIVE, VRP)
        assert r["missing"] == ["move", "put_call"]
        assert sum(r["weights"].values()) == pytest.approx(0.85)
        assert r["score"] == pytest.approx(32.3, abs=0.05)


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
        puts = calculate_composite_risk_score({"cboe_equity_pc": 1.15})["score"]
        calls = calculate_composite_risk_score({"cboe_equity_pc": 0.65})["score"]
        assert puts < calls
