"""Composite risk score (processors/composite_risk.py).

The 2026-10-19 snapshot is the worked example: the page showed 26. With the
credit term scored from a percent input instead of a basis-point formula,
and MOVE actually supplied, the same inputs score 31.8.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from processors.composite_risk import COMPONENTS, calculate_composite_risk_score, normalize  # noqa: E402

LIVE = {
    "vix_spot": 18.6, "credit_spread_hy": 3.23, "fear_greed_score": 19.94,
    "market_breadth": 0.59, "skew": 168.59, "put_call_ratio": 1.2589,
}
VRP = {"vrp": 8.4}
MOVE = 101.1


class TestCreditUnits:
    def test_scored_from_percent(self):
        assert normalize("credit", 3.23) == pytest.approx(24.6)
        assert normalize("credit", 2.0) == 0.0
        assert normalize("credit", 7.0) == 100.0
        assert normalize("credit", 10.9) == 100.0   # 2020 peak saturates, not overflows

    def test_a_crisis_spread_is_not_zero(self):
        # The old formula, (hy - 200) / 500, took percent as if it were basis
        # points and clamped to 0 at every real level — including 7%.
        assert normalize("credit", 7.0) > 0


class TestWorkedExample:
    def test_corrected_score(self):
        r = calculate_composite_risk_score(LIVE, VRP, move_index=MOVE)
        assert r["score"] == pytest.approx(31.8, abs=0.05)
        assert r["components"]["credit"] == pytest.approx(24.6)
        assert r["components"]["move"] == pytest.approx(31.1)
        assert r["missing"] == [] and r["component_count"] == 8
        assert sum(r["weights"].values()) == pytest.approx(1.0)
        assert r["interpretation"] == "LOW RISK"

    def test_the_old_26_was_credit_scoring_zero(self):
        # Force credit to 0 (what the bps formula did) and drop MOVE (which the
        # old code never found): the published 26 falls out.
        r = calculate_composite_risk_score({**LIVE, "credit_spread_hy": 2.0}, VRP)
        assert r["score"] == pytest.approx(26.4, abs=0.05)

    def test_missing_move_is_reported_and_weights_renormalize(self):
        r = calculate_composite_risk_score(LIVE, VRP)
        assert r["missing"] == ["move"]
        assert sum(r["weights"].values()) == pytest.approx(0.90)
        assert r["score"] == pytest.approx(31.8, abs=0.1)   # MOVE at 31 happens to sit near the mean


class TestShape:
    def test_empty_snapshot_scores_none(self):
        r = calculate_composite_risk_score({})
        assert r["score"] is None and r["missing"] == list(COMPONENTS)

    def test_single_input_scores_its_own_normalized_value(self):
        assert calculate_composite_risk_score({"vix_spot": 35.0})["score"] == pytest.approx(50.0)

    def test_unusable_values_are_ignored_not_fatal(self):
        r = calculate_composite_risk_score({"vix_spot": float("nan"), "skew": "abc", "credit_spread_hy": 3.0})
        assert list(r["components"]) == ["credit"]

    @pytest.mark.parametrize("vix, band", [(10, "VERY BULLISH"), (22.5, "LOW RISK"), (30, "NEUTRAL"), (40, "ELEVATED"), (47.5, "HIGH RISK")])
    def test_bands(self, vix, band):
        assert calculate_composite_risk_score({"vix_spot": vix})["interpretation"] == band

    def test_contrarian_inputs_lower_the_score(self):
        # Documented design, kept: fear and heavy put buying read as relief.
        fear = calculate_composite_risk_score({"fear_greed_score": 10})["score"]
        greed = calculate_composite_risk_score({"fear_greed_score": 90})["score"]
        assert fear < greed
        puts = calculate_composite_risk_score({"put_call_ratio": 1.4})["score"]
        calls = calculate_composite_risk_score({"put_call_ratio": 0.6})["score"]
        assert puts < calls

    def test_legacy_ui_import_is_the_same_function(self):
        from dashboard.ui_helpers import calculate_composite_risk_score as legacy
        assert legacy is calculate_composite_risk_score
