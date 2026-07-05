"""
Test Suite for RepoCollector (data_collectors/repo_collector_enhanced.py)

Tests cover:
- fetch_fred_series: CSV parsing, missing-value ("." ) handling, HTTP/network errors
- get_repo_history: SOFR/IORB/RRP merge, spread calc, RRP billions conversion,
  SOFR z-score data-quality handling (insufficient data => NaN, not 0.0)
- get_full_snapshot: liquidity classification by SOFR-IORB spread, empty handling

All network access is mocked at the requests boundary — no real FRED calls.

Run with: python -m pytest tests/test_repo_collector.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import requests

from data_collectors.repo_collector_enhanced import (
    MIN_ZSCORE_OBSERVATIONS,
    RepoCollector,
)

REQUESTS_GET = "data_collectors.repo_collector_enhanced.requests.get"


def _mock_response(csv_text):
    resp = MagicMock()
    resp.text = csv_text
    resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# fetch_fred_series
# ---------------------------------------------------------------------------
class TestFetchFredSeries:
    def test_parses_valid_csv(self):
        # Arrange
        csv = "DATE,SOFR\n2023-01-02,4.30\n2023-01-03,4.31\n2023-01-04,4.32\n"
        collector = RepoCollector()

        # Act
        with patch(REQUESTS_GET, return_value=_mock_response(csv)):
            df = collector.fetch_fred_series("SOFR", days_back=30)

        # Assert
        assert list(df.columns) == ["date", "value"]
        assert len(df) == 3
        assert df["value"].iloc[-1] == pytest.approx(4.32)

    def test_drops_missing_value_markers(self):
        # Arrange - FRED encodes missing observations as "."
        csv = "DATE,SOFR\n2023-01-02,4.30\n2023-01-03,.\n2023-01-04,4.32\n"
        collector = RepoCollector()

        # Act
        with patch(REQUESTS_GET, return_value=_mock_response(csv)):
            df = collector.fetch_fred_series("SOFR", days_back=30)

        # Assert - the "." row is coerced to NaN then dropped
        assert len(df) == 2
        assert not df["value"].isna().any()

    def test_returns_empty_on_http_error(self):
        # Arrange
        resp = MagicMock()
        resp.raise_for_status.side_effect = requests.HTTPError("500")
        collector = RepoCollector()

        # Act
        with patch(REQUESTS_GET, return_value=resp):
            df = collector.fetch_fred_series("SOFR", days_back=30)

        # Assert
        assert df.empty

    def test_returns_empty_on_network_exception(self):
        # Arrange
        collector = RepoCollector()

        # Act
        with patch(REQUESTS_GET, side_effect=requests.ConnectionError("no route")):
            df = collector.fetch_fred_series("SOFR", days_back=30)

        # Assert
        assert df.empty


# ---------------------------------------------------------------------------
# get_repo_history
# ---------------------------------------------------------------------------
def _series_df(dates, values):
    return pd.DataFrame({"date": pd.to_datetime(dates), "value": values})


class TestGetRepoHistory:
    def test_returns_empty_when_sofr_missing(self):
        # Arrange - SOFR fetch returns empty => cannot proceed
        collector = RepoCollector()
        with patch.object(collector, "fetch_fred_series", return_value=pd.DataFrame()):
            # Act
            df = collector.get_repo_history(days_back=30)

        # Assert
        assert df.empty

    def test_merges_iorb_and_computes_spread(self):
        # Arrange
        dates = pd.date_range("2023-01-02", periods=5, freq="D")
        sofr = _series_df(dates, [4.30, 4.31, 4.32, 4.33, 4.34])
        iorb = _series_df(dates, [4.40, 4.40, 4.40, 4.40, 4.40])
        rrp = pd.DataFrame()  # no RRP

        def fake_fetch(series_id, days_back=730):
            return {"SOFR": sofr, "IORB": iorb, "RRPONTSYD": rrp}[series_id].copy()

        collector = RepoCollector()
        with patch.object(collector, "fetch_fred_series", side_effect=fake_fetch):
            # Act
            df = collector.get_repo_history(days_back=30)

        # Assert
        assert "sofr_iorb_spread" in df.columns
        # spread = sofr - iorb = 4.34 - 4.40 = -0.06 on last row
        assert df["sofr_iorb_spread"].iloc[-1] == pytest.approx(-0.06, abs=1e-6)
        assert df["iorb_is_actual"].all()

    def test_converts_rrp_to_billions(self):
        # Arrange
        dates = pd.date_range("2023-01-02", periods=3, freq="D")
        sofr = _series_df(dates, [4.30, 4.31, 4.32])
        rrp = _series_df(dates, [2_000_000.0, 2_100_000.0, 2_200_000.0])  # millions

        def fake_fetch(series_id, days_back=730):
            return {
                "SOFR": sofr,
                "IORB": pd.DataFrame(),
                "RRPONTSYD": rrp,
            }[series_id].copy()

        collector = RepoCollector()
        with patch.object(collector, "fetch_fred_series", side_effect=fake_fetch):
            # Act
            df = collector.get_repo_history(days_back=30)

        # Assert - divided by 1000
        assert df["rrp_on"].iloc[-1] == pytest.approx(2200.0)

    def test_zscore_is_nan_when_insufficient_data(self):
        # Arrange - fewer than MIN_ZSCORE_OBSERVATIONS rows
        n = MIN_ZSCORE_OBSERVATIONS - 5
        dates = pd.date_range("2023-01-02", periods=n, freq="D")
        sofr = _series_df(dates, [4.30] * n)

        def fake_fetch(series_id, days_back=730):
            return {
                "SOFR": sofr,
                "IORB": pd.DataFrame(),
                "RRPONTSYD": pd.DataFrame(),
            }[series_id].copy()

        collector = RepoCollector()
        with patch.object(collector, "fetch_fred_series", side_effect=fake_fetch):
            # Act
            df = collector.get_repo_history(days_back=30)

        # Assert - z-score not falsely reported as 0.0
        assert df["sofr_z_score"].isna().all()
        assert (df["sofr_z_score_quality"] == "INSUFFICIENT_DATA").all()

    def test_zscore_computed_with_sufficient_varied_data(self):
        # Arrange - enough rows AND real variance so std > threshold
        n = 300
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        rng = np.random.default_rng(42)
        values = 4.30 + rng.normal(0, 0.05, n)
        sofr = _series_df(dates, values)

        def fake_fetch(series_id, days_back=730):
            return {
                "SOFR": sofr,
                "IORB": pd.DataFrame(),
                "RRPONTSYD": pd.DataFrame(),
            }[series_id].copy()

        collector = RepoCollector()
        with patch.object(collector, "fetch_fred_series", side_effect=fake_fetch):
            # Act
            df = collector.get_repo_history(days_back=730)

        # Assert - the latest row has a real z-score marked OK
        assert pd.notna(df["sofr_z_score"].iloc[-1])
        assert df["sofr_z_score_quality"].iloc[-1] == "OK"


# ---------------------------------------------------------------------------
# get_full_snapshot - liquidity classification
# ---------------------------------------------------------------------------
class TestGetFullSnapshot:
    def _history_with_spread(self, sofr, iorb):
        """Single-row-ish history where latest SOFR-IORB spread is controlled."""
        dates = pd.date_range("2023-01-02", periods=2, freq="D")
        return pd.DataFrame(
            {
                "date": dates.date,
                "sofr": [sofr, sofr],
                "iorb": [iorb, iorb],
                "iorb_is_actual": [True, True],
                "sofr_iorb_spread": [sofr - iorb, sofr - iorb],
                "sofr_z_score": [np.nan, np.nan],
                "sofr_z_score_quality": ["INSUFFICIENT_DATA", "INSUFFICIENT_DATA"],
            }
        )

    @pytest.mark.parametrize(
        "spread_bps, expected",
        [
            (-2, "ABUNDANT"),   # spread <= 0
            (3, "AMPLE"),       # 0 < spread <= 5
            (10, "NORMAL"),     # 5 < spread <= 15
            (20, "TIGHTENING"), # 15 < spread <= 30
            (40, "STRESS"),     # spread > 30
        ],
    )
    def test_liquidity_classification(self, spread_bps, expected):
        # spread stored in same units as sofr/iorb; classification thresholds are
        # 0/5/15/30 — build sofr/iorb so sofr_iorb_spread == spread_bps
        df = self._history_with_spread(sofr=4.30 + spread_bps, iorb=4.30)
        collector = RepoCollector()
        with patch.object(collector, "get_repo_history", return_value=df):
            snap = collector.get_full_snapshot()
        assert snap["liquidity_status"] == expected

    def test_returns_empty_dict_when_no_history(self):
        collector = RepoCollector()
        with patch.object(collector, "get_repo_history", return_value=pd.DataFrame()):
            assert collector.get_full_snapshot() == {}

    def test_zscore_none_when_insufficient(self):
        df = self._history_with_spread(sofr=4.33, iorb=4.30)
        collector = RepoCollector()
        with patch.object(collector, "get_repo_history", return_value=df):
            snap = collector.get_full_snapshot()
        # None, not 0.0, when data is insufficient
        assert snap["sofr_z_score"] is None
        assert snap["sofr_z_score_quality"] == "INSUFFICIENT_DATA"

    def test_snapshot_includes_core_fields(self):
        df = self._history_with_spread(sofr=4.33, iorb=4.30)
        collector = RepoCollector()
        with patch.object(collector, "get_repo_history", return_value=df):
            snap = collector.get_full_snapshot()
        assert snap["sofr"] == pytest.approx(4.33)
        assert snap["iorb"] == pytest.approx(4.30)
        assert isinstance(snap["repo_df"], pd.DataFrame)
        assert snap["data_quality"]["min_required_for_zscore"] == MIN_ZSCORE_OBSERVATIONS
