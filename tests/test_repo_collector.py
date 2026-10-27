"""
Test Suite for RepoCollector (data_collectors/repo_collector_enhanced.py)

Tests cover:
- fetch_fred_series: CSV parsing, missing-value ("." ) handling, HTTP/network errors
- get_repo_history: SOFR/IORB/RRP merge, spread calc in basis points, RRP $B pass-through,
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
        # Public compatibility key is explicitly basis points:
        # (4.34% - 4.40%) * 100 = -6 bps.
        assert df["sofr_iorb_spread"].iloc[-1] == pytest.approx(-6.0, abs=1e-6)
        assert df["iorb_is_actual"].all()

    def test_preserves_fred_rrp_billions_values(self):
        # Arrange
        dates = pd.date_range("2023-01-02", periods=3, freq="D")
        sofr = _series_df(dates, [4.30, 4.31, 4.32])
        # RRPONTSYD is published by FRED in billions of US dollars.
        rrp = _series_df(dates, [87.4, 82.5, 79.25])

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

        # Assert - realistic source values pass through unchanged in $B.
        assert df["rrp_on"].tolist() == pytest.approx([87.4, 82.5, 79.25])

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
    def _history_with_spread_bps(self, spread_bps):
        """History using realistic percentage rates and a bps spread contract."""
        dates = pd.date_range("2023-01-02", periods=2, freq="D")
        iorb = 4.30
        sofr = iorb + spread_bps / 100.0
        return pd.DataFrame(
            {
                "date": dates.date,
                "sofr": [sofr, sofr],
                "iorb": [iorb, iorb],
                "iorb_is_actual": [True, True],
                "sofr_iorb_spread": [spread_bps, spread_bps],
                "sofr_z_score": [np.nan, np.nan],
                "sofr_z_score_quality": ["INSUFFICIENT_DATA", "INSUFFICIENT_DATA"],
            }
        )

    @pytest.mark.parametrize(
        "spread_bps, expected",
        [
            (-1, "ABUNDANT"),
            (0, "ABUNDANT"),
            (0.01, "AMPLE"),
            (5, "AMPLE"),
            (5.01, "NORMAL"),
            (15, "NORMAL"),
            (15.01, "TIGHTENING"),
            (30, "TIGHTENING"),
            (30.01, "STRESS"),
        ],
    )
    def test_liquidity_classification_at_exact_bps_thresholds(self, spread_bps, expected):
        df = self._history_with_spread_bps(spread_bps)
        collector = RepoCollector()
        with patch.object(collector, "get_repo_history", return_value=df):
            snap = collector.get_full_snapshot()
        assert snap["liquidity_status"] == expected

    def test_returns_empty_dict_when_no_history(self):
        collector = RepoCollector()
        with patch.object(collector, "get_repo_history", return_value=pd.DataFrame()):
            assert collector.get_full_snapshot() == {}

    def test_zscore_none_when_insufficient(self):
        df = self._history_with_spread_bps(3.0)
        collector = RepoCollector()
        with patch.object(collector, "get_repo_history", return_value=df):
            snap = collector.get_full_snapshot()
        # None, not 0.0, when data is insufficient
        assert snap["sofr_z_score"] is None
        assert snap["sofr_z_score_quality"] == "INSUFFICIENT_DATA"

    def test_snapshot_includes_core_fields(self):
        df = self._history_with_spread_bps(3.0)
        collector = RepoCollector()
        with patch.object(collector, "get_repo_history", return_value=df):
            snap = collector.get_full_snapshot()
        assert snap["sofr"] == pytest.approx(4.33)
        assert snap["iorb"] == pytest.approx(4.30)
        assert isinstance(snap["repo_df"], pd.DataFrame)
        assert snap["data_quality"]["min_required_for_zscore"] == MIN_ZSCORE_OBSERVATIONS
