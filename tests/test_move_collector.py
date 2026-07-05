"""
Test Suite for MOVECollector (data_collectors/move_collector.py)

Tests cover:
- get_move_data: FRED primary path, Yahoo fallback, both-fail path
- Source divergence detection (FRED vs Yahoo > 5%)
- get_full_snapshot: stress classification, percentile, empty handling
- get_latest_move: latest value extraction and None on empty

All network access is mocked at the boundary (FREDCollector + yfinance.Ticker).
No real FRED or Yahoo calls are made.

Run with: python -m pytest tests/test_move_collector.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data_collectors.move_collector import MOVECollector, get_move_index

FRED_COLLECTOR = "data_collectors.fred_collector.FREDCollector"
YF_TICKER = "yfinance.Ticker"


# ---------------------------------------------------------------------------
# Helpers to build realistic mocked payloads
# ---------------------------------------------------------------------------
def _fred_series_df(values, start="2023-01-02"):
    """Mimic FREDCollector.get_series output: a 'date' col + a value col."""
    dates = pd.date_range(start=start, periods=len(values), freq="D")
    return pd.DataFrame({"date": dates, "MOVE": values})


def _yahoo_history_df(values, start="2023-01-02"):
    """Mimic yfinance Ticker.history output: DatetimeIndex + OHLC columns."""
    idx = pd.date_range(start=start, periods=len(values), freq="D")
    return pd.DataFrame(
        {
            "Open": values,
            "High": values,
            "Low": values,
            "Close": values,
            "Volume": [0] * len(values),
        },
        index=idx,
    )


def _make_fred_mock(series_df):
    fred_instance = MagicMock()
    fred_instance.get_series.return_value = series_df
    return fred_instance


def _make_ticker_mock(history_df):
    ticker_instance = MagicMock()
    ticker_instance.history.return_value = history_df
    return ticker_instance


# ---------------------------------------------------------------------------
# get_move_data
# ---------------------------------------------------------------------------
class TestGetMoveData:
    def test_returns_fred_dataframe_when_fred_available(self):
        # Arrange
        fred_df = _fred_series_df([95.0, 98.0, 101.0])
        collector = MOVECollector()

        # Act
        with patch(FRED_COLLECTOR, return_value=_make_fred_mock(fred_df)), \
             patch(YF_TICKER, return_value=_make_ticker_mock(pd.DataFrame())):
            result = collector.get_move_data(lookback_days=30)

        # Assert
        assert not result.empty
        assert list(result["source"].unique()) == ["FRED"]
        assert result["move"].iloc[-1] == 101.0

    def test_falls_back_to_yahoo_when_fred_empty(self):
        # Arrange
        collector = MOVECollector()
        yahoo_df = _yahoo_history_df([110.0, 112.5, 115.0])

        # Act
        with patch(FRED_COLLECTOR, return_value=_make_fred_mock(pd.DataFrame())), \
             patch(YF_TICKER, return_value=_make_ticker_mock(yahoo_df)):
            result = collector.get_move_data(lookback_days=30)

        # Assert
        assert not result.empty
        assert list(result["source"].unique()) == ["Yahoo"]
        assert result["move"].iloc[-1] == 115.0

    def test_returns_empty_when_both_sources_fail(self):
        # Arrange
        collector = MOVECollector()

        # Act
        with patch(FRED_COLLECTOR, return_value=_make_fred_mock(pd.DataFrame())), \
             patch(YF_TICKER, return_value=_make_ticker_mock(pd.DataFrame())):
            result = collector.get_move_data(lookback_days=30)

        # Assert
        assert result.empty

    def test_returns_empty_when_fred_raises_and_yahoo_empty(self):
        # Arrange
        collector = MOVECollector()
        fred_instance = MagicMock()
        fred_instance.get_series.side_effect = RuntimeError("FRED down")

        # Act
        with patch(FRED_COLLECTOR, return_value=fred_instance), \
             patch(YF_TICKER, return_value=_make_ticker_mock(pd.DataFrame())):
            result = collector.get_move_data(lookback_days=30)

        # Assert
        assert result.empty

    def test_uses_fred_as_primary_even_when_yahoo_also_present(self):
        # Arrange
        fred_df = _fred_series_df([100.0, 100.0, 100.0])
        yahoo_df = _yahoo_history_df([102.0, 102.0, 102.0])
        collector = MOVECollector()

        # Act
        with patch(FRED_COLLECTOR, return_value=_make_fred_mock(fred_df)), \
             patch(YF_TICKER, return_value=_make_ticker_mock(yahoo_df)):
            result = collector.get_move_data(lookback_days=30, validate_sources=True)

        # Assert - FRED wins when both available
        assert list(result["source"].unique()) == ["FRED"]

    def test_logs_warning_on_source_divergence(self, caplog):
        # Arrange - FRED=100, Yahoo=115 => 15% divergence > 5% threshold
        fred_df = _fred_series_df([100.0, 100.0, 100.0])
        yahoo_df = _yahoo_history_df([115.0, 115.0, 115.0])
        collector = MOVECollector()

        # Act
        with patch(FRED_COLLECTOR, return_value=_make_fred_mock(fred_df)), \
             patch(YF_TICKER, return_value=_make_ticker_mock(yahoo_df)):
            with caplog.at_level("WARNING"):
                collector.get_move_data(lookback_days=30, validate_sources=True)

        # Assert
        assert any("DIVERGENCE" in rec.message for rec in caplog.records)

    def test_no_divergence_warning_when_sources_aligned(self, caplog):
        # Arrange - FRED=100, Yahoo=102 => 2% divergence < 5% threshold
        fred_df = _fred_series_df([100.0, 100.0, 100.0])
        yahoo_df = _yahoo_history_df([102.0, 102.0, 102.0])
        collector = MOVECollector()

        # Act
        with patch(FRED_COLLECTOR, return_value=_make_fred_mock(fred_df)), \
             patch(YF_TICKER, return_value=_make_ticker_mock(yahoo_df)):
            with caplog.at_level("WARNING"):
                collector.get_move_data(lookback_days=30, validate_sources=True)

        # Assert
        assert not any("DIVERGENCE" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# get_full_snapshot - stress classification
# ---------------------------------------------------------------------------
class TestGetFullSnapshot:
    def _snapshot_for_level(self, latest_value):
        """Build a snapshot where the latest MOVE value is `latest_value`."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-02", periods=4, freq="D").date,
                "move": [90.0, 95.0, 100.0, latest_value],
                "source": ["FRED"] * 4,
            }
        )
        collector = MOVECollector()
        with patch.object(collector, "get_move_data", return_value=df):
            return collector.get_full_snapshot()

    def test_classifies_low_stress_below_80(self):
        snap = self._snapshot_for_level(70.0)
        assert snap["stress_level"] == "LOW"
        assert snap["move"] == 70.0

    def test_classifies_normal_stress_between_80_and_120(self):
        snap = self._snapshot_for_level(105.0)
        assert snap["stress_level"] == "NORMAL"

    def test_classifies_elevated_stress_between_120_and_150(self):
        snap = self._snapshot_for_level(135.0)
        assert snap["stress_level"] == "ELEVATED"

    def test_classifies_high_stress_above_150(self):
        snap = self._snapshot_for_level(180.0)
        assert snap["stress_level"] == "HIGH"

    def test_snapshot_includes_compatibility_keys(self):
        snap = self._snapshot_for_level(100.0)
        # Dashboard expects both naming variants
        assert snap["move"] == snap["move_index"]
        assert snap["move_percentile"] == snap["percentile"]
        assert "move_df" in snap
        assert isinstance(snap["move_df"], pd.DataFrame)

    def test_percentile_reflects_rank_within_history(self):
        # latest (180) is the max => strictly greater than all 3 priors => 75th pct
        snap = self._snapshot_for_level(180.0)
        assert snap["percentile"] == pytest.approx(75.0)

    def test_returns_empty_dict_when_no_data(self):
        collector = MOVECollector()
        with patch.object(collector, "get_move_data", return_value=pd.DataFrame()):
            assert collector.get_full_snapshot() == {}


# ---------------------------------------------------------------------------
# get_latest_move
# ---------------------------------------------------------------------------
class TestGetLatestMove:
    def test_returns_latest_value(self):
        df = pd.DataFrame({"date": ["2023-01-02"], "move": [123.4], "source": ["FRED"]})
        collector = MOVECollector()
        with patch.object(collector, "get_move_data", return_value=df):
            assert collector.get_latest_move() == pytest.approx(123.4)

    def test_returns_none_when_empty(self):
        collector = MOVECollector()
        with patch.object(collector, "get_move_data", return_value=pd.DataFrame()):
            assert collector.get_latest_move() is None

    def test_returns_none_when_get_move_data_raises(self):
        collector = MOVECollector()
        with patch.object(collector, "get_move_data", side_effect=ValueError("boom")):
            assert collector.get_latest_move() is None


# ---------------------------------------------------------------------------
# Legacy compatibility function
# ---------------------------------------------------------------------------
def test_legacy_get_move_index_delegates_to_collector():
    fred_df = _fred_series_df([100.0, 101.0])
    with patch(FRED_COLLECTOR, return_value=_make_fred_mock(fred_df)), \
         patch(YF_TICKER, return_value=_make_ticker_mock(pd.DataFrame())):
        result = get_move_index(lookback_days=30)
    assert not result.empty
    assert "move" in result.columns
