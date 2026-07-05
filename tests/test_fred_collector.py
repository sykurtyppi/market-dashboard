"""
Test Suite for FREDCollector (data_collectors/fred_collector.py)

Tests cover:
- Construction / API key resolution
- get_series parsing of FRED JSON observations (including "." missing markers)
- Error paths (network errors, HTTP errors, malformed JSON)
- get_all_indicators / get_latest_values aggregation
- calculate_credit_spread_signals LEFT-style signals
- get_baa_aaa_spread
- get_data_with_status status classification (ok / partial / unavailable)

All network access is mocked at the requests boundary — no real FRED calls.

Run with: python -m pytest tests/test_fred_collector.py -v
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import requests

from data_collectors.fred_collector import FREDCollector

FAKE_API_KEY = "fake-test-fred-key"

REQUESTS_GET = "data_collectors.fred_collector.requests.get"
TIME_SLEEP = "data_collectors.fred_collector.time.sleep"
GET_SECRET = "data_collectors.fred_collector.get_secret"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_observations(values, start="2024-01-02"):
    """Build a realistic FRED observations list from raw value strings/numbers."""
    dates = pd.bdate_range(start=start, periods=len(values))
    return [
        {
            "realtime_start": "2024-06-01",
            "realtime_end": "2024-06-01",
            "date": date.strftime("%Y-%m-%d"),
            "value": str(value),
        }
        for date, value in zip(dates, values)
    ]


def make_json_response(payload):
    """Mock requests.Response returning the given JSON payload."""
    response = Mock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


def make_series_df(series_id, values, start="2024-01-02"):
    """Build a DataFrame shaped like FREDCollector.get_series output."""
    dates = pd.bdate_range(start=start, periods=len(values))
    return pd.DataFrame({"date": dates, series_id: values})


@pytest.fixture
def collector():
    return FREDCollector(api_key=FAKE_API_KEY)


# ---------------------------------------------------------------------------
# Construction / API key handling
# ---------------------------------------------------------------------------

class TestInit:
    def test_explicit_api_key_is_used(self):
        # Arrange / Act
        collector = FREDCollector(api_key=FAKE_API_KEY)

        # Assert
        assert collector.api_key == FAKE_API_KEY
        assert "api.stlouisfed.org" in collector.base_url

    def test_falls_back_to_secret_when_no_key_given(self):
        # Arrange
        with patch(GET_SECRET, return_value="secret-key") as mock_secret:
            # Act
            collector = FREDCollector()

        # Assert
        mock_secret.assert_called_once_with("FRED_API_KEY")
        assert collector.api_key == "secret-key"

    def test_raises_value_error_when_key_missing(self):
        # Arrange
        with patch(GET_SECRET, return_value=None):
            # Act / Assert
            with pytest.raises(ValueError, match="FRED API key not found"):
                FREDCollector()


# ---------------------------------------------------------------------------
# get_series
# ---------------------------------------------------------------------------

class TestGetSeries:
    def test_parses_observations_into_dataframe(self, collector):
        # Arrange
        payload = {"observations": make_observations([4.51, 4.62, 4.48])}
        with patch(REQUESTS_GET, return_value=make_json_response(payload)):
            # Act
            df = collector.get_series("BAMLH0A0HYM2", start_date="2024-01-01")

        # Assert
        assert list(df.columns) == ["date", "BAMLH0A0HYM2"]
        assert len(df) == 3
        assert pd.api.types.is_datetime64_any_dtype(df["date"])
        assert df["BAMLH0A0HYM2"].tolist() == [4.51, 4.62, 4.48]

    def test_missing_value_markers_are_dropped(self, collector):
        # Arrange - FRED uses "." for missing observations (holidays etc.)
        payload = {"observations": make_observations([4.51, ".", 4.48, "."])}
        with patch(REQUESTS_GET, return_value=make_json_response(payload)):
            # Act
            df = collector.get_series("DGS10", start_date="2024-01-01")

        # Assert
        assert len(df) == 2
        assert df["DGS10"].tolist() == [4.51, 4.48]
        assert not df["DGS10"].isna().any()

    def test_all_missing_values_returns_empty_dataframe(self, collector):
        # Arrange
        payload = {"observations": make_observations([".", ".", "."])}
        with patch(REQUESTS_GET, return_value=make_json_response(payload)):
            # Act
            df = collector.get_series("DGS10", start_date="2024-01-01")

        # Assert
        assert df.empty

    def test_request_params_include_key_series_and_start(self, collector):
        # Arrange
        payload = {"observations": make_observations([1.0])}
        with patch(REQUESTS_GET, return_value=make_json_response(payload)) as mock_get:
            # Act
            collector.get_series("DFF", start_date="2023-05-01", limit=500)

        # Assert
        params = mock_get.call_args.kwargs["params"]
        assert params["series_id"] == "DFF"
        assert params["api_key"] == FAKE_API_KEY
        assert params["file_type"] == "json"
        assert params["observation_start"] == "2023-05-01"
        assert params["limit"] == 500

    def test_default_start_date_is_about_two_years_back(self, collector):
        # Arrange
        payload = {"observations": make_observations([1.0])}
        with patch(REQUESTS_GET, return_value=make_json_response(payload)) as mock_get:
            # Act
            collector.get_series("DFF")

        # Assert
        params = mock_get.call_args.kwargs["params"]
        start = datetime.strptime(params["observation_start"], "%Y-%m-%d")
        expected = datetime.now() - timedelta(days=730)
        assert abs((start - expected).total_seconds()) < 86400 * 2

    def test_missing_observations_key_returns_empty(self, collector):
        # Arrange - e.g. FRED error payload
        payload = {"error_code": 400, "error_message": "Bad Request"}
        with patch(REQUESTS_GET, return_value=make_json_response(payload)):
            # Act
            df = collector.get_series("BOGUS", start_date="2024-01-01")

        # Assert
        assert df.empty

    def test_empty_observations_list_returns_empty(self, collector):
        # Arrange
        payload = {"observations": []}
        with patch(REQUESTS_GET, return_value=make_json_response(payload)):
            # Act
            df = collector.get_series("DGS10", start_date="2024-01-01")

        # Assert
        assert df.empty

    def test_connection_error_returns_empty_without_retrying(self, collector):
        # Arrange - the internal try/except swallows RequestException before the
        # exponential_backoff_retry decorator can see it, so exactly one attempt
        # is made (documents current behavior).
        with patch(
            REQUESTS_GET,
            side_effect=requests.exceptions.ConnectionError("boom"),
        ) as mock_get:
            # Act
            df = collector.get_series("DGS10", start_date="2024-01-01")

        # Assert
        assert df.empty
        assert mock_get.call_count == 1

    def test_http_error_status_returns_empty(self, collector):
        # Arrange
        response = Mock()
        response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "500 Server Error"
        )
        with patch(REQUESTS_GET, return_value=response):
            # Act
            df = collector.get_series("DGS10", start_date="2024-01-01")

        # Assert
        assert df.empty

    def test_malformed_json_returns_empty(self, collector):
        # Arrange
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.side_effect = ValueError("No JSON object could be decoded")
        with patch(REQUESTS_GET, return_value=response):
            # Act
            df = collector.get_series("DGS10", start_date="2024-01-01")

        # Assert
        assert df.empty


# ---------------------------------------------------------------------------
# get_all_indicators
# ---------------------------------------------------------------------------

class TestGetAllIndicators:
    EXPECTED_NAMES = {
        "credit_spread_hy", "credit_spread_ig", "treasury_10y", "treasury_3m",
        "treasury_2y", "fed_funds", "m2_supply", "corporate_aaa", "corporate_baa",
    }

    def test_returns_all_indicators_when_every_series_succeeds(self, collector):
        # Arrange
        def fake_get_series(series_id, start_date=None, limit=100000):
            return make_series_df(series_id, [1.0, 2.0, 3.0])

        with patch.object(collector, "get_series", side_effect=fake_get_series), \
                patch(TIME_SLEEP):
            # Act
            results = collector.get_all_indicators(lookback_days=90)

        # Assert
        assert set(results.keys()) == self.EXPECTED_NAMES
        assert all(isinstance(df, pd.DataFrame) for df in results.values())
        assert list(results["treasury_10y"].columns) == ["date", "DGS10"]

    def test_skips_indicators_with_empty_data(self, collector):
        # Arrange - only the HY spread series returns data
        def fake_get_series(series_id, start_date=None, limit=100000):
            if series_id == "BAMLH0A0HYM2":
                return make_series_df(series_id, [4.5, 4.6])
            return pd.DataFrame()

        with patch.object(collector, "get_series", side_effect=fake_get_series), \
                patch(TIME_SLEEP):
            # Act
            results = collector.get_all_indicators(lookback_days=90)

        # Assert
        assert set(results.keys()) == {"credit_spread_hy"}

    def test_passes_lookback_derived_start_date(self, collector):
        # Arrange
        with patch.object(
            collector, "get_series", return_value=pd.DataFrame()
        ) as mock_series, patch(TIME_SLEEP):
            # Act
            collector.get_all_indicators(lookback_days=10)

        # Assert
        start_arg = mock_series.call_args_list[0].kwargs["start_date"]
        start = datetime.strptime(start_arg, "%Y-%m-%d")
        expected = datetime.now() - timedelta(days=10)
        assert abs((start - expected).total_seconds()) < 86400 * 2


# ---------------------------------------------------------------------------
# get_latest_values
# ---------------------------------------------------------------------------

class TestGetLatestValues:
    def test_returns_last_row_value_date_and_series_id(self, collector):
        # Arrange
        def fake_get_series(series_id, start_date=None, limit=100000):
            return make_series_df(series_id, [1.5, 2.5, 3.75], start="2024-03-01")

        with patch.object(collector, "get_series", side_effect=fake_get_series), \
                patch(TIME_SLEEP):
            # Act
            latest = collector.get_latest_values()

        # Assert
        entry = latest["fed_funds"]
        assert entry["value"] == 3.75
        assert entry["series_id"] == "DFF"
        # Third business day from 2024-03-01 (Fri) is 2024-03-05 (Tue)
        assert entry["date"] == "2024-03-05"
        assert isinstance(entry["value"], float)

    def test_returns_empty_dict_when_nothing_fetched(self, collector):
        # Arrange
        with patch.object(collector, "get_series", return_value=pd.DataFrame()), \
                patch(TIME_SLEEP):
            # Act
            latest = collector.get_latest_values()

        # Assert
        assert latest == {}


# ---------------------------------------------------------------------------
# calculate_credit_spread_signals
# ---------------------------------------------------------------------------

class TestCreditSpreadSignals:
    def _run_with_spreads(self, collector, spreads):
        df = make_series_df("BAMLH0A0HYM2", spreads, start="2023-01-02")
        with patch.object(collector, "get_series", return_value=df):
            return collector.calculate_credit_spread_signals()

    def test_neutral_when_spread_at_ema(self, collector):
        # Arrange / Act - flat series, spread == EMA
        result = self._run_with_spreads(collector, [5.0] * 400)

        # Assert
        assert result["signal"] == "NEUTRAL"
        assert result["current_spread"] == 5.0
        assert result["pct_from_ema"] == pytest.approx(0.0, abs=1e-6)

    def test_buy_when_spread_far_below_ema(self, collector):
        # Arrange / Act - long history at 5.0, sudden drop to 2.0 (~60% below EMA)
        result = self._run_with_spreads(collector, [5.0] * 500 + [2.0])

        # Assert
        assert result["signal"] == "BUY"
        assert result["pct_from_ema"] < -35.0

    def test_sell_when_spread_far_above_ema(self, collector):
        # Arrange / Act - long history at 5.0, sudden spike to 8.0 (~60% above EMA)
        result = self._run_with_spreads(collector, [5.0] * 500 + [8.0])

        # Assert
        assert result["signal"] == "SELL"
        assert result["pct_from_ema"] > 40.0

    def test_contains_required_fields(self, collector):
        # Arrange / Act
        result = self._run_with_spreads(collector, [5.0] * 400)

        # Assert
        for field in ["current_spread", "ema_330", "pct_from_ema", "signal", "date"]:
            assert field in result, f"Missing field: {field}"
        datetime.strptime(result["date"], "%Y-%m-%d")  # must be valid date string

    def test_error_dict_when_data_unavailable(self, collector):
        # Arrange
        with patch.object(collector, "get_series", return_value=pd.DataFrame()):
            # Act
            result = collector.calculate_credit_spread_signals()

        # Assert
        assert result == {"error": "Could not fetch credit spread data"}


# ---------------------------------------------------------------------------
# get_baa_aaa_spread
# ---------------------------------------------------------------------------

class TestBaaAaaSpread:
    def test_returns_latest_baa_minus_aaa(self, collector):
        # Arrange
        def fake_get_series(series_id, start_date=None, limit=100000):
            values = {"DBAA": [5.4, 5.6], "DAAA": [4.7, 4.9]}[series_id]
            return make_series_df(series_id, values)

        with patch.object(collector, "get_series", side_effect=fake_get_series):
            # Act
            spread = collector.get_baa_aaa_spread()

        # Assert
        assert spread == pytest.approx(0.7)

    def test_returns_none_when_either_series_empty(self, collector):
        # Arrange - AAA fetch fails
        def fake_get_series(series_id, start_date=None, limit=100000):
            if series_id == "DBAA":
                return make_series_df(series_id, [5.4])
            return pd.DataFrame()

        with patch.object(collector, "get_series", side_effect=fake_get_series):
            # Act
            spread = collector.get_baa_aaa_spread()

        # Assert
        assert spread is None


# ---------------------------------------------------------------------------
# get_data_with_status
# ---------------------------------------------------------------------------

class TestGetDataWithStatus:
    STATUS_INDICATORS = {
        "credit_spread_hy", "credit_spread_ig", "treasury_10y",
        "treasury_2y", "fed_funds",
    }

    def test_status_ok_when_all_indicators_fetched(self, collector):
        # Arrange
        def fake_get_series(series_id, start_date=None, limit=100000):
            return make_series_df(series_id, [1.0, 2.0])

        with patch.object(collector, "get_series", side_effect=fake_get_series), \
                patch(TIME_SLEEP):
            # Act
            result = collector.get_data_with_status()

        # Assert
        assert result["status"] == "ok"
        assert result["fetched_count"] == 5
        assert result["failed_count"] == 0
        assert result["is_complete"] is True
        assert set(result["data"].keys()) == self.STATUS_INDICATORS
        assert result["data"]["fed_funds"]["value"] == 2.0
        assert result["data"]["fed_funds"]["series_id"] == "DFF"

    def test_status_unavailable_when_all_fail(self, collector):
        # Arrange
        with patch.object(collector, "get_series", return_value=pd.DataFrame()), \
                patch(TIME_SLEEP):
            # Act
            result = collector.get_data_with_status()

        # Assert
        assert result["status"] == "unavailable"
        assert result["fetched_count"] == 0
        assert result["failed_count"] == 5
        assert result["is_complete"] is False
        assert set(result["failed_indicators"]) == self.STATUS_INDICATORS
        assert len(result["errors"]) == 5

    def test_status_partial_when_some_fail(self, collector):
        # Arrange - only treasuries succeed
        def fake_get_series(series_id, start_date=None, limit=100000):
            if series_id in ("DGS10", "DGS2"):
                return make_series_df(series_id, [4.2])
            return pd.DataFrame()

        with patch.object(collector, "get_series", side_effect=fake_get_series), \
                patch(TIME_SLEEP):
            # Act
            result = collector.get_data_with_status()

        # Assert
        assert result["status"] == "partial"
        assert result["fetched_count"] == 2
        assert result["failed_count"] == 3
        assert "credit_spread_hy" in result["failed_indicators"]
        assert any("Empty response" in err for err in result["errors"])

    def test_exception_in_get_series_is_recorded_not_raised(self, collector):
        # Arrange
        with patch.object(
            collector, "get_series", side_effect=RuntimeError("kaboom")
        ), patch(TIME_SLEEP):
            # Act
            result = collector.get_data_with_status()

        # Assert
        assert result["status"] == "unavailable"
        assert result["failed_count"] == 5
        assert all("kaboom" in err for err in result["errors"])
