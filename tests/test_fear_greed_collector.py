"""
Test Suite for FearGreedCollector (data_collectors/fear_greed_collector.py)

Covers:
- Parsing of CNN Fear & Greed API JSON payloads (no network access)
- Handling of malformed/empty responses and HTTP/network errors
- Rating threshold classification (_get_rating)
- Status-tracked variant (get_data_with_status) OK/UNAVAILABLE/ERROR paths

Run with: python -m pytest tests/test_fear_greed_collector.py -v
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import Mock, patch

import pytest
import requests

from data_collectors.fear_greed_collector import FearGreedCollector
from utils.data_status import DataStatus

REQUESTS_GET = "data_collectors.fear_greed_collector.requests.get"


def make_cnn_payload(score=62.5):
    """Minimal but structurally accurate CNN graphdata payload."""
    return {
        "fear_and_greed": {
            "score": score,
            "rating": "greed",
            "timestamp": "2026-07-02T23:59:56+00:00",
            "previous_close": 60.1,
            "previous_1_week": 55.0,
            "one_week_ago": 55.0,
            "one_month_ago": 48.3,
            "one_year_ago": 71.9,
        },
        "fear_and_greed_historical": {"timestamp": 1751500796000, "score": 62.5, "data": []},
        "market_momentum_sp500": {"timestamp": 1751500796000, "score": 70.0},
    }


def make_response(payload=None, json_error=None):
    """Build a mocked requests.Response-like object."""
    response = Mock()
    response.raise_for_status.return_value = None
    if json_error is not None:
        response.json.side_effect = json_error
    else:
        response.json.return_value = payload
    return response


def make_http_error_response(status_code=404, reason="Not Found"):
    """Build a mocked response whose raise_for_status raises HTTPError."""
    response = Mock()
    error_response = Mock(status_code=status_code, reason=reason)
    response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        f"{status_code} Client Error", response=error_response
    )
    return response


@pytest.fixture
def collector():
    return FearGreedCollector()


class TestGetFearGreedScore:
    """Parsing and error handling for get_fear_greed_score"""

    def test_parses_score_and_historical_values_from_payload(self, collector):
        # Arrange
        response = make_response(make_cnn_payload(score=62.5))

        # Act
        with patch(REQUESTS_GET, return_value=response) as get_mock:
            result = collector.get_fear_greed_score()

        # Assert
        assert result["score"] == pytest.approx(62.5)
        assert result["rating"] == "Greed"
        assert result["previous_close"] == pytest.approx(60.1)
        assert result["one_week_ago"] == pytest.approx(55.0)
        assert result["one_month_ago"] == pytest.approx(48.3)
        assert result["one_year_ago"] == pytest.approx(71.9)
        assert "timestamp" in result
        get_mock.assert_called_once_with(
            collector.url, headers=collector.headers, timeout=10
        )

    def test_coerces_string_score_to_float(self, collector):
        # Arrange - CNN has served scores as strings before
        payload = make_cnn_payload()
        payload["fear_and_greed"]["score"] = "37.2"
        response = make_response(payload)

        # Act
        with patch(REQUESTS_GET, return_value=response):
            result = collector.get_fear_greed_score()

        # Assert
        assert result["score"] == pytest.approx(37.2)
        assert result["rating"] == "Fear"

    def test_returns_none_when_fear_and_greed_key_missing(self, collector):
        # Arrange - structurally valid JSON without the expected key
        response = make_response({"market_momentum_sp500": {"score": 70.0}})

        # Act
        with patch(REQUESTS_GET, return_value=response):
            result = collector.get_fear_greed_score()

        # Assert
        assert result is None

    def test_returns_none_when_response_not_json(self, collector):
        # Arrange - e.g. CNN serving an HTML error page
        response = make_response(json_error=ValueError("No JSON object could be decoded"))

        # Act
        with patch(REQUESTS_GET, return_value=response):
            result = collector.get_fear_greed_score()

        # Assert
        assert result is None

    def test_returns_none_when_score_not_numeric(self, collector):
        # Arrange - malformed score value
        payload = make_cnn_payload()
        payload["fear_and_greed"]["score"] = "not-a-number"
        response = make_response(payload)

        # Act
        with patch(REQUESTS_GET, return_value=response):
            result = collector.get_fear_greed_score()

        # Assert
        assert result is None

    def test_returns_none_on_timeout(self, collector):
        with patch(REQUESTS_GET, side_effect=requests.exceptions.Timeout()):
            result = collector.get_fear_greed_score()

        assert result is None

    def test_returns_none_on_connection_error(self, collector):
        with patch(REQUESTS_GET, side_effect=requests.exceptions.ConnectionError("refused")):
            result = collector.get_fear_greed_score()

        assert result is None

    def test_returns_none_on_http_error(self, collector):
        # Arrange
        response = make_http_error_response(status_code=503, reason="Service Unavailable")

        # Act
        with patch(REQUESTS_GET, return_value=response):
            result = collector.get_fear_greed_score()

        # Assert
        assert result is None


class TestGetRating:
    """Score → rating threshold classification"""

    @pytest.mark.parametrize("score,expected_rating", [
        (0.0, "Extreme Fear"),
        (2.0, "Extreme Fear"),     # COVID-19 March 2020 level
        (24.9, "Extreme Fear"),
        (25.0, "Fear"),            # lower Fear boundary is inclusive
        (44.9, "Fear"),
        (45.0, "Neutral"),         # lower Neutral boundary is inclusive
        (50.0, "Neutral"),
        (54.9, "Neutral"),
        (55.0, "Greed"),           # lower Greed boundary is inclusive
        (74.9, "Greed"),
        (75.0, "Extreme Greed"),   # lower Extreme Greed boundary is inclusive
        (100.0, "Extreme Greed"),
    ])
    def test_rating_for_score(self, collector, score, expected_rating):
        result = collector._get_rating(score)

        assert result == expected_rating


class TestGetDataWithStatus:
    """Status-tracked variant returning OK/UNAVAILABLE/ERROR envelopes"""

    def test_returns_ok_status_with_data_on_success(self, collector):
        # Arrange
        response = make_response(make_cnn_payload(score=18.0))

        # Act
        with patch(REQUESTS_GET, return_value=response):
            result = collector.get_data_with_status()

        # Assert
        assert result["status"] == DataStatus.OK.value
        assert result["error"] is None
        assert result["source"] == "CNN Fear & Greed Index"
        assert result["data"]["score"] == pytest.approx(18.0)
        assert result["data"]["rating"] == "Extreme Fear"

    def test_returns_unavailable_when_key_missing(self, collector):
        # Arrange
        response = make_response({"unexpected": {}})

        # Act
        with patch(REQUESTS_GET, return_value=response):
            result = collector.get_data_with_status()

        # Assert
        assert result["status"] == DataStatus.UNAVAILABLE.value
        assert result["data"] is None
        assert "fear_and_greed" in result["error"]

    def test_returns_error_status_on_timeout(self, collector):
        with patch(REQUESTS_GET, side_effect=requests.exceptions.Timeout()):
            result = collector.get_data_with_status()

        assert result["status"] == DataStatus.ERROR.value
        assert result["data"] is None
        assert "timeout" in result["error"].lower()

    def test_returns_error_status_on_connection_error(self, collector):
        with patch(REQUESTS_GET, side_effect=requests.exceptions.ConnectionError("refused")):
            result = collector.get_data_with_status()

        assert result["status"] == DataStatus.ERROR.value
        assert "Connection failed" in result["error"]

    def test_returns_error_status_with_http_code_and_reason(self, collector):
        # Arrange
        response = make_http_error_response(status_code=404, reason="Not Found")

        # Act
        with patch(REQUESTS_GET, return_value=response):
            result = collector.get_data_with_status()

        # Assert
        assert result["status"] == DataStatus.ERROR.value
        assert result["error"] == "HTTP 404: Not Found"

    def test_returns_error_status_on_unexpected_exception(self, collector):
        # Arrange - non-JSON body raises inside .json()
        response = make_response(json_error=ValueError("bad payload"))

        # Act
        with patch(REQUESTS_GET, return_value=response):
            result = collector.get_data_with_status()

        # Assert
        assert result["status"] == DataStatus.ERROR.value
        assert "ValueError" in result["error"]
