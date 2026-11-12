# Economic calendar FRED cache regressions.

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from data_collectors import economic_calendar_collector as module


def test_fred_cache_ages_over_a_full_day():
    """A cache entry just over a day old must be refreshed, not reused.

    timedelta.seconds discards whole days, so a 24h1m-old entry previously
    looked fresh (its sub-day remainder is only 60s). The complete elapsed
    age is now measured with total_seconds(), so it exceeds the 1800s TTL and
    the stale cache is refreshed from FRED.
    """
    now = datetime(2026, 11, 11, 12)
    monkeypatched_clock = SimpleNamespace(now=lambda: now)

    collector = module.EconomicCalendarCollector(fred_api_key="dummy")
    cached = [{"date": "2026-10-01", "value": 1.0}]
    expected = [{"date": "2026-11-01", "value": 2.0}]
    collector._fred_data_cache["CPIAUCSL_15"] = cached
    collector._fred_cache_time["CPIAUCSL_15"] = now - timedelta(days=1, minutes=1)

    response = Mock()
    response.json.return_value = {
        "observations": [{"date": "2026-11-01", "value": "2"}]
    }
    request = Mock(return_value=response)

    old_datetime = module.datetime
    old_get = module.requests.get
    try:
        module.datetime = monkeypatched_clock
        module.requests.get = request
        result = collector._fetch_fred_series("CPIAUCSL")
    finally:
        module.datetime = old_datetime
        module.requests.get = old_get

    assert result == expected
    assert request.call_count == 1


def test_fred_cache_avoids_request_when_fresh():
    """A cache entry within the TTL must be reused without a network request."""
    now = datetime(2026, 11, 11, 12)
    monkeypatched_clock = SimpleNamespace(now=lambda: now)

    collector = module.EconomicCalendarCollector(fred_api_key="dummy")
    cached = [{"date": "2026-10-01", "value": 1.0}]
    expected = cached
    collector._fred_data_cache["CPIAUCSL_15"] = cached
    collector._fred_cache_time["CPIAUCSL_15"] = now - timedelta(minutes=1)

    response = Mock()
    response.json.return_value = {
        "observations": [{"date": "2026-11-01", "value": "2"}]
    }
    request = Mock(return_value=response)

    old_datetime = module.datetime
    old_get = module.requests.get
    try:
        module.datetime = monkeypatched_clock
        module.requests.get = request
        result = collector._fetch_fred_series("CPIAUCSL")
    finally:
        module.datetime = old_datetime
        module.requests.get = old_get

    assert result is cached
    assert request.call_count == 0


def test_fred_cache_first_fetch_with_cold_cache():
    """A cold cache must fetch without subtracting a None cache time.

    _fred_cache_time starts as None, so the age check must be guarded or the
    first fetch raises a TypeError. With a new collector, a dummy credential
    and a mocked request, the fetch returns the parsed rows, makes exactly one
    request and initializes the cache.
    """
    collector = module.EconomicCalendarCollector(fred_api_key="dummy")

    response = Mock()
    response.json.return_value = {
        "observations": [
            {"date": "2026-11-01", "value": "2"},
            {"date": "2026-10-01", "value": "1"},
        ]
    }
    request = Mock(return_value=response)
    expected = [
        {"date": "2026-11-01", "value": 2.0},
        {"date": "2026-10-01", "value": 1.0},
    ]

    old_get = module.requests.get
    try:
        module.requests.get = request
        first_fetch = collector._fetch_fred_series("CPIAUCSL")
    finally:
        module.requests.get = old_get

    assert first_fetch == expected
    assert request.call_count == 1
    assert collector._fred_cache_time["CPIAUCSL_15"] is not None
    assert collector._fred_data_cache["CPIAUCSL_15"] == expected


@pytest.mark.parametrize("other_series,other_limit", [("PAYEMS", 15), ("CPIAUCSL", 10)])
def test_fred_cache_keys_age_independently(monkeypatch, other_series, other_limit):
    now = datetime(2026, 11, 11, 12)
    monkeypatch.setattr(module, "datetime", SimpleNamespace(now=lambda: now))
    collector = module.EconomicCalendarCollector(fred_api_key="dummy")

    def fake_get(url, params, timeout):
        response = Mock()
        response.json.return_value = {
            "observations": [{"date": "2026-11-01", "value": str(request.call_count)}]
        }
        return response

    request = Mock(side_effect=fake_get)
    monkeypatch.setattr(module.requests, "get", request)
    assert collector._fetch_fred_series("CPIAUCSL", 15)[0]["value"] == 1
    assert request.call_count == 1
    now += timedelta(minutes=20)
    assert collector._fetch_fred_series(other_series, other_limit)[0]["value"] == 2
    assert request.call_count == 2
    now += timedelta(minutes=11)
    assert collector._fetch_fred_series("CPIAUCSL", 15)[0]["value"] == 3
    assert collector._fetch_fred_series(other_series, other_limit)[0]["value"] == 2
    assert request.call_count == 3
    assert request.call_args_list == [
        call("https://api.stlouisfed.org/fred/series/observations",
             params={"series_id": "CPIAUCSL", "api_key": "dummy", "file_type": "json",
                     "sort_order": "desc", "limit": 15},
             timeout=10),
        call("https://api.stlouisfed.org/fred/series/observations",
             params={"series_id": other_series, "api_key": "dummy", "file_type": "json",
                     "sort_order": "desc", "limit": other_limit},
             timeout=10),
        call("https://api.stlouisfed.org/fred/series/observations",
             params={"series_id": "CPIAUCSL", "api_key": "dummy", "file_type": "json",
                     "sort_order": "desc", "limit": 15},
             timeout=10),
    ]
