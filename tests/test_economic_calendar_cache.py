# Economic calendar FRED cache regressions.

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

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


def test_fred_cache_expires_independently_by_series():
    """A fetch for key B must not extend the freshness of key A.

    With a fixed clock, fetch A, then fetch B twenty minutes later, then at
    minute 31 key A is older than the 1800s TTL and refreshes while key B is
    still within the TTL and stays cached. Each series owns its own timestamp.
    """
    base = datetime(2026, 11, 11, 12)
    collector = module.EconomicCalendarCollector(fred_api_key="dummy")
    collector._fred_cache_time = {}
    collector._fred_data_cache = {}

    a_cold = [{"date": "2026-10-01", "value": 1.0}]
    a_refreshed = [{"date": "2026-11-01", "value": 2.0}]
    b_expected = [{"date": "2026-09-01", "value": 3.0}]

    def make_get(rows):
        response = Mock()
        response.json.return_value = {"observations": rows}
        return Mock(return_value=response)

    old_datetime = module.datetime
    old_get = module.requests.get
    try:
        module.datetime = SimpleNamespace(now=lambda: base)
        module.requests.get = make_get([
            {"date": "2026-10-01", "value": "1"},
        ])
        collector._fetch_fred_series("A")

        module.datetime = SimpleNamespace(now=lambda: base + timedelta(minutes=20))
        module.requests.get = make_get([
            {"date": "2026-09-01", "value": "3"},
        ])
        b_first = collector._fetch_fred_series("B")

        module.datetime = SimpleNamespace(now=lambda: base + timedelta(minutes=31))
        module.requests.get = make_get([
            {"date": "2026-11-01", "value": "2"},
        ])
        refreshed = collector._fetch_fred_series("A")
        expected = a_refreshed
        b_second = collector._fetch_fred_series("B")
    finally:
        module.datetime = old_datetime
        module.requests.get = old_get

    assert refreshed == expected
    assert b_second == b_expected
    assert collector._fred_cache_time.get("A_15") != collector._fred_cache_time.get("B_15")
