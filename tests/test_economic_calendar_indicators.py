# FRED indicator regressions.
from datetime import date
from unittest.mock import call, patch

from data_collectors.economic_calendar_collector import EconomicCalendarCollector


def _indicator_history(current, year_ago=None):
    """Build 13 monthly FRED observations so data[12] is the year-ago value.

    Dates run consecutively monthly, newest first: 2026-11-01 (current) down
    to 2025-11-01 (year-ago).
    """
    if year_ago is None:
        year_ago = current
    values = [current] * 12 + [year_ago]
    observations = []
    for offset, value in enumerate(values):
        year, month = divmod(2026 * 12 + 10 - offset, 12)
        observations.append({"date": date(year, month + 1, 1).isoformat(), "value": value})
    return observations


def test_zero_year_over_year_change_is_preserved():
    collector = EconomicCalendarCollector(fred_api_key="dummy")
    history = _indicator_history(100.0, 100.0)
    with patch.object(collector, "_fetch_fred_series",
                      return_value=history) as fetch:
        result = collector._get_indicator_data("CPIAUCSL")
    # Equal nonzero current and year-ago values must yield a real 0.0, not None.
    assert result["yoy_change"] == 0.0
    assert len(history) == 13
    assert fetch.call_args_list == [call("CPIAUCSL", observations=15)]
    assert result["current"] == 100.0
    assert result["previous"] == 100.0


def test_insufficient_history_yoy_stays_none():
    collector = EconomicCalendarCollector(fred_api_key="dummy")
    with patch.object(collector, "_fetch_fred_series",
                      return_value=_indicator_history(100.0)[:11]):
        result = collector._get_indicator_data("CPIAUCSL")
    assert result["yoy_change"] is None


def test_zero_denominator_yoy_stays_none():
    collector = EconomicCalendarCollector(fred_api_key="dummy")
    with patch.object(collector, "_fetch_fred_series",
                      return_value=_indicator_history(50.0, 0.0)):
        result = collector._get_indicator_data("CPIAUCSL")
    assert result["yoy_change"] is None


def test_positive_year_over_year_change_is_computed():
    collector = EconomicCalendarCollector(fred_api_key="dummy")
    history = _indicator_history(100.0, 80.0)
    with patch.object(collector, "_fetch_fred_series",
                      return_value=history) as fetch:
        result = collector._get_indicator_data("CPIAUCSL")
    assert result["yoy_change"] == 25.0
    assert len(history) == 13
    assert fetch.call_args_list == [call("CPIAUCSL", observations=15)]
