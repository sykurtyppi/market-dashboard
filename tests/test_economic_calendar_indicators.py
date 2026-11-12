# FRED indicator regressions.
from unittest.mock import patch

from data_collectors.economic_calendar_collector import EconomicCalendarCollector


def _indicator_history(current, year_ago=None):
    """Build 13 monthly FRED observations so data[12] is the year-ago value."""
    if year_ago is None:
        year_ago = current
    values = [current] * 12 + [year_ago]
    return [
        {"date": "2026-11-01", "value": v} if i == 0 else
        {"date": f"2026-{i:02d}-01", "value": v}
        for i, v in enumerate(values)
    ]


def test_zero_year_over_year_change_is_preserved():
    collector = EconomicCalendarCollector()
    with patch.object(collector, "_fetch_fred_series",
                      return_value=_indicator_history(100.0, 100.0)) as fetch:
        result = collector._get_indicator_data("CPIAUCSL")
    # Equal nonzero current and year-ago values must yield a real 0.0, not None.
    assert result["yoy_change"] == 0.0
    assert fetch.return_value is not None
    assert result["current"] == 100.0
    assert result["previous"] == 100.0


def test_insufficient_history_yoy_stays_none():
    collector = EconomicCalendarCollector()
    with patch.object(collector, "_fetch_fred_series",
                      return_value=_indicator_history(100.0)[:11]):
        result = collector._get_indicator_data("CPIAUCSL")
    assert result["yoy_change"] is None


def test_zero_denominator_yoy_stays_none():
    collector = EconomicCalendarCollector()
    with patch.object(collector, "_fetch_fred_series",
                      return_value=_indicator_history(50.0, 0.0)):
        result = collector._get_indicator_data("CPIAUCSL")
    assert result["yoy_change"] is None
