# GDP YoY uses four observation intervals; monthly series keep the twelve rule.
from unittest.mock import call, patch

from data_collectors.economic_calendar_collector import EconomicCalendarCollector


def _gdp_quarter_history(current, year_ago):
    """Build 13 quarterly FRED observations, newest first.

    data[4] is the value one year (four quarters) back, so the GDP YoY uses
    four observation intervals rather than twelve.
    """
    values = [current] * 4 + [year_ago] + [100.0] * 8
    observations = []
    for offset, value in enumerate(values):
        total = 2026 * 4 + 2 - offset  # 2026-Q3 down to 2023-Q1
        year = total // 4
        quarter = total % 4 + 1
        month = (quarter - 1) * 3 + 1
        observations.append({"date": f"{year:04d}-{month:02d}-01", "value": value})
    return observations


def _monthly_history(current, year_ago):
    """Build 13 monthly FRED observations so data[12] is the year-ago value."""
    values = [current] * 12 + [year_ago]
    observations = []
    for offset, value in enumerate(values):
        year, month = divmod(2026 * 12 + 10 - offset, 12)
        observations.append({"date": f"{year:04d}-{month + 1:02d}-01", "value": value})
    return observations


def test_gdp_uses_four_quarters_for_yoy():
    collector = EconomicCalendarCollector(fred_api_key="dummy")
    history = _gdp_quarter_history(120.0, 100.0)
    history[12]["value"] = 80.0
    with patch.object(collector, "_fetch_fred_series",
                      return_value=history) as fetch:
        result = collector._get_indicator_data("GDP")
    assert result["yoy_change"] == 20.0
    assert fetch.call_args_list == [call("GDP", observations=15)]


def test_monthly_series_keeps_twelve_interval_rule():
    collector = EconomicCalendarCollector(fred_api_key="dummy")
    history = _monthly_history(100.0, 80.0)
    with patch.object(collector, "_fetch_fred_series",
                      return_value=history) as fetch:
        result = collector._get_indicator_data("CPIAUCSL")
    assert result["yoy_change"] == 25.0
    assert fetch.call_args_list == [call("CPIAUCSL", observations=15)]


def test_gdp_insufficient_quarters_yoy_stays_none():
    collector = EconomicCalendarCollector(fred_api_key="dummy")
    with patch.object(collector, "_fetch_fred_series",
                      return_value=_gdp_quarter_history(120.0, 80.0)[:4]):
        result = collector._get_indicator_data("GDP")
    assert result["yoy_change"] is None


def test_gdp_zero_denominator_yoy_stays_none():
    collector = EconomicCalendarCollector(fred_api_key="dummy")
    with patch.object(collector, "_fetch_fred_series",
                      return_value=_gdp_quarter_history(120.0, 0.0)):
        result = collector._get_indicator_data("GDP")
    assert result["yoy_change"] is None
