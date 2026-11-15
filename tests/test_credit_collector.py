"""Credential-free credit fallback must contain actual dated FRED OAS only."""
from datetime import date
import sqlite3
from unittest.mock import Mock

import pytest
import requests

from data_collectors.credit_collector import CreditCollector
from database.db_manager import DatabaseManager


CSV = """observation_date,BAMLH0A0HYM2,BAMLC0A0CM
2026-11-13,.,.
2026-11-12,3.20,0.89
2026-11-11,3.30,.
2026-11-09,3.34,0.91
"""


def response(text=CSV):
    result = Mock()
    result.text = text
    result.raise_for_status.return_value = None
    return result


def fetch(monkeypatch, text=CSV):
    get = Mock(return_value=response(text))
    monkeypatch.setattr("data_collectors.credit_collector.requests.get", get)
    return CreditCollector(), get


def test_no_key_required_and_one_bounded_get_for_exact_oas_series(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    collector, get = fetch(monkeypatch)
    result = collector.get_all_indicators(lookback_days=10, end_date=date(2026, 11, 15))
    assert set(result) == {"credit_spread_hy", "credit_spread_ig"}
    get.assert_called_once_with(
        "https://fred.stlouisfed.org/graph/fredgraph.csv",
        params={"id": "BAMLH0A0HYM2,BAMLC0A0CM", "cosd": "2026-11-05", "coed": "2026-11-15"},
        timeout=(5, 20),
    )


def test_preserves_observation_dates_and_units_without_forward_fill(monkeypatch):
    collector, _ = fetch(monkeypatch)
    result = collector.get_all_indicators(end_date=date(2026, 11, 15))
    hy, ig = result["credit_spread_hy"], result["credit_spread_ig"]
    assert hy.columns.tolist() == ["date", "BAMLH0A0HYM2"]
    assert hy["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-11-09", "2026-11-11", "2026-11-12"]
    assert hy["BAMLH0A0HYM2"].tolist() == [3.34, 3.30, 3.20]
    assert ig["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-11-09", "2026-11-12"]
    assert ig["BAMLC0A0CM"].tolist() == [0.91, 0.89]


def test_filters_provider_ignored_window_bad_dates_and_nonfinite_values(monkeypatch):
    collector, _ = fetch(monkeypatch, """observation_date,BAMLH0A0HYM2,BAMLC0A0CM
2026-10-01,2.9,0.8
2026-11-11,inf,-1
2026-11-12,3.2,0.89
2026-11-16,3.0,0.7
invalid,3.4,0.9
""")
    result = collector.get_all_indicators(lookback_days=10, end_date=date(2026, 11, 15))
    for frame in result.values():
        assert frame["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-11-12"]


def test_missing_one_series_is_partial_not_a_substituted_yield(monkeypatch):
    collector, _ = fetch(monkeypatch, "observation_date,BAMLH0A0HYM2,BAMLC0A0CMEY\n2026-11-12,3.20,4.81\n")
    result = collector.get_all_indicators(end_date=date(2026, 11, 15))
    assert set(result) == {"credit_spread_hy"}


@pytest.mark.parametrize("text", ["<html>blocked</html>", "observation_date,BAMLH0A0HYM2\n2026-11-12,.\n", ""])
def test_invalid_or_empty_response_is_unavailable(monkeypatch, text):
    collector, _ = fetch(monkeypatch, text)
    assert collector.get_all_indicators(end_date=date(2026, 11, 15)) == {}


def test_http_failure_returns_no_fabricated_observations(monkeypatch):
    collector, get = fetch(monkeypatch)
    get.side_effect = requests.Timeout("provider unavailable")
    assert collector.get_all_indicators(end_date=date(2026, 11, 15)) == {}
    assert get.call_count == 1


def test_credit_fallback_roundtrips_real_source_dates_and_identity(tmp_path, monkeypatch):
    collector, _ = fetch(monkeypatch)
    db = DatabaseManager(str(tmp_path / "credit.db"))
    db.save_indicators_batch(collector.get_all_indicators(end_date=date(2026, 11, 15)))
    history = db.get_indicator_history("credit_spread_hy", days=100000)
    latest = history.sort_values("date").iloc[-1]
    assert str(latest["date"])[:10] == "2026-11-12"
    assert latest["value"] == 3.20
    with sqlite3.connect(db.db_path) as connection:
        assert connection.execute(
            "SELECT series_id, date, value FROM indicators "
            "WHERE indicator_name = 'credit_spread_hy' ORDER BY date DESC LIMIT 1"
        ).fetchone() == ("BAMLH0A0HYM2", "2026-11-12", 3.20)


@pytest.mark.parametrize("days", [0, -1, 1096])
def test_history_window_is_bounded_before_network_access(monkeypatch, days):
    collector, get = fetch(monkeypatch)
    with pytest.raises(ValueError, match="lookback_days"):
        collector.get_all_indicators(lookback_days=days)
    get.assert_not_called()
