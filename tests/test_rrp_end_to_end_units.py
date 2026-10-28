import pandas as pd
import pytest

from api import pages_service
from data_collectors.repo_collector_enhanced import RepoCollector
from database.db_manager import DatabaseManager


def _fred_series(dates, values):
    return pd.DataFrame({"date": dates, "value": values})


def test_rrp_billions_contract_survives_source_storage_and_api(tmp_path, monkeypatch):
    dates = pd.to_datetime(["2026-10-24", "2026-10-27"])
    source_rrp_billions = [87.4, 82.5]

    collector = RepoCollector()

    def fake_fetch(series_id, days_back=730):
        assert days_back == 30
        return {
            "SOFR": _fred_series(dates, [4.28, 4.31]),
            "IORB": pd.DataFrame(),
            # FRED RRPONTSYD is already published in billions of US dollars.
            "RRPONTSYD": _fred_series(dates, source_rrp_billions),
        }[series_id].copy()

    monkeypatch.setattr(collector, "fetch_fred_series", fake_fetch)
    collected = collector.get_repo_history(days_back=30)
    assert collected["rrp_on"].tolist() == pytest.approx(source_rrp_billions)

    db = DatabaseManager(db_path=str(tmp_path / "rrp-units.db"))
    assert db.save_repo_data(collected) is True
    stored = db.get_repo_history(days=365).sort_values("date")
    assert stored["rrp_on"].tolist() == pytest.approx(source_rrp_billions)

    monkeypatch.setattr(pages_service, "get_db", lambda: db)
    payload = pages_service.build_repo()
    rrp_metric = next(metric for metric in payload["metrics"] if metric["key"] == "rrp")

    assert rrp_metric == {
        "key": "rrp",
        "label": "RRP (Overnight)",
        "value": 82.5,
        "unit": "B",
        "state": "neutral",
        "source": "FRED (RRPONTSYD), $B",
    }
    assert payload["charts"]["rrp_history"] == [
        {"date": "2026-10-24", "value": 87.4},
        {"date": "2026-10-27", "value": 82.5},
    ]
