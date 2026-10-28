import pandas as pd
import pytest

from api import pages_service


class _RepoDb:
    def get_repo_history(self, days=365):
        assert days == 365
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-10-24", "2026-10-27"]),
                "sofr": [4.28, 4.31],
                "rrp_on": [87.4, 82.5],
                "sofr_z_score": [-0.2, 0.1],
                "stress_level": ["NORMAL", "NORMAL"],
            }
        )


def test_repo_page_preserves_rrp_billions_contract(monkeypatch):
    monkeypatch.setattr(pages_service, "get_db", lambda: _RepoDb())

    payload = pages_service.build_repo()

    rrp_metric = next(metric for metric in payload["metrics"] if metric["key"] == "rrp")
    assert rrp_metric["value"] == pytest.approx(82.5)
    assert rrp_metric["unit"] == "B"
    assert payload["charts"]["rrp_history"] == [
        {"date": "2026-10-24", "value": 87.4},
        {"date": "2026-10-27", "value": 82.5},
    ]