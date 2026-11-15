# Data status regressions.
from types import SimpleNamespace

from api.snapshot_service import snapshot_response


def test_invalid_snapshot_timestamp_is_stale():
    db = SimpleNamespace(get_api_snapshot=lambda dataset: {
        "payload": {"positions": [1]},
        "refreshed_at": 12345,
        "refresh_status": "ok",
    })
    result = snapshot_response(db, "options_flow", {}, stale_after_seconds=60)
    assert result["data_age_seconds"] is None
    assert result["stale"] is True
    assert result["positions"] == [1]
