"""Regression coverage for DataResult.to_dict partial-result serialization."""
from datetime import datetime

from utils.data_status import DataResult, DataStatus


def test_overnight_partial_result_to_dict_serializes_fixed_inputs():
    fixed_ts = datetime(2026, 11, 14, 3, 30, 0)
    actual = DataResult(
        data={},
        status=DataStatus.PARTIAL,
        timestamp=fixed_ts,
        source="CBOE",
        message="Some fields estimated",
        age_hours=5.5,
        estimated_fields=["volume", "vix3m"],
        errors=["CBOE offline"],
    ).to_dict()
    expected = {
        "status": "partial",
        "timestamp": "2026-11-14T03:30:00",
        "source": "CBOE",
        "message": "Some fields estimated",
        "age_hours": 5.5,
        "estimated_fields": ["volume", "vix3m"],
        "errors": ["CBOE offline"],
        "is_fresh": False,
        "is_usable": True,
    }
    assert actual == expected
