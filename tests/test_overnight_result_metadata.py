"""Regression coverage for DataResult.to_dict partial-result serialization."""
from datetime import datetime

from utils.data_status import DataResult, DataStatus, DataStatusTracker


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


def test_estimated_warnings_distinguish_named_and_unknown_fields():
    tracker = DataStatusTracker()
    tracker.update(
        "CBOE",
        DataStatus.ESTIMATED,
        estimated_fields=["volume", "vix3m"],
    )
    tracker.update("FRED", DataStatus.ESTIMATED, estimated_fields=[])
    tracker.update("HEALTHY", DataStatus.OK)

    expected = ["CBOE: volume, vix3m estimated", "FRED: some fields estimated"]
    actual = tracker.get_summary()["warnings"]
    assert sorted(actual) == expected


def test_summary_reports_error_and_unavailable_warnings():
    tracker = DataStatusTracker()
    tracker.update("API", DataStatus.ERROR, errors=["timeout", "retry exhausted"])
    tracker.update("FEED", DataStatus.ERROR)
    tracker.update("CACHE", DataStatus.UNAVAILABLE)
    tracker.update("HEALTHY", DataStatus.OK)

    expected = ["API: timeout", "CACHE: Data unavailable", "FEED: Unknown error"]
    warnings = tracker.get_summary()["warnings"]
    assert sorted(warnings) == expected
