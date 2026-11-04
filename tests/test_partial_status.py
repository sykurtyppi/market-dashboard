"""Regression coverage for partial source aggregation.

Bug: get_overall_status returned OK when a source was PARTIAL (including
mixed with OK). The fix makes any PARTIAL source keep the overall status
PARTIAL unless ERROR takes precedence, preserving empty-tracker
UNAVAILABLE behavior and all other precedence rules.
"""
import pytest

from utils.data_status import DataStatus, DataStatusTracker


def _tracker_with(statuses):
    tracker = DataStatusTracker()
    for i, status in enumerate(statuses):
        tracker.update(str(i), status)
    return tracker


@pytest.mark.parametrize("statuses,expected", [
    ([], DataStatus.UNAVAILABLE),
    ([DataStatus.OK], DataStatus.OK),
    ([DataStatus.OK, DataStatus.OK], DataStatus.OK),
    ([DataStatus.PARTIAL], DataStatus.PARTIAL),
    ([DataStatus.OK, DataStatus.PARTIAL], DataStatus.PARTIAL),
    ([DataStatus.PARTIAL, DataStatus.OK], DataStatus.PARTIAL),
    ([DataStatus.OK, DataStatus.OK, DataStatus.PARTIAL], DataStatus.PARTIAL),
    ([DataStatus.ERROR, DataStatus.PARTIAL], DataStatus.ERROR),
    ([DataStatus.PARTIAL, DataStatus.ERROR], DataStatus.ERROR),
    ([DataStatus.ERROR, DataStatus.OK, DataStatus.PARTIAL], DataStatus.ERROR),
    ([DataStatus.UNAVAILABLE, DataStatus.PARTIAL], DataStatus.PARTIAL),
    ([DataStatus.PARTIAL, DataStatus.UNAVAILABLE], DataStatus.PARTIAL),
    ([DataStatus.UNAVAILABLE], DataStatus.PARTIAL),
    ([DataStatus.STALE], DataStatus.PARTIAL),
    ([DataStatus.STALE, DataStatus.OK], DataStatus.PARTIAL),
    ([DataStatus.ESTIMATED], DataStatus.PARTIAL),
    ([DataStatus.ESTIMATED, DataStatus.OK], DataStatus.PARTIAL),
    ([DataStatus.STALE, DataStatus.ESTIMATED, DataStatus.PARTIAL], DataStatus.PARTIAL),
])
def test_aggregate_overall_status(statuses, expected):
    tracker = _tracker_with(statuses)
    assert tracker.get_overall_status() is expected
    # get_summary agrees because it delegates to the aggregator.
    assert tracker.get_summary()["overall_status"] == expected.value


def test_partial_string_update_and_recover_to_ok():
    tracker = DataStatusTracker()
    tracker.update("one", "partial")  # string coerces to DataStatus.PARTIAL
    assert tracker.get_overall_status() is DataStatus.PARTIAL
    # Source update recovers the individual source, and the aggregate recovers too.
    tracker.update("one", DataStatus.OK)
    assert tracker.get_overall_status() is DataStatus.OK


def test_unavailable_source_recovers_when_overwritten():
    tracker = DataStatusTracker()
    tracker.update("a", DataStatus.OK)
    tracker.update("b", DataStatus.UNAVAILABLE)
    assert tracker.get_overall_status() is DataStatus.PARTIAL
    # Overwriting the unavailable source with OK clears the downgrade.
    tracker.update("b", DataStatus.OK)
    assert tracker.get_overall_status() is DataStatus.OK


def test_mixed_partial_ok_summary_status():
    tracker = DataStatusTracker()
    tracker.update("fred", DataStatus.OK)
    tracker.update("cbob", DataStatus.PARTIAL, estimated_fields=["vix3m"])
    summary = tracker.get_summary()
    assert summary["overall_status"] == "partial"
