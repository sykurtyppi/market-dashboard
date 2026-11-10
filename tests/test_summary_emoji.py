"""Regression tests: populated summary emoji follows the aggregate status."""
from itertools import product

import pytest

from utils.data_status import DataStatus, DataStatusTracker, DataResult

EMOJI = {
    DataStatus.OK: "✅",
    DataStatus.STALE: "⚠️",
    DataStatus.ESTIMATED: "📊",
    DataStatus.UNAVAILABLE: "❌",
    DataStatus.ERROR: "🚨",
    DataStatus.PARTIAL: "⚡",
}


def test_mixed_statuses_first_inserted_order():
    tracker = DataStatusTracker()
    tracker.update("first", DataStatus.OK)
    tracker.update("second", DataStatus.ERROR)
    overall = tracker.get_overall_status()
    summary = tracker.get_summary()
    assert summary["overall_status"] == overall.value
    assert summary["overall_emoji"] == EMOJI[overall]
    assert summary["sources"]["first"]["emoji"] == EMOJI[DataStatus.OK]
    assert summary["sources"]["second"]["emoji"] == EMOJI[DataStatus.ERROR]


def test_mixed_statuses_reversed_insertion_order():
    tracker = DataStatusTracker()
    tracker.update("first", DataStatus.ERROR)
    tracker.update("second", DataStatus.OK)
    overall = tracker.get_overall_status()
    summary = tracker.get_summary()
    assert summary["overall_status"] == overall.value
    assert summary["overall_emoji"] == EMOJI[overall]
    assert summary["sources"]["first"]["emoji"] == EMOJI[DataStatus.ERROR]
    assert summary["sources"]["second"]["emoji"] == EMOJI[DataStatus.OK]


def test_error_source_sets_error_emoji_regardless_of_first_source():
    tracker = DataStatusTracker()
    tracker.update("good", DataStatus.OK)
    tracker.update("bad", DataStatus.ERROR)
    assert tracker.get_summary()["overall_emoji"] == EMOJI[DataStatus.ERROR]


def test_partial_downgraded_summary_uses_partial_emoji():
    tracker = DataStatusTracker()
    tracker.update("a", DataStatus.OK)
    tracker.update("b", DataStatus.STALE)
    assert tracker.get_summary()["overall_emoji"] == EMOJI[DataStatus.PARTIAL]


def test_updates_and_recovers_after_overwrite():
    tracker = DataStatusTracker()
    tracker.update("first", DataStatus.OK)
    tracker.update("second", DataStatus.ERROR)
    assert tracker.get_summary()["overall_emoji"] == EMOJI[DataStatus.ERROR]
    tracker.update("second", "ok")
    assert tracker.get_summary()["overall_emoji"] == EMOJI[DataStatus.OK]


def test_unavailable_source_downgrades_to_partial_emoji():
    tracker = DataStatusTracker()
    tracker.update("a", DataStatus.OK)
    tracker.update("b", DataStatus.UNAVAILABLE)
    assert tracker.get_summary()["overall_emoji"] == EMOJI[DataStatus.PARTIAL]


def test_estimated_source_downgrades_to_partial_emoji():
    tracker = DataStatusTracker()
    tracker.update("a", DataStatus.OK)
    tracker.update("b", DataStatus.ESTIMATED)
    assert tracker.get_summary()["overall_emoji"] == EMOJI[DataStatus.PARTIAL]


def test_all_sources_ok_uses_ok_emoji():
    tracker = DataStatusTracker()
    tracker.update("a", DataStatus.OK)
    tracker.update("b", DataStatus.OK)
    assert tracker.get_summary()["overall_emoji"] == EMOJI[DataStatus.OK]


def test_summary_emoji_follows_aggregate_method_result(monkeypatch):
    tracker = DataStatusTracker()
    tracker.update("first", DataStatus.OK)
    for overall in list(DataStatus):
        monkeypatch.setattr(tracker, "get_overall_status", lambda: overall)
        summary = tracker.get_summary()
        assert summary["overall_status"] == overall.value
        assert summary["overall_emoji"] == EMOJI[overall]


def test_individual_source_emoji_unchanged_by_summary_fix():
    tracker = DataStatusTracker()
    tracker.update("a", DataStatus.OK)
    tracker.update("b", DataStatus.STALE)
    tracker.update("c", DataStatus.ESTIMATED)
    summary = tracker.get_summary()
    for name, status in (("a", DataStatus.OK), ("b", DataStatus.STALE), ("c", DataStatus.ESTIMATED)):
        assert summary["sources"][name]["emoji"] == EMOJI[status]
        assert summary["sources"][name]["status"] == status.value
