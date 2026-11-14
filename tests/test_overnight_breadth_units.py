from types import SimpleNamespace

import pandas as pd
import pytest

from api import pages_service


@pytest.mark.parametrize("stored,advancing,total,expected", [
    (0.0, 0, 100, 0.0),
    (0.5, 1, 200, 0.5),
    (1.0, 1, 100, 1.0),
    (0.55, 55, 100, 55.0),
    (60.0, 60, 100, 60.0),
    (1.0, 100, 100, 100.0),
    (0.5, None, None, 0.5),
    (1.0, 1, 0, 1.0),
    (0.5, 101, 100, 0.5),
    (0.5, 0.5, 100, 0.5),
])
def test_breadth_chart_and_headline_use_same_percentage(monkeypatch, stored, advancing, total, expected):
    history = pd.DataFrame({
        "date": ["2026-11-10", "2026-11-11"],
        "breadth_pct": [55.0, stored],
        "advancing": [55, advancing],
        "total": [100, total],
    })
    db = SimpleNamespace(get_breadth_history=lambda **kwargs: history)
    monkeypatch.setattr(pages_service, "get_db", lambda: db)
    monkeypatch.setattr(pages_service, "load_constituents", lambda: None)

    result = pages_service.build_breadth()
    assert [point["value"] for point in result["charts"]["breadth_pct"]] == pytest.approx([55.0, expected])
    assert result["metrics"][0]["value"] == pytest.approx(expected)


def test_breadth_builder_filters_then_samples_mixed_units(monkeypatch):
    dates = pd.date_range("2026-06-01", periods=181).strftime("%Y-%m-%d").tolist()
    rows = [(day, 0.55, 55, 100) if i % 2 == 0 else (day, 62.0, 62, 100)
            for i, day in enumerate(dates)]
    rows.insert(15, (None, 99.0, 99, 100))
    rows.insert(30, ("2026-06-29T12:00:00", float("inf"), 55, 100))
    rows[-1] = (dates[-1], 0.60, 60, 100)
    history = pd.DataFrame(rows, columns=["date", "breadth_pct", "advancing", "total"])
    original = history.copy(deep=True)
    monkeypatch.setattr(pages_service, "get_db", lambda: SimpleNamespace(
        get_breadth_history=lambda **kwargs: history))
    monkeypatch.setattr(pages_service, "load_constituents", lambda: None)
    result = pages_service.build_breadth()
    points = result["charts"]["breadth_pct"]
    assert len(points) == 180
    seen = [point["date"] for point in points]
    assert seen == dates[:179] + dates[-1:]
    assert [point["value"] for point in points[:2]] == pytest.approx([55.0, 62.0])
    assert points[-1]["value"] == pytest.approx(60.0)
    assert result["as_of"] == dates[-1]
    metric = next(item for item in result["metrics"] if item["key"] == "breadth_pct")
    assert metric["value"] == pytest.approx(60.0)
    pd.testing.assert_frame_equal(history, original)
