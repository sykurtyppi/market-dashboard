# Breadth chart mixed-unit normalization regressions.
#
# build_breadth normalizes each breadth_pct chart point the same per-reading way
# as the headline: a value <= 1.0 is a 0-1 share and becomes a 0-100 percent,
# otherwise it is already a percent. A single percentage point must not leave the
# rest of the series as a 0-1 share (the old series-wide "all <= 1.0" guard
# skipped normalization whenever any point was already a percentage).
from types import SimpleNamespace

import pandas as pd
import pytest

from api import pages_service


# Frozen acceptance example: raw mixed-unit inputs and their expected normalized
# chart values. Every case yields [55.0, 60.0] because the per-point rule
# normalizes each reading independently.
_RAW_CASES = [
    [0.55, 60.0],   # ratio then percentage
    [55.0, 0.60],   # percentage then ratio
    [0.55, 0.60],   # two ratios
    [55.0, 60.0],   # two percentages
]


@pytest.mark.parametrize("raw", _RAW_CASES)
def test_breadth_chart_normalizes_mixed_units(monkeypatch, raw):
    history = pd.DataFrame({"date": ["2026-11-10", "2026-11-11"], "breadth_pct": raw})
    db = SimpleNamespace(get_breadth_history=lambda **kwargs: history)
    monkeypatch.setattr(pages_service, "get_db", lambda: db)
    monkeypatch.setattr(pages_service, "load_constituents", lambda: None)

    result = pages_service.build_breadth()
    actual = [p["value"] for p in result["charts"]["breadth_pct"]]
    expected = pytest.approx([55.0, 60.0])
    assert actual == expected  # exact mixed-unit normalization, per point


def test_breadth_builder_filters_then_samples_mixed_units(monkeypatch):
    dates = pd.date_range("2026-06-01", periods=181).strftime("%Y-%m-%d").tolist()
    rows = [(day, 0.55 if i % 2 == 0 else 62.0) for i, day in enumerate(dates)]
    rows.insert(15, (None, 99.0))
    rows.insert(30, ("2026-06-29T12:00:00", float("inf")))
    rows[-1] = (dates[-1], 0.60)
    history = pd.DataFrame(rows, columns=["date", "breadth_pct"])
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
