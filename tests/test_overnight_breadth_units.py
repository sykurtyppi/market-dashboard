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
