"""Finite chart payload regressions."""
import json
from types import SimpleNamespace

import pandas as pd

from api import overview_service as overview
from api.pages_service import _aligned_series


def test_aligned_drops_whole_row_on_invalid_present_column():
    # A non-finite value in one present column drops the WHOLE row, before
    # downsampling, so the aligned series share dates index-for-index.
    frame = pd.DataFrame({'date': ['a', 'b', None, 'c'], 'vix': [10, float('inf'), 11, 12],
                          'vrp': [1, 2, 9, 3]})
    result = _aligned_series(frame, ['vix', 'vrp'])
    assert result == {
        'vix': [{'date': 'a', 'value': 10.0}, {'date': 'c', 'value': 12.0}],
        'vrp': [{'date': 'a', 'value': 1.0}, {'date': 'c', 'value': 3.0}],
    }


def test_series_filters_invalid_before_sampling_and_keeps_last_valid():
    # Invalid points are filtered before sampling; a non-positive budget keeps
    # nothing while a budget of one keeps only the latest valid point.
    frame = pd.DataFrame({'date': list('abcdefg'),
                          'value': [float('inf'), 1, float('inf'), 3,
                                    float('inf'), 5, float('inf')]})
    assert overview._series(frame, 'value', max_points=3) == [
        {'date': 'b', 'value': 1.0}, {'date': 'd', 'value': 3.0}, {'date': 'f', 'value': 5.0}]


def test_overview_scaled_breadth_degrades_to_null(monkeypatch):
    # A finite 1e308 breadth overflows on the *100 scale; the detail row must
    # degrade to null instead of emitting an "inf% advancing" note.
    snapshot = {'date': '2026-11-11', 'market_breadth': 1e308}
    db = SimpleNamespace(
        get_latest_snapshot=lambda **kwargs: snapshot,
        get_latest_vrp=lambda: {},
        get_latest_move=lambda: {},
        get_vrp_history=lambda **kwargs: pd.DataFrame(),
        get_indicator_history=lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(overview, 'get_db', lambda: db)
    body = overview.build_overview()
    json.dumps(body, allow_nan=False)
    breadth_row = next(row for row in body['detail'] if row['indicator'] == 'Market Breadth')
    assert breadth_row["value"] is None
    assert breadth_row["state"] == "neutral"
    breadth_component = next(row for row in body["regime"]["components"] if row["key"] == "breadth")
    assert breadth_component["note"] == "n/a"
