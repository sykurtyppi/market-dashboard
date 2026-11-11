"""Finite chart payload regressions."""
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from api import overview_service as overview
from api.signals_service import _snapshot_fallback_series
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


def test_aligned_over_cap_sampling_shares_index():
    # Two columns with different NaN/inf patterns drop the WHOLE row before
    # sampling, so after filtering six valid rows both series pick the same
    # three points and stay index-aligned.
    frame = pd.DataFrame({
        'date': list('abcdefgh'),
        'vix': [0, float('nan'), 2, 3, 4, 5, 6, float('inf')],
        'vrp': [0, 1, 2, 3, 4, 5, 6, 7]})
    result = _aligned_series(frame, ['vix', 'vrp'], max_points=3)
    assert result == {
        'vix': [{'date': 'a', 'value': 0.0}, {'date': 'd', 'value': 3.0},
                {'date': 'g', 'value': 6.0}],
        'vrp': [{'date': 'a', 'value': 0.0}, {'date': 'd', 'value': 3.0},
                {'date': 'g', 'value': 6.0}],
    }


@pytest.mark.parametrize("breadth,expected_label,expected_state,expected_note,finite", [
    (None, "Unavailable", "neutral", "n/a", False),
    (1e308, "Unavailable", "neutral", "n/a", False),
    (0.8, "Healthy", "good", "80% advancing", True),
    (0.2, "Mixed", "warn", "20% advancing", True),
])
def test_overview_breadth_labels(monkeypatch, breadth, expected_label, expected_state, expected_note, finite):
    # Breadth degrades to a null detail (Unavailable) when missing or overflowed,
    # otherwise the regime component maps the percentage to a label and note.
    expected_pct = None if not finite else breadth * 100
    snapshot = {'date': '2026-11-11', 'market_breadth': breadth}
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
    assert breadth_row["value"] == expected_pct
    assert breadth_row["state"] == expected_state
    breadth_component = next(row for row in body["regime"]["components"] if row["key"] == "breadth")
    assert breadth_component["note"] == expected_note
    assert breadth_component["value"] == expected_label


def test_snapshot_fallback_drops_missing_dates_and_keeps_source_priority():
    # Missing dates (None, NaN, NaT and pandas NA) must be dropped so a gap can
    # never become a bogus chart point, while row-level source priority and the
    # input frame are preserved.
    frame = pd.DataFrame({
        "date": pd.Series(
            ["2026-11-06", None, "2026-11-08", float("nan"),
             "2026-11-10", pd.NaT, "2026-11-11", pd.NA],
            dtype=object),
        "cboe_equity_pc": [0.7, 9.0, 0.8, 9.0, 0.9, 9.0, float("inf"), 9.0],
        "spy_put_call": [1.5, 1.5, 1.6, 1.6, 1.7, 1.7, 1.2, 1.8],
        "put_call_ratio": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.3, 2.0],
    })
    expected = [
        {"date": "2026-11-06", "value": 0.7},
        {"date": "2026-11-08", "value": 0.8},
        {"date": "2026-11-10", "value": 0.9},
        {"date": "2026-11-11", "value": 1.2},
    ]
    before = frame.copy(deep=True)
    points = _snapshot_fallback_series(
        frame, ("cboe_equity_pc", "spy_put_call", "put_call_ratio"))
    assert points == expected
    pd.testing.assert_frame_equal(frame, before)
