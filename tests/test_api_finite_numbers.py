"""Regression coverage for the shared _num helper used in Market Dashboard
API payloads: non-finite and overflowing inputs must degrade to None instead of
escaping into strict-JSON-unsafe values.
"""
import math

import numpy as np
import pandas as pd
import pytest

from api.overview_service import _num, _series


@pytest.mark.parametrize('value', [
    float('inf'), float('-inf'),
    'NaN', 'Infinity', '1e999',
    np.float32('nan'), np.float32('inf'),
    10 ** 400, None, 'unavailable',
])
def test_nonfinite_and_overflow_become_none(value):
    assert _num(value) is None


@pytest.mark.parametrize(('value', 'expected'), [
    (0, 0.0),
    ('-2.5', -2.5),
    (1.25, 1.25),
    (True, 1.0),
])
def test_existing_finite_conversions_preserved(value, expected):
    assert _num(value) == expected


def test_finite_numeric_string_control():
    assert _num('3.14159') == 3.14159


def test_history_drops_nonfinite_points_and_stays_strict_json():
    import json

    frame = pd.DataFrame(
        {'date': ['2026-11-01', '2026-11-02', '2026-11-03'],
         'value': [1.25, float('inf'), -2.5]}
    )
    points = _series(frame, 'value')
    assert points == [
        {'date': '2026-11-01', 'value': 1.25},
        {'date': '2026-11-03', 'value': -2.5},
    ]
    json.dumps(points, allow_nan=False)
