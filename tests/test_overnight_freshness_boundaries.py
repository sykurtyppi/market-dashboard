# Data status regressions.
"""Regression coverage for get_staleness_status freshness boundaries.

Guards the overnight boundary contract in utils/data_status.py:
- fresh:  age_hours < 4h            -> DataStatus.OK
- stale:  4h <= age_hours < 24h     -> DataStatus.STALE  (overnight acceptable)
- unavail: age_hours >= 24h         -> DataStatus.UNAVAILABLE

The thresholds are strict on the lower edge: age == fresh -> STALE, and
age == stale -> UNAVAILABLE. Custom threshold dicts shift both edges.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from utils.data_status import DataStatus, get_staleness_status


@pytest.mark.parametrize(
    "age_hours, expected",
    [
        (0, DataStatus.OK),
        (3.999, DataStatus.OK),      # immediately below the 4h edge -> OK
        (4, DataStatus.STALE),       # exactly at 4h -> STALE (strict <)
        (12, DataStatus.STALE),      # mid-overnight
        (23.999, DataStatus.STALE),  # immediately below the 24h edge -> STALE
        (24, DataStatus.UNAVAILABLE),  # exactly at 24h -> UNAVAILABLE (strict <)
        (48, DataStatus.UNAVAILABLE),
    ],
)
def test_overnight_default_thresholds(age_hours, expected):
    status = get_staleness_status(age_hours)
    assert status == expected, f"default edge {age_hours}h"


@pytest.mark.parametrize(
    "age_hours, thresholds, expected",
    [
        (1.5, {"fresh": 2, "stale": 6}, DataStatus.OK),
        (2.0, {"fresh": 2, "stale": 6}, DataStatus.STALE),
        (5.9, {"fresh": 2, "stale": 6}, DataStatus.STALE),
        (6.0, {"fresh": 2, "stale": 6}, DataStatus.UNAVAILABLE),
        (3.99, {"fresh": 4, "stale": 24}, DataStatus.OK),
        (4.0, {"fresh": 4, "stale": 24}, DataStatus.STALE),
        (24.0, {"fresh": 4, "stale": 24}, DataStatus.UNAVAILABLE),
    ],
)
def test_overnight_custom_thresholds(age_hours, thresholds, expected):
    original_thresholds = dict(thresholds)
    actual = get_staleness_status(age_hours, thresholds)
    assert actual == expected
    assert thresholds == original_thresholds
