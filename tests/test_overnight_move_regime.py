"""Offline regressions for the overnight MOVE stress-regime payload.

The stored MOVE history row may keep a NaN stress_level (a numpy float64 /
float('nan')). Treat missing or non-string regime values as None before the
existing state and note rules run, so the payload never crashes and a missing
regime serializes as JSON null. A recognized string regime is preserved.
"""
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from api import pages_service


@pytest.fixture(autouse=True)
def _restore_state():
    """Clear shared module state so tests do not leak into one another."""
    yield


@pytest.mark.parametrize("stress", [float("nan"), 123])
def test_invalid_move_regime_preserves_both_chart_series(monkeypatch, stress):
    history = pd.DataFrame({
        "date": ["2026-11-13", "2026-11-12"],
        "move": [95.0, 65.0],
        "percentile": [72.0, 25.0],
        "stress_level": [stress, "LOW"],
    })
    db = SimpleNamespace(get_move_history=lambda **kwargs: history)
    monkeypatch.setattr(pages_service, "get_db", lambda: db)

    result = pages_service.build_treasury_stress()

    assert result["regime"] is None
    assert result["state"] == "neutral"
    assert result["regime_note"] == "Treasury volatility regime"
    assert result["metrics"][0]["value"] == 95.0
    assert result["charts"] == {
        "move_history": [{"date": "2026-11-12", "value": 65.0}, {"date": "2026-11-13", "value": 95.0}],
        "percentile_history": [{"date": "2026-11-12", "value": 25.0}, {"date": "2026-11-13", "value": 72.0}],
    }
    json.dumps(result, allow_nan=False)


def test_recognized_stress_regime_is_preserved(monkeypatch):
    history = pd.DataFrame({
        "date": ["2026-11-13"],
        "move": [65.0],
        "stress_level": ["LOW"],
    })
    db = SimpleNamespace(get_move_history=lambda **kwargs: history)
    monkeypatch.setattr(pages_service, "get_db", lambda: db)

    result = pages_service.build_treasury_stress()

    assert result["regime"] == "LOW"
    assert result["state"] == "good"
    assert result["regime_note"] == "Calm Treasury market — MOVE below 80."
    assert result["metrics"][0]["value"] == 65.0
    json.dumps(result, allow_nan=False)
