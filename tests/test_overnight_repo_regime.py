"""Repo-market regime regressions: a missing (NaN) or non-string stress_level
must not leak into the payload as non-JSON or a bogus regime, and a valid
string regime must be preserved along with metrics and charts."""
import json
from types import SimpleNamespace

import pandas as pd

from api import pages_service


def _install_repo_history(monkeypatch, stress_level):
    history = pd.DataFrame({
        'date': ['2026-11-14', '2026-11-13'],
        'sofr': [4.5, 4.4],
        'rrp_on': [100.0, 110.0],
        'stress_level': [stress_level, 'NORMAL'],
    })
    monkeypatch.setattr(pages_service, 'get_db', lambda: SimpleNamespace(
        get_repo_history=lambda **kwargs: history))


def test_repo_missing_regime_preserves_snapshot(monkeypatch):
    _install_repo_history(monkeypatch, float('nan'))
    result = pages_service.build_repo()
    assert result["regime"] is None
    assert result['state'] == 'neutral'
    assert result['charts'] == {
        'sofr_history': [{'date': '2026-11-13', 'value': 4.4}, {'date': '2026-11-14', 'value': 4.5}],
        'rrp_history': [{'date': '2026-11-13', 'value': 110.0}, {'date': '2026-11-14', 'value': 100.0}],
    }
    json.dumps(result, allow_nan=False)


def test_repo_valid_string_regime_preserved(monkeypatch):
    _install_repo_history(monkeypatch, 'AMPLE')
    result = pages_service.build_repo()
    assert result['regime'] == 'AMPLE'
    assert result['state'] == 'good'
    # metrics and charts must survive the fix unchanged.
    keys = [m['key'] for m in result['metrics']]
    assert keys == ['sofr', 'rrp', 'sofr_z']
    assert result['charts']['sofr_history'] == [
        {'date': '2026-11-13', 'value': 4.4}, {'date': '2026-11-14', 'value': 4.5}
    ]
    assert result['charts']['rrp_history'] == [
        {'date': '2026-11-13', 'value': 110.0}, {'date': '2026-11-14', 'value': 100.0}
    ]
    json.dumps(result, allow_nan=False)
