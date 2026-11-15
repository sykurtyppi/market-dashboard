"""Liquidity output keeps aligned proxy and component observation dates distinct."""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import overview_service, pages_service
from api.main import app
from database.db_manager import DatabaseManager


def test_api_exposes_sparse_liquidity_with_actual_dates(tmp_path, monkeypatch):
    db = DatabaseManager(str(tmp_path / 'liquidity.db'))
    db.save_daily_snapshot({'date': '2026-11-13'})
    db.save_liquidity_history(pd.DataFrame({
        'date': pd.to_datetime(['2026-11-04', '2026-11-11', '2026-11-13']),
        'rrp_on': [1.0, None, 0.325],
        'tga': [900.0, 914.37, None],
        'fed_bs': [6700.0, 6758.867, None],
    }))
    monkeypatch.setattr(overview_service, 'get_db', lambda: db)
    monkeypatch.setattr(pages_service, 'get_db', lambda: db)
    for endpoint in ['/api/overview', '/api/credit-liquidity']:
        response = TestClient(app).get(endpoint)
        assert response.status_code == 200
        metrics = {m['key']: m for m in response.json()['metrics']}
        assert metrics['net_liquidity']['value'] == pytest.approx(5799.0)
        assert metrics['net_liquidity']['as_of'] == '2026-11-04'
        assert 'proxy' in metrics['net_liquidity']['label'].lower()
        assert metrics['rrp_on']['as_of'] == '2026-11-13'
        assert metrics['rrp_on']['value'] == 0.325
        assert metrics['tga']['as_of'] == '2026-11-11'
        assert metrics['tga']['value'] == 914.37
        assert any('weekly' in w.lower() for w in response.json()['warnings'])
        assert metrics['hy']['as_of'] is None
        # Missing optional metadata must not be synthesized by response defaults.
        if 'vix' in metrics:
            assert 'as_of' not in metrics['vix']


def test_liquidity_getter_never_fills_missing_dates(tmp_path):
    db = DatabaseManager(str(tmp_path / 'sparse.db'))
    db.save_liquidity_history(pd.DataFrame({
        'date': pd.to_datetime(['2026-11-04', '2026-11-13']),
        'rrp_on': [1.0, 0.325], 'tga': [900.0, None], 'fed_bs': [6700.0, None],
    }))
    history = db.get_liquidity_history(days=100000)
    assert len(history) == 2
    assert history['net_liquidity'].notna().sum() == 1
