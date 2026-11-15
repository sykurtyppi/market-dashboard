"""Credit observation identity/date regressions; temporary SQLite only."""
from datetime import datetime
import sqlite3
from types import SimpleNamespace

import pandas as pd
import pytest

from database.db_manager import DatabaseManager
from api import overview_service, pages_service
from api.schemas import Metric
from database.health_check import HealthCheckSystem, HealthStatus


def test_credit_health_uses_observation_date_not_snapshot_date(tmp_path):
    path = str(tmp_path / 'health.db')
    db = DatabaseManager(path)
    db.save_daily_snapshot({'date': '2026-11-13', 'credit_spread_hy': 3.2,
                            'credit_spread_hy_date': '2026-11-12',
                            'credit_spread_hy_source': 'FRED (BAMLH0A0HYM2)'})
    health = HealthCheckSystem(path).check_data_source('Credit Spread (HY)', 'credit_spread_hy')
    assert health.last_update.date().isoformat() == '2026-11-12'


def test_credit_health_unknown_when_observation_date_unknown(tmp_path):
    path = str(tmp_path / 'undated-health.db')
    db = DatabaseManager(path)
    db.save_daily_snapshot({'date': '2026-11-13', 'credit_spread_hy': 3.2})
    health = HealthCheckSystem(path).check_data_source('Credit Spread (HY)', 'credit_spread_hy')
    assert health.status == HealthStatus.UNKNOWN
    assert health.last_update is None


def test_metric_response_schema_preserves_actual_observation_date():
    payload = dict(key='hy', label='HY OAS', value=3.2, unit='%',
                   state='good', source='FRED (BAMLH0A0HYM2)', as_of='2026-11-12')
    assert Metric.model_validate(payload).model_dump()['as_of'] == '2026-11-12'


def test_snapshot_credit_dates_migrate_nullable_and_roundtrip(tmp_path):
    path = str(tmp_path / 'credit.db')
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE daily_snapshots (id INTEGER PRIMARY KEY, date TEXT UNIQUE, credit_spread_hy REAL)')
        conn.execute("INSERT INTO daily_snapshots (date, credit_spread_hy) VALUES ('2026-11-15', 3.2)")
    db = DatabaseManager(path)
    DatabaseManager(path)
    legacy = db.get_latest_snapshot()
    assert legacy['credit_spread_hy_date'] is None
    assert legacy['credit_spread_ig_date'] is None
    # The deliberately minimal legacy fixture lacks unrelated snapshot columns;
    # exercise the full write contract on the normal complete schema.
    db = DatabaseManager(str(tmp_path / 'complete.db'))
    saved, result = db.save_daily_snapshot({
        'date': '2026-11-15', 'credit_spread_hy': 3.2, 'credit_spread_ig': .89,
        'credit_spread_hy_date': '2026-11-12', 'credit_spread_ig_date': '2026-11-11',
    })
    assert saved, result.errors
    latest = db.get_latest_snapshot()
    assert latest['credit_spread_hy_date'] == '2026-11-12'
    assert latest['credit_spread_ig_date'] == '2026-11-11'


@pytest.mark.parametrize('unique', [False, True])
def test_yield_migration_preserves_all_rows_and_occupied_target(tmp_path, unique):
    path = str(tmp_path / 'legacy.db')
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE indicators (id INTEGER PRIMARY KEY, indicator_name TEXT, series_id TEXT, date TEXT, value REAL, created_at TEXT)')
        conn.executemany('INSERT INTO indicators VALUES (?, ?, ?, ?, ?, ?)', [
            (1, 'credit_spread_ig', 'BAMLC0A0CMEY', '2026-11-11', 4.8, 'original'),
            (2, 'credit_spread_ig', 'BAMLC0A0CM', '2026-11-10' if unique else '2026-11-11', .9, 'oas'),
            (3, 'corporate_ig_yield', 'OTHER', '2026-11-11', 7., 'occupied'),
            (4, 'credit_spread_ig', 'BAMLC0A0CMEY', '2026-11-12', 4.9, 'original'),
        ])
        if unique:
            conn.execute('CREATE UNIQUE INDEX idx_indicators_name_date ON indicators (indicator_name, date)')
    DatabaseManager(path)
    with sqlite3.connect(path) as conn:
        rows = conn.execute('SELECT * FROM indicators ORDER BY id').fetchall()
    assert len(rows) == 4
    assert rows[0][1].startswith('corporate_ig_yield')
    assert rows[0][2:] == ('BAMLC0A0CMEY', '2026-11-11', 4.8, 'original')
    assert rows[1][1] == 'credit_spread_ig'
    assert rows[2][1:] == ('corporate_ig_yield', 'OTHER', '2026-11-11', 7., 'occupied')
    assert rows[3][1] == 'corporate_ig_yield'
    DatabaseManager(path)
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT * FROM indicators ORDER BY id').fetchall() == rows


def test_ig_history_excludes_wrong_and_unknown_series_even_after_migration(tmp_path):
    db = DatabaseManager(str(tmp_path / 'history.db'))
    for day, value, series in [(10, .8, 'BAMLC0A0CM'), (11, 4.9, 'BAMLC0A0CMEY'), (12, 4.7, None)]:
        db.save_indicator('credit_spread_ig', datetime(2026, 11, day), value, series)
    hist = db.get_indicator_history('credit_spread_ig', days=100000)
    assert hist['value'].tolist() == [.8]


def fake_db(snapshot, histories=None):
    return SimpleNamespace(
        get_latest_snapshot=lambda **kwargs: snapshot,
        get_latest_vrp=lambda: {}, get_latest_move=lambda: {},
        get_vrp_history=lambda **kwargs: pd.DataFrame(),
        get_indicator_history=lambda name, **kwargs: (histories or {}).get(name, pd.DataFrame()),
        get_fed_balance_sheet_history=lambda **kwargs: pd.DataFrame(),
    )


@pytest.mark.parametrize('service,builder', [(overview_service, overview_service.build_overview), (pages_service, pages_service.build_credit_liquidity)])
@pytest.mark.parametrize('dated', [True, False])
def test_credit_metrics_never_use_global_snapshot_date(monkeypatch, service, builder, dated):
    snapshot = {'date': '2026-11-15', 'credit_spread_hy': 3.2, 'credit_spread_ig': .89,
                'credit_spread_hy_source': 'FRED (BAMLH0A0HYM2)'}
    if dated:
        snapshot.update(credit_spread_hy_date='2026-11-12', credit_spread_ig_date='2026-11-11')
    monkeypatch.setattr(service, 'get_db', lambda: fake_db(snapshot))
    body = builder()
    metrics = {m['key']: m for m in body['metrics']}
    for key, expected in [('hy', '2026-11-12'), ('ig', '2026-11-11')]:
        assert metrics[key]['as_of'] == (expected if dated else None)
        if not dated:
            assert any(key.upper() in warning and 'observation date' in warning for warning in body['warnings'])
    if not dated:
        assert 'BAMLC0A0CM' not in metrics['ig']['source']


def test_credit_page_history_fallback_keeps_source_and_date_together(monkeypatch):
    hist = pd.DataFrame({'date': pd.to_datetime(['2026-11-10', '2026-11-12']), 'value': [3.3, 3.2]})
    monkeypatch.setattr(pages_service, 'get_db', lambda: fake_db(
        {'date': '2026-11-15', 'credit_spread_hy_source': 'Unrelated legacy proxy'},
        {'credit_spread_hy': hist}))
    body = pages_service.build_credit_liquidity()
    metric = next(m for m in body['metrics'] if m['key'] == 'hy')
    assert metric['as_of'] == '2026-11-12'
    assert metric['source'] == 'FRED (BAMLH0A0HYM2)'
    assert metric['value'] == 3.2


@pytest.mark.parametrize('raw', ['', '2026-02-30', '2026-1-1', 20261112])
def test_invalid_optional_credit_dates_drop_with_warning(tmp_path, raw):
    db = DatabaseManager(str(tmp_path / 'dates.db'))
    saved, result = db.save_daily_snapshot({'date': '2026-11-15', 'credit_spread_hy': 3.2,
                                           'credit_spread_hy_date': raw})
    assert saved, result.errors
    assert db.get_latest_snapshot()['credit_spread_hy_date'] is None
    assert any('credit_spread_hy_date' in warning for warning in result.warnings)


@pytest.mark.parametrize('service,builder', [(overview_service, overview_service.build_overview), (pages_service, pages_service.build_credit_liquidity)])
def test_hy_proxy_is_explicitly_warned_even_with_observation_date(monkeypatch, service, builder):
    snapshot = {'date': '2026-11-15', 'credit_spread_hy': 1.09,
                'credit_spread_hy_date': '2026-11-11',
                'credit_spread_hy_source': 'Yahoo Finance (HYG yield − 10Y Treasury proxy)'}
    monkeypatch.setattr(service, 'get_db', lambda: fake_db(snapshot))
    body = builder()
    metric = next(m for m in body['metrics'] if m['key'] == 'hy')
    assert metric['source'] == snapshot['credit_spread_hy_source']
    assert any('HY' in warning and 'proxy' in warning and 'not OAS' in warning for warning in body['warnings'])
