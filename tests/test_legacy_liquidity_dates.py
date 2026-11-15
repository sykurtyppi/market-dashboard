"""Legacy headline dates obey the evidence cutoff without changing stored data."""
from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import overview_service, pages_service
from api.main import app
from processors.liquidity_evidence import build_liquidity_evidence

TODAY = date(2026, 11, 15)
COLUMNS = ('net_liquidity', 'rrp_on', 'tga')


@pytest.fixture(autouse=True)
def frozen_today(monkeypatch):
    class FixedDate(date):
        @classmethod
        def today(cls):
            return TODAY

    monkeypatch.setattr(overview_service, 'date', FixedDate)


def history_frame(rows):
    return pd.DataFrame(rows, columns=['date', *COLUMNS], dtype=object)


def metrics(history):
    result, _ = overview_service._liquidity_observations(None, history=history)
    return {m['key']: m for m in result}


@pytest.mark.parametrize('bad_date', [
    '2026-11-16', 'invalid', '2026-02-30', '20261115', '2026-W46-7',
    '2026-11-15T00:00:00', None, pd.NaT,
])
def test_ineligible_dates_cannot_replace_latest_legacy_observation(bad_date):
    history = history_frame([
        ['2026-11-13', 5799, 1, 900],
        [bad_date, 9999, 9999, 9999],
    ])
    original = history.copy(deep=True)
    result = metrics(history)
    for column, value in zip(COLUMNS, [5799, 1, 900]):
        assert result[column]['value'] == value
        assert result[column]['as_of'] == '2026-11-13'
    pd.testing.assert_frame_equal(history, original)


@pytest.mark.parametrize('observed', [TODAY, datetime(2026, 11, 15, 23, 59), pd.Timestamp('2026-11-15'), '2026-11-15'])
def test_today_boundary_and_date_types_match_evidence(observed):
    history = history_frame([
        ['2026-11-13', 5799, 1, 900],
        [observed, 5800, 0, None],
        ['2026-11-16', 9999, 9999, 9999],
    ])
    result = metrics(history)
    evidence = build_liquidity_evidence(history, today=TODAY)
    assert {k: result['net_liquidity'][k] for k in ('value', 'as_of')} == evidence['strict']
    assert result['rrp_on']['value'] == 0
    assert result['rrp_on']['as_of'] == '2026-11-15'
    # Each series retains its own prior eligible observation, not another's date.
    assert result['tga']['value'] == 900
    assert result['tga']['as_of'] == '2026-11-13'


def test_only_ineligible_dates_leave_legacy_values_and_dates_unavailable():
    result = metrics(history_frame([['2026-11-16', 1, 2, 3], ['invalid', 4, 5, 6]]))
    assert set(result) == set(COLUMNS)
    assert all(m['value'] is None and m['as_of'] is None for m in result.values())


def test_missing_date_column_is_unavailable_not_an_exception():
    result = metrics(pd.DataFrame({'net_liquidity': [1]}))
    assert all(m['value'] is None and m['as_of'] is None for m in result.values())


def test_current_and_prior_cutoffs_select_only_eligible_rows_and_matching_dates():
    history = history_frame([
        ['2026-11-16', 9999, 9999, 9999],
        ['2026-11-15', 5800, 0, 910],
        ['2026-11-13', 5799, 1, 900],
        ['invalid', 8888, 8888, 8888],
    ])
    for cutoff, expected in [(date(2026, 11, 13), [5799, 1, 900]), (TODAY, [5800, 0, 910])]:
        rows, _ = overview_service._liquidity_observations(None, history=history, today=cutoff)
        result = {m['key']: m for m in rows}
        evidence = build_liquidity_evidence(history, today=cutoff)
        assert [result[c]['value'] for c in COLUMNS] == expected
        assert all(result[c]['as_of'] == cutoff.isoformat() for c in COLUMNS)
        assert {k: result['net_liquidity'][k] for k in ('value', 'as_of')} == evidence['strict']


def test_legacy_duplicate_rows_keep_existing_last_row_semantics():
    result = metrics(history_frame([
        ['2026-11-13', 1, 2, 3], ['2026-11-13', 4, None, 6],
    ]))
    assert [result[c]['value'] for c in COLUMNS] == [4, 2, 6]


@pytest.mark.parametrize('endpoint', ['/api/overview', '/api/credit-liquidity'])
def test_public_response_keeps_legacy_and_evidence_observation_dates_aligned(monkeypatch, endpoint):
    history = history_frame([
        ['2026-11-16', 9999, 9999, 9999],
        ['2026-11-13', 5799, 0, None],
        ['invalid', 8888, 8888, 8888],
        ['2026-11-11', None, 1, 900],
    ])
    original = history.copy(deep=True)
    getter = Mock(return_value=history)
    db = SimpleNamespace(
        get_latest_snapshot=lambda **kw: {'date': '2026-11-15'},
        get_latest_vrp=lambda: {}, get_latest_move=lambda: {},
        get_vrp_history=lambda **kw: pd.DataFrame(),
        get_indicator_history=lambda *a, **kw: pd.DataFrame(),
        get_fed_balance_sheet_history=lambda **kw: pd.DataFrame(),
        get_liquidity_history=getter,
    )
    monkeypatch.setattr(overview_service, 'get_db', lambda: db)
    monkeypatch.setattr(pages_service, 'get_db', lambda: db)
    response = TestClient(app).get(endpoint)
    assert response.status_code == 200
    body = response.json()
    result = {m['key']: m for m in body['metrics']}
    assert (result['net_liquidity']['value'], result['net_liquidity']['as_of']) == (5799, '2026-11-13')
    assert (result['rrp_on']['value'], result['rrp_on']['as_of']) == (0, '2026-11-13')
    assert (result['tga']['value'], result['tga']['as_of']) == (900, '2026-11-11')
    if endpoint == '/api/overview':
        assert body['liquidity']['strict'] == {'value': 5799, 'as_of': '2026-11-13'}
        latest = {c['key']: c for c in body['liquidity']['latest_components']}
        for column in ('rrp_on', 'tga'):
            assert all(result[column][k] == latest[column][k] for k in ('value', 'as_of'))
    getter.assert_called_once_with(days=120)
    pd.testing.assert_frame_equal(history, original)
