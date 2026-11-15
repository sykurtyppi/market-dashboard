"""Read-only research evidence; fixtures are already USD billions."""
from datetime import date
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import overview_service
from api.main import app

TODAY = date(2026, 11, 15)


def evidence(rows):
    # Local import lets RED identify the missing behavior without collection errors.
    from processors import liquidity_evidence
    return liquidity_evidence.build_liquidity_evidence(pd.DataFrame(rows, dtype=object), today=TODAY)


def holiday_rows():
    return [
        dict(date='2026-11-04', fed_balance_sheet=6700, tga=900, rrp_on=1, net_liquidity=5799),
        dict(date='2026-11-10', rrp_on=0.4),
        dict(date='2026-11-11', fed_balance_sheet=6758.867, tga=914.37),
        dict(date='2026-11-13', rrp_on=0.325),
    ]


def test_holiday_mismatch_preserves_strict_and_matches_only_backward():
    result = evidence(holiday_rows())
    assert result['status'] == 'available'
    assert result['unit'] == 'USD billions'
    assert result['strict'] == {'value': 5799, 'as_of': '2026-11-04'}
    proxy = result['proxy']
    assert proxy['value'] == pytest.approx(5844.097)
    assert proxy['status'] == 'backward_matched'
    assert proxy['as_of'] == '2026-11-11'
    assert proxy['max_rrp_age_days'] == 4
    assert proxy['reason']
    assert [c['key'] for c in proxy['components']] == ['fed_balance', 'tga', 'rrp_on']
    assert proxy['components'][-1]['as_of'] == '2026-11-10'
    assert proxy['components'][-1]['age_days'] == 1
    latest = {c['key']: c for c in result['latest_components']}
    assert latest['rrp_on']['as_of'] == '2026-11-13'
    assert latest['rrp_on']['value'] == 0.325
    assert [c['series'] for c in proxy['components']] == ['WALCL', 'WTREGEN', 'RRPONTSYD']
    notes = ' '.join(result['notes']).lower()
    for phrase in ('weekly average', 'simultaneous', 'vintage', 'publication', 'backtest', '4', 'calendar', 'exploratory', 'lineage', 'unknown', 'trading'):
        assert phrase in notes


@pytest.mark.parametrize('rrp_date,available', [('2026-11-07', True), ('2026-11-06', False), ('2026-11-12', False), ('2026-11-16', False)])
def test_calendar_tolerance_and_no_forward_or_future_rrp(rrp_date, available):
    result = evidence([dict(date='2026-11-11', fed_balance_sheet=100, tga=20), dict(date=rrp_date, rrp_on=5)])
    assert (result['proxy']['value'] is not None) is available
    assert result['proxy']['as_of'] == '2026-11-11'
    assert result['status'] == 'partial'
    if available:
        assert result['proxy']['value'] == 75
        assert result['proxy']['components'][-1]['age_days'] == 4
    else:
        assert result['proxy']['status'] == 'unavailable'
        assert result['proxy']['reason']


def test_latest_target_not_replaced_with_older_complete_proxy():
    result = evidence([holiday_rows()[0], dict(date='2026-11-11', fed_balance_sheet=6758, tga=914)])
    assert result['proxy']['as_of'] == '2026-11-11'
    assert result['proxy']['value'] is None
    assert result['strict']['value'] == 5799


def test_zero_is_valid_exact_value_not_missing():
    result = evidence([dict(date='2026-11-11', fed_balance_sheet=0, tga=0, rrp_on=0, net_liquidity=0)])
    assert result['status'] == 'available'
    assert result['strict']['value'] == result['proxy']['value'] == 0
    assert result['proxy']['status'] == 'exact'
    assert all(c['age_days'] == 0 for c in result['proxy']['components'])


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf'), 'bad', 10**400])
def test_nonfinite_invalid_and_numeric_overflow_excluded(bad):
    result = evidence([dict(date='2026-11-11', fed_balance_sheet=bad, tga=2, rrp_on=1, net_liquidity=bad)])
    assert result['strict'] == {'value': None, 'as_of': None}
    assert result['proxy']['value'] is None
    assert result['proxy']['as_of'] is None
    assert result['latest_components'][0]['value'] is None
    assert result['status'] == 'partial'


def test_arithmetic_overflow_unavailable_not_infinite():
    result = evidence([dict(date='2026-11-11', fed_balance_sheet=1e308, tga=-1e308, rrp_on=0)])
    assert result['proxy']['value'] is None
    assert result['proxy']['status'] == 'unavailable'
    assert 'finite' in result['proxy']['reason'].lower()


@pytest.mark.parametrize('rows', [[], [dict(date='invalid', fed_balance_sheet=1, tga=0, rrp_on=0)], [dict(date='2026-02-30', net_liquidity=1)], [dict(date='2026-11-16', net_liquidity=1, fed_balance_sheet=2)]])
def test_empty_invalid_dates_future_dates_unavailable(rows):
    result = evidence(rows)
    assert result['status'] == 'unavailable'
    assert result['strict'] == {'value': None, 'as_of': None}
    assert result['proxy']['as_of'] is None
    assert len(result['proxy']['components']) == len(result['latest_components']) == 3
    assert all(c['value'] is None for c in result['latest_components'])


@pytest.mark.parametrize('missing', ['fed_balance_sheet', 'tga', 'rrp_on'])
def test_missing_components_are_explicit(missing):
    row = dict(date='2026-11-11', fed_balance_sheet=100, tga=20, rrp_on=1)
    del row[missing]
    result = evidence([row])
    assert result['status'] == 'partial'
    assert len(result['proxy']['components']) == 3
    assert result['proxy']['value'] is None


def test_unsorted_history_and_latest_finite_strict_before_today():
    rows = holiday_rows() + [dict(date='2026-11-14', net_liquidity=float('inf')), dict(date='2026-11-16', net_liquidity=123)]
    assert evidence(rows) == evidence(list(reversed(rows)))
    assert evidence(rows)['strict'] == {'value': 5799, 'as_of': '2026-11-04'}


def test_duplicate_policy_rejects_conflicting_series_date_deterministically():
    rows = [dict(date='2026-11-11', fed_balance_sheet=100, tga=20, rrp_on=1), dict(date='2026-11-11', rrp_on=2)]
    assert evidence(rows) == evidence(list(reversed(rows)))
    assert evidence(rows)['proxy']['value'] is None
    assert evidence([rows[0], rows[0]])['proxy']['value'] == 79


def test_http_schema_keeps_nested_evidence_reuses_history_and_preserves_metric(monkeypatch):
    history = pd.DataFrame(holiday_rows())
    original = history.copy(deep=True)
    getter = Mock(return_value=history)
    db = SimpleNamespace(
        get_latest_snapshot=lambda **kw: {}, get_latest_vrp=lambda: {}, get_latest_move=lambda: {},
        get_vrp_history=lambda **kw: pd.DataFrame(), get_indicator_history=lambda *a, **kw: pd.DataFrame(),
        get_liquidity_history=getter,
    )
    class FixedDate(date):
        @classmethod
        def today(cls):
            return TODAY

    monkeypatch.setattr(overview_service, 'date', FixedDate)
    monkeypatch.setattr(overview_service, 'get_db', lambda: db)
    response = TestClient(app).get('/api/overview')
    assert response.status_code == 200
    body = response.json()
    assert 'liquidity' in body
    assert body['liquidity'] == evidence(holiday_rows())
    getter.assert_called_once_with(days=120)
    pd.testing.assert_frame_equal(history, original)
    strict_metric = next(m for m in body['metrics'] if m['key'] == 'net_liquidity')
    assert strict_metric['value'] == 5799
    assert strict_metric['as_of'] == '2026-11-04'
    schema = app.openapi()['components']['schemas']
    assert 'liquidity' in schema['OverviewResponse']['properties']
    assert 'liquidity' not in schema['OverviewResponse'].get('required', [])
