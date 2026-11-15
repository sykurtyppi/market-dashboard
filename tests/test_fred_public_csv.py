"""Public CSV is opt-in, strict, and preserves observed dates and caller schemas."""
import importlib
from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from data_collectors.fred_collector import FREDCollector
from data_collectors.fed_balance_sheet_collector import FedBalanceSheetCollector
from data_collectors.liquidity_collector import LiquidityCollector


@pytest.fixture(autouse=True)
def no_credentials(monkeypatch):
    monkeypatch.delenv('FRED_API_KEY', raising=False)
    monkeypatch.delenv('FRED_USE_PUBLIC_CSV', raising=False)
    monkeypatch.setattr('data_collectors.fred_collector.get_secret', lambda name: None)


def response(text):
    result = Mock()
    result.text = text
    result.raise_for_status.return_value = None
    return result


def transport():
    return importlib.import_module('data_collectors.fred_public_csv')


def test_csv_parses_missing_sorts_clips_limits_and_uses_only_official_endpoint(monkeypatch):
    helper = transport()
    get = Mock(return_value=response('observation_date,RRPONTSYD\n2026-11-13,0\n2026-11-12,.\n2026-11-11,1.25\n2026-10-30,2\n2026-11-16,3\n'))
    monkeypatch.setattr(helper.requests, 'get', get)
    frame = helper.fetch_public_fred_csv('RRPONTSYD', '2026-11-01', '2026-11-15', limit=1)
    assert list(frame.columns) == ['date', 'value']
    assert frame['date'].dt.strftime('%Y-%m-%d').tolist() == ['2026-11-11']
    assert frame['value'].tolist() == [1.25]
    assert get.call_args.args == ('https://fred.stlouisfed.org/graph/fredgraph.csv',)
    assert get.call_args.kwargs == {'params': {'id': 'RRPONTSYD', 'cosd': '2026-11-01', 'coed': '2026-11-15'}, 'timeout': 10, 'allow_redirects': False}


@pytest.mark.parametrize('text', [
    '<html>blocked</html>', '', 'DATE,SOFR\n2026-11-13,4\n',
    'observation_date,OTHER\n2026-11-13,4\n',
    'observation_date,SOFR,OTHER\n2026-11-13,4,5\n',
    'observation_date,SOFR\n2026-11-13,4,5\n',
    'observation_date,SOFR\n2026-11-13\n',
    'observation_date,SOFR\n2026-02-30,4\n',
    'observation_date,SOFR\n2026-1-3,4\n',
    'observation_date,SOFR\n2026-11-13T00:00:00,4\n',
    'observation_date,SOFR\n2026-11-13,NaN\n',
    'observation_date,SOFR\n2026-11-13,inf\n',
    'observation_date,SOFR\n2026-11-13,-inf\n',
    'observation_date,SOFR\n2026-11-13,1e999\n',
    'observation_date,SOFR\n2026-11-13,unknown\n',
    'observation_date,SOFR\n2026-11-13,4\n2026-11-13,5\n',
])
def test_malformed_csv_fails_closed(monkeypatch, text):
    helper = transport()
    monkeypatch.setattr(helper.requests, 'get', Mock(return_value=response(text)))
    frame = helper.fetch_public_fred_csv('SOFR', '2026-11-01', '2026-11-15')
    assert frame.empty
    assert list(frame.columns) == ['date', 'value']


@pytest.mark.parametrize('series,start,end,limit', [
    ('SOFR,OTHER', '2026-11-01', '2026-11-15', 2),
    ('https://example.org', '2026-11-01', '2026-11-15', 2),
    ('SOFR', '2026-11-1', '2026-11-15', 2),
    ('SOFR', '2026-11-15', '2026-11-01', 2),
    ('SOFR', '2026-11-01', 'bad', 2),
    ('SOFR', '2026-11-01', '2026-11-15', 0),
])
def test_invalid_arguments_never_request(monkeypatch, series, start, end, limit):
    helper = transport()
    get = Mock()
    monkeypatch.setattr(helper.requests, 'get', get)
    assert helper.fetch_public_fred_csv(series, start, end, limit=limit).empty
    get.assert_not_called()


@pytest.mark.parametrize('error', [requests.Timeout('sensitive upstream detail'), requests.HTTPError('sensitive upstream detail')])
def test_safe_transport_failure(monkeypatch, caplog, error):
    helper = transport()
    monkeypatch.setattr(helper.requests, 'get', Mock(side_effect=error))
    assert helper.fetch_public_fred_csv('SOFR', '2026-11-01', '2026-11-15').empty
    assert 'sensitive upstream detail' not in caplog.text


def test_graph_export_blank_holiday_is_missing_not_zero(monkeypatch):
    helper = transport()
    monkeypatch.setattr(helper.requests, 'get', Mock(return_value=response(
        'observation_date,RRPONTSYD\n2026-11-10,0.152\n2026-11-11,\n2026-11-12,0.250\n'
    )))
    frame = helper.fetch_public_fred_csv('RRPONTSYD', '2026-11-01', '2026-11-15')
    assert frame['date'].dt.strftime('%Y-%m-%d').tolist() == ['2026-11-10', '2026-11-12']
    assert frame['value'].tolist() == [0.152, 0.25]


def test_disabled_without_explicit_opt_in(monkeypatch):
    get = Mock(side_effect=AssertionError('network not allowed'))
    monkeypatch.setattr(requests, 'get', get)
    with pytest.raises(ValueError, match='FRED API key not found'):
        FREDCollector()
    assert FedBalanceSheetCollector().get_total_assets().empty
    assert LiquidityCollector().get_rrp_on().empty
    get.assert_not_called()


@pytest.mark.parametrize('flag', ['true', '0', 'yes'])
def test_only_literal_one_enables_csv(monkeypatch, flag):
    monkeypatch.setenv('FRED_USE_PUBLIC_CSV', flag)
    with pytest.raises(ValueError):
        FREDCollector()


def test_existing_scheduler_constructors_and_public_schemas(monkeypatch):
    monkeypatch.setenv('FRED_USE_PUBLIC_CSV', '1')
    def get(url, **kwargs):
        series = kwargs['params']['id']
        return response(f'observation_date,{series}\n2026-11-13,0\n')
    mock_get = Mock(side_effect=get)
    monkeypatch.setattr(requests, 'get', mock_get)
    fred = FREDCollector().get_series('SOFR', start_date='2026-11-01')
    assets = FedBalanceSheetCollector().get_total_assets('2026-11-01')
    liquidity = LiquidityCollector()
    rrp = liquidity.get_rrp_on('2026-11-01')
    sofr = liquidity.get_sofr('2026-11-01')
    assert list(fred.columns) == ['date', 'SOFR']
    assert list(assets.columns) == ['date', 'total_assets']
    assert list(rrp.columns) == ['date', 'rrp_on']
    assert list(sofr.columns) == ['date', 'sofr']
    for frame in (fred, assets, rrp, sofr):
        assert frame.iloc[0, 1] == 0
        assert len(frame) == 1
    assert mock_get.call_count == 4


@pytest.mark.parametrize('collector_class,method', [(FREDCollector, 'get_series'), (FedBalanceSheetCollector, '_fetch_series'), (LiquidityCollector, '_fetch_series')])
def test_configured_api_key_keeps_json_path(monkeypatch, collector_class, method):
    monkeypatch.setenv('FRED_USE_PUBLIC_CSV', '1')
    get = Mock(return_value=Mock(json=lambda: {'observations': [{'date': '2026-11-13', 'value': '4'}]}))
    monkeypatch.setattr(requests, 'get', get)
    frame = getattr(collector_class(api_key='test-only'), method)('SOFR', start_date='2026-11-01')
    assert not frame.empty
    assert get.call_args.args[0] == 'https://api.stlouisfed.org/fred/series/observations'
    assert get.call_args.kwargs['params']['api_key'] == 'test-only'


def test_csv_tga_preserves_observed_weekly_dates(monkeypatch):
    monkeypatch.setenv('FRED_USE_PUBLIC_CSV', '1')
    monkeypatch.setattr(requests, 'get', Mock(return_value=response('observation_date,WTREGEN\n2026-11-04,800000\n2026-11-11,900000\n')))
    frame = LiquidityCollector().get_tga('2026-11-01')
    assert frame['date'].dt.strftime('%Y-%m-%d').tolist() == ['2026-11-04', '2026-11-11']
    assert frame['tga'].tolist() == [800000, 900000]


def test_partial_balance_sheet_does_not_invent_reserves_or_require_metadata(monkeypatch):
    monkeypatch.setenv('FRED_USE_PUBLIC_CSV', '1')
    def get(url, **kwargs):
        assert url == 'https://fred.stlouisfed.org/graph/fredgraph.csv'
        series = kwargs['params']['id']
        if series != 'WALCL':
            raise requests.HTTPError('unavailable')
        return response('observation_date,WALCL\n2026-11-04,6500000\n2026-11-11,6501000\n')
    monkeypatch.setattr(requests, 'get', get)
    snapshot = FedBalanceSheetCollector().get_full_snapshot(lookback_days=30)
    assert snapshot['total_assets_billions'] == 6501
    assert snapshot['reserve_balances_billions'] is None
    assert snapshot['latest_date'] == '2026-11-11'
    assert len(snapshot['balance_sheet_df']) == 2


def test_csv_series_failure_is_truthful_partial_status(monkeypatch):
    monkeypatch.setenv('FRED_USE_PUBLIC_CSV', '1')
    monkeypatch.setattr('data_collectors.fred_collector.time.sleep', lambda seconds: None)
    def get(url, **kwargs):
        series = kwargs['params']['id']
        if series.startswith('BAML'):
            return response('<html>restricted</html>')
        return response(f'observation_date,{series}\n2026-11-13,4\n')
    monkeypatch.setattr(requests, 'get', get)
    result = FREDCollector().get_data_with_status()
    assert result['status'] == 'partial'
    assert result['fetched_count'] == 3
    assert result['failed_count'] == 2
    assert result['failed_indicators'] == ['credit_spread_hy', 'credit_spread_ig']
