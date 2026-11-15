"""Task-specific regression tests for DarkPoolCollector Authorization header."""
import pandas as pd
import pytest
from data_collectors import dark_pool_collector as module


@pytest.fixture(autouse=True)
def forbid_unmocked_http(monkeypatch):
    """Guarantee no real network call happens in these tests."""
    def forbidden(*args, **kwargs):
        raise AssertionError('Unmocked HTTP is forbidden')
    monkeypatch.setattr(module.requests.sessions.Session, 'request', forbidden)


class _Response:
    """Minimal fake response used for the request-forwarding checks."""

    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _capture_post(monkeypatch, records):
    """Monkeypatch module.requests.post to record (url, kwargs) and return a fake."""
    def post(url, **kwargs):
        records.append((url, kwargs))
        return _Response([{'symbol': 'SPY', 'dark_pool_pct': 30.0}])
    monkeypatch.setattr(module.requests, 'post', post)


def test_helper_only_key_reaches_header(monkeypatch):
    """A key resolved via get_secret (no argument) must reach the Authorization header."""
    seen = []
    monkeypatch.setattr(module, 'get_secret', lambda key: seen.append(key) or 'helper-key')
    collector = module.DarkPoolCollector()
    assert collector.api_key == 'helper-key'
    assert collector.headers['Authorization'] == 'Bearer helper-key'
    assert seen == ['FINRA_API_KEY']


def test_helper_key_from_empty_string_argument(monkeypatch):
    """An empty-string argument is falsy, so the resolved helper key is used."""
    monkeypatch.setattr(module, 'get_secret', lambda key: 'helper-from-empty')
    collector = module.DarkPoolCollector('')
    assert collector.api_key == 'helper-from-empty'
    assert collector.headers.get('Authorization') == 'Bearer helper-from-empty'


def test_explicit_key_retains_precedence(monkeypatch):
    """An explicit non-empty key must not be replaced by the resolved helper key."""
    seen = []
    monkeypatch.setattr(module, 'get_secret', lambda key: seen.append(key) or 'helper-key')
    collector = module.DarkPoolCollector('explicit-key')
    assert collector.api_key == 'explicit-key'
    assert collector.headers['Authorization'] == 'Bearer explicit-key'
    assert seen == []  # helper is never consulted when an explicit key is supplied


def test_missing_key_yields_no_authorization_header(monkeypatch):
    """No argument and no resolved key must yield no Authorization header."""
    seen = []
    monkeypatch.setattr(module, 'get_secret', lambda key: seen.append(key) or None)
    collector = module.DarkPoolCollector()
    assert collector.api_key is None
    assert collector.headers.get('Authorization') is None
    assert collector.headers['Accept'] == 'application/json'
    assert collector.headers['Content-Type'] == 'application/json'
    assert seen == ['FINRA_API_KEY']


def test_resolved_key_reaches_real_request(monkeypatch):
    """The resolved helper key must forward into the request that is actually sent."""
    monkeypatch.setattr(module, 'get_secret', lambda key: 'helper-key')
    records = []
    _capture_post(monkeypatch, records)
    result = module.DarkPoolCollector().get_weekly_ats_data('SPY')
    assert isinstance(result, pd.DataFrame)
    assert len(records) == 1
    url, kwargs = records[0]
    assert url == 'https://api.finra.org/data/group/otcMarket/name/weeklySummary'
    assert kwargs['headers']['Authorization'] == 'Bearer helper-key'
    assert kwargs['timeout'] == 15
    assert kwargs['json'] == {
        'limit': 100,
        'offset': 0,
        'compareFilters': [
            {'fieldName': 'issueSymbolIdentifier', 'fieldValue': 'SPY', 'compareType': 'EQUAL'}
        ],
    }


def test_missing_key_request_has_no_authorization(monkeypatch):
    """With no key at all, the forwarded request must carry no Authorization header."""
    monkeypatch.setattr(module, 'get_secret', lambda key: None)
    records = []
    _capture_post(monkeypatch, records)
    module.DarkPoolCollector().get_weekly_ats_data()
    assert len(records) == 1
    _, kwargs = records[0]
    assert kwargs['headers'].get('Authorization') is None


def test_403_still_falls_back(monkeypatch):
    """A 403 response must still trigger the existing fallback path."""
    monkeypatch.setattr(module, 'get_secret', lambda key: 'helper-key')
    expected = pd.DataFrame([{'symbol': 'FAKE', 'estimated': True}])
    monkeypatch.setattr(module.DarkPoolCollector, '_get_fallback_data', lambda self: expected)

    class _Resp403:
        status_code = 403
    monkeypatch.setattr(module.requests, 'post', lambda *a, **kw: _Resp403())
    assert module.DarkPoolCollector().get_weekly_ats_data() is expected
