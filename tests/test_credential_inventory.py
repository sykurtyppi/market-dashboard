"""Credential inventory regression tests.

Verifies that the credential inventory lists only the primary keys that have
consumers, that sample.env exposes only those keys, and that the helper's
get_secret/get_all_secrets contract is preserved without disclosing values.
"""
import re
from pathlib import Path

import pytest

from utils import secrets_helper as secrets

EXPECTED = ('FRED_API_KEY', 'NASDAQ_DATA_LINK_KEY', 'FINRA_API_KEY')
REMOVED = ('ALPHA_VANTAGE_KEY', 'POLYGON_API_KEY', 'NEWS_API_KEY')


def test_expected_secrets_only_consumed_primary_keys_in_order():
    assert secrets.EXPECTED_SECRETS == EXPECTED
    for removed in REMOVED:
        assert removed not in secrets.EXPECTED_SECRETS


def test_sample_env_lists_only_consumed_primary_keys():
    text = Path('sample.env').read_text()
    names = [line.split('=', 1)[0].strip() for line in text.splitlines()
             if line.strip() and not line.lstrip().startswith('#') and '=' in line]
    assert names == list(EXPECTED)
    for removed in REMOVED:
        assert removed not in names


def test_sample_env_documents_quandl_alias():
    text = Path('sample.env').read_text()
    assert 'QUANDL_API_KEY' in text
    assert 'alias' in text.lower()
    assert 'QUANDL_API_KEY=' not in re.sub(r'#.*', '', text)


def test_sample_env_finra_note_not_unverified():
    text = Path('sample.env').read_text()
    assert 'Not yet used' not in text
    assert 'live FINRA authentication has not been verified' in text


def test_presence_schema_and_no_secret_disclosure(monkeypatch):
    for key in (*EXPECTED, *REMOVED, 'ARBITRARY_TEST_KEY'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('FINRA_API_KEY', 'synthetic-finra-token')
    status = secrets.get_all_secrets()
    assert tuple(status) == EXPECTED
    assert status['FINRA_API_KEY'] == {'configured': True, 'source': 'environment'}
    assert all(set(row) == {'configured', 'source'} for row in status.values())
    assert 'synthetic-finra-token' not in repr(status)


def test_placeholder_values_count_as_unset(monkeypatch):
    for key in (*EXPECTED, *REMOVED):
        monkeypatch.delenv(key, raising=False)
    for placeholder in ('', 'YOUR_KEY_HERE', 'your_key_here'):
        monkeypatch.setenv('FRED_API_KEY', placeholder)
        assert secrets.get_secret('FRED_API_KEY') is None
    monkeypatch.setenv('FRED_API_KEY', 'real-fred-value')
    assert secrets.get_secret('FRED_API_KEY') == 'real-fred-value'


def test_default_and_arbitrary_key_compatibility(monkeypatch):
    monkeypatch.delenv('ARBITRARY_TEST_KEY', raising=False)
    assert secrets.get_secret('ARBITRARY_TEST_KEY', 'fallback') == 'fallback'
    monkeypatch.setenv('ARBITRARY_TEST_KEY', 'synthetic')
    assert secrets.get_secret('ARBITRARY_TEST_KEY') == 'synthetic'
