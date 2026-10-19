"""Secrets helper: API keys from the environment (or a local .env file).

Usage:
    from utils.secrets_helper import get_secret

    fred_key = get_secret('FRED_API_KEY')

Placeholder values left over from a copied sample.env ('', 'YOUR_KEY_HERE',
'your_key_here') count as unset.
"""

import os
from typing import Optional

from dotenv import load_dotenv

# Load .env for local development; deployed hosts set real environment variables.
load_dotenv()

_PLACEHOLDERS = frozenset({'', 'YOUR_KEY_HERE', 'your_key_here'})

EXPECTED_SECRETS = (
    'FRED_API_KEY',
    'NASDAQ_DATA_LINK_KEY',
    'FINRA_API_KEY',
    'ALPHA_VANTAGE_KEY',
    'POLYGON_API_KEY',
    'NEWS_API_KEY',
)


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Return the environment value for `key`, or `default` if it is unset or a placeholder."""
    value = os.getenv(key)
    if value and value not in _PLACEHOLDERS:
        return value
    return default


def get_all_secrets() -> dict:
    """Presence (never the value) of each expected secret, keyed by name.

    Returns {name: {'configured': bool, 'source': 'environment' | None}}.
    """
    status = {}
    for key in EXPECTED_SECRETS:
        configured = get_secret(key) is not None
        status[key] = {
            'configured': configured,
            'source': 'environment' if configured else None,
        }
    return status

