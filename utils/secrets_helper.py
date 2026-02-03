"""
Secrets Helper - Unified secret management for local and Streamlit Cloud

Priority order:
1. Streamlit secrets (st.secrets) - for Streamlit Cloud deployment
2. Environment variables (.env file) - for local development
3. None - if secret not found

Usage:
    from utils.secrets_helper import get_secret

    fred_key = get_secret('FRED_API_KEY')
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load .env file for local development
load_dotenv()


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get a secret from Streamlit secrets or environment variables.

    Args:
        key: The secret key name (e.g., 'FRED_API_KEY')
        default: Default value if secret not found

    Returns:
        The secret value or default

    Priority:
        1. Streamlit secrets (for cloud deployment)
        2. Environment variables (for local development)
        3. Default value
    """
    # Try Streamlit secrets first (for cloud deployment)
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and key in st.secrets:
            value = st.secrets[key]
            if value and value not in ['', 'YOUR_KEY_HERE', 'your_key_here']:
                return value
    except (ImportError, Exception):
        # Streamlit not available or not in a Streamlit context
        pass

    # Fall back to environment variables (for local development)
    value = os.getenv(key)
    if value and value not in ['', 'YOUR_KEY_HERE', 'your_key_here']:
        return value

    return default


def get_all_secrets() -> dict:
    """
    Get status of all expected secrets.

    Returns:
        Dict with secret names and their availability status
    """
    expected_secrets = [
        'FRED_API_KEY',
        'NASDAQ_DATA_LINK_KEY',
        'FINRA_API_KEY',
        'ALPHA_VANTAGE_KEY',
        'POLYGON_API_KEY',
        'NEWS_API_KEY',
    ]

    status = {}
    for key in expected_secrets:
        value = get_secret(key)
        status[key] = {
            'configured': value is not None,
            'source': _get_secret_source(key) if value else None
        }

    return status


def _get_secret_source(key: str) -> Optional[str]:
    """Determine where a secret is coming from"""
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and key in st.secrets:
            value = st.secrets[key]
            if value and value not in ['', 'YOUR_KEY_HERE']:
                return 'streamlit_secrets'
    except (ImportError, Exception):
        pass

    if os.getenv(key):
        return 'environment'

    return None


# Convenience functions for common secrets
def get_fred_api_key() -> Optional[str]:
    """Get FRED API key"""
    return get_secret('FRED_API_KEY')


def get_nasdaq_api_key() -> Optional[str]:
    """Get Nasdaq Data Link API key"""
    return get_secret('NASDAQ_DATA_LINK_KEY')


def get_finra_api_key() -> Optional[str]:
    """Get FINRA API key"""
    return get_secret('FINRA_API_KEY')
