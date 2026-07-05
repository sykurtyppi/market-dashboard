"""Environment bootstrap: dotenv, logging, and Streamlit Cloud secrets sync.

Importing this module has side effects on purpose; import it before any
collector is instantiated so API keys are present in os.environ.
"""
import logging
import os

import streamlit as st
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


def load_streamlit_secrets():
    """Load Streamlit Cloud secrets into environment variables"""
    try:
        if hasattr(st, 'secrets') and len(st.secrets) > 0:
            for key in st.secrets:
                if isinstance(st.secrets[key], str):
                    os.environ[key] = st.secrets[key]
    except Exception:
        pass  # Secrets not available (local dev without secrets.toml)


load_streamlit_secrets()
