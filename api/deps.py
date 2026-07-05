"""Streamlit-free backend accessors shared across API routes.

The dashboard's init_components() couples collector setup to Streamlit
(st.warning/st.error). The API instantiates the same pure backend objects
directly and caches them for the process lifetime.
"""
from functools import lru_cache

from database.db_manager import DatabaseManager
from database.health_check import HealthCheckSystem


@lru_cache(maxsize=1)
def get_db() -> DatabaseManager:
    return DatabaseManager()


@lru_cache(maxsize=1)
def get_health() -> HealthCheckSystem:
    return HealthCheckSystem()
