"""Backend accessors shared across API routes.

The API instantiates the pure backend objects (database, health check)
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
