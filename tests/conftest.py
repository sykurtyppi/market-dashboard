"""Keep default relative database paths away from the managed live database."""
import pytest


@pytest.fixture(autouse=True)
def isolate_relative_data_paths(tmp_path, monkeypatch):
    # DatabaseManager() and HealthCheckSystem() may initialize/migrate SQLite
    # even for GET requests. The checkout's data directory can be a live symlink.
    monkeypatch.chdir(tmp_path)
    from api.deps import get_db, get_health
    get_db.cache_clear()
    get_health.cache_clear()
    yield
    get_db.cache_clear()
    get_health.cache_clear()
