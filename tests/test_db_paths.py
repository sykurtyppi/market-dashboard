"""Database paths must not drift across cached API/background operations."""
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from database.db_manager import DatabaseManager
from database.health_check import HealthCheckSystem, HealthStatus


def test_relative_database_path_stays_pinned_after_cwd_change(tmp_path, monkeypatch):
    app = tmp_path / "app"
    app.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(app)
    db = DatabaseManager("data/test.db")
    health = HealthCheckSystem("data/test.db")
    assert db.save_daily_snapshot({"date": "2026-11-13", "treasury_10y": 4.2})[0]

    monkeypatch.chdir(elsewhere)
    snapshot = db.get_latest_snapshot()
    assert snapshot is not None
    assert snapshot["treasury_10y"] == 4.2
    assert db.save_daily_snapshot({"date": "2026-11-14", "treasury_10y": 4.3})[0]
    assert health.check_database_connection().status == HealthStatus.HEALTHY
    assert Path(db.db_path) == app / "data/test.db"
    assert not (elsewhere / "data").exists()

    from api import main
    monkeypatch.setattr(main, "get_db", lambda: db)
    with TestClient(main.app) as client:
        response = client.get("/api/freshness")
    assert response.status_code == 200
    assert response.json()["as_of"] == "2026-11-14"


def test_symlinked_relative_database_path_resolves_once(tmp_path, monkeypatch):
    app = tmp_path / "app"
    app.mkdir()
    durable = tmp_path / "data"
    durable.mkdir()
    (app / "data").symlink_to("../data", target_is_directory=True)
    monkeypatch.chdir(app)
    db = DatabaseManager("data/market_data.db")
    monkeypatch.chdir(tmp_path)
    assert db.save_daily_snapshot({"date": "2026-11-13", "treasury_10y": 4.2})[0]
    assert db.db_path == str(durable / "market_data.db")
    with sqlite3.connect(durable / "market_data.db") as conn:
        assert conn.execute("SELECT treasury_10y FROM daily_snapshots").fetchone() == (4.2,)


def test_default_database_is_application_relative_not_launch_cwd(tmp_path, monkeypatch):
    import database.db_manager as db_module
    import database.health_check as health_module

    # Model deployment app/data -> ../data, but never access the real deployment.
    app = tmp_path / "app"
    (app / "database").mkdir(parents=True)
    durable = tmp_path / "data"
    durable.mkdir()
    (app / "data").symlink_to("../data", target_is_directory=True)
    launch = tmp_path / "arbitrary-launch-dir"
    launch.mkdir()
    monkeypatch.setattr(db_module, "__file__", str(app / "database/db_manager.py"))
    monkeypatch.setattr(health_module, "__file__", str(app / "database/health_check.py"))
    monkeypatch.chdir(launch)
    db = DatabaseManager()
    health = HealthCheckSystem()
    assert db.db_path == str(durable / "market_data.db")
    assert health.db_path == db.db_path
    assert health.check_database_connection().status == HealthStatus.HEALTHY
    assert not (launch / "data").exists()
