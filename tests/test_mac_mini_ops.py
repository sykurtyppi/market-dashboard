"""Standalone stdlib CLI regression tests; no application imports or live writes."""
import json
import time
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/mac_mini_ops.py"


@pytest.fixture
def site(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "config/backend.env").write_text(
        'MARKET_API_TOKEN="never-print-this-token"\nFRED_API_KEY=never-print-this-key\n'
    )
    with sqlite3.connect(tmp_path / "data/market_data.db") as conn:
        conn.execute("create table sample(value)")
        conn.execute("insert into sample values (42)")
    payloads = {
        "/api/health": {"overall_status": "healthy", "sources": {"database": {"status": "healthy"}}},
        "/api/refresh/status": {"running": False, "last_run": {"outcome": "successful", "failed_phases": [], "error": None}},
        "/api/freshness": {"is_fresh": True},
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            body = payloads[self.path]
            if callable(body):
                body(self)
            else:
                self.wfile.write(body if isinstance(body, bytes) else json.dumps(body).encode())

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield tmp_path, payloads, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


def run(site, *args):
    root, _, url = site
    return subprocess.run([sys.executable, str(SCRIPT), "--root", str(root),
                           "--base-url", url, *args], capture_output=True, text=True, timeout=10)


def test_check_good_and_redacted(site):
    result = run(site, "check")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "never-print" not in result.stdout + result.stderr
    assert json.loads(result.stdout)["ok"] is True


@pytest.mark.parametrize("endpoint,body", [
    ("/api/health", {"overall_status": "stale", "sources": {"database": {"status": "healthy"}}}),
    ("/api/health", {"overall_status": "healthy", "sources": {"database": {"status": "down"}}}),
    ("/api/health", {}),
    ("/api/refresh/status", {"running": False, "last_run": None}),
    ("/api/refresh/status", {"running": True, "last_run": {"outcome": "successful"}}),
    ("/api/refresh/status", {"running": False, "last_run": {"outcome": "partial"}}),
    ("/api/freshness", {"is_fresh": False}),
    ("/api/freshness", {"is_fresh": "true"}),
    ("/api/freshness", b"not json never-print-this-token"),
])
def test_http_200_is_not_success(site, endpoint, body):
    site[1][endpoint] = body
    result = run(site, "check")
    assert result.returncode == 1
    assert "never-print" not in result.stdout + result.stderr


def test_env_is_never_executed(site):
    root = site[0]
    (root / "config/backend.env").write_text(
        f'MARKET_API_TOKEN=$(touch {root / "executed"})\nFRED_API_KEY=""\n'
    )
    result = run(site, "check")
    assert result.returncode == 1
    assert not (root / "executed").exists()


def test_missing_database_not_created(site):
    db = site[0] / "data/market_data.db"
    db.unlink()
    assert run(site, "check").returncode == 1
    assert not db.exists()


def test_corrupt_database(site):
    (site[0] / "data/market_data.db").write_bytes(b"not a database")
    assert run(site, "check").returncode == 1


def test_online_backup_includes_wal_and_is_private(site):
    db = site[0] / "data/market_data.db"
    dest = site[0] / "backup.db"
    with sqlite3.connect(db) as writer:
        writer.execute("pragma journal_mode=wal")
        writer.execute("insert into sample values (43)")
        writer.commit()
        result = run(site, "backup", "--destination", str(dest))
    assert result.returncode == 0, result.stdout + result.stderr
    with sqlite3.connect(dest) as copy:
        assert copy.execute("select value from sample order by value").fetchall() == [(42,), (43,)]
        assert copy.execute("pragma integrity_check").fetchall() == [("ok",)]
    assert dest.stat().st_mode & 0o777 == 0o600


def test_backup_refuses_existing_and_same_path(site):
    db = site[0] / "data/market_data.db"
    before = db.read_bytes()
    assert run(site, "backup", "--destination", str(db)).returncode == 1
    assert db.read_bytes() == before
    assert not list(site[0].glob(".mac-mini-backup-*"))


def test_backup_failure_removes_partial(site):
    (site[0] / "data/market_data.db").write_bytes(b"corrupt")
    dest = site[0] / "backup.db"
    assert run(site, "backup", "--destination", str(dest)).returncode == 1
    assert not dest.exists()
    assert not list(site[0].glob(".mac-mini-backup-*"))


def test_trickling_response_has_total_deadline(site):
    def trickle(handler):
        try:
            for _ in range(50):
                handler.wfile.write(b" ")
                handler.wfile.flush()
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError):
            pass

    site[1]["/api/freshness"] = trickle
    start = time.monotonic()
    result = run(site, "--timeout", "0.2", "check")
    assert result.returncode == 1
    assert time.monotonic() - start < 2


def test_backup_refuses_unrelated_existing_file(site):
    dest = site[0] / "existing.db"
    dest.write_bytes(b"keep-me")
    assert run(site, "backup", "--destination", str(dest)).returncode == 1
    assert dest.read_bytes() == b"keep-me"


def test_backup_works_without_secrets_or_healthy_api(site):
    (site[0] / "config/backend.env").unlink()
    site[1]["/api/health"] = {}
    assert run(site, "backup", "--destination", str(site[0] / "rescue.db")).returncode == 0


def test_remote_probe_rejected(site):
    result = subprocess.run([sys.executable, str(SCRIPT), "--base-url", "https://example.com", "check"],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
