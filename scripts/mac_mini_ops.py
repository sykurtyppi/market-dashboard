#!/usr/bin/env python3
"""Read-only Mac mini preflight and explicit online SQLite backup (stdlib + curl)."""
import argparse
from contextlib import closing
import json
import math
import os
from pathlib import Path
import shlex
import sqlite3
import subprocess
import tempfile
import time
from urllib.parse import urlsplit

DEFAULT_ROOT = Path("/Users/tristanalejandro/services/market-dashboard")


def secret_presence(path):
    """Parse literal dotenv assignments only. Never source a shell or echo values."""
    found = {"MARKET_API_TOKEN": False, "FRED_API_KEY": False}
    for line in path.read_text().splitlines():
        line = line.strip().removeprefix("export ")
        key, sep, raw = line.partition("=")
        key = key.strip()
        if sep and key in found:
            try:
                parts = shlex.split(raw, comments=True)
                found[key] = (len(parts) == 1 and bool(parts[0])
                              and not any(c in parts[0] for c in "$`"))
            except ValueError:
                found[key] = False
    return found


def fetch(base_url, endpoint, timeout):
    """curl enforces total wall-clock timeout, including trickling responses."""
    result = subprocess.run(
        ["curl", "--disable", "--silent", "--fail", "--noproxy", "*",
         "--max-time", str(timeout), "--max-filesize", "1048576",
         base_url + endpoint], capture_output=True, timeout=timeout + 1, check=True,
    )
    body = json.loads(result.stdout)
    if not isinstance(body, dict):
        raise ValueError("Expected JSON object")
    return body


def open_database(path, timeout):
    # mode=ro refuses missing DBs; do not use immutable=1 (would ignore live WAL).
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=timeout)
    deadline = time.monotonic() + timeout
    connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
    return connection


def integrity(connection):
    return connection.execute("PRAGMA integrity_check").fetchall() == [("ok",)]


def check(args):
    checks = {}
    try:
        checks.update({"secret_" + key: present for key, present in
                       secret_presence(args.root / "config/backend.env").items()})
    except (OSError, UnicodeError):
        checks["secret_MARKET_API_TOKEN"] = False
        checks["secret_FRED_API_KEY"] = False
    validators = {
        "backend_health": ("/api/health", lambda d: d.get("overall_status") == "healthy"
                           and isinstance(d.get("sources"), dict) and bool(d["sources"])
                           and all(isinstance(v, dict) and v.get("status") == "healthy"
                                   for v in d["sources"].values())),
        "refresh_successful_idle": ("/api/refresh/status", lambda d:
                                    d.get("running") is False
                                    and isinstance(d.get("last_run"), dict)
                                    and d["last_run"].get("outcome") == "successful"
                                    and d["last_run"].get("failed_phases") == []
                                    and d["last_run"].get("error") is None),
        "data_fresh": ("/api/freshness", lambda d: d.get("is_fresh") is True),
    }
    for name, (endpoint, validate) in validators.items():
        try:
            checks[name] = bool(validate(fetch(args.base_url, endpoint, args.timeout)))
        except (OSError, ValueError, subprocess.SubprocessError):
            checks[name] = False
    try:
        with closing(open_database(args.root / "data/market_data.db", args.timeout)) as db:
            checks["sqlite_integrity"] = integrity(db)
    except (OSError, sqlite3.Error):
        checks["sqlite_integrity"] = False
    return {"ok": all(checks.values()), "checks": checks}


def backup(args):
    """Publish only a complete integrity-checked copy; never overwrite a target."""
    destination = args.destination.absolute()
    temporary = None
    try:
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("Destination exists")
        # Parent must already exist. No automatic retention, restore or directory creation.
        fd, name = tempfile.mkstemp(prefix=".mac-mini-backup-", dir=destination.parent)
        temporary = Path(name)
        os.close(fd)  # mkstemp creates mode 0600 regardless of the process umask.
        deadline = time.monotonic() + args.timeout

        def progress(status, remaining, total):
            if time.monotonic() >= deadline:
                raise TimeoutError("Backup deadline exceeded")

        with closing(open_database(args.root / "data/market_data.db", args.timeout)) as source:
            with closing(sqlite3.connect(temporary)) as target:
                source.backup(target, pages=128, progress=progress, sleep=0.05)
                target.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
                if not integrity(target):
                    raise ValueError("Backup integrity failed")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        # Atomic no-clobber publication on the destination filesystem, unlike replace().
        os.link(temporary, destination)
        return {"ok": True, "backup_integrity": True, "destination": str(destination)}
    except (OSError, sqlite3.Error, ValueError):
        # Never print exception strings: filesystem/HTTP errors can contain private data.
        return {"ok": False, "backup_integrity": False,
                "error": "Backup failed; check source, deadline, destination parent and collisions"}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Service root, not checkout")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765", help="Loopback HTTP API")
    parser.add_argument("--timeout", type=float, default=15, help="Seconds per probe / whole backup")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Check literal secrets, API semantics and read-only SQLite integrity")
    command = commands.add_parser("backup", help="Online SQLite backup, independent of API/secret health")
    command.add_argument("--destination", type=Path, required=True, help="New file in an existing directory")
    args = parser.parse_args()
    url = urlsplit(args.base_url)
    if (url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost", "::1")
            or url.username or url.password or url.path not in ("", "/") or url.query or url.fragment):
        parser.error("--base-url must be an HTTP loopback origin without credentials")
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 600:
        parser.error("--timeout must be greater than zero and at most 600")
    args.base_url = args.base_url.rstrip("/")
    report = check(args) if args.command == "check" else backup(args)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
