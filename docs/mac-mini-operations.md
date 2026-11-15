# Mac mini operations

## Scope and ownership

The primary deployment is the owner's Mac mini, not Render/Vercel. This runbook records the inspected layout on **November 15, 2026**; it is not an installer and does not authorize a rollout. The [cutover plan](cutover-plan.md) is retained as historical cloud-deployment context, not the current operating procedure.

Do not change production while another worktree is being reviewed. Production currently uses detached commit `1112bae`; the documentation worktree starts from `origin/main` (`06ac362` at inspection). Those are deliberately different. New commits, successful tests and a healthy process do not imply production has been updated.

## Service inventory

Service root: `/Users/tristanalejandro/services/market-dashboard` (called `$ROOT` below).

| Path / component | Responsibility |
| --- | --- |
| `app/` | Production Git checkout; do not run development installs or branch switches here |
| `worktrees/` | Isolated branches; never assume a worktree has production configuration/data |
| `venv/bin/python` | Shared backend environment; observed Python 3.11.15 (README local setup recommends 3.13). Record exact runtime when rebuilding |
| `config/backend.env` | Local backend secrets/settings, sourced by startup wrapper; not committed |
| `config/frontend.env` | Server-side frontend environment, not committed |
| `data/market_data.db` | Persistent production SQLite DB, outside Git |
| `app/data` | Untracked symlink to `../data`; losing/replacing it can silently create a new empty data store |
| `bin/` | Host-owned startup, backup and snapshot wrappers, not tracked by this repository |
| `logs/` | Backend/frontend/backup/snapshot stdout and stderr |
| `backups/` | Host backup wrapper output; retention is owned by that wrapper |

Backend: `127.0.0.1:8765`; frontend: `127.0.0.1:8770`. Tailscale Serve forwards tailnet-only HTTPS port **10004 → backend 8765** and **10005 → frontend 8770**. Do not enable Funnel/public exposure, bind to `0.0.0.0`, publish the private hostname, or add Tailscale OAuth credentials as a workaround for CI connectivity.

`bin/start-backend.zsh` changes into `app`, exports `config/backend.env`, and execs `venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port 8765 --proxy-headers --forwarded-allow-ips 127.0.0.1`. `bin/start-frontend.zsh` changes into `app/frontend`, exports `config/frontend.env`, and execs `/opt/homebrew/bin/node node_modules/next/dist/bin/next start --hostname 127.0.0.1 --port 8770`. The frontend must already have a production build; this runbook does not build or restart it.

### Configuration checks without leaking values

The backend file must have a nonempty literal `MARKET_API_TOKEN`; authenticated refresh/settings/snapshot endpoints depend on it. `FRONTEND_ORIGINS` must match the approved frontend origins. `FRED_API_KEY` is required for FRED-dependent data; Nasdaq Data Link configuration is optional for faster COT access. Use a secret manager for credentials and restrictive local file permissions. Never print/cat env files, use shell tracing around them, paste tokens into commands, or put secrets in `NEXT_PUBLIC_*` variables.

At inspection the backend had `MARKET_API_TOKEN`, `FRONTEND_ORIGINS`, `PYTHONUNBUFFERED`, but **no `FRED_API_KEY`**. No usable FRED key was found locally. The frontend env contained `MARKET_API_URL` and `NODE_ENV`, but no `MARKET_API_TOKEN`; authenticated frontend proxy behavior therefore needs operator review before claiming parity with direct backend authenticated access. Do not fabricate keys or claim FRED completeness because the server returns HTTP 200.

## Scheduling and process ownership

Two mechanisms exist; do not blindly enable both or spawn duplicate processes:

* User LaunchAgents: `ai.hermes.market-dashboard-{backend,frontend,backup,snapshot-sectors,snapshot-options-flow}`. Backend/frontend/snapshot agents were loaded at inspection; backup agent was **not** loaded. Plist presence is not evidence a job is active.
* Enabled Hermes **script-only/no-agent** cron jobs currently keep backend/frontend reachable every minute, invoke authenticated refresh weekdays at **14:00 and 21:30 UTC**, and invoke verified backup daily at **04:15 UTC**. The live backend/frontend listeners had PPID 1; do not assume launchd's reported PID is the current application process.
* Supervisor scripts live in `~/.hermes/scripts/market_dashboard_{backend_supervisor,frontend_supervisor,authenticated_refresh,backup}.sh`. Their HTTP reachability gates are process-availability checks, **not freshness validation**. The refresh script separately checks final outcome, failed phases and freshness.
* launchd snapshot schedules are **local host wall-clock time**, not intrinsically UTC: sectors weekdays **13:15, 17:15, 20:15**; options flow weekdays **14:45, 18:45, 21:45**. Confirm host timezone before translating them to UTC. Each calls `bin/refresh-snapshot.zsh <dataset>`, adds up to 300 seconds of jitter by default, authenticates locally, and requires HTTP 200. **The wrapper does not validate the returned snapshot's ready/degraded/failed-source fields**; review the result and API snapshot payload separately. A schedule does not prove its last run succeeded.
* `bin/backup-database.zsh` uses SQLite CLI `.backup`, checks local and iCloud-path copies with `PRAGMA integrity_check`, and compares SHA-256 hashes. The second copy is under the user's iCloud Drive `Backups/market-dashboard` directory; successful local-path verification is not proof that cloud sync completed. The inspected wrapper has **no retention pruning**. It is separate from the explicit one-shot CLI below.

Inspect loaded agent state with `launchctl print gui/$(id -u)/ai.hermes.market-dashboard-backend` (repeat for the other labels); inspect current listeners with `lsof -nP -iTCP:8765 -sTCP:LISTEN` and port 8770. Read only the relevant logs locally, redacting credentials before sharing. Do not restart/kill a PID based solely on port ownership or create new cron jobs while existing supervisors are active.

GitHub Actions `REFRESH_ENABLED` is intentionally **false**: the hosted runner cannot reach the private backend. Leave it false until an approved connectivity and credential design exists. Do not re-enable the historical workflow or alter existing PRs/issues as part of operations documentation.

## Safe preflight CLI

From any checkout containing this revision:

```bash
ROOT=/Users/tristanalejandro/services/market-dashboard
PY="$ROOT/venv/bin/python"
"$PY" scripts/mac_mini_ops.py --root "$ROOT" check
```

The script uses Python's standard library plus `curl` (available on macOS). It emits one JSON report with boolean checks, never secret values or raw API/error bodies. Exit **0** means all checks passed, **1** means a failed/degraded/unproven check, **2** means invalid CLI arguments. It has no deploy/start/stop/refresh/restore commands.

Checks:

1. Literal, nonempty `MARKET_API_TOKEN` and `FRED_API_KEY` assignments in `config/backend.env`. The file is parsed, **never sourced/executed**; dynamic shell substitutions and malformed assignments fail. This verifies presence only, not upstream credential validity, active-process inheritance, file permissions, CORS correctness or frontend token parity.
2. `GET /api/health`: `overall_status == healthy`, a nonempty source map, and every source healthy. Healthy HTTP transport alone cannot pass.
3. `GET /api/refresh/status`: idle, with a known last run reporting `successful`, an empty failed-phase list and no error. A running refresh or `last_run: null` fails conservatively. Last-run state is in memory and can reset after restart.
4. `GET /api/freshness`: boolean `is_fresh == true`. This endpoint's configured coverage is not proof that every optional source is complete; independently review missing FRED/source coverage and snapshot degradation.
5. Production DB opens in SQLite `mode=ro` (missing files are not created), with full `PRAGMA integrity_check == ok`. This checks structural integrity, not financial-data validity.

`--base-url` accepts only an HTTP loopback origin, with no credentials, redirects or proxy routing. Each HTTP call has a total timeout (default 15 seconds), including trickling bodies; SQLite integrity execution is bounded by a progress handler. The checks are sequential, so the total run can exceed one probe timeout. This CLI does not send a token or call any POST endpoint. `--timeout` accepts up to 600 seconds. Options go **before** `check` / `backup`.

A nonzero preflight on the current host is expected while FRED is unconfigured and/or refresh/freshness is degraded. Diagnose the individual flags; do not weaken the gate or turn missing sources into green checks.

## Explicit online backup and verification

Do **not** copy the live `.db` file with `cp`: committed rows may still be in the WAL. Use the online backup API:

```bash
# Choose a new filename and an already-existing protected destination directory.
"$PY" scripts/mac_mini_ops.py --root "$ROOT" --timeout 120 backup \
  --destination "$ROOT/backups/operator-$(date -u +%Y%m%dT%H%M%SZ).db"
```

Backup is independent of API reachability and secret availability so degraded data can still be preserved. It opens the source read-only without `immutable=1`, captures committed WAL data through SQLite's backup API, writes a private `0600` temporary file in the destination directory, checks the copy's full integrity, fsyncs it, and publishes with a no-overwrite hard link. Existing files/symlinks (including the source itself) are refused; errors remove the incomplete temporary copy. Destination filesystem must support hard links. This tool does not prune backups. A green result guarantees a validated copy was produced, not off-host disaster recovery or power-loss durability of the directory entry.

To independently verify a copy without altering it:

```bash
"$PY" - /path/to/copy.db <<'PY'
from contextlib import closing
from pathlib import Path
import sqlite3, sys
with closing(sqlite3.connect(Path(sys.argv[1]).resolve().as_uri() + '?mode=ro', uri=True)) as db:
    result = db.execute('PRAGMA integrity_check').fetchall()
    print(result)
    raise SystemExit(0 if result == [('ok',)] else 1)
PY
```

Restore is deliberately **manual and out of scope for this CLI**. Before a restore, coordinate downtime, disable all relevant launchd and Hermes restart/refresh/snapshot writers, verify no live SQLite connections, and take a separate verified rollback backup. Never replace the DB beneath running processes or leave old `-wal`/`-shm` sidecars paired with a restored file. Re-establish the data symlink and exact pinned revision/configuration, restart only the selected owner, and verify all health layers before restoring schedules. Rehearse on an isolated DB/alternate ports first.

## Rebuild / recovery checklist (operator approval required)

1. Record the intended Git SHA, runtime versions, package lockfiles and currently active schedulers before changing anything. Back up DB **and** restricted configuration/wrappers/plists/cron definitions to a private off-host store; do not commit these files. Test a copy's integrity. Record Tailscale Serve mappings privately.
2. Recreate the root layout and runtime in isolation. Install backend dependencies from the pinned revision's `requirements.txt` and frontend dependencies from its lockfile; build the frontend for the approved server-side API URL. The shared venv and host wrappers are mutable external dependencies, not fully reproducible from Git alone.
3. Restore approved secrets privately; reconcile frontend token/API origin and backend allowed origins. Restore data outside the checkout, then verify the expected `app/data -> ../data` symlink. Do not run the production wrappers from a review worktree.
4. Recreate wrappers from the behavior described above and privately backed-up originals, then review against the pinned app schema/routes. Restore **one** process owner per service, not every available supervisor. Confirm timezone, scheduling, wrapper authentication and log paths; no script here installs jobs.
5. Validate loopback/backend/frontend and tailnet access without broadening exposure. Run `check`; review each source, refresh completion and snapshot coverage separately. A clean newly initialized DB is not a successful restore. Keep GitHub hosted refresh disabled unless private connectivity has actually been solved.
6. Keep the previous app revision, configuration and verified backup for an approved rollback. Do not switch the production checkout or update the shared venv as a side effect of documentation/testing.

## Regression tests

```bash
"$PY" -m pytest tests/test_mac_mini_ops.py -q \
  --basetemp /Users/tristanalejandro/.hermes/cache/scratch/mac-mini-ops-tests
```

Tests use isolated SQLite databases and loopback HTTP fixtures, including WAL-only committed rows, corrupted/missing DBs, no-clobber publication, private file permissions, missing/dynamic secrets, malformed JSON, degraded HTTP-200 payloads and trickling-response deadlines. The fixture's green data is synthetic test data, not evidence about production sources.
