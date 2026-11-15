# Refresh outcomes and credit/rate provenance

The refresh always expects the `fred_indicators` phase. A missing FRED API key,
an empty result, or a mapping containing only empty frames marks that phase failed.
Independent collection and snapshot writes continue, but the run cannot report a
clean success. Existing CLI/API failed-phase handling receives this failure.
Other optional FRED-dependent collectors retain their existing skip behavior.
This is not a completeness check for every individual FRED series: a nonempty
partial collection still runs with its available observations.

## 10-year Treasury attribution

New snapshots persist `treasury_10y_source` alongside the selected value:
`FRED (DGS10)` when the FRED observation wins, or `Yahoo Finance (^TNX)` when
using Yahoo's fallback. The overview reads that persisted source instead of
assuming every value came from FRED. Old snapshots with a value but no provenance
are labeled `Unknown (legacy snapshot)`; absent values are `Unavailable`.
No historical source is guessed or backfilled. Yahoo cache age is not recorded
by this change; attribution identifies the provider, not observation freshness.

## Legacy IG effective yields are not credit spreads

Repository history (`332b89534065cfcea811f263f1e80be7214ef1f4`) changed the IG
collector from effective-yield series `BAMLC0A0CMEY` to option-adjusted-spread
series `BAMLC0A0CM`. Changing the collector mapping alone cannot repair older
stored observations.

The indicators table stores `series_id`, with one row per indicator/date.
`get_indicator_history("credit_spread_ig")` now admits only the exact OAS series
`BAMLC0A0CM`. Consequently overview and credit-page charts, and the credit-page
headline, cannot consume old yield or unknown-provenance rows as spreads.

Daily snapshots previously stored only the IG numeric value. A nullable additive
`credit_spread_ig_source` column now preserves the actual series used by the
refresh. Latest/recent snapshot readers expose the IG value only with exact
`FRED (BAMLC0A0CM)` provenance. Existing snapshots are not retroactively labeled
from today's collector configuration, value magnitude, or a matching date.
Some valid legacy OAS values will therefore be unavailable until refreshed; this
is intentional rather than risking a yield displayed as a spread.

These guards do not delete or rewrite legacy values: raw rows remain available
for audit in SQLite. Existing database migration/write semantics (including the
older deduplication migration and per-date replacement) are unchanged. No
historical refetch, destructive migration, production DB access, service config
change, or deployment is required to test these guards.

## Stable database paths

DatabaseManager and HealthCheckSystem now anchor their default database to the
application root and resolve it once to an absolute path. Explicit relative paths
keep caller-cwd semantics at construction, then remain pinned. Resolving once
also follows `app/data -> ../data` so later cwd changes cannot redirect reads or
writes. This change does not relocate data or repair missing mounts/permissions.

A temporary-database regression reproduced `sqlite3.OperationalError: unable to
open database file` after changing cwd with a cached relative-path manager; the
fixed test also exercises `/api/freshness` and gets HTTP 200. No scheduler
`chdir` was found in repository code, so this reproduces the failure mechanism
but does not prove the exact trigger of the reported live incident.

## Regression coverage

`tests/test_backend_data_integrity.py` exercises failed FRED outcomes while
independent phases continue; FRED-preferred/Yahoo-fallback/missing/legacy Treasury
attribution through validation, SQLite persistence, and overview assembly;
legacy schema migration idempotence; mixed-source IG history through both page
builders; raw-row preservation; and snapshot provenance allowlisting.

Run with the shared service environment from the isolated worktree:

```sh
/Users/tristanalejandro/services/market-dashboard/venv/bin/python -m pytest -q
```
