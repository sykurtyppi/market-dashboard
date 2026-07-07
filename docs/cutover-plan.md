# Deployment Cutover Plan — Streamlit → FastAPI + Next.js

**Status:** Decisions made (§9); config artifacts committed (`render.yaml`, `.github/workflows/refresh.yml`). Infrastructure not yet provisioned — Phase A–D pending.
**Author:** Claude Code · **Date:** 2026-07-06 · **Revision:** 3 (§9 decisions recorded; deploy artifacts added)

This plan describes how to take the completed custom frontend (16 analysis pages
+ Settings + System Health) from local-only to a deployed stack, keep its data
current, and retire the Streamlit app — without faking data or exposing secrets.

---

## 1. The one architectural constraint that drives everything

The backend **serves a SQLite file** (`data/market_data.db`, ~27 MB) that a
**separate refresh job writes**. For the served pages to reflect a refresh, the
API process and the refresh write must land on the **same persistent
filesystem**.

Three consequences fall out of this, and they shape every hosting decision:

1. **The backend needs a persistent disk.** Ephemeral/serverless hosts
   (Vercel Functions, Render's free web tier, scale-to-zero containers) lose the
   DB on restart. The backend must run on an instance with an attached volume
   mounted where the app expects `data/`.
2. **The refresh must write to that same disk.** A refresh running on a
   *separate* instance (e.g. a provider "cron job" container with its own
   filesystem) writes to a disk the API never reads. The refresh must execute
   **inside the web service process** — which is exactly what the token-gated
   `POST /api/refresh` already does.
3. **The DB is gitignored, so a fresh deploy starts empty.** `data/` and `*.db`
   are in `.gitignore` (correct — 27 MB of data shouldn't be in git). A
   newly-deployed backend has **no data** until it's seeded (§6 Phase B). Pages
   degrade honestly (System Health shows `Down`, data pages show empty/notice)
   until then.

> **Why not just host the backend on Vercel too?** Vercel Functions are
> stateless and ephemeral — no shared writable disk between invocations, no way
> for a scheduled write to persist for the next read. SQLite-as-served-store
> needs a real disk. Frontend on Vercel is fine (it's stateless and only
> proxies to the backend).

---

## 2. Target topology

```
                    ┌──────────────────────────┐
  Browser  ───────► │  Next.js (Vercel)        │   server-only env:
                    │  - 16 pages + 2 admin    │   MARKET_API_URL  ─┐
                    │  - Route handlers proxy  │   MARKET_API_TOKEN │
                    └───────────┬──────────────┘                    │
                                │ same-origin proxy                 │
                                │ (backend URL never hits browser)  │
                                ▼                                   │
                    ┌──────────────────────────┐ ◄─────────────────┘
                    │  FastAPI (Render/Railway) │   env: FRED_API_KEY,
                    │  - reads/writes SQLite    │   MARKET_API_TOKEN,
                    │  + PERSISTENT DISK at     │   FRONTEND_ORIGINS
                    │    data/market_data.db    │
                    └───────────▲──────────────┘
                                │ POST /api/refresh  (X-API-Token)
                                │ then poll status + freshness
                    ┌───────────┴──────────────┐
                    │  GitHub Actions cron      │   repo secrets:
                    │  (scheduled workflow)     │   BACKEND_URL, MARKET_API_TOKEN
                    └──────────────────────────┘
```

---

## 3. Hosting options — honest trade-offs

Frontend is settled: **Vercel** (or any Next.js host). The decision is the
backend + disk.

| Option | Backend + disk | Refresh | Cost reality | Notes |
|---|---|---|---|---|
| **A. Render (recommended)** | Web Service + **Render Disk** mounted at `/opt/render/project/src/data` | GH Actions cron → `POST /api/refresh` | **Persistent disk requires a paid instance** (~$7/mo Starter); free tier has *no* disk and sleeps | Well-trodden path; `render.yaml` can pin disk + env |
| **B. Railway** | Service + **Volume** mounted at `data/` | GH Actions cron → `POST /api/refresh` | Small monthly credit, then usage-based | Single provider can host both; volume attaches to the web service |
| **C. Single VPS (Fly.io / Hetzner / DO)** | Container + mounted volume | Same, or host-local cron | Most control, most ops | Overkill unless you want it |

**Recommendation: A (Render backend) + Vercel frontend.** It matches the
env-var names the code already reads, keeps the refresh writing to the web
service's own disk, and uses GitHub Actions (free, already hosting CI) as the
scheduler. The honest catch: **the persistent disk is a paid tier** — there is
no zero-cost path that also keeps the DB across restarts.

---

## 4. Environment / secrets matrix

Generate a strong `MARKET_API_TOKEN` (e.g. `openssl rand -hex 32`). The **same
value** goes on the backend, the frontend, and the GH Actions cron.

**Backend (Render/Railway service env):**

| Var | Value | Purpose |
|---|---|---|
| `FRED_API_KEY` | real key | FRED data collectors |
| `NASDAQ_DATA_LINK_KEY` | real key (optional) | COT data; degrades gracefully if absent |
| `MARKET_API_TOKEN` | generated secret | gates `POST /api/refresh` **and** `GET /api/settings` |
| `FRONTEND_ORIGINS` | `https://<your-app>.vercel.app` | CORS allowlist (comma-sep) |

**Frontend (Vercel project env — server-only, NOT `NEXT_PUBLIC_*`):**

| Var | Value | Purpose |
|---|---|---|
| `MARKET_API_URL` | `https://<backend-host>` | base URL for server-side fetches |
| `MARKET_API_TOKEN` | same as backend | forwarded as `X-API-Token` for Settings |

**GitHub Actions (repo secrets):**

| Secret | Value |
|---|---|
| `BACKEND_URL` | `https://<backend-host>` |
| `MARKET_API_TOKEN` | same as backend |

> Secrets hygiene: `.streamlit/secrets.toml` and `data/*.db` are already
> gitignored (verified). Only `.example` files are tracked. Do **not** commit
> real keys — set them in each platform's env UI / secret store.

---

## 5. Scheduled refresh design

The scheduler (`python -m scheduler.daily_update`) is **one-shot**:
`run_full_update()` then exit. Two equivalent triggers, both writing to the web
host's persistent disk:

- **Preferred — GitHub Actions cron → `POST /api/refresh`.** A scheduled
  workflow calls the token-gated endpoint; FastAPI runs the update as a
  background task *in the web process*, writing to the mounted disk. The
  `_refresh_lock` already prevents overlapping runs; the endpoint returns
  `already_running` if one is in flight. Free, no always-on worker, reuses what
  we built.
- **Alternative — in-process APScheduler thread.** Runs inside uvicorn; no
  external trigger. Simpler ops but couples refresh lifetime to the web process
  and needs an added dependency. Fall back to this only if the external cron is
  unreliable.

### Verify completion, not just acceptance

`POST /api/refresh` returns **as soon as the background task starts** — a `200`
means "refresh began," not "data updated." A cron that checks only the status
code would report success even if the update later failed, silently serving
stale data. The workflow must therefore **poll to completion and gate on
freshness**. (`/api/refresh/status` and `/api/freshness` are open, non-secret
endpoints — no token needed for the polling steps.)

**Committed as `.github/workflows/refresh.yml`** (twice-daily cadence per §9,
plus a system-health step that fails on `down` and warns on `degraded`). The
sketch below is the original single-run design for reference:

```yaml
name: Scheduled data refresh
on:
  schedule:
    - cron: "30 21 * * 1-5"   # 21:30 UTC weekdays = 16:30 ET (EST) / 17:30 ET (EDT)
  workflow_dispatch: {}
jobs:
  refresh:
    runs-on: ubuntu-latest
    steps:
      - name: Trigger refresh
        run: |
          code=$(curl -s -o /dev/null -w '%{http_code}' -X POST \
            -H "X-API-Token: ${{ secrets.MARKET_API_TOKEN }}" \
            "${{ secrets.BACKEND_URL }}/api/refresh")
          echo "refresh trigger -> HTTP $code"
          test "$code" = "200"   # 401 = bad/missing token; anything non-200 fails here

      - name: Wait for the background refresh to finish
        run: |
          running=true
          for i in $(seq 1 60); do          # up to ~20 min (60 × 20s)
            running=$(curl -s "${{ secrets.BACKEND_URL }}/api/refresh/status" | jq -r '.running')
            echo "poll $i: running=$running"
            [ "$running" = "false" ] && break
            sleep 20
          done
          if [ "$running" != "false" ]; then
            echo "::error::refresh did not complete within timeout"; exit 1
          fi

      - name: Gate on data freshness
        run: |
          body=$(curl -s "${{ secrets.BACKEND_URL }}/api/freshness")
          echo "$body"
          fresh=$(echo "$body" | jq -r '.is_fresh')
          if [ "$fresh" != "true" ]; then
            echo "::error::refresh completed but data is not fresh — update likely failed"; exit 1
          fi
```

> `jq` is preinstalled on `ubuntu-latest`. For a stricter gate, add a step that
> checks `GET /api/system-health` and fails if `overall_status` is `down`.
>
> Timing note: `config/config.yaml` sets `update_time: "16:30"` ET. GitHub cron
> is fixed-UTC and does **not** follow DST, so `21:30 UTC` lands at 16:30 ET in
> winter (EST) but 17:30 ET in summer (EDT). One hour after close is harmless;
> if you want tighter summer alignment use `20:30 UTC`, or just refresh twice
> daily — the `_refresh_lock` makes extra calls safe.

---

## 6. Cutover runbook (execution order — for a later session)

Nothing below is done yet. This is the sequence when you give the go-ahead.

**Phase A — Backend up (no traffic yet)**
1. Provision the backend service + persistent disk mounted at the app's `data/`.
2. Set backend env vars (§4). Start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`.
3. Confirm `GET /api/health` returns 200 (it will report `down`/stale sources — DB not seeded yet; expected and honest).

**Phase B — Seed the database (prefer uploading the existing DB)**

The DB accumulates **daily point-in-time snapshots** that a cold refresh cannot
reconstruct retroactively (yesterday's readings can't be re-fetched). So prefer
seeding with the history you already have:

4. **Preferred — upload the local DB.** Copy the current
   `data/market_data.db` (~27 MB) onto the backend's persistent disk (host file
   manager, `scp`, or a one-off deploy step). This preserves accumulated history
   from day one.
   - Copy it consistently, not mid-write: `sqlite3 data/market_data.db ".backup /tmp/seed.db"` then upload `/tmp/seed.db` as `data/market_data.db`.
5. **Fallback — cold refresh.** If no upload path is available, trigger a
   refresh (§5 / Phase D step 9) to build a fresh DB. Accepts the loss of
   pre-launch history.
6. Verify: `GET /api/freshness` shows fresh, `GET /api/system-health` shows
   healthy sources.

**Phase C — Frontend up**
7. Deploy the Next.js app to Vercel; set `MARKET_API_URL` + `MARKET_API_TOKEN`.
8. Set the backend's `FRONTEND_ORIGINS` to the Vercel URL; redeploy backend.
9. Smoke-test live: Overview + 3–4 data pages render; Settings shows
   `protected: true` (token enforced) with **no** unprotected warning; System
   Health shows healthy sources.

**Phase D — Automate + cut over**
10. Add `.github/workflows/refresh.yml` (§5); set `BACKEND_URL` +
    `MARKET_API_TOKEN` repo secrets; run it once via `workflow_dispatch` to
    confirm the trigger → poll → freshness-gate path works end-to-end.
11. (If a custom domain exists) point it at Vercel.
12. **Retire Streamlit:** take the Streamlit Cloud app **offline but unlisted**
    (do not delete). Keep it recoverable as rollback for several days before
    deleting.

**Rollback:** until Streamlit is deleted, reverting is just re-pointing
users/DNS back to the Streamlit app. The new stack shares no state with it, so
there's no data migration to undo.

---

## 7. DB durability & backup

The served SQLite file is the system's only stateful asset, and part of its
value (daily history) is **not regenerable** by a fresh refresh. Protect it:

- **Before retiring Streamlit:** take a one-time off-host export of the current
  DB — `sqlite3 data/market_data.db ".backup backup-YYYYMMDD.db"` — and store it
  somewhere durable (object storage, or just off the server). This is also the
  seed artifact from Phase B step 4.
- **Ongoing, pick one:**
  - Enable the host's **persistent-disk snapshots** (Render Disks and Railway
    Volumes both support periodic snapshots) — lowest effort.
  - Or add a weekly GitHub Actions job that pulls a `.backup` copy and uploads
    it to object storage (Backblaze B2 / S3) — portable, provider-independent.
- **Restore drill:** confirm once that uploading a `.backup` file to the disk
  and restarting the service brings the data back — an untested backup isn't a
  backup.

---

## 8. Known gotchas & follow-ups (non-blocking)

- **Relative DB path.** `data/market_data.db` is resolved relative to the
  process working directory. The persistent disk must mount at
  `<app-root>/data`, and the service must run from the repo root. Verify the
  mount path matches the working dir on the chosen host.
- **Second SQLite file.** `data/cta_prices.db` also exists; the CTA page uses
  the *cloud* collector (live fetch), so this is a local cache and not required
  on the server, but confirm during Phase B that `/api/cta` works without it.
- **Streamlit still in `requirements.txt`.** `secrets_helper.py` imports
  streamlit lazily inside a `try/except`, so the API runs fine without it — but
  the pinned dep bloats the backend image and slows cold starts. Optional
  follow-up: a `requirements-api.txt` without streamlit for the backend build.
- **Full Streamlit code removal is a separate cleanup.** ~20 files still import
  streamlit. Cutover only means *stop serving* Streamlit; deleting that code is
  its own later phase, not a prerequisite.
- **Node/Python action versions.** CI logs a Node 20 deprecation warning
  (`actions/checkout@v4`, `setup-node@v4`, `setup-python@v5`). Bump to current
  majors when convenient — unrelated to deploy but touches the same workflows.
- **Cost.** The only real recurring cost is the backend's persistent-disk
  instance (~$7/mo on Render Starter, or Railway usage). Vercel + GitHub Actions
  fit their free tiers for this workload.

---

## 9. Decisions (settled 2026-07-07)

1. **Backend host: Render** — Web Service (starter) + Render Disk. Blueprint
   committed as `render.yaml`.
2. **Domain: default Vercel URL for now.** A custom domain can be attached
   later with zero code changes.
3. **Refresh cadence: twice each weekday** — 14:00 UTC (after the open) and
   21:30 UTC (after the close). Workflow committed as
   `.github/workflows/refresh.yml`; the `_refresh_lock` makes extra runs safe.
4. **Streamlit: unlist-and-keep** as rollback for a couple of weeks after the
   new stack is verified live, then delete.

With the artifacts committed, what remains is Phase A–D execution (§6) — the
provisioning steps that need dashboard access: create the Render service from
the blueprint, set the secrets, seed the DB, deploy the Vercel project, add the
two repo secrets, and run the refresh workflow once by hand.
