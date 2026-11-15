# Meridian — Market Risk (frontend)

Custom frontend for the Market Risk Dashboard. Next.js (App Router) +
TypeScript + Tailwind v4, talking to the FastAPI backend in [`../api`](../api).

Replaced the Streamlit UI (removed from the repo in October 2026). Background:
[`../claudedocs/frontend-rewrite-plan.md`](../claudedocs/frontend-rewrite-plan.md).

## Prerequisites

- Node 20.9+ (developed on Node 24; Next.js 16 requires Node >=20.9)
- The Python API running (see below)

## Run locally

**1. Start the API** (from the repo root):

```bash
# venv with requirements.txt installed
venv/bin/uvicorn api.main:app --port 8000
```

**2. Point the frontend at it.** Create `frontend/.env.local`:

```
MARKET_API_URL=http://localhost:8000
```

`MARKET_API_URL` is server-only (the Overview page fetches in a Server
Component). Defaults to `http://localhost:8000` if unset. Use a different port
if 8000 is taken (e.g. `http://localhost:8010`).

**3. Start the frontend:**

```bash
cd frontend
npm install
npm run dev          # http://localhost:3000
```

## Scripts

| Command | Purpose |
|---|---|
| `npm run dev` | Dev server on :3000 |
| `npm run build` | Production build (self-hosted fonts, no network needed) |
| `npm start` | Serve the production build |
| `npx tsc --noEmit` | Type-check |

## Source health

The topbar **Sources** button and `/system-health` read `/api/source-health`, a
same-origin, GET-only proxy for the backend's public `/api/health`. Status follows
the backend's per-source calendar/cadence rules, not the age of the latest daily
snapshot. Healthy Friday data therefore stays healthy over a weekend or holiday
when the backend says it is current; stale/down/unknown sources remain visible.
Snapshot age is shown separately as context, not used as a health verdict.

Expand the badge to filter by status (including **Needs attention**), search names
or backend reasons, inspect last-update dates and elapsed ages, or recheck health.
Rechecking is read-only: no collectors, admin calls or snapshot writes. Checks
refresh every minute while the page is active. Failed checks show **Unavailable**;
previous results, when present, are explicitly last-known rather than current.
System Health remains usable when the snapshot freshness endpoint fails.

Verification:

```bash
npm test
npm run lint
npm run typecheck
npm run build
# Uses a deterministic mock API, never production write endpoints.
PLAYWRIGHT_PORT=3180 PLAYWRIGHT_API_PORT=3181 npm run test:e2e
```

Both Playwright ports are optional (defaults: frontend 3100, mock API 3101).

## Notes

- **Fonts** are self-hosted via the `geist` package (`geist/font/*`), so
  `npm run build` works in offline/locked-down CI without fetching Google Fonts.
- **Data freshness**: the Overview fetch uses `cache: "no-store"` so market data
  is never served stale from Next's cache.
- **Design system** lives in `app/globals.css` (Swiss/data-terminal, light, no
  emoji). Status is encoded with semantic color dots; figures use the mono face.
