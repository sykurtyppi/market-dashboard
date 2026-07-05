# Meridian — Market Risk (frontend)

Custom frontend for the Market Risk Dashboard. Next.js (App Router) +
TypeScript + Tailwind v4, talking to the FastAPI backend in [`../api`](../api).

Part of the Phase 0 rewrite that replaces the Streamlit UI. See
[`../claudedocs/frontend-rewrite-plan.md`](../claudedocs/frontend-rewrite-plan.md).

## Prerequisites

- Node 18.18+ (developed on Node 24)
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

## Notes

- **Fonts** are self-hosted via the `geist` package (`geist/font/*`), so
  `npm run build` works in offline/locked-down CI without fetching Google Fonts.
- **Data freshness**: the Overview fetch uses `cache: "no-store"` so market data
  is never served stale from Next's cache.
- **Design system** lives in `app/globals.css` (Swiss/data-terminal, light, no
  emoji). Status is encoded with semantic color dots; figures use the mono face.
