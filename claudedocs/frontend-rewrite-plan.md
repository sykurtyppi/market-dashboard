# Frontend Rewrite Plan — Market Risk Dashboard

**Goal:** Replace the Streamlit UI with a decoupled, product-grade frontend: an
HTTP API over the existing Python backend, plus a custom SPA that is clean,
professional, well-layered, easy to navigate, emoji-free, and something other
people would actually use.

**Non-goals:** Rewriting the data layer. The 26 collectors, 11 processors, and
SQLite store stay as-is — they work and are now tested. This is a presentation +
API project, not a backend project.

---

## 1. Why rewrite (not redesign in place)

The backend is well-structured and decoupled enough to serve an API. The UI is
the weak point:

- Emoji used as UI vocabulary throughout (`📊`, `🎯`, `💰`, `🌙`, `✅/⚠️/🚨`).
- Generic Streamlit layout: vertical stacks of `st.subheader` + `st.columns`,
  gray metric-cards, one default blue. No hierarchy, no depth, no editorial layout.
- 18 flat radio buttons for navigation — not scannable.
- Hard ceiling on layout/interaction control; always reads as "Streamlit".

A custom frontend removes the ceiling and gives full control over hierarchy,
navigation, typography, and chart presentation.

---

## 2. Target architecture

```
┌─────────────────────┐     HTTPS/JSON      ┌──────────────────────┐
│  Next.js frontend   │  ───────────────►   │   FastAPI service     │
│  (Vercel)           │  ◄───────────────   │   (Render/Railway)    │
│  - grouped nav      │                     │   - reuses collectors │
│  - design system    │                     │   - reads SQLite      │
│  - Plotly charts    │                     │   - triggers refresh  │
└─────────────────────┘                     └──────────┬───────────┘
                                                        │
                                       ┌────────────────┴───────────────┐
                                       │  Existing backend (unchanged)   │
                                       │  collectors · processors · DB   │
                                       │  scheduler/daily_update.py      │
                                       └─────────────────────────────────┘
```

**Data flow principle:** the API serves *cached* data from SQLite for speed and
reliability (the collectors are slow and rate-limited — never call them live per
request). A background job (the existing `scheduler/daily_update.py`, run on a
timer) refreshes the DB. A `POST /api/refresh` endpoint triggers an on-demand
update, mirroring today's "Update Data" button.

---

## 3. Tech stack

| Layer | Choice | Rationale |
|---|---|---|
| API | **FastAPI** (Python) | Imports the existing collectors directly; async; auto OpenAPI docs. Pydantic response models added in Phase 0. |
| Frontend | **Next.js (App Router) + TypeScript** | Mature, great DX, easy Vercel deploy, file-based routing maps cleanly to pages. |
| Styling | **Tailwind CSS + CSS variables** | Design tokens as CSS vars; utility classes for speed; no component-library default look. |
| Data fetching | Server-Component `fetch` (Phase 0) → **TanStack Query** (Phase 1+) | Phase 0 fetches server-side with `cache: "no-store"`. TanStack Query comes with the first interactive/client feature (the Refresh button), for background revalidation and loading/error states. |
| Charts | Hand-built SVG (Phase 0) → **Plotly.js** (Phase 1+) | Phase 0 renders lightweight themed SVG area/line charts. Plotly (ported from the existing figures) comes as chart richness grows — kept out of Phase 0 to keep the bundle small while validating the design. |
| Fonts | **`geist` package (self-hosted)** | Geist Sans/Mono bundled locally — no build-time Google Fonts fetch, so builds work in offline/locked-down CI. |
| Icons | **Inline SVG** (→ Lucide) | Line icons replacing all emoji; status shown via semantic color dots, not glyphs. Lucide can standardize the set later. |

---

## 4. API surface

One endpoint per page, each returning exactly what that page renders. All read
from SQLite except where noted (live). Payloads finalized during Phase 0.

| Endpoint | Backs page | Source |
|---|---|---|
| `GET /api/overview` | Overview | `daily_snapshots` + composite risk + alerts |
| `GET /api/left-strategy` | LEFT Strategy | `left_strategy` processor + `signals` |
| `GET /api/sentiment` | Sentiment | fear/greed + put/call (snapshot) |
| `GET /api/credit-liquidity` | Credit & Liquidity | `liquidity_history`, `fed_balance_sheet`, analyzers |
| `GET /api/volatility` | Volatility & VRP | `vrp_data` (current + history) |
| `GET /api/sectors` | Sectors & VIX | sector perf (live) + VIX term structure (live) |
| `GET /api/breadth` | Market Breadth | `breadth_history` + McClellan |
| `GET /api/treasury-stress` | Treasury Stress | `move_index` + regime |
| `GET /api/repo` | Repo Market | `repo_market` + analyzer |
| `GET /api/cot` | COT Positioning | COT collector |
| `GET /api/cta` | CTA Flow | CTA engine |
| `GET /api/institutional` | Institutional Flow | dark pool, insider, auctions |
| `GET /api/economic-calendar` | Economic Calendar | econ calendar collector |
| `GET /api/fed-watch` | Fed Watch | fed watch collector |
| `GET /api/cross-asset` | Cross-Asset | cross-asset collector (live) |
| `GET /api/options-flow` | Options Flow | options flow collector (live) |
| `GET /api/health` | System Health | `HealthCheckSystem` |
| `GET /api/freshness` | global topbar | latest snapshot age (now honest — `created_at`) |
| `POST /api/refresh` | "Update Data" | runs `MarketDataUpdater` (auth-gated) |
| `GET /api/reports/pdf` | PDF export | existing `pdf_generator_v2` server-side |
| `GET/POST /api/settings` | Settings | env/secrets (auth-gated) |

---

## 5. Navigation restructure

The 18 flat items become **6 grouped sections** — the single biggest
navigability win:

- **Overview**
- **Volatility** — Volatility & VRP · Sectors & VIX · Treasury Stress
- **Credit & Rates** — Credit & Liquidity · Repo Market · Fed Watch
- **Positioning** — COT · CTA Flow · Options Flow · Institutional Flow
- **Macro & Breadth** — Cross-Asset · Economic Calendar · Market Breadth · LEFT Strategy · Sentiment
- **System** — Settings · System Health

---

## 6. Design system

Full direction is in the interactive Overview mockup (shared separately). Summary:

- **Direction:** Swiss / data-terminal, modernized — precise, restrained,
  typographic hierarchy, tabular figures. Light theme (professional, premium;
  not the dark-terminal cliché).
- **Palette:** cool near-neutral ground, hairline borders, one restrained
  slate-blue accent. Semantic status colors (muted green / amber / brick) are
  **separate** from the accent and used only to encode state.
- **Type:** grotesk system sans for UI/headings; **monospace for all numeric
  data** (a subject-grounded choice — financial figures read as tabular and
  deliberate). `tabular-nums` everywhere digits align.
- **State without emoji:** small colored dots + text labels ("Fresh", "Stale",
  "Stress"), not glyphs.
- **Charts as first-class:** faint grid, area fills, emphasized endpoints,
  tabular tooltips, consistent color mapping.

---

## 7. Page inventory & phasing

| Phase | Pages | Size |
|---|---|---|
| **0 — Foundation** | API skeleton + design system + **Overview** end-to-end | M |
| **1 — Core markets** | Volatility & VRP · Sectors & VIX · Credit & Liquidity · Market Breadth | L |
| **2 — Rates & macro** | Treasury Stress · Repo Market · Fed Watch · Economic Calendar · Cross-Asset | L |
| **3 — Positioning & flows** | COT · CTA Flow · Options Flow · Institutional Flow · Sentiment · LEFT Strategy | L |
| **4 — System & polish** | Settings (auth) · System Health · PDF export · a11y · responsive · perf | M |
| **5 — Cutover** | Parity check · deploy · retire Streamlit | S |

(Relative sizing S/M/L — not hour estimates. Phase 0 is the proof that de-risks
everything after it.)

---

## 8. Deployment

- **Frontend:** Vercel (Next.js native).
- **API:** Render or Railway (Python service + persistent volume for SQLite).
- **Scheduler:** a cron/worker on the same host running `daily_update.py` on a
  timer (replaces manual refresh; the app currently has 18-week-stale data
  precisely because nothing runs it automatically).
- **Secrets:** move from Streamlit secrets to the API host's env vars. The
  settings/refresh endpoints require an auth token.
- **SQLite note:** fine for a single API instance. If the API ever scales
  horizontally, migrate to Postgres — out of scope now, flagged for later.

---

## 9. Coexistence & cutover

The Streamlit app stays live and untouched on Streamlit Cloud throughout. We cut
over only when the new frontend reaches feature parity (Phase 5). No flag day, no
loss of a working product mid-build.

---

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Scope — 18 pages is a large build | Phased; Phase 0 validates effort before committing to 1–4; shared card/chart primitives keep per-page cost low. |
| Collector latency in the API | Serve from SQLite + background refresh, never live-per-request. |
| Two codebases during transition | Streamlit frozen (bugfix-only) once Phase 0 starts; all new work on the API/frontend. |
| Secrets/settings now need auth | Token-gated settings + refresh endpoints; settings admin-only. |
| Chart parity effort | Reuse existing Plotly figures via a shared themed wrapper; don't rebuild from scratch. |
| Deployment complexity (two services) | Simple managed hosts (Vercel + Render); Dockerize the API. |

---

## 11. Immediate next steps (Phase 0)

1. Approve the design direction (mockup).
2. Scaffold `api/` (FastAPI) — reuse `init_components`; ship `/api/overview`,
   `/api/freshness`, `/api/health`.
3. Scaffold `frontend/` (Next.js + Tailwind) — design tokens, app shell (grouped
   nav + topbar), card/chart primitives.
4. Build the **Overview** page end-to-end against the live API.
5. Review the working POC → confirm effort and roll into Phase 1.
