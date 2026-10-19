# Market Risk Dashboard

Dashboard for monitoring cross-asset risk, volatility regime, credit stress, positioning, and liquidity context. A FastAPI backend over a SQLite store, a Next.js frontend, and a refresh pipeline that fills the database from FRED, Yahoo Finance, CBOE and CFTC.

Creator: Tristan Alejandro  
Notice: Not financial advice.

## What It Covers
- Volatility: VIX, VIX9D, VIX3M, VVIX, SKEW, term structure/contango
- Credit: HY and IG spread context, credit ETF behavior
- Positioning: COT extremes, put/call context, sentiment gauges
- Liquidity and rates: Fed balance sheet, repo/SOFR context, treasury stress, Fed Watch
- Breadth and internal market health metrics

## Data Reliability Design
- Data source + delay labels on key metrics
- Per-source freshness judged in business days (`/api/health`), with the grace each source actually needs
- Page payloads carry `degraded` and `warnings`, so fallback inputs are labeled rather than hidden
- Yahoo/CBOE/COT memoization and last-known-good fallback persistence
- Breadth sample reconciled against a tracked S&P 500 constituent list (`config/sp500_constituents.csv`); refresh it with `python scripts/update_sp500_constituents.py` — the breadth page warns when it is stale

## Layout
- `api/` — FastAPI app (`api.main:app`); one service module per page
- `frontend/` — Next.js app; see [frontend/README.md](frontend/README.md)
- `scheduler/daily_update.py` — the refresh pipeline that writes `data/market_data.db`
- `data_collectors/`, `processors/`, `database/`, `utils/` — collectors, signal math, SQLite access
- `scripts/` — maintenance scripts
- `tests/` — pytest suite

## Local Run
1. Create a Python 3.13 virtual environment and install the backend:
```bash
pip install -r requirements.txt
```
2. Add keys. The FRED key powers the credit, liquidity and Fed pages (without it those collectors are disabled with a warning); the Nasdaq Data Link key speeds up COT data. If you kept keys in `.streamlit/secrets.toml` before, move them into `.env`: the app no longer reads that file.
```bash
cp sample.env .env
```
3. Fill the database (several minutes; hits the upstream sources):
```bash
python scheduler/daily_update.py
```
4. Start the API:
```bash
uvicorn api.main:app --port 8000
```
5. Start the frontend (see `frontend/README.md` for `MARKET_API_URL`):
```bash
cd frontend && npm install && npm run dev
```

## Deploy
See [docs/cutover-plan.md](docs/cutover-plan.md): Render backend from `render.yaml`, Vercel frontend, and the GitHub Actions refresh job (`.github/workflows/refresh.yml`) gated by the `REFRESH_ENABLED` repository variable.

## Tests
```bash
pip install pytest pytest-timeout
```
```bash
pytest -q
```
```bash
cd frontend && npm run lint && npx tsc --noEmit
```

## License
Licensed under GNU AGPL v3. See [LICENSE](LICENSE).

## Source
- [https://github.com/sykurtyppi/market-dashboard](https://github.com/sykurtyppi/market-dashboard)
