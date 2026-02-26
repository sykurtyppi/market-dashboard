# Market Risk Dashboard

Public Streamlit dashboard for monitoring cross-asset risk, volatility regime, credit stress, positioning, and liquidity context.

Creator: Tristan Alejandro  
Notice: Not financial advice.

## What It Covers
- Volatility: VIX, VIX9D, VIX3M, VVIX, SKEW, term structure/contango
- Credit: HY and IG spread context, credit ETF behavior
- Positioning: COT extremes, put/call context, sentiment gauges
- Liquidity and rates: Fed balance sheet, repo/sofr context, treasury stress
- Breadth and internal market health metrics

## Data Reliability Design
- Data source + delay labels on key metrics
- Per-metric status (fresh/stale/estimated/unavailable)
- Yahoo/CBOE/COT memoization and last-known-good fallback persistence
- Holiday-aware market status logic

## Local Run
1. Create and activate a virtual environment.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Add secrets:
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```
4. Run:
```bash
streamlit run dashboard/app.py
```

## Deploy Publicly (Streamlit Cloud)
Use:
- `/Users/tristanalejandro/Market Dashboard/DEPLOY.md`

Main app entry:
- `dashboard/app.py`

## Public Release Checklist
- Keep `.streamlit/secrets.toml` local only (already gitignored)
- Use `.streamlit/secrets.toml.example` for collaborators
- Confirm tests pass before each push:
```bash
pytest -q
```
- Confirm app boot:
```bash
streamlit run dashboard/app.py
```
- Rotate any keys that were ever shared outside your private environment

## License
Licensed under GNU AGPL v3. See:
- `/Users/tristanalejandro/Market Dashboard/LICENSE`

## Source
- [https://github.com/sykurtyppi/market-dashboard](https://github.com/sykurtyppi/market-dashboard)

