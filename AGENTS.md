# AGENTS.md — market-dashboard

Per-repo facts for the Hermes review agent (adversarial quant reviewer). This file is the
repo-specific layer under the global SOUL.md identity and the shared six-check review skill
(look-ahead bias, point-in-time violations, survivorship, overfitting/multiple-testing,
cost/execution realism, regime cherry-picking). Report findings as a severity-ranked list
citing exact lines — never a prose summary.

> **Audit provenance.** Verified against `main` at commit `2c63a4b` on 2026-07-16. Package
> versions, line numbers, sample counts, and "currently surviving" observations are
> point-in-time snapshots — re-verify any specific line / number / version against the
> current tree before relying on it. The structural invariants (data providers, risk tier,
> the classes of pitfall) are durable; the exact citations are not.

## Orientation

An advisory market-risk dashboard that turns institutional metrics — VIX term structure, volatility
risk premium, Fed liquidity (balance sheet / RRP / TGA), NYSE breadth, repo/SOFR stress, credit spreads,
CTA trend — into **categorical signals** (BUY/SELL/SUPPORTIVE/STRESS/LONG). No capital, no execution:
signals are strings written to SQLite and rendered in the UI. The dominant risk is **macro/econometric
correctness**, not slippage.

## 1. Stack & dependencies

- **Python 3.13** (`render.yaml`). `streamlit>=1.28`, `pandas>=2.0`, `numpy`, `plotly`, `yfinance>=0.2.28`,
  `requests`/`beautifulsoup4`/`lxml`, `cot_reports` (CFTC COT), `scipy`, `pandas_market_calendars`, `fpdf2`
  (PDF), `pyyaml`, plus `fastapi>=0.115` + `uvicorn` + `httpx`. Separate Next.js/TypeScript frontend in
  `frontend/`.
- **Three surfaces over one analytics core:** (1) **Streamlit app** `dashboard/app.py` (primary deploy target,
  Streamlit Cloud) calling collectors/processors live; (2) **FastAPI backend** `api/main.py` (read = cached
  SQLite, `POST /api/refresh` runs `scheduler/daily_update.py::MarketDataUpdater.run_full_update()` under a
  lock), deployed on **Render** (`render.yaml`); (3) **Next.js frontend** consuming the JSON (the cutover
  target replacing Streamlit).
- **Data store:** SQLite `data/market_data.db` (`database/db_manager.py`; tables `indicators`, `signals`,
  `daily_snapshots`, `vrp_data`).
- **Refresh:** GitHub Actions `.github/workflows/refresh.yml` — twice each weekday (14:00 & 21:30 UTC, fixed,
  no DST), POSTs `/api/refresh` then polls status/freshness/health.

## 2. Risk tier — Macro/econometric (advisory, no capital)

No execution layer was found (grep for `broker|alpaca|ibkr|submit_order|place_order` finds nothing operative — strong evidence, not exhaustive proof).
Footer and README both say "Not financial advice." So the review focus is the macro/econometric set:
**point-in-time data violations (raw vs revised FRED series), seasonal-adjustment/revision consistency,
stationarity of z-scores, and hardcoded regime thresholds.** Cost/fill realism is N/A (no backtest P&L). All
signal/threshold findings are **flag-only**; collector code that decides revised-vs-PIT or synthetic-vs-real
data is numerically load-bearing and also flag-worthy even though it looks like plumbing.

## 3. Known pitfalls specific to this repo (verified from code)

1. **No point-in-time / vintage data anywhere (systemic).** Every FRED call
   (`data_collectors/fred_collector.py`, `fed_balance_sheet_collector.py`, `liquidity_collector.py`,
   `repo_collector_enhanced.py`) uses `series/observations` with only `observation_start/end` — **no
   `realtime_start/end`, no ALFRED vintages** (grep-confirmed). All macro data is **latest-revised**. Any
   *historical* signal series recomputed from them (`LEFTStrategy.get_historical_signals`,
   `VRPAnalyzer.get_historical_vrp`, liquidity/SOFR z-score histories) is **look-ahead-contaminated**: today's
   revised numbers applied to past dates. HY/IG OAS, `WALCL`, `WRESBAL`, and especially `M2SL` (monthly,
   seasonally adjusted, heavily revised) are all restated after the fact. The SQLite `daily_snapshots`/`signals`
   history is therefore **not** a clean vintage archive and can't be treated as one for backtesting.
2. **Release-lag / staleness in liquidity signals.** `WALCL`/`WRESBAL` are weekly (Wed); `WTREGEN` (TGA) is a
   weekly average `.resample("D").ffill()` (`liquidity_collector.py:804-811`). Net-liquidity z-scores
   (`processors/liquidity_signals.py`) mix a daily RRP with a forward-filled weekly TGA and a stale weekly Fed
   BS — the "latest" net-liq value can be up to a week stale and step-shaped.
3. **QT monthly-pace is a ~4-week approximation.** `fed_balance_sheet_collector.py::calculate_qt_metrics` uses
   `.diff(periods=4)` (28d) as "monthly pace" — self-documented ~8% underestimate vs a 30.44-day month
   (`qt_pace_is_approximate=True`), yet presented as monthly.
4. **VRP regime table is hardcoded, unsourced, and forward-labeled.** `processors/vrp_module.py`: `VRP = VIX −
   realized_vol(21d)`; VIX regime buckets carry a hardcoded `expected_6m_return` (VIX<12 → 15.2%, VIX>40 →
   25.0%) presented as forward return — classic in-sample/curve-fit regime labeling. VRP interpretation
   thresholds (8/4/0/−4) are hardcoded in the function, not in YAML.
5. **Hardcoded thresholds, no multiple-testing control across ~15 simultaneous signals.** `config/parameters.yaml`
   holds LEFT 0.65/1.40, VVIX 120/110, SKEW 160/145/130, net-liq z ±0.8, SOFR z 2.0/1.0, Zweig 0.40/0.615,
   RRP $50B/$200B, `RRP_PEAK=2600`. Several are attributed to practitioners (McMillan, McClellan, Zweig) but
   none are validated in-repo.
6. **Breadth survivorship (confirmed).** `data_collectors/sp500_adline_calculator.py` computes advance/decline
   from a **hardcoded static ~500-ticker list** applied across all history — includes delisted/renamed names
   (`SBNY` Signature Bank failed Mar 2023, `PKI`→RVTY, `CTLT` acquired). Historical breadth reflects today's
   membership. Same static-universe issue feeds `cta_engine.py` and cross-asset.
7. **Synthetic/sample data silently substituted.** `dark_pool_collector.py::_estimate_dark_pool_pct` generates
   **deterministic md5-hash "variance" around hardcoded baselines** when FINRA is unavailable (the normal case) —
   the dark-pool panel is largely fabricated (`data_source → 'Historical Baseline Estimates'`).
   `insider_trading_collector.py` falls back to `'Sample Data'` (`:357`). **Check the `data_source`/`is_estimated`/
   `is_sample` flags before trusting either panel.** CNN Fear&Greed is a single-source undocumented scrape with
   no fallback.
8. **z-score stationarity / window edges.** Repo SOFR z-score (`repo_collector_enhanced.py:159-175`) uses an
   adaptive window `min(252, len-1)` — early series z-scores are not comparable to later ones (a `LOW_VARIANCE`
   flag guards std≈0). Liquidity z-score needs only `min_data_points=30`, so a "z-score" can be emitted off 30 obs.
9. **Duplicate divergent LEFT logic.** `data_collectors/fred_collector.py::calculate_credit_spread_signals`
   (−0.35/+0.40 pct-from-EMA) duplicates `processors/left_strategy.py` (ratio ≤0.65 / ≥1.40) — two thresholds
   for the same signal; risk of the wrong one being surfaced.
10. **Timezone join hazard (handled, fragile).** `vrp_module.py:532-537` normalizes VIX(Chicago)/SPY(NY) to
    date-only before joining; a regression here silently drops rows via inner-join.

## 4. What the agent may fix directly vs only flag

**Default posture is read-only.** During a review-only task, report proposed changes as
findings and do not edit; post inline PR comments only as the configured review bot or when
explicitly asked, not merely because a PR exists. Fixes apply only when explicitly
authorized — and even then, numerical / signal / statistical changes require focused
before/after validation and human review, never a silent edit. "Low-risk" is not risk-free:
UI, scheduler, deploy, CORS, and DB code can still be consequential — treat every item below
as a candidate, not standing authorization.


**Flag only (signal math / thresholds — editing changes every signal):**
`config/parameters.yaml` thresholds and the `expected_6m_return` table; `processors/left_strategy.py`,
`vrp_module.py`, `liquidity_signals.py`, `repo_analyzer.py`, `rrp_analyzer.py`, `cta_engine.py`,
`breadth_signals.py`; the point-in-time gap (all FRED collectors — moving to ALFRED vintages is a methodology
change); `data_collectors/sp500_adline_calculator.py` static universe.

**Numerically load-bearing collector plumbing — flag (decides revised-vs-PIT / synthetic-vs-real):**
FRED fetchers' endpoint/param choice, TGA `resample("D").ffill()`, QT `.diff(periods=4)`, SOFR adaptive-window
z-score, and the `dark_pool_collector`/`insider_trading_collector` fallbacks.

**Low-risk — only if a fix is explicitly requested, (infra/UI/plumbing, no signal impact):**
`api/main.py` wiring/CORS/lock; `dashboard/core/*`, `dashboard/views/*`, `frontend/*` rendering;
`dashboard/pdf_*`; `utils/retry_utils.py`, `utils/secrets_helper.py`; `render.yaml`, `.github/workflows/*`,
`DEPLOY.md`; SQLite schema/migrations in `database/`; timezone-normalization helpers (re-verify the VRP join if
touched).

## 5. PR etiquette

Findings as **inline review comments on exact lines**, severity-ranked, each naming the concrete way the
number could mislead (revised-not-PIT, stale weekly liquidity, hardcoded forward-return table, breadth
survivorship, synthetic panel). No full-file rewrites unless explicitly asked to push a fix commit. For
flag-only areas, comment and stop. When flagging a collector, state whether the issue is revised-vs-PIT,
release-lag, or synthetic-fallback — each has a different remedy.
