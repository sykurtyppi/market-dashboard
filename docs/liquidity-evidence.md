# Liquidity evidence: first data-trust improvement

## Scope

Extend the existing overview with a read-only evidence view. Preserve the stored exact-date net-liquidity series, collector behavior, historical rows, composite risk model, and briefing records. No migration or refresh is required for this view. This feature does not address source-wide freshness or publication-time data ingestion.

The overview snapshot's freshness describes the snapshot, not the completeness or recency of every underlying observation. Individual observation dates remain visible independently.

## Two distinct views

- **Stored strict net:** latest finite stored `net_liquidity` observation on or before the evaluation date. Its date is not advanced when newer component observations arrive. The original component lineage of a stored derived result is not reconstructed or asserted.
- **Exploratory proxy:** use the latest eligible date with finite WALCL and WTREGEN observations on the same date. Select only an RRP observation on or before that target date, with a maximum age of four calendar days. If the newest target cannot be matched within that bound, return unavailable rather than substituting an older target. Keep all actual component dates visible. Do not overwrite the strict value.

The four-calendar-day bound is a deliberately limited exploratory policy, **not** a validated business-day or holiday calendar policy. It is not an investment recommendation, freshness guarantee, or production signal rule. An `available` coverage status means values exist under the stated policy, not that those values are current relative to today's release schedule.

## Definition and units

Formula: **WALCL − WTREGEN − RRPONTSYD**, using already normalized **USD billions** from `liquidity_history`. WALCL and WTREGEN enter ingestion as USD millions and are converted there; they must not be divided again in the evidence view. WTREGEN is a **weekly-average TGA** series; WDTGAL's Wednesday balance is not an interchangeable replacement. Even an exact observation-date match is not simultaneous daily balance data.

Zero is a legitimate observation, including zero RRP. Missing, invalid, nonfinite, or future-dated observations are not evidence. Dates refer to observations, never refresh timestamps.

## Timing limits

The current table does not establish publication timestamps or historical data vintages. Backward matching by observation date does **not** prove that a weekly release was available to a historical decision maker. This panel is a current-data research view, not a point-in-time historical dataset, trading signal, or backtest input. These limitations must appear in the UI, not only this document.

## Acceptance checks

Backend fixtures must establish:

- a missing same-day RRP leaves the stored strict result untouched;
- the proxy uses an eligible earlier RRP and exposes its real date;
- future RRP is never used;
- four-day carry is accepted; five-day carry is rejected;
- zero survives, and absent/nonfinite values degrade to unavailable;
- newest unmatched target does not silently fall back to an older proxy;
- empty/partial history remains serializable and typed through the HTTP response model;
- the processor does not mutate input data or write database state.

Browser tests must establish:

- fresh snapshot copy coexists with older strict and separately dated proxy data;
- component source identifiers, actual dates, units, and policy limitations render;
- unavailable and zero states are distinct;
- older API responses without this optional field degrade honestly;
- existing overview/briefing flows continue to work, including mobile layout.

## Rollout boundary

Development is isolated on `feat/liquidity-evidence`, based on upstream `4fe51bd`. PR #72 rollout reconciliation is separate; this branch does not establish its current merge or deployment state. Keep deployment, production refreshes, service configuration changes, and PR #72 reconciliation separate from feature implementation. Run tests against fixture/temporary databases, never the production database.

## Next experiments (not part of this slice)

1. Compare a stress/opportunity two-view mock against the existing composite without introducing unvalidated scores.
2. Trial one conditional research card per session: hypothesis, horizon, opposing evidence, invalidation, review trigger.
3. Add versioned, point-in-time eligible evaluation before predictive claims.
