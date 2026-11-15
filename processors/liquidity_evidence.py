"""Pure request-time liquidity evidence, never persisted or used for scoring.

Inputs from liquidity_history are already USD billions (including WALCL and
WTREGEN, normalized by the scheduler before storage). No unit conversion here.
"""
from datetime import date, datetime
import math
from typing import Any

import pandas as pd

MAX_RRP_AGE_DAYS = 4
_COMPONENTS = (
    ('fed_balance', 'fed_balance_sheet', 'WALCL', 'Federal Reserve total assets'),
    ('tga', 'tga', 'WTREGEN', 'Treasury General Account (weekly average)'),
    ('rrp_on', 'rrp_on', 'RRPONTSYD', 'ON RRP'),
)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _observation_date(value: Any) -> date | None:
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            parsed = date.fromisoformat(value)
            # Accept only the stored ISO calendar-date format, not compact/week dates.
            return parsed if value == parsed.isoformat() else None
        except ValueError:
            pass
    return None


def build_liquidity_evidence(history: pd.DataFrame | None, *, today: date) -> dict:
    """Describe latest eligible observations, without claiming vintage eligibility.

    Identical duplicates collapse; conflicting finite values invalidate that
    series/date, independently of input order. Missing values cannot erase a
    finite duplicate. No observation is inferred from another series or date.
    """
    columns = [column for _, column, _, _ in _COMPONENTS] + ['net_liquidity']
    candidates = {column: {} for column in columns}
    if history is not None:
        for row in history.to_dict(orient='records'):
            observed = _observation_date(row.get('date'))
            if observed is None or observed > today:
                continue
            for column in columns:
                value = _finite(row.get(column))
                if value is not None:
                    candidates[column].setdefault(observed, set()).add(value)
    observations = {
        column: {observed: next(iter(values)) for observed, values in dates.items() if len(values) == 1}
        for column, dates in candidates.items()
    }

    def component(spec, observed=None, target=None):
        key, column, series, label = spec
        return {
            'key': key, 'series': series, 'label': label,
            'value': observations[column].get(observed),
            'as_of': observed.isoformat() if observed is not None else None,
            'age_days': (target - observed).days if target is not None and observed is not None else None,
        }

    latest = [component(spec, max(observations[spec[1]], default=None)) for spec in _COMPONENTS]
    strict_date = max(observations['net_liquidity'], default=None)
    strict = {
        'value': observations['net_liquidity'].get(strict_date),
        'as_of': strict_date.isoformat() if strict_date is not None else None,
    }
    target = max(observations['fed_balance_sheet'].keys() & observations['tga'].keys(), default=None)
    rrp_date = max((d for d in observations['rrp_on'] if target is not None and d <= target), default=None)
    components = [component(spec, target if target is not None else None, target) for spec in _COMPONENTS[:2]]
    components.append(component(_COMPONENTS[2], rrp_date, target))
    proxy = {
        'value': None, 'as_of': target.isoformat() if target is not None else None,
        'status': 'unavailable', 'max_rrp_age_days': MAX_RRP_AGE_DAYS,
        'reason': None, 'components': components,
    }
    if target is None:
        proxy['reason'] = 'No joint date with finite WALCL and WTREGEN observations.'
    elif rrp_date is None:
        proxy['reason'] = 'No finite ON RRP observation at or before the latest joint weekly date.'
    elif (target - rrp_date).days > MAX_RRP_AGE_DAYS:
        proxy['reason'] = 'Latest backward ON RRP observation exceeds the 4-calendar-day exploratory limit.'
    else:
        value = _finite(components[0]['value'] - components[1]['value'] - components[2]['value'])
        if value is None:
            proxy['reason'] = 'Proxy arithmetic did not produce a finite value.'
        else:
            proxy['value'] = value
            proxy['status'] = 'exact' if rrp_date == target else 'backward_matched'
            proxy['reason'] = (
                'All three components share the target observation date.' if rrp_date == target else
                f'ON RRP matched backward by {(target - rrp_date).days} calendar day(s).'
            )
    some_valid = strict['value'] is not None or any(c['value'] is not None for c in latest)
    return {
        'status': 'available' if strict['value'] is not None and proxy['value'] is not None else
                  ('partial' if some_valid else 'unavailable'),
        'unit': 'USD billions', 'strict': strict, 'proxy': proxy,
        'latest_components': latest,
        'notes': [
            'WTREGEN is a weekly average; these are not simultaneous daily balances. No WDTGAL substitution is made.',
            'Current research view only, not a historical or trading signal. Publication/availability timestamps and vintage eligibility are unknown; not suitable for backtests.',
            'Backward ON RRP matching uses an exploratory maximum of 4 calendar days, not a holiday-calendar-aware policy; never matches forward or falls back to an older weekly target.',
            'Strict is the latest finite stored net observation; its original component lineage is unknown and is not reconstructed from current components.',
            'Identical duplicates collapse; conflicting finite values invalidate that series/date regardless of row order.',
        ],
    }
