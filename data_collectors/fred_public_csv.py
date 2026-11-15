"""Strict, unauthenticated transport for the official FRED graph CSV export.

Only used by collectors when explicitly enabled and no API key is configured.
No metadata APIs, fallback hosts, imputation, or extrapolation are used here.
"""
import csv
from datetime import date
from io import StringIO
import logging
import math
import os
import re
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)
PUBLIC_FRED_CSV_URL = 'https://fred.stlouisfed.org/graph/fredgraph.csv'


def public_csv_enabled(api_key: Optional[str]) -> bool:
    """An API key always takes precedence; only the literal '1' opts in."""
    return not api_key and os.getenv('FRED_USE_PUBLIC_CSV') == '1'


def _date(value: str) -> date:
    if not isinstance(value, str) or not re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}', value):
        raise ValueError('Expected ISO date')
    return date.fromisoformat(value)


def fetch_public_fred_csv(
    series_id: str,
    start_date: str,
    end_date: Optional[str] = None,
    *,
    limit: int = 100000,
    optional: bool = False,
) -> pd.DataFrame:
    """Return observed ``date, value`` rows, or an empty frame on any failure.

    Require exactly ``observation_date,<requested series>`` and two fields per
    row. '.' and empty fields mean missing (the graph export uses empty holiday
    values); other invalid/nonfinite values reject the
    response. Validate every row before clipping, sorting and applying limit.
    Exceptions are logged by class only, never response bodies or request URLs.
    """
    empty = pd.DataFrame({'date': pd.Series(dtype='datetime64[ns]'),
                          'value': pd.Series(dtype='float64')})
    log = logger.debug if optional else logger.warning
    try:
        if not isinstance(series_id, str) or not re.fullmatch(r'[A-Za-z0-9_]+', series_id):
            raise ValueError('Invalid series ID')
        start = _date(start_date)
        end_date = end_date if end_date is not None else date.today().isoformat()
        end = _date(end_date)
        if start > end or type(limit) is not int or limit <= 0:
            raise ValueError('Invalid range or limit')
    except (ValueError, TypeError):
        log('Public FRED CSV rejected invalid request arguments')
        return empty

    try:
        response = requests.get(
            PUBLIC_FRED_CSV_URL,
            params={'id': series_id, 'cosd': start_date, 'coed': end_date},
            timeout=10,
            allow_redirects=False,
        )
        response.raise_for_status()
        reader = csv.reader(StringIO(response.text), strict=True)
        if next(reader, None) != ['observation_date', series_id]:
            raise ValueError('Unexpected CSV schema')
        rows = []
        seen = set()
        for row in reader:
            if len(row) != 2:
                raise ValueError('Unexpected CSV row width')
            observed = _date(row[0])
            if observed in seen:
                raise ValueError('Duplicate observation date')
            seen.add(observed)
            if row[1] in ('.', ''):
                continue
            value = float(row[1])
            if not math.isfinite(value):
                raise ValueError('Nonfinite observation')
            if start <= observed <= end:
                rows.append((observed, value))
        if not rows:
            return empty
        frame = pd.DataFrame(rows, columns=['date', 'value'])
        frame['date'] = pd.to_datetime(frame['date'])
        return frame.sort_values('date').head(limit).reset_index(drop=True)
    except (requests.RequestException, ValueError, TypeError, csv.Error, OverflowError) as exc:
        log('Public FRED CSV unavailable for %s (%s)', series_id, type(exc).__name__)
        return empty
