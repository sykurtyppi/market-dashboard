"""Actual HY/IG OAS observations from FRED's public CSV download.

This is the credential-free fallback for the credit subset of FREDCollector,
not a yield/ETF proxy. Its frames use the existing save_indicators_batch shape,
including the original series ID and observation date. No observation is dated
with the retrieval time and missing releases are never forward-filled.
"""
import logging
from datetime import date, timedelta
from io import StringIO

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


class CreditCollector:
    SERIES = {
        "credit_spread_hy": "BAMLH0A0HYM2",
        "credit_spread_ig": "BAMLC0A0CM",
    }
    URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    def get_all_indicators(
        self, lookback_days: int = 730, *, end_date: date | None = None
    ) -> dict[str, pd.DataFrame]:
        """Fetch both exact OAS series with one bounded, uncredentialed GET.

        Return only independently valid series, so an unavailable IG release
        cannot discard valid HY. A failed response returns no observations;
        callers must not replace existing dated history with fabricated data.
        """
        if not 1 <= lookback_days <= 1095:
            raise ValueError("lookback_days must be between 1 and 1095")
        end = end_date or date.today()
        start = end - timedelta(days=lookback_days)
        try:
            response = requests.get(
                self.URL,
                params={
                    "id": ",".join(self.SERIES.values()),
                    "cosd": start.isoformat(),
                    "coed": end.isoformat(),
                },
                timeout=(5, 20),
            )
            response.raise_for_status()
            raw = pd.read_csv(StringIO(response.text))
            if "observation_date" not in raw.columns:
                logger.warning("FRED public credit response has no observation_date")
                return {}
            dates = pd.to_datetime(raw["observation_date"], errors="coerce")
            in_window = dates.between(pd.Timestamp(start), pd.Timestamp(end))
            results = {}
            for indicator, series_id in self.SERIES.items():
                if series_id not in raw.columns:
                    logger.warning("FRED public credit response missing %s", series_id)
                    continue
                values = pd.to_numeric(raw[series_id], errors="coerce")
                valid = in_window & np.isfinite(values) & (values >= 0)
                frame = pd.DataFrame({"date": dates[valid], series_id: values[valid]})
                frame = frame.sort_values("date").drop_duplicates("date", keep="last")
                if not frame.empty:
                    results[indicator] = frame.reset_index(drop=True)
                else:
                    logger.warning("No valid public FRED credit observations for %s", series_id)
            return results
        except (requests.RequestException, pd.errors.ParserError, pd.errors.EmptyDataError, ValueError):
            # Never log response bodies or request details (even though this
            # endpoint requires no secret). Fail closed instead of using an ETF.
            logger.warning("FRED public credit observations unavailable")
            return {}
