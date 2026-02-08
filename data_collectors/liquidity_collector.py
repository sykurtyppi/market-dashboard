# data_collectors/liquidity_collector.py

"""
Liquidity Collector
Fetches key US liquidity metrics from FRED:
- Fed Overnight Reverse Repo (ON RRP)
- Treasury General Account (TGA)
- Secured Overnight Financing Rate (SOFR)

These are intended to feed the credit/liquidity section of the dashboard.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Optional
import logging

import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.retry_utils import exponential_backoff_retry
class LiquidityCollector:
    """
    Collects macro liquidity data from FRED.

    Series used (all from FRED):
        - RRPONTSYD : Overnight Reverse Repurchase Agreements (ON RRP), total
        - WTREGEN   : Treasury General Account (TGA), weekly average
        - SOFR      : Secured Overnight Financing Rate (repo benchmark)

    Notes
    -----
    - Requires FRED_API_KEY to be set in your .env file
    - Returns tidy DataFrames with a `date` column and one value column
    """

    FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    SERIES = {
        "rrp_on": {
            "id": "RRPONTSYD",
            "title": "Fed ON RRP (RRPONTSYD)",
        },
        "tga": {
            "id": "WTREGEN",
            "title": "Treasury General Account (WTREGEN)",
        },
        "sofr": {
            "id": "SOFR",
            "title": "Secured Overnight Financing Rate (SOFR)",
        },
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        self.logger = logging.getLogger(__name__)
        self._disabled = False
        if not self.api_key:
            self.logger.warning(
                "FRED_API_KEY not found. Liquidity data will be unavailable. "
                "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
            )
            self._disabled = True

    # ---------- Internal helpers ----------

    def _fetch_series(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch a single FRED series as a DataFrame.

        Parameters
        ----------
        series_id : str
            FRED series ID (e.g. 'RRPONTSYD')
        start_date : str, optional
            'YYYY-MM-DD'. If None, defaults to 2 years back.
        end_date : str, optional
            'YYYY-MM-DD'. If None, defaults to today.
        """
        # Return empty DataFrame if API key not configured
        if self._disabled:
            return pd.DataFrame(columns=["date", series_id])

        if end_date is None:
            end_date = datetime.today().strftime("%Y-%m-%d")
        if start_date is None:
            start_dt = datetime.today() - timedelta(days=365 * 2)
            start_date = start_dt.strftime("%Y-%m-%d")

        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start_date,
            "observation_end": end_date,
        }

        try:
            resp = requests.get(self.FRED_BASE_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if "observations" not in data or not data["observations"]:
                logger.warning(f"No data returned for {series_id}")
                return pd.DataFrame(columns=["date", "value"])

            df = pd.DataFrame(data["observations"])
            df = df.rename(columns={"date": "date", "value": "value"})
            df["date"] = pd.to_datetime(df["date"])
            # FRED uses '.' for missing values
            df["value"] = pd.to_numeric(df["value"].replace(".", pd.NA), errors="coerce")
            df = df.dropna(subset=["value"]).sort_values("date").reset_index(drop=True)
            
            logger.info(f"Fetched {len(df)} rows for {series_id}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {series_id}: {e}")
            return pd.DataFrame(columns=["date", "value"])

    # ---------- Public fetch methods ----------

    def get_rrp_on(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """Fed Overnight Reverse Repo (RRPONTSYD)."""
        series_id = self.SERIES["rrp_on"]["id"]
        df = self._fetch_series(series_id, start_date=start_date)
        df = df.rename(columns={"value": "rrp_on"})
        return df

    def get_tga(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """
        Treasury General Account balance (WTREGEN).

        This is weekly data; we forward-fill to daily so it lines up
        with other series on the dashboard.
        """
        series_id = self.SERIES["tga"]["id"]
        df = self._fetch_series(series_id, start_date=start_date)
        if df.empty:
            return pd.DataFrame(columns=["date", "tga"])
        df = df.rename(columns={"value": "tga"})
        # Resample to daily and forward fill for smooth charts
        df = (
            df.set_index("date")
            .resample("D")
            .ffill()
            .reset_index()
        )
        return df

    def get_sofr(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """Secured Overnight Financing Rate (SOFR)."""
        series_id = self.SERIES["sofr"]["id"]
        df = self._fetch_series(series_id, start_date=start_date)
        df = df.rename(columns={"value": "sofr"})
        return df

    def get_all_liquidity(
        self,
        lookback_days: int = 365,
    ) -> pd.DataFrame:
        """
        Fetch all liquidity metrics and combine into one DataFrame.

        Returns
        -------
        DataFrame with columns:
            - date
            - rrp_on      (billions USD)
            - tga         (billions USD)
            - sofr        (%)
        """
        end = datetime.today()
        start = end - timedelta(days=lookback_days)
        start_str = start.strftime("%Y-%m-%d")

        logger.info(f"Fetching liquidity data from {start_str}")
        
        rrp = self.get_rrp_on(start_str)
        tga = self.get_tga(start_str)
        sofr = self.get_sofr(start_str)

        # Debug: Log sample values
        if not rrp.empty:
            logger.info(f"RRP sample values: min={rrp['rrp_on'].min():.2f}, max={rrp['rrp_on'].max():.2f}, latest={rrp['rrp_on'].iloc[-1]:.2f}")
            logger.info(f"RRP date sample: {rrp['date'].iloc[0]}, type: {type(rrp['date'].iloc[0])}")
        if not tga.empty:
            logger.info(f"TGA sample values: min={tga['tga'].min():.2f}, max={tga['tga'].max():.2f}, latest={tga['tga'].iloc[-1]:.2f}")
            logger.info(f"TGA date sample: {tga['date'].iloc[0]}, type: {type(tga['date'].iloc[0])}")

        if rrp.empty and tga.empty and sofr.empty:
            logger.warning("All liquidity series returned empty")
            return pd.DataFrame()

        # Create date range - normalize to date only (no time component)
        df = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
        df['date'] = pd.to_datetime(df['date']).dt.normalize()
        
        logger.info(f"Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")

        # CRITICAL: Normalize dates in all series to match (remove time/timezone)
        for sub in (rrp, tga, sofr):
            if not sub.empty:
                sub['date'] = pd.to_datetime(sub['date']).dt.normalize()
                logger.info(f"Merging {len(sub)} rows, date range: {sub['date'].iloc[0]} to {sub['date'].iloc[-1]}")
                df = df.merge(sub, on="date", how="left")
                logger.info(f"After merge: {df.shape}")

        df = df.sort_values("date").reset_index(drop=True)
        logger.info(f"Combined liquidity data: {len(df)} rows")
        
        # Debug: Check for NaN after merge
        logger.info(f"RRP NaN count: {df['rrp_on'].isna().sum() if 'rrp_on' in df.columns else 'N/A'}")
        logger.info(f"TGA NaN count: {df['tga'].isna().sum() if 'tga' in df.columns else 'N/A'}")
        
        return df