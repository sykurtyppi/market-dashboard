"""
Fed Balance Sheet Collector - ULTIMATE VERSION

Fetches Federal Reserve balance sheet data from FRED:
- WALCL: Total Assets (Fed Balance Sheet size)
- WRESBAL: Reserve Balances with Federal Reserve Banks
- WLCFLPCL: Loans and Credit (emergency facilities)

This is THE most important macro liquidity indicator.

Net Liquidity (proper formula) = Fed BS - TGA - RRP

ENHANCED: Now includes balance_sheet_df and all dashboard-required keys
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional

import requests
import pandas as pd

logger = logging.getLogger(__name__)


import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.retry_utils import exponential_backoff_retry
class FedBalanceSheetCollector:
    """
    Collects Federal Reserve balance sheet data from FRED.
    
    Key series:
    - WALCL: Total Fed assets (the "balance sheet")
    - WRESBAL: Reserve balances
    - WLCFLPCL: Loans (emergency facilities)
    """

    FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    SERIES = {
        "total_assets": {"id": "WALCL", "name": "Fed Total Assets"},
        "reserve_balances": {"id": "WRESBAL", "name": "Reserve Balances"},
        "loans": {"id": "WLCFLPCL", "name": "Fed Loans"},
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: FRED API key (falls back to env FRED_API_KEY)
        """
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        if not self.api_key:
            raise ValueError(
                "FRED_API_KEY not found. Set it in .env or pass explicitly."
            )

    @exponential_backoff_retry(max_retries=3, base_delay=2.0)
    def _fetch_series(
        self,
        series_id: str,
        start_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch a FRED series and return DataFrame with columns: date, value
        """
        if start_date is None:
            # Default: 3 years of history
            start = datetime.today() - timedelta(days=365*3)
            start_date = start.strftime("%Y-%m-%d")

        params = {
            "api_key": self.api_key,
            "series_id": series_id,
            "file_type": "json",
            "observation_start": start_date,
        }

        try:
            logger.info(f"Fetching FRED series {series_id} from {start_date}")
            resp = requests.get(self.FRED_BASE_URL, params=params, timeout=15)
            resp.raise_for_status()
            
            data = resp.json().get("observations", [])
            
            if not data:
                logger.warning(f"No data returned for {series_id}")
                return pd.DataFrame(columns=["date", "value"])

            df = pd.DataFrame(data)[["date", "value"]]
            df["date"] = pd.to_datetime(df["date"])
            # FRED uses '.' for missing values
            df["value"] = pd.to_numeric(df["value"].replace(".", pd.NA), errors="coerce")
            df = df.dropna(subset=["value"]).sort_values("date").reset_index(drop=True)
            
            logger.info(f"Fetched {len(df)} rows for {series_id}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {series_id}: {e}")
            return pd.DataFrame(columns=["date", "value"])

    def get_total_assets(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """Get Fed total assets (WALCL) - the Fed's balance sheet size."""
        series_id = self.SERIES["total_assets"]["id"]
        df = self._fetch_series(series_id, start_date=start_date)
        df = df.rename(columns={"value": "total_assets"})
        return df

    def get_reserve_balances(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """Get reserve balances held at Fed (WRESBAL)."""
        series_id = self.SERIES["reserve_balances"]["id"]
        df = self._fetch_series(series_id, start_date=start_date)
        df = df.rename(columns={"value": "reserve_balances"})
        return df

    def get_loans(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """Get Fed loans and credit (WLCFLPCL) - emergency facilities."""
        series_id = self.SERIES["loans"]["id"]
        df = self._fetch_series(series_id, start_date=start_date)
        df = df.rename(columns={"value": "loans"})
        return df

    def get_balance_sheet_history(
        self,
        lookback_days: int = 730,
    ) -> pd.DataFrame:
        """
        Get complete Fed balance sheet history.
        
        Returns DataFrame with columns:
        - date
        - total_assets (millions USD)
        - reserve_balances (millions USD)
        - loans (millions USD)
        """
        end = datetime.today()
        start = end - timedelta(days=lookback_days)
        start_str = start.strftime("%Y-%m-%d")

        logger.info(f"Fetching Fed balance sheet data from {start_str}")

        # Fetch all series
        assets = self.get_total_assets(start_str)
        reserves = self.get_reserve_balances(start_str)
        loans = self.get_loans(start_str)

        if assets.empty:
            logger.error("Failed to fetch Fed total assets")
            return pd.DataFrame()

        # Merge all series
        df = assets.copy()
        
        if not reserves.empty:
            df = df.merge(reserves, on="date", how="left")
        if not loans.empty:
            df = df.merge(loans, on="date", how="left")

        # Forward fill any gaps
        df = df.sort_values("date")
        for col in ["total_assets", "reserve_balances", "loans"]:
            if col in df.columns:
                df[col] = df[col].ffill()

        logger.info(f"Fed balance sheet: {len(df)} rows")
        
        return df

    def calculate_qt_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate QT (Quantitative Tightening) metrics from balance sheet data.
        
        Adds columns:
        - qt_cumulative: Cumulative change since QT peak (millions USD)
        - qt_monthly_pace: 30-day rolling change (millions USD)
        - qt_pace_billions_month: Monthly pace in billions/month
        """
        if df.empty or 'total_assets' not in df.columns:
            return df

        df = df.sort_values("date").copy()

        # Find QT peak (likely around April 2022)
        peak_idx = df["total_assets"].idxmax()
        peak_value = df.loc[peak_idx, "total_assets"]
        peak_date = df.loc[peak_idx, "date"]

        logger.info(f"Fed BS peak: ${peak_value:,.0f}M on {peak_date.strftime('%Y-%m-%d')}")

        # Cumulative QT since peak
        df["qt_cumulative"] = df["total_assets"] - peak_value

        # Monthly pace (30-day change)
        # Weekly data: 4 periods ≈ 1 month
        df["qt_monthly_pace"] = df["total_assets"].diff(periods=4)
        
        # Convert to billions/month for readability
        df["qt_pace_billions_month"] = df["qt_monthly_pace"] / 1000.0

        return df

    def get_full_snapshot(self, lookback_days: int = 730) -> dict:
        """
        Get complete Fed balance sheet snapshot with all metrics.
        
        Returns dict with ALL keys needed by dashboard:
        - latest_date: str
        - total_assets: float (billions) ← Dashboard key
        - total_assets_billions: float (alias)
        - reserve_balances_billions: float
        - qt_cumulative: float (billions) ← Dashboard key
        - qt_cumulative_billions: float (alias)
        - qt_pace_billions_month: float ← Dashboard key
        - qt_monthly_pace_billions: float (alias)
        - balance_sheet_df: pd.DataFrame ← CRITICAL for charts
        - history: list of dicts
        """
        df = self.get_balance_sheet_history(lookback_days)
        
        if df.empty:
            return {
                "error": "No Fed balance sheet data available",
                "latest_date": None,
                "total_assets": None,
                "total_assets_billions": None,
                "reserve_balances_billions": None,
                "qt_cumulative": None,
                "qt_cumulative_billions": None,
                "qt_pace_billions_month": None,
                "qt_monthly_pace_billions": None,
                "balance_sheet_df": pd.DataFrame(),
                "history": []
            }

        df = self.calculate_qt_metrics(df)

        latest = df.iloc[-1]
        latest_date = latest["date"].strftime("%Y-%m-%d")
        
        # Convert from millions to billions
        total_assets_millions = latest["total_assets"]
        total_assets_billions = total_assets_millions / 1000.0 if pd.notna(total_assets_millions) else None
        
        reserve_balances_millions = latest.get("reserve_balances", 0)
        reserve_balances_billions = reserve_balances_millions / 1000.0 if pd.notna(reserve_balances_millions) else None
        
        qt_cumulative_millions = latest.get("qt_cumulative", 0)
        qt_cumulative_billions = qt_cumulative_millions / 1000.0 if pd.notna(qt_cumulative_millions) else None
        
        qt_pace = latest.get("qt_pace_billions_month", 0)
        qt_pace_billions = float(qt_pace) if pd.notna(qt_pace) else None

        snapshot = {
            "latest_date": latest_date,
            "total_assets": total_assets_billions,  # Dashboard expects this key
            "total_assets_billions": total_assets_billions,  # Alias
            "reserve_balances_billions": reserve_balances_billions,
            "qt_cumulative": qt_cumulative_billions,  # Dashboard expects this key
            "qt_cumulative_billions": qt_cumulative_billions,  # Alias
            "qt_pace_billions_month": qt_pace_billions,  # Dashboard expects this key
            "qt_monthly_pace_billions": qt_pace_billions,  # Alias
            "balance_sheet_df": df,  # ← CRITICAL: DataFrame for charts
            "history": df.to_dict('records'),
        }

        logger.info(
            f"Fed BS: ${total_assets_billions:,.0f}B, "
            f"QT: {qt_cumulative_billions:,.0f}B cumulative"
        )

        return snapshot
    
    def get_latest_data(self) -> dict:
        """Get latest Fed balance sheet data"""
        df = self.get_balance_sheet_history(lookback_days=30)
        
        if df.empty:
            return {}
        
        latest = df.iloc[-1]
        return {
            'total_assets': latest.get('total_assets'),
            'reserves': latest.get('reserves'),
            'securities': latest.get('securities'),
            'date': latest.get('date'),
            'timestamp': pd.Timestamp.now().isoformat()
        }
