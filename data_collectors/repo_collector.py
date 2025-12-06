"""
Repo Market Collector - ULTIMATE VERSION

Tracks repo (repurchase agreement) market conditions - critical for funding liquidity.

Key indicators:
1. GC Repo Rate (General Collateral) - FRED: GCFRED
2. SOFR (Secured Overnight Financing Rate) - FRED: SOFR  
3. SOFR-RRP spread - Key stress indicator
4. Tri-party repo volume - FRED: RPONTTD (if available)

The repo market is where the financial system gets short-term funding.
Stress here can cascade into broader market issues.

ENHANCED: Now includes repo_df and all dashboard-required keys
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional

import requests
import pandas as pd

logger = logging.getLogger(__name__)


class RepoCollector:
    """
    Collects repo market data from FRED.
    
    Tracks funding stress and repo market conditions.
    """

    FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    SERIES = {
        "sofr": {"id": "SOFR", "name": "Secured Overnight Financing Rate"},
        "gc_repo": {"id": "GCFRED", "name": "GC Repo Rate"},
        "rrp": {"id": "RRPONTSYD", "name": "Overnight RRP"},
        "triparty_volume": {"id": "RPONTTD", "name": "Tri-party Repo Volume"},
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: FRED API key (falls back to env FRED_API_KEY)
        """
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        if not self.api_key:
            raise ValueError(
                "FRED_API_KEY not found. Set in .env or pass explicitly."
            )

    def _fetch_series(
        self,
        series_id: str,
        start_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch a FRED series.
        
        Returns:
            DataFrame with columns: date, value
        """
        if start_date is None:
            start = datetime.today() - timedelta(days=730)
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
                logger.warning(f"No data for {series_id}")
                return pd.DataFrame(columns=["date", "value"])

            df = pd.DataFrame(data)[["date", "value"]]
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"].replace(".", pd.NA), errors="coerce")
            df = df.dropna(subset=["value"]).sort_values("date").reset_index(drop=True)
            
            logger.info(f"Fetched {len(df)} rows for {series_id}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {series_id}: {e}")
            return pd.DataFrame(columns=["date", "value"])

    def get_sofr(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """Get SOFR (Secured Overnight Financing Rate)."""
        series_id = self.SERIES["sofr"]["id"]
        df = self._fetch_series(series_id, start_date=start_date)
        df = df.rename(columns={"value": "sofr"})
        return df

    def get_gc_repo(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """Get GC Repo Rate (General Collateral)."""
        series_id = self.SERIES["gc_repo"]["id"]
        df = self._fetch_series(series_id, start_date=start_date)
        df = df.rename(columns={"value": "gc_repo"})
        return df

    def get_rrp(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """Get Overnight RRP volume."""
        series_id = self.SERIES["rrp"]["id"]
        df = self._fetch_series(series_id, start_date=start_date)
        df = df.rename(columns={"value": "rrp"})
        return df

    def get_triparty_volume(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """Get tri-party repo volume (if available from FRED)."""
        series_id = self.SERIES["triparty_volume"]["id"]
        df = self._fetch_series(series_id, start_date=start_date)
        df = df.rename(columns={"value": "triparty_volume"})
        return df

    def get_repo_history(
        self,
        lookback_days: int = 730,
    ) -> pd.DataFrame:
        """
        Get complete repo market history.
        
        Returns DataFrame with columns:
        - date
        - sofr (percent)
        - gc_repo (percent)
        - rrp (millions -> converted to rrp_on billions in snapshot)
        - sofr_deviation (basis points)
        """
        end = datetime.today()
        start = end - timedelta(days=lookback_days)
        start_str = start.strftime("%Y-%m-%d")

        logger.info(f"Fetching repo market data from {start_str}")

        # Fetch all series
        sofr = self.get_sofr(start_str)
        gc = self.get_gc_repo(start_str)
        rrp = self.get_rrp(start_str)

        if sofr.empty:
            logger.error("Failed to fetch SOFR")
            return pd.DataFrame()

        # Merge all series
        df = sofr.copy()
        
        if not gc.empty:
            df = df.merge(gc, on="date", how="left")
        if not rrp.empty:
            df = df.merge(rrp, on="date", how="left")
            # Convert to billions and rename for dashboard compatibility
            df['rrp_on'] = df['rrp'] / 1000.0

        # Forward fill
        for col in df.columns:
            if col != "date":
                df[col] = df[col].ffill()

        # Calculate SOFR deviation (key stress indicator)
        sofr_mean = df["sofr"].mean()
        df["sofr_deviation"] = (df["sofr"] - sofr_mean) * 100  # In basis points

        df = df.sort_values("date").reset_index(drop=True)
        
        logger.info(f"Repo market data: {len(df)} rows")
        
        return df

    def calculate_stress_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate repo stress metrics.
        
        Adds columns:
        - sofr_z_score: Z-score of SOFR vs history
        - stress_level: "Normal", "Elevated", "Stress"
        """
        if df.empty or "sofr" not in df.columns:
            return df

        df = df.copy()

        # Calculate SOFR z-score
        sofr_mean = df["sofr"].mean()
        sofr_std = df["sofr"].std()

        if sofr_std > 0:
            df["sofr_z_score"] = (df["sofr"] - sofr_mean) / sofr_std
        else:
            df["sofr_z_score"] = 0.0

        # Classify stress
        def classify_stress(z):
            abs_z = abs(z)
            if abs_z < 1.0:
                return "Normal"
            elif abs_z < 2.0:
                return "Elevated"
            else:
                return "Stress"

        df["stress_level"] = df["sofr_z_score"].apply(classify_stress)

        return df

    def get_full_snapshot(self, lookback_days: int = 730) -> dict:
        """
        Get complete repo market snapshot.
        
        Returns dict with ALL keys needed by dashboard:
        - latest_date: str
        - sofr: float (current SOFR rate)
        - gc_repo: float (current GC repo rate)
        - rrp_volume: float (RRP in billions) ← Dashboard key
        - rrp_billions: float (alias)
        - sofr_z_score: float
        - stress_level: str
        - repo_df: pd.DataFrame ← CRITICAL for charts
        - history: list of dicts
        """
        df = self.get_repo_history(lookback_days)
        
        if df.empty:
            return {
                "error": "No repo market data available",
                "latest_date": None,
                "sofr": None,
                "gc_repo": None,
                "rrp_volume": None,
                "rrp_billions": None,
                "sofr_z_score": None,
                "stress_level": "UNKNOWN",
                "repo_df": pd.DataFrame(),
                "history": []
            }

        df = self.calculate_stress_metrics(df)

        latest = df.iloc[-1]
        latest_date = latest["date"].strftime("%Y-%m-%d")
        
        # Get values with proper null handling
        sofr_val = float(latest["sofr"]) if pd.notna(latest["sofr"]) else None
        gc_val = float(latest.get("gc_repo", 0)) if "gc_repo" in df.columns and pd.notna(latest.get("gc_repo")) else None
        rrp_val = float(latest.get("rrp_on", 0)) if "rrp_on" in df.columns and pd.notna(latest.get("rrp_on")) else 0.0
        z_score = float(latest["sofr_z_score"]) if pd.notna(latest["sofr_z_score"]) else None
        stress = latest["stress_level"]

        snapshot = {
            "latest_date": latest_date,
            "sofr": sofr_val,
            "gc_repo": gc_val,
            "rrp_volume": rrp_val,  # Dashboard expects this key
            "rrp_billions": rrp_val,  # Alias for compatibility
            "sofr_z_score": z_score,
            "stress_level": stress,
            "repo_df": df,  # ← CRITICAL: DataFrame for charts
            "history": df.to_dict('records'),
        }

        logger.info(
            f"Repo: SOFR={sofr_val:.2f}%, "
            f"z={z_score:+.2f}, "
            f"stress={stress}"
        )

        return snapshot