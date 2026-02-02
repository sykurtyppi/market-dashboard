"""
Enhanced Repo Market Data Collector with IORB
Collects SOFR, IORB, and calculates key spreads
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
import io

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


class RepoCollector:
    """
    Collect repo market data including SOFR, IORB, and RRP
    
    Key data sources:
    - SOFR: Federal Reserve (primary repo rate)
    - IORB: Interest on Reserve Balances (Fed floor rate)
    - RRP: Reverse Repo Program volume
    """
    
    FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def fetch_fred_series(
        self, 
        series_id: str, 
        days_back: int = 730
    ) -> pd.DataFrame:
        """
        Fetch data from FRED
        
        Args:
            series_id: FRED series identifier
            days_back: Days of history to fetch
            
        Returns:
            DataFrame with date and value columns
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            params = {
                "id": series_id,
                "cosd": start_date.strftime("%Y-%m-%d"),
                "coed": end_date.strftime("%Y-%m-%d"),
            }
            
            response = requests.get(self.FRED_BASE, params=params, timeout=10)
            response.raise_for_status()
            
            # Read CSV - FRED uses first column as date, unnamed
            df = pd.read_csv(
                io.StringIO(response.text),
                parse_dates=[0],  # Parse first column as date
                index_col=0,  # Use first column as index
            )
            
            # Reset index to make date a column
            df = df.reset_index()
            
            # Rename columns - first is date, second is value
            df.columns = ["date", "value"]
            
            # Convert to numeric, handle missing values
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna()
            
            self.logger.info(f"Fetched {len(df)} rows for {series_id}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching {series_id}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def get_repo_history(self, days_back: int = 730) -> pd.DataFrame:
        """
        Get comprehensive repo market history
        
        Returns DataFrame with:
        - date
        - sofr: Secured Overnight Financing Rate
        - iorb: Interest on Reserve Balances  
        - sofr_iorb_spread: SOFR - IORB (key liquidity indicator)
        - rrp_on: Overnight Reverse Repo volume ($B)
        - sofr_z_score: SOFR z-score vs historical
        """
        try:
            # Fetch SOFR
            sofr_df = self.fetch_fred_series("SOFR", days_back)
            if sofr_df.empty:
                self.logger.error("Failed to fetch SOFR")
                return pd.DataFrame()
            
            sofr_df = sofr_df.rename(columns={"value": "sofr"})
            
            # Fetch IORB
            iorb_df = self.fetch_fred_series("IORB", days_back)
            if not iorb_df.empty:
                iorb_df = iorb_df.rename(columns={"value": "iorb"})
            else:
                self.logger.warning("IORB data unavailable")
                iorb_df = pd.DataFrame()
            
            # Fetch RRP
            rrp_df = self.fetch_fred_series("RRPONTSYD", days_back)
            if not rrp_df.empty:
                rrp_df = rrp_df.rename(columns={"value": "rrp_on"})
                # Convert to billions
                rrp_df["rrp_on"] = rrp_df["rrp_on"] / 1000.0
            else:
                self.logger.warning("RRP data unavailable")
                rrp_df = pd.DataFrame()
            
            # Merge all data
            df = sofr_df.copy()
            
            if not iorb_df.empty:
                df = df.merge(iorb_df, on="date", how="left")
                # Forward fill IORB (it changes infrequently)
                df["iorb"] = df["iorb"].ffill()
                
                # Calculate SOFR-IORB spread (KEY LIQUIDITY INDICATOR)
                df["sofr_iorb_spread"] = df["sofr"] - df["iorb"]
            
            if not rrp_df.empty:
                df = df.merge(rrp_df, on="date", how="left")
                df["rrp_on"] = df["rrp_on"].ffill()
            
            # Calculate SOFR z-score
            if len(df) > 252:
                df["sofr_z_score"] = (
                    df["sofr"] - df["sofr"].rolling(252).mean()
                ) / df["sofr"].rolling(252).std()
            else:
                df["sofr_z_score"] = 0.0
            
            df = df.sort_values("date").reset_index(drop=True)
            
            self.logger.info(f"Compiled repo history: {len(df)} rows")
            return df
            
        except Exception as e:
            self.logger.error(f"Error compiling repo history: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def get_full_snapshot(self) -> Dict:
        """
        Get current repo market snapshot
        
        Returns:
            Dict with current values and historical DataFrame
        """
        try:
            repo_df = self.get_repo_history(days_back=730)
            
            if repo_df.empty:
                return {}
            
            latest = repo_df.iloc[-1]
            
            snapshot = {
                "date": latest["date"],
                "sofr": float(latest["sofr"]),
                "sofr_z_score": float(latest.get("sofr_z_score", 0.0)),
                "rrp_volume": float(latest.get("rrp_on", 0.0)),
                "repo_df": repo_df,
            }
            
            # Add IORB data if available
            if "iorb" in latest and pd.notna(latest["iorb"]):
                snapshot["iorb"] = float(latest["iorb"])
                snapshot["sofr_iorb_spread"] = float(latest["sofr_iorb_spread"])
                
                # Classify liquidity based on spread
                spread = snapshot["sofr_iorb_spread"]
                if spread <= 0:
                    snapshot["liquidity_status"] = "ABUNDANT"
                    snapshot["liquidity_color"] = "#4CAF50"
                elif spread <= 5:
                    snapshot["liquidity_status"] = "AMPLE"
                    snapshot["liquidity_color"] = "#4CAF50"
                elif spread <= 15:
                    snapshot["liquidity_status"] = "NORMAL"
                    snapshot["liquidity_color"] = "#FFC107"
                elif spread <= 30:
                    snapshot["liquidity_status"] = "TIGHTENING"
                    snapshot["liquidity_color"] = "#FF9800"
                else:
                    snapshot["liquidity_status"] = "STRESS"
                    snapshot["liquidity_color"] = "#F44336"
            
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Error creating repo snapshot: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {}
