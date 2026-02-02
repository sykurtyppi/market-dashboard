"""
FRED Data Collector - UPDATED with correct API structure
Fetches credit spreads, interest rates, and macro data from Federal Reserve

Parameters loaded from config/parameters.yaml
"""

import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.retry_utils import exponential_backoff_retry
from config import cfg

load_dotenv()

logger = logging.getLogger(__name__)

class FREDCollector:
    """Collects data from FRED API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('FRED_API_KEY')
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"
        
        if not self.api_key:
            raise ValueError(
                "FRED API key not found. Get one at https://fred.stlouisfed.org/docs/api/api_key.html"
            )
    
    @exponential_backoff_retry(max_retries=3, base_delay=2.0, max_delay=30.0)
    def get_series(
        self, 
        series_id: str, 
        start_date: Optional[str] = None,
        limit: int = 100000
    ) -> pd.DataFrame:
        """
        Fetch a single series from FRED
        
        Args:
            series_id: FRED series ID (e.g., 'BAMLH0A0HYM2')
            start_date: Start date in YYYY-MM-DD format (default: 2 years ago)
            limit: Maximum observations to return (default: 100000)
        
        Returns:
            DataFrame with date and value columns
        """
        if not start_date:
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json',
            'observation_start': start_date,
            'limit': limit
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'observations' not in data:
                logger.warning(f"No observations found for {series_id}")
                return pd.DataFrame()
            
            # Parse observations
            observations = data['observations']
            
            # Convert to DataFrame
            df = pd.DataFrame(observations)
            
            if df.empty:
                logger.warning(f"Empty data for {series_id}")
                return pd.DataFrame()
            
            # Clean and process data
            df['date'] = pd.to_datetime(df['date'])
            
            # Handle missing values (FRED uses "." for missing)
            df['value'] = df['value'].replace('.', pd.NA)
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            # Drop rows with missing values
            df = df.dropna(subset=['value'])
            
            # Rename value column to series_id for clarity
            df = df[['date', 'value']].rename(columns={'value': series_id})
            
            logger.info(f"Fetched {len(df)} observations for {series_id}")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching FRED data for {series_id}: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error for {series_id}: {e}")
            return pd.DataFrame()
    
    def get_all_indicators(self, lookback_days: int = 730) -> Dict[str, pd.DataFrame]:
        """
        Fetch all key market indicators
        
        Args:
            lookback_days: How many days of history to fetch (default: 2 years)
        
        Returns:
            Dictionary of DataFrames keyed by indicator name
        """
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        
        indicators = {
            'credit_spread_hy': 'BAMLH0A0HYM2',    # High Yield OAS
            'credit_spread_ig': 'BAMLC0A0CMEY',    # Investment Grade OAS
            'treasury_10y': 'DGS10',                # 10-Year Treasury
            'treasury_3m': 'DGS3MO',                # 3-Month Treasury
            'treasury_2y': 'DGS2',                  # 2-Year Treasury
            'fed_funds': 'DFF',                     # Fed Funds Rate
            'm2_supply': 'M2SL',                    # M2 Money Supply (Monthly)
            'corporate_aaa': 'DAAA',                # AAA Corporate Bond Yield
            'corporate_baa': 'DBAA',                # BAA Corporate Bond Yield
        }
        
        results = {}

        logger.info(f"Fetching FRED data (last {lookback_days} days)")

        for name, series_id in indicators.items():
            logger.debug(f"Fetching {name} ({series_id})...")
            df = self.get_series(series_id, start_date=start_date)

            if not df.empty:
                results[name] = df
                # Add a small delay to be nice to FRED's servers (from config)
                time.sleep(cfg.data_collection.rate_limits.fred_delay_seconds)
            else:
                logger.warning(f"No data returned for {name}")

        logger.info(f"Fetched {len(results)} FRED indicators")

        return results
    
    def get_latest_values(self) -> Dict[str, Dict]:
        """
        Get the most recent value for each indicator
        
        Returns:
            Dictionary with latest value, date, and metadata
        """
        data = self.get_all_indicators(lookback_days=90)  # Only need recent data
        latest = {}
        
        for name, df in data.items():
            if not df.empty:
                last_row = df.iloc[-1]
                latest[name] = {
                    'value': float(last_row[df.columns[1]]),
                    'date': last_row['date'].strftime('%Y-%m-%d'),
                    'series_id': df.columns[1]
                }
        
        return latest
    
    def calculate_credit_spread_signals(self) -> Dict:
        """
        Calculate LEFT strategy signals based on credit spreads
        
        Returns:
            Dictionary with signals and metadata
        """
        # Get 2 years of HYG OAS data for EMA calculation
        hyg_data = self.get_series('BAMLH0A0HYM2', 
                                   start_date=(datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d'))
        
        if hyg_data.empty:
            return {'error': 'Could not fetch credit spread data'}
        
        # Calculate 330-day EMA
        hyg_data['ema_330'] = hyg_data['BAMLH0A0HYM2'].ewm(span=330, adjust=False).mean()
        
        # Get current values
        current_spread = hyg_data['BAMLH0A0HYM2'].iloc[-1]
        current_ema = hyg_data['ema_330'].iloc[-1]
        
        # Calculate percentage from EMA
        pct_from_ema = (current_spread - current_ema) / current_ema
        
        # LEFT strategy signals
        signal = 'NEUTRAL'
        if pct_from_ema < -0.35:  # 35% below EMA
            signal = 'BUY'
        elif pct_from_ema > 0.40:  # 40% above EMA
            signal = 'SELL'
        
        return {
            'current_spread': float(current_spread),
            'ema_330': float(current_ema),
            'pct_from_ema': float(pct_from_ema * 100),
            'signal': signal,
            'date': hyg_data['date'].iloc[-1].strftime('%Y-%m-%d')
        }
    
    def get_baa_aaa_spread(self) -> Optional[float]:
        """
        Calculate traditional Baa-Aaa credit spread
        
        Returns:
            Spread in percentage points
        """
        baa = self.get_series('DBAA', start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'))
        aaa = self.get_series('DAAA', start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'))
        
        if baa.empty or aaa.empty:
            return None

        baa_latest = baa['DBAA'].iloc[-1]
        aaa_latest = aaa['DAAA'].iloc[-1]

        return float(baa_latest - aaa_latest)

    def get_data_with_status(self) -> Dict:
        """
        Get all FRED data with detailed status tracking.

        Returns:
            Dict with:
                - data: The actual indicator data
                - status: "ok", "partial", "error", "unavailable"
                - fetched_count: Number of indicators successfully fetched
                - failed_count: Number of indicators that failed
                - failed_indicators: List of failed indicator names
                - timestamp: When this fetch occurred
                - errors: List of error messages
        """
        from utils.data_status import DataStatus, DataResult

        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

        indicators = {
            'credit_spread_hy': 'BAMLH0A0HYM2',
            'credit_spread_ig': 'BAMLC0A0CMEY',
            'treasury_10y': 'DGS10',
            'treasury_2y': 'DGS2',
            'fed_funds': 'DFF',
        }

        results = {}
        failed = []
        errors = []

        for name, series_id in indicators.items():
            try:
                df = self.get_series(series_id, start_date=start_date)
                if not df.empty:
                    results[name] = {
                        'value': float(df.iloc[-1][series_id]),
                        'date': df.iloc[-1]['date'].strftime('%Y-%m-%d'),
                        'series_id': series_id
                    }
                else:
                    failed.append(name)
                    errors.append(f"{name}: Empty response from FRED API")
                time.sleep(cfg.data_collection.rate_limits.fred_delay_seconds)
            except Exception as e:
                failed.append(name)
                errors.append(f"{name}: {str(e)}")
                logger.error(f"Failed to fetch {name}: {e}")

        # Determine overall status
        total = len(indicators)
        fetched = len(results)

        if fetched == 0:
            status = DataStatus.UNAVAILABLE
        elif fetched < total:
            status = DataStatus.PARTIAL
        else:
            status = DataStatus.OK

        return {
            'data': results,
            'status': status.value,
            'fetched_count': fetched,
            'failed_count': len(failed),
            'total_count': total,
            'failed_indicators': failed,
            'timestamp': datetime.now().isoformat(),
            'errors': errors,
            'is_complete': fetched == total
        }
