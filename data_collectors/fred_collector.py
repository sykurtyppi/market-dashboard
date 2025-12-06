"""
FRED Data Collector - UPDATED with correct API structure
Fetches credit spreads, interest rates, and macro data from Federal Reserve
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import os
from dotenv import load_dotenv
import time

load_dotenv()

class FREDCollector:
    """Collects data from FRED API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('FRED_API_KEY')
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"
        
        if not self.api_key:
            raise ValueError(
                "FRED API key not found. Get one at https://fred.stlouisfed.org/docs/api/api_key.html"
            )
    
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
                print(f"Warning: No observations found for {series_id}")
                return pd.DataFrame()
            
            # Parse observations
            observations = data['observations']
            
            # Convert to DataFrame
            df = pd.DataFrame(observations)
            
            if df.empty:
                print(f"Warning: Empty data for {series_id}")
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
            
            print(f"✓ Fetched {len(df)} observations for {series_id}")
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Error fetching FRED data for {series_id}: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"✗ Unexpected error for {series_id}: {e}")
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
        
        print(f"\n{'='*60}")
        print(f"Fetching FRED data (last {lookback_days} days)")
        print(f"{'='*60}")
        
        for name, series_id in indicators.items():
            print(f"Fetching {name} ({series_id})...", end=' ')
            df = self.get_series(series_id, start_date=start_date)
            
            if not df.empty:
                results[name] = df
                # Add a small delay to be nice to FRED's servers
                time.sleep(0.1)
            else:
                print(f"  ⚠ No data returned")
        
        print(f"{'='*60}\n")
        
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


# Test function
if __name__ == "__main__":
    print("\n" + "="*80)
    print("FRED DATA COLLECTOR TEST")
    print("="*80)
    
    try:
        collector = FREDCollector()
        
        # Test 1: Get latest values
        print("\n📊 TEST 1: Latest Indicator Values")
        print("-" * 80)
        latest = collector.get_latest_values()
        
        for key, data in latest.items():
            print(f"{key:20s} | {data['value']:8.4f} | Date: {data['date']}")
        
        # Test 2: Calculate LEFT strategy signals
        print("\n📈 TEST 2: LEFT Strategy Signals")
        print("-" * 80)
        signals = collector.calculate_credit_spread_signals()
        
        if 'error' not in signals:
            print(f"Current HYG OAS:     {signals['current_spread']:.4f}%")
            print(f"330-Day EMA:         {signals['ema_330']:.4f}%")
            print(f"Distance from EMA:   {signals['pct_from_ema']:+.2f}%")
            print(f"Signal:              {signals['signal']}")
            print(f"As of:               {signals['date']}")
        else:
            print(f"Error: {signals['error']}")
        
        # Test 3: Traditional Baa-Aaa spread
        print("\n📉 TEST 3: Traditional Credit Spread")
        print("-" * 80)
        baa_aaa = collector.get_baa_aaa_spread()
        if baa_aaa:
            print(f"Baa-Aaa Spread:      {baa_aaa:.4f}%")
        else:
            print("Could not calculate Baa-Aaa spread")
        
        print("\n" + "="*80)
        print("✓ ALL TESTS COMPLETED")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}\n")