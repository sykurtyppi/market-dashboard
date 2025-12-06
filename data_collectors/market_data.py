"""
Market Data Collector
Collects general market data (VIX, SPY, etc.) for Phase 2 analysis
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class MarketDataCollector:
    """Collector for general market data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_vix_history(self, lookback_days: int = 365) -> pd.DataFrame:
        """
        Get VIX historical data
        
        Args:
            lookback_days: Number of days to look back
            
        Returns:
            DataFrame with columns: date, close
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            vix = yf.Ticker("^VIX")
            hist = vix.history(start=start_date, end=end_date)
            
            if hist.empty:
                self.logger.warning("No VIX data returned from yfinance")
                return pd.DataFrame()
            
            # Format for consistency
            df = pd.DataFrame({
                'date': hist.index,
                'close': hist['Close'].values
            })
            
            df['date'] = pd.to_datetime(df['date']).dt.date
            
            self.logger.info(f"Retrieved {len(df)} days of VIX data")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching VIX history: {e}")
            return pd.DataFrame()
    
    def get_spy_history(self, lookback_days: int = 365) -> pd.DataFrame:
        """
        Get SPY historical data
        
        Args:
            lookback_days: Number of days to look back
            
        Returns:
            DataFrame with columns: date, close, volume
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            spy = yf.Ticker("SPY")
            hist = spy.history(start=start_date, end=end_date)
            
            if hist.empty:
                self.logger.warning("No SPY data returned from yfinance")
                return pd.DataFrame()
            
            # Format for consistency
            df = pd.DataFrame({
                'date': hist.index,
                'close': hist['Close'].values,
                'volume': hist['Volume'].values
            })
            
            df['date'] = pd.to_datetime(df['date']).dt.date
            
            self.logger.info(f"Retrieved {len(df)} days of SPY data")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching SPY history: {e}")
            return pd.DataFrame()
    
    def get_current_vix(self) -> float:
        """Get current VIX spot price"""
        try:
            vix = yf.Ticker("^VIX")
            data = vix.history(period="1d")
            
            if data.empty:
                return None
            
            return float(data['Close'].iloc[-1])
            
        except Exception as e:
            self.logger.error(f"Error fetching current VIX: {e}")
            return None
    
    def get_current_spy(self) -> float:
        """Get current SPY price"""
        try:
            spy = yf.Ticker("SPY")
            data = spy.history(period="1d")
            
            if data.empty:
                return None
            
            return float(data['Close'].iloc[-1])
            
        except Exception as e:
            self.logger.error(f"Error fetching current SPY: {e}")
            return None


# For backward compatibility
def get_vix_history(lookback_days: int = 365) -> pd.DataFrame:
    """Legacy function - use MarketDataCollector instead"""
    collector = MarketDataCollector()
    return collector.get_vix_history(lookback_days)