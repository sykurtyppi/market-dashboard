"""
Advance-Decline Line Collector using Stooq
FREE, no API key, institutional-grade data
"""

import pandas as pd
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ADLineCollector:
    """
    Collects Advance-Decline Line data from Stooq
    - S&P 500 A/D Line ($ADSPD)
    - NASDAQ A/D Line ($ADCN)
    - NYSE A/D Line ($ADD)
    """
    
    SOURCES = {
        'sp500': "https://stooq.com/q/d/l/?s=$adspd&i=d",
        'nasdaq': "https://stooq.com/q/d/l/?s=$adcn&i=d",
        'nyse': "https://stooq.com/q/d/l/?s=$add&i=d"
    }
    
    def fetch(self, market='sp500', lookback_days=365):
        """Fetch A/D line data for specified market"""
        try:
            url = self.SOURCES[market]
            df = pd.read_csv(url)
            
            # FLEXIBLE COLUMN HANDLING - works with any case
            df.columns = df.columns.str.strip().str.lower()
            
            # Rename columns to standard format
            column_mapping = {}
            for col in df.columns:
                if 'date' in col.lower():
                    column_mapping[col] = 'date'
                elif 'close' in col.lower():
                    column_mapping[col] = 'close'
                elif 'open' in col.lower():
                    column_mapping[col] = 'open'
                elif 'high' in col.lower():
                    column_mapping[col] = 'high'
                elif 'low' in col.lower():
                    column_mapping[col] = 'low'
                elif 'volume' in col.lower() or 'vol' in col.lower():
                    column_mapping[col] = 'volume'
            
            df = df.rename(columns=column_mapping)
            
            # Validate we have required columns
            if 'date' not in df.columns or 'close' not in df.columns:
                logger.error(f"Missing required columns. Found: {df.columns.tolist()}")
                return pd.DataFrame()
            
            # Clean data
            df['date'] = pd.to_datetime(df['date'])
            df = df.dropna(subset=['date', 'close'])
            df = df.sort_values('date')
            
            # Filter to lookback period
            cutoff = datetime.now() - timedelta(days=lookback_days)
            df = df[df['date'] >= cutoff]
            
            logger.info(f"✅ Fetched {len(df)} days of {market.upper()} A/D data")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error fetching {market} A/D line: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def get_all(self, lookback_days=365):
        """Fetch all three A/D lines"""
        return {
            'sp500': self.fetch('sp500', lookback_days),
            'nasdaq': self.fetch('nasdaq', lookback_days),
            'nyse': self.fetch('nyse', lookback_days)
        }
    
    def get_current_breadth(self, market='sp500'):
        """
        Calculate current breadth metrics from A/D line
        Returns latest A/D value and momentum
        """
        df = self.fetch(market, lookback_days=60)
        
        if df.empty:
            logger.warning(f"No data for {market}")
            return None
        
        # Get latest value
        latest = df.iloc[-1]
        prev_week = df.iloc[-6] if len(df) >= 6 else df.iloc[0]
        prev_month = df.iloc[-22] if len(df) >= 22 else df.iloc[0]
        
        # Calculate momentum
        week_change = latest['close'] - prev_week['close']
        month_change = latest['close'] - prev_month['close']
        
        # Calculate 10-day EMA for breadth thrust
        df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
        ema_current = df['ema10'].iloc[-1]
        ema_prev = df['ema10'].iloc[-11] if len(df) >= 11 else df['ema10'].iloc[0]
        
        breadth_thrust = ((ema_current - ema_prev) / abs(ema_prev) * 100) if ema_prev != 0 else 0
        
        return {
            'ad_line': float(latest['close']),
            'date': latest['date'],
            'week_change': float(week_change),
            'month_change': float(month_change),
            'breadth_thrust': float(breadth_thrust),
            'trend': 'Improving' if week_change > 0 else 'Weakening'
        }
    
    def calculate_mcclellan_oscillator(self, market='sp500'):
        """
        Calculate McClellan Oscillator
        (19-day EMA - 39-day EMA) of daily A/D line
        """
        df = self.fetch(market, lookback_days=90)
        
        if df.empty or len(df) < 39:
            logger.warning(f"Insufficient data for McClellan: {len(df)} days")
            return None
        
        # Calculate EMAs
        df['ema19'] = df['close'].ewm(span=19, adjust=False).mean()
        df['ema39'] = df['close'].ewm(span=39, adjust=False).mean()
        df['mcclellan'] = df['ema19'] - df['ema39']
        
        latest = df.iloc[-1]
        
        # Interpretation
        value = float(latest['mcclellan'])
        if value > 100:
            signal = "Strong Bullish"
        elif value > 50:
            signal = "Bullish"
        elif value > -50:
            signal = "Neutral"
        elif value > -100:
            signal = "Bearish"
        else:
            signal = "Strong Bearish"
        
        return {
            'value': value,
            'signal': signal,
            'date': latest['date'],
            'history': df[['date', 'mcclellan']].tail(90)
        }
    
    def get_health_check(self):
        """Check if data source is working"""
        try:
            df = self.fetch('sp500', lookback_days=5)
            
            if df.empty:
                return {
                    'status': 'error',
                    'message': 'No data returned from Stooq'
                }
            
            latest_date = df['date'].max()
            days_old = (datetime.now() - latest_date).days
            
            if days_old > 5:
                return {
                    'status': 'warning',
                    'message': f'Data is {days_old} days old'
                }
            
            return {
                'status': 'ok',
                'message': f'Fresh data ({days_old} days old)',
                'latest_date': latest_date,
                'record_count': len(df)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

