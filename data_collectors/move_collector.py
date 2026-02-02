"""
MOVE Index Collector
Fetches MOVE Index data (Merrill Option Volatility Estimate - Treasury volatility)

Parameters loaded from config/parameters.yaml
"""

import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, Optional

from config import cfg

logger = logging.getLogger(__name__)


class MOVECollector:
    """Collector for MOVE Index (Treasury volatility)"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_move_data(self, lookback_days: int = 730) -> pd.DataFrame:
        """
        Get MOVE Index data
        
        Args:
            lookback_days: Number of days to look back
            
        Returns:
            DataFrame with columns: date, move
        """
        try:
            # Try FRED first
            from data_collectors.fred_collector import FREDCollector
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            self.logger.info(f"Trying FRED for MOVE Index from {start_date.date()}")
            
            try:
                fred = FREDCollector()
                move_data = fred.get_series("MOVE", start_date=start_date.strftime('%Y-%m-%d'))
                
                if not move_data.empty:
                    self.logger.info(f"FRED: Fetched {len(move_data)} MOVE observations")
                    df = pd.DataFrame({
                        'date': pd.to_datetime(move_data.index).date,
                        'move': move_data.values
                    })
                    return df
                else:
                    self.logger.warning("FRED returned empty DataFrame for MOVE")
            except Exception as e:
                self.logger.warning(f"FRED MOVE fetch failed: {e}")
            
            # Fallback to Yahoo Finance
            self.logger.info(f"Trying Yahoo Finance for ^MOVE from {start_date.date()}")
            
            import yfinance as yf
            ticker = yf.Ticker("^MOVE")
            hist = ticker.history(start=start_date, end=end_date)
            
            if hist.empty:
                self.logger.error("Yahoo Finance: No MOVE data returned")
                return pd.DataFrame()
            
            self.logger.info(f"Yahoo: Fetched {len(hist)} MOVE observations")
            
            df = pd.DataFrame({
                'date': pd.to_datetime(hist.index).date,
                'move': hist['Close'].values
            })
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching MOVE data: {e}")
            return pd.DataFrame()
    
    def get_full_snapshot(self) -> Dict:
        """
        Get complete MOVE snapshot with analysis
        
        Returns:
            dict with MOVE data and analysis
        """
        try:
            df = self.get_move_data(lookback_days=730)
            
            if df.empty:
                self.logger.error("No MOVE data available for snapshot")
                return {}
            
            latest = df.iloc[-1]
            move_value = float(latest['move'])
            
            # Calculate percentile
            percentile = (df['move'] < move_value).sum() / len(df) * 100
            
            # Classify stress level using config thresholds
            move_cfg = cfg.treasury.move
            if move_value < move_cfg.low_max:
                stress_level = "LOW"
                stress_color = "#4CAF50"
            elif move_value < move_cfg.normal_max:
                stress_level = "NORMAL"
                stress_color = "#8BC34A"
            elif move_value < move_cfg.elevated_max:
                stress_level = "ELEVATED"
                stress_color = "#FF9800"
            else:
                stress_level = "HIGH"
                stress_color = "#F44336"
            
            self.logger.info(
                f"MOVE: {move_value:.1f} ({percentile:.0f}th percentile, {stress_level} stress)"
            )
            
            return {
                'latest_date': latest['date'],
                'move': move_value,
                'move_index': move_value,  # FIX: Add this key for dashboard compatibility
                'move_percentile': percentile,
                'percentile': percentile,  # FIX: Add this key too
                'stress_level': stress_level,
                'stress_color': stress_color,
                'move_df': df,  # FIX: Add the DataFrame for charts
                'history': df.to_dict('records'),
                'thresholds': {
                    'low': move_cfg.low_max,
                    'normal': move_cfg.normal_max,
                    'elevated': move_cfg.elevated_max,
                    'high': move_cfg.high_min
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error creating MOVE snapshot: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {}
    
    def get_latest_move(self) -> Optional[float]:
        """Get latest MOVE value only"""
        try:
            df = self.get_move_data(lookback_days=7)
            if df.empty:
                return None
            return float(df.iloc[-1]['move'])
        except Exception as e:
            self.logger.error(f"Error getting latest MOVE: {e}")
            return None


# Backward compatibility
def get_move_index(lookback_days: int = 730) -> pd.DataFrame:
    """Legacy function"""
    collector = MOVECollector()
    return collector.get_move_data(lookback_days)