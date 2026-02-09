"""
MOVE Index Collector
Fetches MOVE Index data (Merrill Option Volatility Estimate - Treasury volatility)

HISTORICAL BASIS FOR THRESHOLDS:
    The MOVE Index measures implied volatility of Treasury options.
    Historical statistics (2000-2024):
        - Mean: ~107
        - Median: ~97
        - 25th percentile: ~80
        - 75th percentile: ~120
        - 90th percentile: ~150
        - Max (COVID panic): ~164 (March 2020)
        - Max (2023 SVB crisis): ~198

    Threshold calibration:
        LOW (<80):       Below 25th percentile - calm Treasury markets
        NORMAL (80-120): Middle 50% of distribution - typical conditions
        ELEVATED (120-150): 75th-90th percentile - increased uncertainty
        HIGH (>150):     Above 90th percentile - Treasury market stress/crisis

    These thresholds are based on historical distribution, not arbitrary cutoffs.

Parameters loaded from config/parameters.yaml
"""

import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, Optional

from config import cfg

logger = logging.getLogger(__name__)


class MOVECollector:
    """
    Collector for MOVE Index (Treasury volatility)

    The MOVE Index is to Treasuries what VIX is to equities:
    - Measures implied volatility from Treasury options
    - High MOVE = bond market stress/uncertainty
    - Low MOVE = calm fixed income markets

    Key levels (based on 2000-2024 historical distribution):
        < 80:  LOW stress (below 25th percentile)
        80-120: NORMAL (middle 50% of observations)
        120-150: ELEVATED (75th-90th percentile)
        > 150: HIGH stress (above 90th percentile, crisis territory)

    Notable peaks:
        - March 2020 (COVID): 164
        - March 2023 (SVB crisis): 198
        - 2008 Financial Crisis: ~264
    """
    
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
                    series_col = move_data.columns[1] if len(move_data.columns) > 1 else "MOVE"
                    df = move_data.rename(columns={series_col: 'move'}).copy()
                    df['date'] = pd.to_datetime(df['date']).dt.date
                    df['source'] = 'FRED'
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
            df['source'] = 'Yahoo'
            
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

            # Calculate percentile (with division by zero protection)
            df_len = len(df)
            percentile = (df['move'] < move_value).sum() / df_len * 100 if df_len > 0 else 50.0
            
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
