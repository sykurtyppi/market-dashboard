"""
LEFT Strategy implementation
Based on credit spreads and 330-day EMA

Source: Larry McMillan's "Buy at Extreme Lows" research
"""

import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Optional

from config import cfg

logger = logging.getLogger(__name__)


class LEFTStrategy:
    """
    LEFT Strategy: Leveraged ETF Trading based on credit spreads

    Entry: Credit spreads fall 35% below 330-day EMA (ratio <= 0.65)
    Exit: Credit spreads rise 40% above 330-day EMA (ratio >= 1.40)

    Parameters loaded from config/parameters.yaml
    """

    def __init__(
        self,
        ema_period: Optional[int] = None,
        entry_threshold: Optional[float] = None,
        exit_threshold: Optional[float] = None
    ):
        # Load from config if not provided
        left_cfg = cfg.credit.left_strategy
        self.ema_period = ema_period or left_cfg.ema_period
        self.entry_threshold = entry_threshold or left_cfg.entry_threshold
        self.exit_threshold = exit_threshold or left_cfg.exit_threshold
    
    def calculate_signal(self, credit_spread_data: pd.DataFrame) -> Dict:
        """
        Calculate LEFT strategy signal
        
        Args:
            credit_spread_data: DataFrame with 'date' and spread value columns
        
        Returns:
            Dictionary with signal and metadata
        """
        if credit_spread_data.empty or len(credit_spread_data) < self.ema_period:
            return {
                'signal': 'INSUFFICIENT_DATA',
                'reason': f'Need at least {self.ema_period} days of data'
            }
        
        # Get the spread column (second column after date)
        spread_col = credit_spread_data.columns[1]
        
        # Calculate 330-day EMA
        credit_spread_data['ema_330'] = credit_spread_data[spread_col].ewm(
            span=self.ema_period, 
            adjust=False
        ).mean()
        
        # Get current values
        current_row = credit_spread_data.iloc[-1]
        current_spread = current_row[spread_col]
        current_ema = current_row['ema_330']

        # Guard against division by zero
        if current_ema == 0 or pd.isna(current_ema):
            logger.warning("EMA is zero or NaN, cannot calculate ratio")
            return {
                'signal': 'INSUFFICIENT_DATA',
                'reason': 'EMA value is zero or invalid',
                'strength': 0,
                'current_spread': float(current_spread) if pd.notna(current_spread) else None,
                'ema_330': None,
                'ratio': None,
                'pct_from_ema': None,
                'date': current_row['date'].strftime('%Y-%m-%d'),
                'days_of_data': len(credit_spread_data)
            }

        # Calculate ratio (current / EMA)
        ratio = current_spread / current_ema
        
        # Determine signal
        if ratio <= self.entry_threshold:
            signal = 'BUY'
            strength = min(100, ((self.entry_threshold - ratio) / self.entry_threshold) * 100)
        elif ratio >= self.exit_threshold:
            signal = 'SELL'
            strength = min(100, ((ratio - self.exit_threshold) / self.exit_threshold) * 100)
        else:
            signal = 'NEUTRAL'
            # Calculate how far between thresholds
            range_size = self.exit_threshold - self.entry_threshold
            position = (ratio - self.entry_threshold) / range_size
            strength = 50 + (position - 0.5) * 50  # Map to 25-75 range
        
        return {
            'signal': signal,
            'strength': float(strength),
            'current_spread': float(current_spread),
            'ema_330': float(current_ema),
            'ratio': float(ratio),
            'pct_from_ema': float((ratio - 1) * 100),
            'date': current_row['date'].strftime('%Y-%m-%d'),
            'days_of_data': len(credit_spread_data)
        }
    
    def get_historical_signals(self, credit_spread_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate historical signals for backtesting
        
        Returns:
            DataFrame with signals for each date
        """
        if credit_spread_data.empty or len(credit_spread_data) < self.ema_period:
            return pd.DataFrame()
        
        spread_col = credit_spread_data.columns[1]
        
        # Calculate EMA
        credit_spread_data['ema_330'] = credit_spread_data[spread_col].ewm(
            span=self.ema_period, 
            adjust=False
        ).mean()
        
        # Calculate ratio
        credit_spread_data['ratio'] = credit_spread_data[spread_col] / credit_spread_data['ema_330']
        
        # Generate signals
        def get_signal(ratio):
            if pd.isna(ratio):
                return 'NEUTRAL'
            elif ratio <= self.entry_threshold:
                return 'BUY'
            elif ratio >= self.exit_threshold:
                return 'SELL'
            else:
                return 'NEUTRAL'
        
        credit_spread_data['signal'] = credit_spread_data['ratio'].apply(get_signal)
        
        return credit_spread_data[['date', spread_col, 'ema_330', 'ratio', 'signal']]
