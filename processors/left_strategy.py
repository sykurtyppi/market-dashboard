"""
LEFT Strategy implementation
Based on credit spreads and 330-day EMA

Source: Larry McMillan's "Buy at Extreme Lows" research

Methodology:
    The strategy compares current credit spread to its 330-day EMA.
    The RATIO = current_spread / ema_330

    When spreads are LOW relative to EMA (ratio < 1.0):
        - Credit conditions are TIGHT (favorable)
        - Risk appetite is HIGH
        - Bullish for equities

    When spreads are HIGH relative to EMA (ratio > 1.0):
        - Credit conditions are LOOSE/stressed
        - Risk appetite is LOW
        - Bearish for equities
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

    RATIO DEFINITION:
        ratio = current_spread / ema_330

        ratio = 0.65 means spread is 35% BELOW its EMA (spread is LOW = bullish)
        ratio = 1.00 means spread equals its EMA (neutral)
        ratio = 1.40 means spread is 40% ABOVE its EMA (spread is HIGH = bearish)

    SIGNAL LOGIC:
        BUY when ratio <= 0.65:
            Credit spreads have compressed 35%+ below average
            → Credit markets are calm/tight → bullish for equities

        SELL when ratio >= 1.40:
            Credit spreads have widened 40%+ above average
            → Credit stress elevated → bearish for equities

    NOTE: This is CONTRARIAN to credit spread direction:
        - Low spreads (ratio << 1.0) = BUY signal (market complacent but supportive)
        - High spreads (ratio >> 1.0) = SELL signal (market stressed)

    Parameters loaded from config/parameters.yaml

    Historical Basis (330-day EMA):
        330 trading days ≈ 15.7 calendar months ≈ 1.3 years
        This captures a full credit cycle while smoothing seasonal noise.
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
