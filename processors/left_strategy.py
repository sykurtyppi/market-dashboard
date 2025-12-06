"""
LEFT Strategy implementation
Based on credit spreads and 330-day EMA
"""

import pandas as pd
from typing import Dict, Optional
from datetime import datetime

class LEFTStrategy:
    """
    LEFT Strategy: Leveraged ETF Trading based on credit spreads
    
    Entry: Credit spreads fall 35% below 330-day EMA
    Exit: Credit spreads rise 40% above 330-day EMA
    """
    
    def __init__(self, ema_period: int = 330, entry_threshold: float = 0.65, exit_threshold: float = 1.40):
        self.ema_period = ema_period
        self.entry_threshold = entry_threshold  # 0.65 = 35% below EMA
        self.exit_threshold = exit_threshold     # 1.40 = 40% above EMA
    
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


# Test function
if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from data_collectors.fred_collector import FREDCollector
    
    print("\n" + "="*80)
    print("LEFT STRATEGY TEST")
    print("="*80)
    
    # Fetch data
    collector = FREDCollector()
    hyg_data = collector.get_series('BAMLH0A0HYM2', start_date='2023-01-01')
    
    if not hyg_data.empty:
        # Calculate signal
        strategy = LEFTStrategy()
        signal = strategy.calculate_signal(hyg_data)
        
        print(f"\n Current Market Status:")
        print(f"  Date:              {signal['date']}")
        print(f"  HYG OAS:           {signal['current_spread']:.4f}%")
        print(f"  330-Day EMA:       {signal['ema_330']:.4f}%")
        print(f"  Ratio:             {signal['ratio']:.4f}")
        print(f"  Distance from EMA: {signal['pct_from_ema']:+.2f}%")
        print(f"\n Signal:           {signal['signal']}")
        print(f"  Strength:          {signal['strength']:.1f}/100")
        
        # Show historical
        print(f"\n Generating historical signals...")
        historical = strategy.get_historical_signals(hyg_data)
        
        print(f"\nLast 5 signals:")
        print(historical.tail().to_string(index=False))
        
        # Signal counts
        signal_counts = historical['signal'].value_counts()
        print(f"\n Signal Distribution:")
        for signal_type, count in signal_counts.items():
            pct = (count / len(historical)) * 100
            print(f"  {signal_type:10s} {count:4d} ({pct:5.1f}%)")
    
    print("\n" + "="*80)
    print("✓ LEFT STRATEGY TEST COMPLETE")
    print("="*80 + "\n")