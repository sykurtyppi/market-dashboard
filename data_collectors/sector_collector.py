"""
Sector Performance Collector
Tracks sector ETF performance for rotation analysis
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict

class SectorCollector:
    """Collects sector performance data"""
    
    def __init__(self):
        self.sectors = {
            'XLK': 'Technology',
            'XLF': 'Financials',
            'XLV': 'Healthcare',
            'XLE': 'Energy',
            'XLY': 'Consumer Discretionary',
            'XLP': 'Consumer Staples',
            'XLI': 'Industrials',
            'XLB': 'Materials',
            'XLU': 'Utilities',
            'XLRE': 'Real Estate',
            'XLC': 'Communication Services'
        }
    
    def get_sector_performance(self, period: str = '1d') -> Dict:
        """
        Get sector performance
        period: '1d', '5d', '1mo', '3mo', 'ytd'
        """
        performance = {}
        
        for ticker, name in self.sectors.items():
            try:
                etf = yf.Ticker(ticker)
                hist = etf.history(period='5d')
                
                if len(hist) >= 2:
                    current = hist['Close'].iloc[-1]
                    previous = hist['Close'].iloc[-2]
                    change_pct = ((current - previous) / previous) * 100
                    
                    performance[ticker] = {
                        'name': name,
                        'price': float(current),
                        'change_pct': float(change_pct)
                    }
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")
                continue
        
        return performance
    
    def get_sector_rankings(self) -> pd.DataFrame:
        """Get sector performance ranked"""
        perf = self.get_sector_performance()
        
        df = pd.DataFrame.from_dict(perf, orient='index')
        df = df.sort_values('change_pct', ascending=False)
        
        return df


if __name__ == "__main__":
    collector = SectorCollector()
    rankings = collector.get_sector_rankings()
    print(rankings)
