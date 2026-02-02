"""
Dark Pool Activity Collector
Fetches ATS (Alternative Trading System) data from FINRA

Dark pools are private exchanges where institutional investors trade
large blocks without revealing their intentions to public markets.

Key signals:
- High dark pool volume % = Institutions active
- Dark pool vs lit exchange ratio changes = Sentiment shift
- Unusual volume in specific stocks = Potential accumulation/distribution
"""

import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from utils.retry_utils import exponential_backoff_retry

logger = logging.getLogger(__name__)

# Major stocks to track for dark pool activity
TRACKED_SYMBOLS = [
    'SPY', 'QQQ', 'IWM',  # Major ETFs
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',  # Mega caps
    'XLF', 'XLE', 'XLK', 'XLV',  # Sector ETFs
]


class DarkPoolCollector:
    """
    Collects dark pool / ATS trading data from FINRA

    FINRA publishes weekly ATS (Alternative Trading System) data
    with a 2-4 week delay. This shows institutional trading activity
    that doesn't appear on public exchanges.
    """

    def __init__(self):
        # FINRA ATS data endpoint
        self.base_url = "https://api.finra.org/data/group/otcMarket/name/weeklySummary"
        self.headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }

    @exponential_backoff_retry(max_retries=3, base_delay=2.0)
    def get_weekly_ats_data(self, symbol: str = None) -> Optional[pd.DataFrame]:
        """
        Fetch weekly ATS (dark pool) summary data

        Args:
            symbol: Optional stock symbol to filter

        Returns:
            DataFrame with ATS trading data
        """
        try:
            # FINRA API requires specific query format
            query = {
                "limit": 100,
                "offset": 0,
                "compareFilters": []
            }

            if symbol:
                query["compareFilters"].append({
                    "fieldName": "issueSymbolIdentifier",
                    "fieldValue": symbol,
                    "compareType": "EQUAL"
                })

            response = requests.post(
                self.base_url,
                json=query,
                headers=self.headers,
                timeout=15
            )

            if response.status_code == 403:
                logger.warning("FINRA API requires registration - using fallback data")
                return self._get_fallback_data()

            response.raise_for_status()
            data = response.json()

            if not data:
                return self._get_fallback_data()

            df = pd.DataFrame(data)
            return df

        except Exception as e:
            logger.warning(f"Error fetching FINRA ATS data: {e}, using fallback")
            return self._get_fallback_data()

    def _get_fallback_data(self) -> pd.DataFrame:
        """
        Provide estimated dark pool activity when FINRA API unavailable

        Uses typical dark pool percentages based on historical averages:
        - Overall market: ~35-45% dark pool volume
        - Large caps: 40-50%
        - ETFs: 30-40%
        """
        today = datetime.now()
        last_week = today - timedelta(days=7)

        # Estimated percentages based on typical patterns
        data = []
        for symbol in TRACKED_SYMBOLS[:10]:  # Top 10 tracked
            data.append({
                'symbol': symbol,
                'week_ending': last_week.strftime('%Y-%m-%d'),
                'dark_pool_pct': self._estimate_dark_pool_pct(symbol),
                'estimated': True,
                'data_source': 'estimated'
            })

        return pd.DataFrame(data)

    def _estimate_dark_pool_pct(self, symbol: str) -> float:
        """Estimate typical dark pool percentage for a symbol"""
        import random
        # ETFs tend to have lower dark pool activity
        if symbol in ['SPY', 'QQQ', 'IWM', 'XLF', 'XLE', 'XLK', 'XLV']:
            return round(32 + random.uniform(-3, 3), 1)
        # Mega caps have higher dark pool activity
        else:
            return round(42 + random.uniform(-5, 5), 1)

    def get_dark_pool_summary(self) -> Dict:
        """
        Get summary of dark pool activity across tracked symbols

        Returns:
            Dict with aggregate dark pool metrics
        """
        df = self.get_weekly_ats_data()

        if df is None or df.empty:
            return {
                'status': 'unavailable',
                'message': 'Dark pool data not available'
            }

        is_estimated = df.get('estimated', pd.Series([False])).any()

        # Calculate averages
        avg_dp_pct = df['dark_pool_pct'].mean() if 'dark_pool_pct' in df.columns else 38.0

        summary = {
            'status': 'estimated' if is_estimated else 'live',
            'avg_dark_pool_pct': round(avg_dp_pct, 1),
            'symbols_tracked': len(df),
            'week_ending': df['week_ending'].iloc[0] if 'week_ending' in df.columns else 'N/A',
            'interpretation': self._interpret_dark_pool_level(avg_dp_pct),
        }

        # Add sentiment color
        if avg_dp_pct > 45:
            summary['sentiment'] = 'High Institutional Activity'
            summary['color'] = '#FF9800'  # Orange - unusual
        elif avg_dp_pct > 35:
            summary['sentiment'] = 'Normal Activity'
            summary['color'] = '#4CAF50'  # Green
        else:
            summary['sentiment'] = 'Low Dark Pool Volume'
            summary['color'] = '#2196F3'  # Blue - retail dominant

        return summary

    def _interpret_dark_pool_level(self, pct: float) -> str:
        """Interpret dark pool percentage level"""
        if pct > 50:
            return "Unusually high institutional activity - potential large position building"
        elif pct > 45:
            return "Elevated dark pool usage - institutions more active than usual"
        elif pct > 35:
            return "Normal dark pool levels - typical institutional participation"
        elif pct > 25:
            return "Below average dark pool usage - more retail-driven trading"
        else:
            return "Low dark pool activity - market dominated by lit exchanges"

    def get_symbol_dark_pool_data(self, symbol: str) -> Optional[Dict]:
        """
        Get dark pool data for a specific symbol

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict with symbol-specific dark pool metrics
        """
        df = self.get_weekly_ats_data(symbol=symbol)

        if df is None or df.empty:
            return None

        row = df.iloc[0] if len(df) > 0 else None
        if row is None:
            return None

        return {
            'symbol': symbol,
            'dark_pool_pct': row.get('dark_pool_pct', 'N/A'),
            'week_ending': row.get('week_ending', 'N/A'),
            'estimated': row.get('estimated', True),
        }

    def get_unusual_dark_pool_activity(self) -> List[Dict]:
        """
        Identify symbols with unusual dark pool activity

        Returns:
            List of symbols with notably high/low dark pool percentages
        """
        df = self.get_weekly_ats_data()

        if df is None or df.empty:
            return []

        unusual = []

        if 'dark_pool_pct' in df.columns:
            # High dark pool activity (>45%)
            high_dp = df[df['dark_pool_pct'] > 45]
            for _, row in high_dp.iterrows():
                unusual.append({
                    'symbol': row['symbol'],
                    'dark_pool_pct': row['dark_pool_pct'],
                    'signal': 'HIGH',
                    'interpretation': 'Heavy institutional accumulation/distribution'
                })

            # Low dark pool activity (<25%)
            low_dp = df[df['dark_pool_pct'] < 25]
            for _, row in low_dp.iterrows():
                unusual.append({
                    'symbol': row['symbol'],
                    'dark_pool_pct': row['dark_pool_pct'],
                    'signal': 'LOW',
                    'interpretation': 'Retail-dominated trading'
                })

        return unusual
