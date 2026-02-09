"""
Fear & Greed Index Collector
Fetches CNN's Fear & Greed Index

IMPORTANT: This is CNN's PROPRIETARY composite index.
The exact methodology is not publicly documented, but it combines:
    1. Stock Price Momentum (S&P 500 vs 125-day MA)
    2. Stock Price Strength (52-week highs vs lows)
    3. Stock Price Breadth (McClellan Volume Summation)
    4. Put/Call Ratio
    5. Market Volatility (VIX vs 50-day MA)
    6. Safe Haven Demand (stocks vs bonds)
    7. Junk Bond Demand (yield spread)

THRESHOLD CALIBRATION:
    CNN uses a 0-100 scale where 50 is "Neutral".
    Our thresholds align with CNN's published methodology:
        0-25:  Extreme Fear (strong contrarian buy signal)
        25-45: Fear
        45-55: Neutral
        55-75: Greed
        75-100: Extreme Greed (contrarian caution)

    Note: These thresholds are slightly wider than CNN's official
    display (which uses 0-25, 25-50, 50-75, 75-100) to reduce
    noise and false signals around boundaries.

DATA SOURCE FRAGILITY:
    This scrapes CNN's internal API endpoint. If CNN changes
    their API structure, this collector will fail. There is
    no official API or fallback source for this index.
"""

import logging
from datetime import datetime
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)


class FearGreedCollector:
    """
    Collects CNN Fear & Greed Index

    IMPORTANT: This is a PROPRIETARY index with no official API.
    The data is scraped from CNN's internal visualization endpoint.
    This may break if CNN changes their website structure.

    Contrarian usage:
        - Extreme Fear (<25): Historically bullish entry point
        - Extreme Greed (>75): Historically time for caution
        - Neutral (45-55): No strong signal

    Limitations:
        - Proprietary methodology (not fully transparent)
        - Single source (no fallback if CNN API changes)
        - Intraday updates, but we cache for dashboard use
    """
    
    def __init__(self):
        self.url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def get_fear_greed_score(self) -> Optional[Dict]:
        """
        Get current Fear & Greed Index score
        
        Returns:
            Dictionary with score, rating, and timestamp
        """
        try:
            response = requests.get(self.url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'fear_and_greed' in data:
                score = float(data['fear_and_greed']['score'])
                rating = self._get_rating(score)
                
                return {
                    'score': score,
                    'rating': rating,
                    'previous_close': data['fear_and_greed'].get('previous_close'),
                    'one_week_ago': data['fear_and_greed'].get('one_week_ago'),
                    'one_month_ago': data['fear_and_greed'].get('one_month_ago'),
                    'one_year_ago': data['fear_and_greed'].get('one_year_ago'),
                    'timestamp': datetime.now().isoformat()
                }
            
            return None
            
        except requests.exceptions.Timeout:
            logger.error("Fear & Greed API timeout (10s)")
            return None
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Fear & Greed API connection error: {e}")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"Fear & Greed API HTTP error: {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Fear & Greed unexpected error: {type(e).__name__}: {e}")
            return None

    def get_data_with_status(self) -> Dict:
        """
        Get Fear & Greed data with detailed status tracking.

        Returns:
            Dict with data, status, and error details
        """
        from utils.data_status import DataStatus

        try:
            response = requests.get(self.url, headers=self.headers, timeout=10)
            response.raise_for_status()

            data = response.json()

            if 'fear_and_greed' in data:
                score = float(data['fear_and_greed']['score'])
                rating = self._get_rating(score)

                return {
                    'data': {
                        'score': score,
                        'rating': rating,
                        'previous_close': data['fear_and_greed'].get('previous_close'),
                        'one_week_ago': data['fear_and_greed'].get('one_week_ago'),
                        'one_month_ago': data['fear_and_greed'].get('one_month_ago'),
                        'one_year_ago': data['fear_and_greed'].get('one_year_ago'),
                    },
                    'status': DataStatus.OK.value,
                    'timestamp': datetime.now().isoformat(),
                    'source': 'CNN Fear & Greed Index',
                    'error': None
                }

            return {
                'data': None,
                'status': DataStatus.UNAVAILABLE.value,
                'timestamp': datetime.now().isoformat(),
                'source': 'CNN Fear & Greed Index',
                'error': 'API returned data without fear_and_greed key'
            }

        except requests.exceptions.Timeout:
            return {
                'data': None,
                'status': DataStatus.ERROR.value,
                'timestamp': datetime.now().isoformat(),
                'source': 'CNN Fear & Greed Index',
                'error': 'Request timeout (10 seconds)'
            }
        except requests.exceptions.ConnectionError:
            return {
                'data': None,
                'status': DataStatus.ERROR.value,
                'timestamp': datetime.now().isoformat(),
                'source': 'CNN Fear & Greed Index',
                'error': 'Connection failed - check internet'
            }
        except requests.exceptions.HTTPError as e:
            return {
                'data': None,
                'status': DataStatus.ERROR.value,
                'timestamp': datetime.now().isoformat(),
                'source': 'CNN Fear & Greed Index',
                'error': f'HTTP {e.response.status_code}: {e.response.reason}'
            }
        except Exception as e:
            return {
                'data': None,
                'status': DataStatus.ERROR.value,
                'timestamp': datetime.now().isoformat(),
                'source': 'CNN Fear & Greed Index',
                'error': f'{type(e).__name__}: {str(e)}'
            }

    def _get_rating(self, score: float) -> str:
        """
        Convert score to rating label.

        Thresholds based on CNN's methodology with slight widening
        to reduce boundary noise:

            0-25:  Extreme Fear  (contrarian BUY zone)
            25-45: Fear          (cautious sentiment)
            45-55: Neutral       (no strong signal)
            55-75: Greed         (risk-on sentiment)
            75-100: Extreme Greed (contrarian CAUTION zone)

        Historical context:
            - Extreme Fear readings (<25) have preceded major
              market bottoms (COVID-19 March 2020: score=2)
            - Extreme Greed readings (>75) often precede
              corrections but timing is unreliable
        """
        if score >= 75:
            return "Extreme Greed"
        elif score >= 55:
            return "Greed"
        elif score >= 45:
            return "Neutral"
        elif score >= 25:
            return "Fear"
        else:
            return "Extreme Fear"
