"""
Fear & Greed Index Collector
Fetches CNN's Fear & Greed Index
"""

import logging
from datetime import datetime
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)


class FearGreedCollector:
    """Collects CNN Fear & Greed Index"""
    
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
        """Convert score to rating"""
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
