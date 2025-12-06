"""
Fear & Greed Index Collector
Fetches CNN's Fear & Greed Index
"""

import requests
from datetime import datetime
from typing import Dict, Optional

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
            
        except Exception as e:
            print(f"Error fetching Fear & Greed Index: {e}")
            return None
    
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


# Test function
if __name__ == "__main__":
    collector = FearGreedCollector()
    
    print("Testing Fear & Greed Collector...")
    data = collector.get_fear_greed_score()
    
    if data:
        print(f"\nScore: {data['score']}")
        print(f"Rating: {data['rating']}")
        print(f"Previous Close: {data['previous_close']}")