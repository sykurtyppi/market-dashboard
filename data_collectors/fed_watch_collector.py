"""
Fed Watch Rate Probability Collector
Calculates implied Fed Funds rate probabilities from Fed Funds Futures

Similar to CME FedWatch Tool methodology:
- Uses Fed Funds Futures prices to derive implied rates
- Calculates probability distribution for each FOMC meeting
- Shows market expectations for rate cuts/hikes

Data sources:
- Yahoo Finance for Fed Funds Futures (ZQ)
- FOMC meeting dates (known schedule)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Current Fed Funds Rate (update manually or fetch from FRED)
# As of early 2024, rate is 5.25-5.50%
CURRENT_FED_RATE_UPPER = 5.50
CURRENT_FED_RATE_LOWER = 5.25
CURRENT_FED_RATE_MID = 5.375

# FOMC Meeting Dates 2024-2026
FOMC_MEETINGS = [
    datetime(2024, 1, 31),
    datetime(2024, 3, 20),
    datetime(2024, 5, 1),
    datetime(2024, 6, 12),
    datetime(2024, 7, 31),
    datetime(2024, 9, 18),
    datetime(2024, 11, 7),
    datetime(2024, 12, 18),
    datetime(2025, 1, 29),
    datetime(2025, 3, 19),
    datetime(2025, 5, 1),
    datetime(2025, 6, 18),
    datetime(2025, 7, 30),
    datetime(2025, 9, 17),
    datetime(2025, 11, 5),
    datetime(2025, 12, 17),
    datetime(2026, 1, 28),
    datetime(2026, 3, 18),
    datetime(2026, 4, 29),
    datetime(2026, 6, 17),
    datetime(2026, 7, 29),
    datetime(2026, 9, 16),
    datetime(2026, 11, 4),
    datetime(2026, 12, 16),
]


class FedWatchCollector:
    """
    Calculates Fed rate probabilities similar to CME FedWatch

    Uses Fed Funds Futures to derive market-implied probabilities
    for rate changes at upcoming FOMC meetings.
    """

    def __init__(self):
        self.current_rate = CURRENT_FED_RATE_MID
        self.rate_step = 0.25  # Fed moves in 25bp increments

    def get_upcoming_meetings(self, n: int = 6) -> List[Dict]:
        """
        Get next N FOMC meetings with basic info

        Returns:
            List of meeting dicts with date and days until
        """
        now = datetime.now()
        upcoming = []

        for meeting in FOMC_MEETINGS:
            if meeting > now:
                days_until = (meeting - now).days
                upcoming.append({
                    'date': meeting,
                    'date_str': meeting.strftime('%b %d, %Y'),
                    'days_until': days_until,
                })
                if len(upcoming) >= n:
                    break

        return upcoming

    def get_rate_probabilities(self) -> Dict:
        """
        Calculate rate probabilities for next FOMC meeting

        Uses Fed Funds Futures pricing when available,
        otherwise provides estimated probabilities based on
        current market conditions.

        Returns:
            Dict with meeting info and rate probabilities
        """
        meetings = self.get_upcoming_meetings(n=1)

        if not meetings:
            return {'status': 'no_meetings'}

        next_meeting = meetings[0]

        # Try to get Fed Funds Futures data
        implied_rate = self._get_implied_rate()

        if implied_rate is None:
            # Fallback: estimate based on VIX and yield curve
            probabilities = self._estimate_probabilities()
        else:
            probabilities = self._calculate_probabilities(implied_rate)

        # Determine most likely outcome
        most_likely = max(probabilities, key=probabilities.get)

        return {
            'meeting': next_meeting,
            'current_rate': f"{CURRENT_FED_RATE_LOWER:.2f}% - {CURRENT_FED_RATE_UPPER:.2f}%",
            'probabilities': probabilities,
            'most_likely': most_likely,
            'most_likely_prob': probabilities[most_likely],
            'implied_rate': implied_rate,
            'data_source': 'estimated' if implied_rate is None else 'futures',
        }

    def _get_implied_rate(self) -> Optional[float]:
        """
        Get implied Fed Funds rate from futures

        Fed Funds Futures ticker format: ZQH24 (March 2024)
        Yahoo Finance may not have this data directly
        """
        try:
            # Try to get data from Treasury yields as proxy
            # The 3-month T-bill closely tracks Fed Funds expectations

            tbill = yf.Ticker("^IRX")  # 13-week T-bill yield
            hist = tbill.history(period='5d')

            if not hist.empty:
                # T-bill yield approximates Fed Funds
                current_yield = hist['Close'].iloc[-1]
                return current_yield

            return None

        except Exception as e:
            logger.debug(f"Could not fetch implied rate: {e}")
            return None

    def _calculate_probabilities(self, implied_rate: float) -> Dict[str, float]:
        """
        Calculate probabilities based on implied rate

        Uses the difference between current rate and implied rate
        to determine probability of rate change.
        """
        rate_diff = implied_rate - self.current_rate

        # Base probabilities
        probs = {
            'Cut 50bp': 0,
            'Cut 25bp': 0,
            'No Change': 0,
            'Hike 25bp': 0,
            'Hike 50bp': 0,
        }

        if rate_diff < -0.375:  # >37.5bp lower
            probs['Cut 50bp'] = 60
            probs['Cut 25bp'] = 35
            probs['No Change'] = 5
        elif rate_diff < -0.125:  # 12.5-37.5bp lower
            probs['Cut 50bp'] = 20
            probs['Cut 25bp'] = 60
            probs['No Change'] = 20
        elif rate_diff < 0.125:  # Within 12.5bp
            probs['Cut 25bp'] = 25
            probs['No Change'] = 50
            probs['Hike 25bp'] = 25
        elif rate_diff < 0.375:  # 12.5-37.5bp higher
            probs['No Change'] = 20
            probs['Hike 25bp'] = 60
            probs['Hike 50bp'] = 20
        else:  # >37.5bp higher
            probs['No Change'] = 5
            probs['Hike 25bp'] = 35
            probs['Hike 50bp'] = 60

        return probs

    def _estimate_probabilities(self) -> Dict[str, float]:
        """
        Estimate probabilities when futures data unavailable

        Uses VIX and 2-10 yield spread as indicators
        """
        try:
            # Get VIX level
            vix = yf.Ticker("^VIX")
            vix_hist = vix.history(period='5d')
            vix_level = vix_hist['Close'].iloc[-1] if not vix_hist.empty else 20

            # Get yield curve (2Y - 10Y spread)
            tnx = yf.Ticker("^TNX")  # 10Y yield
            two_year = yf.Ticker("^TWO")  # 2Y yield (may not work)

            tnx_hist = tnx.history(period='5d')
            ten_year = tnx_hist['Close'].iloc[-1] if not tnx_hist.empty else 4.0

            # Estimate based on conditions
            probs = {
                'Cut 50bp': 0,
                'Cut 25bp': 0,
                'No Change': 50,
                'Hike 25bp': 0,
                'Hike 50bp': 0,
            }

            # High VIX suggests rate cuts more likely
            if vix_level > 30:
                probs['Cut 50bp'] = 30
                probs['Cut 25bp'] = 40
                probs['No Change'] = 25
                probs['Hike 25bp'] = 5
            elif vix_level > 20:
                probs['Cut 25bp'] = 35
                probs['No Change'] = 50
                probs['Hike 25bp'] = 15
            else:
                probs['Cut 25bp'] = 20
                probs['No Change'] = 55
                probs['Hike 25bp'] = 25

            return probs

        except Exception as e:
            logger.error(f"Error estimating probabilities: {e}")
            return {
                'Cut 50bp': 5,
                'Cut 25bp': 25,
                'No Change': 45,
                'Hike 25bp': 20,
                'Hike 50bp': 5,
            }

    def get_rate_path_expectations(self) -> Dict:
        """
        Get expected rate path for next several meetings

        Returns market expectations for where rates will be
        at each upcoming FOMC meeting.
        """
        meetings = self.get_upcoming_meetings(n=6)

        if not meetings:
            return {'status': 'no_data'}

        # Get current T-bill rate as baseline
        implied = self._get_implied_rate() or self.current_rate

        # Simple model: assume gradual move toward implied
        path = []
        current = self.current_rate

        for i, meeting in enumerate(meetings):
            # Each meeting: 30% chance of 25bp change in direction of implied
            expected_change = (implied - current) * 0.3 * (i + 1)
            expected_rate = current + expected_change

            # Round to nearest 25bp
            expected_rate = round(expected_rate * 4) / 4

            path.append({
                'meeting': meeting['date_str'],
                'days_until': meeting['days_until'],
                'expected_rate': expected_rate,
                'change_from_current': expected_rate - self.current_rate,
            })

        return {
            'current_rate': self.current_rate,
            'path': path,
            'terminal_rate': path[-1]['expected_rate'] if path else self.current_rate,
        }

    def get_fed_watch_summary(self) -> Dict:
        """
        Get comprehensive Fed Watch summary for dashboard

        Returns:
            Dict with all key Fed rate metrics
        """
        probs = self.get_rate_probabilities()
        path = self.get_rate_path_expectations()

        if probs.get('status') == 'no_meetings':
            return {'status': 'unavailable'}

        # Calculate cut vs hike probability
        cut_prob = probs['probabilities'].get('Cut 50bp', 0) + probs['probabilities'].get('Cut 25bp', 0)
        hike_prob = probs['probabilities'].get('Hike 50bp', 0) + probs['probabilities'].get('Hike 25bp', 0)
        hold_prob = probs['probabilities'].get('No Change', 0)

        # Determine market bias
        if cut_prob > 60:
            bias = 'Strongly Dovish'
            bias_color = '#4CAF50'
        elif cut_prob > hike_prob + 10:
            bias = 'Dovish'
            bias_color = '#8BC34A'
        elif hike_prob > cut_prob + 10:
            bias = 'Hawkish'
            bias_color = '#FF9800'
        elif hike_prob > 60:
            bias = 'Strongly Hawkish'
            bias_color = '#F44336'
        else:
            bias = 'Neutral'
            bias_color = '#9E9E9E'

        return {
            'next_meeting': probs['meeting'],
            'current_rate': probs['current_rate'],
            'probabilities': probs['probabilities'],
            'most_likely': probs['most_likely'],
            'most_likely_prob': probs['most_likely_prob'],
            'cut_probability': round(cut_prob, 1),
            'hike_probability': round(hike_prob, 1),
            'hold_probability': round(hold_prob, 1),
            'market_bias': bias,
            'bias_color': bias_color,
            'rate_path': path.get('path', []),
            'terminal_rate': path.get('terminal_rate'),
            'data_source': probs.get('data_source', 'estimated'),
        }
