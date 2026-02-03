"""
Fed Watch Rate Probability Collector - Institutional Grade

Implements CME FedWatch Tool methodology for calculating Fed rate probabilities
from 30-Day Fed Funds Futures (ZQ contracts).

Methodology (per CME Group):
- Fed Funds Futures price = 100 - Expected Average EFFR for the month
- Implied Rate = 100 - Futures Price
- Probability = (End Rate - Start Rate) / 0.25%

Data Sources:
- FRED API: Real-time Federal Funds Target Rate (upper/lower bounds + EFFR)
- Yahoo Finance: 30-Day Fed Funds Futures (ZQ contracts)
- Federal Reserve: Official FOMC meeting schedule

References:
- https://www.cmegroup.com/articles/2023/understanding-the-cme-group-fedwatch-tool-methodology.html
- https://fred.stlouisfed.org/series/DFEDTARU (Target Rate Upper)
- https://fred.stlouisfed.org/series/DFEDTARL (Target Rate Lower)
"""

import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# ============================================================
# FOMC MEETING SCHEDULE 2025-2027
# Source: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
# ============================================================

@dataclass
class FOMCMeeting:
    """FOMC Meeting with metadata"""
    date: datetime  # Decision announcement date (2nd day of meeting)
    has_sep: bool   # Has Summary of Economic Projections

    @property
    def date_str(self) -> str:
        return self.date.strftime('%b %d, %Y')

    @property
    def month_code(self) -> str:
        """Get futures month code for this meeting's contract month"""
        # Futures month codes: F=Jan, G=Feb, H=Mar, J=Apr, K=May, M=Jun
        # N=Jul, Q=Aug, U=Sep, V=Oct, X=Nov, Z=Dec
        codes = {1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
                 7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'}
        return codes[self.date.month]


# Official FOMC Meeting Schedule
FOMC_MEETINGS = [
    # 2025
    FOMCMeeting(datetime(2025, 1, 29), has_sep=False),
    FOMCMeeting(datetime(2025, 3, 19), has_sep=True),
    FOMCMeeting(datetime(2025, 5, 7), has_sep=False),
    FOMCMeeting(datetime(2025, 6, 18), has_sep=True),
    FOMCMeeting(datetime(2025, 7, 30), has_sep=False),
    FOMCMeeting(datetime(2025, 9, 17), has_sep=True),
    FOMCMeeting(datetime(2025, 10, 29), has_sep=False),
    FOMCMeeting(datetime(2025, 12, 10), has_sep=True),
    # 2026
    FOMCMeeting(datetime(2026, 1, 28), has_sep=False),
    FOMCMeeting(datetime(2026, 3, 18), has_sep=True),
    FOMCMeeting(datetime(2026, 4, 29), has_sep=False),
    FOMCMeeting(datetime(2026, 6, 17), has_sep=True),
    FOMCMeeting(datetime(2026, 7, 29), has_sep=False),
    FOMCMeeting(datetime(2026, 9, 16), has_sep=True),
    FOMCMeeting(datetime(2026, 10, 28), has_sep=False),
    FOMCMeeting(datetime(2026, 12, 9), has_sep=True),
    # 2027
    FOMCMeeting(datetime(2027, 1, 27), has_sep=False),
    FOMCMeeting(datetime(2027, 3, 17), has_sep=True),
    FOMCMeeting(datetime(2027, 4, 28), has_sep=False),
    FOMCMeeting(datetime(2027, 6, 16), has_sep=True),
    FOMCMeeting(datetime(2027, 7, 28), has_sep=False),
    FOMCMeeting(datetime(2027, 9, 22), has_sep=True),
    FOMCMeeting(datetime(2027, 11, 3), has_sep=False),
    FOMCMeeting(datetime(2027, 12, 15), has_sep=True),
]


# ============================================================
# FRED DATA FETCHER
# ============================================================

class FREDDataFetcher:
    """
    Fetches Federal Reserve data from FRED (Federal Reserve Economic Data)
    Uses the public CSV endpoint which doesn't require an API key
    """

    BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    # FRED Series IDs
    SERIES = {
        'target_upper': 'DFEDTARU',  # Fed Funds Target Rate - Upper Bound
        'target_lower': 'DFEDTARL',  # Fed Funds Target Rate - Lower Bound
        'effr': 'EFFR',              # Effective Federal Funds Rate (daily)
        'effr_monthly': 'FEDFUNDS',  # Effective Federal Funds Rate (monthly avg)
    }

    def __init__(self, cache_ttl: int = 300):
        """
        Initialize FRED fetcher

        Args:
            cache_ttl: Cache time-to-live in seconds (default 5 minutes)
        """
        self._cache = {}
        self._cache_time = {}
        self._cache_ttl = cache_ttl

    def _fetch_series(self, series_id: str, lookback_days: int = 30) -> Optional[pd.DataFrame]:
        """
        Fetch a FRED series

        Args:
            series_id: FRED series identifier
            lookback_days: Number of days to fetch

        Returns:
            DataFrame with DATE and value columns, or None on error
        """
        # Check cache
        cache_key = f"{series_id}_{lookback_days}"
        if cache_key in self._cache:
            if datetime.now().timestamp() - self._cache_time[cache_key] < self._cache_ttl:
                return self._cache[cache_key]

        try:
            start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            url = f"{self.BASE_URL}?id={series_id}&cosd={start_date}"

            response = requests.get(url, timeout=10)
            response.raise_for_status()

            # Parse CSV
            lines = response.text.strip().split('\n')
            if len(lines) < 2:
                return None

            data = []
            for line in lines[1:]:  # Skip header
                parts = line.split(',')
                if len(parts) >= 2 and parts[1] != '.':
                    try:
                        data.append({
                            'date': datetime.strptime(parts[0], '%Y-%m-%d'),
                            'value': float(parts[1])
                        })
                    except (ValueError, IndexError):
                        continue

            if not data:
                return None

            df = pd.DataFrame(data)

            # Cache result
            self._cache[cache_key] = df
            self._cache_time[cache_key] = datetime.now().timestamp()

            return df

        except Exception as e:
            logger.error(f"Error fetching FRED series {series_id}: {e}")
            return None

    def get_current_target_rate(self) -> Optional[Dict]:
        """
        Get current Fed Funds Target Rate range

        Returns:
            Dict with 'upper', 'lower', 'mid', 'range_str', and 'as_of' date
        """
        upper_df = self._fetch_series(self.SERIES['target_upper'])
        lower_df = self._fetch_series(self.SERIES['target_lower'])

        if upper_df is None or lower_df is None:
            return None

        upper = upper_df['value'].iloc[-1]
        lower = lower_df['value'].iloc[-1]
        as_of = upper_df['date'].iloc[-1]

        return {
            'upper': upper,
            'lower': lower,
            'mid': (upper + lower) / 2,
            'range_str': f"{lower:.2f}% - {upper:.2f}%",
            'as_of': as_of.strftime('%Y-%m-%d'),
        }

    def get_effective_rate(self) -> Optional[Dict]:
        """
        Get current Effective Federal Funds Rate (EFFR)

        Returns:
            Dict with 'rate' and 'as_of' date
        """
        df = self._fetch_series(self.SERIES['effr'])

        if df is None:
            return None

        return {
            'rate': df['value'].iloc[-1],
            'as_of': df['date'].iloc[-1].strftime('%Y-%m-%d'),
        }


# ============================================================
# FED FUNDS FUTURES FETCHER
# ============================================================

class FedFundsFuturesFetcher:
    """
    Fetches 30-Day Fed Funds Futures (ZQ) data from Yahoo Finance

    Ticker format: ZQ{MONTH_CODE}{YY}.CBT
    Example: ZQH26.CBT = March 2026 contract
    """

    MONTH_CODES = {
        1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
        7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'
    }

    def __init__(self, cache_ttl: int = 60):
        """
        Initialize futures fetcher

        Args:
            cache_ttl: Cache time-to-live in seconds (default 1 minute)
        """
        self._cache = {}
        self._cache_time = {}
        self._cache_ttl = cache_ttl

    def _get_ticker(self, year: int, month: int) -> str:
        """Get Yahoo Finance ticker for a specific contract month"""
        month_code = self.MONTH_CODES[month]
        return f"ZQ{month_code}{year % 100:02d}.CBT"

    def get_contract_price(self, year: int, month: int) -> Optional[float]:
        """
        Get current price for a specific Fed Funds Futures contract

        Args:
            year: Contract year (e.g., 2026)
            month: Contract month (1-12)

        Returns:
            Contract price (e.g., 96.54) or None on error
        """
        ticker = self._get_ticker(year, month)

        # Check cache
        if ticker in self._cache:
            if datetime.now().timestamp() - self._cache_time[ticker] < self._cache_ttl:
                return self._cache[ticker]

        try:
            contract = yf.Ticker(ticker)
            hist = contract.history(period='5d')

            if hist.empty:
                logger.debug(f"No data for {ticker}")
                return None

            price = hist['Close'].iloc[-1]

            # Cache result
            self._cache[ticker] = price
            self._cache_time[ticker] = datetime.now().timestamp()

            return price

        except Exception as e:
            logger.debug(f"Error fetching {ticker}: {e}")
            return None

    def get_implied_rate(self, year: int, month: int) -> Optional[float]:
        """
        Get implied Fed Funds rate from futures price

        Formula: Implied Rate = 100 - Futures Price

        Args:
            year: Contract year
            month: Contract month

        Returns:
            Implied rate as percentage (e.g., 3.46) or None
        """
        price = self.get_contract_price(year, month)
        if price is None:
            return None
        return 100 - price

    def get_multiple_contracts(self, months_ahead: int = 12) -> Dict[str, Dict]:
        """
        Get data for multiple upcoming contract months

        Args:
            months_ahead: Number of months to fetch

        Returns:
            Dict keyed by 'YYYY-MM' with price and implied rate
        """
        results = {}
        now = datetime.now()

        for i in range(months_ahead):
            target = now + timedelta(days=i * 30)
            year = target.year
            month = target.month

            price = self.get_contract_price(year, month)
            if price is not None:
                key = f"{year}-{month:02d}"
                results[key] = {
                    'year': year,
                    'month': month,
                    'ticker': self._get_ticker(year, month),
                    'price': price,
                    'implied_rate': 100 - price,
                }

        return results


# ============================================================
# CME FEDWATCH PROBABILITY CALCULATOR
# ============================================================

class FedWatchCalculator:
    """
    Calculates Fed rate probabilities using CME FedWatch methodology

    Key Methodology:
    1. Get EFFR for anchor month (month before FOMC meeting if no meeting in that month)
    2. Get implied rate from futures for meeting month
    3. Calculate probability based on rate difference and 25bp step size

    Formula:
    - If implied rate change < 0 (cut expected):
      P(cut) = |implied_change| / 0.25
      P(no change) = 1 - P(cut)

    - If implied rate change > 0 (hike expected):
      P(hike) = implied_change / 0.25
      P(no change) = 1 - P(hike)
    """

    RATE_STEP = 0.25  # Fed moves in 25bp increments

    def __init__(self, fred_fetcher: FREDDataFetcher, futures_fetcher: FedFundsFuturesFetcher):
        self.fred = fred_fetcher
        self.futures = futures_fetcher

    def _get_rate_levels(self, center_rate: float, num_levels: int = 5) -> List[Dict]:
        """
        Generate possible rate levels around center rate

        Args:
            center_rate: Current target rate midpoint
            num_levels: Number of levels in each direction

        Returns:
            List of rate level dicts with 'lower', 'upper', 'mid', 'label'
        """
        levels = []

        for i in range(-num_levels, num_levels + 1):
            mid = center_rate + (i * self.RATE_STEP)
            if mid >= 0:  # No negative rates
                lower = mid - 0.125
                upper = mid + 0.125

                # Generate label
                if i < 0:
                    label = f"{lower:.2f}-{upper:.2f}"
                elif i > 0:
                    label = f"{lower:.2f}-{upper:.2f}"
                else:
                    label = f"{lower:.2f}-{upper:.2f} (current)"

                levels.append({
                    'lower': lower,
                    'upper': upper,
                    'mid': mid,
                    'label': label,
                    'change_bps': i * 25,
                })

        return levels

    def calculate_meeting_probabilities(
        self,
        meeting: FOMCMeeting,
        current_rate_mid: float,
        prior_month_implied: Optional[float] = None
    ) -> Dict:
        """
        Calculate rate probabilities for a specific FOMC meeting

        Uses proper CME methodology:
        1. Anchor rate = prior month's implied rate (or current rate if first meeting)
        2. Meeting month implied rate from futures
        3. Probability = (meeting_implied - anchor) / 0.25

        Args:
            meeting: FOMC meeting to calculate for
            current_rate_mid: Current target rate midpoint
            prior_month_implied: Implied rate from prior month (anchor)

        Returns:
            Dict with probabilities for each possible outcome
        """
        # Get meeting month implied rate
        meeting_implied = self.futures.get_implied_rate(meeting.date.year, meeting.date.month)

        if meeting_implied is None:
            return self._fallback_probabilities(current_rate_mid)

        # Use prior month as anchor, or current rate if not available
        anchor_rate = prior_month_implied if prior_month_implied else current_rate_mid

        # Calculate implied change
        implied_change = meeting_implied - anchor_rate

        # CME methodology: calculate probability based on distance from anchor
        # The probability is proportional to how far the implied rate has moved

        # Determine the two most likely outcomes
        prob_dict = {}

        # Calculate number of 25bp moves implied
        moves = implied_change / self.RATE_STEP

        # Floor and ceil to get the two bracketing rate levels
        lower_moves = int(np.floor(moves))
        upper_moves = int(np.ceil(moves))

        # Probability interpolation
        if lower_moves == upper_moves:
            # Exactly on a rate level
            prob_lower = 1.0
            prob_upper = 0.0
        else:
            # Between two levels - linear interpolation
            prob_upper = moves - lower_moves
            prob_lower = 1.0 - prob_upper

        # Build probability dictionary
        # We show probabilities for: -50bp, -25bp, 0bp, +25bp, +50bp
        outcomes = [
            ('Cut 50bp', -2),
            ('Cut 25bp', -1),
            ('No Change', 0),
            ('Hike 25bp', 1),
            ('Hike 50bp', 2),
        ]

        for label, move_count in outcomes:
            if move_count == lower_moves:
                prob_dict[label] = round(prob_lower * 100, 1)
            elif move_count == upper_moves:
                prob_dict[label] = round(prob_upper * 100, 1)
            else:
                prob_dict[label] = 0.0

        # Ensure probabilities sum to 100
        total = sum(prob_dict.values())
        if total > 0 and abs(total - 100) > 0.1:
            # Normalize
            factor = 100 / total
            for key in prob_dict:
                prob_dict[key] = round(prob_dict[key] * factor, 1)

        return {
            'probabilities': prob_dict,
            'implied_rate': round(meeting_implied, 4),
            'anchor_rate': round(anchor_rate, 4),
            'implied_change_bps': round(implied_change * 100, 1),
            'data_source': 'fed_funds_futures',
        }

    def _fallback_probabilities(self, current_rate: float) -> Dict:
        """Return neutral probabilities when futures data unavailable"""
        return {
            'probabilities': {
                'Cut 50bp': 5.0,
                'Cut 25bp': 15.0,
                'No Change': 60.0,
                'Hike 25bp': 15.0,
                'Hike 50bp': 5.0,
            },
            'implied_rate': None,
            'anchor_rate': current_rate,
            'implied_change_bps': 0,
            'data_source': 'fallback',
        }


# ============================================================
# MAIN FED WATCH COLLECTOR
# ============================================================

class FedWatchCollector:
    """
    Institutional-grade Fed Watch collector

    Provides:
    - Real-time Fed Funds Target Rate from FRED
    - Market-implied probabilities from Fed Funds Futures
    - CME FedWatch methodology for probability calculations
    - Rate path expectations for multiple meetings

    Usage:
        collector = FedWatchCollector()
        summary = collector.get_fed_watch_summary()
    """

    def __init__(self):
        self.fred = FREDDataFetcher(cache_ttl=300)  # 5 minute cache
        self.futures = FedFundsFuturesFetcher(cache_ttl=60)  # 1 minute cache
        self.calculator = FedWatchCalculator(self.fred, self.futures)

        # Cache for computed results
        self._summary_cache = None
        self._summary_cache_time = None
        self._summary_cache_ttl = 60  # 1 minute

    def get_upcoming_meetings(self, n: int = 8) -> List[Dict]:
        """
        Get next N FOMC meetings with metadata

        Returns:
            List of meeting dicts with date, days until, and SEP indicator
        """
        now = datetime.now()
        upcoming = []

        for meeting in FOMC_MEETINGS:
            if meeting.date > now:
                days_until = (meeting.date - now).days
                upcoming.append({
                    'date': meeting.date,
                    'date_str': meeting.date_str,
                    'days_until': days_until,
                    'has_sep': meeting.has_sep,
                    'month_code': meeting.month_code,
                    'year': meeting.date.year,
                    'month': meeting.date.month,
                })
                if len(upcoming) >= n:
                    break

        return upcoming

    def get_current_rate(self) -> Dict:
        """
        Get current Fed Funds Rate from FRED

        Returns:
            Dict with target range, effective rate, and timestamps
        """
        target = self.fred.get_current_target_rate()
        effr = self.fred.get_effective_rate()

        if target is None:
            # Fallback to hardcoded if FRED unavailable
            logger.warning("FRED unavailable, using fallback rate")
            return {
                'upper': 3.75,
                'lower': 3.50,
                'mid': 3.625,
                'range_str': '3.50% - 3.75%',
                'effr': 3.64,
                'as_of': datetime.now().strftime('%Y-%m-%d'),
                'source': 'fallback',
            }

        result = {
            'upper': target['upper'],
            'lower': target['lower'],
            'mid': target['mid'],
            'range_str': target['range_str'],
            'effr': effr['rate'] if effr else target['mid'],
            'as_of': target['as_of'],
            'source': 'FRED',
        }

        return result

    def get_rate_probabilities(self) -> Dict:
        """
        Calculate rate probabilities for next FOMC meeting

        Returns:
            Dict with meeting info and rate probabilities
        """
        meetings = self.get_upcoming_meetings(n=1)

        if not meetings:
            return {'status': 'no_meetings'}

        next_meeting = meetings[0]
        current = self.get_current_rate()

        # Get prior month implied rate for anchor
        meeting_date = next_meeting['date']
        prior_month = meeting_date.month - 1 if meeting_date.month > 1 else 12
        prior_year = meeting_date.year if meeting_date.month > 1 else meeting_date.year - 1
        prior_implied = self.futures.get_implied_rate(prior_year, prior_month)

        # Create meeting object for calculator
        meeting_obj = FOMCMeeting(meeting_date, next_meeting['has_sep'])

        # Calculate probabilities
        result = self.calculator.calculate_meeting_probabilities(
            meeting_obj,
            current['mid'],
            prior_implied
        )

        # Determine most likely outcome
        probs = result['probabilities']
        most_likely = max(probs, key=probs.get)

        return {
            'meeting': next_meeting,
            'current_rate': current['range_str'],
            'current_rate_mid': current['mid'],
            'probabilities': probs,
            'most_likely': most_likely,
            'most_likely_prob': probs[most_likely],
            'implied_rate': result['implied_rate'],
            'implied_change_bps': result['implied_change_bps'],
            'data_source': result['data_source'],
        }

    def get_rate_path_expectations(self) -> Dict:
        """
        Get expected rate path for next several meetings

        Uses Fed Funds Futures for each meeting month to derive
        the market-implied expected rate at each FOMC meeting.

        Returns:
            Dict with current rate, expected path, and terminal rate
        """
        meetings = self.get_upcoming_meetings(n=8)
        current = self.get_current_rate()

        if not meetings:
            return {'status': 'no_data'}

        path = []
        prior_implied = current['mid']

        for meeting in meetings:
            # Get implied rate for meeting month
            implied = self.futures.get_implied_rate(meeting['year'], meeting['month'])

            if implied is not None:
                # Round to nearest 12.5bp (half step) for cleaner display
                expected_rate = round(implied * 8) / 8

                # Calculate probability-weighted expected rate
                # (using the implied rate directly as the expectation)
                change_from_current = expected_rate - current['mid']
                change_from_prior = expected_rate - prior_implied

                path.append({
                    'meeting': meeting['date_str'],
                    'date': meeting['date'],
                    'days_until': meeting['days_until'],
                    'has_sep': meeting['has_sep'],
                    'implied_rate': round(implied, 3),
                    'expected_rate': expected_rate,
                    'change_from_current': round(change_from_current, 3),
                    'change_from_prior': round(change_from_prior, 3),
                    'change_bps': round(change_from_current * 100),
                })

                prior_implied = implied
            else:
                # Use prior implied rate if futures unavailable
                path.append({
                    'meeting': meeting['date_str'],
                    'date': meeting['date'],
                    'days_until': meeting['days_until'],
                    'has_sep': meeting['has_sep'],
                    'implied_rate': None,
                    'expected_rate': prior_implied,
                    'change_from_current': round(prior_implied - current['mid'], 3),
                    'change_from_prior': 0,
                    'change_bps': round((prior_implied - current['mid']) * 100),
                })

        # Calculate terminal rate (last meeting in path)
        terminal = path[-1]['expected_rate'] if path else current['mid']

        # Calculate total expected cuts/hikes
        total_change_bps = round((terminal - current['mid']) * 100)

        return {
            'current_rate': current['mid'],
            'current_rate_str': current['range_str'],
            'path': path,
            'terminal_rate': terminal,
            'total_change_bps': total_change_bps,
            'expected_cuts': abs(total_change_bps) // 25 if total_change_bps < 0 else 0,
            'expected_hikes': total_change_bps // 25 if total_change_bps > 0 else 0,
        }

    def get_futures_term_structure(self) -> Dict:
        """
        Get full term structure of Fed Funds Futures implied rates

        Returns:
            Dict with contract data for visualization
        """
        contracts = self.futures.get_multiple_contracts(months_ahead=18)

        if not contracts:
            return {'status': 'no_data'}

        # Sort by date
        sorted_contracts = sorted(contracts.items(), key=lambda x: x[0])

        term_structure = []
        for key, data in sorted_contracts:
            term_structure.append({
                'contract': key,
                'ticker': data['ticker'],
                'price': round(data['price'], 4),
                'implied_rate': round(data['implied_rate'], 4),
            })

        return {
            'contracts': term_structure,
            'front_month': term_structure[0] if term_structure else None,
            'back_month': term_structure[-1] if term_structure else None,
        }

    def get_fed_watch_summary(self) -> Dict:
        """
        Get comprehensive Fed Watch summary for dashboard

        Returns:
            Dict with all key Fed rate metrics
        """
        # Check cache
        if self._summary_cache is not None and self._summary_cache_time is not None:
            if datetime.now().timestamp() - self._summary_cache_time < self._summary_cache_ttl:
                return self._summary_cache

        try:
            probs = self.get_rate_probabilities()
            path = self.get_rate_path_expectations()
            current = self.get_current_rate()

            if probs.get('status') == 'no_meetings':
                return {'status': 'unavailable'}

            # Calculate cut vs hike probability
            prob_dict = probs['probabilities']
            cut_prob = prob_dict.get('Cut 50bp', 0) + prob_dict.get('Cut 25bp', 0)
            hike_prob = prob_dict.get('Hike 50bp', 0) + prob_dict.get('Hike 25bp', 0)
            hold_prob = prob_dict.get('No Change', 0)

            # Determine market bias
            if cut_prob > 70:
                bias = 'Strongly Dovish'
                bias_color = '#4CAF50'
            elif cut_prob > 55:
                bias = 'Dovish'
                bias_color = '#8BC34A'
            elif hike_prob > 70:
                bias = 'Strongly Hawkish'
                bias_color = '#F44336'
            elif hike_prob > 55:
                bias = 'Hawkish'
                bias_color = '#FF9800'
            elif hold_prob > 60:
                bias = 'Hold Expected'
                bias_color = '#9E9E9E'
            else:
                bias = 'Uncertain'
                bias_color = '#FFC107'

            summary = {
                # Current rate info
                'current_rate': current['range_str'],
                'current_rate_mid': current['mid'],
                'current_rate_upper': current['upper'],
                'current_rate_lower': current['lower'],
                'effr': current.get('effr'),
                'rate_source': current.get('source', 'unknown'),
                'rate_as_of': current.get('as_of'),

                # Next meeting info
                'next_meeting': probs['meeting'],

                # Probabilities
                'probabilities': prob_dict,
                'most_likely': probs['most_likely'],
                'most_likely_prob': probs['most_likely_prob'],
                'cut_probability': round(cut_prob, 1),
                'hike_probability': round(hike_prob, 1),
                'hold_probability': round(hold_prob, 1),

                # Market-implied data
                'implied_rate': probs.get('implied_rate'),
                'implied_change_bps': probs.get('implied_change_bps'),

                # Market bias
                'market_bias': bias,
                'bias_color': bias_color,

                # Rate path
                'rate_path': path.get('path', []),
                'terminal_rate': path.get('terminal_rate'),
                'total_change_bps': path.get('total_change_bps', 0),
                'expected_cuts': path.get('expected_cuts', 0),
                'expected_hikes': path.get('expected_hikes', 0),

                # Data quality
                'data_source': probs.get('data_source', 'unknown'),
                'timestamp': datetime.now().isoformat(),
            }

            # Cache result
            self._summary_cache = summary
            self._summary_cache_time = datetime.now().timestamp()

            return summary

        except Exception as e:
            logger.error(f"Error generating Fed Watch summary: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
            }

    def get_historical_comparison(self) -> Dict:
        """
        Get data for comparing current expectations vs historical

        Useful for showing how expectations have shifted
        """
        # This could be enhanced to store/retrieve historical expectations
        # For now, return current data with metadata for future comparison

        summary = self.get_fed_watch_summary()

        return {
            'current': summary,
            'note': 'Historical tracking requires persistent storage',
        }


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def get_fed_watch_data() -> Dict:
    """Quick access function for Fed Watch data"""
    collector = FedWatchCollector()
    return collector.get_fed_watch_summary()


def get_next_fomc_meeting() -> Optional[Dict]:
    """Get info about the next FOMC meeting"""
    collector = FedWatchCollector()
    meetings = collector.get_upcoming_meetings(n=1)
    return meetings[0] if meetings else None


def get_current_fed_rate() -> Dict:
    """Get current Fed Funds Rate from FRED"""
    collector = FedWatchCollector()
    return collector.get_current_rate()
