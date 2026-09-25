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

import calendar
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


def _merge_warnings(*groups) -> List[str]:
    """Concatenate warning lists, dropping blanks and duplicates, order kept."""
    merged: List[str] = []
    for group in groups:
        for w in group or []:
            if w and w not in merged:
                merged.append(w)
    return merged


def _shift_month(year: int, month: int, delta: int) -> Tuple[int, int]:
    """(year, month) moved by `delta` calendar months."""
    idx = year * 12 + (month - 1) + delta
    return idx // 12, idx % 12 + 1


def _has_meeting(year: int, month: int) -> bool:
    return any(m.date.year == year and m.date.month == month for m in FOMC_MEETINGS)


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
    1. Start rate = the rate prevailing before the meeting (chosen by the collector:
       prior-month contract if it is still trading and has no meeting, else EFFR)
    2. End rate = the rate after the meeting:
       - next-month contract, if the following month has no FOMC meeting
         (its whole-month average IS the post-meeting rate), else
       - backed out of the meeting-month contract, whose price is the AVERAGE rate
         over the whole month — pre-meeting days at the start rate, post-meeting
         days at the end rate:
           end = (month_avg * N - start * pre_days) / post_days
    3. Probability = (end - start) / 0.25, split between the two bracketing
       25bp outcomes.

    Comparing the raw meeting-month average against the start rate (the old
    behavior) dilutes the implied move by the pre-meeting days — for a Sep 16
    meeting it roughly halved the implied hike probability.
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

    # Fewer post-meeting days than this and a back-out from the meeting-month
    # contract amplifies quote noise (1bp of price -> N/post_days bp of rate).
    MIN_RELIABLE_POST_DAYS = 7

    @staticmethod
    def post_meeting_rate(month_avg: float, start_rate: float,
                          meeting_day: int, days_in_month: int) -> Optional[float]:
        """Rate after the meeting, backed out of the meeting-month average.

        The new rate takes effect the day after the decision, so days 1..meeting_day
        run at the start rate. Returns None if no post-meeting days remain.
        """
        post_days = days_in_month - meeting_day
        if post_days <= 0:
            return None
        return (month_avg * days_in_month - start_rate * meeting_day) / post_days

    def meeting_end_rate(self, meeting_date: datetime,
                         start_rate: float) -> Tuple[Optional[float], str, List[str]]:
        """Market-implied post-meeting rate for the meeting on `meeting_date`.

        Returns (end_rate, source, caveats). end_rate is None when no usable
        contract exists.
        """
        year, month = meeting_date.year, meeting_date.month
        next_y, next_m = _shift_month(year, month, 1)
        if not _has_meeting(next_y, next_m):
            next_avg = self.futures.get_implied_rate(next_y, next_m)
            if next_avg is not None:
                return next_avg, 'next_month_contract', []

        month_avg = self.futures.get_implied_rate(year, month)
        if month_avg is None:
            return None, 'unavailable', []

        days = calendar.monthrange(year, month)[1]
        end = self.post_meeting_rate(month_avg, start_rate, meeting_date.day, days)
        if end is None:
            return None, 'unavailable', []

        caveats: List[str] = []
        post_days = days - meeting_date.day
        if post_days < self.MIN_RELIABLE_POST_DAYS:
            caveats.append(
                f"Late-month meeting: the post-meeting rate is backed out of only {post_days} "
                "days of the meeting-month contract, so small quote moves shift the "
                "probabilities sharply."
            )
        return end, 'meeting_month_contract', caveats

    def probabilities_from_rates(self, start_rate: float, end_rate: float) -> Dict:
        """Split the implied start->end move between the two bracketing 25bp outcomes."""
        implied_change = end_rate - start_rate
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
            'implied_rate': round(end_rate, 4),
            'anchor_rate': round(start_rate, 4),
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

    def _now(self) -> datetime:
        """Current time; a method so tests can pin the calendar."""
        return datetime.now()

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
                'effr_source': 'fallback',
                'as_of': datetime.now().strftime('%Y-%m-%d'),
                'source': 'fallback',
            }

        result = {
            'upper': target['upper'],
            'lower': target['lower'],
            'mid': target['mid'],
            'range_str': target['range_str'],
            'effr': effr['rate'] if effr else target['mid'],
            # Consumers anchoring on EFFR need to know when it is really the midpoint.
            'effr_source': 'FRED' if effr else 'target_midpoint',
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
        meeting_date = next_meeting['date']

        # `warnings` is everything the page should disclose; `degraded` is only
        # set when an input fell back (the frontend labels that a fallback).
        warnings: List[str] = []
        degraded = False

        start_rate, start_source = self._pre_meeting_rate(meeting_date, current, warnings)
        if start_source in ('target_midpoint', 'missing_prior_contract_effr'):
            degraded = True

        end_rate, end_source, caveats = self.calculator.meeting_end_rate(meeting_date, start_rate)
        warnings.extend(caveats)

        if end_rate is None:
            result = self.calculator._fallback_probabilities(current['mid'])
            warnings.append(
                f"Meeting-month contract {self.futures._get_ticker(meeting_date.year, meeting_date.month)} "
                "unavailable; probabilities are a neutral placeholder, not market-implied."
            )
            degraded = True
        else:
            result = self.calculator.probabilities_from_rates(start_rate, end_rate)

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
            'anchor_rate': result['anchor_rate'],
            'implied_change_bps': result['implied_change_bps'],
            'start_rate_source': start_source,
            'end_rate_source': end_source,
            'data_source': result['data_source'],
            'warnings': warnings,
            'degraded': degraded,
        }

    def _pre_meeting_rate(self, meeting_date: datetime, current: Dict,
                          warnings: List[str]) -> Tuple[float, str]:
        """Rate prevailing until the next meeting, and where it came from.

        The prior-month contract is used only while it still trades and that
        month has no meeting. Otherwise the prior month is already realized, so
        the observed EFFR is the correct pre-meeting rate. This is the normal
        path inside the meeting month, not a fallback.
        """
        now = self._now()
        prior_y, prior_m = _shift_month(meeting_date.year, meeting_date.month, -1)
        prior_trading = (now.year, now.month) <= (prior_y, prior_m)
        missing_prior = False

        if prior_trading and not _has_meeting(prior_y, prior_m):
            prior_avg = self.futures.get_implied_rate(prior_y, prior_m)
            if prior_avg is not None:
                return prior_avg, 'prior_month_contract'
            missing_prior = True
            warnings.append(
                f"Anchor contract {self.futures._get_ticker(prior_y, prior_m)} should be trading "
                "but returned no data; pre-meeting rate uses the effective rate instead."
            )

        if current.get('effr') is not None and current.get('effr_source') == 'FRED':
            return current['effr'], 'missing_prior_contract_effr' if missing_prior else 'effr'

        warnings.append(
            f"Effective rate (EFFR) unavailable; pre-meeting rate uses the target midpoint "
            f"({current['mid']:.3f}%), which usually sits a few bp off EFFR — enough to move "
            "these probabilities noticeably."
        )
        return current['mid'], 'target_midpoint'

    def get_rate_path_expectations(self) -> Dict:
        """
        Get expected rate path for the next several meetings

        Chained CME-style: each meeting's post-meeting rate becomes the
        pre-meeting rate for the next one, and every step uses the same weighted
        end-rate logic as the next-meeting probabilities. Reading the raw
        meeting-month average as "the rate after that meeting" (the old
        behavior) blends in pre-meeting days at the previous rate, so every
        step of the path — and the terminal rate — was pulled toward the
        current rate.

        Returns:
            Dict with current rate, expected path, terminal rate, and any
            disclosures about gaps in the curve.
        """
        meetings = self.get_upcoming_meetings(n=8)
        current = self.get_current_rate()

        if not meetings:
            return {'status': 'no_data'}

        warnings: List[str] = []
        path = []
        first_gap: Optional[str] = None

        # Anchor on the same pre-meeting rate the probabilities use. Its own
        # fallback warnings are reported there, so they are discarded here.
        start_rate, _src = self._pre_meeting_rate(meetings[0]['date'], current, [])

        for meeting in meetings:
            end_rate, source, _caveats = self.calculator.meeting_end_rate(
                meeting['date'], start_rate
            )

            if end_rate is None:
                # No quote for this meeting: hold the rate flat rather than
                # inventing a move, and remember where the curve ran out.
                end_rate, source = start_rate, 'carried_forward'
                if first_gap is None:
                    first_gap = meeting['date_str']

            path.append({
                'meeting': meeting['date_str'],
                'date': meeting['date'],
                'days_until': meeting['days_until'],
                'has_sep': meeting['has_sep'],
                'implied_rate': None if source == 'carried_forward' else round(end_rate, 3),
                # Rounded to the nearest 12.5bp for display only — the chain
                # itself carries the unrounded rate so errors don't accumulate.
                'expected_rate': round(end_rate * 8) / 8,
                'change_from_current': round(end_rate - current['mid'], 3),
                'change_from_prior': round(end_rate - start_rate, 3),
                'change_bps': round((end_rate - current['mid']) * 100),
                'source': source,
            })

            start_rate = end_rate

        if first_gap is not None:
            warnings.append(
                f"No futures quotes for the rate path from {first_gap} onward; those "
                "meetings are carried forward flat, so the terminal rate is a floor, "
                "not a market-implied estimate."
            )

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
            'warnings': warnings,
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

                # Data quality. Path gaps are disclosed but do NOT set
                # `degraded` — that flag labels the next-meeting probability
                # panel, which can be perfectly good while a far-dated contract
                # is missing.
                'data_source': probs.get('data_source', 'unknown'),
                'warnings': _merge_warnings(probs.get('warnings'), path.get('warnings')),
                'degraded': probs.get('degraded', False),
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
