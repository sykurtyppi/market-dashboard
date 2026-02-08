"""
Options Flow Scanner - Institutional Grade

Detects unusual options activity using Yahoo Finance data with improved
scoring and multi-ticker support.

Features:
- Unusual volume detection (Volume >> Open Interest)
- Put/Call ratio analysis per ticker
- ATM straddle activity detection (rangebound signals)
- Multi-expiry analysis (weekly + monthly)
- Relative scoring (normalized by ticker's typical volume)
- Per-ticker diversity (max findings per symbol)

Limitations (free data):
- Yahoo Finance data is delayed (not real-time)
- No block trade or sweep detection
- Cannot distinguish buy vs sell
- No dark pool data

For institutional-grade flow data, consider:
- Unusual Whales, FlowAlgo, Tradytics, Cboe LiveVol
"""

import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
import yfinance as yf
import pytz

logger = logging.getLogger(__name__)

# ============================================================
# MARKET HOURS & EXPIRATION UTILITIES
# ============================================================

def get_market_status() -> Dict:
    """
    Get current market status and timing info

    Returns:
        Dict with is_open, status_text, next_open, last_close
    """
    eastern = pytz.timezone('US/Eastern')
    now = datetime.now(eastern)

    # Market hours: 9:30 AM - 4:00 PM ET, Mon-Fri
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

    is_weekday = now.weekday() < 5
    is_market_hours = market_open <= now <= market_close
    is_open = is_weekday and is_market_hours

    if is_open:
        status = "OPEN"
        status_color = "#4CAF50"
    elif now < market_open and is_weekday:
        status = "PRE-MARKET"
        status_color = "#FF9800"
    elif now > market_close and is_weekday:
        status = "AFTER-HOURS"
        status_color = "#9E9E9E"
    else:
        status = "CLOSED"
        status_color = "#F44336"

    # Calculate last trading day
    last_trading_day = now.date()
    if now.weekday() == 5:  # Saturday
        last_trading_day = now.date() - timedelta(days=1)
    elif now.weekday() == 6:  # Sunday
        last_trading_day = now.date() - timedelta(days=2)
    elif now < market_open:
        # Before market open, use previous day
        if now.weekday() == 0:  # Monday
            last_trading_day = now.date() - timedelta(days=3)
        else:
            last_trading_day = now.date() - timedelta(days=1)

    return {
        'is_open': is_open,
        'status': status,
        'status_color': status_color,
        'current_time': now.strftime('%Y-%m-%d %H:%M:%S ET'),
        'last_trading_day': last_trading_day.strftime('%Y-%m-%d'),
        'market_close_time': market_close.strftime('%H:%M ET'),
    }


def is_expiration_valid(expiry_str: str, min_dte: int = 1, max_dte: int = 45) -> Tuple[bool, int]:
    """
    Check if an expiration date is valid for analysis

    Args:
        expiry_str: Expiration date string (YYYY-MM-DD)
        min_dte: Minimum days to expiration (default 1, excludes 0DTE)
        max_dte: Maximum days to expiration (default 45)

    Returns:
        Tuple of (is_valid, days_to_expiry)
    """
    try:
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

        expiry_date = datetime.strptime(expiry_str, '%Y-%m-%d').date()
        today = now.date()

        dte = (expiry_date - today).days

        # If it's 0DTE and after market close, it's expired
        if dte == 0 and now >= market_close:
            return False, 0

        # Check if within valid range
        is_valid = min_dte <= dte <= max_dte

        return is_valid, dte

    except Exception:
        return False, -1


def filter_valid_expirations(expirations: List[str], min_dte: int = 1, max_dte: int = 45) -> List[Tuple[str, int]]:
    """
    Filter expirations to only valid ones

    Returns:
        List of (expiry_str, dte) tuples, sorted by DTE
    """
    valid = []
    for exp in expirations:
        is_valid, dte = is_expiration_valid(exp, min_dte, max_dte)
        if is_valid:
            valid.append((exp, dte))

    # Sort by DTE
    valid.sort(key=lambda x: x[1])
    return valid


# ============================================================
# CONFIGURATION
# ============================================================

class FlowSignal(Enum):
    """Types of options flow signals"""
    UNUSUAL_VOLUME = "unusual_volume"
    HIGH_VOL_OI = "high_vol_oi"
    PUT_HEAVY = "put_heavy"
    CALL_HEAVY = "call_heavy"
    ATM_ACTIVITY = "atm_activity"
    STRADDLE_SIGNAL = "straddle_signal"
    LARGE_PREMIUM = "large_premium"


# Tickers to scan - organized by category
SCAN_TICKERS = {
    'indices': ['SPY', 'QQQ', 'IWM', 'DIA', 'VIX'],
    'mega_cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'],
    'financials': ['JPM', 'BAC', 'GS', 'MS', 'C'],
    'tech': ['AMD', 'INTC', 'CRM', 'ORCL', 'ADBE'],
    'consumer': ['NFLX', 'DIS', 'SBUX', 'MCD', 'NKE'],
    'speculative': ['COIN', 'GME', 'AMC', 'MARA', 'RIOT'],
    'semis': ['AVGO', 'QCOM', 'MU', 'AMAT', 'LRCX'],
}

# Flatten for scanning
ALL_TICKERS = []
for category_tickers in SCAN_TICKERS.values():
    ALL_TICKERS.extend(category_tickers)

# Thresholds - tiered by ticker type
THRESHOLDS = {
    'indices': {
        'min_volume': 5000,        # Lower threshold to catch more
        'vol_oi_ratio': 2.0,       # Volume > 2x OI
        'atm_range_pct': 2.0,      # Within 2% of current price
    },
    'mega_cap': {
        'min_volume': 2000,
        'vol_oi_ratio': 2.5,
        'atm_range_pct': 3.0,
    },
    'default': {
        'min_volume': 1000,
        'vol_oi_ratio': 3.0,
        'atm_range_pct': 5.0,
    }
}

# Per-ticker findings limit
MAX_FINDINGS_PER_TICKER = 3


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class OptionActivity:
    """Represents unusual options activity"""
    ticker: str
    option_type: str  # CALL or PUT
    strike: float
    expiry: str
    volume: int
    open_interest: int
    vol_oi_ratio: float
    last_price: float
    implied_vol: Optional[float]
    moneyness_pct: float
    stock_price: float
    signal_type: FlowSignal
    signal_description: str
    sentiment: str  # BULLISH, BEARISH, NEUTRAL
    score: float
    premium_value: float  # Volume * Last Price * 100
    category: str


@dataclass
class StraddleActivity:
    """Represents potential straddle/rangebound activity"""
    ticker: str
    strike: float
    expiry: str
    call_volume: int
    put_volume: int
    combined_volume: int
    call_premium: float
    put_premium: float
    straddle_cost: float
    implied_range_pct: float
    stock_price: float
    score: float


# ============================================================
# OPTIONS FLOW COLLECTOR
# ============================================================

class OptionsFlowCollector:
    """
    Institutional-grade options flow scanner

    Improvements over basic version:
    - Normalized scoring (relative to ticker's typical volume)
    - Per-ticker diversity (max findings per symbol)
    - Multi-expiry analysis
    - Straddle/rangebound detection
    - Premium-based weighting
    - Category-aware thresholds
    """

    def __init__(self):
        self.tickers = ALL_TICKERS
        self._cache = {}
        self._cache_time = {}
        self._cache_ttl = 120  # 2 minute cache

    def _get_ticker_category(self, ticker: str) -> str:
        """Get category for a ticker"""
        for category, tickers in SCAN_TICKERS.items():
            if ticker in tickers:
                return category
        return 'default'

    def _get_thresholds(self, ticker: str) -> Dict:
        """Get appropriate thresholds for a ticker"""
        category = self._get_ticker_category(ticker)
        if category in THRESHOLDS:
            return THRESHOLDS[category]
        return THRESHOLDS['default']

    def scan_unusual_activity(self, tickers: List[str] = None) -> List[Dict]:
        """
        Scan tickers for unusual options activity

        Returns:
            List of unusual activity findings, diversified across tickers
        """
        if tickers is None:
            tickers = self.tickers

        all_findings = []
        ticker_findings_count = {}

        for ticker in tickers:
            try:
                results = self._analyze_ticker_options(ticker)
                if results:
                    # Limit findings per ticker
                    for finding in results[:MAX_FINDINGS_PER_TICKER]:
                        all_findings.append(finding)
                    ticker_findings_count[ticker] = len(results)
            except Exception as e:
                logger.debug(f"Error scanning {ticker}: {e}")
                continue

        # Sort by score (now properly normalized)
        all_findings.sort(key=lambda x: x.get('score', 0), reverse=True)

        return all_findings[:30]  # Top 30 findings

    def _analyze_ticker_options(self, ticker: str) -> List[Dict]:
        """
        Analyze options chain for a single ticker

        Returns:
            List of unusual activity findings
        """
        findings = []
        thresholds = self._get_thresholds(ticker)
        category = self._get_ticker_category(ticker)

        try:
            stock = yf.Ticker(ticker)
            expirations = stock.options

            if not expirations:
                return []

            # Get current stock price
            current_price = self._get_stock_price(stock)
            if not current_price:
                return []

            # Filter to valid expirations (skip 0DTE after market close, focus on 1-45 DTE)
            valid_expirations = filter_valid_expirations(expirations, min_dte=1, max_dte=45)

            if not valid_expirations:
                # Fallback: if no valid expirations, try including 0DTE during market hours
                valid_expirations = filter_valid_expirations(expirations, min_dte=0, max_dte=45)

            if not valid_expirations:
                logger.debug(f"{ticker}: No valid expirations found")
                return []

            # Analyze up to 3 nearest valid expirations
            for expiry, dte in valid_expirations[:3]:
                try:
                    chain = stock.option_chain(expiry)
                    calls = chain.calls
                    puts = chain.puts

                    # Analyze individual contracts (pass DTE for weighting)
                    call_findings = self._find_unusual_contracts(
                        calls, ticker, 'CALL', current_price, expiry, thresholds, category, dte
                    )
                    findings.extend(call_findings)

                    put_findings = self._find_unusual_contracts(
                        puts, ticker, 'PUT', current_price, expiry, thresholds, category, dte
                    )
                    findings.extend(put_findings)

                    # Check for straddle activity
                    straddle = self._detect_straddle_activity(
                        calls, puts, ticker, current_price, expiry, category, dte
                    )
                    if straddle:
                        findings.append(straddle)

                    # Check P/C ratio extremes
                    pc_finding = self._check_put_call_ratio(
                        calls, puts, ticker, expiry, category, dte
                    )
                    if pc_finding:
                        findings.append(pc_finding)

                except Exception as e:
                    logger.debug(f"Error getting chain for {ticker} {expiry}: {e}")
                    continue

            # Sort by score within this ticker
            findings.sort(key=lambda x: x.get('score', 0), reverse=True)

            return findings

        except Exception as e:
            logger.debug(f"Error analyzing {ticker}: {e}")
            return []

    def _get_stock_price(self, stock) -> Optional[float]:
        """Get current stock price"""
        try:
            info = stock.info
            price = info.get('currentPrice') or info.get('regularMarketPrice')
            if price:
                return price

            hist = stock.history(period='1d')
            if not hist.empty:
                return hist['Close'].iloc[-1]

            return None
        except Exception as e:
            self.logger.debug(f"Could not get stock price for {ticker}: {e}")
            return None

    def _find_unusual_contracts(
        self,
        options_df: pd.DataFrame,
        ticker: str,
        option_type: str,
        stock_price: float,
        expiry: str,
        thresholds: Dict,
        category: str,
        dte: int = None
    ) -> List[Dict]:
        """Find unusual activity in options contracts"""
        findings = []

        if options_df.empty:
            return findings

        required_cols = ['strike', 'volume', 'openInterest', 'lastPrice']
        if not all(col in options_df.columns for col in required_cols):
            return findings

        min_vol = thresholds['min_volume']
        min_vol_oi = thresholds['vol_oi_ratio']
        atm_range = thresholds['atm_range_pct']

        # Calculate total volume for this chain (for relative scoring)
        total_chain_volume = options_df['volume'].sum() or 1

        for _, row in options_df.iterrows():
            volume = row.get('volume', 0) or 0
            open_interest = row.get('openInterest', 0) or 1
            strike = row.get('strike', 0)
            last_price = row.get('lastPrice', 0) or 0
            implied_vol = row.get('impliedVolatility', 0)

            # Skip very low volume
            if volume < min_vol:
                continue

            # Calculate metrics
            vol_oi_ratio = volume / max(open_interest, 1)
            moneyness_pct = ((strike / stock_price) - 1) * 100 if stock_price > 0 else 0
            premium_value = volume * last_price * 100  # Total premium traded

            # Relative volume (what % of total chain volume is this contract?)
            relative_volume = volume / total_chain_volume

            # Determine if unusual
            is_unusual = False
            signal_type = None
            signal_desc = None
            score = 0

            # Check 1: High volume/OI ratio
            if vol_oi_ratio >= min_vol_oi:
                is_unusual = True
                signal_type = FlowSignal.HIGH_VOL_OI
                signal_desc = f"Volume {vol_oi_ratio:.1f}x Open Interest"
                # Score based on ratio + relative volume
                score = min(100, (vol_oi_ratio * 15) + (relative_volume * 200))

            # Check 2: High ATM activity
            if abs(moneyness_pct) <= atm_range and volume >= min_vol * 2:
                if not is_unusual or vol_oi_ratio >= min_vol_oi * 1.5:
                    is_unusual = True
                    signal_type = FlowSignal.ATM_ACTIVITY
                    signal_desc = f"High ATM volume ({volume:,})"
                    score = max(score, min(100, 40 + (relative_volume * 300)))

            # Check 3: Large premium (institutional size)
            if premium_value >= 500000:  # $500k+ premium
                is_unusual = True
                signal_type = FlowSignal.LARGE_PREMIUM
                signal_desc = f"Large premium (${premium_value/1e6:.1f}M)"
                score = max(score, min(100, 50 + (premium_value / 100000)))

            if is_unusual and signal_type:
                sentiment = 'BULLISH' if option_type == 'CALL' else 'BEARISH'
                emoji = '📈' if sentiment == 'BULLISH' else '📉'

                # Format DTE label
                if dte is not None:
                    if dte == 0:
                        dte_label = "0DTE"
                    elif dte == 1:
                        dte_label = "1 day"
                    elif dte <= 7:
                        dte_label = f"{dte}d (weekly)"
                    elif dte <= 30:
                        dte_label = f"{dte}d"
                    else:
                        dte_label = f"{dte}d (monthly)"
                else:
                    dte_label = None

                findings.append({
                    'ticker': ticker,
                    'type': option_type,
                    'strike': strike,
                    'expiry': expiry,
                    'dte': dte,
                    'dte_label': dte_label,
                    'volume': int(volume),
                    'open_interest': int(open_interest),
                    'vol_oi_ratio': round(vol_oi_ratio, 2),
                    'last_price': last_price,
                    'moneyness_pct': round(moneyness_pct, 1),
                    'implied_vol': round(implied_vol * 100, 1) if implied_vol else None,
                    'premium_value': round(premium_value),
                    'signal_type': signal_type.value,
                    'signal': signal_desc,
                    'sentiment': sentiment,
                    'emoji': emoji,
                    'score': round(score, 1),
                    'category': category,
                    'stock_price': round(stock_price, 2),
                })

        return findings

    def _detect_straddle_activity(
        self,
        calls_df: pd.DataFrame,
        puts_df: pd.DataFrame,
        ticker: str,
        stock_price: float,
        expiry: str,
        category: str,
        dte: int = None
    ) -> Optional[Dict]:
        """
        Detect potential straddle activity (ATM calls + puts)

        Straddle activity suggests market expects rangebound price action
        or is hedging for big moves.
        """
        if calls_df.empty or puts_df.empty:
            return None

        # Find ATM strike (closest to current price)
        all_strikes = set(calls_df['strike'].tolist()) & set(puts_df['strike'].tolist())
        if not all_strikes:
            return None

        atm_strike = min(all_strikes, key=lambda x: abs(x - stock_price))

        # Get ATM call and put
        atm_calls = calls_df[calls_df['strike'] == atm_strike]
        atm_puts = puts_df[puts_df['strike'] == atm_strike]

        if atm_calls.empty or atm_puts.empty:
            return None

        call_row = atm_calls.iloc[0]
        put_row = atm_puts.iloc[0]

        call_vol = call_row.get('volume', 0) or 0
        put_vol = put_row.get('volume', 0) or 0
        call_price = call_row.get('lastPrice', 0) or 0
        put_price = put_row.get('lastPrice', 0) or 0

        # Check if both have significant volume
        min_straddle_vol = 1000 if category in ['indices', 'mega_cap'] else 500

        if call_vol >= min_straddle_vol and put_vol >= min_straddle_vol:
            # Calculate straddle metrics
            combined_vol = call_vol + put_vol
            straddle_cost = call_price + put_price
            implied_range_pct = (straddle_cost / stock_price) * 100

            # Check if volumes are roughly balanced (indicates straddle vs directional)
            vol_ratio = min(call_vol, put_vol) / max(call_vol, put_vol, 1)

            if vol_ratio >= 0.4:  # At least 40% balanced
                score = min(100, 30 + (combined_vol / 500) + (vol_ratio * 30))

                # Format DTE label
                if dte is not None:
                    if dte == 0:
                        dte_label = "0DTE"
                    elif dte == 1:
                        dte_label = "1 day"
                    elif dte <= 7:
                        dte_label = f"{dte}d (weekly)"
                    elif dte <= 30:
                        dte_label = f"{dte}d"
                    else:
                        dte_label = f"{dte}d (monthly)"
                else:
                    dte_label = None

                return {
                    'ticker': ticker,
                    'type': 'STRADDLE',
                    'strike': atm_strike,
                    'expiry': expiry,
                    'dte': dte,
                    'dte_label': dte_label,
                    'volume': combined_vol,
                    'call_volume': int(call_vol),
                    'put_volume': int(put_vol),
                    'open_interest': int((call_row.get('openInterest', 0) or 0) +
                                        (put_row.get('openInterest', 0) or 0)),
                    'vol_oi_ratio': None,
                    'last_price': straddle_cost,
                    'straddle_cost': round(straddle_cost, 2),
                    'implied_range_pct': round(implied_range_pct, 1),
                    'moneyness_pct': 0,
                    'implied_vol': None,
                    'premium_value': round(combined_vol * straddle_cost * 100),
                    'signal_type': FlowSignal.STRADDLE_SIGNAL.value,
                    'signal': f"ATM Straddle Activity (±{implied_range_pct:.1f}% range)",
                    'sentiment': 'NEUTRAL',
                    'emoji': '⚖️',
                    'score': round(score, 1),
                    'category': category,
                    'stock_price': round(stock_price, 2),
                }

        return None

    def _check_put_call_ratio(
        self,
        calls_df: pd.DataFrame,
        puts_df: pd.DataFrame,
        ticker: str,
        expiry: str,
        category: str,
        dte: int = None
    ) -> Optional[Dict]:
        """Check for extreme put/call ratio"""
        if calls_df.empty or puts_df.empty:
            return None

        call_vol = calls_df['volume'].sum() if 'volume' in calls_df else 0
        put_vol = puts_df['volume'].sum() if 'volume' in puts_df else 0

        if call_vol == 0:
            return None

        pc_ratio = put_vol / call_vol

        # Format DTE label
        if dte is not None:
            if dte == 0:
                dte_label = "0DTE"
            elif dte == 1:
                dte_label = "1 day"
            elif dte <= 7:
                dte_label = f"{dte}d (weekly)"
            elif dte <= 30:
                dte_label = f"{dte}d"
            else:
                dte_label = f"{dte}d (monthly)"
        else:
            dte_label = None

        # Only flag extreme ratios
        if pc_ratio > 2.0:
            score = min(100, 40 + (pc_ratio * 15))
            return {
                'ticker': ticker,
                'type': 'PUT_HEAVY',
                'strike': None,
                'expiry': expiry,
                'dte': dte,
                'dte_label': dte_label,
                'volume': int(put_vol),
                'open_interest': None,
                'vol_oi_ratio': None,
                'last_price': None,
                'moneyness_pct': None,
                'implied_vol': None,
                'premium_value': None,
                'signal_type': FlowSignal.PUT_HEAVY.value,
                'signal': f"P/C Ratio {pc_ratio:.2f} - Heavy put activity",
                'put_call_ratio': round(pc_ratio, 2),
                'sentiment': 'BEARISH',
                'emoji': '🐻',
                'score': round(score, 1),
                'category': category,
                'stock_price': None,
            }
        elif pc_ratio < 0.4:
            score = min(100, 40 + ((1/pc_ratio) * 10))
            return {
                'ticker': ticker,
                'type': 'CALL_HEAVY',
                'strike': None,
                'expiry': expiry,
                'dte': dte,
                'dte_label': dte_label,
                'volume': int(call_vol),
                'open_interest': None,
                'vol_oi_ratio': None,
                'last_price': None,
                'moneyness_pct': None,
                'implied_vol': None,
                'premium_value': None,
                'signal_type': FlowSignal.CALL_HEAVY.value,
                'signal': f"P/C Ratio {pc_ratio:.2f} - Heavy call activity",
                'put_call_ratio': round(pc_ratio, 2),
                'sentiment': 'BULLISH',
                'emoji': '🐂',
                'score': round(score, 1),
                'category': category,
                'stock_price': None,
            }

        return None

    def get_market_options_summary(self) -> Dict:
        """Get overall market options activity summary"""
        spy_analysis = self._get_index_options_summary('SPY')
        qqq_analysis = self._get_index_options_summary('QQQ')
        iwm_analysis = self._get_index_options_summary('IWM')

        return {
            'spy': spy_analysis,
            'qqq': qqq_analysis,
            'iwm': iwm_analysis,
            'timestamp': datetime.now().isoformat(),
        }

    def _get_index_options_summary(self, ticker: str) -> Dict:
        """Get detailed options summary for an index ETF"""
        try:
            stock = yf.Ticker(ticker)
            expirations = stock.options

            if not expirations:
                return {'status': 'unavailable', 'ticker': ticker}

            # Get current price
            current_price = self._get_stock_price(stock)

            # Filter to valid expirations
            valid_expirations = filter_valid_expirations(expirations, min_dte=1, max_dte=45)
            if not valid_expirations:
                # Fallback to first expiration if all are 0DTE
                valid_expirations = [(expirations[0], 0)]

            expiry_to_use, dte = valid_expirations[0]
            chain = stock.option_chain(expiry_to_use)
            calls = chain.calls
            puts = chain.puts

            total_call_vol = calls['volume'].sum() if 'volume' in calls.columns else 0
            total_put_vol = puts['volume'].sum() if 'volume' in puts.columns else 0
            total_call_oi = calls['openInterest'].sum() if 'openInterest' in calls.columns else 1
            total_put_oi = puts['openInterest'].sum() if 'openInterest' in puts.columns else 1

            pc_ratio = total_put_vol / max(total_call_vol, 1)

            # Calculate GEX proxy (simplified)
            # Real GEX requires dealer positioning data
            call_oi_near_atm = 0
            put_oi_near_atm = 0
            if current_price:
                atm_range = current_price * 0.03  # 3% range
                for _, row in calls.iterrows():
                    if abs(row['strike'] - current_price) <= atm_range:
                        call_oi_near_atm += row.get('openInterest', 0) or 0
                for _, row in puts.iterrows():
                    if abs(row['strike'] - current_price) <= atm_range:
                        put_oi_near_atm += row.get('openInterest', 0) or 0

            # Determine sentiment
            if pc_ratio > 1.3:
                sentiment = 'BEARISH'
                color = '#F44336'
            elif pc_ratio > 0.9:
                sentiment = 'NEUTRAL'
                color = '#9E9E9E'
            else:
                sentiment = 'BULLISH'
                color = '#4CAF50'

            return {
                'ticker': ticker,
                'current_price': current_price,
                'expiry': expiry_to_use,
                'dte': dte,
                'total_call_volume': int(total_call_vol),
                'total_put_volume': int(total_put_vol),
                'total_call_oi': int(total_call_oi),
                'total_put_oi': int(total_put_oi),
                'put_call_ratio': round(pc_ratio, 3),
                'call_oi_near_atm': int(call_oi_near_atm),
                'put_oi_near_atm': int(put_oi_near_atm),
                'sentiment': sentiment,
                'color': color,
                'status': 'ok',
            }

        except Exception as e:
            logger.error(f"Error getting {ticker} options summary: {e}")
            return {'status': 'error', 'ticker': ticker, 'message': str(e)}

    def detect_straddle_setups(self) -> List[Dict]:
        """
        Scan for potential straddle setups across all tickers

        Returns list of tickers with balanced ATM call/put volume,
        indicating expected rangebound action.
        """
        straddles = []

        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                expirations = stock.options

                if not expirations:
                    continue

                current_price = self._get_stock_price(stock)
                if not current_price:
                    continue

                # Filter to valid expirations (skip 0DTE after hours)
                valid_expirations = filter_valid_expirations(expirations, min_dte=1, max_dte=30)
                if not valid_expirations:
                    continue

                expiry, dte = valid_expirations[0]
                chain = stock.option_chain(expiry)
                calls = chain.calls
                puts = chain.puts

                straddle = self._detect_straddle_activity(
                    calls, puts, ticker, current_price,
                    expiry, self._get_ticker_category(ticker), dte
                )

                if straddle:
                    straddles.append(straddle)

            except Exception as e:
                logger.debug(f"Error checking straddle for {ticker}: {e}")
                continue

        # Sort by score
        straddles.sort(key=lambda x: x.get('score', 0), reverse=True)

        return straddles

    def get_options_flow_summary(self) -> Dict:
        """
        Get comprehensive options flow summary for dashboard

        Returns:
            Dict with unusual activity, market sentiment, and straddle signals
        """
        # Get market status for context
        market_status = get_market_status()

        # Scan for unusual activity
        unusual = self.scan_unusual_activity()

        # Get market summaries
        market_summary = self.get_market_options_summary()

        # Detect straddle setups
        straddles = [u for u in unusual if u.get('type') == 'STRADDLE']

        # Count signals by sentiment
        bullish_count = len([u for u in unusual if u.get('sentiment') == 'BULLISH'])
        bearish_count = len([u for u in unusual if u.get('sentiment') == 'BEARISH'])
        neutral_count = len([u for u in unusual if u.get('sentiment') == 'NEUTRAL'])

        # Count unique tickers
        unique_tickers = set(u.get('ticker') for u in unusual)

        # Calculate aggregate premium
        total_premium = sum(u.get('premium_value', 0) or 0 for u in unusual)

        # Determine overall signal
        if bullish_count > bearish_count * 1.5 and bullish_count > 3:
            overall_signal = 'BULLISH FLOW'
            signal_color = '#4CAF50'
        elif bearish_count > bullish_count * 1.5 and bearish_count > 3:
            overall_signal = 'BEARISH FLOW'
            signal_color = '#F44336'
        elif neutral_count > max(bullish_count, bearish_count):
            overall_signal = 'HEDGING/STRADDLE ACTIVITY'
            signal_color = '#9E9E9E'
        else:
            overall_signal = 'MIXED FLOW'
            signal_color = '#FF9800'

        # Build data freshness message
        if market_status['is_open']:
            freshness_msg = f"Live data as of {market_status['current_time']}"
            freshness_color = '#4CAF50'
        else:
            freshness_msg = f"Data from {market_status['last_trading_day']} market close"
            freshness_color = '#FF9800'

        return {
            'unusual_activity': unusual[:15],  # Top 15
            'total_unusual_count': len(unusual),
            'unique_tickers': len(unique_tickers),
            'bullish_signals': bullish_count,
            'bearish_signals': bearish_count,
            'neutral_signals': neutral_count,
            'straddle_count': len(straddles),
            'total_premium': total_premium,
            'overall_signal': overall_signal,
            'signal_color': signal_color,
            'spy_summary': market_summary.get('spy', {}),
            'qqq_summary': market_summary.get('qqq', {}),
            'iwm_summary': market_summary.get('iwm', {}),
            'straddle_setups': straddles[:5],  # Top 5 straddles
            'timestamp': datetime.now().isoformat(),
            'market_status': market_status,
            'freshness_msg': freshness_msg,
            'freshness_color': freshness_color,
            'data_note': 'Yahoo Finance options data (15-20 min delay). Excludes expired 0DTE options.',
        }

    def get_ticker_flow(self, ticker: str) -> Dict:
        """Get detailed options flow for a specific ticker"""
        try:
            stock = yf.Ticker(ticker)
            expirations = stock.options

            if not expirations:
                return {'status': 'no_options', 'ticker': ticker}

            current_price = self._get_stock_price(stock)
            category = self._get_ticker_category(ticker)

            # Filter to valid expirations
            valid_expirations = filter_valid_expirations(expirations, min_dte=1, max_dte=45)

            # Get chains for multiple valid expirations
            all_findings = []
            expiry_summaries = []

            for expiry, dte in valid_expirations[:5]:  # Up to 5 expirations
                try:
                    chain = stock.option_chain(expiry)
                    calls = chain.calls
                    puts = chain.puts

                    call_vol = calls['volume'].sum() if 'volume' in calls.columns else 0
                    put_vol = puts['volume'].sum() if 'volume' in puts.columns else 0
                    pc_ratio = put_vol / max(call_vol, 1)

                    expiry_summaries.append({
                        'expiry': expiry,
                        'dte': dte,
                        'call_volume': int(call_vol),
                        'put_volume': int(put_vol),
                        'put_call_ratio': round(pc_ratio, 3),
                    })

                    # Get unusual activity for this expiry
                    thresholds = self._get_thresholds(ticker)
                    call_findings = self._find_unusual_contracts(
                        calls, ticker, 'CALL', current_price, expiry, thresholds, category, dte
                    )
                    put_findings = self._find_unusual_contracts(
                        puts, ticker, 'PUT', current_price, expiry, thresholds, category, dte
                    )
                    all_findings.extend(call_findings)
                    all_findings.extend(put_findings)

                except Exception:
                    continue

            all_findings.sort(key=lambda x: x.get('score', 0), reverse=True)

            # Get market status for context
            market_status = get_market_status()

            return {
                'status': 'ok',
                'ticker': ticker,
                'current_price': current_price,
                'category': category,
                'expiry_summaries': expiry_summaries,
                'unusual_activity': all_findings[:10],
                'timestamp': datetime.now().isoformat(),
                'market_status': market_status,
            }

        except Exception as e:
            return {'status': 'error', 'ticker': ticker, 'message': str(e)}


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def get_options_flow() -> Dict:
    """Quick access to options flow summary"""
    collector = OptionsFlowCollector()
    return collector.get_options_flow_summary()


def get_straddle_signals() -> List[Dict]:
    """Get all straddle/rangebound signals"""
    collector = OptionsFlowCollector()
    return collector.detect_straddle_setups()


def get_ticker_options(ticker: str) -> Dict:
    """Get options flow for specific ticker"""
    collector = OptionsFlowCollector()
    return collector.get_ticker_flow(ticker)
