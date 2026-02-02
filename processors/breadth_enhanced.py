"""
Enhanced Market Breadth Analysis - Institutional Grade

Adds advanced breadth metrics used by institutional traders:
- McClellan Summation Index (cumulative breadth momentum)
- New Highs vs New Lows ratio
- Percent of stocks above 50/200 SMA
- Breadth Thrust indicators
- Breadth regime classification

Parameters loaded from config/parameters.yaml
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import cfg

logger = logging.getLogger(__name__)


# ============================================================================
# NOTE ON NYSE TICK DATA
# ============================================================================
# Real-time NYSE TICK requires professional data feeds (Bloomberg, TradeStation,
# ThinkorSwim, etc.). It is NOT available via free APIs like Yahoo Finance.
#
# For institutional-grade TICK data, users should:
# 1. Use a broker platform with real-time TICK (TD Ameritrade, Interactive Brokers)
# 2. Subscribe to a professional data feed
# 3. Use the reversal levels defined in config/parameters.yaml:
#    - +1200 / -1200: EXTREME reversal zones
#    - +800 / -800: STRONG momentum levels
#    - +600 / -600: MODERATE levels
# ============================================================================


class EnhancedBreadthAnalyzer:
    """
    Institutional-grade market breadth analysis.

    Key metrics:
    1. McClellan Oscillator & Summation Index
    2. New Highs / New Lows
    3. % Stocks above 50/200 SMA
    4. Breadth regime classification
    5. Thrust signals
    """

    # Representative S&P 500 sample for SMA calculations
    SAMPLE_STOCKS = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
        'LLY', 'V', 'JPM', 'UNH', 'XOM', 'MA', 'PG', 'JNJ', 'HD', 'AVGO',
        'MRK', 'CVX', 'BAC', 'ADBE', 'CRM', 'ORCL', 'WMT', 'KO', 'PEP',
        'CSCO', 'TMO', 'ACN', 'MCD', 'NFLX', 'ABT', 'DIS', 'AMD', 'INTC',
        'NKE', 'VZ', 'CMCSA', 'TXN', 'UPS', 'RTX', 'HON', 'PM', 'NEE',
        'QCOM', 'AMGN', 'LOW', 'UNP', 'IBM'
    ]

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_mcclellan_summation(self, breadth_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate McClellan Summation Index - cumulative version of oscillator.

        The Summation Index is more useful for identifying major market turns.
        - Above +1000: Strong bull market
        - +500 to +1000: Moderately bullish
        - -500 to +500: Neutral/transitional
        - -1000 to -500: Moderately bearish
        - Below -1000: Strong bear market

        Args:
            breadth_df: DataFrame with advancing, declining columns

        Returns:
            Dict with summation index value, signal, and history
        """
        if breadth_df.empty or len(breadth_df) < 39:
            return {
                'value': None,
                'signal': 'INSUFFICIENT_DATA',
                'color': '#9E9E9E',
                'description': 'Need at least 39 days of data'
            }

        df = breadth_df.copy()

        # Calculate net advances
        df['ad_diff'] = df['advancing'] - df['declining']

        # McClellan Oscillator components
        df['ema19'] = df['ad_diff'].ewm(span=19, adjust=False).mean()
        df['ema39'] = df['ad_diff'].ewm(span=39, adjust=False).mean()
        df['mcclellan'] = df['ema19'] - df['ema39']

        # Summation Index = cumulative sum of McClellan Oscillator
        # Start at 0 (or can use a seed value)
        df['summation'] = df['mcclellan'].cumsum()

        current_value = float(df['summation'].iloc[-1])
        current_oscillator = float(df['mcclellan'].iloc[-1])

        # Classify signal
        if current_value > 1000:
            signal = 'STRONG_BULLISH'
            color = '#4CAF50'
            description = f'Summation at {current_value:+,.0f} - Strong bull market regime'
        elif current_value > 500:
            signal = 'BULLISH'
            color = '#8BC34A'
            description = f'Summation at {current_value:+,.0f} - Moderately bullish'
        elif current_value > -500:
            signal = 'NEUTRAL'
            color = '#FF9800'
            description = f'Summation at {current_value:+,.0f} - Transitional/neutral'
        elif current_value > -1000:
            signal = 'BEARISH'
            color = '#FF6B6B'
            description = f'Summation at {current_value:+,.0f} - Moderately bearish'
        else:
            signal = 'STRONG_BEARISH'
            color = '#F44336'
            description = f'Summation at {current_value:+,.0f} - Strong bear market regime'

        return {
            'value': current_value,
            'oscillator': current_oscillator,
            'signal': signal,
            'color': color,
            'description': description,
            'history': df[['date', 'mcclellan', 'summation']].tail(90).to_dict('records')
        }

    def calculate_percent_above_sma(self, period: int = 200, max_workers: int = 10) -> Dict[str, Any]:
        """
        Calculate percentage of stocks trading above their moving average.

        Institutional traders use this to gauge overall market health:
        - >70% above 200 SMA: Strong bull market
        - 50-70%: Healthy market
        - 30-50%: Weakening
        - <30%: Bear market conditions

        Args:
            period: SMA period (50 or 200 typically)
            max_workers: Parallel fetch workers

        Returns:
            Dict with percentage, signal, and stock details
        """
        results = {'above': [], 'below': [], 'error': []}

        def check_stock(ticker: str) -> Dict:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=f'{period + 50}d')

                if len(hist) < period:
                    return {'ticker': ticker, 'status': 'insufficient_data'}

                current_price = hist['Close'].iloc[-1]
                sma = hist['Close'].rolling(period).mean().iloc[-1]

                return {
                    'ticker': ticker,
                    'price': current_price,
                    'sma': sma,
                    'above': current_price > sma,
                    'pct_from_sma': ((current_price / sma) - 1) * 100
                }
            except Exception as e:
                return {'ticker': ticker, 'status': 'error', 'error': str(e)}

        # Parallel fetch
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(check_stock, t): t for t in self.SAMPLE_STOCKS}

            for future in as_completed(futures):
                result = future.result()
                if 'above' in result:
                    if result['above']:
                        results['above'].append(result)
                    else:
                        results['below'].append(result)
                else:
                    results['error'].append(result)

        total_valid = len(results['above']) + len(results['below'])

        if total_valid == 0:
            return {
                'percentage': None,
                'signal': 'NO_DATA',
                'color': '#9E9E9E',
                'description': 'Could not fetch stock data'
            }

        pct_above = (len(results['above']) / total_valid) * 100

        # Signal classification
        if pct_above > 70:
            signal = 'STRONG_BULLISH'
            color = '#4CAF50'
            description = f'{pct_above:.0f}% above {period} SMA - Strong bull market'
        elif pct_above > 50:
            signal = 'BULLISH'
            color = '#8BC34A'
            description = f'{pct_above:.0f}% above {period} SMA - Healthy breadth'
        elif pct_above > 30:
            signal = 'WEAKENING'
            color = '#FF9800'
            description = f'{pct_above:.0f}% above {period} SMA - Breadth weakening'
        else:
            signal = 'BEARISH'
            color = '#F44336'
            description = f'{pct_above:.0f}% above {period} SMA - Bear market breadth'

        return {
            'percentage': pct_above,
            'period': period,
            'stocks_above': len(results['above']),
            'stocks_below': len(results['below']),
            'total_stocks': total_valid,
            'signal': signal,
            'color': color,
            'description': description,
            'above_list': results['above'][:10],  # Top 10 strongest
            'below_list': sorted(results['below'], key=lambda x: x.get('pct_from_sma', 0))[:10]  # Weakest 10
        }

    def classify_breadth_regime(self, breadth_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Classify overall breadth regime using multiple indicators.

        Combines:
        - Current breadth % (daily advancing)
        - Trend (5-day vs 20-day)
        - McClellan Oscillator direction
        - Breadth thrust status

        Returns institutional regime classification.
        """
        if breadth_df.empty or len(breadth_df) < 20:
            return {
                'regime': 'UNKNOWN',
                'confidence': 0,
                'description': 'Insufficient data',
                'color': '#9E9E9E'
            }

        df = breadth_df.copy()

        # Calculate components
        current_breadth = df['breadth_pct'].iloc[-1]
        avg_5d = df['breadth_pct'].tail(5).mean()
        avg_20d = df['breadth_pct'].tail(20).mean()

        # McClellan if available
        if 'mcclellan' in df.columns:
            mcclellan = df['mcclellan'].iloc[-1]
        else:
            df['ad_diff'] = df['advancing'] - df['declining']
            df['ema19'] = df['ad_diff'].ewm(span=19, adjust=False).mean()
            df['ema39'] = df['ad_diff'].ewm(span=39, adjust=False).mean()
            mcclellan = (df['ema19'] - df['ema39']).iloc[-1]

        # Score components (each 0-25 points)
        score = 0
        components = []

        # 1. Current breadth (0-25)
        if current_breadth > 65:
            score += 25
            components.append('Strong daily breadth')
        elif current_breadth > 55:
            score += 18
            components.append('Healthy daily breadth')
        elif current_breadth > 45:
            score += 12
            components.append('Neutral daily breadth')
        elif current_breadth > 35:
            score += 6
            components.append('Weak daily breadth')
        else:
            components.append('Very weak daily breadth')

        # 2. Short-term trend (0-25)
        if avg_5d > avg_20d + 5:
            score += 25
            components.append('Breadth accelerating')
        elif avg_5d > avg_20d:
            score += 18
            components.append('Breadth improving')
        elif avg_5d > avg_20d - 5:
            score += 12
            components.append('Breadth stable')
        else:
            score += 0
            components.append('Breadth deteriorating')

        # 3. McClellan momentum (0-25)
        if mcclellan > 50:
            score += 25
            components.append('Strong momentum')
        elif mcclellan > 20:
            score += 18
            components.append('Positive momentum')
        elif mcclellan > -20:
            score += 12
            components.append('Neutral momentum')
        elif mcclellan > -50:
            score += 6
            components.append('Negative momentum')
        else:
            components.append('Strong negative momentum')

        # 4. Consistency (0-25) - how many recent days above 50%
        recent_strong = (df['breadth_pct'].tail(10) > 50).sum()
        score += int(recent_strong * 2.5)
        components.append(f'{recent_strong}/10 days above 50%')

        # Classify regime
        if score >= 80:
            regime = 'STRONG_BULL'
            color = '#4CAF50'
            description = 'Strong bullish breadth - healthy market'
        elif score >= 60:
            regime = 'BULL'
            color = '#8BC34A'
            description = 'Bullish breadth - favorable conditions'
        elif score >= 40:
            regime = 'NEUTRAL'
            color = '#FF9800'
            description = 'Mixed breadth - watch for direction'
        elif score >= 20:
            regime = 'BEAR'
            color = '#FF6B6B'
            description = 'Bearish breadth - risk elevated'
        else:
            regime = 'STRONG_BEAR'
            color = '#F44336'
            description = 'Strongly bearish breadth - defensive positioning'

        return {
            'regime': regime,
            'score': score,
            'confidence': min(100, score + 20),  # Confidence 20-100
            'components': components,
            'description': description,
            'color': color,
            'details': {
                'current_breadth': current_breadth,
                'avg_5d': avg_5d,
                'avg_20d': avg_20d,
                'mcclellan': mcclellan
            }
        }

    def get_breadth_dashboard_data(self, breadth_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get all breadth data for dashboard display in one call.

        Returns comprehensive breadth analysis package.
        """
        if breadth_df.empty:
            return {'error': 'No breadth data available'}

        # Calculate all metrics
        summation = self.calculate_mcclellan_summation(breadth_df)
        regime = self.classify_breadth_regime(breadth_df)

        # Latest stats
        latest = breadth_df.iloc[-1]

        return {
            'latest': {
                'date': latest['date'],
                'breadth_pct': float(latest['breadth_pct']),
                'advancing': int(latest['advancing']),
                'declining': int(latest['declining']),
                'ad_line': float(latest.get('ad_line', 0))
            },
            'mcclellan_summation': summation,
            'regime': regime,
            'stats': {
                'avg_30d': float(breadth_df['breadth_pct'].tail(30).mean()),
                'std_30d': float(breadth_df['breadth_pct'].tail(30).std()),
                'strong_days_30d': int((breadth_df['breadth_pct'].tail(30) > 60).sum()),
                'weak_days_30d': int((breadth_df['breadth_pct'].tail(30) < 40).sum())
            }
        }


def get_new_highs_lows(sample_stocks: List[str] = None, lookback_days: int = 252) -> Dict[str, Any]:
    """
    Calculate New Highs vs New Lows from a sample of stocks.

    Since NYSE NH/NL tickers aren't available via Yahoo Finance,
    we calculate this from our representative S&P 500 sample.

    A stock is at a "new high" if current price is at 52-week high.
    A stock is at a "new low" if current price is at 52-week low.

    Returns:
        Dict with new highs, new lows, ratio, and signal
    """
    if sample_stocks is None:
        sample_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
            'LLY', 'V', 'JPM', 'UNH', 'XOM', 'MA', 'PG', 'JNJ', 'HD', 'AVGO',
            'MRK', 'CVX', 'BAC', 'ADBE', 'CRM', 'ORCL', 'WMT', 'KO', 'PEP',
            'CSCO', 'TMO', 'ACN', 'MCD', 'NFLX', 'ABT', 'DIS', 'AMD', 'INTC',
            'NKE', 'VZ', 'CMCSA', 'TXN', 'UPS', 'RTX', 'HON', 'PM', 'NEE',
            'QCOM', 'AMGN', 'LOW', 'UNP', 'IBM'
        ]

    new_highs = 0
    new_lows = 0
    near_highs = 0  # Within 5% of high
    near_lows = 0   # Within 5% of low
    total_checked = 0

    def check_stock(ticker: str) -> Dict:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=f'{lookback_days}d')

            if len(hist) < 20:
                return None

            current = hist['Close'].iloc[-1]
            high_52w = hist['High'].max()
            low_52w = hist['Low'].min()

            # Within 2% of 52-week high = "new high"
            at_high = current >= high_52w * 0.98
            # Within 2% of 52-week low = "new low"
            at_low = current <= low_52w * 1.02
            # Within 5% = "near"
            near_high = current >= high_52w * 0.95
            near_low = current <= low_52w * 1.05

            return {
                'at_high': at_high,
                'at_low': at_low,
                'near_high': near_high,
                'near_low': near_low
            }
        except Exception:
            return None

    # Parallel fetch
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(check_stock, t): t for t in sample_stocks}

        for future in as_completed(futures):
            result = future.result()
            if result:
                total_checked += 1
                if result['at_high']:
                    new_highs += 1
                if result['at_low']:
                    new_lows += 1
                if result['near_high']:
                    near_highs += 1
                if result['near_low']:
                    near_lows += 1

    if total_checked == 0:
        return {
            'new_highs': None,
            'new_lows': None,
            'net': None,
            'signal': 'NO_DATA',
            'color': '#9E9E9E',
            'description': 'Could not fetch stock data'
        }

    # Calculate net and percentages
    net = new_highs - new_lows
    pct_at_high = (new_highs / total_checked) * 100
    pct_at_low = (new_lows / total_checked) * 100

    # Signal classification based on sample
    if new_highs > new_lows + 10:
        signal = 'STRONG_BULLISH'
        color = '#4CAF50'
        description = f'{new_highs} stocks at 52w highs vs {new_lows} at lows ({pct_at_high:.0f}% at highs) - Strong breadth'
    elif new_highs > new_lows + 3:
        signal = 'BULLISH'
        color = '#8BC34A'
        description = f'{new_highs} stocks at 52w highs vs {new_lows} at lows - Bullish breadth'
    elif abs(new_highs - new_lows) <= 3:
        signal = 'NEUTRAL'
        color = '#FF9800'
        description = f'{new_highs} stocks at 52w highs vs {new_lows} at lows - Mixed breadth'
    elif new_lows > new_highs + 3:
        signal = 'BEARISH'
        color = '#FF6B6B'
        description = f'{new_highs} stocks at 52w highs vs {new_lows} at lows - Bearish breadth'
    else:
        signal = 'STRONG_BEARISH'
        color = '#F44336'
        description = f'{new_highs} stocks at 52w highs vs {new_lows} at lows ({pct_at_low:.0f}% at lows) - Very weak'

    return {
        'new_highs': new_highs,
        'new_lows': new_lows,
        'net': net,
        'near_highs': near_highs,
        'near_lows': near_lows,
        'total_stocks': total_checked,
        'pct_at_high': pct_at_high,
        'pct_at_low': pct_at_low,
        'signal': signal,
        'color': color,
        'description': description
    }
