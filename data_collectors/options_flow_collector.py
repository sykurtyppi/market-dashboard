"""
Options Flow Scanner
Detects unusual options activity using free data from Yahoo Finance

Signals tracked:
- Unusual volume (volume >> open interest)
- Large open interest changes
- Put/Call ratio extremes by ticker
- Near-expiry high volume (weekly options activity)

Limitations:
- Yahoo Finance doesn't provide real-time flow data
- No block trade or sweep detection (requires paid data)
- Data is delayed and aggregated

For real institutional-grade options flow, consider:
- Unusual Whales ($)
- FlowAlgo ($)
- Tradytics ($)
- Cboe LiveVol ($)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Tickers to scan for unusual options activity
SCAN_TICKERS = [
    # Major indices/ETFs
    'SPY', 'QQQ', 'IWM', 'DIA',
    # Mega caps
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
    # Financials
    'JPM', 'BAC', 'GS',
    # Other high-activity
    'AMD', 'NFLX', 'COIN', 'GME', 'AMC',
]

# Volume thresholds for "unusual"
UNUSUAL_VOLUME_MULTIPLIER = 3.0  # Volume > 3x open interest
HIGH_VOLUME_THRESHOLD = 10000    # Minimum volume to consider


class OptionsFlowCollector:
    """
    Scans for unusual options activity using Yahoo Finance data

    Note: This is a simplified scanner. Professional options flow
    analysis requires real-time data feeds with block/sweep detection.
    """

    def __init__(self):
        self.tickers = SCAN_TICKERS
        self._cache = {}
        self._cache_time = None

    def scan_unusual_activity(self, tickers: List[str] = None) -> List[Dict]:
        """
        Scan tickers for unusual options activity

        Args:
            tickers: List of tickers to scan (default: SCAN_TICKERS)

        Returns:
            List of unusual activity findings
        """
        if tickers is None:
            tickers = self.tickers

        findings = []

        for ticker in tickers:
            try:
                result = self._analyze_ticker_options(ticker)
                if result:
                    findings.extend(result)
            except Exception as e:
                logger.debug(f"Error scanning {ticker}: {e}")
                continue

        # Sort by significance score
        findings.sort(key=lambda x: x.get('score', 0), reverse=True)

        return findings[:20]  # Return top 20 findings

    def _analyze_ticker_options(self, ticker: str) -> List[Dict]:
        """
        Analyze options chain for a single ticker

        Returns:
            List of unusual activity findings for this ticker
        """
        findings = []

        try:
            stock = yf.Ticker(ticker)

            # Get available expiration dates
            expirations = stock.options
            if not expirations:
                return []

            # Get current stock price
            info = stock.info
            current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)

            if not current_price:
                hist = stock.history(period='1d')
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]

            # Analyze front-month options (most liquid)
            front_expiry = expirations[0]

            try:
                chain = stock.option_chain(front_expiry)
            except Exception:
                return []

            calls = chain.calls
            puts = chain.puts

            # Analyze calls
            call_findings = self._find_unusual_options(
                calls, ticker, 'CALL', current_price, front_expiry
            )
            findings.extend(call_findings)

            # Analyze puts
            put_findings = self._find_unusual_options(
                puts, ticker, 'PUT', current_price, front_expiry
            )
            findings.extend(put_findings)

            # Check overall put/call ratio
            total_call_vol = calls['volume'].sum() if 'volume' in calls else 0
            total_put_vol = puts['volume'].sum() if 'volume' in puts else 0

            if total_call_vol > 0:
                pc_ratio = total_put_vol / total_call_vol

                if pc_ratio > 1.5:
                    findings.append({
                        'ticker': ticker,
                        'type': 'PUT_HEAVY',
                        'signal': 'High Put Volume',
                        'description': f'P/C ratio {pc_ratio:.2f} - Heavy put buying',
                        'put_call_ratio': round(pc_ratio, 2),
                        'expiry': front_expiry,
                        'sentiment': 'BEARISH',
                        'score': min(100, pc_ratio * 30),
                    })
                elif pc_ratio < 0.5:
                    findings.append({
                        'ticker': ticker,
                        'type': 'CALL_HEAVY',
                        'signal': 'High Call Volume',
                        'description': f'P/C ratio {pc_ratio:.2f} - Heavy call buying',
                        'put_call_ratio': round(pc_ratio, 2),
                        'expiry': front_expiry,
                        'sentiment': 'BULLISH',
                        'score': min(100, (1/pc_ratio) * 30),
                    })

        except Exception as e:
            logger.debug(f"Error analyzing {ticker} options: {e}")

        return findings

    def _find_unusual_options(self, options_df: pd.DataFrame, ticker: str,
                               option_type: str, stock_price: float,
                               expiry: str) -> List[Dict]:
        """
        Find unusual activity in options DataFrame

        Looks for:
        - High volume relative to open interest
        - Large absolute volume
        - Near-the-money high activity
        """
        findings = []

        if options_df.empty:
            return findings

        required_cols = ['strike', 'volume', 'openInterest', 'lastPrice']
        if not all(col in options_df.columns for col in required_cols):
            return findings

        for _, row in options_df.iterrows():
            volume = row.get('volume', 0) or 0
            open_interest = row.get('openInterest', 0) or 1
            strike = row.get('strike', 0)
            last_price = row.get('lastPrice', 0)
            implied_vol = row.get('impliedVolatility', 0)

            # Skip low volume
            if volume < HIGH_VOLUME_THRESHOLD:
                continue

            # Calculate volume/OI ratio
            vol_oi_ratio = volume / max(open_interest, 1)

            # Calculate moneyness
            if stock_price > 0:
                moneyness = (strike / stock_price - 1) * 100
            else:
                moneyness = 0

            # Check for unusual activity
            is_unusual = False
            signal = None
            score = 0

            # High volume relative to open interest
            if vol_oi_ratio > UNUSUAL_VOLUME_MULTIPLIER:
                is_unusual = True
                signal = f'Volume {vol_oi_ratio:.1f}x Open Interest'
                score = min(100, vol_oi_ratio * 20)

            # Very high absolute volume near the money
            if volume > 50000 and abs(moneyness) < 5:
                is_unusual = True
                signal = f'Massive volume ({volume:,}) near ATM'
                score = max(score, min(100, volume / 1000))

            if is_unusual and signal:
                # Determine sentiment
                if option_type == 'CALL':
                    sentiment = 'BULLISH'
                    emoji = '📈'
                else:
                    sentiment = 'BEARISH'
                    emoji = '📉'

                findings.append({
                    'ticker': ticker,
                    'type': option_type,
                    'strike': strike,
                    'expiry': expiry,
                    'volume': int(volume),
                    'open_interest': int(open_interest),
                    'vol_oi_ratio': round(vol_oi_ratio, 2),
                    'last_price': last_price,
                    'moneyness_pct': round(moneyness, 1),
                    'implied_vol': round(implied_vol * 100, 1) if implied_vol else None,
                    'signal': signal,
                    'sentiment': sentiment,
                    'emoji': emoji,
                    'score': score,
                })

        return findings

    def get_market_options_summary(self) -> Dict:
        """
        Get overall market options activity summary

        Returns:
            Dict with market-wide options metrics
        """
        spy_analysis = self._get_spy_options_summary()
        qqq_analysis = self._get_spy_options_summary('QQQ')

        return {
            'spy': spy_analysis,
            'qqq': qqq_analysis,
            'timestamp': datetime.now().isoformat(),
        }

    def _get_spy_options_summary(self, ticker: str = 'SPY') -> Dict:
        """Get options summary for SPY or QQQ"""
        try:
            stock = yf.Ticker(ticker)
            expirations = stock.options

            if not expirations:
                return {'status': 'unavailable'}

            chain = stock.option_chain(expirations[0])

            total_call_vol = chain.calls['volume'].sum() if 'volume' in chain.calls.columns else 0
            total_put_vol = chain.puts['volume'].sum() if 'volume' in chain.puts.columns else 0
            total_call_oi = chain.calls['openInterest'].sum() if 'openInterest' in chain.calls.columns else 1
            total_put_oi = chain.puts['openInterest'].sum() if 'openInterest' in chain.puts.columns else 1

            pc_ratio = total_put_vol / max(total_call_vol, 1)

            # Sentiment based on P/C ratio
            if pc_ratio > 1.2:
                sentiment = 'BEARISH'
                color = '#F44336'
            elif pc_ratio > 0.8:
                sentiment = 'NEUTRAL'
                color = '#9E9E9E'
            else:
                sentiment = 'BULLISH'
                color = '#4CAF50'

            return {
                'ticker': ticker,
                'expiry': expirations[0],
                'total_call_volume': int(total_call_vol),
                'total_put_volume': int(total_put_vol),
                'total_call_oi': int(total_call_oi),
                'total_put_oi': int(total_put_oi),
                'put_call_ratio': round(pc_ratio, 3),
                'sentiment': sentiment,
                'color': color,
            }

        except Exception as e:
            logger.error(f"Error getting {ticker} options summary: {e}")
            return {'status': 'error', 'message': str(e)}

    def get_options_flow_summary(self) -> Dict:
        """
        Get comprehensive options flow summary for dashboard

        Returns:
            Dict with unusual activity and market sentiment
        """
        # Scan for unusual activity
        unusual = self.scan_unusual_activity()

        # Get market summaries
        market_summary = self.get_market_options_summary()

        # Count bullish vs bearish signals
        bullish_count = len([u for u in unusual if u.get('sentiment') == 'BULLISH'])
        bearish_count = len([u for u in unusual if u.get('sentiment') == 'BEARISH'])

        # Overall signal
        if bullish_count > bearish_count * 1.5:
            overall_signal = 'BULLISH FLOW'
            signal_color = '#4CAF50'
        elif bearish_count > bullish_count * 1.5:
            overall_signal = 'BEARISH FLOW'
            signal_color = '#F44336'
        else:
            overall_signal = 'MIXED FLOW'
            signal_color = '#FF9800'

        return {
            'unusual_activity': unusual[:10],  # Top 10
            'total_unusual_count': len(unusual),
            'bullish_signals': bullish_count,
            'bearish_signals': bearish_count,
            'overall_signal': overall_signal,
            'signal_color': signal_color,
            'spy_summary': market_summary.get('spy', {}),
            'qqq_summary': market_summary.get('qqq', {}),
            'timestamp': datetime.now().isoformat(),
            'data_note': 'Based on Yahoo Finance options data (delayed, not real-time flow)',
        }
