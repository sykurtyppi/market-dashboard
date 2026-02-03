"""
CBOE (Chicago Board Options Exchange) Data Collector
Fetches VIX, VIX3M, put/call ratios, and volatility data from official sources

Parameters loaded from config/parameters.yaml
"""

import yfinance as yf
import requests
from bs4 import BeautifulSoup
from typing import Dict, Optional
import logging
from utils.validators import validate_vix, validate_ratio
import pandas as pd
from datetime import datetime

from config import cfg

logger = logging.getLogger(__name__)


class CBOECollector:
    """
    Collects CBOE market data including:
    - VIX (spot)
    - VIX3M (3-month VIX for real term structure)
    - Equity put/call ratios
    - Volatility metrics

    Tracks which values are estimated vs real data.
    Access via self.estimated_fields after calling get_all_data()
    """

    def __init__(self):
        """Initialize CBOE collector"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        # Track which fields contain estimated (not real) data
        self.estimated_fields = []

    def _mark_estimated(self, field_name: str, reason: str):
        """Mark a field as containing estimated data"""
        self.estimated_fields.append({
            'field': field_name,
            'reason': reason
        })
        logger.warning(f"ESTIMATED DATA: {field_name} - {reason}")
    
    def get_vix(self) -> Optional[float]:
        """
        Get current VIX spot price from Yahoo Finance
        
        Returns:
            VIX value or None
        """
        try:
            vix = yf.Ticker("^VIX")
            data = vix.history(period="1d")
            
            if not data.empty:
                vix_value = float(data['Close'].iloc[-1])
                logger.info(f"VIX: {vix_value:.2f}")
                return vix_value
            
            logger.warning("No VIX data available")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching VIX: {e}")
            return None
    
    def get_vix3m(self, track_estimation: bool = True) -> Optional[float]:
        """
        Get VIX3M (3-month VIX) for real term structure

        Args:
            track_estimation: Whether to track if this value is estimated

        Returns:
            VIX3M value or None
        """
        try:
            # Try multiple possible tickers for VIX3M
            tickers_to_try = ["^VIX3M"]  # VXV delisted

            for ticker_symbol in tickers_to_try:
                try:
                    vix3m = yf.Ticker(ticker_symbol)
                    data = vix3m.history(period="5d")  # Try 5 days to get recent data

                    if not data.empty:
                        vix3m_value = float(data['Close'].iloc[-1])
                        logger.info(f"VIX3M (using {ticker_symbol}): {vix3m_value:.2f}")
                        return vix3m_value
                except Exception as e:
                    logger.debug(f"Ticker {ticker_symbol} failed: {e}")
                    continue

            # If all tickers fail, estimate from VIX (typically VIX3M ~= VIX * 0.7-0.9)
            vix = self.get_vix()
            if vix:
                estimated_vix3m = vix * 0.85  # Conservative estimate
                if track_estimation:
                    self._mark_estimated('vix3m', 'Calculated as VIX × 0.85 (real VIX3M unavailable)')
                logger.warning(f"VIX3M unavailable, using estimate: {estimated_vix3m:.2f}")
                return estimated_vix3m

            logger.warning("No VIX3M data available from any source")
            return None

        except Exception as e:
            logger.error(f"Error fetching VIX3M: {e}")
            return None
    
    
    def get_vix9d(self) -> Optional[float]:
        """Get VIX9D (9-day implied volatility) from Yahoo Finance"""
        try:
            vix9d = yf.Ticker('^VIX9D')
            data = vix9d.history(period='5d')
            
            if not data.empty:
                value = float(data['Close'].iloc[-1])
                logger.info(f"VIX9D: {value:.2f}")
                return value
            else:
                logger.warning("VIX9D data unavailable")
                return None
        except Exception as e:
            logger.error(f"Error fetching VIX9D: {e}")
            return None
    
    def get_skew(self) -> Optional[float]:
        """Get CBOE SKEW Index (tail risk measure) from Yahoo Finance"""
        try:
            skew = yf.Ticker('^SKEW')
            data = skew.history(period='5d')
            
            if not data.empty:
                value = float(data['Close'].iloc[-1])
                logger.info(f"SKEW: {value:.2f}")
                return value
            else:
                logger.warning("SKEW data unavailable")
                return None
        except Exception as e:
            logger.error(f"Error fetching SKEW: {e}")
            return None
    

    def get_vvix(self) -> Optional[float]:
        """
        Get VVIX (VIX of VIX) - volatility of volatility index.

        VVIX measures expected volatility OF the VIX itself.

        Key levels:
        - < 80: Low vol-of-vol, complacent
        - 80-100: Normal range
        - 100-120: Elevated uncertainty
        - > 120: EXTREME - often marks capitulation/turning points
                 Strong buy signal for equities (mean reversion incoming)

        When VVIX spikes > 120, dealers are scrambling for gamma protection.
        The subsequent mean reversion creates powerful vanna/charm tailwinds.
        """
        try:
            vvix = yf.Ticker('^VVIX')
            data = vvix.history(period='5d')

            if not data.empty:
                value = float(data['Close'].iloc[-1])
                logger.info(f"VVIX: {value:.2f}")
                return value
            else:
                logger.warning("VVIX data unavailable")
                return None
        except Exception as e:
            logger.error(f"Error fetching VVIX: {e}")
            return None

    def get_vvix_history(self, days: int = 90) -> pd.DataFrame:
        """Get historical VVIX data"""
        try:
            vvix_ticker = yf.Ticker('^VVIX')
            data = vvix_ticker.history(period=f'{days}d')

            if not data.empty:
                df = pd.DataFrame({
                    'date': data.index,
                    'vvix': data['Close']
                })
                df['date'] = pd.to_datetime(df['date']).dt.date
                logger.info(f"Fetched {len(df)} days of VVIX history")
                return df
            else:
                logger.warning("No VVIX history available")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching VVIX history: {e}")
            return pd.DataFrame()

    def get_vvix_signal(self) -> dict:
        """
        Generate VVIX-based buy signal.

        Returns:
            dict with signal, level, description, and strength

        Signal Logic (from config/parameters.yaml):
        - VVIX >= 120: STRONG BUY - Historic turning point, mean reversion imminent
        - VVIX >= 110: BUY ALERT - Elevated, watching for spike
        - VVIX 80-110: NEUTRAL - Normal range
        - VVIX < 80: CAUTION - Complacency, potential for vol expansion
        """
        vvix = self.get_vvix()

        # Load thresholds from config
        vvix_cfg = cfg.volatility.vvix
        strong_buy_threshold = vvix_cfg.strong_buy_threshold
        buy_alert_threshold = vvix_cfg.buy_alert_threshold
        normal_min = vvix_cfg.normal_min
        colors = vvix_cfg.colors

        if vvix is None:
            return {
                'signal': 'UNAVAILABLE',
                'level': None,
                'strength': 0,
                'color': '#9E9E9E',
                'description': 'VVIX data unavailable'
            }

        if vvix >= strong_buy_threshold:
            return {
                'signal': 'STRONG BUY',
                'level': vvix,
                'strength': min(100, 70 + (vvix - strong_buy_threshold) * 2),
                'color': colors.strong_buy,
                'description': (
                    f'VVIX at {vvix:.1f} - EXTREME vol-of-vol! '
                    'Historic turning point. Dealers scrambling for gamma. '
                    'Mean reversion → vanna/charm tailwinds incoming.'
                )
            }
        elif vvix >= buy_alert_threshold:
            return {
                'signal': 'BUY ALERT',
                'level': vvix,
                'strength': 50 + (vvix - buy_alert_threshold) * 2,
                'color': colors.buy_alert,
                'description': (
                    f'VVIX at {vvix:.1f} - Elevated vol-of-vol. '
                    f'Watch for spike to {strong_buy_threshold}+ for strong buy signal.'
                )
            }
        elif vvix >= normal_min:
            return {
                'signal': 'NEUTRAL',
                'level': vvix,
                'strength': 30,
                'color': colors.neutral,
                'description': f'VVIX at {vvix:.1f} - Normal range. No signal.'
            }
        else:
            return {
                'signal': 'CAUTION',
                'level': vvix,
                'strength': 40,
                'color': colors.caution,
                'description': (
                    f'VVIX at {vvix:.1f} - Low vol-of-vol, complacency. '
                    'Potential for vol expansion.'
                )
            }

    def get_skew_history(self, days: int = 90) -> pd.DataFrame:
        """Get historical SKEW data"""
        try:
            skew_ticker = yf.Ticker('^SKEW')
            data = skew_ticker.history(period=f'{days}d')
            
            if not data.empty:
                df = pd.DataFrame({
                    'date': data.index,
                    'skew': data['Close']
                })
                df['date'] = pd.to_datetime(df['date']).dt.date
                logger.info(f"Fetched {len(df)} days of SKEW history")
                return df
            else:
                logger.warning("No SKEW history available")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching SKEW history: {e}")
            return pd.DataFrame()
    
    def get_vix9d_history(self, days: int = 90) -> pd.DataFrame:
        """Get historical VIX9D data"""
        try:
            vix9d_ticker = yf.Ticker('^VIX9D')
            data = vix9d_ticker.history(period=f'{days}d')
            
            if not data.empty:
                df = pd.DataFrame({
                    'date': data.index,
                    'vix9d': data['Close']
                })
                df['date'] = pd.to_datetime(df['date']).dt.date
                logger.info(f"Fetched {len(df)} days of VIX9D history")
                return df
            else:
                logger.warning("No VIX9D history available")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching VIX9D history: {e}")
            return pd.DataFrame()

    def get_real_contango(self) -> Optional[float]:
        """
        Calculate REAL VIX contango using VIX/VIX3M ratio

        Real contango should be single-digit percentages, not 65%!

        Returns:
            Contango percentage or None
        """
        try:
            vix = self.get_vix()
            # Don't double-track estimation - vix3m already tracks it
            vix3m = self.get_vix3m(track_estimation=False)

            if vix and vix3m:
                # Real contango: (VIX3M - VIX) / VIX * 100
                contango = ((vix3m - vix) / vix) * 100
                logger.info(f"Real VIX Contango: {contango:+.2f}%")
                return contango
            elif vix:
                # If VIX3M unavailable, estimate contango from VIX level
                # Typically: VIX > 20 → slight backwardation, VIX < 15 → contango
                if vix > 25:
                    estimated_contango = -2.0  # Backwardation when fear high
                elif vix > 20:
                    estimated_contango = 0.0  # Flat
                else:
                    estimated_contango = 3.0  # Contango when calm

                self._mark_estimated('vix_contango', f'VIX-based estimate (VIX={vix:.1f})')
                logger.warning(f"VIX3M unavailable, estimated contango: {estimated_contango:+.2f}%")
                return estimated_contango

            return None

        except Exception as e:
            logger.error(f"Error calculating contango: {e}")
            return None
    
    def get_equity_put_call_ratio(self) -> Optional[float]:
        """
        Get equity-only put/call ratio

        Note: Real CBOE data requires scraping or paid API.
        This uses a proxy calculation from options activity.

        Returns:
            Equity P/C ratio or None
        """
        try:
            # Try to get from CBOE website (may require parsing)
            # For now, use SPY options as equity proxy
            spy = yf.Ticker("SPY")

            # Get options chain
            try:
                options_dates = spy.options
                if options_dates:
                    # Get front month
                    front_month = options_dates[0]
                    opt_chain = spy.option_chain(front_month)

                    # Calculate put/call ratio from open interest
                    put_oi = opt_chain.puts['openInterest'].sum()
                    call_oi = opt_chain.calls['openInterest'].sum()

                    if call_oi > 0:
                        pc_ratio = put_oi / call_oi
                        logger.info(f"Equity P/C (SPY proxy): {pc_ratio:.3f}")
                        return pc_ratio
            except Exception as e:
                logger.debug(f"Options chain method failed: {e}")

            # Fallback: Use VIX/VIX3M as fear proxy
            vix = self.get_vix()
            vix3m = self.get_vix3m(track_estimation=False)

            if vix and vix3m:
                # When VIX > VIX3M, fear is high (P/C should be high)
                ratio = vix / vix3m
                # Map to typical P/C range (0.7-1.5)
                estimated_pc = 0.5 + (ratio * 0.7)
                self._mark_estimated('equity_put_call', 'Derived from VIX/VIX3M ratio (real P/C unavailable)')
                logger.info(f"Equity P/C (VIX/VIX3M proxy): {estimated_pc:.3f}")
                return estimated_pc

            return None

        except Exception as e:
            logger.error(f"Error getting equity P/C: {e}")
            return None
    
    def get_total_put_call_ratio(self) -> Optional[float]:
        """
        Get total market put/call ratio (all options)
        
        Returns:
            Total P/C ratio or None
        """
        try:
            # Try multiple market-wide proxies
            vix = self.get_vix()
            vix3m = self.get_vix3m()
            
            if vix and vix3m:
                # VIX/VIX3M ratio correlates with total P/C
                ratio = vix / vix3m
                # Map to typical total P/C range (0.8-1.3)
                estimated_pc = 0.6 + (ratio * 0.6)
                logger.info(f"Total P/C (estimated): {estimated_pc:.3f}")
                return estimated_pc
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting total P/C: {e}")
            return None
    
    def get_put_call_ratios(self) -> Optional[Dict]:
        """
        Get all put/call ratios

        Priority:
        1. CBOE official data (scraped)
        2. SPY options chain (from Yahoo Finance)
        3. VIX/VIX3M proxy estimate

        Returns:
            Dict with equity_pc, total_pc, source, and metadata
        """
        try:
            # Try to get CBOE official P/C first (if available)
            cboe_pc = self._scrape_cboe_put_call()

            if cboe_pc and cboe_pc.get('equity_pc'):
                result = {
                    'equity_pc': cboe_pc.get('equity_pc'),
                    'total_pc': cboe_pc.get('total_pc'),
                    'index_pc': cboe_pc.get('index_pc'),
                    'source': 'CBOE',
                    'is_estimated': False,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                # Fall back to computed values (SPY options or VIX proxy)
                equity_pc = self.get_equity_put_call_ratio()
                total_pc = self.get_total_put_call_ratio()

                # Determine source based on whether we used SPY options or VIX proxy
                # The get_equity_put_call_ratio tries SPY first, then falls back to VIX
                is_estimated = 'equity_put_call' in self._estimated_fields

                result = {
                    'equity_pc': equity_pc,
                    'total_pc': total_pc,
                    'index_pc': None,
                    'source': 'VIX_PROXY' if is_estimated else 'SPY_OPTIONS',
                    'is_estimated': is_estimated,
                    'timestamp': datetime.now().isoformat()
                }

            # Add sentiment interpretation
            pc = result['equity_pc'] or result['total_pc']
            if pc:
                result['sentiment'] = self._interpret_put_call(pc)

            return result

        except Exception as e:
            logger.error(f"Error getting P/C ratios: {e}")
            return None

    def _scrape_cboe_put_call(self) -> Optional[Dict]:
        """
        Attempt to scrape CBOE put/call ratio page

        CBOE publishes daily P/C ratios but may block scraping.
        Falls back gracefully if unavailable.

        Returns:
            Dict with equity_pc, index_pc, total_pc or None
        """
        try:
            # CBOE P/C data page
            url = "https://www.cboe.com/us/options/market_statistics/daily/"

            response = self.session.get(url, timeout=10)
            if response.status_code != 200:
                logger.debug(f"CBOE P/C page returned {response.status_code}")
                return None

            soup = BeautifulSoup(response.text, 'html.parser')

            # Look for P/C ratio data in page
            # Note: CBOE page structure changes - this may need updates
            tables = soup.find_all('table')

            for table in tables:
                text = table.get_text().lower()
                if 'put/call' in text or 'put call' in text:
                    # Found P/C table - parse it
                    rows = table.find_all('tr')
                    for row in rows:
                        cells = row.find_all(['td', 'th'])
                        if len(cells) >= 2:
                            label = cells[0].get_text().strip().lower()
                            try:
                                value = float(cells[1].get_text().strip())
                                if 'equity' in label:
                                    return {'equity_pc': value, 'source': 'CBOE'}
                                elif 'total' in label:
                                    return {'total_pc': value, 'source': 'CBOE'}
                            except ValueError:
                                continue

            return None

        except Exception as e:
            logger.debug(f"CBOE scrape failed (expected): {e}")
            return None

    def _interpret_put_call(self, pc_ratio: float) -> Dict:
        """
        Interpret put/call ratio sentiment

        P/C > 1.0: More puts than calls = bearish sentiment
        P/C < 1.0: More calls than puts = bullish sentiment

        Contrarian view: Extreme readings often mark reversals
        """
        if pc_ratio > 1.3:
            return {
                'reading': 'Extreme Fear',
                'signal': 'CONTRARIAN BUY',
                'color': '#4CAF50',
                'description': 'Heavy put buying - potential bottom forming'
            }
        elif pc_ratio > 1.1:
            return {
                'reading': 'Fearful',
                'signal': 'BEARISH',
                'color': '#FF9800',
                'description': 'Elevated hedging activity'
            }
        elif pc_ratio > 0.9:
            return {
                'reading': 'Neutral',
                'signal': 'NEUTRAL',
                'color': '#9E9E9E',
                'description': 'Balanced options activity'
            }
        elif pc_ratio > 0.7:
            return {
                'reading': 'Complacent',
                'signal': 'BULLISH',
                'color': '#8BC34A',
                'description': 'Call buying dominates - bullish sentiment'
            }
        else:
            return {
                'reading': 'Extreme Greed',
                'signal': 'CONTRARIAN SELL',
                'color': '#F44336',
                'description': 'Heavy call buying - potential top forming'
            }

    def get_multi_ticker_put_call(self, tickers: list = None) -> Dict:
        """
        Get put/call ratios for multiple tickers

        Useful for comparing P/C across QQQ, IWM, SPY
        to identify sector-specific sentiment

        Args:
            tickers: List of tickers to analyze (default: major ETFs)

        Returns:
            Dict with P/C ratios per ticker
        """
        if tickers is None:
            tickers = ['SPY', 'QQQ', 'IWM']

        results = {}

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                options_dates = stock.options

                if not options_dates:
                    continue

                # Get front month options
                front_month = options_dates[0]
                chain = stock.option_chain(front_month)

                put_oi = chain.puts['openInterest'].sum()
                call_oi = chain.calls['openInterest'].sum()

                if call_oi > 0:
                    pc_ratio = put_oi / call_oi
                    results[ticker] = {
                        'put_call_ratio': round(pc_ratio, 3),
                        'put_oi': int(put_oi),
                        'call_oi': int(call_oi),
                        'expiry': front_month,
                        'sentiment': self._interpret_put_call(pc_ratio)
                    }

            except Exception as e:
                logger.debug(f"Could not get P/C for {ticker}: {e}")
                continue

        return results
    
    def get_all_data(self) -> Dict:
        """
        Get all CBOE data in one call

        Returns:
            Dict with all available CBOE data, including metadata about estimated values
        """
        logger.info("Fetching CBOE data...")

        # Reset estimated fields tracking for this fetch
        self.estimated_fields = []

        vix_spot = self.get_vix()
        vix9d = self.get_vix9d()
        vix3m = self.get_vix3m()
        vvix = self.get_vvix()
        skew = self.get_skew()
        vix_contango = self.get_real_contango()
        put_call_ratios = self.get_put_call_ratios()
        vvix_signal = self.get_vvix_signal()

        result = {
            'vix_spot': vix_spot,
            'vix9d': vix9d,
            'vix3m': vix3m,
            'vvix': vvix,
            'vvix_signal': vvix_signal,
            'skew': skew,
            'vix_contango': vix_contango,
            'put_call_ratios': put_call_ratios or {},
            'timestamp': pd.Timestamp.now().isoformat(),
            # Include metadata about which fields are estimated (not real data)
            'estimated_fields': self.estimated_fields.copy(),
            'has_estimated_data': len(self.estimated_fields) > 0
        }

        if self.estimated_fields:
            logger.warning(f"CBOE data contains {len(self.estimated_fields)} estimated field(s): "
                          f"{[e['field'] for e in self.estimated_fields]}")

        return result
