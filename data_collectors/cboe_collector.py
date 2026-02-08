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

            # If all tickers fail, estimate from VIX using regime-aware approach
            # Historical VIX/VIX3M relationship varies by market regime:
            #
            # Calm markets (VIX < 15):   VIX3M typically 5-10% HIGHER (contango)
            # Normal (VIX 15-20):        VIX3M typically 0-5% higher (mild contango)
            # Elevated (VIX 20-30):      VIX3M roughly equal to VIX (flat)
            # Crisis (VIX > 30):         VIX3M typically 5-15% LOWER (backwardation)
            # Panic (VIX > 50):          VIX3M can be 15-25% LOWER (steep backwardation)
            vix = self.get_vix()
            if vix:
                if vix < 15:
                    # Calm market: contango, VIX3M > VIX
                    multiplier = 1.07
                    regime = "calm (contango)"
                elif vix < 20:
                    # Normal: mild contango
                    multiplier = 1.03
                    regime = "normal (mild contango)"
                elif vix < 30:
                    # Elevated: roughly flat
                    multiplier = 0.98
                    regime = "elevated (flat)"
                elif vix < 50:
                    # Crisis: backwardation, VIX3M < VIX
                    multiplier = 0.90
                    regime = "crisis (backwardation)"
                else:
                    # Panic: steep backwardation
                    multiplier = 0.80
                    regime = "panic (steep backwardation)"

                estimated_vix3m = vix * multiplier
                if track_estimation:
                    self._mark_estimated(
                        'vix3m',
                        f'Estimated VIX × {multiplier:.2f} for {regime} regime (VIX={vix:.1f})'
                    )
                logger.warning(
                    f"VIX3M unavailable, using regime-aware estimate: {estimated_vix3m:.2f} "
                    f"(VIX={vix:.1f}, regime={regime})"
                )
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

        This is the CORRECT methodology for VIX term structure:
        - Uses actual VIX indices, not ETN prices
        - Typical range: -15% (backwardation) to +10% (contango)
        - NOT the 65%+ fake values from ETN price ratios

        The previous ETN-based calculation (VIXY/VXZ) was fundamentally
        flawed because ETN prices are eroded by contango drag and fees.

        Returns:
            Contango percentage or None
        """
        try:
            vix = self.get_vix()
            # Don't double-track estimation - vix3m already tracks it
            vix3m = self.get_vix3m(track_estimation=False)

            if vix and vix3m:
                # Real contango: (VIX3M - VIX) / VIX * 100
                # Positive = contango (normal, bullish)
                # Negative = backwardation (fear, bearish)
                contango = ((vix3m - vix) / vix) * 100
                logger.info(f"Real VIX Contango: {contango:+.2f}% (VIX={vix:.2f}, VIX3M={vix3m:.2f})")
                return round(contango, 2)
            elif vix:
                # If VIX3M unavailable but VIX is, estimate based on VIX regime
                # Historical relationship based on empirical data:
                if vix < 15:
                    # Calm market: strong contango
                    estimated_contango = 7.0
                    regime = "calm"
                elif vix < 20:
                    # Normal: mild contango
                    estimated_contango = 3.0
                    regime = "normal"
                elif vix < 25:
                    # Elevated: roughly flat
                    estimated_contango = -2.0
                    regime = "elevated"
                elif vix < 35:
                    # Crisis: backwardation
                    estimated_contango = -8.0
                    regime = "crisis"
                else:
                    # Panic: steep backwardation
                    estimated_contango = -15.0
                    regime = "panic"

                self._mark_estimated(
                    'vix_contango',
                    f'VIX-regime estimate for {regime} market (VIX={vix:.1f})'
                )
                logger.warning(
                    f"VIX3M unavailable, estimated contango: {estimated_contango:+.2f}% "
                    f"(VIX={vix:.1f}, regime={regime})"
                )
                return estimated_contango

            return None

        except Exception as e:
            logger.error(f"Error calculating contango: {e}")
            return None
    
    def get_cboe_equity_put_call(self) -> Optional[float]:
        """
        Get official CBOE Equity Put/Call Ratio (PCCE).

        This is the actual CBOE-published ratio covering ALL equity options
        (options on individual stocks like AAPL, TSLA, NVDA, etc.)

        IMPORTANT: CBOE moved their P/C ratio data to paid DataShop in 2019.
        Free CSV files at cdn.cboe.com only contain data through 2019.

        Data sources tried:
        1. CBOE daily statistics page scrape (usually blocked)
        2. Yahoo Finance ^PCCE ticker (does not exist)

        For reliable CBOE PCCE data, users should:
        - Enter manual PCCE from trading platforms (ThinkorSwim, TradingView)
        - Or subscribe to CBOE DataShop (paid)

        Returns:
            CBOE Equity P/C ratio or None
        """
        try:
            # Try CBOE website scrape (usually returns None - page uses JS)
            cboe_data = self._scrape_cboe_put_call()
            if cboe_data and cboe_data.get('equity_pc'):
                value = cboe_data['equity_pc']
                logger.info(f"CBOE Equity P/C (scraped): {value:.3f}")
                return value

            # Note: CBOE P/C indices not available on free sources
            # - ^PCCE, ^CPCE do not exist on Yahoo Finance
            # - cdn.cboe.com CSVs ended in 2019
            # - Current data requires CBOE DataShop subscription
            logger.debug("CBOE PCCE not available via free sources - use manual input")
            return None

        except Exception as e:
            logger.debug(f"CBOE PCCE fetch failed: {e}")
            return None

    def get_spy_put_call_ratio(self) -> Optional[Dict]:
        """
        Get SPY-specific put/call ratios from options chain.

        Provides BOTH:
        - Volume-based P/C (daily sentiment, comparable to CBOE methodology)
        - Open Interest P/C (positioning/inventory)

        SPY is NOT the same as CBOE Equity P/C (which covers all stocks),
        but it's the best free proxy for market-wide options sentiment.

        Returns:
            Dict with volume_ratio, oi_ratio, put/call volumes and OI
        """
        try:
            spy = yf.Ticker("SPY")
            options_dates = spy.options

            if not options_dates:
                logger.warning("No SPY options dates available")
                return None

            # Get front month options
            front_month = options_dates[0]
            opt_chain = spy.option_chain(front_month)

            # Calculate from VOLUME (matches CBOE methodology)
            put_vol = int(opt_chain.puts['volume'].fillna(0).sum())
            call_vol = int(opt_chain.calls['volume'].fillna(0).sum())

            # Calculate from OPEN INTEREST (positioning/inventory)
            put_oi = int(opt_chain.puts['openInterest'].fillna(0).sum())
            call_oi = int(opt_chain.calls['openInterest'].fillna(0).sum())

            result = {
                'expiry': front_month,
                'source': 'SPY_OPTIONS'
            }

            # Volume-based ratio (primary - matches CBOE methodology)
            if call_vol > 0:
                vol_ratio = put_vol / call_vol
                result['volume_ratio'] = vol_ratio
                result['put_volume'] = put_vol
                result['call_volume'] = call_vol
                logger.info(f"SPY P/C (Volume): {vol_ratio:.3f} (Puts: {put_vol:,}, Calls: {call_vol:,})")
            else:
                result['volume_ratio'] = None

            # OI-based ratio (secondary - positioning view)
            if call_oi > 0:
                oi_ratio = put_oi / call_oi
                result['oi_ratio'] = oi_ratio
                result['put_oi'] = put_oi
                result['call_oi'] = call_oi
                logger.info(f"SPY P/C (OI): {oi_ratio:.3f} (Puts: {put_oi:,}, Calls: {call_oi:,})")
            else:
                result['oi_ratio'] = None

            # Set 'ratio' to volume-based (CBOE-comparable) or fallback to OI
            result['ratio'] = result.get('volume_ratio') or result.get('oi_ratio')

            return result if result.get('ratio') else None

        except Exception as e:
            logger.debug(f"SPY options chain failed: {e}")
            return None

    def get_equity_put_call_ratio(self) -> Optional[float]:
        """
        Get equity put/call ratio with fallback chain.

        Priority:
        1. Official CBOE PCCE (^PCCE ticker)
        2. SPY options OI as proxy
        3. VIX/VIX3M estimation (last resort)

        Returns:
            Equity P/C ratio or None
        """
        # Try official CBOE PCCE first
        pcce = self.get_cboe_equity_put_call()
        if pcce is not None:
            return pcce

        # Fall back to SPY options
        spy_data = self.get_spy_put_call_ratio()
        if spy_data and spy_data.get('ratio'):
            self._mark_estimated('equity_put_call', 'Using SPY OI as proxy (CBOE PCCE unavailable)')
            return spy_data['ratio']

        # Last resort: VIX proxy
        try:
            vix = self.get_vix()
            vix3m = self.get_vix3m(track_estimation=False)

            if vix and vix3m:
                ratio = vix / vix3m
                estimated_pc = 0.5 + (ratio * 0.7)
                self._mark_estimated('equity_put_call', 'Derived from VIX/VIX3M ratio (real P/C unavailable)')
                logger.info(f"Equity P/C (VIX/VIX3M proxy): {estimated_pc:.3f}")
                return estimated_pc
        except Exception as e:
            logger.debug(f"VIX proxy calculation failed: {e}")

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
        Get all put/call ratios with proper sourcing.

        Returns dict with:
        - cboe_equity_pc: Official CBOE PCCE (all equity options)
        - spy_pc: SPY-specific put/call (institutional hedging gauge)
        - total_pc: Estimated total market P/C
        - equity_pc: Best available equity P/C (CBOE > SPY > VIX proxy)
        - source: Where equity_pc came from

        This provides both the official macro gauge (CBOE PCCE) and
        a tradable micro gauge (SPY positioning).
        """
        try:
            result = {
                'cboe_equity_pc': None,
                # SPY metrics (volume = daily sentiment, OI = positioning)
                'spy_pc': None,           # Primary: volume-based (CBOE-comparable)
                'spy_pc_oi': None,        # Secondary: OI-based (positioning)
                'spy_put_volume': None,
                'spy_call_volume': None,
                'spy_put_oi': None,
                'spy_call_oi': None,
                # Aggregates
                'total_pc': None,
                'equity_pc': None,  # Best available
                'index_pc': None,
                'source': None,
                'is_estimated': False,
                'timestamp': datetime.now().isoformat()
            }

            # 1. Get official CBOE PCCE (best source for macro sentiment)
            # Note: Rarely available - CBOE moved data to paid DataShop
            cboe_pcce = self.get_cboe_equity_put_call()
            if cboe_pcce is not None:
                result['cboe_equity_pc'] = cboe_pcce
                result['equity_pc'] = cboe_pcce
                result['source'] = 'CBOE_PCCE'
                logger.info(f"Using official CBOE PCCE: {cboe_pcce:.3f}")

            # 2. Get SPY-specific P/C (primary free data source)
            spy_data = self.get_spy_put_call_ratio()
            if spy_data:
                # Volume-based ratio (matches CBOE methodology)
                result['spy_pc'] = spy_data.get('volume_ratio') or spy_data.get('ratio')
                result['spy_put_volume'] = spy_data.get('put_volume')
                result['spy_call_volume'] = spy_data.get('call_volume')

                # OI-based ratio (positioning view)
                result['spy_pc_oi'] = spy_data.get('oi_ratio')
                result['spy_put_oi'] = spy_data.get('put_oi')
                result['spy_call_oi'] = spy_data.get('call_oi')

                # If no CBOE data, use SPY volume as fallback for equity_pc
                if result['equity_pc'] is None and result['spy_pc'] is not None:
                    result['equity_pc'] = result['spy_pc']
                    result['source'] = 'SPY_VOLUME'  # Explicitly note it's volume-based
                    result['is_estimated'] = True
                    self._mark_estimated('equity_put_call', 'Using SPY Volume (CBOE PCCE unavailable)')

            # 3. Get total P/C estimate
            result['total_pc'] = self.get_total_put_call_ratio()

            # 4. Last resort: VIX proxy if nothing else available
            if result['equity_pc'] is None:
                vix = self.get_vix()
                vix3m = self.get_vix3m(track_estimation=False)
                if vix and vix3m:
                    ratio = vix / vix3m
                    estimated_pc = 0.5 + (ratio * 0.7)
                    result['equity_pc'] = estimated_pc
                    result['source'] = 'VIX_PROXY'
                    result['is_estimated'] = True
                    self._mark_estimated('equity_put_call', 'Derived from VIX/VIX3M ratio')

            # Add sentiment interpretation based on best available
            pc = result['cboe_equity_pc'] or result['equity_pc']
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
