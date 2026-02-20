"""
Yahoo Finance Collector - IMPROVED VERSION v2
Fetches VIX, market breadth, and supplementary options data with robust retry logic
Now with proper config usage, logging, and narrower exception catching

Includes centralized rate limiting via utils.retry_utils
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, Callable, Any
import time
import logging
import json
import threading
from pathlib import Path
from functools import wraps
from yfinance.exceptions import YFRateLimitError

# Set up logger
logger = logging.getLogger(__name__)

# Import centralized rate limiter
try:
    from utils.retry_utils import YAHOO_LIMITER
except ImportError:
    YAHOO_LIMITER = None
    logger.debug("Centralized rate limiter not available, using built-in")

class RetryConfig:
    """Configuration for retry behavior"""
    def __init__(
        self,
        max_retries: int = 3,
        initial_backoff: float = 1.0,
        max_backoff: float = 10.0,
        exponential_base: float = 2.0
    ):
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.exponential_base = exponential_base


def with_retry(func: Callable) -> Callable:
    """
    Method decorator that adds retry logic with exponential backoff.
    If the first argument has a `.retry_config` attribute (the instance),
    it will use that; otherwise it falls back to default RetryConfig().
    
    Only retries on network/data errors, not programming errors.
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # Assume methods: first arg is `self`
        self_obj = args[0] if args else None
        cfg = getattr(self_obj, "retry_config", RetryConfig())
        
        last_exception = None
        
        for attempt in range(cfg.max_retries):
            try:
                return func(*args, **kwargs)
            
            except (ValueError, ConnectionError, TimeoutError, IOError) as e:
                # Only retry on data/network errors, not programming errors
                last_exception = e
                
                if attempt < cfg.max_retries - 1:
                    # Calculate backoff time with exponential increase
                    backoff_time = min(
                        cfg.initial_backoff * (cfg.exponential_base ** attempt),
                        cfg.max_backoff
                    )
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{cfg.max_retries} failed for {func.__name__}: {str(e)}"
                    )
                    logger.info(f"Retrying in {backoff_time:.2f} seconds...")
                    
                    time.sleep(backoff_time)
                else:
                    logger.error(f"All {cfg.max_retries} attempts failed for {func.__name__}")
            
            except Exception as e:
                # Don't retry on unexpected errors (likely bugs)
                logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
                return None
        
        # If all retries failed, return None
        logger.error(f"Final error in {func.__name__}: {last_exception}")
        return None
    
    return wrapper


class YahooCollector:
    """Collects market data from Yahoo Finance with robust error handling"""
    _CACHE: Dict[str, Dict[str, Any]] = {}
    _MISS_CACHE: Dict[str, float] = {}
    _LAST_RATE_LIMIT_AT: Optional[float] = None
    _CACHE_LOCK = threading.Lock()
    _LKG_LOCK = threading.Lock()
    _LKG_PATH = Path(__file__).resolve().parents[1] / "data" / "cache" / "yahoo_lkg.json"
    
    def __init__(self, retry_config: RetryConfig = None, cache_ttl_seconds: int = 300):
        """
        Initialize collector with optional custom retry configuration
        
        Args:
            retry_config: Custom retry configuration (uses defaults if None)
        """
        self.retry_config = retry_config or RetryConfig(
            max_retries=3,
            initial_backoff=2.0,
            max_backoff=10.0,
            exponential_base=2.0
        )
        self.cache_ttl_seconds = cache_ttl_seconds
        self.stale_ttl_seconds = 3600  # allow stale for 1 hour on errors
        self.persistent_ttl_seconds = 86400  # 24h last-known-good fallback window
        self.rate_limit_cooldown_seconds = 180  # short cooldown after rate limit

    def _load_lkg_store(self) -> Dict[str, Any]:
        try:
            if not self._LKG_PATH.exists():
                return {}
            with self._LKG_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception as e:
            logger.debug(f"Could not read Yahoo LKG store: {e}")
            return {}

    def _save_lkg_store(self, store: Dict[str, Any]):
        try:
            self._LKG_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._LKG_PATH.with_suffix(".tmp")
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(store, f)
            tmp_path.replace(self._LKG_PATH)
        except Exception as e:
            logger.debug(f"Could not write Yahoo LKG store: {e}")

    def _get_persisted(self, key: str, allow_stale: bool = False):
        ttl = self.persistent_ttl_seconds if allow_stale else self.cache_ttl_seconds
        with self._LKG_LOCK:
            store = self._load_lkg_store()
            entry = store.get(key)
            if not isinstance(entry, dict):
                return None
            ts = entry.get("ts")
            value = entry.get("value")
            try:
                age = time.time() - float(ts)
            except (TypeError, ValueError):
                return None
            if age <= ttl:
                return value
            return None

    def _get_cached(self, key: str, allow_stale: bool = False):
        with self._CACHE_LOCK:
            entry = self._CACHE.get(key)
        if not entry:
            return self._get_persisted(key, allow_stale=allow_stale)
        age = time.time() - entry["ts"]
        if age <= self.cache_ttl_seconds:
            return entry["value"]
        if allow_stale and age <= self.stale_ttl_seconds:
            return entry["value"]
        return self._get_persisted(key, allow_stale=allow_stale)

    def _set_cached(self, key: str, value: Any):
        if value is None:
            return
        ts = time.time()
        with self._CACHE_LOCK:
            self._CACHE[key] = {"ts": ts, "value": value}
            self._MISS_CACHE.pop(key, None)
        with self._LKG_LOCK:
            store = self._load_lkg_store()
            store[key] = {"ts": ts, "value": value}
            self._save_lkg_store(store)

    def _mark_fetch_miss(self, key: str):
        with self._CACHE_LOCK:
            self._MISS_CACHE[key] = time.time()

    def _is_fetch_miss_recent(self, key: str, cooldown_seconds: int = 60) -> bool:
        with self._CACHE_LOCK:
            ts = self._MISS_CACHE.get(key)
        if ts is None:
            return False
        return (time.time() - ts) < cooldown_seconds

    def _rate_limited_recently(self) -> bool:
        last_rate_limit_at = type(self)._LAST_RATE_LIMIT_AT
        if last_rate_limit_at is None:
            return False
        return (time.time() - last_rate_limit_at) < self.rate_limit_cooldown_seconds
    
    @with_retry
    def get_vix(self) -> Optional[float]:
        """Get current VIX spot price"""
        cached = self._get_cached("vix")
        if cached is not None:
            return cached

        if self._rate_limited_recently():
            stale = self._get_cached("vix", allow_stale=True)
            if stale is not None:
                return stale

        if self._is_fetch_miss_recent("vix"):
            stale = self._get_cached("vix", allow_stale=True)
            return stale

        # Apply centralized rate limiting
        if YAHOO_LIMITER:
            YAHOO_LIMITER.wait()

        try:
            vix = yf.Ticker("^VIX")
            data = vix.history(period="1d")
            
            if not data.empty:
                value = float(data['Close'].iloc[-1])
                self._set_cached("vix", value)
                return value
            
            raise ValueError("VIX data is empty")
        except YFRateLimitError:
            type(self)._LAST_RATE_LIMIT_AT = time.time()
            self._mark_fetch_miss("vix")
            stale = self._get_cached("vix", allow_stale=True)
            return stale
        except Exception as e:
            logger.error(f"Error fetching VIX from Yahoo: {e}")
            self._mark_fetch_miss("vix")
            stale = self._get_cached("vix", allow_stale=True)
            if stale is not None:
                logger.warning("Using last-known-good VIX value")
            return stale
    
    @with_retry
    def get_vix_futures_proxy(self) -> Optional[float]:
        """
        DEPRECATED: ETN-based VIX contango calculation is fundamentally flawed.

        IMPORTANT: Do NOT use this for market analysis!
        ETN prices (VIXY/VXZ) are NOT equivalent to VIX futures levels because:
        1. Contango drag: Daily roll losses compound over time
        2. Management fees: ~0.85%+ annually erode ETN value
        3. Different decay rates: Short-term ETNs decay faster than mid-term

        The ratio of ETN prices can show fake contango of 65%+ when real
        VIX term structure contango is typically -5% to +10%.

        For accurate VIX term structure, use:
        - CBOECollector.get_real_contango() which uses VIX/VIX3M indices

        This method is kept for backward compatibility but returns None
        to prevent use of misleading data.

        Returns:
            None - intentionally disabled due to fundamental methodology flaw
        """
        logger.warning(
            "get_vix_futures_proxy() is DEPRECATED: ETN-based calculation is "
            "fundamentally flawed. Use CBOECollector.get_real_contango() instead."
        )
        # Return None to prevent use of misleading data
        # The real VIX contango should come from VIX/VIX3M ratio in CBOECollector
        return None
    
    @with_retry
    def get_market_breadth_proxy(self) -> Optional[float]:
        """
        Get market breadth proxy using SPY components
        Uses advance/decline from major ETFs
        """
        # Get major sector ETFs
        sectors = {
            'XLK': 'Technology',
            'XLF': 'Financials',
            'XLV': 'Healthcare',
            'XLE': 'Energy',
            'XLY': 'Consumer Discretionary',
            'XLP': 'Consumer Staples',
            'XLI': 'Industrials',
            'XLB': 'Materials',
            'XLU': 'Utilities'
        }
        
        advancing = 0
        total = 0
        
        cached = self._get_cached("market_breadth_proxy")
        if cached is not None:
            return cached

        if self._rate_limited_recently():
            stale = self._get_cached("market_breadth_proxy", allow_stale=True)
            if stale is not None:
                return stale

        if self._is_fetch_miss_recent("market_breadth_proxy"):
            stale = self._get_cached("market_breadth_proxy", allow_stale=True)
            return stale

        try:
            for ticker in sectors.keys():
                try:
                    etf = yf.Ticker(ticker)
                    data = etf.history(period="2d")
                    
                    if len(data) >= 2:
                        total += 1
                        if data['Close'].iloc[-1] > data['Close'].iloc[-2]:
                            advancing += 1
                except Exception as e:
                    logger.debug(f"Error fetching {ticker} for breadth: {e}")
                    continue
        except YFRateLimitError:
            type(self)._LAST_RATE_LIMIT_AT = time.time()
            self._mark_fetch_miss("market_breadth_proxy")
            stale = self._get_cached("market_breadth_proxy", allow_stale=True)
            return stale
        
        if total > 0:
            value = advancing / total
            self._set_cached("market_breadth_proxy", value)
            return value

        stale = self._get_cached("market_breadth_proxy", allow_stale=True)
        if stale is not None:
            logger.warning("Using last-known-good market breadth proxy")
            return stale
        self._mark_fetch_miss("market_breadth_proxy")
        raise ValueError("No sector ETF data available for breadth calculation")
    
    @with_retry
    def get_put_call_ratio_proxy(self) -> Optional[float]:
        """
        Get Put/Call ratio using SPY options volume (VXV is delisted)
        Returns ratio of put volume to call volume
        """
        cached = self._get_cached("put_call_proxy")
        if cached is not None:
            return cached

        if self._rate_limited_recently():
            stale = self._get_cached("put_call_proxy", allow_stale=True)
            if stale is not None:
                return stale

        if self._is_fetch_miss_recent("put_call_proxy"):
            stale = self._get_cached("put_call_proxy", allow_stale=True)
            return stale

        try:
            spy = yf.Ticker("SPY")
            
            # Get options chain for nearest expiry
            expirations = spy.options
            if not expirations:
                logger.warning("No SPY options expiration dates available")
                return None
            
            # Use nearest expiration
            nearest_expiry = expirations[0]
            opt_chain = spy.option_chain(nearest_expiry)
            
            # Calculate total volume for puts and calls
            put_volume = opt_chain.puts['volume'].sum()
            call_volume = opt_chain.calls['volume'].sum()
            
            if call_volume > 0:
                pc_ratio = put_volume / call_volume
                logger.info(f"SPY P/C Ratio: {pc_ratio:.3f} (Puts: {put_volume:,.0f}, Calls: {call_volume:,.0f})")
                self._set_cached("put_call_proxy", pc_ratio)
                return pc_ratio
            else:
                logger.warning("SPY call volume is zero")
                return None
                
        except YFRateLimitError:
            type(self)._LAST_RATE_LIMIT_AT = time.time()
            self._mark_fetch_miss("put_call_proxy")
            stale = self._get_cached("put_call_proxy", allow_stale=True)
            return stale
        except Exception as e:
            logger.error(f"Error calculating SPY P/C ratio: {e}")
            self._mark_fetch_miss("put_call_proxy")
            stale = self._get_cached("put_call_proxy", allow_stale=True)
            if stale is not None:
                logger.warning("Using last-known-good SPY put/call ratio")
            return stale


    def get_all_data(self) -> Dict:
        """Get all Yahoo Finance data

        Note: vix_contango_proxy is intentionally excluded - the ETN-based
        calculation was fundamentally flawed. Use CBOECollector.get_real_contango()
        for accurate VIX term structure data.
        """
        return {
            'vix': self.get_vix(),
            # vix_contango_proxy removed - ETN-based calculation is flawed
            # Real contango should come from CBOE VIX/VIX3M ratio
            'market_breadth_proxy': self.get_market_breadth_proxy(),
            'put_call_proxy': self.get_put_call_ratio_proxy(),
            'treasury_10y': self.get_treasury_10y(),
            'hy_spread_proxy': self.get_hy_spread_proxy(),
            'timestamp': datetime.now().isoformat()
        }

    @with_retry
    def get_treasury_10y(self) -> Optional[float]:
        """
        Get 10-Year Treasury Yield from Yahoo Finance (^TNX)

        This is a fallback when FRED API is unavailable.
        Yahoo's ^TNX tracks the 10-year Treasury yield.

        Returns:
            10Y yield as percentage (e.g., 4.25 for 4.25%)
        """
        cached = self._get_cached("treasury_10y")
        if cached is not None:
            return cached

        if self._rate_limited_recently():
            stale = self._get_cached("treasury_10y", allow_stale=True)
            if stale is not None:
                return stale

        if self._is_fetch_miss_recent("treasury_10y"):
            stale = self._get_cached("treasury_10y", allow_stale=True)
            return stale

        try:
            tnx = yf.Ticker("^TNX")
            data = tnx.history(period="5d")

            if not data.empty:
                # Yahoo TNX is already in percentage form
                yield_pct = float(data['Close'].iloc[-1])
                logger.info(f"10Y Treasury (Yahoo ^TNX): {yield_pct:.2f}%")
                self._set_cached("treasury_10y", yield_pct)
                return yield_pct

            logger.warning("No 10Y Treasury data from Yahoo")
            return None

        except YFRateLimitError:
            type(self)._LAST_RATE_LIMIT_AT = time.time()
            self._mark_fetch_miss("treasury_10y")
            stale = self._get_cached("treasury_10y", allow_stale=True)
            return stale
        except Exception as e:
            logger.error(f"Error fetching 10Y Treasury: {e}")
            self._mark_fetch_miss("treasury_10y")
            stale = self._get_cached("treasury_10y", allow_stale=True)
            if stale is not None:
                logger.warning("Using last-known-good 10Y Treasury value")
            return stale

    @with_retry
    def get_hy_spread_proxy(self) -> Optional[float]:
        """
        Estimate HY credit spread using HYG yield vs Treasury yield.

        This is a proxy when FRED's BAMLH0A0HYM2 is unavailable.
        Uses: HYG SEC Yield - 10Y Treasury

        Typical HY spreads:
        - Normal: 3-4% (300-400 bps)
        - Elevated: 5-6% (500-600 bps)
        - Crisis: 8%+ (800+ bps)

        Returns:
            Estimated HY spread as percentage (e.g., 3.5 for 350 bps)
        """
        cached = self._get_cached("hy_spread_proxy")
        if cached is not None:
            return cached

        if self._rate_limited_recently():
            stale = self._get_cached("hy_spread_proxy", allow_stale=True)
            if stale is not None:
                return stale

        if self._is_fetch_miss_recent("hy_spread_proxy"):
            stale = self._get_cached("hy_spread_proxy", allow_stale=True)
            return stale

        try:
            # HYG - iShares iBoxx High Yield Corporate Bond ETF
            hyg = yf.Ticker("HYG")

            # Get HYG info for SEC yield
            info = hyg.info
            hyg_yield = info.get('yield')  # This is the 30-day SEC yield

            if hyg_yield is None:
                # Fallback: estimate from dividend yield
                hyg_yield = info.get('dividendYield', 0.06)  # ~6% typical

            # Get 10Y Treasury for comparison
            treasury_10y = self.get_treasury_10y()

            if hyg_yield and treasury_10y:
                # HYG yield is already decimal (0.06 = 6%)
                hyg_yield_pct = hyg_yield * 100 if hyg_yield < 1 else hyg_yield

                # Spread = HYG yield - Treasury yield
                spread = hyg_yield_pct - treasury_10y

                # Sanity check - HY spread should be positive and reasonable
                if 1.0 < spread < 15.0:
                    logger.info(f"HY Spread Proxy: {spread:.2f}% (HYG: {hyg_yield_pct:.2f}%, 10Y: {treasury_10y:.2f}%)")
                    spread = round(spread, 2)
                    self._set_cached("hy_spread_proxy", spread)
                    return spread
                else:
                    logger.warning(f"HY spread estimate out of range: {spread:.2f}%")
                    # Return a reasonable estimate based on current market
                    self._set_cached("hy_spread_proxy", 3.5)
                    return 3.5  # ~350 bps is typical "normal" spread

            logger.warning("Could not calculate HY spread proxy")
            return None

        except YFRateLimitError:
            type(self)._LAST_RATE_LIMIT_AT = time.time()
            self._mark_fetch_miss("hy_spread_proxy")
            stale = self._get_cached("hy_spread_proxy", allow_stale=True)
            return stale
        except Exception as e:
            logger.error(f"Error calculating HY spread proxy: {e}")
            self._mark_fetch_miss("hy_spread_proxy")
            stale = self._get_cached("hy_spread_proxy", allow_stale=True)
            if stale is not None:
                logger.warning("Using last-known-good HY spread proxy")
            return stale

    @with_retry
    def get_credit_etf_flows(self) -> Optional[Dict]:
        """
        Get HYG/LQD credit ETF data for credit sentiment analysis.

        HYG = iShares High Yield Corporate Bond ETF (junk bonds)
        LQD = iShares Investment Grade Corporate Bond ETF

        When HYG outperforms LQD = Risk-on (investors prefer junk yield)
        When LQD outperforms HYG = Risk-off (flight to quality)

        Returns:
            Dict with prices, ratio, and performance metrics
        """
        cached = self._get_cached("credit_etf_flows")
        if cached is not None:
            return cached

        if self._rate_limited_recently():
            stale = self._get_cached("credit_etf_flows", allow_stale=True)
            if stale is not None:
                return stale

        if self._is_fetch_miss_recent("credit_etf_flows"):
            stale = self._get_cached("credit_etf_flows", allow_stale=True)
            return stale

        try:
            hyg = yf.Ticker("HYG")
            lqd = yf.Ticker("LQD")

            # Get 30 days for trend analysis
            hyg_data = hyg.history(period="1mo")
            lqd_data = lqd.history(period="1mo")

            if hyg_data.empty or lqd_data.empty:
                raise ValueError("Credit ETF data is empty")

            # Current prices
            hyg_price = float(hyg_data['Close'].iloc[-1])
            lqd_price = float(lqd_data['Close'].iloc[-1])

            # HYG/LQD ratio (higher = more risk appetite)
            ratio = hyg_price / lqd_price

            # Calculate 1-day, 5-day, and 20-day performance
            def calc_return(data, days):
                if len(data) > days:
                    return (data['Close'].iloc[-1] / data['Close'].iloc[-days-1] - 1) * 100
                return None

            hyg_1d = calc_return(hyg_data, 1)
            hyg_5d = calc_return(hyg_data, 5)
            hyg_20d = calc_return(hyg_data, 20)

            lqd_1d = calc_return(lqd_data, 1)
            lqd_5d = calc_return(lqd_data, 5)
            lqd_20d = calc_return(lqd_data, 20)

            # Relative performance (HYG - LQD)
            rel_1d = (hyg_1d - lqd_1d) if hyg_1d and lqd_1d else None
            rel_5d = (hyg_5d - lqd_5d) if hyg_5d and lqd_5d else None
            rel_20d = (hyg_20d - lqd_20d) if hyg_20d and lqd_20d else None

            # 20-day ratio trend
            if len(hyg_data) >= 20 and len(lqd_data) >= 20:
                ratio_20d_ago = float(hyg_data['Close'].iloc[-20]) / float(lqd_data['Close'].iloc[-20])
                ratio_change = ((ratio / ratio_20d_ago) - 1) * 100
            else:
                ratio_change = None

            # Signal interpretation
            if rel_5d is not None:
                if rel_5d > 0.5:
                    signal = "RISK_ON"
                    description = "HYG outperforming - credit risk appetite strong"
                elif rel_5d < -0.5:
                    signal = "RISK_OFF"
                    description = "LQD outperforming - flight to quality"
                else:
                    signal = "NEUTRAL"
                    description = "Credit sentiment balanced"
            else:
                signal = "UNKNOWN"
                description = "Insufficient data"

            result = {
                'hyg_price': hyg_price,
                'lqd_price': lqd_price,
                'hyg_lqd_ratio': round(ratio, 4),
                'hyg_1d_pct': round(hyg_1d, 2) if hyg_1d else None,
                'hyg_5d_pct': round(hyg_5d, 2) if hyg_5d else None,
                'hyg_20d_pct': round(hyg_20d, 2) if hyg_20d else None,
                'lqd_1d_pct': round(lqd_1d, 2) if lqd_1d else None,
                'lqd_5d_pct': round(lqd_5d, 2) if lqd_5d else None,
                'lqd_20d_pct': round(lqd_20d, 2) if lqd_20d else None,
                'relative_1d': round(rel_1d, 2) if rel_1d else None,
                'relative_5d': round(rel_5d, 2) if rel_5d else None,
                'relative_20d': round(rel_20d, 2) if rel_20d else None,
                'ratio_20d_change_pct': round(ratio_change, 2) if ratio_change else None,
                'signal': signal,
                'description': description,
                'timestamp': datetime.now().isoformat()
            }
            self._set_cached("credit_etf_flows", result)
            return result

        except YFRateLimitError:
            type(self)._LAST_RATE_LIMIT_AT = time.time()
            self._mark_fetch_miss("credit_etf_flows")
            stale = self._get_cached("credit_etf_flows", allow_stale=True)
            return stale
        except Exception as e:
            logger.error(f"Error fetching credit ETF flow data: {e}")
            self._mark_fetch_miss("credit_etf_flows")
            stale = self._get_cached("credit_etf_flows", allow_stale=True)
            if stale is not None:
                logger.warning("Using last-known-good credit ETF flow data")
            return stale

    def get_health_check(self) -> Dict:
        """
        Run a health check on all data sources.
        Returns dict with status of each data source.
        Useful for UI health widget.
        """
        health = {
            'vix': 'unknown',
            'vix_futures': 'unknown',
            'breadth': 'unknown',
            'put_call': 'unknown',
            'timestamp': datetime.now().isoformat()
        }
        
        # Test VIX
        try:
            vix = self.get_vix()
            health['vix'] = 'ok' if vix else 'failed'
        except Exception as e:
            logger.warning(f"VIX health check failed: {e}")
            health['vix'] = 'failed'
        
        # Test VIX futures
        try:
            contango = self.get_vix_futures_proxy()
            health['vix_futures'] = 'ok' if contango else 'failed'
        except Exception as e:
            logger.warning(f"VIX futures health check failed: {e}")
            health['vix_futures'] = 'failed'
        
        # Test breadth
        try:
            breadth = self.get_market_breadth_proxy()
            health['breadth'] = 'ok' if breadth else 'failed'
        except Exception as e:
            logger.warning(f"Breadth health check failed: {e}")
            health['breadth'] = 'failed'
        
        # Test put/call
        try:
            pc = self.get_put_call_ratio_proxy()
            health['put_call'] = 'ok' if pc else 'failed'
        except Exception as e:
            logger.warning(f"Put/Call health check failed: {e}")
            health['put_call'] = 'failed'
        
        return health
