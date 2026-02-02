"""
Yahoo Finance Collector - IMPROVED VERSION v2
Fetches VIX, market breadth, and supplementary options data with robust retry logic
Now with proper config usage, logging, and narrower exception catching
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, Callable, Any
import time
import logging
from functools import wraps

# Set up logger
logger = logging.getLogger(__name__)

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
    
    def __init__(self, retry_config: RetryConfig = None):
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
    
    @with_retry
    def get_vix(self) -> Optional[float]:
        """Get current VIX spot price"""
        vix = yf.Ticker("^VIX")
        data = vix.history(period="1d")
        
        if not data.empty:
            return float(data['Close'].iloc[-1])
        
        raise ValueError("VIX data is empty")
    
    @with_retry
    def get_vix_futures_proxy(self) -> Optional[float]:
        """
        Calculate VIX term structure using VIX ETFs as proxy
        VIXY = short-term VIX futures
        VXZ = mid-term VIX futures
        """
        vixy = yf.Ticker("VIXY")  # Short-term VIX futures ETN
        vxz = yf.Ticker("VXZ")    # Mid-term VIX futures ETN
        
        vixy_data = vixy.history(period="5d")
        vxz_data = vxz.history(period="5d")
        
        if not vixy_data.empty and not vxz_data.empty:
            vixy_price = float(vixy_data['Close'].iloc[-1])
            vxz_price = float(vxz_data['Close'].iloc[-1])
            
            # Approximate contango as (mid-term / short-term - 1) * 100
            contango = (vxz_price / vixy_price - 1) * 100
            
            return contango
        
        raise ValueError("VIX ETF data is empty")
    
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
        
        if total > 0:
            return advancing / total
        
        raise ValueError("No sector ETF data available for breadth calculation")
    
    @with_retry
    def get_put_call_ratio_proxy(self) -> Optional[float]:
        """
        Get Put/Call ratio using SPY options volume (VXV is delisted)
        Returns ratio of put volume to call volume
        """
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
                return pc_ratio
            else:
                logger.warning("SPY call volume is zero")
                return None
                
        except Exception as e:
            logger.error(f"Error calculating SPY P/C ratio: {e}")
            return None
    
        """Get all Yahoo Finance data with retry logic"""
        return {
            'vix': self.get_vix(),
            'vix_contango_proxy': self.get_vix_futures_proxy(),
            'market_breadth_proxy': self.get_market_breadth_proxy(),
            'put_call_proxy': self.get_put_call_ratio_proxy(),
            'timestamp': datetime.now().isoformat()
        }
    
    
    def get_all_data(self) -> Dict:
        """Get all Yahoo Finance data"""
        return {
            'vix': self.get_vix(),
            'vix_contango_proxy': self.get_vix_futures_proxy(),
            'market_breadth_proxy': self.get_market_breadth_proxy(),
            'put_call_proxy': self.get_put_call_ratio_proxy(),
            'timestamp': datetime.now().isoformat()
        }

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

        return {
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
