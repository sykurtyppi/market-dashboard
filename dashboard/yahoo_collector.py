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
        Approximate Put/Call using VIX vs VXV ratio
        High VIX/VXV = more fear (higher P/C)
        
        Note: This calls get_vix() which has its own retry logic.
        In case of nested failures, total attempts = outer_retries * inner_retries.
        """
        vix = self.get_vix()
        
        # VXV is 3-month VIX - use as proxy
        vxv = yf.Ticker("^VXV")
        vxv_data = vxv.history(period="1d")
        
        if vix and not vxv_data.empty:
            vxv_value = float(vxv_data['Close'].iloc[-1])
            
            # Normalize to typical P/C range (0.7-1.1)
            # VIX/VXV typically ranges 0.8-1.2
            ratio = vix / vxv_value
            
            # Map to P/C: high ratio = high fear = high P/C
            pc_proxy = 0.5 + (ratio * 0.5)  # Maps ~0.8-1.2 to ~0.9-1.1
            
            return pc_proxy
        
        raise ValueError("VIX or VXV data unavailable")
    
    def get_all_data(self) -> Dict:
        """Get all Yahoo Finance data with retry logic"""
        return {
            'vix': self.get_vix(),
            'vix_contango_proxy': self.get_vix_futures_proxy(),
            'market_breadth_proxy': self.get_market_breadth_proxy(),
            'put_call_proxy': self.get_put_call_ratio_proxy(),
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


# Test function
if __name__ == "__main__":
    print("Testing Yahoo Finance Collector with Retry Logic...")
    collector = YahooCollector()
    
    data = collector.get_all_data()
    
    print(f"\nVIX: {data['vix']}")
    print(f"VIX Contango Proxy: {data['vix_contango_proxy']}")
    print(f"Market Breadth Proxy: {data['market_breadth_proxy']}")
    print(f"Put/Call Proxy: {data['put_call_proxy']}")