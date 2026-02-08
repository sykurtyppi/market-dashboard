"""
Retry utilities for robust API calls.

Default parameters loaded from config/parameters.yaml

Includes:
- exponential_backoff_retry: Decorator for retrying failed API calls
- simple_retry: Simple fixed-delay retry decorator
- RateLimiter: Class for rate limiting API calls
- rate_limit: Decorator for applying rate limits
"""
import time
import logging
import threading
from functools import wraps
from typing import Callable, Optional, Tuple, Dict
from collections import defaultdict
import requests

from config import cfg

logger = logging.getLogger(__name__)

# Load defaults from config with fallbacks for Streamlit Cloud compatibility
try:
    _retry_cfg = cfg.data_collection.retry
except AttributeError:
    # Fallback defaults if config doesn't have retry section
    class _RetryDefaults:
        max_retries = 3
        initial_delay = 1.0
        max_delay = 60.0
        backoff_multiplier = 2.0
    _retry_cfg = _RetryDefaults()
    logger.info("Using fallback retry defaults")


def exponential_backoff_retry(
    max_retries: Optional[int] = None,
    base_delay: Optional[float] = None,
    max_delay: Optional[float] = None,
    backoff_factor: Optional[float] = None,
    retry_on: Tuple = (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.HTTPError)
):
    """
    Decorator for exponential backoff retry logic.

    Args:
        max_retries: Maximum number of retry attempts (default from config)
        base_delay: Initial delay in seconds (default from config)
        max_delay: Maximum delay between retries (default from config)
        backoff_factor: Multiplier for delay on each retry (default from config)
        retry_on: Tuple of exceptions to retry on
    """
    # Use config defaults if not specified
    max_retries = max_retries if max_retries is not None else _retry_cfg.max_retries
    base_delay = base_delay if base_delay is not None else _retry_cfg.initial_delay
    max_delay = max_delay if max_delay is not None else _retry_cfg.max_delay
    backoff_factor = backoff_factor if backoff_factor is not None else _retry_cfg.backoff_multiplier
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except retry_on as e:
                    last_exception = e
                    
                    # Check if it's an HTTPError we should NOT retry
                    if isinstance(e, requests.exceptions.HTTPError):
                        # Don't retry 4xx errors (except 429 rate limit)
                        if hasattr(e, 'response') and e.response is not None:
                            status_code = e.response.status_code
                            if 400 <= status_code < 500 and status_code != 429:
                                logger.warning(f"{func.__name__}: Client error {status_code}, not retrying")
                                raise
                    
                    if attempt < max_retries:
                        logger.warning(
                            f"{func.__name__}: Attempt {attempt + 1}/{max_retries} failed: {str(e)}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        delay = min(delay * backoff_factor, max_delay)
                    else:
                        logger.error(
                            f"{func.__name__}: All {max_retries} retries exhausted. "
                            f"Last error: {str(e)}"
                        )
                        raise
                        
                except Exception as e:
                    # Don't retry unexpected exceptions
                    logger.error(f"{func.__name__}: Unexpected error: {str(e)}")
                    raise
            
            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


def simple_retry(max_attempts: int = 3, delay: float = 1.0):
    """
    Simple retry decorator with fixed delay.

    Args:
        max_attempts: Number of attempts
        delay: Delay between attempts in seconds
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    logger.warning(f"{func.__name__}: Attempt {attempt + 1} failed, retrying...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator


# =============================================================================
# RATE LIMITING
# =============================================================================

class RateLimiter:
    """
    Token bucket rate limiter for API calls.

    Provides thread-safe rate limiting with configurable requests per second.

    Usage:
        limiter = RateLimiter(requests_per_second=2)  # 2 requests/sec

        @limiter.limit
        def fetch_data():
            return requests.get(url)

        # Or manually:
        limiter.wait()  # Blocks until rate limit allows
        response = requests.get(url)
    """

    # Shared limiters by API name (thread-safe singleton pattern)
    _instances: Dict[str, 'RateLimiter'] = {}
    _lock = threading.Lock()

    def __init__(self, requests_per_second: float = 1.0, burst: int = 1):
        """
        Initialize rate limiter.

        Args:
            requests_per_second: Maximum requests per second
            burst: Maximum burst size (tokens available immediately)
        """
        self.rate = requests_per_second
        self.burst = burst
        self.tokens = float(burst)
        self.last_update = time.monotonic()
        self._lock = threading.Lock()

    @classmethod
    def get_limiter(cls, name: str, requests_per_second: float = 1.0, burst: int = 1) -> 'RateLimiter':
        """
        Get or create a named rate limiter (singleton per API).

        Args:
            name: API/service name (e.g., 'yahoo', 'fred', 'cboe')
            requests_per_second: Rate limit
            burst: Burst size

        Returns:
            RateLimiter instance
        """
        with cls._lock:
            if name not in cls._instances:
                cls._instances[name] = RateLimiter(requests_per_second, burst)
                logger.debug(f"Created rate limiter '{name}': {requests_per_second} req/s, burst={burst}")
            return cls._instances[name]

    def _add_tokens(self):
        """Add tokens based on elapsed time"""
        now = time.monotonic()
        elapsed = now - self.last_update
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        self.last_update = now

    def wait(self):
        """
        Wait until a request is allowed.

        Blocks the thread until the rate limit permits a request.
        """
        with self._lock:
            self._add_tokens()

            if self.tokens < 1:
                # Calculate wait time
                wait_time = (1 - self.tokens) / self.rate
                logger.debug(f"Rate limited, waiting {wait_time:.2f}s")
                time.sleep(wait_time)
                self._add_tokens()

            self.tokens -= 1

    def try_acquire(self) -> bool:
        """
        Try to acquire permission for a request without blocking.

        Returns:
            True if request is allowed, False if rate limited
        """
        with self._lock:
            self._add_tokens()
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

    def limit(self, func: Callable) -> Callable:
        """
        Decorator to rate limit a function.

        Usage:
            limiter = RateLimiter(requests_per_second=2)

            @limiter.limit
            def fetch_data():
                return requests.get(url)
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.wait()
            return func(*args, **kwargs)
        return wrapper


def rate_limit(api_name: str, requests_per_second: float = 1.0, burst: int = 1):
    """
    Decorator for rate limiting function calls.

    Uses a shared rate limiter per API name, so all calls to the same API
    share the rate limit even across different functions.

    Args:
        api_name: Name of the API (e.g., 'yahoo', 'fred')
        requests_per_second: Maximum requests per second
        burst: Maximum burst size

    Usage:
        @rate_limit('yahoo', requests_per_second=2)
        def fetch_yahoo_data(ticker):
            return yf.Ticker(ticker).history(period='5d')
    """
    def decorator(func: Callable) -> Callable:
        limiter = RateLimiter.get_limiter(api_name, requests_per_second, burst)

        @wraps(func)
        def wrapper(*args, **kwargs):
            limiter.wait()
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Pre-configured rate limiters for common APIs
YAHOO_LIMITER = RateLimiter.get_limiter('yahoo', requests_per_second=2.0, burst=5)
FRED_LIMITER = RateLimiter.get_limiter('fred', requests_per_second=5.0, burst=10)
CBOE_LIMITER = RateLimiter.get_limiter('cboe', requests_per_second=1.0, burst=3)
