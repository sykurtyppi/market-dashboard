"""
Retry utilities for robust API calls.

Default parameters loaded from config/parameters.yaml
"""
import time
import logging
from functools import wraps
from typing import Callable, Optional, Tuple
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
