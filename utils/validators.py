import logging
from typing import Optional
import math

logger = logging.getLogger(__name__)

def validate_vix(vix: Optional[float]) -> Optional[float]:
    """Validate VIX value is reasonable (5-100 range)"""
    if vix is None:
        logger.warning("VIX is None")
        return None
    
    if math.isnan(vix) or math.isinf(vix):
        logger.error(f"VIX is NaN or Inf: {vix}")
        return None
    
    if vix < 5 or vix > 100:
        logger.error(f"VIX out of range: {vix}")
        return None
    
    return vix

def validate_realized_vol(vol: Optional[float]) -> Optional[float]:
    """Validate realized volatility is reasonable (0-100 range)"""
    if vol is None:
        logger.warning("Realized vol is None")
        return None
    
    if math.isnan(vol) or math.isinf(vol):
        logger.error(f"Realized vol is NaN or Inf: {vol}")
        return None
    
    if vol < 0 or vol > 100:
        logger.error(f"Realized vol out of range: {vol}")
        return None
    
    return vol

def validate_price(price: Optional[float], min_val: float = 0, max_val: float = 1000000) -> Optional[float]:
    """Validate price is reasonable"""
    if price is None:
        return None
    
    if math.isnan(price) or math.isinf(price):
        logger.error(f"Price is NaN or Inf: {price}")
        return None
    
    if price < min_val or price > max_val:
        logger.error(f"Price out of range: {price}")
        return None
    
    return price

def validate_ratio(ratio: Optional[float], min_val: float = 0, max_val: float = 10) -> Optional[float]:
    """Validate ratio (like P/C) is reasonable"""
    if ratio is None:
        return None
    
    if math.isnan(ratio) or math.isinf(ratio):
        logger.error(f"Ratio is NaN or Inf: {ratio}")
        return None
    
    if ratio < min_val or ratio > max_val:
        logger.error(f"Ratio out of range: {ratio}")
        return None
    
    return ratio
