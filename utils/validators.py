"""
Input Validators for Market Dashboard

Validates financial data inputs to catch data feed errors,
NaN values, and obviously incorrect data.

VIX HISTORICAL RANGE:
    The VIX index has a theoretical floor near 0 but practically
    almost never goes below ~9 (extreme complacency).

    Historical extremes:
        - All-time high: ~89.53 (intraday, March 16, 2020 COVID crash)
        - All-time closing high: ~82.69 (March 16, 2020)
        - 1987 crash equivalent: ~150+ (VIX didn't exist, but VXO proxy)
        - 2008 crisis peak: ~80.86 (November 2008)
        - All-time low: ~8.56 (November 2017)

    VALIDATION RANGE: 3-200
        - Lower bound (3): Below any recorded value, catches zero/negative errors
        - Upper bound (200): Accommodates 1987-level events with margin for error

    Previous range (5-100) would have rejected valid data during extreme events.

REALIZED VOLATILITY RANGE:
    Similar logic - during crashes, realized vol can spike dramatically.
    The 2008 and 2020 crashes saw SPY realized vol >100% annualized.

    VALIDATION RANGE: 0-250
        - Allows for extreme market dislocations
        - Catches obviously erroneous negative values
"""

import logging
from typing import Optional
import math

logger = logging.getLogger(__name__)


# Historical reference points for validation boundaries
VIX_MIN = 3      # Below any recorded value (catches zero/negative errors)
VIX_MAX = 200    # Accommodates 1987-equivalent events with margin
REALIZED_VOL_MIN = 0
REALIZED_VOL_MAX = 250  # Extreme crashes can exceed 100% annualized


def validate_vix(vix: Optional[float]) -> Optional[float]:
    """
    Validate VIX value is within historically plausible range.

    Range: 3-200 (see module docstring for historical basis)

    Note: Previous range of 5-100 would have rejected valid crisis data.
    The expanded range accommodates historical extremes while still
    catching data feed errors (negative values, astronomical numbers).
    """
    if vix is None:
        logger.warning("VIX is None")
        return None

    if math.isnan(vix) or math.isinf(vix):
        logger.error(f"VIX is NaN or Inf: {vix}")
        return None

    if vix < VIX_MIN or vix > VIX_MAX:
        logger.error(f"VIX out of plausible range ({VIX_MIN}-{VIX_MAX}): {vix}")
        return None

    # Log warning for extreme but valid values
    if vix > 80:
        logger.warning(f"VIX at crisis level: {vix} (valid but extreme)")

    return vix

def validate_realized_vol(vol: Optional[float]) -> Optional[float]:
    """
    Validate realized volatility is within historically plausible range.

    Range: 0-250 (see module docstring for historical basis)

    Note: During extreme crashes (2008, 2020), SPY realized vol exceeded
    100% annualized. Previous range of 0-100 would have rejected valid data.
    """
    if vol is None:
        logger.warning("Realized vol is None")
        return None

    if math.isnan(vol) or math.isinf(vol):
        logger.error(f"Realized vol is NaN or Inf: {vol}")
        return None

    if vol < REALIZED_VOL_MIN or vol > REALIZED_VOL_MAX:
        logger.error(f"Realized vol out of plausible range ({REALIZED_VOL_MIN}-{REALIZED_VOL_MAX}): {vol}")
        return None

    # Log warning for extreme but valid values
    if vol > 80:
        logger.warning(f"Realized vol at extreme level: {vol} (valid but unusual)")

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
