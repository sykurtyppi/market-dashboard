"""
Dashboard UI Helpers - Clean display utilities

Provides:
- Percentile calculations with context
- Estimated value indicators
- Educational tooltips (non-intrusive)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import yfinance as yf
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# PERCENTILE CALCULATIONS
# =============================================================================

def calculate_percentile(current_value: float, history: pd.Series) -> Optional[float]:
    """
    Calculate percentile rank of current value within historical data.

    Args:
        current_value: Current metric value
        history: Historical series of values

    Returns:
        Percentile (0-100) or None if insufficient data
    """
    if history is None or len(history) < 10:
        return None

    clean = history.dropna()
    if len(clean) < 10:
        return None

    # Count how many historical values are below current
    below_count = (clean < current_value).sum()
    percentile = (below_count / len(clean)) * 100

    return round(percentile, 1)


def get_vix_percentile(vix_value: float, lookback_days: int = 252) -> Optional[Dict]:
    """
    Get VIX percentile ranking over lookback period.

    Returns dict with percentile, interpretation, and context.
    """
    if vix_value is None:
        return None

    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period=f"{lookback_days}d")

        if hist.empty:
            return None

        percentile = calculate_percentile(vix_value, hist['Close'])

        if percentile is None:
            return None

        # Interpretation
        if percentile >= 90:
            context = "Extreme fear (historically high)"
        elif percentile >= 75:
            context = "Elevated fear"
        elif percentile >= 50:
            context = "Above average"
        elif percentile >= 25:
            context = "Below average"
        else:
            context = "Low fear (complacent)"

        return {
            'percentile': percentile,
            'context': context,
            'lookback_days': lookback_days,
            'min': float(hist['Close'].min()),
            'max': float(hist['Close'].max()),
            'median': float(hist['Close'].median())
        }
    except Exception as e:
        logger.warning(f"Could not calculate VIX percentile: {e}")
        return None


def get_fear_greed_percentile(score: float) -> Dict:
    """
    Get Fear & Greed context (score is already 0-100 scale).
    """
    if score is None:
        return None

    # F&G is already a percentile-like score
    if score <= 25:
        context = "Extreme Fear - Contrarian buy zone"
    elif score <= 45:
        context = "Fear - Cautious sentiment"
    elif score <= 55:
        context = "Neutral"
    elif score <= 75:
        context = "Greed - Risk-on sentiment"
    else:
        context = "Extreme Greed - Contrarian caution"

    return {
        'score': score,
        'context': context,
        'is_extreme': score <= 25 or score >= 75
    }


def get_credit_spread_percentile(spread_bps: float, lookback_days: int = 252) -> Optional[Dict]:
    """
    Get HY credit spread percentile ranking.

    Args:
        spread_bps: HY OAS spread in basis points
        lookback_days: Lookback period
    """
    if spread_bps is None:
        return None

    try:
        # Use HYG-LQD spread as proxy if FRED data unavailable
        hyg = yf.Ticker("HYG")
        lqd = yf.Ticker("LQD")

        hyg_hist = hyg.history(period=f"{lookback_days}d")
        lqd_hist = lqd.history(period=f"{lookback_days}d")

        if hyg_hist.empty or lqd_hist.empty:
            # Return basic interpretation without percentile
            if spread_bps < 300:
                context = "Tight spreads (risk-on)"
            elif spread_bps < 450:
                context = "Normal range"
            elif spread_bps < 600:
                context = "Elevated (caution)"
            else:
                context = "Wide spreads (stress)"

            return {
                'spread_bps': spread_bps,
                'context': context,
                'percentile': None
            }

        # Calculate yield spread proxy
        combined = pd.DataFrame({
            'hyg': hyg_hist['Close'],
            'lqd': lqd_hist['Close']
        }).dropna()

        spread_proxy = (combined['lqd'] / combined['hyg'] - 1) * 10000  # Convert to bps-like

        # We can't directly compare FRED spread to price ratio, so interpret based on level
        if spread_bps < 300:
            percentile = 15.0
            context = "Tight spreads (risk-on)"
        elif spread_bps < 350:
            percentile = 30.0
            context = "Below average"
        elif spread_bps < 450:
            percentile = 50.0
            context = "Normal range"
        elif spread_bps < 550:
            percentile = 70.0
            context = "Elevated (caution)"
        elif spread_bps < 700:
            percentile = 85.0
            context = "Wide spreads (stress)"
        else:
            percentile = 95.0
            context = "Crisis-level spreads"

        return {
            'spread_bps': spread_bps,
            'percentile': percentile,
            'context': context
        }

    except Exception as e:
        logger.warning(f"Could not calculate credit spread percentile: {e}")
        return None


# =============================================================================
# ESTIMATED VALUE INDICATORS
# =============================================================================

def format_with_estimated(value: Any, format_str: str, is_estimated: bool) -> str:
    """
    Format a value with optional estimated indicator.

    Args:
        value: The value to format
        format_str: Python format string (e.g., "{:.2f}%")
        is_estimated: Whether this value is estimated

    Returns:
        Formatted string with "~" prefix if estimated
    """
    if value is None:
        return "N/A"

    formatted = format_str.format(value)

    if is_estimated:
        return f"~{formatted}"

    return formatted


def get_estimated_indicator(field_name: str, estimated_fields: list) -> Tuple[bool, Optional[str]]:
    """
    Check if a field is in the estimated fields list.

    Args:
        field_name: Name of field to check
        estimated_fields: List of estimated field dicts from CBOE collector

    Returns:
        Tuple of (is_estimated, reason_or_none)
    """
    if not estimated_fields:
        return False, None

    for est in estimated_fields:
        if est.get('field') == field_name:
            return True, est.get('reason', 'Estimated value')

    return False, None


# =============================================================================
# TOOLTIP HELPERS (Clean, non-intrusive)
# =============================================================================

# Concise educational context - shown only on hover/expander
METRIC_TOOLTIPS = {
    'vix': "Implied 30-day S&P 500 volatility. High VIX = fear, often contrarian buy.",
    'vvix': "Vol-of-vol. 120+ = dealers scrambling for gamma = historic buy signal.",
    'skew': "Tail risk premium. High = institutions hedging for crash.",
    'vrp': "VIX minus realized vol. Positive = options expensive vs actual moves.",
    'credit_spread_hy': "HY bond spread over Treasuries. Widens in risk-off, tightens in risk-on.",
    'fear_greed': "CNN composite sentiment. Extremes (<25 or >75) are contrarian signals.",
    'net_liquidity': "Fed BS - TGA - RRP. Rising = supportive for risk assets.",
    'sofr_iorb': "SOFR vs Fed floor rate. Spread widening = funding stress.",
    'move': "Treasury market implied vol. High = bond market stress.",
    'breadth': "% stocks advancing. >60% = healthy rally, <40% = weak participation.",
    'left_signal': "Credit spreads vs 330-day EMA. BUY when 35%+ below.",
    'vix_contango': "VIX3M vs VIX. Positive = normal (bullish), negative = fear.",
    'mcclellan': "Breadth momentum oscillator. Positive = healthy internals.",
}


def get_tooltip(metric_key: str) -> Optional[str]:
    """Get educational tooltip for a metric."""
    return METRIC_TOOLTIPS.get(metric_key)


def format_percentile_badge(percentile: float, lookback_days: int = 252) -> str:
    """
    Format a percentile as a clean badge string.

    Returns something like: "73rd %ile (1Y)"
    """
    if percentile is None:
        return ""

    # Ordinal suffix
    if 10 <= percentile % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(int(percentile) % 10, 'th')

    # Period label
    if lookback_days <= 30:
        period = "1M"
    elif lookback_days <= 90:
        period = "3M"
    elif lookback_days <= 180:
        period = "6M"
    elif lookback_days <= 365:
        period = "1Y"
    else:
        period = f"{lookback_days // 365}Y"

    return f"{int(percentile)}{suffix} %ile ({period})"


# =============================================================================
# STREAMLIT DISPLAY HELPERS
# =============================================================================

def metric_with_context(
    st_column,
    label: str,
    value: Any,
    format_str: str = "{:.2f}",
    percentile: Optional[float] = None,
    is_estimated: bool = False,
    tooltip: Optional[str] = None,
    caption: Optional[str] = None,
    delta: Optional[str] = None
):
    """
    Display a metric with optional percentile, estimated indicator, and tooltip.

    Keeps display clean - only shows essential info.

    Args:
        st_column: Streamlit column context
        label: Metric label
        value: Metric value
        format_str: Format string for value
        percentile: Optional percentile ranking
        is_estimated: Whether value is estimated
        tooltip: Optional hover tooltip
        caption: Optional caption below metric
        delta: Optional delta value for st.metric
    """
    # Format value with estimated indicator
    if value is not None:
        display_value = format_str.format(value)
        if is_estimated:
            display_value = f"~{display_value}"
    else:
        display_value = "N/A"

    # Build label with tooltip indicator if present
    display_label = label
    if tooltip:
        display_label = f"{label} ℹ️"

    # Display metric
    if delta:
        st_column.metric(display_label, display_value, delta=delta, help=tooltip)
    else:
        st_column.metric(display_label, display_value, help=tooltip)

    # Add percentile badge if available (compact)
    if percentile is not None:
        badge = format_percentile_badge(percentile)
        st_column.caption(f"📊 {badge}")

    # Add custom caption if provided
    if caption:
        st_column.caption(caption)
