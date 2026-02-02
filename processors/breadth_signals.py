"""
Advanced Breadth Signals Module
Implements: Zweig Thrust, A/D Ratio, Divergence Detection, Z-Scores

Parameters loaded from config/parameters.yaml
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from config import cfg

logger = logging.getLogger(__name__)


def compute_daily_breadth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure we have breadth_pct in [0, 100] and breadth_ratio in [0, 1].
    
    Accepts either:
      - columns: ['date', 'advancing', 'declining']
      - or:      ['date', 'breadth_pct']
    
    Returns a new DataFrame sorted by date with:
      ['date', 'advancing', 'declining', 'breadth_pct', 'breadth_ratio']
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    
    if "breadth_pct" not in df.columns:
        if not {"advancing", "declining"}.issubset(df.columns):
            raise ValueError(
                "compute_daily_breadth needs either 'breadth_pct' or "
                "both 'advancing' and 'declining' columns"
            )
        total = (df["advancing"] + df["declining"]).replace(0, pd.NA)
        df["breadth_pct"] = (df["advancing"] / total * 100).astype(float)
    
    df["breadth_ratio"] = df["breadth_pct"] / 100.0
    return df


def calculate_ad_ratio(advancing: int, declining: int) -> Dict[str, Any]:
    """
    Calculate Advance/Decline Ratio and interpret it

    Returns:
        dict with ratio, z_score, and interpretation
    """
    # Validate inputs
    if advancing < 0 or declining < 0:
        logger.warning(f"Invalid breadth data: advancing={advancing}, declining={declining}")
        return {
            "ratio": None,
            "interpretation": "Invalid Data",
            "color": "#9E9E9E",
            "advancing": advancing,
            "declining": declining,
            "error": "Negative values not allowed"
        }

    if declining == 0:
        # Handle division by zero gracefully
        if advancing == 0:
            ratio = 1.0  # No activity = neutral
        else:
            ratio = 10.0  # Cap at 10 instead of infinity for display purposes
            logger.info(f"No declining issues, capping ratio at 10.0 (actual advancing={advancing})")
    else:
        ratio = advancing / declining
    
    # Interpretation
    if ratio > 2.5:
        interpretation = "Strong Buying"
        color = "#4CAF50"
    elif ratio > 1.5:
        interpretation = "Moderate Buying"
        color = "#8BC34A"
    elif ratio > 0.67:
        interpretation = "Neutral"
        color = "#FF9800"
    elif ratio > 0.4:
        interpretation = "Moderate Selling"
        color = "#FF6B6B"
    else:
        interpretation = "Strong Selling"
        color = "#F44336"
    
    return {
        'ratio': ratio,
        'advancing': advancing,
        'declining': declining,
        'interpretation': interpretation,
        'color': color
    }


def detect_zweig_breadth_thrust(
    breadth_df: pd.DataFrame,
    ema_span: Optional[int] = None,
    window_days: Optional[int] = None,
    low_thresh: Optional[float] = None,
    high_thresh: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Detect Zweig Breadth Thrust signal.

    A Zweig Thrust occurs when:
    - 10-day EMA of breadth ratio goes from <0.40 to >0.615
    - Within 10 trading days or less

    This is a rare, powerful bullish signal (happens few times per decade)

    Parameters loaded from config/parameters.yaml

    Parameters
    ----------
    breadth_df : DataFrame
        Must contain 'date' and either 'breadth_pct' or 'advancing'+'declining'
    ema_span : int, optional
        EMA length for Zweig (default from config)
    window_days : int, optional
        Max number of days allowed for thrust (default from config)
    low_thresh : float, optional
        Oversold boundary for EMA (default from config)
    high_thresh : float, optional
        Overbought boundary for EMA (default from config)

    Returns
    -------
    dict with keys:
        active: bool
        signal_date: Optional[date]
        days_since_signal: Optional[int]
        ema10: float
        from_level: float
        to_level: float
        description: str
    """
    # Load defaults from config
    zweig_cfg = cfg.breadth.zweig_thrust
    ema_span = ema_span or zweig_cfg.ema_span
    window_days = window_days or zweig_cfg.window_days
    low_thresh = low_thresh or zweig_cfg.low_threshold
    high_thresh = high_thresh or zweig_cfg.high_threshold
    active_days = zweig_cfg.active_days
    df = compute_daily_breadth(breadth_df)
    
    if len(df) < ema_span + window_days + 2:
        return {
            "active": False,
            "signal_date": None,
            "days_since_signal": None,
            "ema10": float("nan"),
            "from_level": float("nan"),
            "to_level": float("nan"),
            "description": "Insufficient data"
        }
    
    # Calculate 10-day EMA of breadth ratio
    df["ema10"] = df["breadth_ratio"].ewm(span=ema_span, adjust=False).mean()
    df = df.reset_index(drop=True)
    
    signal_date: Optional[pd.Timestamp] = None
    
    # Scan rolling windows to find last valid thrust
    for end_idx in range(ema_span + window_days, len(df)):
        start_idx = end_idx - window_days
        
        window = df.iloc[start_idx:end_idx + 1]
        ema_start = window["ema10"].iloc[0]
        ema_end = window["ema10"].iloc[-1]
        
        # Check if thrust occurred
        if ema_start < low_thresh and ema_end > high_thresh:
            signal_date = df.loc[end_idx, "date"]
    
    latest_ema = float(df["ema10"].iloc[-1])
    from_level = float(df["ema10"].iloc[-window_days - 1]) if len(df) > window_days else latest_ema
    to_level = latest_ema
    
    if signal_date is not None:
        days_since = int((df["date"].iloc[-1] - signal_date).days)
        active = days_since <= active_days  # Thrust "active" period from config

        description = f"Zweig Thrust triggered on {signal_date.date()} ({days_since} days ago)"
    else:
        days_since = None
        active = False
        description = "No thrust detected (last 60 days)"
    
    return {
        "active": active,
        "signal_date": signal_date.date() if signal_date is not None else None,
        "days_since_signal": days_since,
        "ema10": latest_ema,
        "from_level": from_level,
        "to_level": to_level,
        "description": description,
        "color": "#4CAF50" if active else "#9E9E9E"
    }


def detect_breadth_divergence(
    price_series: pd.Series,
    ad_line_series: pd.Series,
    lookback: Optional[int] = None
) -> Dict[str, Any]:
    """
    Detect bullish/bearish divergences between price and A/D Line
    
    Bearish Divergence: Price makes new high, A/D Line doesn't
    Bullish Divergence: Price makes new low, A/D Line doesn't
    
    Parameters
    ----------
    price_series : pd.Series
        Price data (e.g., SPY close) indexed by date
    ad_line_series : pd.Series
        A/D Line indexed by date
    lookback : int, optional
        Days to look back for divergences (default from config)

    Returns
    -------
    dict with:
        type: 'bearish' | 'bullish' | 'none'
        description: str
        days_ago: int
        color: str
    """
    # Load from config if not provided
    lookback = lookback or cfg.breadth.divergence.lookback_days

    if len(price_series) < lookback or len(ad_line_series) < lookback:
        return {
            'type': 'none',
            'description': 'Insufficient data',
            'days_ago': None,
            'color': '#9E9E9E'
        }
    
    recent_price = price_series.tail(lookback)
    recent_ad = ad_line_series.tail(lookback)
    
    # Find extrema
    price_high_idx = recent_price.idxmax()
    ad_high_idx = recent_ad.idxmax()
    price_low_idx = recent_price.idxmin()
    ad_low_idx = recent_ad.idxmin()
    
    # Check for bearish divergence (price new high, breadth not confirming)
    if price_high_idx == recent_price.index[-1]:
        # Price just made a new high
        if ad_high_idx != recent_ad.index[-1]:
            # But A/D line high was earlier
            days_ago = (recent_price.index[-1] - ad_high_idx).days
            
            if days_ago >= 5:  # Require meaningful separation
                return {
                    'type': 'bearish',
                    'description': f'Price at {lookback}d high, breadth peaked {days_ago}d ago',
                    'days_ago': days_ago,
                    'color': '#F44336',
                    'severity': 'warning'
                }
    
    # Check for bullish divergence (price new low, breadth holding up)
    if price_low_idx == recent_price.index[-1]:
        # Price just made a new low
        if ad_low_idx != recent_ad.index[-1]:
            # But A/D line low was earlier
            days_ago = (recent_price.index[-1] - ad_low_idx).days
            
            if days_ago >= 5:
                return {
                    'type': 'bullish',
                    'description': f'Price at {lookback}d low, breadth holding above {days_ago}d ago low',
                    'days_ago': days_ago,
                    'color': '#4CAF50',
                    'severity': 'positive'
                }
    
    return {
        'type': 'none',
        'description': f'No significant divergence (last {lookback} days)',
        'days_ago': None,
        'color': '#9E9E9E',
        'severity': 'neutral'
    }


def calculate_breadth_zscore(breadth_history: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate z-score for current breadth vs historical average
    
    Returns:
        dict with current breadth, mean, std, z_score, interpretation
    """
    if len(breadth_history) < 30:
        return {
            'current': None,
            'mean': None,
            'std': None,
            'z_score': None,
            'interpretation': 'Insufficient data'
        }
    
    breadth_pct = breadth_history['breadth_pct']
    current = breadth_pct.iloc[-1]
    mean = breadth_pct.mean()
    std = breadth_pct.std()
    
    z_score = (current - mean) / std if std > 0 else 0
    
    # Interpretation
    if z_score > 2:
        interpretation = "Extremely Strong"
    elif z_score > 1:
        interpretation = "Above Average"
    elif z_score > -1:
        interpretation = "Normal"
    elif z_score > -2:
        interpretation = "Below Average"
    else:
        interpretation = "Extremely Weak"
    
    return {
        'current': current,
        'mean': mean,
        'std': std,
        'z_score': z_score,
        'interpretation': interpretation
    }


def get_comprehensive_breadth_signals(
    breadth_history: pd.DataFrame,
    price_series: Optional[pd.Series] = None
) -> Dict[str, Any]:
    """
    Get all breadth signals in one call
    
    Parameters
    ----------
    breadth_history : DataFrame
        With columns: date, advancing, declining, breadth_pct, ad_line
    price_series : pd.Series, optional
        Price data for divergence detection
    
    Returns
    -------
    Comprehensive dict with all signals:
        - ad_ratio
        - zweig_thrust
        - divergence
        - z_score
        - latest_breadth
    """
    if breadth_history.empty:
        return {
            'error': 'No breadth data available',
            'ad_ratio': None,
            'zweig_thrust': None,
            'divergence': None,
            'z_score': None
        }
    
    latest = breadth_history.iloc[-1]
    
    # 1. A/D Ratio
    ad_ratio = calculate_ad_ratio(
        int(latest['advancing']),
        int(latest['declining'])
    )
    
    # 2. Zweig Thrust
    zweig = detect_zweig_breadth_thrust(breadth_history)
    
    # 3. Divergence (if price data provided)
    if price_series is not None and 'ad_line' in breadth_history.columns:
        # Align dates
        ad_line = breadth_history.set_index('date')['ad_line']
        divergence = detect_breadth_divergence(price_series, ad_line)
    else:
        divergence = {
            'type': 'none',
            'description': 'Price data not provided',
            'days_ago': None,
            'color': '#9E9E9E'
        }
    
    # 4. Z-Score
    z_score = calculate_breadth_zscore(breadth_history)
    
    return {
        'latest_breadth': {
            'breadth_pct': float(latest['breadth_pct']),
            'advancing': int(latest['advancing']),
            'declining': int(latest['declining']),
            'date': latest['date']
        },
        'ad_ratio': ad_ratio,
        'zweig_thrust': zweig,
        'divergence': divergence,
        'z_score': z_score
    }

