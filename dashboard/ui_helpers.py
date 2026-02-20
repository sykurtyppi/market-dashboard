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

from utils.data_status import DataResult, DataStatus, format_age_string


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


# =============================================================================
# DATA SOURCE LABELS
# =============================================================================

def data_source_caption(
    st_column,
    source: str,
    delay: Optional[str] = None,
    note: Optional[str] = None
):
    """
    Display a compact data source + delay label under a metric.

    Args:
        st_column: Streamlit column context
        source: Data source label (e.g., "FRED (DGS10)")
        delay: Optional delay label (e.g., "daily", "delayed")
        note: Optional extra note
    """
    parts = [f"Source: {source}"]
    if delay:
        parts.append(f"Delay: {delay}")
    if note:
        parts.append(note)

    st_column.caption(" | ".join(parts))


def metric_status_caption(st_column, data_result: Optional[DataResult]):
    """
    Display a compact status badge for a metric.

    Args:
        st_column: Streamlit column context
        data_result: DataResult from DataStatusTracker
    """
    if not data_result:
        return

    # Handle case where status is None
    if not data_result.status:
        return

    status_value = data_result.status
    status_enum = None
    if isinstance(status_value, DataStatus):
        status_enum = status_value
    elif isinstance(status_value, str):
        try:
            status_enum = DataStatus(status_value.lower())
        except ValueError:
            status_enum = None

    if status_enum:
        label = status_enum.value.replace("_", " ").title()
        emoji = {
            DataStatus.OK: "✅",
            DataStatus.STALE: "⚠️",
            DataStatus.ESTIMATED: "📊",
            DataStatus.UNAVAILABLE: "❌",
            DataStatus.ERROR: "🚨",
            DataStatus.PARTIAL: "⚡",
        }.get(status_enum, "❓")
    else:
        label = str(status_value).replace("_", " ").title()
        emoji = "❓"
    age_note = ""
    if data_result.age_hours and data_result.age_hours > 0:
        age_note = f" ({format_age_string(data_result.age_hours)})"

    st_column.caption(f"Status: {emoji} {label}{age_note}")


# =============================================================================
# DASHBOARD HEALTH CHECK
# =============================================================================

def get_data_health_status() -> Dict[str, Any]:
    """
    Check health/freshness of all data sources.

    Returns:
        Dict with status for each data source and overall health score
    """
    health = {
        'sources': {},
        'overall_status': 'unknown',
        'healthy_count': 0,
        'total_count': 0,
        'timestamp': datetime.now().isoformat()
    }

    # Define data sources to check
    checks = [
        ('VIX', lambda: yf.Ticker("^VIX").history(period="1d")),
        ('SPY', lambda: yf.Ticker("SPY").history(period="1d")),
        ('HYG', lambda: yf.Ticker("HYG").history(period="1d")),
    ]

    for name, check_func in checks:
        health['total_count'] += 1
        try:
            result = check_func()
            if result is not None and not result.empty:
                health['sources'][name] = {
                    'status': 'ok',
                    'message': 'Data available',
                    'last_date': str(result.index[-1].date()) if hasattr(result.index[-1], 'date') else 'today'
                }
                health['healthy_count'] += 1
            else:
                health['sources'][name] = {
                    'status': 'warning',
                    'message': 'Empty data'
                }
        except Exception as e:
            health['sources'][name] = {
                'status': 'error',
                'message': str(e)[:50]
            }

    # Calculate overall status
    if health['healthy_count'] == health['total_count']:
        health['overall_status'] = 'healthy'
    elif health['healthy_count'] >= health['total_count'] * 0.5:
        health['overall_status'] = 'degraded'
    else:
        health['overall_status'] = 'unhealthy'

    return health


def check_snapshot_freshness(snapshot: Dict) -> Dict[str, Any]:
    """
    Check if dashboard snapshot data is fresh.

    Args:
        snapshot: Latest dashboard snapshot from database

    Returns:
        Dict with freshness status and age
    """
    if not snapshot:
        return {
            'is_fresh': False,
            'age_hours': None,
            'status': 'error',
            'message': 'No snapshot available'
        }

    snapshot_date = snapshot.get('date')
    if not snapshot_date:
        return {
            'is_fresh': False,
            'age_hours': None,
            'status': 'warning',
            'message': 'No date in snapshot'
        }

    try:
        if isinstance(snapshot_date, str):
            snapshot_dt = datetime.strptime(snapshot_date, '%Y-%m-%d')
        else:
            snapshot_dt = snapshot_date

        now = datetime.now()
        age = now - snapshot_dt
        age_hours = age.total_seconds() / 3600

        # Fresh = less than 24 hours old (accounting for weekends)
        is_weekend = now.weekday() >= 5
        threshold_hours = 72 if is_weekend else 24

        if age_hours <= threshold_hours:
            return {
                'is_fresh': True,
                'age_hours': age_hours,
                'status': 'ok',
                'message': f'Updated {age_hours:.0f}h ago'
            }
        else:
            return {
                'is_fresh': False,
                'age_hours': age_hours,
                'status': 'stale',
                'message': f'Data is {age_hours:.0f}h old'
            }
    except Exception as e:
        return {
            'is_fresh': False,
            'age_hours': None,
            'status': 'error',
            'message': f'Error: {str(e)[:30]}'
        }


# =============================================================================
# ALERT THRESHOLDS
# =============================================================================

# Define extreme thresholds for key indicators
ALERT_THRESHOLDS = {
    'vix': {
        'extreme_high': 35,
        'high': 25,
        'low': 12,
        'extreme_low': 10,
        'high_message': '⚠️ VIX elevated - high fear',
        'extreme_high_message': '🚨 VIX extreme - panic levels',
        'low_message': '⚠️ VIX low - complacency',
        'extreme_low_message': '🚨 VIX extreme low - caution warranted'
    },
    'vvix': {
        'extreme_high': 140,
        'high': 120,
        'low': 80,
        'high_message': '🟢 VVIX spike - historic buy signal!',
        'extreme_high_message': '🟢🟢 VVIX extreme - strong contrarian buy!'
    },
    'fear_greed': {
        'extreme_high': 80,
        'high': 70,
        'low': 30,
        'extreme_low': 20,
        'high_message': '⚠️ Extreme Greed - contrarian sell zone',
        'extreme_high_message': '🚨 Max Greed - consider de-risking',
        'low_message': '🟢 Fear zone - contrarian opportunity',
        'extreme_low_message': '🟢🟢 Extreme Fear - historic buy zone!'
    },
    'credit_spread_hy': {
        'extreme_high': 600,
        'high': 450,
        'low': 250,
        'high_message': '⚠️ Credit stress - spreads widening',
        'extreme_high_message': '🚨 Credit crisis levels - risk-off!'
    },
    'skew': {
        'extreme_high': 150,
        'high': 140,
        'low': 115,
        'high_message': '⚠️ High tail risk hedging',
        'extreme_high_message': '🚨 Extreme skew - crash protection bid'
    },
    'put_call_ratio': {
        'extreme_high': 1.3,
        'high': 1.1,
        'low': 0.7,
        'extreme_low': 0.5,
        'high_message': '🟢 High put buying - fear (contrarian bullish)',
        'extreme_high_message': '🟢🟢 Extreme put buying - capitulation!',
        'low_message': '⚠️ Low put protection - complacency',
        'extreme_low_message': '🚨 No hedging - extreme greed'
    },
    'move_index': {
        'extreme_high': 150,
        'high': 120,
        'low': 80,
        'high_message': '⚠️ Treasury stress elevated',
        'extreme_high_message': '🚨 Bond market panic!'
    },
    'market_breadth': {
        'extreme_high': 0.80,
        'high': 0.70,
        'low': 0.35,
        'extreme_low': 0.25,
        'high_message': '🟢 Strong breadth - healthy rally',
        'low_message': '⚠️ Weak breadth - selective market',
        'extreme_low_message': '🚨 Very weak breadth - broad selling'
    }
}


def check_indicator_alert(indicator: str, value: float) -> Optional[Dict]:
    """
    Check if an indicator is at extreme levels.

    Args:
        indicator: Name of indicator (e.g., 'vix', 'fear_greed')
        value: Current value

    Returns:
        Dict with alert info if triggered, None otherwise
    """
    if value is None or indicator not in ALERT_THRESHOLDS:
        return None

    thresholds = ALERT_THRESHOLDS[indicator]

    # Check extreme high
    if 'extreme_high' in thresholds and value >= thresholds['extreme_high']:
        return {
            'level': 'extreme_high',
            'indicator': indicator,
            'value': value,
            'threshold': thresholds['extreme_high'],
            'message': thresholds.get('extreme_high_message', f'{indicator} at extreme high'),
            'severity': 'critical'
        }

    # Check high
    if 'high' in thresholds and value >= thresholds['high']:
        return {
            'level': 'high',
            'indicator': indicator,
            'value': value,
            'threshold': thresholds['high'],
            'message': thresholds.get('high_message', f'{indicator} elevated'),
            'severity': 'warning'
        }

    # Check extreme low
    if 'extreme_low' in thresholds and value <= thresholds['extreme_low']:
        return {
            'level': 'extreme_low',
            'indicator': indicator,
            'value': value,
            'threshold': thresholds['extreme_low'],
            'message': thresholds.get('extreme_low_message', f'{indicator} at extreme low'),
            'severity': 'critical'
        }

    # Check low
    if 'low' in thresholds and value <= thresholds['low']:
        return {
            'level': 'low',
            'indicator': indicator,
            'value': value,
            'threshold': thresholds['low'],
            'message': thresholds.get('low_message', f'{indicator} low'),
            'severity': 'info'
        }

    return None


def get_all_alerts(snapshot: Dict, vrp_data: Dict = None) -> list:
    """
    Check all indicators for alerts.

    Args:
        snapshot: Dashboard snapshot data
        vrp_data: VRP analysis data

    Returns:
        List of triggered alerts, sorted by severity
    """
    alerts = []

    if not snapshot:
        return alerts

    # Map snapshot fields to indicator names
    checks = [
        ('vix', snapshot.get('vix_spot')),
        ('vvix', snapshot.get('vvix')),
        ('fear_greed', snapshot.get('fear_greed_score')),
        ('credit_spread_hy', snapshot.get('credit_spread_hy')),
        ('skew', snapshot.get('skew')),
        ('put_call_ratio', snapshot.get('put_call_ratio')),
        ('move_index', snapshot.get('move_index')),
        ('market_breadth', snapshot.get('market_breadth')),
    ]

    for indicator, value in checks:
        if value is not None:
            alert = check_indicator_alert(indicator, value)
            if alert:
                alerts.append(alert)

    # Sort by severity (critical first)
    severity_order = {'critical': 0, 'warning': 1, 'info': 2}
    alerts.sort(key=lambda x: severity_order.get(x['severity'], 3))

    return alerts


# =============================================================================
# CSV EXPORT HELPERS
# =============================================================================

def prepare_snapshot_for_export(snapshot: Dict) -> pd.DataFrame:
    """
    Prepare snapshot data for CSV export.

    Args:
        snapshot: Dashboard snapshot dict

    Returns:
        DataFrame ready for CSV export
    """
    if not snapshot:
        return pd.DataFrame()

    # Flatten and clean the data
    export_data = {}

    # Core metrics
    export_data['Date'] = snapshot.get('date', '')
    export_data['VIX'] = snapshot.get('vix_spot')
    export_data['VVIX'] = snapshot.get('vvix')
    export_data['VIX9D'] = snapshot.get('vix9d')
    export_data['SKEW'] = snapshot.get('skew')
    export_data['VIX_Contango_%'] = snapshot.get('vix_contango')

    # Credit
    export_data['HY_Spread_bps'] = snapshot.get('credit_spread_hy')
    export_data['IG_Spread_bps'] = snapshot.get('credit_spread_ig')

    # Sentiment
    export_data['Fear_Greed'] = snapshot.get('fear_greed_score')
    export_data['Put_Call_Ratio'] = snapshot.get('put_call_ratio')

    # Breadth
    export_data['Market_Breadth_%'] = snapshot.get('market_breadth')
    if export_data['Market_Breadth_%']:
        export_data['Market_Breadth_%'] *= 100  # Convert to percentage

    # Rates
    export_data['Treasury_10Y_%'] = snapshot.get('treasury_10y')
    export_data['Fed_Funds_%'] = snapshot.get('fed_funds')

    # Phase 2
    export_data['MOVE_Index'] = snapshot.get('move_index')
    export_data['SOFR_%'] = snapshot.get('sofr')
    export_data['RRP_Volume_B'] = snapshot.get('rrp_volume')

    # Signals
    export_data['LEFT_Signal'] = snapshot.get('left_signal')
    export_data['VVIX_Signal'] = snapshot.get('vvix_signal')

    return pd.DataFrame([export_data])


def get_historical_export_data(db_manager, days: int = 90) -> pd.DataFrame:
    """
    Get historical data for CSV export.

    Args:
        db_manager: Database manager instance
        days: Number of days of history

    Returns:
        DataFrame with historical data
    """
    try:
        # Try to get snapshot history
        if hasattr(db_manager, 'get_snapshot_history'):
            return db_manager.get_snapshot_history(days=days)

        # Fallback: get individual indicator histories and merge
        indicators = ['vix_spot', 'vvix', 'fear_greed_score', 'credit_spread_hy']
        all_data = {}

        for indicator in indicators:
            try:
                hist = db_manager.get_indicator_history(indicator, days=days)
                if not hist.empty:
                    all_data[indicator] = hist['value']
            except Exception as e:
                logger.debug(f"Could not fetch {indicator} history: {e}")
                continue

        if all_data:
            return pd.DataFrame(all_data)

        return pd.DataFrame()

    except Exception as e:
        logger.warning(f"Could not get historical export data: {e}")
        return pd.DataFrame()


# =============================================================================
# COMPOSITE RISK SCORE
# =============================================================================

def calculate_composite_risk_score(snapshot: Dict, vrp_data: Dict = None) -> Dict[str, Any]:
    """
    Calculate a composite risk score (0-100) combining multiple indicators.

    Higher score = More risk/caution warranted
    Lower score = More bullish/risk-on

    Components (weighted):
    - VIX level (15%)
    - Credit spreads (20%)
    - Fear & Greed inverted (15%)
    - VRP (10%)
    - Breadth inverted (15%)
    - MOVE Index (10%)
    - SKEW (10%)
    - Put/Call (5%)

    Args:
        snapshot: Dashboard snapshot
        vrp_data: VRP analysis data

    Returns:
        Dict with score, components, and interpretation
    """
    if not snapshot:
        return {'score': None, 'status': 'No data'}

    components = {}
    weights = {}

    # VIX (0-100 scale, higher VIX = higher risk score)
    vix = snapshot.get('vix_spot')
    if vix is not None:
        # VIX typically ranges 10-80, normalize to 0-100
        vix_score = min(100, max(0, (vix - 10) / 50 * 100))
        components['vix'] = vix_score
        weights['vix'] = 0.15

    # Credit Spreads (higher spread = higher risk)
    hy_spread = snapshot.get('credit_spread_hy')
    if hy_spread is not None:
        # HY spread typically 200-800 bps
        spread_score = min(100, max(0, (hy_spread - 200) / 500 * 100))
        components['credit'] = spread_score
        weights['credit'] = 0.20

    # Fear & Greed (INVERTED: low F&G = high fear = lower risk score for contrarians)
    fg = snapshot.get('fear_greed_score')
    if fg is not None:
        # Invert: high greed (80+) = high risk, extreme fear (20) = low risk
        fg_score = fg  # High greed = high score = more risk
        components['sentiment'] = fg_score
        weights['sentiment'] = 0.15

    # VRP (negative VRP = higher risk)
    if vrp_data and vrp_data.get('vrp') is not None:
        vrp = vrp_data['vrp']
        # VRP typically -10 to +15, negative = bad
        vrp_score = min(100, max(0, (10 - vrp) / 20 * 100))
        components['vrp'] = vrp_score
        weights['vrp'] = 0.10

    # Breadth (INVERTED: low breadth = higher risk)
    breadth = snapshot.get('market_breadth')
    if breadth is not None:
        # Breadth 0-1, invert so low breadth = high risk score
        breadth_score = (1 - breadth) * 100
        components['breadth'] = breadth_score
        weights['breadth'] = 0.15

    # MOVE Index (higher = more treasury stress = risk)
    move = snapshot.get('move_index')
    if move is not None:
        # MOVE typically 70-200
        move_score = min(100, max(0, (move - 70) / 100 * 100))
        components['move'] = move_score
        weights['move'] = 0.10

    # SKEW (higher = more tail risk = risk)
    skew = snapshot.get('skew')
    if skew is not None:
        # SKEW typically 110-160
        skew_score = min(100, max(0, (skew - 110) / 40 * 100))
        components['skew'] = skew_score
        weights['skew'] = 0.10

    # Put/Call (INVERTED: high P/C = fear = contrarian bullish = lower risk)
    pc = snapshot.get('put_call_ratio')
    if pc is not None:
        # P/C typically 0.5-1.5, invert for contrarian view
        pc_score = min(100, max(0, (1.5 - pc) / 1.0 * 100))
        components['put_call'] = pc_score
        weights['put_call'] = 0.05

    # Calculate weighted score
    if not components:
        return {'score': None, 'status': 'Insufficient data'}

    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    if total_weight == 0:
        return {'score': None, 'status': 'No valid weights'}

    score = sum(components[k] * weights[k] for k in components) / total_weight
    score = round(score, 1)

    # Interpretation
    if score >= 75:
        interpretation = "HIGH RISK"
        color = "#FF4444"
        description = "Multiple stress indicators elevated. Defensive positioning warranted."
    elif score >= 60:
        interpretation = "ELEVATED"
        color = "#FFA500"
        description = "Some caution warranted. Monitor for deterioration."
    elif score >= 40:
        interpretation = "NEUTRAL"
        color = "#FFD700"
        description = "Mixed signals. Normal market conditions."
    elif score >= 25:
        interpretation = "LOW RISK"
        color = "#90EE90"
        description = "Favorable conditions. Risk-on appropriate."
    else:
        interpretation = "VERY BULLISH"
        color = "#00CC00"
        description = "Extreme fear/pessimism. Strong contrarian buy signals."

    return {
        'score': score,
        'interpretation': interpretation,
        'color': color,
        'description': description,
        'components': components,
        'weights': weights,
        'component_count': len(components),
        'timestamp': datetime.now().isoformat()
    }


# =============================================================================
# TIMESTAMP DISPLAY HELPERS
# =============================================================================

def format_last_updated(timestamp: Optional[datetime] = None, prefix: str = "Last updated") -> str:
    """
    Format a timestamp for display as "Last updated: X min ago".

    Args:
        timestamp: The timestamp to format. If None, uses current time.
        prefix: Text to show before the time (default: "Last updated")

    Returns:
        Formatted string like "Last updated: 5 min ago"
    """
    if timestamp is None:
        return f"{prefix}: just now"

    now = datetime.now()
    if timestamp.tzinfo is not None:
        # Make naive for comparison
        timestamp = timestamp.replace(tzinfo=None)

    delta = now - timestamp

    if delta.total_seconds() < 60:
        return f"{prefix}: just now"
    elif delta.total_seconds() < 3600:
        mins = int(delta.total_seconds() / 60)
        return f"{prefix}: {mins} min ago"
    elif delta.total_seconds() < 86400:
        hours = int(delta.total_seconds() / 3600)
        return f"{prefix}: {hours}h ago"
    else:
        days = int(delta.total_seconds() / 86400)
        return f"{prefix}: {days}d ago"


def section_header_with_timestamp(title: str, timestamp: Optional[datetime] = None,
                                   help_text: Optional[str] = None) -> None:
    """
    Display a section header with an embedded timestamp.

    Args:
        title: Section title
        timestamp: Last update timestamp
        help_text: Optional help tooltip
    """
    import streamlit as st

    time_str = format_last_updated(timestamp) if timestamp else ""

    col1, col2 = st.columns([3, 1])
    with col1:
        if help_text:
            st.subheader(title, help=help_text)
        else:
            st.subheader(title)
    with col2:
        if time_str:
            st.caption(f" {time_str.replace('Last updated: ', '')}")


def data_freshness_badge(timestamp: Optional[datetime], container=None) -> None:
    """
    Display a small freshness badge showing data age.

    Args:
        timestamp: Data timestamp
        container: Streamlit container to use (defaults to st)
    """
    import streamlit as st

    if container is None:
        container = st

    if timestamp is None:
        container.caption("⚪ No timestamp")
        return

    now = datetime.now()
    if timestamp.tzinfo is not None:
        timestamp = timestamp.replace(tzinfo=None)

    delta = now - timestamp
    hours = delta.total_seconds() / 3600

    if hours < 1:
        container.caption(f"🟢 Fresh ({int(delta.total_seconds()/60)}m ago)")
    elif hours < 4:
        container.caption(f"🟡 Recent ({int(hours)}h ago)")
    elif hours < 24:
        container.caption(f"🟠 Stale ({int(hours)}h ago)")
    else:
        container.caption(f"🔴 Old ({int(hours/24)}d ago)")


# =============================================================================
# HISTORICAL COMPARISON
# =============================================================================

# Reference periods for comparison
HISTORICAL_PERIODS = {
    'covid_crash_2020': {
        'name': 'COVID Crash (Mar 2020)',
        'date': '2020-03-23',
        'vix': 82.69,
        'fear_greed': 2,
        'hy_spread': 1100,
        'description': 'Market bottom during pandemic panic'
    },
    'bear_2022_low': {
        'name': '2022 Bear Low (Oct 2022)',
        'date': '2022-10-12',
        'vix': 32.5,
        'fear_greed': 18,
        'hy_spread': 540,
        'description': 'Market bottom during Fed tightening'
    },
    'vix_spike_aug_2024': {
        'name': 'VIX Spike (Aug 2024)',
        'date': '2024-08-05',
        'vix': 65.7,
        'fear_greed': 17,
        'hy_spread': 380,
        'description': 'Yen carry trade unwind panic'
    },
    'pre_covid_high_2020': {
        'name': 'Pre-COVID High (Feb 2020)',
        'date': '2020-02-19',
        'vix': 14.4,
        'fear_greed': 68,
        'hy_spread': 310,
        'description': 'Market top before COVID crash'
    },
    'all_time_high_2021': {
        'name': 'ATH (Dec 2021)',
        'date': '2021-12-29',
        'vix': 17.2,
        'fear_greed': 62,
        'hy_spread': 280,
        'description': 'Market top before 2022 bear'
    }
}


def compare_to_historical(snapshot: Dict) -> Dict[str, Any]:
    """
    Compare current market conditions to historical reference points.

    Args:
        snapshot: Current dashboard snapshot

    Returns:
        Dict with comparisons to each historical period
    """
    if not snapshot:
        return {'comparisons': [], 'closest_match': None}

    current_vix = snapshot.get('vix_spot')
    current_fg = snapshot.get('fear_greed_score')
    current_spread = snapshot.get('credit_spread_hy')

    comparisons = []

    for period_id, period in HISTORICAL_PERIODS.items():
        comparison = {
            'period_id': period_id,
            'name': period['name'],
            'date': period['date'],
            'description': period['description'],
            'differences': {}
        }

        # Calculate differences
        if current_vix and period.get('vix'):
            diff = current_vix - period['vix']
            pct = (diff / period['vix']) * 100
            comparison['differences']['vix'] = {
                'current': current_vix,
                'historical': period['vix'],
                'diff': round(diff, 1),
                'pct': round(pct, 1)
            }

        if current_fg is not None and period.get('fear_greed'):
            diff = current_fg - period['fear_greed']
            comparison['differences']['fear_greed'] = {
                'current': current_fg,
                'historical': period['fear_greed'],
                'diff': round(diff, 1)
            }

        if current_spread and period.get('hy_spread'):
            diff = current_spread - period['hy_spread']
            pct = (diff / period['hy_spread']) * 100
            comparison['differences']['hy_spread'] = {
                'current': current_spread,
                'historical': period['hy_spread'],
                'diff': round(diff, 1),
                'pct': round(pct, 1)
            }

        # Calculate similarity score (lower = more similar)
        similarity = 0
        count = 0

        if 'vix' in comparison['differences']:
            similarity += abs(comparison['differences']['vix']['pct'])
            count += 1

        if 'fear_greed' in comparison['differences']:
            similarity += abs(comparison['differences']['fear_greed']['diff'])
            count += 1

        if 'hy_spread' in comparison['differences']:
            similarity += abs(comparison['differences']['hy_spread']['pct'])
            count += 1

        comparison['similarity_score'] = round(similarity / count, 1) if count > 0 else 999

        comparisons.append(comparison)

    # Sort by similarity (most similar first)
    comparisons.sort(key=lambda x: x['similarity_score'])

    return {
        'comparisons': comparisons,
        'closest_match': comparisons[0] if comparisons else None
    }
