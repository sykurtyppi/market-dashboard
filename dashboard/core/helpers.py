"""Shared formatting, status, and chart helpers for the dashboard."""
import os
from datetime import datetime

import numpy as np
import plotly.graph_objects as go

from utils.data_status import DataStatus, calculate_data_age, get_staleness_status


def get_breadth_mode():
    """Get breadth mode from environment (set in Settings page)."""
    return os.getenv('BREADTH_MODE', 'fast').lower()
def get_mcclellan_scale_factor():
    """Get McClellan scale factor based on breadth mode setting.
    Fast mode (100 stocks): 5.0x scaling
    Full mode (500 stocks): 1.0x (no scaling)
    """
    return 1.0 if get_breadth_mode() == 'full' else 5.0
def parse_timestamp(value):
    """Parse date/timestamp strings or datetime objects into datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            try:
                return datetime.strptime(value, "%Y-%m-%d")
            except ValueError:
                return None
    return None
def status_from_timestamp(value):
    """Return (DataStatus, age_hours) based on a timestamp."""
    ts = parse_timestamp(value)
    if ts is None:
        return DataStatus.UNAVAILABLE, None
    age_hours = calculate_data_age(ts)
    return get_staleness_status(age_hours), age_hours
def format_large_number(value, prefix="$", suffix="B"):
    """Format large numbers with proper suffix"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{prefix}{value:.1f}{suffix}"
def get_status_color(level):
    """Get color for status level"""
    colors = {
        "NORMAL": "#4CAF50",
        "LOW": "#4CAF50",
        "SUPPORTIVE": "#4CAF50",
        "ELEVATED": "#FF9800",
        "NEUTRAL": "#FF9800",
        "STRESS": "#F44336",
        "HIGH": "#F44336",
        "DRAINING": "#F44336",
    }
    return colors.get(level, "#9E9E9E")
def create_gauge_chart(value, title, max_value=100, color="#1f77b4"):
    """Create a gauge chart"""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title, "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, max_value]},
                "bar": {"color": color},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, max_value / 3], "color": "lightgray"},
                    {"range": [max_value / 3, 2 * max_value / 3], "color": "lightgray"},
                    {"range": [2 * max_value / 3, max_value], "color": "lightgray"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": max_value * 0.8,
                },
            },
        )
    )

    fig.update_layout(height=200, margin=dict(l=10, r=10, t=40, b=10))
    return fig
