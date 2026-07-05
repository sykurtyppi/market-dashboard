"""
Market Risk Dashboard - PROFESSIONAL EDITION v2

Thin entrypoint: environment setup, navigation, and page dispatch.
Page implementations live in dashboard/views/; shared code in dashboard/core/.
"""

import sys
from pathlib import Path

# Path setup must run before any project imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import streamlit as st

from dashboard.core import setup  # noqa: F401  (dotenv, logging, secrets sync)

st.set_page_config(
    page_title="Market Risk Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    from dashboard.core.init import init_components
    from dashboard.core.sidebar import render_sidebar
    from dashboard.core.styles import inject_custom_css
    from dashboard.views import (
        overview,
    left_strategy,
    sentiment,
    credit_liquidity,
    volatility_vrp,
    sectors_vix,
    market_breadth,
    treasury_stress,
    repo_market,
    cot_positioning,
    cta_flow,
    institutional_flow,
    economic_calendar,
    fed_watch,
    cross_asset,
    options_flow,
    settings,
    system_health,
    )
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Make sure all required files are in their correct folders")
    st.stop()

inject_custom_css()

components = init_components()
if not components:
    st.error("Failed to initialize dashboard.")
    st.stop()

page = render_sidebar(components)

st.title("Market Risk Dashboard")

PAGES = {
    "Overview": overview.render,
    "LEFT Strategy": left_strategy.render,
    "Sentiment": sentiment.render,
    "Credit & Liquidity": credit_liquidity.render,
    "Volatility & VRP": volatility_vrp.render,
    "Sectors & VIX": sectors_vix.render,
    "Market Breadth": market_breadth.render,
    "Treasury Stress (MOVE)": treasury_stress.render,
    "Repo Market (SOFR)": repo_market.render,
    "COT Positioning": cot_positioning.render,
    "CTA Flow Tracker": cta_flow.render,
    "Institutional Flow": institutional_flow.render,
    "Economic Calendar": economic_calendar.render,
    "Fed Watch": fed_watch.render,
    "Cross-Asset": cross_asset.render,
    "Options Flow": options_flow.render,
    "Settings": settings.render,
    "System Health": system_health.render,
}

renderer = PAGES.get(page)
if renderer is not None:
    renderer(components)

st.divider()
st.caption(
    "Market Risk Dashboard | Not financial advice | Data: FRED, CNN, CBOE, Yahoo Finance, Fed, Treasury"
)
