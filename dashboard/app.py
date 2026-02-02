"""
Market Risk Dashboard - PROFESSIONAL EDITION v2
+ Phase 2: Enhanced Liquidity, Treasury Stress, and Repo Market
Fully integrated app.py (Phase 1 + Phase 2)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
from pathlib import Path
import os
import yfinance as yf
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# PATH / ENV SETUP
# -------------------------------------------------------------------
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from dotenv import load_dotenv
load_dotenv()

# -------------------------------------------------------------------
# IMPORT PROJECT MODULES
# -------------------------------------------------------------------
try:
    # Database / health
    from database.db_manager import DatabaseManager
    from database.health_check import HealthCheckSystem, HealthStatus

    # Collectors - Phase 1
    from data_collectors.fred_collector import FREDCollector
    from data_collectors.fear_greed_collector import FearGreedCollector
    from data_collectors.cboe_collector import CBOECollector
    from data_collectors.sp500_adline_calculator import SP500ADLineCalculator
    
    from data_collectors.yahoo_collector import YahooCollector
    # removed old import SP500ADLineCalculator
    from data_collectors.liquidity_collector import LiquidityCollector
    
    # PDF Chart builders
    from dashboard.pdf_chart_builder import create_vrp_chart, create_credit_spreads_chart, create_liquidity_chart
    from dashboard.pdf_chart_builder import create_vix_term_structure_chart, create_adline_chart, create_mcclellan_chart, create_treasury_stress_chart

    # Collectors - Phase 2
    from data_collectors.market_data import MarketDataCollector
    from data_collectors.fed_balance_sheet_collector import FedBalanceSheetCollector
    from data_collectors.move_collector import MOVECollector
    from data_collectors.repo_collector_enhanced import RepoCollector
    from data_collectors.cot_collector import COTCollector

    # Processors
    from processors.left_strategy import LEFTStrategy
    from processors.vrp_module import VRPAnalyzer           # VRP analysis
    from processors.liquidity_signals import (
        LiquidityAnalyzer,
        NetLiquiditySignal,
    )
    from processors.qt_analyzer import QTAnalyzer
    from processors.treasury_liquidity_analyzer import TreasuryLiquidityAnalyzer
    from processors.repo_analyzer import RepoAnalyzer

    # Enhanced Breadth Analysis (institutional-grade)
    from processors.breadth_enhanced import (
        EnhancedBreadthAnalyzer,
        get_new_highs_lows,
    )

    # Settings page
    from settings_page import render_settings_page
    # PDF Export
    from dashboard.pdf_generator_v2 import PDFReportGenerator
    # UI Helpers
    from dashboard.ui_helpers import (
        get_vix_percentile,
        get_fear_greed_percentile,
        get_credit_spread_percentile,
        get_estimated_indicator,
        format_with_estimated,
        format_percentile_badge,
        get_tooltip,
        METRIC_TOOLTIPS,
    )

except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Make sure all required files are in their correct folders")
    st.stop()

# -------------------------------------------------------------------
# CONSTANTS & HELPERS
# -------------------------------------------------------------------
def get_breadth_mode():
    """Get breadth mode from environment (set in Settings page)."""
    return os.getenv('BREADTH_MODE', 'fast').lower()

def get_mcclellan_scale_factor():
    """Get McClellan scale factor based on breadth mode setting.
    Fast mode (100 stocks): 5.0x scaling
    Full mode (500 stocks): 1.0x (no scaling)
    """
    return 1.0 if get_breadth_mode() == 'full' else 5.0

# Dynamic scale factor based on mode
MCCLELLAN_SCALE_FACTOR = get_mcclellan_scale_factor()

# -------------------------------------------------------------------
# PAGE CONFIG + CSS
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Market Risk Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .status-good { color: #4CAF50; font-weight: bold; }
    .status-warning { color: #FF9800; font-weight: bold; }
    .status-bad { color: #F44336; font-weight: bold; }
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# INITIALIZATION
# -------------------------------------------------------------------
@st.cache_resource
def init_components():
    try:
        # FRED is optional
        try:
            fred_collector = FREDCollector()
        except Exception as fred_error:
            st.warning(f"FRED API not configured: {fred_error}")
            st.info("You can add your FRED API key in Settings page")
            fred_collector = None

        components = {
            # Core DB / health
            "db": DatabaseManager(),
            "health": HealthCheckSystem(),

            # Phase 1 collectors
            "fred": fred_collector,
            "fear_greed": FearGreedCollector(),
            "cboe": CBOECollector(),
            "breadth": SP500ADLineCalculator(mode=get_breadth_mode()),
            "yahoo": YahooCollector(),
            "left_strategy": LEFTStrategy(),
            "vrp": VRPAnalyzer(lookback_days=21),

            # Liquidity collectors & analyzers
            "liquidity": LiquidityCollector(),
            "liquidity_analyzer": LiquidityAnalyzer(),  # Now uses correct formula: Fed BS - TGA - RRP

            # Phase 2 collectors & processors
            "market": MarketDataCollector(),
            "fed_bs": FedBalanceSheetCollector(),
            "move": MOVECollector(),
            "repo": RepoCollector(),
            "qt_analyzer": QTAnalyzer(),
            "treasury_analyzer": TreasuryLiquidityAnalyzer(),
            "repo_analyzer": RepoAnalyzer(),
        }

        return components
    except Exception as e:
        st.error(f"Failed to initialize: {e}")
        return None


components = init_components()
if not components:
    st.error("Failed to initialize dashboard.")
    st.stop()

# -------------------------------------------------------------------
# HELPER FUNCTIONS (PHASE 1)
# -------------------------------------------------------------------
@st.cache_data(ttl=300)
def get_vrp_analysis_cached():
    """Cached wrapper for VRP analysis (5 min TTL)"""
    try:
        yahoo = YahooCollector()
        vix = yahoo.get_vix()
        analyzer = VRPAnalyzer(lookback_days=21)
        
        if vix is not None:
            return analyzer.get_complete_analysis(vix=vix)
        else:
            return analyzer.get_complete_analysis()
    except Exception as e:
        logger.error(f"Failed to get VRP analysis: {e}")
        return {"error": str(e)}


@st.cache_data(ttl=300)
def get_vrp_history_cached(days: int = 180):
    """Cached wrapper for VRP historical data (5 min TTL)"""
    try:
        analyzer = VRPAnalyzer(lookback_days=21)
        history = analyzer.get_historical_vrp(days=days)
        
        if history.empty:
            logger.warning(f"VRP history returned empty for {days} days")
        else:
            logger.info(f"Successfully loaded VRP history: {len(history)} rows")
        
        return history
    except Exception as e:
        logger.error(f"Failed to get VRP history: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_vix_term_structure():
    """Approximate VIX term structure using VIX and VIX ETFs."""
    try:
        vix = yf.Ticker("^VIX").history(period="1d")["Close"].iloc[-1]
        vixy = yf.Ticker("VIXY").history(period="1d")["Close"].iloc[-1]
        vxz = yf.Ticker("VXZ").history(period="1d")["Close"].iloc[-1]

        term_structure = {
            "Spot": float(vix),
            "1-Month": float(vix * 1.02),
            "2-Month": float(vix * (vixy / 20)),
            "3-Month": float(vix * (vxz / 20)),
            "4-Month": float(vix * (vxz / 20) * 1.03),
        }

        return pd.DataFrame(list(term_structure.items()), columns=["Maturity", "VIX Level"])
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_sector_performance(period: str = "1d") -> pd.DataFrame:
    """Get sector ETF performance for different time periods."""
    sectors = {
        "XLK": "Technology",
        "XLF": "Financials",
        "XLV": "Healthcare",
        "XLE": "Energy",
        "XLY": "Consumer Disc",
        "XLP": "Consumer Staples",
        "XLI": "Industrials",
        "XLB": "Materials",
        "XLU": "Utilities",
        "XLRE": "Real Estate",
        "XLC": "Communication",
    }

    performance = []
    period_days = {
        "1d": 2,
        "5d": 6,
        "1mo": 30,
        "3mo": 90,
        "6mo": 180,
        "1y": 365,
        "5y": 1825,
        "ytd": (datetime.now() - datetime(datetime.now().year, 1, 1)).days + 5,
    }

    days = period_days.get(period, 2)

    for ticker, name in sectors.items():
        try:
            etf = yf.Ticker(ticker)
            hist = etf.history(period=f"{days}d")

            if len(hist) >= 2:
                start_price = hist["Close"].iloc[0]
                end_price = hist["Close"].iloc[-1]
                change = ((end_price / start_price) - 1) * 100
                returns = hist["Close"].pct_change().dropna()
                volatility = returns.std() * 100

                performance.append({
                    "Sector": name,
                    "Ticker": ticker,
                    "Change %": float(change),
                    "Price": float(end_price),
                    "Volatility": float(volatility),
                    "Start Price": float(start_price),
                })
        except Exception as e:
            logger.warning(f"Error fetching {ticker}: {e}")
            continue

    df = pd.DataFrame(performance)
    if not df.empty:
        df = df.sort_values("Change %", ascending=False)
    return df


@st.cache_data(ttl=300)
def get_sector_comparison_chart(period: str = "1y") -> pd.DataFrame:
    """Get historical sector performance for comparison (normalized base 100)."""
    sectors = {
        "XLK": "Technology",
        "XLF": "Financials",
        "XLV": "Healthcare",
        "XLE": "Energy",
        "XLY": "Consumer Disc",
        "XLP": "Consumer Staples",
        "XLI": "Industrials",
        "XLB": "Materials",
        "XLU": "Utilities",
        "XLRE": "Real Estate",
        "XLC": "Communication",
    }

    all_data = {}

    for ticker, name in sectors.items():
        try:
            etf = yf.Ticker(ticker)
            hist = etf.history(period=period)

            if not hist.empty:
                normalized = (hist["Close"] / hist["Close"].iloc[0]) * 100
                all_data[name] = normalized
        except Exception:
            continue

    if all_data:
        return pd.DataFrame(all_data)
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_breadth_metrics_cached():
    """Cached wrapper around breadth analysis."""
    from processors.breadth_signals import get_comprehensive_breadth_signals
    from database.db_manager import DatabaseManager
    import yfinance as yf
    
    db = DatabaseManager()
    breadth_history = db.get_breadth_history(days=90)
    
    if breadth_history.empty:
        return None
    
    # Get SPY for divergence detection
    spy = yf.Ticker('SPY')
    spy_data = spy.history(period='90d')
    spy_price = spy_data['Close'] if not spy_data.empty else None
    
    # Get comprehensive signals
    signals = get_comprehensive_breadth_signals(breadth_history, spy_price)
    return signals

# -------------------------------------------------------------------
# HELPER FUNCTIONS (PHASE 2)
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# SIDEBAR - NAVIGATION
# -------------------------------------------------------------------
with st.sidebar:
    st.title("Market Dashboard")

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "LEFT Strategy",
            "Sentiment",
            "Credit & Liquidity",
            "CTA Flow Tracker",
            "COT Positioning",         # CFTC institutional positioning
            "Volatility & VRP",
            "Sectors & VIX",
            "Market Breadth",
            "Treasury Stress (MOVE)",  # Phase 2
            "Repo Market (SOFR)",      # Phase 2
            "Settings",
        ],
    )

    st.divider()

    # Get latest data WITH age tracking
    latest = components["db"].get_latest_snapshot(include_age=True)
    if latest:
        age_status = latest.get('_status', 'unknown')
        age_string = latest.get('_age_string', 'unknown')
        is_fresh = latest.get('_is_fresh', False)

        if is_fresh:
            st.success(f"✅ Data: {latest['date']} ({age_string})")
        elif age_status == 'stale':
            st.warning(f"⚠️ Data: {latest['date']} ({age_string})")
        else:
            st.error(f"🚨 Data: {latest['date']} ({age_string})")
            st.caption("Data may be outdated - click 'Update Data' below")
    else:
        st.error("❌ No data available")

    # Health Check Widget
    st.divider()
    st.subheader("System Health")

    with st.spinner("Checking..."):
        health_summary = components["health"].get_health_summary()
        overall_status = HealthStatus(health_summary["overall_status"])

        status_emoji = components["health"].get_status_emoji(overall_status)
        status_color = components["health"].get_status_color(overall_status)

        st.markdown(
            f"<div style='padding: 10px; background-color: {status_color}20; border-left: 4px solid {status_color}; border-radius: 4px;'>"
            f"<strong>{status_emoji} {overall_status.value.title()}</strong>"
            f"</div>",
            unsafe_allow_html=True,
        )

        with st.expander(" Data Source Status"):
            st.markdown("**Yahoo Finance APIs:**")
            yahoo_health = components["yahoo"].get_health_check()

            for source, status in yahoo_health.items():
                if source != "timestamp":
                    emoji = "✅" if status == "ok" else "❌"
                    st.text(f"{emoji} {source.replace('_', ' ').title()}")

            st.divider()

            st.markdown("**Data Sources:**")
            for name, check_data in health_summary["sources"].items():
                status = HealthStatus(check_data["status"])
                emoji = components["health"].get_status_emoji(status)

                st.text(f"{emoji} {check_data['name']}")

                if "age_hours" in check_data and check_data["age_hours"] is not None:
                    st.caption(f"   Last updated: {check_data['age_hours']:.1f}h ago")

                if status in [HealthStatus.DEGRADED, HealthStatus.DOWN]:
                    if "error" in check_data and check_data["error"]:
                        st.caption(f"   ⚠️ {check_data['error'][:50]}...")

    if st.button("Update Data", width='stretch'):
        with st.spinner("Updating data..."):
            try:
                from scheduler.daily_update import MarketDataUpdater
                updater = MarketDataUpdater()
                updater.run_full_update()
                st.success("Update complete.")
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Update failed: {e}")

    if st.button(" Clear Cache", width='stretch'):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache cleared! Refresh page.")
        st.rerun()
    # PDF Export Button
    st.divider()
    st.subheader(" Export Report")
    
    if st.button(" Generate PDF", width='stretch'):
        with st.spinner("Generating PDF with charts (30-60 sec)..."):
            try:
                snapshot = components["db"].get_latest_snapshot()
                vrp_data = components["db"].get_latest_vrp()
                
                charts = {}
                
                # VRP Chart
                try:
                    vrp_history = get_vrp_history_cached(days=180)
                    if not vrp_history.empty:
                        charts['VRP_Analysis'] = create_vrp_chart(vrp_history)
                except Exception as e:
                    logger.error(f"VRP chart: {e}")
                
                # Credit Spreads
                try:
                    hy = components["db"].get_indicator_history("credit_spread_hy", days=365)
                    ig = components["db"].get_indicator_history("credit_spread_ig", days=365)
                    if not hy.empty or not ig.empty:
                        charts['Credit_Spreads'] = create_credit_spreads_chart(hy, ig)
                except Exception as e:
                    logger.error(f"Credit chart: {e}")
                
                # Liquidity
                try:
                    liq = components["liquidity"].get_liquidity_history(lookback_days=365)
                    if not liq.empty:
                        charts['Net_Liquidity'] = create_liquidity_chart(liq)
                except Exception as e:
                    logger.error(f"Liquidity chart: {e}")
                
                # VIX Term Structure - fetch directly from CBOE
                try:
                    from data_collectors.cboe_collector import CBOECollector
                    cboe = CBOECollector()
                    
                    vix_data = {
                        'vix': cboe.get_vix(),  # Fetch directly
                        'vix9d': cboe.get_vix9d(),  # Fetch VIX9D
                        'vix3m': cboe.get_vix3m(),  # Fetch directly
                        'vix6m': None
                    }
                    logger.info(f"VIX data for chart: vix={vix_data['vix']}, vix3m={vix_data['vix3m']}")
                    
                    vix_chart = create_vix_term_structure_chart(vix_data)
                    if vix_chart is not None:
                        charts['VIX_Term_Structure'] = vix_chart
                        logger.info(" VIX term structure chart created")
                    else:
                        logger.warning("VIX term structure returned None - insufficient data")
                except Exception as e:
                    logger.error(f"VIX term: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                
                # Breadth Analysis Charts
                try:
                    breadth_history = components["db"].get_breadth_history(days=90)
                    if not breadth_history.empty:
                        # Ensure McClellan is calculated (scale factor depends on breadth mode)
                        if 'mcclellan' not in breadth_history.columns:
                            sf = get_mcclellan_scale_factor()
                            breadth_history['ema19'] = breadth_history['ad_diff'].ewm(span=19, adjust=False).mean() * sf
                            breadth_history['ema39'] = breadth_history['ad_diff'].ewm(span=39, adjust=False).mean() * sf
                            breadth_history['mcclellan'] = breadth_history['ema19'] - breadth_history['ema39']

                        charts['AD_Line'] = create_adline_chart(breadth_history)
                        charts['McClellan'] = create_mcclellan_chart(breadth_history)
                except Exception as e:
                    logger.error(f"Breadth charts: {e}")
                
                # Treasury Stress (MOVE)
                try:
                    move_history = components["db"].get_indicator_history("move_index", days=180)
                    if not move_history.empty:
                        charts['Treasury_Stress'] = create_treasury_stress_chart(move_history)
                except Exception as e:
                    logger.error(f"MOVE chart: {e}")
                
                # Generate insights
                insights = []
                if snapshot and vrp_data:
                    fg = snapshot.get('fear_greed_score', 50)
                    vrp = vrp_data.get('vrp', 0)
                    hy_spread = snapshot.get('credit_spread_hy', 0) * 100
                    breadth = snapshot.get('market_breadth', 0) * 100
                    
                    if fg < 25 and vrp > 0:
                        insights.append(
                            f"Extreme Fear ({fg:.0f}) + positive VRP ({vrp:+.2f}) = contrarian buy signal. "
                            "Historically, readings below 25 precede strong rallies."
                        )
                    if hy_spread < 300:
                        insights.append(f"Credit spreads tight (HYG {hy_spread:.0f} bps) - no systemic stress.")
                    if breadth > 70:
                        insights.append(f"Strong breadth ({breadth:.1f}%) = healthy continuation signal.")
                
                # Generate PDF
                generator = PDFReportGenerator()
                pdf_path = generator.generate_report(
                    snapshot=snapshot,
                    vrp_data=vrp_data,
                    charts=charts,
                    insights=insights if insights else None
                )
                
                # Download button
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="Download PDF",
                        data=f,
                        file_name=f"market_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf",
                        width='stretch'
                    )
                
                st.success(f"✅ Generated with {len(charts)} charts!")
                
            except Exception as e:
                st.error(f"Error: {e}")
                logger.error(f"PDF error: {e}", exc_info=True)

# -------------------------------------------------------------------
# MAIN TITLE
# -------------------------------------------------------------------
st.title("Market Risk Dashboard")

# ===================================================================
# PAGES
# ===================================================================

# ============================================================
# OVERVIEW
# ============================================================

@st.cache_resource
def get_cta_collector():
    """Cached CTA collector to avoid multiple DB connections"""
    from data_collectors.cta_collector import CTACollector
    return CTACollector(db_path="data/cta_prices.db")


def save_to_env(key: str, value: str):
    """Save or update a key-value pair in .env file"""
    env_path = Path('.env')
    
    if env_path.exists():
        with open(env_path, 'r') as f:
            lines_env = f.readlines()
        
        found = False
        for i, line in enumerate(lines_env):
            if line.startswith(f'{key}='):
                lines_env[i] = f'{key}={value}\n'
                found = True
                break
        
        if not found:
            lines_env.append(f'{key}={value}\n')
        
        with open(env_path, 'w') as f:
            f.writelines(lines_env)
    else:
        with open(env_path, 'w') as f:
            f.write(f'{key}={value}\n')



if page == "Overview":
    snapshot = components["db"].get_latest_snapshot()

    if not snapshot:
        st.warning("No data available. Run: python scheduler/daily_update.py")
        st.code("python scheduler/daily_update.py")
        st.stop()
    
    # ========== REGIME SUMMARY BANNER ==========
    st.markdown("###  Market Regime Summary")
    
    # Get VRP data for volatility regime
    vrp_data = components["db"].get_latest_vrp()
    
    # Build regime components
    regime_parts = []
    
    # 1. Credit / Macro
    if snapshot.get('credit_spread_hy'):
        hy_spread = snapshot['credit_spread_hy']
        # hy_spread is in bps (e.g. 350 = 3.5%)
        if hy_spread < 300:
            credit_status = "🟢 Supportive"
        elif hy_spread < 450:
            credit_status = "🟡 Neutral"
        else:
            credit_status = "🔴 Risk-Off"
        regime_parts.append(f"**Credit:** {credit_status}")
    
    # 2. Volatility Regime
    if vrp_data:
        vrp = vrp_data.get('vrp', 0)
        regime = vrp_data.get('regime', 'Unknown')
        
        if vrp > 8:
            vol_status = f"🟢 {regime} / VRP High"
        elif vrp > 4:
            vol_status = f"🟡 {regime} / VRP Moderate"
        elif vrp > 0:
            vol_status = f"🟡 {regime} / VRP Positive"
        else:
            vol_status = f"🔴 {regime} / VRP Negative"
        
        regime_parts.append(f"**Vol:** {vol_status}")
    
    # 3. Sentiment
    if snapshot.get('fear_greed_score'):
        fg_score = snapshot['fear_greed_score']
        if fg_score < 25:
            sentiment = "🔴 Extreme Fear (Buy Signal)"
        elif fg_score < 45:
            sentiment = "🟡 Fear"
        elif fg_score < 55:
            sentiment = "⚪ Neutral"
        elif fg_score < 75:
            sentiment = "🟢 Greed"
        else:
            sentiment = "🔴 Extreme Greed (Caution)"
        
        regime_parts.append(f"**Sentiment:** {sentiment}")
    
    # Display banner
    if regime_parts:
        banner_text = " | ".join(regime_parts)
        st.info(banner_text)
    
    st.divider()
    # ========== END REGIME BANNER ==========

    st.subheader("Key Market Indicators")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        val = snapshot.get("credit_spread_hy")
        st.metric(
            "HY Spread",
            f"{val:.2f}%" if val is not None else "N/A",
            help=get_tooltip('credit_spread_hy')
        )
        if val is not None:
            credit_pct = get_credit_spread_percentile(val * 100)  # Convert to bps
            if credit_pct and credit_pct.get('context'):
                st.caption(credit_pct['context'])

    with col2:
        val = snapshot.get("treasury_10y")
        st.metric(
            "10Y Treasury",
            f"{val:.2f}%" if val is not None else "N/A",
            help="Benchmark risk-free rate. Rising = tighter financial conditions."
        )

    with col3:
        val = snapshot.get("fear_greed_score")
        st.metric(
            "Fear & Greed",
            f"{val:.0f}" if val is not None else "N/A",
            help=get_tooltip('fear_greed')
        )
        if val is not None:
            fg_ctx = get_fear_greed_percentile(val)
            if fg_ctx:
                if fg_ctx.get('is_extreme'):
                    st.caption(f"⚡ {fg_ctx['context']}")
                else:
                    st.caption(fg_ctx['context'])

    with col4:
        val = snapshot.get("left_signal")
        signal_colors = {"BUY": "🟢", "SELL": "🔴", "NEUTRAL": "⚪"}
        signal_icon = signal_colors.get(val, "")
        st.metric(
            "LEFT Signal",
            f"{signal_icon} {val}" if val else "N/A",
            help=get_tooltip('left_signal')
        )

    st.divider()

    st.subheader("Options & Volatility Metrics")
    col1, col2, col3, col4 = st.columns(4)

    # Get fresh CBOE data (with proper error handling)
    fresh_cboe = {}
    try:
        cboe_fresh = CBOECollector()
        fresh_cboe = cboe_fresh.get_all_data()

        if fresh_cboe:
            logging.info(
                f"Fresh CBOE data: VIX={fresh_cboe.get('vix_spot')}, "
                f"VIX3M={fresh_cboe.get('vix3m')}, "
                f"Contango={fresh_cboe.get('vix_contango')}"
            )

            # Display prominent warning banner if any data is estimated
            if fresh_cboe.get('has_estimated_data'):
                estimated = fresh_cboe.get('estimated_fields', [])
                estimated_names = [e['field'] for e in estimated]

                # Show inline warning banner (always visible)
                st.markdown(
                    f"""<div style='padding: 8px 12px; background: #FFF3E0; border-left: 4px solid #FF9800;
                    border-radius: 4px; margin-bottom: 10px;'>
                    <strong>📊 Estimated Data:</strong> {', '.join(estimated_names)}
                    <span style='color: #666; font-size: 0.85em;'> (not real-time)</span>
                    </div>""",
                    unsafe_allow_html=True
                )

                with st.expander("ℹ️ View estimation details", expanded=False):
                    for est in estimated:
                        st.markdown(f"- **{est['field']}**: {est['reason']}")
                    st.caption("For institutional-grade data, consider professional data feeds (Bloomberg, Refinitiv)")

    except Exception as e:
        st.warning(f"Could not fetch fresh CBOE data: {e}")
        fresh_cboe = {}

    with col1:
        vix = fresh_cboe.get("vix_spot") or snapshot.get("vix_spot")
        st.metric(
            "VIX Spot",
            f"{vix:.2f}" if vix is not None else "N/A",
            help=get_tooltip('vix')
        )
        # Add percentile context
        if vix is not None:
            vix_pct = get_vix_percentile(vix, lookback_days=252)
            if vix_pct:
                badge = format_percentile_badge(vix_pct['percentile'], 252)
                st.caption(f"📊 {badge}")

    with col2:
        contango = fresh_cboe.get("vix_contango")
        is_estimated, est_reason = get_estimated_indicator('vix_contango', fresh_cboe.get('estimated_fields', []))

        if contango is not None:
            # Use ~ prefix for estimated values
            display_val = f"~{contango:+.2f}%" if is_estimated else f"{contango:+.2f}%"
            st.metric(
                "VIX Contango",
                display_val,
                help=get_tooltip('vix_contango')
            )
            if contango > 0:
                st.caption("📈 Contango (bullish)")
            else:
                st.caption("📉 Backwardation (fear)")
        else:
            contango = snapshot.get("vix_contango")
            if contango is not None:
                st.metric("VIX Contango", f"{contango:+.2f}%", help=get_tooltip('vix_contango'))
                st.caption("📦 Cached data")
            else:
                st.metric("VIX Contango", "N/A")

    with col3:
        load_dotenv()
        manual_pcce = os.getenv('MANUAL_PCCE', '0.0')
        manual_pcce_date = os.getenv('MANUAL_PCCE_DATE', '')
        
        try:
            manual_pcce_value = float(manual_pcce) if manual_pcce else 0.0
        except Exception:
            manual_pcce_value = 0.0
        
        if manual_pcce_value > 0:
            st.metric("Equity Put/Call (PCCE)", f"{manual_pcce_value:.3f}")
            st.caption(f"✅ Manual ({manual_pcce_date or 'Today'})")
            
            if manual_pcce_value > 1.0:
                st.caption("Bearish (high P/C)")
            elif manual_pcce_value < 0.7:
                st.caption("Bullish (low P/C)")
            else:
                st.caption("Neutral range")
        else:
            pc_ratios = fresh_cboe.get("put_call_ratios", {})
            equity_pc = pc_ratios.get("equity_pc")
            
            if equity_pc is not None:
                st.metric("Equity P/C (Estimated)", f"{equity_pc:.2f}")
                st.caption(" VIX/VXV proxy")
                
                if equity_pc > 1.0:
                    st.caption("Bearish (high)")
                elif equity_pc < 0.7:
                    st.caption("Bullish (low)")
                else:
                    st.caption("Neutral")
            else:
                pc = snapshot.get("put_call_ratio")
                if pc is not None:
                    st.metric("Equity P/C (Cached)", f"{pc:.2f}")
                    st.caption("⚠️ Stale data")
                    st.caption("Update or set manual")
                else:
                    st.metric("Equity Put/Call", "N/A")
                    st.caption("Set in Settings →")

    with col4:
        breadth = snapshot.get("market_breadth")
        if breadth is not None:
            pct = breadth * 100 if breadth <= 1 else breadth
            st.metric("Market Breadth", f"{pct:.1f}%")
            if pct > 60:
                st.caption("Strong participation")
            elif pct < 40:
                st.caption("Weak participation")
        else:
            st.metric("Market Breadth", "N/A")

    st.subheader("Advanced Volatility Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        vix9d = snapshot.get("vix9d")
        vix = snapshot.get("vix_spot")
        if vix9d is not None:
            st.metric(
                "VIX9D",
                f"{vix9d:.2f}",
                help="9-day implied vol. Spread vs VIX shows near-term event risk."
            )
            if vix is not None:
                spread = vix9d - vix
                spread_pct = (spread / vix) * 100
                if spread_pct < -10:
                    st.caption(f"✅ {spread_pct:+.0f}% calm")
                elif spread_pct > 10:
                    st.caption(f"⚠️ {spread_pct:+.0f}% event risk")
                else:
                    st.caption(f"{spread_pct:+.0f}% normal")
        else:
            st.metric("VIX9D", "N/A")

    with col2:
        vvix = fresh_cboe.get("vvix") if fresh_cboe else None
        if vvix is not None:
            if vvix >= 120:
                st.metric("🎯 VVIX", f"{vvix:.1f}", delta="BUY", help=get_tooltip('vvix'))
                st.caption("🟢 Historic buy zone")
            elif vvix >= 110:
                st.metric("VVIX", f"{vvix:.1f}", help=get_tooltip('vvix'))
                st.caption("🟡 Elevated")
            elif vvix >= 80:
                st.metric("VVIX", f"{vvix:.1f}", help=get_tooltip('vvix'))
                st.caption("Normal")
            else:
                st.metric("VVIX", f"{vvix:.1f}", help=get_tooltip('vvix'))
                st.caption("🟠 Complacent")
        else:
            st.metric("VVIX", "N/A")

    with col3:
        skew = snapshot.get("skew")
        if skew is not None:
            st.metric("SKEW", f"{skew:.0f}", help=get_tooltip('skew'))
            if skew > 150:
                st.caption("🔴 Extreme hedging")
            elif skew > 145:
                st.caption("🟡 Elevated")
            elif skew > 130:
                st.caption("Normal")
            else:
                st.caption("Low protection")
        else:
            st.metric("SKEW", "N/A")

    with col4:
        vrp = snapshot.get("vrp")
        if vrp is not None:
            st.metric("VRP", f"{vrp:+.1f}", help=get_tooltip('vrp'))
            if vrp > 5:
                st.caption("🟢 Options expensive")
            elif vrp > 0:
                st.caption("Positive premium")
            else:
                st.caption("🔴 Negative (risk)")
        else:
            st.metric("VRP", "N/A")

    with col5:
        vix3m = fresh_cboe.get("vix3m")
        is_vix3m_est, _ = get_estimated_indicator('vix3m', fresh_cboe.get('estimated_fields', []))

        if vix is not None and vix3m:
            slope = vix3m - vix
            slope_per_day = slope / 63
            display_val = f"~{slope_per_day:+.3f}" if is_vix3m_est else f"{slope_per_day:+.3f}"
            st.metric(
                "Term Slope",
                display_val,
                help="VIX term structure slope. Positive = contango (normal)."
            )
            if slope_per_day > 0.05:
                st.caption("🟢 Steep contango")
            elif slope_per_day > 0:
                st.caption("Normal")
            else:
                st.caption("🔴 Inverted")
        else:
            st.metric("Term Slope", "N/A")
    
    st.divider()
    signal = snapshot.get("left_signal")
    if signal:
        st.subheader(f"Current Signal: {signal}")

# ============================================================
# LEFT STRATEGY
# ============================================================
elif page == "LEFT Strategy":
    st.header("LEFT Strategy Analysis")

    try:
        if components["fred"] is None:
            st.warning("FRED API not configured. LEFT Strategy needs HYG OAS from FRED.")
        else:
            hyg_data = components["fred"].get_series("BAMLH0A0HYM2", start_date="2023-01-01")

            if not hyg_data.empty:
                signals = components["left_strategy"].calculate_signal(hyg_data)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Signal", signals["signal"])
                with col2:
                    st.metric("Strength", f"{signals['strength']:.1f}/100")
                with col3:
                    st.metric("From EMA", f"{signals['pct_from_ema']:+.2f}%")

                st.divider()

                historical = components["left_strategy"].get_historical_signals(hyg_data)

                if not historical.empty:
                    fig = go.Figure()

                    fig.add_trace(
                        go.Scatter(
                            x=historical["date"],
                            y=historical["BAMLH0A0HYM2"],
                            mode="lines",
                            name="HYG OAS",
                            line=dict(color="royalblue", width=2),
                        )
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=historical["date"],
                            y=historical["ema_330"],
                            mode="lines",
                            name="330-Day EMA",
                            line=dict(color="orange", width=2, dash="dash"),
                        )
                    )

                    fig.update_layout(
                        title="Credit Spreads vs EMA",
                        xaxis_title="Date",
                        yaxis_title="Spread (%)",
                        height=500,
                        hovermode="x unified",
                    )

                    st.plotly_chart(fig, width='stretch')
            else:
                st.warning("No HYG OAS data from FRED.")
    except Exception as e:
        st.error(f"Error: {e}")

# ============================================================
# SENTIMENT
# ============================================================
elif page == "Sentiment":
    st.header("Market Sentiment")

    try:
        fg_data = components["fear_greed"].get_fear_greed_score()

        if fg_data:
            score = fg_data["score"]

            if score < 25:
                color, label = "red", "EXTREME FEAR"
            elif score < 45:
                color, label = "orange", "FEAR"
            elif score < 55:
                color, label = "yellow", "NEUTRAL"
            elif score < 75:
                color, label = "lightgreen", "GREED"
            else:
                color, label = "green", "EXTREME GREED"

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=score,
                    title={"text": f"Fear & Greed<br>{label}"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": color},
                        "steps": [
                            {"range": [0, 25], "color": "rgba(255,0,0,0.2)"},
                            {"range": [25, 45], "color": "rgba(255,165,0,0.2)"},
                            {"range": [45, 55], "color": "rgba(255,255,0,0.2)"},
                            {"range": [55, 75], "color": "rgba(144,238,144,0.2)"},
                            {"range": [75, 100], "color": "rgba(0,128,0,0.2)"},
                        ],
                    },
                )
            )

            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current", f"{score:.0f}")
            with col2:
                if fg_data.get("previous_close") is not None:
                    st.metric("Yesterday", f"{fg_data['previous_close']:.0f}")
            with col3:
                if fg_data.get("one_week_ago") is not None:
                    st.metric("Last Week", f"{fg_data['one_week_ago']:.0f}")
        else:
            st.warning("No Fear & Greed data available.")
    except Exception as e:
        st.error(f"Error: {e}")

    # ============================================================
    # PUT/CALL RATIOS
    # ============================================================
    st.divider()
    st.subheader("📊 Put/Call Ratios")
    
    try:
        # Get CBOE data
        cboe = components["cboe"]
        cboe_data = cboe.get_all_data()
        
        # Check for manual override
        use_manual = st.session_state.get('use_manual_equity_pc', False)
        
        if use_manual:
            equity_pc = st.session_state.get('manual_equity_pc', 1.0)
            st.info(f"📝 Using manual equity P/C ratio: {equity_pc:.3f}")
        else:
            # Get nested put_call_ratios dict
            pc_ratios = cboe_data.get("put_call_ratios", {})
            equity_pc = pc_ratios.get("equity_pc")
            total_pc = pc_ratios.get("total_pc")
        
        if equity_pc is not None or total_pc is not None:
            col1, col2 = st.columns(2)
            
            # Equity P/C Ratio
            with col1:
                if equity_pc is not None:
                    # Determine sentiment
                    if equity_pc > 1.2:
                        color, label = "#f44336", "VERY BEARISH"
                    elif equity_pc > 1.0:
                        color, label = "#ff9800", "BEARISH"
                    elif equity_pc > 0.8:
                        color, label = "#9e9e9e", "NEUTRAL"
                    elif equity_pc > 0.6:
                        color, label = "#8bc34a", "BULLISH"
                    else:
                        color, label = "#4caf50", "VERY BULLISH"
                    
                    st.markdown(
                        f"<div style='text-align: center; padding: 1.5rem; background: {color}20; border-radius: 0.5rem; border: 2px solid {color};'>"
                        f"<h3 style='margin:0;'>Equity P/C Ratio</h3>"
                        f"<h1 style='margin:0.5rem 0; color: {color};'>{equity_pc:.3f}</h1>"
                        f"<p style='margin:0; font-size: 1.1rem; font-weight: bold;'>{label}</p>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    
                    st.caption("**Interpretation:** >1.0 = More puts than calls (bearish), <1.0 = More calls than puts (bullish)")
                else:
                    st.warning("Equity P/C ratio unavailable")
            
            # Total P/C Ratio
            with col2:
                if total_pc is not None:
                    # Determine sentiment
                    if total_pc > 1.1:
                        color, label = "#4caf50", "EXTREME FEAR (Contrarian Buy)"
                    elif total_pc > 0.90:
                        color, label = "#8bc34a", "NORMAL/HEALTHY"
                    elif total_pc > 0.80:
                        color, label = "#9e9e9e", "NEUTRAL"
                    elif total_pc > 0.70:
                        color, label = "#ff9800", "LOW (Complacent)"
                    else:
                        color, label = "#f44336", "EXTREME LOW (Danger)"
                    
                    st.markdown(
                        f"<div style='text-align: center; padding: 1.5rem; background: {color}20; border-radius: 0.5rem; border: 2px solid {color};'>"
                        f"<h3 style='margin:0;'>Total P/C Ratio</h3>"
                        f"<h1 style='margin:0.5rem 0; color: {color};'>{total_pc:.3f}</h1>"
                        f"<p style='margin:0; font-size: 1.1rem; font-weight: bold;'>{label}</p>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    
                    st.caption("**Interpretation:** Includes index + equity options (broader sentiment)")
                else:
                    st.warning("Total P/C ratio unavailable")
            
            # Explanation
            st.markdown("""
            ---
            **📖 What are Put/Call Ratios?**
            
            Put/Call ratios measure options trading activity:
            - **Equity P/C**: Individual stock options only (pure directional bets)
            - **Total P/C**: All options including indices (broader sentiment)
            
            **How to interpret:**
            - **> 1.0**: More put buying than calls → **Bearish/protective sentiment**
            - **< 1.0**: More call buying than puts → **Bullish sentiment**
            - **Extreme readings** (>1.5 or <0.5) often mark sentiment extremes and potential reversals
            
            **Note:** Can be manually overridden in Settings page if data is unavailable.
            """)
        else:
            st.warning("⚠️ Put/Call ratio data unavailable. You can set manual values in Settings.")
            
    except Exception as e:
        st.error(f"Error loading P/C ratios: {e}")
        import traceback
        st.code(traceback.format_exc())


# ============================================================
# CREDIT & LIQUIDITY  (Phase 1 + Phase 2 combined)
# ============================================================
elif page == "Credit & Liquidity":
    st.markdown(
        "<h1 class='main-header'> Credit Spreads & Liquidity</h1>",
        unsafe_allow_html=True,
    )
    
    st.markdown("""
    **Credit spreads** measure the risk premium for corporate debt, while **macro liquidity** 
    (RRP, TGA, Fed balance sheet) shows the plumbing behind market moves. Together they paint
    the complete credit picture.
    """)
    
    st.divider()
    
    try:
        # ---------- Liquidity data ----------
        liq_collector = components.get("liquidity")
        if liq_collector:
            if hasattr(liq_collector, "get_liquidity_history"):
                liquidity_df = liq_collector.get_liquidity_history(lookback_days=365)
            elif hasattr(liq_collector, "get_all_liquidity"):
                liquidity_df = liq_collector.get_all_liquidity(lookback_days=365)
            else:
                liquidity_df = pd.DataFrame()
        else:
            liquidity_df = pd.DataFrame()

        # ---------- Credit spreads snapshot ----------
        snapshot = components["db"].get_latest_snapshot()

        # ---------- Fed Balance Sheet / Net Liquidity ----------
        # Using CORRECT formula: Net Liquidity = Fed BS - TGA - RRP
        fed_bs_snapshot = components["fed_bs"].get_full_snapshot()
        net_liq_signal = None

        if fed_bs_snapshot and "balance_sheet_df" in fed_bs_snapshot:
            # Prepare TGA and RRP series for the liquidity analyzer
            tga_series = None
            rrp_series = None

            if not liquidity_df.empty:
                if 'tga' in liquidity_df.columns and 'date' in liquidity_df.columns:
                    tga_series = liquidity_df.set_index('date')['tga']
                if 'rrp_on' in liquidity_df.columns and 'date' in liquidity_df.columns:
                    rrp_series = liquidity_df.set_index('date')['rrp_on']

            # Use the corrected LiquidityAnalyzer (Fed BS - TGA - RRP)
            try:
                net_liq_signal = components["liquidity_analyzer"].analyze(
                    fed_bs_snapshot["balance_sheet_df"],
                    tga_series,
                    rrp_series
                )
            except Exception as e:
                logger.error(f"Error generating net liquidity signal: {e}")

        # =======================
        # TOP METRICS
        # =======================
        st.subheader(" Key Indicators")
        col1, col2, col3, col4 = st.columns(4)

        # Credit regime (HYG / LQD)
        with col1:
            if snapshot and snapshot.get('credit_spread_hy') is not None:
                hy_spread_bps = snapshot['credit_spread_hy'] * 100
                ig_spread_bps = (
                    snapshot.get('credit_spread_ig', None) * 100
                    if snapshot.get('credit_spread_ig') is not None
                    else None
                )

                if hy_spread_bps < 300:
                    regime = "🟢 Tight"
                    color = "#4CAF50"
                elif hy_spread_bps < 500:
                    regime = "🟡 Neutral"
                    color = "#FF9800"
                else:
                    regime = "🔴 Wide"
                    color = "#F44336"

                st.markdown(
                    f"<div style='text-align: center; padding: 1rem; background: {color}20; border-radius: 0.5rem;'>"
                    f"<h4 style='margin:0;'>Credit Regime</h4>"
                    f"<h2 style='margin:0; color: {color};'>{regime}</h2>"
                    f"<p style='margin:0;'>HYG: {hy_spread_bps:.0f} bps</p>"
                    + (f"<p style='margin:0;'>LQD: {ig_spread_bps:.0f} bps</p>" if ig_spread_bps is not None else "")
                    + "</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.warning("Credit data unavailable")

        # Net Liquidity (CORRECT formula: Fed BS - TGA - RRP)
        with col2:
            if net_liq_signal and net_liq_signal.signal != "UNAVAILABLE":
                net_liq = net_liq_signal.net_liquidity_billions
                regime = net_liq_signal.signal
                color = net_liq_signal.regime_color
                fed_bs = net_liq_signal.fed_bs_billions
                tga = net_liq_signal.tga_billions
                rrp = net_liq_signal.rrp_billions

                # Add emoji based on regime
                if regime == "SUPPORTIVE":
                    regime_display = "🟢 " + regime
                elif regime == "DRAINING":
                    regime_display = "🔴 " + regime
                else:
                    regime_display = "🟡 " + regime

                st.markdown(
                    f"<div style='text-align: center; padding: 1rem; background: {color}20; border-radius: 0.5rem;'>"
                    f"<h4 style='margin:0;'>Net Liquidity</h4>"
                    f"<h2 style='margin:0; color: {color};'>{regime_display}</h2>"
                    f"<p style='margin:0; font-size: 1.2rem; font-weight: bold;'>{format_large_number(net_liq) if net_liq else 'N/A'}</p>"
                    f"<p style='margin:0; font-size: 0.75rem; color: #666;'>Fed BS - TGA - RRP</p>"
                    f"<p style='margin:0; font-size: 0.7rem; color: #888;'>Z-score: {net_liq_signal.z_score:+.2f}</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.warning("Net liquidity data unavailable")

        # QT Pace & Fed Balance Sheet (combined in one column)
        with col3:
            if fed_bs_snapshot and fed_bs_snapshot.get('qt_pace_billions_month'):
                qt_pace = fed_bs_snapshot['qt_pace_billions_month']
                qt_cumulative = fed_bs_snapshot.get('qt_cumulative', 0)

                if qt_pace < -100:
                    color = "#F44336"
                elif qt_pace < 0:
                    color = "#FF9800"
                else:
                    color = "#4CAF50"

                st.markdown(
                    f"<div style='text-align: center; padding: 1rem; background: {color}20; border-radius: 0.5rem;'>"
                    f"<h4 style='margin:0;'>QT Pace</h4>"
                    f"<h2 style='margin:0; color: {color};'>{format_large_number(qt_pace, prefix='', suffix='B/mo')}</h2>"
                    f"<p style='margin:0;'>Cumulative QT: {format_large_number(qt_cumulative)}</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.warning("QT data unavailable")

        # Fed balance sheet size
        with col4:
            if fed_bs_snapshot and fed_bs_snapshot.get('total_assets'):
                fed_bs_billions = fed_bs_snapshot['total_assets']
                fed_bs_trillions = fed_bs_billions / 1000.0

                st.markdown(
                    "<div style='text-align: center; padding: 1rem; background: #1f77b420; border-radius: 0.5rem;'>"
                    "<h4 style='margin:0;'>Fed Balance Sheet</h4>"
                    f"<h2 style='margin:0; color: #1f77b4;'>${fed_bs_trillions:.2f}T</h2>",
                    unsafe_allow_html=True,
                )

                peak_bs = 8900  # billions, approx April 2022
                change_pct = ((fed_bs_billions - peak_bs) / peak_bs) * 100
                st.markdown(
                    f"<p style='margin:0;'>{change_pct:+.1f}% from peak</p></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.warning("Fed BS data unavailable")
        
        st.markdown("---")
        
        # =======================
        # CREDIT SPREADS HISTORY
        # =======================
        st.subheader("Credit Spreads History")
        hyg = components["db"].get_indicator_history("credit_spread_hy", days=365)
        lqd = components["db"].get_indicator_history("credit_spread_ig", days=365)

        if not hyg.empty or not lqd.empty:
            fig = go.Figure()
            
            if not hyg.empty:
                fig.add_trace(
                    go.Scatter(
                        x=hyg["date"],
                        y=hyg["value"] * 100,
                        name="High Yield (HYG)",
                        line=dict(color='#FF6B6B', width=2.5),
                        hovertemplate='HYG: %{y:.0f} bps<extra></extra>'
                    )
                )
            
            if not lqd.empty:
                fig.add_trace(
                    go.Scatter(
                        x=lqd["date"],
                        y=lqd["value"] * 100,
                        name="Investment Grade (LQD)",
                        line=dict(color='#4ECDC4', width=2.5),
                        hovertemplate='LQD: %{y:.0f} bps<extra></extra>'
                    )
                )
            
            fig.update_layout(
                title="Credit Spreads Over Time",
                xaxis_title="Date",
                yaxis_title="Spread (basis points)",
                height=400,
                hovermode='x unified',
                showlegend=True
            )
            
            st.plotly_chart(fig, width='stretch')
        else:
            st.warning("No credit spread history available.")
        
        # =======================
        # LIQUIDITY ANALYSIS
        # =======================
        if net_liq_signal and net_liq_signal.signal != "UNAVAILABLE" and not liquidity_df.empty:
            st.divider()
            st.subheader("📊 Macro Liquidity Analysis")

            # Using the correct formula: Fed BS - TGA - RRP
            fed_bs = net_liq_signal.fed_bs_billions
            formula_label = "Net Liquidity = Fed Balance Sheet - TGA - RRP"
            formula_subtitle = "Institutional liquidity calculation"
            st.success("✅ Using correct institutional liquidity formula")

            # Calculate net liquidity time series for charting
            liquidity_df_calc = liquidity_df.copy()
            if fed_bs is not None and fed_bs > 0:
                liquidity_df_calc["net_liquidity"] = (
                    (fed_bs * 1000) -  # Convert back to millions for calculation
                    liquidity_df_calc["tga"].fillna(0) -
                    liquidity_df_calc["rrp_on"].fillna(0)
                ) / 1000  # Convert to billions for display
            else:
                # If fed_bs not available, try to get from fed_bs_snapshot
                if fed_bs_snapshot and 'balance_sheet_df' in fed_bs_snapshot:
                    bs_df = fed_bs_snapshot['balance_sheet_df']
                    if not bs_df.empty and 'total_assets' in bs_df.columns:
                        # Merge Fed BS data with liquidity data
                        bs_df_copy = bs_df[['date', 'total_assets']].copy()
                        bs_df_copy = bs_df_copy.set_index('date')
                        liquidity_df_calc = liquidity_df_calc.set_index('date')
                        liquidity_df_calc = liquidity_df_calc.join(bs_df_copy, how='left')
                        liquidity_df_calc['total_assets'] = liquidity_df_calc['total_assets'].ffill()
                        liquidity_df_calc["net_liquidity"] = (
                            liquidity_df_calc["total_assets"].fillna(0) / 1000 -
                            liquidity_df_calc["tga"].fillna(0) / 1000 -
                            liquidity_df_calc["rrp_on"].fillna(0) / 1000
                        )
                        liquidity_df_calc = liquidity_df_calc.reset_index()

            # Net Liquidity chart
            st.markdown("**Net Liquidity Trend**")
            st.caption(formula_subtitle)

            fig = go.Figure()
            if "net_liquidity" in liquidity_df_calc.columns:
                fig.add_trace(
                    go.Scatter(
                        x=liquidity_df_calc["date"],
                        y=liquidity_df_calc["net_liquidity"],
                        name="Net Liquidity",
                        line=dict(color='#00D9FF', width=2.5),
                        fill='tozeroy',
                        fillcolor='rgba(0, 217, 255, 0.1)',
                        hovertemplate='Net Liq: $%{y:,.0f}B<extra></extra>'
                    )
                )

                # Add zero line
                fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

                # Show average line
                mean_liq = liquidity_df_calc["net_liquidity"].mean()
                if not pd.isna(mean_liq):
                    fig.add_hline(
                        y=mean_liq,
                        line_dash="dot",
                        line_color="yellow",
                        opacity=0.3,
                        annotation_text=f"Average: ${mean_liq:,.0f}B",
                        annotation_position="right"
                    )

            fig.update_layout(
                title=formula_label,
                xaxis_title="Date",
                yaxis_title="Net Liquidity (Billions USD)",
                height=450,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Explanation
            with st.expander("ℹ️ Understanding Net Liquidity"):
                current_description = net_liq_signal.description if net_liq_signal else "N/A"
                st.markdown(f"""
                **Net Liquidity** measures the actual cash available to financial markets:

                **Formula:** `Fed Balance Sheet - TGA - RRP`

                - **Fed Balance Sheet ↑** = Fed injecting liquidity (QE) → **Bullish** 🟢
                - **TGA ↑** = Treasury hoarding cash at Fed → **Draining** 🔴
                - **RRP ↑** = Banks parking cash at Fed → **Draining** 🔴

                **When net liquidity is:**
                - **Increasing** = More cash flowing to markets → Supportive for risk assets
                - **Decreasing** = Cash being drained → Headwind for risk assets

                **Current Status:** {current_description}
                """)

            # RRP + TGA stacked chart (keep this - it's useful)
            st.markdown("**Liquidity Components: RRP & TGA**")
            st.caption("Fed Reverse Repo + Treasury General Account (Drains)")

            fig = go.Figure()

            if "rrp_on" in liquidity_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=liquidity_df["date"],
                        y=liquidity_df["rrp_on"],
                        name="ON RRP",
                        mode='lines',
                        stackgroup='one',
                        fillcolor='rgba(255, 107, 107, 0.5)',
                        line=dict(color='#FF6B6B', width=1),
                        hovertemplate='RRP: $%{y:,.0f}B<extra></extra>'
                    )
                )

            if "tga" in liquidity_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=liquidity_df["date"],
                        y=liquidity_df["tga"],
                        name="TGA",
                        mode='lines',
                        stackgroup='one',
                        fillcolor='rgba(255, 165, 0, 0.5)',
                        line=dict(color='#FFA500', width=1),
                        hovertemplate='TGA: $%{y:,.0f}B<extra></extra>'
                    )
                )

            fig.update_layout(
                title="RRP + TGA Stacked (Liquidity Drains)",
                xaxis_title="Date",
                yaxis_title="Billions USD",
                height=400,
                hovermode='x unified',
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)

            # Summary metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                current_liq = net_liq_signal.net_liquidity_billions
                z_score = net_liq_signal.z_score
                st.metric(
                    "Current Net Liquidity",
                    f"${current_liq:,.0f}B" if current_liq else "N/A",
                    delta=f"{z_score:.2f}σ" if z_score is not None else "No data"
                )

            with col2:
                rrp_value = net_liq_signal.rrp_billions
                st.metric("Current RRP", f"${rrp_value:,.0f}B" if rrp_value else "N/A")

            with col3:
                tga_value = net_liq_signal.tga_billions
                st.metric("Current TGA", f"${tga_value:,.0f}B" if tga_value else "N/A")
    except Exception as e:
        logger.error(f"Error loading liquidity data: {e}")
        st.warning(f"⚠️ Liquidity data unavailable: {e}")
# ============================================================
# VOLATILITY & VRP
# ============================================================
elif page == "Volatility & VRP":
    st.header("Volatility Risk Premium Analysis")
    
    st.markdown("""
    **Volatility Risk Premium (VRP)** measures the difference between implied volatility (VIX) 
    and realized volatility. A positive VRP indicates options are expensive relative to actual market moves.
    """)
    
    st.divider()

        # ============================================
        # SKEW INDEX CHART
        # ============================================

        
    
    with st.spinner("Calculating VRP..."):
        vrp_analysis = get_vrp_analysis_cached()
    
    if "error" in vrp_analysis:
        st.error(f"Error: {vrp_analysis['error']}")
    else:
        components["db"].save_vrp_data(vrp_analysis)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("VIX", f"{vrp_analysis['vix']:.2f}")
            st.caption("Current VIX (implied volatility)")
        
        with col2:
            st.metric("Realized Vol (21d)", f"{vrp_analysis['realized_vol']:.2f}")
            st.caption("SPY realized volatility over 21 days")
        
        with col3:
            vrp_val = vrp_analysis['vrp']
            vrp_delta = "Rich" if vrp_val > 0 else "Cheap"
            st.metric("VRP", f"{vrp_val:+.2f}", vrp_delta)
            st.caption("VIX - Realized Vol (positive = options expensive)")
        
        with col4:
            st.metric(
                "Hist. Avg Return",
                f"{vrp_analysis['expected_6m_return']:.1f}%"
            )
            st.caption("6M avg return in this VIX regime (historical)")
        
        st.divider()
        
        # ============================================
        # SKEW INDEX CHART
        # ============================================
        st.subheader(" CBOE SKEW Index - Tail Risk Premium")
        
        st.info("""
        ** What is SKEW?**  
        SKEW measures tail risk - how much investors are paying for deep out-of-the-money puts (crash protection).
        
        - **100-130:** Low tail hedging (complacency)
        - **130-145:** Normal tail protection  
        - **145-160:** Elevated hedging (institutions worried)
        - **>160:** Extreme tail hedging (crisis mode)
        
        High SKEW doesn’t predict a crash — it shows demand for crash insurance, not an imminent event.
        *Used by hedge funds to gauge institutional positioning and crash fear.*
        """)
        
        with st.spinner("Loading SKEW history..."):
            from data_collectors.cboe_collector import CBOECollector
            from dashboard.pdf_chart_builder import create_skew_history_chart
            
            cboe = CBOECollector()
            skew_df = cboe.get_skew_history(days=90)
            
            if not skew_df.empty:
                skew_chart = create_skew_history_chart(skew_df)
                if skew_chart:
                    st.plotly_chart(skew_chart, width='stretch')
                    
                    # Current interpretation
                    latest_skew = skew_df.iloc[-1]['skew']
                    if latest_skew > 160:
                        st.error(f"🔴 **Extreme tail hedging:** SKEW at {latest_skew:.1f} - institutions heavily hedged for crash")
                    elif latest_skew > 145:
                        st.warning(f"🟡 **Elevated protection:** SKEW at {latest_skew:.1f} - above-normal tail hedging")
                    elif latest_skew > 130:
                        st.success(f"🟢 **Normal range:** SKEW at {latest_skew:.1f} - typical tail protection")
                    else:
                        st.info(f"⚪ **Low hedging:** SKEW at {latest_skew:.1f} - minimal crash protection")
            else:
                st.warning("SKEW historical data unavailable")
        
        st.divider()

        # ============================================
        # VVIX (VIX OF VIX) - BUY SIGNAL
        # ============================================
        st.subheader("🎯 VVIX (VIX of VIX) - Institutional Buy Signal")

        st.info("""
        **What is VVIX?**
        VVIX measures the expected volatility OF the VIX itself - the "fear of fear" index.

        **Key Levels:**
        - **< 80:** Low vol-of-vol (complacency, potential for vol expansion)
        - **80-100:** Normal range
        - **100-120:** Elevated uncertainty
        - **≥ 120:** 🟢 **STRONG BUY SIGNAL** - Historic turning point!

        **Why 120+ matters:**
        When VVIX spikes above 120, dealers are scrambling for gamma protection.
        This almost always marks capitulation. The subsequent mean reversion creates
        powerful **vanna and charm tailwinds** as implied vol collapses → S&P500 rallies.

        *Used by institutional traders to identify major market bottoms.*
        """)

        with st.spinner("Loading VVIX data..."):
            from dashboard.pdf_chart_builder import create_vvix_history_chart

            # Get VVIX history
            vvix_df = cboe.get_vvix_history(days=90)
            vvix_signal = cboe.get_vvix_signal()

            # Display current signal prominently
            if vvix_signal and vvix_signal.get('signal') != 'UNAVAILABLE':
                signal = vvix_signal['signal']
                level = vvix_signal['level']
                color = vvix_signal['color']
                description = vvix_signal['description']
                strength = vvix_signal['strength']

                # Large signal box
                if signal == 'STRONG BUY':
                    st.success(f"""
                    ### 🟢 {signal} - VVIX at {level:.1f}
                    **Strength: {strength:.0f}/100**

                    {description}
                    """)
                elif signal == 'BUY ALERT':
                    st.warning(f"""
                    ### 🟡 {signal} - VVIX at {level:.1f}
                    **Strength: {strength:.0f}/100**

                    {description}
                    """)
                elif signal == 'CAUTION':
                    st.warning(f"""
                    ### 🟠 {signal} - VVIX at {level:.1f}

                    {description}
                    """)
                else:
                    st.info(f"""
                    ### ⚪ {signal} - VVIX at {level:.1f}

                    {description}
                    """)

            # Chart
            if not vvix_df.empty:
                vvix_chart = create_vvix_history_chart(vvix_df)
                if vvix_chart:
                    st.plotly_chart(vvix_chart, use_container_width=True)

                    # Historical context
                    latest_vvix = vvix_df.iloc[-1]['vvix']
                    max_vvix = vvix_df['vvix'].max()
                    min_vvix = vvix_df['vvix'].min()
                    avg_vvix = vvix_df['vvix'].mean()

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current VVIX", f"{latest_vvix:.1f}")
                    with col2:
                        st.metric("90d High", f"{max_vvix:.1f}")
                    with col3:
                        st.metric("90d Low", f"{min_vvix:.1f}")
                    with col4:
                        st.metric("90d Average", f"{avg_vvix:.1f}")

                    # Days above 120 in period
                    days_above_120 = (vvix_df['vvix'] >= 120).sum()
                    if days_above_120 > 0:
                        st.caption(f"📊 VVIX was ≥120 on **{days_above_120}** of the last 90 days")
            else:
                st.warning("VVIX historical data unavailable")

        st.divider()

        # ============================================
        # VIX9D vs VIX SPREAD CHART
        # ============================================
        st.subheader(" VIX9D vs VIX (30d) Spread - Near-Term Risk")
        
        st.info("""
        ** What is VIX9D?**  
        VIX9D is 9-day implied volatility - the market's expectation for volatility over the **NEXT WEEK**.
        
        - **VIX9D < VIX (negative spread):** Calm near-term (good for selling weekly options)
        - **VIX9D ≈ VIX (flat):** Normal term structure
        - **VIX9D > VIX (positive spread):** Near-term event risk (FOMC, CPI, earnings)
        
        *Used by traders for weekly options strategies and event hedging.*
        """)
        
        with st.spinner("Loading VIX9D spread..."):
            from dashboard.pdf_chart_builder import create_vix9d_spread_chart
            
            vix9d_df = cboe.get_vix9d_history(days=90)
            
            # Get VIX history for comparison
            import yfinance as yf
            vix_ticker = yf.Ticker('^VIX')
            vix_data = vix_ticker.history(period='90d')
            vix_df = pd.DataFrame({
                'date': vix_data.index.date,
                'vix': vix_data['Close'].values
            })
            
            if not vix9d_df.empty and not vix_df.empty:
                spread_chart = create_vix9d_spread_chart(vix9d_df, vix_df)
                if spread_chart:
                    st.plotly_chart(spread_chart, width='stretch')
                    
                    # Current interpretation
                    merged = pd.merge(vix9d_df, vix_df, on='date', how='inner')
                    merged['spread_pct'] = ((merged['vix9d'] - merged['vix']) / merged['vix']) * 100
                    latest_spread = merged.iloc[-1]['spread_pct']
                    
                    if latest_spread < -10:
                        st.success(f"✅ **Calm near-term:** Spread at {latest_spread:.1f}% - market expects quiet week ahead")
                    elif latest_spread > 10:
                        st.error(f"⚠️ **Event risk ahead:** Spread at {latest_spread:.1f}% - market pricing near-term volatility spike")
                    else:
                        st.info(f"⚪ **Normal structure:** Spread at {latest_spread:.1f}% - typical term structure")
            else:
                st.warning("VIX9D spread data unavailable")
        
        
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Volatility Regime")
            regime_color = vrp_analysis['regime_color']
            st.markdown(
                f"<div style='padding: 20px; background-color: {regime_color}20; border-left: 6px solid {regime_color}; border-radius: 8px; text-align: center;'>"
                f"<h2 style='margin: 0; color: {regime_color};'>{vrp_analysis['regime']}</h2>"
                f"<p style='margin: 10px 0 0 0; font-size: 16px; color: #666;'>VIX: {vrp_analysis['vix_range']}</p>"
                f"</div>",
                unsafe_allow_html=True
            )
            st.markdown("#### Regime Characteristics")
            regime_text = {
                "Complacent": "Very low volatility. Market calm, potentially over-complacent.",
                "Normal": "Typical volatility levels. Healthy market conditions.",
                "Elevated": "Above-average volatility. Increased uncertainty.",
                "Fearful": "High volatility. Significant market concern.",
                "Panic": "Extreme volatility. Market stress - historically strong buying opportunity.",
                "Extreme Panic": "Crisis-level volatility. Maximum fear - exceptional buying opportunity."
            }
            st.info(regime_text.get(vrp_analysis['regime'], "Unknown regime"))
        
        with col2:
            st.subheader("VRP Interpretation")
            vrp_color = vrp_analysis['vrp_color']
            st.markdown(
                f"<div style='padding: 20px; background-color: {vrp_color}20; border-left: 6px solid {vrp_color}; border-radius: 8px; text-align: center;'>"
                f"<h2 style='margin: 0; color: {vrp_color};'>{vrp_analysis['vrp_level']}</h2>"
                f"<p style='margin: 10px 0 0 0; font-size: 16px; color: #666;'>VRP: {vrp_analysis['vrp']:+.2f}</p>"
                f"</div>",
                unsafe_allow_html=True
            )
            st.markdown("#### VRP Analysis")
            st.write(f"**{vrp_analysis['vrp_interpretation']}**")
            st.caption(f"Trading implication: {vrp_analysis['vrp_implication']}")
        
        st.divider()
        
        st.subheader("VRP Gauge")
        vrp_val = vrp_analysis['vrp']
        vrp_color = vrp_analysis['vrp_color']
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=vrp_val,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Volatility Risk Premium", 'font': {'size': 24}},
            delta={'reference': 0, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
            gauge={
                'axis': {'range': [-10, 15], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': vrp_color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [-10, -4], 'color': '#FFCDD2'},
                    {'range': [-4, 0], 'color': '#FFE0B2'},
                    {'range': [0, 4], 'color': '#FFF9C4'},
                    {'range': [4, 8], 'color': '#C8E6C9'},
                    {'range': [8, 15], 'color': '#A5D6A7'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0
                }
            }
        ))
        fig.update_layout(height=400, font={'color': "darkblue", 'family': "Arial"})
        st.plotly_chart(fig, width='stretch')
        
        st.divider()
        
        st.subheader("Historical VRP")
        with st.spinner("Loading historical data..."):
            history_days = st.select_slider(
                "Time Period",
                options=[30, 90, 180, 252, 504],
                value=180,
                format_func=lambda x: f"{x} days" if x < 252 else f"{x//252} year" + ("s" if x > 252 else "")
            )
            vrp_history = get_vrp_history_cached(days=history_days)
        
        if not vrp_history.empty:
            from plotly.subplots import make_subplots
            
            vrp_percentile = (vrp_history['vrp'] < vrp_analysis['vrp']).sum() / len(vrp_history) * 100
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Scatter(
                    x=vrp_history['date'],
                    y=vrp_history['vix'],
                    name='VIX (Implied)',
                    line=dict(color='#FF6B6B', width=2.5),
                    hovertemplate='VIX: %{y:.2f}<extra></extra>'
                ),
                secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(
                    x=vrp_history['date'],
                    y=vrp_history['realized_vol'],
                    name='RVol 21d',
                    line=dict(color='#4ECDC4', width=2.5, dash='dash'),
                    hovertemplate='RVol 21d: %{y:.2f}<extra></extra>'
                ),
                secondary_y=False
            )
            
            if 'realized_vol_50d' in vrp_history.columns:
                fig.add_trace(
                    go.Scatter(
                        x=vrp_history['date'],
                        y=vrp_history['realized_vol_50d'],
                        name='RVol 50d (Trend)',
                        line=dict(color='rgba(255, 165, 0, 0.4)', width=1.5, dash='dot'),
                        hovertemplate='RVol 50d: %{y:.2f}<extra></extra>'
                    ),
                    secondary_y=False
                )
            
            fig.add_trace(
                go.Scatter(
                    x=vrp_history['date'],
                    y=vrp_history['vrp'],
                    name='VRP Spread',
                    line=dict(color='#95E1D3', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(149, 225, 211, 0.3)',
                    hovertemplate='VRP: %{y:.2f}<extra></extra>'
                ),
                secondary_y=True
            )
            
            vrp_expensive = vrp_history['vrp'].copy()
            vrp_expensive[vrp_expensive <= 8] = 8
            fig.add_trace(
                go.Scatter(
                    x=vrp_history['date'],
                    y=vrp_expensive,
                    name='VRP >8',
                    line=dict(color='#2ecc71', width=0),
                    fill='tonexty',
                    fillcolor='rgba(46, 204, 113, 0.2)',
                    hoverinfo='skip',
                    showlegend=False
                ),
                secondary_y=True
            )
            
            vrp_cheap = vrp_history['vrp'].copy()
            vrp_cheap[vrp_cheap >= 0] = 0
            fig.add_trace(
                go.Scatter(
                    x=vrp_history['date'],
                    y=vrp_cheap,
                    name='VRP <0',
                    line=dict(color='#e74c3c', width=0),
                    fill='tozeroy',
                    fillcolor='rgba(231, 76, 60, 0.2)',
                    hoverinfo='skip',
                    showlegend=False
                ),
                secondary_y=True
            )
            
            fig.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.4, secondary_y=True)
            fig.add_hline(y=8, line_dash="dot", line_color="green", opacity=0.2, secondary_y=True)
            
            fig.add_trace(
                go.Scatter(
                    x=[vrp_history['date'].iloc[-1]],
                    y=[vrp_analysis['vrp']],
                    mode='markers',
                    name='Current',
                    marker=dict(size=15, color='#FF1744', symbol='diamond', line=dict(width=2, color='white')),
                    hovertemplate=f'Now: {vrp_analysis["vrp"]:.2f}<extra></extra>',
                    showlegend=True
                ),
                secondary_y=True
            )
            
            fig.update_xaxes(title_text="Date", showgrid=False)
            fig.update_yaxes(
                title_text="Volatility (%)", 
                secondary_y=False,
                showgrid=True,
                gridcolor='rgba(128,128,128,0.15)'
            )
            fig.update_yaxes(
                title_text="VRP (pts)", 
                secondary_y=True,
                showgrid=False
            )
            
            fig.update_layout(
                title={
                    'text': f"VIX vs Realized Vol & VRP Spread (Percentile: {vrp_percentile:.0f}%)",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16}
                },
                height=550,
                hovermode='x unified',
                legend=dict(
                    orientation="h", 
                    yanchor="bottom", 
                    y=1.02, 
                    xanchor="center", 
                    x=0.5,
                    font=dict(size=11)
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, width='stretch')
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg VRP", f"{vrp_history['vrp'].mean():.2f}")
            with col2:
                st.metric("VRP Std Dev", f"{vrp_history['vrp'].std():.2f}")
            with col3:
                percentile = (vrp_history['vrp'] < vrp_analysis['vrp']).mean() * 100
                st.metric("Current Percentile", f"{percentile:.1f}%")
            with col4:
                max_vrp = vrp_history['vrp'].max()
                st.metric("Max VRP", f"{max_vrp:.2f}")
        else:
            st.warning("Could not load historical VRP data")
        
        st.divider()
        with st.expander(" How to Use VRP in Your Trading"):
            st.markdown("""
            ### Volatility Risk Premium (VRP) Guide
            
            **High VRP (> 4):**
            - Options are expensive relative to realized moves
            - Historically supportive for equities
            - Consider: selling volatility, risk-on positioning
            
            **Neutral VRP (0 to 4):**
            - Options fairly priced
            - Balanced risk/reward environment
            
            **Negative VRP (< 0):**
            - Realized volatility exceeding implied
            - Market potentially underpricing risk
            - Consider: buying protection, reducing exposure
            """)

# ============================================================
# SECTORS & VIX
# ============================================================
elif page == "Sectors & VIX":
    try:
        st.header("Sector Rotation & VIX Analysis")

        st.subheader("VIX Term Structure")
        vix_term = get_vix_term_structure()

        if not vix_term.empty:
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=vix_term["Maturity"],
                    y=vix_term["VIX Level"],
                    mode="lines+markers",
                    name="VIX Term Structure",
                    line=dict(color="royalblue", width=3),
                    marker=dict(size=10),  
                )
            )

            if vix_term["VIX Level"].iloc[-1] > vix_term["VIX Level"].iloc[0]:
                contango_text = "Contango (bullish)"
                color = "green"
            else:
                contango_text = "Backwardation (risk-off)"
                color = "red"

            fig.update_layout(
                title=f"VIX Term Structure - {contango_text}",
                xaxis_title="Maturity",
                yaxis_title="VIX Level",
                height=400,
                annotations=[
                    dict(
                        text=contango_text,
                        x=0.5,
                        y=0.95,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        font=dict(size=14, color=color),
                    )
                ],
            )

            st.plotly_chart(fig, width='stretch')

        st.divider()

        st.subheader("Sector Performance Analysis")

        col1, col2 = st.columns([3, 1])

        with col1:
            period_options = {
                "1 Day": "1d",
                "1 Week": "5d",
                "1 Month": "1mo",
                "3 Months": "3mo",
                "6 Months": "6mo",
                "YTD": "ytd",
                "1 Year": "1y",
                "5 Years": "5y",
            }

            selected_period_label = st.radio(
                "Select Time Period:",
                options=list(period_options.keys()),
                horizontal=True,
                index=2,
            )

        with col2:
            view_mode = st.selectbox("View:", ["Bar Chart", "Line Chart", "Table Only"])

        period_code = period_options[selected_period_label]
        
        try:
            sectors = get_sector_performance(period_code)
        except Exception as e:
            st.error(f"Error fetching sector data: {e}")
            st.stop()

        if not sectors.empty:
            if view_mode == "Bar Chart":
                fig = px.bar(
                    sectors,
                    x="Sector",
                    y="Change %",
                    color="Change %",
                    color_continuous_scale=["red", "yellow", "green"],
                    title=f"Sector Returns - {selected_period_label}",
                    text="Change %",
                )

                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig.update_layout(height=450, showlegend=False)
                st.plotly_chart(fig, width='stretch')

            elif view_mode == "Line Chart":
                comparison_data = get_sector_comparison_chart(period_code)

                if not comparison_data.empty:
                    fig = go.Figure()

                    for col in comparison_data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=comparison_data.index,
                                y=comparison_data[col],
                                mode="lines",
                                name=col,
                                line=dict(width=2),
                            )
                        )

                    fig.update_layout(
                        title=f"Sector Performance Comparison - {selected_period_label} (Base 100)",
                        xaxis_title="Date",
                        yaxis_title="Performance (Base 100)",
                        height=500,
                        hovermode="x unified",
                    )

                    st.plotly_chart(fig, width='stretch')

            st.divider()

            st.subheader("Detailed Sector Data")

            sectors["Rank"] = range(1, len(sectors) + 1)
            display_df = sectors[["Rank", "Sector", "Ticker", "Change %", "Price", "Volatility"]]

            st.dataframe(
                display_df.style.format(
                    {
                        "Change %": "{:+.2f}%",
                        "Price": "${:.2f}",
                        "Volatility": "{:.2f}%",
                    }
                ),
                width='stretch',
                hide_index=True,
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Best Performer",
                    sectors.iloc[0]["Sector"],
                    f"{sectors.iloc[0]['Change %']:+.2f}%",
                )
            with col2:
                st.metric(
                    "Worst Performer",
                    sectors.iloc[-1]["Sector"],
                    f"{sectors.iloc[-1]['Change %']:+.2f}%",
                )
            with col3:
                st.metric("Average Return", f"{sectors['Change %'].mean():+.2f}%")
            with col4:
                advancing = (sectors["Change %"] > 0).sum()
                st.metric("Sectors Advancing", f"{advancing}/{len(sectors)}")
        else:
            st.error("Unable to fetch sector data.")
    except Exception as e:
        st.error(f"Error loading Sectors page: {e}")
        import traceback
        st.code(traceback.format_exc())

# ============================================================
# MARKET BREADTH
# ============================================================
elif page == "Market Breadth":
    st.header("📊 S&P 500 Market Breadth Analysis")

    st.markdown("""
    Market breadth measures the number of advancing vs declining stocks.
    **Strong breadth** = healthy market participation. **Weak breadth** = narrow leadership (risk of reversal).
    """)

    # Get current breadth mode from settings
    current_breadth_mode = get_breadth_mode()
    scale_factor = get_mcclellan_scale_factor()
    stock_count = "500" if current_breadth_mode == 'full' else "100"
    time_estimate = "3-5 minutes" if current_breadth_mode == 'full' else "30-60 seconds"

    # Add refresh button at the top
    col_refresh, col_info = st.columns([1, 3])
    with col_refresh:
        refresh_breadth = st.button("🔄 Refresh Breadth Data", help=f"Recalculate from {stock_count} S&P 500 stocks (takes {time_estimate})")
    with col_info:
        mode_icon = "🎯" if current_breadth_mode == 'full' else "⚡"
        st.caption(f"{mode_icon} Mode: {current_breadth_mode.upper()} ({stock_count} stocks) - Change in Settings")

    try:
        # Force refresh if button clicked
        if refresh_breadth:
            st.info(f"📊 Recalculating breadth data from {stock_count} stocks ({time_estimate})...")
            from data_collectors.sp500_adline_calculator import SP500ADLineCalculator

            calc = SP500ADLineCalculator(mode=current_breadth_mode)
            breadth_history = calc.get_breadth_history(days=90)

            if not breadth_history.empty:
                # Calculate McClellan (scale factor depends on mode)
                breadth_history['ema19'] = breadth_history['ad_diff'].ewm(span=19, adjust=False).mean() * scale_factor
                breadth_history['ema39'] = breadth_history['ad_diff'].ewm(span=39, adjust=False).mean() * scale_factor
                breadth_history['mcclellan'] = breadth_history['ema19'] - breadth_history['ema39']

                # Save to DB
                components["db"].save_breadth_data(breadth_history)
                st.success(f"✅ Breadth data refreshed successfully! ({stock_count} stocks)")
                st.rerun()
        else:
            # Get breadth data from DB
            breadth_history = components["db"].get_breadth_history(days=90)

        if breadth_history.empty:
            st.info(f"📊 Calculating fresh breadth data ({time_estimate})...")
            from data_collectors.sp500_adline_calculator import SP500ADLineCalculator

            calc = SP500ADLineCalculator(mode=current_breadth_mode)
            breadth_history = calc.get_breadth_history(days=90)

            if not breadth_history.empty:
                # Calculate McClellan (scale factor depends on mode)
                breadth_history['ema19'] = breadth_history['ad_diff'].ewm(span=19, adjust=False).mean() * scale_factor
                breadth_history['ema39'] = breadth_history['ad_diff'].ewm(span=39, adjust=False).mean() * scale_factor
                breadth_history['mcclellan'] = breadth_history['ema19'] - breadth_history['ema39']

                # Save to DB
                components["db"].save_breadth_data(breadth_history)

        if not breadth_history.empty:
            # ALWAYS recalculate McClellan from ad_diff to ensure accuracy
            # (DB values may be stale, NULL, or 0.0)
            # Scale factor: 5.0 for fast mode (100 stocks), 1.0 for full mode (500 stocks)
            if 'ad_diff' in breadth_history.columns:
                breadth_history['ema19'] = breadth_history['ad_diff'].ewm(span=19, adjust=False).mean() * scale_factor
                breadth_history['ema39'] = breadth_history['ad_diff'].ewm(span=39, adjust=False).mean() * scale_factor
                breadth_history['mcclellan'] = breadth_history['ema19'] - breadth_history['ema39']
            elif 'advancing' in breadth_history.columns and 'declining' in breadth_history.columns:
                # Calculate ad_diff if missing
                breadth_history['ad_diff'] = breadth_history['advancing'] - breadth_history['declining']
                breadth_history['ema19'] = breadth_history['ad_diff'].ewm(span=19, adjust=False).mean() * scale_factor
                breadth_history['ema39'] = breadth_history['ad_diff'].ewm(span=39, adjust=False).mean() * scale_factor
                breadth_history['mcclellan'] = breadth_history['ema19'] - breadth_history['ema39']

        if not breadth_history.empty:
            # Get SPY price for divergence detection
            import yfinance as yf
            spy = yf.Ticker('SPY')
            spy_data = spy.history(period='90d')
            spy_price = spy_data['Close']
            
            # Calculate all signals
            from processors.breadth_signals import get_comprehensive_breadth_signals
            signals = get_comprehensive_breadth_signals(breadth_history, spy_price)
            
            latest = signals['latest_breadth']
            ad_ratio = signals['ad_ratio']
            zweig = signals['zweig_thrust']
            divergence = signals['divergence']
            z_score = signals['z_score']
            
            # ========== TOP METRICS ROW ==========
            st.subheader("📈 Current Breadth Snapshot")

            # Get current McClellan value
            current_mcclellan = breadth_history['mcclellan'].iloc[-1] if 'mcclellan' in breadth_history.columns else 0

            # Determine McClellan status
            if current_mcclellan > 50:
                mc_color = "#4CAF50"
                mc_status = "Strong Bull"
            elif current_mcclellan > 20:
                mc_color = "#8BC34A"
                mc_status = "Bullish"
            elif current_mcclellan > -20:
                mc_color = "#FF9800"
                mc_status = "Neutral"
            elif current_mcclellan > -50:
                mc_color = "#FF6B6B"
                mc_status = "Bearish"
            else:
                mc_color = "#F44336"
                mc_status = "Strong Bear"

            col1, col2, col3, col4, col5, col6 = st.columns(6)

            with col1:
                breadth_pct = latest['breadth_pct']

                if breadth_pct > 70:
                    color = "#4CAF50"
                    status = "Strong"
                elif breadth_pct > 55:
                    color = "#8BC34A"
                    status = "Healthy"
                elif breadth_pct > 45:
                    color = "#FF9800"
                    status = "Neutral"
                else:
                    color = "#F44336"
                    status = "Weak"

                st.markdown(
                    f"<div style='text-align: center; padding: 1rem; background: {color}20; border-radius: 0.5rem;'>"
                    f"<h5 style='margin:0;'>Market Breadth</h5>"
                    f"<h2 style='margin:0.3rem 0; color: {color};'>{breadth_pct:.1f}%</h2>"
                    f"<p style='margin:0; color: {color}; font-weight: bold;'>{status}</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            with col2:
                # McClellan Oscillator - THE KEY METRIC
                st.markdown(
                    f"<div style='text-align: center; padding: 1rem; background: {mc_color}20; border-radius: 0.5rem;'>"
                    f"<h5 style='margin:0;'>McClellan Osc</h5>"
                    f"<h2 style='margin:0.3rem 0; color: {mc_color};'>{current_mcclellan:+.1f}</h2>"
                    f"<p style='margin:0; color: {mc_color}; font-weight: bold;'>{mc_status}</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            with col3:
                adv = latest['advancing']
                dec = latest['declining']
                total = adv + dec

                st.markdown(
                    "<div style='text-align: center; padding: 1rem; background: #4CAF5020; border-radius: 0.5rem;'>"
                    f"<h5 style='margin:0;'>Advancing</h5>"
                    f"<h2 style='margin:0.3rem 0; color: #4CAF50;'>{adv}</h2>"
                    f"<p style='margin:0;'>of {total} stocks</p>"
                    "</div>",
                    unsafe_allow_html=True
                )

            with col4:
                ratio = ad_ratio['ratio']
                ratio_color = ad_ratio['color']
                ratio_interp = ad_ratio['interpretation']

                ratio_display = f"{ratio:.2f}x" if ratio < 10 else "∞"

                st.markdown(
                    f"<div style='text-align: center; padding: 1rem; background: {ratio_color}20; border-radius: 0.5rem;'>"
                    f"<h5 style='margin:0;'>A/D Ratio</h5>"
                    f"<h2 style='margin:0.3rem 0; color: {ratio_color};'>{ratio_display}</h2>"
                    f"<p style='margin:0;'>{ratio_interp}</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            with col5:
                zweig_color = zweig['color']
                zweig_status = "✅ ACTIVE" if zweig['active'] else "None"

                st.markdown(
                    f"<div style='text-align: center; padding: 1rem; background: {zweig_color}20; border-radius: 0.5rem;'>"
                    f"<h5 style='margin:0;'>Zweig Thrust</h5>"
                    f"<h2 style='margin:0.3rem 0; color: {zweig_color};'>{zweig_status}</h2>"
                    f"<p style='margin:0; font-size: 0.8rem;'>Rare bull signal</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            with col6:
                div_color = divergence['color']
                div_type = divergence['type'].upper()
                
                if div_type == "BEARISH":
                    div_icon = "⚠️"
                elif div_type == "BULLISH":
                    div_icon = "✅"
                else:
                    div_icon = "⚪"
                
                st.markdown(
                    f"<div style='text-align: center; padding: 1.5rem; background: {div_color}20; border-radius: 0.5rem;'>"
                    f"<h4 style='margin:0;'>Divergence</h4>"
                    f"<h1 style='margin:0.5rem 0; color: {div_color};'>{div_icon}</h1>"
                    f"<p style='margin:0;'>{div_type}</p>"
                    "</div>",
                    unsafe_allow_html=True
                )
            
            # Z-Score banner
            if z_score['z_score'] is not None:
                z_val = z_score['z_score']
                z_interp = z_score['interpretation']
                
                st.info(f" **Statistical Context:** Current breadth is **{z_val:+.2f}σ** ({z_interp}) vs 90-day average of {z_score['mean']:.1f}%")
            
            st.divider()
            
            # Signal Details
            with st.expander(" Signal Details", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**⚡ Zweig Breadth Thrust**")
                    st.write(zweig['description'])
                    st.caption(f"10-day EMA: {zweig['ema10']:.1%} (need <40% → >61.5% in 10 days)")
                
                with col2:
                    st.markdown("** Price-Breadth Divergence**")
                    st.write(divergence['description'])
                    if divergence['type'] == 'bearish':
                        st.caption(" Warning: Price strength not confirmed by breadth")
                    elif divergence['type'] == 'bullish':
                        st.caption(" Positive: Breadth holding despite price weakness")
            
            st.divider()
            
            # Charts
            if 'mcclellan' in breadth_history.columns:
                tab1, tab2, tab3 = st.tabs([" A/D Line", " Breadth %", " McClellan"])
                
                with tab1:
                    st.subheader("Advance-Decline Line")
                    st.caption("Cumulative measure of market breadth - higher = healthier market")
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=breadth_history['date'],
                        y=breadth_history['ad_line'],
                        name='A/D Line',
                        line=dict(color='#1f77b4', width=3),
                        fill='tozeroy',
                        fillcolor='rgba(31, 119, 180, 0.2)',
                        hovertemplate='A/D Line: %{y:,.0f}<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title="",
                        xaxis_title="Date",
                        yaxis_title="A/D Line Value",
                        height=500,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, width='stretch')
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        current_ad = breadth_history['ad_line'].iloc[-1]
                        st.metric("Current A/D Line", f"{current_ad:,.0f}")
                    with col2:
                        week_ago = breadth_history['ad_line'].iloc[-6] if len(breadth_history) >= 6 else current_ad
                        week_change = current_ad - week_ago
                        st.metric("Week Change", f"{week_change:+,.0f}")
                    with col3:
                        month_ago = breadth_history['ad_line'].iloc[-22] if len(breadth_history) >= 22 else current_ad
                        month_change = current_ad - month_ago
                        st.metric("Month Change", f"{month_change:+,.0f}")
                
                with tab2:
                    st.subheader("Daily Breadth Percentage")
                    st.caption("% of stocks advancing each day - above 50% = bullish, below 50% = bearish")
                    
                    fig = go.Figure()
                    
                    colors = ['#4CAF50' if x > 50 else '#F44336' for x in breadth_history['breadth_pct']]
                    
                    fig.add_trace(go.Bar(
                        x=breadth_history['date'],
                        y=breadth_history['breadth_pct'],
                        name='Breadth %',
                        marker_color=colors,
                        hovertemplate='Breadth: %{y:.1f}%<extra></extra>'
                    ))
                    
                    fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
                    fig.add_hline(y=70, line_dash="dot", line_color="green", opacity=0.3)
                    fig.add_hline(y=30, line_dash="dot", line_color="red", opacity=0.3)
                    
                    # Add Zweig levels
                    fig.add_hline(y=61.5, line_dash="dot", line_color="blue", opacity=0.4,
                                annotation_text="Zweig High (61.5%)", annotation_position="right")
                    fig.add_hline(y=40, line_dash="dot", line_color="orange", opacity=0.4,
                                annotation_text="Zweig Low (40%)", annotation_position="right")
                    
                    fig.update_layout(
                        title="",
                        xaxis_title="Date",
                        yaxis_title="Breadth %",
                        height=500,
                        hovermode='x unified',
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, width='stretch')
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        avg_breadth = breadth_history['breadth_pct'].tail(30).mean()
                        st.metric("30-Day Average", f"{avg_breadth:.1f}%")
                    with col2:
                        strong_days = (breadth_history['breadth_pct'].tail(30) > 60).sum()
                        st.metric("Strong Days (>60%)", f"{strong_days}/30")
                    with col3:
                        weak_days = (breadth_history['breadth_pct'].tail(30) < 40).sum()
                        st.metric("Weak Days (<40%)", f"{weak_days}/30")
                
                with tab3:
                    st.subheader("McClellan Oscillator")
                    st.caption("Breadth momentum indicator - positive = improving, negative = weakening")
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=breadth_history['date'],
                        y=breadth_history['mcclellan'],
                        name='McClellan',
                        line=dict(color='#4ECDC4', width=3),
                        fill='tozeroy',
                        fillcolor='rgba(78, 205, 196, 0.2)',
                        hovertemplate='McClellan: %{y:.1f}<extra></extra>'
                    ))
                    
                    fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5)
                    fig.add_hline(y=50, line_dash="dash", line_color="green", opacity=0.3)
                    fig.add_hline(y=-50, line_dash="dash", line_color="red", opacity=0.3)
                    
                    fig.update_layout(
                        title="",
                        xaxis_title="Date",
                        yaxis_title="Oscillator Value",
                        height=500,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, width='stretch')
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        current_mc = breadth_history['mcclellan'].iloc[-1]
                        st.metric("Current Value", f"{current_mc:+.1f}")
                    with col2:
                        mc_max = breadth_history['mcclellan'].max()
                        st.metric("90-Day High", f"{mc_max:+.1f}")
                    with col3:
                        mc_min = breadth_history['mcclellan'].min()
                        st.metric("90-Day Low", f"{mc_min:+.1f}")

            st.divider()

            # ========== INSTITUTIONAL BREADTH SECTION ==========
            st.subheader(" Institutional Breadth Metrics")
            st.caption("Advanced breadth analysis used by professional traders")

            # Initialize enhanced breadth analyzer
            enhanced_analyzer = EnhancedBreadthAnalyzer()

            # Get enhanced metrics
            with st.spinner("Calculating institutional metrics..."):
                summation = enhanced_analyzer.calculate_mcclellan_summation(breadth_history)
                regime = enhanced_analyzer.classify_breadth_regime(breadth_history)

            # Top row: Summation Index and Regime
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("##### McClellan Summation Index")
                if summation['value'] is not None:
                    sum_val = summation['value']
                    sum_color = summation['color']
                    sum_signal = summation['signal'].replace('_', ' ')

                    st.markdown(
                        f"<div style='text-align: center; padding: 1.5rem; background: {sum_color}20; border-radius: 0.5rem;'>"
                        f"<h1 style='margin:0; color: {sum_color};'>{sum_val:+,.0f}</h1>"
                        f"<h4 style='margin:0.5rem 0; color: {sum_color};'>{sum_signal}</h4>"
                        f"<p style='margin:0;'>{summation['description']}</p>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                    with st.expander("Summation Index Guide"):
                        st.markdown("""
                        **McClellan Summation Index** = Cumulative sum of McClellan Oscillator

                        | Range | Interpretation |
                        |-------|----------------|
                        | > +1000 | Strong bull market regime |
                        | +500 to +1000 | Moderately bullish |
                        | -500 to +500 | Neutral/transitional |
                        | -1000 to -500 | Moderately bearish |
                        | < -1000 | Strong bear market regime |

                        *Zero line crossings often signal trend changes.*
                        """)
                else:
                    st.warning("Insufficient data for Summation Index")

            with col2:
                st.markdown("##### Breadth Regime Classification")
                if regime['regime'] != 'UNKNOWN':
                    regime_color = regime['color']
                    regime_name = regime['regime'].replace('_', ' ')

                    st.markdown(
                        f"<div style='text-align: center; padding: 1.5rem; background: {regime_color}20; border-radius: 0.5rem;'>"
                        f"<h1 style='margin:0; color: {regime_color};'>{regime_name}</h1>"
                        f"<h4 style='margin:0.5rem 0;'>Score: {regime['score']}/100</h4>"
                        f"<p style='margin:0;'>{regime['description']}</p>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                    # Show regime components
                    with st.expander("Regime Components"):
                        for component in regime.get('components', []):
                            st.write(f"- {component}")

                        if 'details' in regime:
                            d = regime['details']
                            st.markdown(f"""
                            **Current Breadth:** {d.get('current_breadth', 'N/A'):.1f}%
                            **5-Day Avg:** {d.get('avg_5d', 'N/A'):.1f}%
                            **20-Day Avg:** {d.get('avg_20d', 'N/A'):.1f}%
                            **McClellan:** {d.get('mcclellan', 0):+.1f}
                            """)
                else:
                    st.warning("Insufficient data for regime classification")

            st.divider()

            # New Highs vs New Lows (real data from stock analysis)
            st.markdown("##### 52-Week New Highs vs New Lows")
            st.caption("Calculated from 50 representative S&P 500 stocks - shows market leadership quality")

            with st.spinner("Analyzing 52-week highs/lows..."):
                nh_nl = get_new_highs_lows()

            if nh_nl.get('new_highs') is not None:
                col1, col2, col3, col4 = st.columns(4)

                nh_color = nh_nl['color']

                with col1:
                    st.markdown(
                        f"<div style='text-align: center; padding: 1rem; background: #4CAF5020; border-radius: 0.5rem;'>"
                        f"<h4 style='margin:0;'>New Highs</h4>"
                        f"<h2 style='margin:0.3rem 0; color: #4CAF50;'>{nh_nl['new_highs']}</h2>"
                        f"<p style='margin:0;'>{nh_nl['pct_at_high']:.0f}% of sample</p>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                with col2:
                    st.markdown(
                        f"<div style='text-align: center; padding: 1rem; background: #F4433620; border-radius: 0.5rem;'>"
                        f"<h4 style='margin:0;'>New Lows</h4>"
                        f"<h2 style='margin:0.3rem 0; color: #F44336;'>{nh_nl['new_lows']}</h2>"
                        f"<p style='margin:0;'>{nh_nl['pct_at_low']:.0f}% of sample</p>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                with col3:
                    net = nh_nl['net']
                    net_color = '#4CAF50' if net > 0 else '#F44336' if net < 0 else '#FF9800'
                    st.markdown(
                        f"<div style='text-align: center; padding: 1rem; background: {net_color}20; border-radius: 0.5rem;'>"
                        f"<h4 style='margin:0;'>Net</h4>"
                        f"<h2 style='margin:0.3rem 0; color: {net_color};'>{net:+d}</h2>"
                        f"<p style='margin:0;'>Highs - Lows</p>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                with col4:
                    st.markdown(
                        f"<div style='text-align: center; padding: 1rem; background: {nh_color}20; border-radius: 0.5rem;'>"
                        f"<h4 style='margin:0;'>Signal</h4>"
                        f"<h2 style='margin:0.3rem 0; color: {nh_color};'>{nh_nl['signal'].replace('_', ' ')}</h2>"
                        f"<p style='margin:0;'>{nh_nl['total_stocks']} stocks analyzed</p>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                st.info(f" {nh_nl['description']}")
            else:
                st.warning("Could not fetch new highs/lows data")

            st.divider()

            # % Above 200 SMA - Market Health Indicator
            st.markdown("##### Percent of Stocks Above 200-Day SMA")
            st.caption("Institutional health check - measures percentage of stocks in long-term uptrends")

            # Add button to fetch (since this takes time)
            if st.button(" Calculate % Above 200 SMA", help="Analyzes 50 stocks - takes 30-60 seconds"):
                with st.spinner("Analyzing 50 stocks vs their 200-day SMA..."):
                    sma_result = enhanced_analyzer.calculate_percent_above_sma(period=200)

                if sma_result.get('percentage') is not None:
                    col1, col2, col3, col4 = st.columns(4)

                    sma_color = sma_result['color']
                    pct = sma_result['percentage']

                    with col1:
                        st.markdown(
                            f"<div style='text-align: center; padding: 1rem; background: {sma_color}20; border-radius: 0.5rem;'>"
                            f"<h4 style='margin:0;'>% Above 200 SMA</h4>"
                            f"<h1 style='margin:0.3rem 0; color: {sma_color};'>{pct:.0f}%</h1>"
                            f"<p style='margin:0;'>{sma_result['signal'].replace('_', ' ')}</p>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                    with col2:
                        st.metric("Stocks Above", f"{sma_result['stocks_above']}")

                    with col3:
                        st.metric("Stocks Below", f"{sma_result['stocks_below']}")

                    with col4:
                        st.metric("Total Analyzed", f"{sma_result['total_stocks']}")

                    st.info(f" {sma_result['description']}")

                    # Show strongest/weakest stocks
                    with st.expander("Top 10 Strongest vs Weakest"):
                        col_a, col_b = st.columns(2)

                        with col_a:
                            st.markdown("**Strongest (Furthest Above SMA)**")
                            for stock in sma_result.get('above_list', [])[:5]:
                                st.write(f"  {stock['ticker']}: {stock['pct_from_sma']:+.1f}% above SMA")

                        with col_b:
                            st.markdown("**Weakest (Furthest Below SMA)**")
                            for stock in sma_result.get('below_list', [])[:5]:
                                st.write(f"  {stock['ticker']}: {stock['pct_from_sma']:+.1f}% from SMA")
                else:
                    st.warning("Could not calculate % above SMA")
            else:
                st.info(" Click the button above to calculate (takes 30-60 seconds)")

            st.divider()

            with st.expander(" How to Interpret These Signals"):
                st.markdown("""
                ### Basic Breadth Metrics

                **A/D Ratio:**
                - **>2.5x:** Strong buying pressure
                - **1.5-2.5x:** Moderate buying
                - **0.67-1.5x:** Neutral/balanced
                - **0.4-0.67x:** Moderate selling
                - **<0.4x:** Strong selling pressure

                **Zweig Breadth Thrust:**
                - Rare signal (happens few times per decade)
                - Triggers when 10-day EMA of breadth goes from <40% to >61.5%
                - Historically precedes strong bull moves

                **Divergence:**
                - **Bearish:** Price new high BUT breadth weak = warning
                - **Bullish:** Price new low BUT breadth holds = potential bottom
                - **None:** Price and breadth aligned = healthy

                **Z-Score:**
                - **>+2σ:** Extremely strong (rare)
                - **+1 to +2σ:** Above average strength
                - **-1 to +1σ:** Normal range
                - **<-2σ:** Extremely weak (rare)

                ---

                ### Institutional Breadth Metrics

                **McClellan Summation Index:**
                - Cumulative version of McClellan Oscillator
                - **>+1000:** Strong bull market regime
                - **+500 to +1000:** Moderately bullish
                - **-500 to +500:** Transitional/neutral
                - **-1000 to -500:** Moderately bearish
                - **<-1000:** Strong bear market regime

                **Breadth Regime Classification:**
                - Combines multiple breadth factors into single regime score
                - Components: current breadth, trend momentum, McClellan direction, consistency
                - Regimes: STRONG BULL → BULL → NEUTRAL → BEAR → STRONG BEAR

                **New Highs vs New Lows:**
                - Measures stocks at 52-week extremes
                - More new highs = healthy leadership
                - More new lows = deteriorating internals
                - Extreme readings often mark market turns

                **% Above 200 SMA:**
                - **>70%:** Strong bull market - most stocks in uptrends
                - **50-70%:** Healthy market conditions
                - **30-50%:** Breadth weakening - caution warranted
                - **<30%:** Bear market conditions - most stocks in downtrends

                ---
                **Data Source:** Calculated from 50-100 representative S&P 500 stocks using Yahoo Finance (real-time data).
                """)
        
        else:
            st.warning("No breadth data available. Click 'Update Data' to calculate.")
    
    except Exception as e:
        st.error(f"Error loading breadth signals: {e}")
        import traceback
        st.code(traceback.format_exc())

elif page == "Treasury Stress (MOVE)":
    st.markdown(
        "<h1 class='main-header'> Treasury Market Stress</h1>",
        unsafe_allow_html=True,
    )
    
    try:
        move_snapshot = components["move"].get_full_snapshot()
        
        if not move_snapshot or 'move_index' not in move_snapshot:
            st.error("MOVE Index data unavailable. Please check data collection.")
            st.stop()
        
        move_df = move_snapshot.get('move_df')
        
        # Get VIX history
        try:
            vix_df = components["market"].get_vix_history(lookback_days=365)
        except Exception:
            vix_df = pd.DataFrame()
        
        if move_df is not None and not move_df.empty and not vix_df.empty:
            vix_series = vix_df['close'] if 'close' in vix_df.columns else vix_df.iloc[:, 0]
            treasury_signal = components["treasury_analyzer"].analyze(move_df, vix_series)
        else:
            treasury_signal = None
    
    except Exception as e:
        st.error(f"Error loading Treasury data: {e}")
        logger.error(f"Error in treasury_stress page: {e}")
        st.stop()
    
    st.subheader(" Current Treasury Market Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        move_value = move_snapshot['move_index']
        stress_level = move_snapshot['stress_level']
        color = get_status_color(stress_level)
        
        st.markdown(
            f"<div style='text-align: center; padding: 1.5rem; background: {color}20; border-radius: 0.5rem;'>"
            f"<h4 style='margin:0;'>MOVE Index</h4>"
            f"<h1 style='margin:0.5rem 0; color: {color};'>{move_value:.1f}</h1>"
            f"<h3 style='margin:0; color: {color};'>{stress_level}</h3>"
            f"</div>",
            unsafe_allow_html=True,
        )
    
    with col2:
        percentile = move_snapshot.get('percentile', 50)
        
        if percentile < 25:
            desc = "Very Calm"
            color = "#4CAF50"
        elif percentile < 50:
            desc = "Calm"
            color = "#8BC34A"
        elif percentile < 75:
            desc = "Active"
            color = "#FF9800"
        else:
            desc = "Elevated"
            color = "#F44336"
        
        st.markdown(
            f"<div style='text-align: center; padding: 1.5rem; background: {color}20; border-radius: 0.5rem;'>"
            f"<h4 style='margin:0;'>Historical Percentile</h4>"
            f"<h1 style='margin:0.5rem 0; color: {color};'>{percentile:.0f}th</h1>"
            f"<h3 style='margin:0; color: {color};'>{desc}</h3>"
            f"</div>",
            unsafe_allow_html=True,
        )
    
    with col3:
        if treasury_signal:
            stress_regime = treasury_signal.stress_level
            strength = treasury_signal.strength
            color = get_status_color(stress_regime)
            
            st.markdown(
                f"<div style='text-align: center; padding: 1.5rem; background: {color}20; border-radius: 0.5rem;'>"
                f"<h4 style='margin:0;'>Treasury Regime</h4>"
                f"<h1 style='margin:0.5rem 0; color: {color};'>{stress_regime}</h1>"
                f"<p style='margin:0;'>Strength: {strength:.0f}/100</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.warning("Analysis unavailable")
    
    with col4:
        if treasury_signal and hasattr(treasury_signal, 'divergence_type') and treasury_signal.divergence_type:
            div_type = treasury_signal.divergence_type
            
            if "Leading" in div_type:
                color = "#FF9800"
                icon = "⚠️"
            else:
                color = "#4CAF50"
                icon = "✓"
            
            st.markdown(
                f"<div style='text-align: center; padding: 1.5rem; background: {color}20; border-radius: 0.5rem;'>"
                f"<h4 style='margin:0;'>MOVE-VIX Divergence</h4>"
                f"<h2 style='margin:0.5rem 0; color: {color};'>{icon}</h2>"
                f"<p style='margin:0; font-size: 0.9rem;'>{div_type}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("Divergence analysis unavailable")
    
    st.markdown("---")
    
    with st.expander("ℹ️ What is the MOVE Index?", expanded=False):
        st.markdown("""
        **The MOVE Index** (Merrill Option Volatility Estimate) is like the VIX but for Treasury bonds:
        
        - **< 80**: Low stress - calm Treasury market  
        - **80-120**: Normal stress - typical market conditions  
        - **120-150**: Elevated stress - increased uncertainty  
        - **> 150**: High stress - crisis territory
        
        Treasury stress often **precedes** equity stress.
        """)
    
    tab1, tab2, tab3 = st.tabs([" MOVE History", " MOVE vs VIX", "Treasury Stress Regime"])
    
    with tab1:
        st.subheader("MOVE Index Historical Chart")
        
        if move_df is not None and not move_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=move_df['date'],
                y=move_df['move'],
                name='MOVE Index',
                line=dict(color='#1f77b4', width=2),
                fill='tozeroy',
                fillcolor='rgba(31, 119, 180, 0.2)'
            ))
            fig.add_hline(y=80, line_dash="dash", line_color="green",
                         annotation_text="Normal Lower", annotation_position="right")
            fig.add_hline(y=120, line_dash="dash", line_color="orange",
                         annotation_text="Elevated", annotation_position="right")
            fig.add_hline(y=150, line_dash="dash", line_color="red",
                         annotation_text="High Stress", annotation_position="right")
            fig.update_layout(
                title="MOVE Index - Last 2 Years",
                xaxis_title="Date",
                yaxis_title="MOVE Index",
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig, width='stretch')
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current MOVE", f"{move_df['move'].iloc[-1]:.1f}")
            with col2:
                st.metric("30-Day Average", f"{move_df['move'].tail(30).mean():.1f}")
            with col3:
                st.metric("1-Year High", f"{move_df['move'].max():.1f}")
        else:
            st.warning("MOVE historical data unavailable")
    
    with tab2:
        st.subheader("MOVE vs VIX Divergence Analysis")
        
        if move_df is not None and not move_df.empty and not vix_df.empty:
            merged = move_df.merge(vix_df[['date', 'close']], on='date', how='inner')
            merged.rename(columns={'close': 'vix'}, inplace=True)
            
            merged['move_norm'] = (merged['move'] - merged['move'].min()) / (merged['move'].max() - merged['move'].min()) * 100
            merged['vix_norm'] = (merged['vix'] - merged['vix'].min()) / (merged['vix'].max() - merged['vix'].min()) * 100
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=merged['date'],
                y=merged['move_norm'],
                name='MOVE (normalized)',
                line=dict(color='#1f77b4', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=merged['date'],
                y=merged['vix_norm'],
                name='VIX (normalized)',
                line=dict(color='#F44336', width=2)
            ))
            fig.update_layout(
                title="MOVE vs VIX - Normalized Comparison",
                xaxis_title="Date",
                yaxis_title="Normalized Value (0-100)",
                hovermode='x unified',
                height=500,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig, width='stretch')
            
            if treasury_signal and hasattr(treasury_signal, 'divergence_type') and treasury_signal.divergence_type:
                st.info(f"**Current Divergence:** {treasury_signal.divergence_type}")
                if hasattr(treasury_signal, 'description'):
                    st.write(treasury_signal.description)
        else:
            st.warning("Insufficient data for divergence analysis")
    
    with tab3:
        st.subheader("Treasury Stress Regime Classification")
        
        if treasury_signal and move_df is not None and not move_df.empty:
            regimes = []
            for _, row in move_df.iterrows():
                if row['move'] < 80:
                    regimes.append(('LOW', '#4CAF50'))
                elif row['move'] < 120:
                    regimes.append(('NORMAL', '#8BC34A'))
                elif row['move'] < 150:
                    regimes.append(('ELEVATED', '#FF9800'))
                else:
                    regimes.append(('HIGH', '#F44336'))
            
            move_df['regime'] = [r[0] for r in regimes]
            move_df['regime_color'] = [r[1] for r in regimes]
            
            fig = go.Figure()
            for regime_name, color in [('LOW', '#4CAF50'), ('NORMAL', '#8BC34A'),
                                       ('ELEVATED', '#FF9800'), ('HIGH', '#F44336')]:
                regime_data = move_df[move_df['regime'] == regime_name]
                if not regime_data.empty:
                    fig.add_trace(go.Scatter(
                        x=regime_data['date'],
                        y=[1] * len(regime_data),
                        name=regime_name,
                        mode='markers',
                        marker=dict(color=color, size=8, symbol='square')
                    ))
            fig.update_layout(
                title="Treasury Stress Regime Over Time",
                xaxis_title="Date",
                yaxis_title="",
                yaxis=dict(showticklabels=False),
                hovermode='x unified',
                height=200,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.warning("Regime classification unavailable")

# ============================================================
# REPO MARKET (SOFR) - Phase 2
# ============================================================
elif page == "Repo Market (SOFR)":
    st.markdown(
        "<h1 class='main-header'>💰 Repo Market & Liquidity Stress</h1>",
        unsafe_allow_html=True,
    )
    
    try:
        repo_snapshot = components["repo"].get_full_snapshot()
        
        if not repo_snapshot or 'sofr' not in repo_snapshot:
            st.error("Repo market data unavailable. Please check data collection.")
            st.stop()
        
        repo_df = repo_snapshot.get('repo_df')
        
        if repo_df is not None and not repo_df.empty:
            repo_signal = components["repo_analyzer"].analyze(repo_df)
        else:
            repo_signal = None
            
    except Exception as e:
        st.error(f"Error loading Repo data: {e}")
        logger.error(f"Error in repo_stress page: {e}")
        st.stop()
    
    st.subheader("📊 Current Repo Market Status")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        sofr = repo_snapshot['sofr']
        color = "#1f77b4"
        
        st.markdown(
            f"<div style='text-align: center; padding: 1.5rem; background: {color}20; border-radius: 0.5rem;'>"
            f"<h4 style='margin:0;'>SOFR Rate</h4>"
            f"<h1 style='margin:0.5rem 0; color: {color};'>{sofr:.2f}%</h1>"
            f"<p style='margin:0; font-size:0.9em;'>Secured Overnight</p>"
            f"</div>",
            unsafe_allow_html=True,
        )
    
    with col2:
        if 'iorb' in repo_snapshot:
            iorb = repo_snapshot['iorb']
            color = "#9C27B0"
            
            st.markdown(
                f"<div style='text-align: center; padding: 1.5rem; background: {color}20; border-radius: 0.5rem;'>"
                f"<h4 style='margin:0;'>IORB Rate</h4>"
                f"<h1 style='margin:0.5rem 0; color: {color};'>{iorb:.2f}%</h1>"
                f"<p style='margin:0; font-size:0.9em;'>Fed Floor Rate</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.warning("IORB unavailable")
    
    with col3:
        if 'sofr_iorb_spread' in repo_snapshot:
            spread = repo_snapshot['sofr_iorb_spread']
            status = repo_snapshot.get('liquidity_status', 'UNKNOWN')
            color = repo_snapshot.get('liquidity_color', '#9E9E9E')
            
            st.markdown(
                f"<div style='text-align: center; padding: 1.5rem; background: {color}20; border-radius: 0.5rem;'>"
                f"<h4 style='margin:0;'>SOFR-IORB Spread</h4>"
                f"<h1 style='margin:0.5rem 0; color: {color};'>{spread:+.1f} bps</h1>"
                f"<p style='margin:0; font-size:0.9em;'>{status}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.warning("Spread unavailable")
    
    with col4:
        if repo_signal:
            stress_level = repo_signal.stress_level
            strength = repo_signal.strength
            color = get_status_color(stress_level)
            
            st.markdown(
                f"<div style='text-align: center; padding: 1.5rem; background: {color}20; border-radius: 0.5rem;'>"
                f"<h4 style='margin:0;'>Funding Stress</h4>"
                f"<h1 style='margin:0.5rem 0; color: {color};'>{stress_level}</h1>"
                f"<p style='margin:0; font-size:0.9em;'>Strength: {strength:.0f}/100</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.warning("Stress analysis unavailable")
    
    with col5:
        rrp_volume = repo_snapshot.get('rrp_volume', 0)
        
        if rrp_volume < 50:
            color = "#F44336"
            desc = "Depleted"
        elif rrp_volume < 500:
            color = "#FF9800"
            desc = "Low"
        else:
            color = "#4CAF50"
            desc = "Elevated"
        
        st.markdown(
            f"<div style='text-align: center; padding: 1.5rem; background: {color}20; border-radius: 0.5rem;'>"
            f"<h4 style='margin:0;'>RRP Volume</h4>"
            f"<h1 style='margin:0.5rem 0; color: {color};'>${rrp_volume:.0f}B</h1>"
            f"<p style='margin:0; font-size:0.9em;'>{desc}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )
    
    st.markdown("---")
    
    with st.expander("ℹ️ What is the Repo Market, SOFR & IORB?", expanded=False):
        st.markdown("""
        **The Repo Market** is where the financial system gets overnight funding:
        
        **SOFR (Secured Overnight Financing Rate):**
        - Benchmark rate for overnight Treasury repo transactions  
        - Replaced LIBOR as primary U.S. rate benchmark  
        - Reflects actual market funding conditions
        
        **IORB (Interest on Reserve Balances):**
        - Rate the Fed pays banks on reserves held at the Fed
        - Acts as a "floor" for overnight rates
        - Changed via Fed policy (FOMC decisions)
        
        **SOFR-IORB Spread = KEY LIQUIDITY INDICATOR:**
        - **Normal**: 5-15 bps positive spread
        - **Abundant**: SOFR ≤ IORB or 0-5 bps (plenty of cash)
        - **Tightening**: 15-30 bps (liquidity getting scarce)
        - **Stress**: >30 bps (funding pressure building)
        - **Crisis**: >50-100 bps (September 2019: 390 bps!)
        
        **Why it matters:**
        - Wide spread = Banks prefer lending in repo over holding reserves
        - Signals declining liquidity and potential funding stress
        - Fed monitors closely to gauge reserve adequacy
        """)
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 SOFR vs IORB", 
        "📊 SOFR-IORB Spread", 
        "📉 Z-Score & Stress", 
        "💵 RRP Volume"
    ])
    
    with tab1:
        st.subheader("SOFR vs IORB - Funding Rate Comparison")
        
        if repo_df is not None and not repo_df.empty and 'iorb' in repo_df.columns:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=repo_df['date'],
                y=repo_df['sofr'],
                name='SOFR (Market Rate)',
                line=dict(color='#1f77b4', width=2),
            ))
            
            fig.add_trace(go.Scatter(
                x=repo_df['date'],
                y=repo_df['iorb'],
                name='IORB (Fed Floor)',
                line=dict(color='#9C27B0', width=2, dash='dash'),
            ))
            
            fig.update_layout(
                title="SOFR vs IORB - Last 2 Years",
                xaxis_title="Date",
                yaxis_title="Rate (%)",
                hovermode='x unified',
                height=500,
                legend=dict(x=0.01, y=0.99)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current SOFR", f"{repo_df['sofr'].iloc[-1]:.2f}%")
            with col2:
                st.metric("Current IORB", f"{repo_df['iorb'].iloc[-1]:.2f}%")
            with col3:
                current_spread = repo_df['sofr_iorb_spread'].iloc[-1]
                st.metric("Spread", f"{current_spread:+.1f} bps")
            with col4:
                avg_spread = repo_df['sofr_iorb_spread'].tail(252).mean()
                st.metric("1Y Avg Spread", f"{avg_spread:+.1f} bps")
            
            if current_spread <= 5:
                st.success("✅ **ABUNDANT LIQUIDITY**: SOFR trading at or below IORB - funding conditions very easy")
            elif current_spread <= 15:
                st.info("➡️ **NORMAL CONDITIONS**: SOFR slightly above IORB - healthy funding market")
            elif current_spread <= 30:
                st.warning("⚠️ **TIGHTENING**: SOFR-IORB spread widening - liquidity starting to thin")
            else:
                st.error("🚨 **FUNDING STRESS**: Wide spread indicates significant liquidity pressure")
        else:
            st.warning("IORB data unavailable for comparison")
    
    with tab2:
        st.subheader("SOFR-IORB Spread History (Liquidity Indicator)")
        
        if repo_df is not None and not repo_df.empty and 'sofr_iorb_spread' in repo_df.columns:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=repo_df['date'],
                y=repo_df['sofr_iorb_spread'],
                name='SOFR-IORB Spread',
                line=dict(color='#2E86AB', width=2),
                fill='tozeroy',
                fillcolor='rgba(46, 134, 171, 0.3)'
            ))
            
            fig.add_hrect(y0=-5, y1=5, fillcolor="green", opacity=0.1,
                         annotation_text="ABUNDANT", annotation_position="left")
            fig.add_hrect(y0=5, y1=15, fillcolor="lightgreen", opacity=0.1,
                         annotation_text="NORMAL", annotation_position="left")
            fig.add_hrect(y0=15, y1=30, fillcolor="orange", opacity=0.1,
                         annotation_text="TIGHTENING", annotation_position="left")
            fig.add_hrect(y0=30, y1=100, fillcolor="red", opacity=0.1,
                         annotation_text="STRESS", annotation_position="left")
            
            fig.add_hline(y=0, line_dash="solid", line_color="gray", 
                         annotation_text="IORB Floor")
            
            fig.update_layout(
                title="SOFR-IORB Spread with Liquidity Bands",
                xaxis_title="Date",
                yaxis_title="Spread (basis points)",
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                current = repo_df['sofr_iorb_spread'].iloc[-1]
                st.metric("Current Spread", f"{current:+.1f} bps")
            with col2:
                max_spread = repo_df['sofr_iorb_spread'].max()
                st.metric("2Y Maximum", f"{max_spread:+.1f} bps")
            with col3:
                days_above_15 = (repo_df['sofr_iorb_spread'] > 15).sum()
                pct = (days_above_15 / len(repo_df)) * 100
                st.metric("Days >15 bps", f"{days_above_15} ({pct:.1f}%)")
            
            st.info("""
            **Interpretation Guide:**
            - **Negative or 0-5 bps**: Abundant liquidity, SOFR at or below floor
            - **5-15 bps**: Normal conditions, typical small premium
            - **15-30 bps**: Liquidity tightening, monitor for escalation
            - **>30 bps**: Funding stress, potential liquidity shortage
            - **>50 bps**: Severe stress (see September 2019: 390 bps spike!)
            """)
        else:
            st.warning("Spread data unavailable")
    
    with tab3:
        st.subheader("SOFR Z-Score & Stress Bands")
        
        if repo_df is not None and not repo_df.empty and 'sofr_z_score' in repo_df.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=repo_df['date'],
                y=repo_df['sofr_z_score'],
                name='SOFR Z-Score',
                line=dict(color='#1f77b4', width=2)
            ))
            fig.add_hrect(y0=-1, y1=1, fillcolor="green", opacity=0.1,
                         annotation_text="NORMAL", annotation_position="left")
            fig.add_hrect(y0=1, y1=2, fillcolor="orange", opacity=0.1,
                         annotation_text="ELEVATED", annotation_position="left")
            fig.add_hrect(y0=2, y1=5, fillcolor="red", opacity=0.1,
                         annotation_text="STRESS", annotation_position="left")
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(
                title="SOFR Z-Score with Stress Bands",
                xaxis_title="Date",
                yaxis_title="Standard Deviations (σ)",
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            if repo_signal and hasattr(repo_signal, 'description'):
                st.info(f"**Current Status:** {repo_signal.description}")
        else:
            st.warning("Z-score data unavailable")
    
    with tab4:
        st.subheader("Overnight RRP Volume")
        
        if repo_df is not None and not repo_df.empty and 'rrp_on' in repo_df.columns:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=repo_df['date'],
                y=repo_df['rrp_on'],
                name='RRP Volume',
                marker_color='#F44336'
            ))
            fig.update_layout(
                title="Overnight Reverse Repo (RRP) Volume - Last 2 Years",
                xaxis_title="Date",
                yaxis_title="Volume ($B)",
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            current_rrp = repo_df['rrp_on'].iloc[-1]
            peak_rrp = repo_df['rrp_on'].max()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Current RRP", f"${current_rrp:.0f}B")
            with col2:
                st.metric(
                    "Peak RRP",
                    f"${peak_rrp:.0f}B",
                    f"{((current_rrp - peak_rrp) / peak_rrp * 100):.1f}% vs peak"
                )
        else:
            st.warning("RRP volume data unavailable")

# ============================================================
# COT POSITIONING (CFTC Institutional Data)
# ============================================================
elif page == "COT Positioning":
    st.markdown("<h1 class='main-header'>📊 CFTC Commitments of Traders</h1>", unsafe_allow_html=True)

    st.markdown("""
    **Institutional futures positioning** from CFTC weekly reports. Shows how hedge funds (speculators)
    and commercial hedgers are positioned. Extreme positioning often precedes reversals.
    """)

    st.info("""
    **How to read COT data:**
    - **Speculators (Non-Commercial)**: Hedge funds, CTAs - trend followers
    - **Commercials**: Hedgers (producers/consumers) - usually fade extremes
    - **Extreme Long**: Speculators max long → potential top (contrarian sell)
    - **Extreme Short**: Speculators max short → potential bottom (contrarian buy)
    """)

    st.divider()

    # Initialize COT collector
    @st.cache_resource
    def get_cot_collector():
        return COTCollector()

    cot = get_cot_collector()

    # Fetch positioning extremes
    with st.spinner("Analyzing institutional positioning..."):
        try:
            extremes = cot.get_all_extremes(weeks_back=52)
        except Exception as e:
            st.error(f"Error fetching COT data: {e}")
            extremes = []

    if not extremes:
        st.warning("""
        ⚠️ **COT data unavailable**

        CFTC data requires downloading annual ZIP files. This may take a moment on first load.
        If this persists, the CFTC website may be temporarily unavailable.
        """)
    else:
        # Summary of extreme positions
        st.subheader("🎯 Positioning Extremes")

        extreme_signals = [e for e in extremes if 'EXTREME' in e.get('signal', '')]

        if extreme_signals:
            st.warning(f"**{len(extreme_signals)} market(s) at extreme positioning levels**")

            for ext in extreme_signals:
                pct = ext['percentile']
                if ext['signal'] == 'EXTREME_LONG':
                    st.error(f"🔴 **{ext['name']}**: Speculators {pct:.0f}th percentile LONG - Contrarian SELL signal")
                else:
                    st.success(f"🟢 **{ext['name']}**: Speculators {pct:.0f}th percentile SHORT - Contrarian BUY signal")
        else:
            st.success("✅ No extreme positioning detected across tracked markets")

        st.divider()

        # Detailed positioning table
        st.subheader("📋 Current Positioning by Market")

        # Group by category
        categories = {}
        for ext in extremes:
            cat = cot.CONTRACTS.get(ext['symbol'], {}).get('category', 'other')
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(ext)

        # Display by category
        for category, items in categories.items():
            st.markdown(f"**{category.upper()}**")

            cols = st.columns(len(items)) if len(items) <= 4 else st.columns(4)

            for i, item in enumerate(items[:4]):
                with cols[i % 4]:
                    pct = item['percentile']

                    # Color based on percentile
                    if pct >= 80:
                        color = "#F44336"  # Red - extreme long
                        status = "Crowded Long"
                    elif pct >= 60:
                        color = "#FF9800"  # Orange
                        status = "Long"
                    elif pct <= 20:
                        color = "#4CAF50"  # Green - extreme short
                        status = "Crowded Short"
                    elif pct <= 40:
                        color = "#8BC34A"  # Light green
                        status = "Short"
                    else:
                        color = "#9E9E9E"  # Gray
                        status = "Neutral"

                    st.markdown(
                        f"""<div style='text-align: center; padding: 1rem;
                        background: {color}20; border-radius: 0.5rem; border-left: 4px solid {color};'>
                        <h4 style='margin:0;'>{item['symbol']}</h4>
                        <p style='margin:0.2rem 0; font-size: 0.85em;'>{item['name']}</p>
                        <h2 style='margin:0.5rem 0; color: {color};'>{pct:.0f}th</h2>
                        <p style='margin:0; color: {color};'>{status}</p>
                        </div>""",
                        unsafe_allow_html=True
                    )

                    # Net position info
                    if 'current_net' in item:
                        net = item['current_net']
                        st.caption(f"Net: {net:+,} contracts")

            st.markdown("")  # Spacing

        st.divider()

        # Historical context
        st.subheader("📈 How to Use This Data")

        st.markdown("""
        **Contrarian Signals:**
        - When speculators hit **90th+ percentile** long, they're "all in" → less buying power left
        - When speculators hit **10th percentile** short, they're max bearish → short covering rally potential

        **Confirmation Signals:**
        - Extreme positioning + price reversal = higher probability trade
        - COT data is **weekly** (released Fridays) - use for swing/position trades, not day trading

        **Key Markets to Watch:**
        - **ES (S&P 500)**: Overall equity sentiment
        - **TY (10Y Treasury)**: Bond market positioning
        - **GC (Gold)**: Safe haven flows
        - **DX (Dollar)**: Currency/risk sentiment
        - **VX (VIX)**: Volatility positioning
        """)

elif page == "CTA Flow Tracker":
    st.markdown("<h1 class='main-header'> CTA Flow Tracker</h1>", unsafe_allow_html=True)

    st.markdown("""
    Track systematic trend-following flows. CTAs control $400B+ and create self-reinforcing
    momentum at key technical levels. **Flip levels** show where CTAs likely adjust positions.
    """)
    
    # Get cached CTA collector
    cta_collector = get_cta_collector()
    
    # Control buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button(" Update Prices", help="Fetch latest price data (incremental)"):
            with st.spinner("Updating prices..."):
                cta_collector.update_prices()
                st.success("✓ Prices updated")
                st.rerun()
    
    with col2:
        if st.button(" Refresh Analysis", help="Recompute CTA signals"):
            st.session_state["cta_refresh"] = st.session_state.get("cta_refresh", 0) + 1
    
    # Run analysis (cached by session state)
    refresh_key = st.session_state.get("cta_refresh", 0)
    
    @st.cache_data(ttl=900)  # Cache for 15 minutes
    def get_cta_result(_refresh):
        return cta_collector.get_cta_analysis()
    
    with st.spinner("Running CTA analysis..."):
        result = get_cta_result(refresh_key)
    
    if result is None:
        st.warning(" No CTA data available. Click 'Update Prices' to fetch data.")
    else:
        # Asset class groupings
        equities = ["SPY", "QQQ", "IWM", "EEM"]
        bonds = ["TLT", "HYG"]
        commodities = ["GLD"]
        fx = ["UUP"]
        
        def safe_sum(symbols):
            return result.latest_exposure[[s for s in symbols if s in result.latest_exposure.index]].sum()
        
        eq_exp = safe_sum(equities)
        bond_exp = safe_sum(bonds)
        comm_exp = safe_sum(commodities)
        fx_exp = safe_sum(fx)
        total_gross = result.latest_exposure.abs().sum()
        
        # Top metrics
        st.subheader("🎯 Aggregate CTA Positioning")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        def render_exposure_card(col, title, exposure):
            color = "#4CAF50" if exposure > 0.1 else ("#f44336" if exposure < -0.1 else "#757575")
            state = 'LONG' if exposure > 0.1 else ('SHORT' if exposure < -0.1 else 'FLAT')
            col.markdown(
                f"<div style='text-align: center; padding: 1rem; background: {color}20; border-radius: 0.5rem;'>"
                f"<h4 style='margin:0;'>{title}</h4>"
                f"<h2 style='margin:0; color: {color};'>{exposure:+.2f}</h2>"
                f"<p style='margin:0; font-size: 0.8rem;'>{state}</p>"
                f"</div>",
                unsafe_allow_html=True
            )
        
        render_exposure_card(col1, "Equities", eq_exp)
        render_exposure_card(col2, "Bonds", bond_exp)
        render_exposure_card(col3, "Gold", comm_exp)
        render_exposure_card(col4, "USD", fx_exp)
        
        with col5:
            leverage_pct = (total_gross / 2.0) * 100
            st.metric("Gross Leverage", f"{leverage_pct:.0f}%", help="Total absolute exposure (max 200%)")
        
        st.divider()
        
        # Flip levels table
        st.subheader(" CTA Flip Levels")
        st.markdown("**Flip levels** = prices where momentum turns. Breaking above/below triggers CTA flows.")
        
        if not result.flip_levels.empty:
            display_data = []
            for symbol in result.flip_levels['symbol'].unique():
                row = {'Symbol': symbol}
                current = result.flip_levels[result.flip_levels['symbol'] == symbol]['current_price'].iloc[0]
                row['Current'] = f"${current:.2f}"
                
                for horizon in [21, 63, 126, 252]:
                    sym_data = result.flip_levels[
                        (result.flip_levels['symbol'] == symbol) & 
                        (result.flip_levels['horizon_days'] == horizon)
                    ]
                    if not sym_data.empty:
                        flip = sym_data['flip_price'].iloc[0]
                        dist = sym_data['distance_pct'].iloc[0]
                        row[f'{horizon}d'] = f"${flip:.2f} ({dist:+.1f}%)"
                    else:
                        row[f'{horizon}d'] = "N/A"
                
                row['State'] = result.latest_state.get(symbol, 'N/A')
                display_data.append(row)
            
            df_display = pd.DataFrame(display_data)
            st.dataframe(df_display, use_container_width=True)
        
        st.divider()
        
        # Exposure heatmap
        st.subheader("📊 Historical CTA Exposures (90 Days)")
        exposures_90d = result.exposures.tail(90)
        
        if not exposures_90d.empty:
            fig = go.Figure(data=go.Heatmap(
                z=exposures_90d.T.values,
                x=exposures_90d.index,
                y=exposures_90d.columns,
                colorscale='RdYlGn',
                zmid=0,
                colorbar=dict(title="Exposure"),
            ))
            fig.update_layout(
                title="CTA Trend Strength Over Time",
                xaxis_title="Date",
                yaxis_title="Symbol",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Signal interpretation
        st.subheader(" Trading Implications")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("** Strong Longs (>0.3)**")
            strong_longs = result.latest_exposure[result.latest_exposure > 0.3].sort_values(ascending=False)
            if not strong_longs.empty:
                for sym, exp in strong_longs.items():
                    st.markdown(f"- **{sym}**: {exp:.2f}")
            else:
                st.markdown("*None*")
        
        with col2:
            st.markdown("** Strong Shorts (<-0.3)**")
            strong_shorts = result.latest_exposure[result.latest_exposure < -0.3].sort_values()
            if not strong_shorts.empty:
                for sym, exp in strong_shorts.items():
                    st.markdown(f"- **{sym}**: {exp:.2f}")
            else:
                st.markdown("*None*")
        
        st.info(" **Strategy:** Watch price action near flip levels. Breaking above = CTA buying. Breaking below = CTA selling.")

        

# ============================================================
# SETTINGS
# ============================================================

elif page == "Settings":
    st.markdown("<h1 class='main-header'> Settings</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    Configure API keys and dashboard preferences. Changes are saved to `.env` file.
    """)
    
    # API Keys Section
    st.subheader(" API Keys")
    
    # FRED API Key
    with st.expander("FRED API Key (Required)", expanded=False):
        st.markdown("Get your free API key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        
        current_fred_key = os.getenv('FRED_API_KEY', '')
        fred_key = st.text_input(
            "FRED API Key",
            value=current_fred_key,
            type="password",
            key="fred_api_key"
        )
        
        if st.button("Save FRED API Key"):
            if fred_key:
                save_to_env('FRED_API_KEY', fred_key)
                st.success("✓ FRED API key saved! Restart dashboard to apply.")
            else:
                st.error("Please enter an API key")
    
    # Alpha Vantage API Key
    with st.expander("Alpha Vantage API Key (Optional)", expanded=False):
        st.markdown("Get your free API key at: https://www.alphavantage.co/support/#api-key")
        st.info("Used for: Alternative market data, economic indicators")
        
        current_alpha_key = os.getenv('ALPHA_VANTAGE_API_KEY', '')
        alpha_key = st.text_input(
            "Alpha Vantage API Key",
            value=current_alpha_key,
            type="password",
            key="alpha_api_key"
        )
        
        if st.button("Save Alpha Vantage API Key"):
            if alpha_key:
                save_to_env('ALPHA_VANTAGE_API_KEY', alpha_key)
                st.success("✓ Alpha Vantage API key saved! Restart dashboard to apply.")
            else:
                st.error("Please enter an API key")
    
    # Polygon API Key
    with st.expander("Polygon.io API Key (Optional)", expanded=False):
        st.markdown("Get your API key at: https://polygon.io/")
        st.info("Used for: Real-time market data, options flow")
        
        current_polygon_key = os.getenv('POLYGON_API_KEY', '')
        polygon_key = st.text_input(
            "Polygon API Key",
            value=current_polygon_key,
            type="password",
            key="polygon_api_key"
        )
        
        if st.button("Save Polygon API Key"):
            if polygon_key:
                save_to_env('POLYGON_API_KEY', polygon_key)
                st.success("✓ Polygon API key saved! Restart dashboard to apply.")
            else:
                st.error("Please enter an API key")
    
    st.divider()
    
    # Manual Overrides Section
    st.subheader(" Manual Data Overrides")
    
    with st.expander("Equity Put/Call Ratio Override", expanded=False):
        st.markdown("""
        Manually set the equity put/call ratio when automatic data is unavailable or incorrect.
        """)
        
        use_manual = st.checkbox("Use Manual Equity P/C Ratio", key="use_manual_pc")
        
        if use_manual:
            manual_pc = st.number_input(
                "Manual Equity P/C Ratio",
                min_value=0.0,
                max_value=5.0,
                value=1.0,
                step=0.01,
                help="Typical range: 0.5-2.0. Above 1.0 = bearish, below 1.0 = bullish"
            )
            
            if st.button("Save Manual P/C Ratio"):
                st.session_state['manual_equity_pc'] = manual_pc
                st.session_state['use_manual_equity_pc'] = True
                st.success(f"✓ Manual equity P/C ratio set to {manual_pc:.2f}")
        else:
            if 'use_manual_equity_pc' in st.session_state:
                st.session_state['use_manual_equity_pc'] = False
            st.info("Using automatic equity P/C ratio from CBOE")

    st.divider()

    # Market Breadth Settings
    st.subheader("📊 Market Breadth Settings")

    st.markdown("""
    **Stock Sample Size for Breadth Calculation**

    Choose between speed and precision for market breadth analysis:
    """)

    # Get current setting from env
    current_breadth_mode = os.getenv('BREADTH_MODE', 'fast').lower()

    breadth_mode = st.radio(
        "Breadth Calculation Mode",
        options=['fast', 'full'],
        index=0 if current_breadth_mode == 'fast' else 1,
        format_func=lambda x: {
            'fast': '⚡ Fast Mode (100 stocks) - Recommended',
            'full': '🎯 Full Mode (500 stocks) - More precise'
        }[x],
        key="breadth_mode_setting",
        help="Fast mode uses 100 representative stocks (~30-60 sec). Full mode uses all 500 S&P stocks (~3-5 min)."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **⚡ Fast Mode (100 stocks)**
        - ~30-60 seconds refresh
        - Representative sample
        - McClellan scaled to NYSE standard
        """)
    with col2:
        st.markdown("""
        **🎯 Full Mode (500 stocks)**
        - ~3-5 minutes refresh
        - True S&P 500 breadth
        - No scaling needed
        """)

    if st.button("Save Breadth Mode", key="save_breadth_mode_btn"):
        save_to_env('BREADTH_MODE', breadth_mode)
        st.success(f"✅ Breadth mode saved: {breadth_mode.upper()}")
        st.info("🔄 Go to Market Breadth page and click 'Refresh Breadth Data' to apply.")

    # Show current status
    mode_icon = "⚡" if current_breadth_mode == 'fast' else "🎯"
    stock_count = "100" if current_breadth_mode == 'fast' else "500"
    st.caption(f"Current mode: {mode_icon} {current_breadth_mode.upper()} ({stock_count} stocks)")

    st.divider()

    # Dashboard Preferences
    st.subheader(" Dashboard Preferences")
    
    st.markdown("""
    **Streamlit Built-in Settings** (accessible via ☰ menu in top right):
    - Theme (Light/Dark mode)
    - Wide mode toggle
    - Run on save
    - Developer options
    """)
    
    st.divider()
    
    # About Section
    st.subheader("About")
    
    st.markdown("""
    **Market Dashboard** - Institutional-grade market analysis
    
    **Core Features:**
    -  VIX & Volatility Risk Premium tracking
    -  Fed liquidity monitoring (RRP, TGA, Balance Sheet)
    -  CTA systematic flow tracking
    -  Credit spreads & repo stress
    -  Market breadth & sentiment
    
    **Data Sources:**
    - Federal Reserve Economic Data (FRED)
    - CBOE (VIX, SKEW, Options)
    - Yahoo Finance (Market data)
    - CNN Fear & Greed Index
    - Alpha Vantage (Optional)
    - Polygon.io (Optional)
    
    **Version:** 2.0 | **Last Updated:** December 2024
    """)

# FOOTER
# -------------------------------------------------------------------
st.divider()
st.caption(
    "Market Risk Dashboard | Not financial advice | Data: FRED, CNN, CBOE, Yahoo Finance, Fed, Treasury"
)