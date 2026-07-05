"""Sidebar: navigation, market status, data freshness, health, and PDF export."""
import logging
from datetime import datetime

import streamlit as st

from dashboard.core.data import get_vrp_history_cached
from dashboard.core.helpers import get_mcclellan_scale_factor
from dashboard.pdf_chart_builder import create_adline_chart, create_credit_spreads_chart, create_liquidity_chart, create_mcclellan_chart, create_treasury_stress_chart, create_vix_term_structure_chart, create_vrp_chart
from dashboard.pdf_generator_v2 import PDFReportGenerator
from data_collectors.market_status_collector import MarketStatusCollector
from database.health_check import HealthStatus

logger = logging.getLogger(__name__)


def render_sidebar(components):
    """Render the sidebar and return the selected page name."""
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
                "Institutional Flow",      # Phase 3 - Dark Pool, Insider, Auctions
                "Economic Calendar",       # Phase 3 - Events & Countdown
                "Fed Watch",               # Phase 4 - Rate probabilities
                "Cross-Asset",             # Phase 4 - Correlations & Regime
                "Options Flow",            # Phase 4 - Unusual activity scanner
                "Settings",
                "System Health",           # Health check and diagnostics
            ],
        )

        st.divider()

        # --- MARKET STATUS INDICATOR (Phase 4) ---
        try:
            market_status_collector = MarketStatusCollector()
            market_status = market_status_collector.get_market_status()

            status_text = market_status.get('status', 'UNKNOWN')
            status_color = market_status.get('status_color', '#9E9E9E')
            status_emoji = market_status.get('emoji', '')
            current_time = market_status.get('current_time_et', '')

            st.markdown(
                f"<div style='padding: 8px; background-color: {status_color}20; border-left: 4px solid {status_color}; border-radius: 4px; margin-bottom: 10px;'>"
                f"<strong>{status_emoji} {status_text}</strong><br/>"
                f"<small style='color: #888;'>{current_time}</small>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Show time until next session
            if status_text == 'OPEN':
                time_left = market_status.get('time_until_close', '')
                st.caption(f"Closes in: {time_left}")
            elif status_text == 'PRE-MARKET':
                time_left = market_status.get('time_until_open', '')
                st.caption(f"Market opens in: {time_left}")
            elif status_text in ['CLOSED', 'AFTER-HOURS']:
                time_left = market_status.get('time_until_open') or market_status.get('time_until_end', '')
                if time_left:
                    st.caption(f"Next session: {time_left}")
        except Exception as e:
            logger.debug(f"Market status error: {e}")

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

        st.divider()
        st.caption(
            "Data notice: This product uses the FRED® API but is not endorsed or certified by "
            "the Federal Reserve Bank of St. Louis. Market data sources may be delayed."
        )
        st.caption("Created by Tristan Alejandro / Not financial advice.")
        st.caption("Source code: [github.com/sykurtyppi/market-dashboard](https://github.com/sykurtyppi/market-dashboard)")
    return page
