"""Overview page."""
import logging
import os
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from config import cfg
from dashboard.core.helpers import status_from_timestamp
from dashboard.ui_helpers import calculate_composite_risk_score, check_snapshot_freshness, compare_to_historical, data_source_caption, format_percentile_badge, get_all_alerts, get_credit_spread_percentile, get_estimated_indicator, get_fear_greed_percentile, get_tooltip, get_vix_percentile, metric_status_caption, prepare_snapshot_for_export, section_header_with_timestamp
from data_collectors.cboe_collector import CBOECollector
from data_collectors.yahoo_collector import YahooCollector
from utils.data_status import DataStatus, DataStatusTracker

logger = logging.getLogger(__name__)


def render(components):
    snapshot = components["db"].get_latest_snapshot(include_age=True)

    if not snapshot:
        st.warning("⏳ No cached data available. Fetching fresh data...")

        # Try to fetch fresh data on-demand for Streamlit Cloud
        try:
            with st.spinner("Loading market data (this may take a moment on first load)..."):
                from scheduler.daily_update import MarketDataUpdater
                updater = MarketDataUpdater()
                updater.run_full_update()

                # Reload snapshot after update
                snapshot = components["db"].get_latest_snapshot(include_age=True)

                if snapshot:
                    st.success("✅ Data loaded successfully!")
                    st.rerun()
                else:
                    st.error("❌ Could not fetch data. Please check API keys in Settings → Secrets.")
                    st.info("Required: FRED_API_KEY. Get one free at https://fred.stlouisfed.org/docs/api/api_key.html")
                    st.stop()
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            st.info("💡 Make sure FRED_API_KEY is set in Streamlit Cloud Secrets (Settings → Secrets)")
            st.code("""# Add to Streamlit Secrets:
FRED_API_KEY = "your_key_here"
NASDAQ_DATA_LINK_KEY = "your_key_here"  # Optional""")
            st.stop()

    status_tracker = DataStatusTracker()
    snapshot_status, snapshot_age_hours = status_from_timestamp(snapshot.get("date"))
    status_tracker.update("snapshot", snapshot_status, age_hours=snapshot_age_hours or 0.0)

    # Check if FRED data is missing and show warning
    fred_data_missing = (
        snapshot.get("credit_spread_hy") is None or
        snapshot.get("treasury_10y") is None
    )

    if fred_data_missing:
        from utils.secrets_helper import get_secret
        fred_key = get_secret('FRED_API_KEY')

        if not fred_key:
            st.warning("""
            ⚠️ **FRED API Key Not Configured** - Some data using Yahoo Finance fallbacks

            For full functionality, add your FRED API key:
            1. Get a **free** key at https://fred.stlouisfed.org/docs/api/api_key.html
            2. Go to **Settings** → **Secrets** (in Streamlit Cloud)
            3. Add: `FRED_API_KEY = "your_key_here"`
            """)
        else:
            st.info("ℹ️ Some FRED data unavailable - using Yahoo Finance fallbacks where possible")

    def latest_indicator_timestamp(indicator_name: str, days: int = 30):
        df = components["db"].get_indicator_history(indicator_name, days=days)
        if df.empty:
            return None
        return df["date"].iloc[-1]

    credit_ts = latest_indicator_timestamp("credit_spread_hy") or snapshot.get("date")
    credit_status, credit_age = status_from_timestamp(credit_ts)
    treasury_ts = latest_indicator_timestamp("treasury_10y") or snapshot.get("date")
    treasury_status, treasury_age = status_from_timestamp(treasury_ts)

    breadth_latest = components["db"].get_latest_breadth()
    breadth_ts = (breadth_latest or {}).get("date") or snapshot.get("date")
    breadth_status, breadth_age = status_from_timestamp(breadth_ts)
    
    # ========== REGIME SUMMARY BANNER ==========
    # Display header with last update timestamp
    snapshot_date = snapshot.get("date")
    if snapshot_date:
        if isinstance(snapshot_date, str):
            try:
                snapshot_date = datetime.fromisoformat(snapshot_date.replace('Z', '+00:00'))
            except (ValueError, TypeError):
                snapshot_date = None
        section_header_with_timestamp(" Market Regime Summary", snapshot_date)
    else:
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
            credit_regime_label = "🟢 Supportive"
        elif hy_spread < 450:
            credit_regime_label = "🟡 Neutral"
        else:
            credit_regime_label = "🔴 Risk-Off"
        regime_parts.append(f"**Credit:** {credit_regime_label}")
    
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

    # ========== COMPOSITE RISK SCORE & ALERTS ==========
    risk_col, alert_col, status_col = st.columns([1, 1, 1])

    with risk_col:
        # Composite Risk Score
        risk_score = calculate_composite_risk_score(snapshot, vrp_data)
        if risk_score.get('score') is not None:
            st.markdown("#### 📊 Risk Score")
            score = risk_score['score']
            interpretation = risk_score['interpretation']
            color = risk_score['color']

            # Create a visual gauge
            st.markdown(
                f"""
                <div style="text-align: center; padding: 10px;">
                    <div style="font-size: 48px; font-weight: bold; color: {color};">{score:.0f}</div>
                    <div style="font-size: 14px; color: {color}; font-weight: bold;">{interpretation}</div>
                    <div style="font-size: 12px; color: #888; margin-top: 5px;">{risk_score['description']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Expandable details
            with st.expander("Score Components"):
                for comp, val in risk_score['components'].items():
                    weight = risk_score['weights'].get(comp, 0) * 100
                    st.caption(f"• {comp.title()}: {val:.1f} ({weight:.0f}% weight)")

    with alert_col:
        # Active Alerts
        st.markdown("#### ⚠️ Active Alerts")
        alerts = get_all_alerts(snapshot, vrp_data)

        if alerts:
            for alert in alerts[:4]:  # Show max 4 alerts
                severity = alert['severity']
                if severity == 'critical':
                    st.error(alert['message'])
                elif severity == 'warning':
                    st.warning(alert['message'])
                else:
                    st.info(alert['message'])
        else:
            st.success("✅ No alerts - all indicators normal")

    with status_col:
        # Data Freshness Status
        st.markdown("#### 🔄 Data Status")
        freshness = check_snapshot_freshness(snapshot)

        if freshness['status'] == 'ok':
            st.success(f"✅ {freshness['message']}")
        elif freshness['status'] == 'stale':
            st.warning(f"⚠️ {freshness['message']}")
        else:
            st.error(f"❌ {freshness['message']}")

        # Export button
        st.markdown("---")
        export_df = prepare_snapshot_for_export(snapshot)
        if not export_df.empty:
            csv_data = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Export to CSV",
                data=csv_data,
                file_name=f"market_data_{snapshot.get('date', 'latest')}.csv",
                mime="text/csv",
                width='stretch'
            )

    st.divider()

    # ========== HISTORICAL COMPARISON (Collapsible) ==========
    with st.expander("📈 Historical Comparison", expanded=False):
        comparison = compare_to_historical(snapshot)

        if comparison.get('closest_match'):
            closest = comparison['closest_match']
            st.info(f"**Most similar to:** {closest['name']} - {closest['description']}")

        # Show comparison table
        comp_data = []
        for period in comparison.get('comparisons', [])[:4]:
            row = {'Period': period['name']}

            if 'vix' in period['differences']:
                diff = period['differences']['vix']
                row['VIX'] = f"{diff['current']:.1f} vs {diff['historical']:.1f} ({diff['pct']:+.0f}%)"

            if 'fear_greed' in period['differences']:
                diff = period['differences']['fear_greed']
                row['F&G'] = f"{diff['current']:.0f} vs {diff['historical']:.0f} ({diff['diff']:+.0f})"

            if 'hy_spread' in period['differences']:
                diff = period['differences']['hy_spread']
                row['HY Spread'] = f"{diff['current']:.0f} vs {diff['historical']:.0f} ({diff['pct']:+.0f}%)"

            comp_data.append(row)

        if comp_data:
            st.table(pd.DataFrame(comp_data))

    st.subheader("Key Market Indicators")

    # Fetch live Yahoo data for fallbacks when FRED is unavailable
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_live_yahoo_fallbacks():
        """Fetch Treasury and HY spread from Yahoo as fallback"""
        try:
            yahoo = YahooCollector()
            return {
                'treasury_10y': yahoo.get_treasury_10y(),
                'hy_spread_proxy': yahoo.get_hy_spread_proxy(),
            }
        except Exception as e:
            logger.warning(f"Yahoo fallback fetch failed: {e}")
            return {}

    # Get fallback data if needed
    yahoo_fallbacks = {}
    if snapshot.get("credit_spread_hy") is None or snapshot.get("treasury_10y") is None:
        yahoo_fallbacks = get_live_yahoo_fallbacks()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        val = snapshot.get("credit_spread_hy")
        source_text = "FRED (BAMLH0A0HYM2)"
        is_fallback = False

        # Use Yahoo fallback if FRED data unavailable
        if val is None and yahoo_fallbacks.get('hy_spread_proxy'):
            val = yahoo_fallbacks['hy_spread_proxy']
            source_text = "Yahoo (HYG proxy)"
            is_fallback = True

        st.metric(
            "HY Spread",
            f"{val:.2f}%" if val is not None else "N/A",
            help=get_tooltip('credit_spread_hy')
        )

        if is_fallback:
            status_tracker.update("credit_spread_hy", DataStatus.ESTIMATED, age_hours=0.0)
            data_source_caption(col1, source_text, "estimated")
            st.caption("📊 Yahoo proxy (FRED unavailable)")
        else:
            status_tracker.update("credit_spread_hy", credit_status, age_hours=credit_age or 0.0)
            data_source_caption(col1, source_text, "daily")

        metric_status_caption(col1, status_tracker.get_source("credit_spread_hy"))

        if val is not None:
            credit_pct = get_credit_spread_percentile(val * 100)  # Convert to bps
            if credit_pct and credit_pct.get('context'):
                st.caption(credit_pct['context'])

    with col2:
        val = snapshot.get("treasury_10y")
        source_text = "FRED (DGS10)"
        is_fallback = False

        # Use Yahoo fallback if FRED data unavailable
        if val is None and yahoo_fallbacks.get('treasury_10y'):
            val = yahoo_fallbacks['treasury_10y']
            source_text = "Yahoo (^TNX)"
            is_fallback = True

        st.metric(
            "10Y Treasury",
            f"{val:.2f}%" if val is not None else "N/A",
            help="Benchmark risk-free rate. Rising = tighter financial conditions."
        )

        if is_fallback:
            status_tracker.update("treasury_10y", DataStatus.ESTIMATED, age_hours=0.0)
            data_source_caption(col2, source_text, "live")
            st.caption("📊 Yahoo live (FRED unavailable)")
        else:
            status_tracker.update("treasury_10y", treasury_status, age_hours=treasury_age or 0.0)
            data_source_caption(col2, source_text, "daily")

        metric_status_caption(col2, status_tracker.get_source("treasury_10y"))

    with col3:
        val = snapshot.get("fear_greed_score")
        st.metric(
            "Fear & Greed",
            f"{val:.0f}" if val is not None else "N/A",
            help=get_tooltip('fear_greed')
        )
        status_tracker.update("fear_greed_score", snapshot_status, age_hours=snapshot_age_hours or 0.0)
        data_source_caption(col3, "CNN Fear & Greed Index", "daily (approx)")
        metric_status_caption(col3, status_tracker.get_source("fear_greed_score"))
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
        status_tracker.update("left_signal", credit_status, age_hours=credit_age or 0.0)
        data_source_caption(col4, "Derived (FRED HYG OAS)", "daily")
        metric_status_caption(col4, status_tracker.get_source("left_signal"))

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
        if fresh_cboe.get("vix_spot") is not None:
            vix_status, vix_age = status_from_timestamp(fresh_cboe.get("timestamp"))
            status_tracker.update("vix_spot", vix_status, age_hours=vix_age or 0.0)
        elif vix is not None:
            status_tracker.update("vix_spot", snapshot_status, age_hours=snapshot_age_hours or 0.0)
        else:
            status_tracker.update("vix_spot", DataStatus.UNAVAILABLE, age_hours=snapshot_age_hours or 0.0)
        data_source_caption(col1, "Yahoo Finance (^VIX)", "delayed")
        metric_status_caption(col1, status_tracker.get_source("vix_spot"))
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
            contango_status = DataStatus.ESTIMATED if is_estimated else DataStatus.OK
            contango_status, contango_age = (
                (DataStatus.ESTIMATED, 0.0) if is_estimated else status_from_timestamp(fresh_cboe.get("timestamp"))
            )
            status_tracker.update("vix_contango", contango_status, age_hours=contango_age or 0.0)
            data_source_caption(col2, "Derived (VIX/VIX3M indices)", "delayed")
            metric_status_caption(col2, status_tracker.get_source("vix_contango"))
            if contango > 0:
                st.caption("📈 Contango (bullish)")
            else:
                st.caption("📉 Backwardation (fear)")
        else:
            contango = snapshot.get("vix_contango")
            if contango is not None:
                st.metric("VIX Contango", f"{contango:+.2f}%", help=get_tooltip('vix_contango'))
                status_tracker.update("vix_contango", snapshot_status, age_hours=snapshot_age_hours or 0.0)
                data_source_caption(col2, "Local DB snapshot", "varies")
                metric_status_caption(col2, status_tracker.get_source("vix_contango"))
                st.caption("📦 Cached data")
            else:
                st.metric("VIX Contango", "N/A")
                status_tracker.update("vix_contango", DataStatus.UNAVAILABLE, age_hours=snapshot_age_hours or 0.0)
                data_source_caption(col2, "Derived (VIX/VIX3M indices)", "delayed")
                metric_status_caption(col2, status_tracker.get_source("vix_contango"))

    with col3:
        # Get put/call ratios - SPY P/C is primary (free data), CBOE PCCE requires manual input
        pc_ratios = fresh_cboe.get("put_call_ratios", {})
        spy_pc = pc_ratios.get("spy_pc")  # SPY-specific P/C (primary)
        cboe_pcce = pc_ratios.get("cboe_equity_pc")  # Official CBOE Equity P/C (rare)
        equity_pc = pc_ratios.get("equity_pc")  # Best available
        pc_source = pc_ratios.get("source", "")

        # Thresholds from config (SPY typically runs higher than CBOE PCCE)
        pc_bearish = cfg.get('options.equity_put_call.bearish_threshold', 1.0)
        pc_bullish = cfg.get('options.equity_put_call.bullish_threshold', 0.7)

        # Check for manual PCCE override first
        load_dotenv()
        manual_pcce = os.getenv('MANUAL_PCCE', '0.0')
        manual_pcce_date = os.getenv('MANUAL_PCCE_DATE', '')
        try:
            manual_pcce_value = float(manual_pcce) if manual_pcce else 0.0
        except (ValueError, TypeError) as e:
            logger.debug(f"Invalid manual PCCE value '{manual_pcce}': {e}")
            manual_pcce_value = 0.0

        # Priority: Manual PCCE > CBOE PCCE > SPY P/C > VIX Proxy > Cached
        if manual_pcce_value > 0:
            # User has entered manual CBOE PCCE
            st.metric("CBOE Equity P/C", f"{manual_pcce_value:.3f}",
                      help="Manual PCCE from trading platform (all equity options)")
            status_tracker.update("cboe_equity_pc", DataStatus.OK, age_hours=0.0)
            data_source_caption(col3, "Manual input", "user-provided")
            st.caption(f"✏️ Manual ({manual_pcce_date or 'Today'})")

            if manual_pcce_value > pc_bearish:
                st.caption("🔴 Bearish sentiment")
            elif manual_pcce_value < pc_bullish:
                st.caption("🟢 Bullish sentiment")
            else:
                st.caption("🟡 Neutral range")

        elif cboe_pcce is not None:
            # Rare: Official CBOE PCCE available
            st.metric("CBOE Equity P/C", f"{cboe_pcce:.3f}",
                      help="Official CBOE ratio for ALL equity options")
            pc_status, pc_age = status_from_timestamp(fresh_cboe.get("timestamp"))
            status_tracker.update("cboe_equity_pc", pc_status, age_hours=pc_age or 0.0)
            data_source_caption(col3, "CBOE (scraped)", "delayed")
            st.caption("📊 Official CBOE PCCE")

            if cboe_pcce > pc_bearish:
                st.caption("🔴 Bearish sentiment")
            elif cboe_pcce < pc_bullish:
                st.caption("🟢 Bullish sentiment")
            else:
                st.caption("🟡 Neutral range")

        elif spy_pc is not None:
            # Primary free data: SPY Put/Call (clearly labeled as SPY, not CBOE)
            st.metric("SPY Put/Call", f"{spy_pc:.3f}",
                      help="SPY options open interest ratio - institutional hedging gauge")
            pc_status, pc_age = status_from_timestamp(fresh_cboe.get("timestamp"))
            status_tracker.update("spy_put_call", pc_status, age_hours=pc_age or 0.0)
            data_source_caption(col3, "Yahoo Finance (SPY options)", "delayed")
            st.caption("📈 SPY Options OI")

            # SPY thresholds (typically runs higher than CBOE PCCE)
            if spy_pc > 1.2:
                st.caption("🔴 Heavy hedging")
            elif spy_pc < 0.8:
                st.caption("🟢 Bullish bias")
            else:
                st.caption("🟡 Normal range")

        elif equity_pc is not None and pc_source == "VIX_PROXY":
            # VIX proxy estimation (least reliable)
            st.metric("Est. Equity P/C", f"{equity_pc:.3f}",
                      help="Estimated from VIX term structure (proxy)")
            status_tracker.update("equity_put_call", DataStatus.ESTIMATED, age_hours=0.0)
            data_source_caption(col3, "VIX/VIX3M estimate", "delayed")
            st.caption("📉 VIX proxy estimate")

            if equity_pc > pc_bearish:
                st.caption("🔴 Bearish")
            elif equity_pc < pc_bullish:
                st.caption("🟢 Bullish")
            else:
                st.caption("🟡 Neutral")
        else:
            # Fallback to cached data
            pc = snapshot.get("put_call_ratio") or snapshot.get("spy_put_call")
            if pc is not None:
                st.metric("Put/Call", f"{pc:.3f}")
                status_tracker.update("equity_put_call", snapshot_status, age_hours=snapshot_age_hours or 0.0)
                data_source_caption(col3, "Local DB snapshot", "varies")
                st.caption("📦 Cached data")
            else:
                st.metric("Put/Call", "N/A")
                status_tracker.update("equity_put_call", DataStatus.UNAVAILABLE, age_hours=snapshot_age_hours or 0.0)
                data_source_caption(col3, "Unknown source", "unknown")
                st.caption("Data unavailable")

    with col4:
        breadth = snapshot.get("market_breadth")
        if breadth is not None:
            pct = breadth * 100 if breadth <= 1 else breadth
            st.metric("Market Breadth", f"{pct:.1f}%")
            status_tracker.update("market_breadth", breadth_status, age_hours=breadth_age or 0.0)
            data_source_caption(col4, "Yahoo Finance (S&P 500 components)", "delayed")
            metric_status_caption(col4, status_tracker.get_source("market_breadth"))
            if pct > 60:
                st.caption("Strong participation")
            elif pct < 40:
                st.caption("Weak participation")
        else:
            st.metric("Market Breadth", "N/A")
            status_tracker.update("market_breadth", DataStatus.UNAVAILABLE, age_hours=snapshot_age_hours or 0.0)
            data_source_caption(col4, "Yahoo Finance (S&P 500 components)", "delayed")
            metric_status_caption(col4, status_tracker.get_source("market_breadth"))

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
            status_tracker.update("vix9d", snapshot_status, age_hours=snapshot_age_hours or 0.0)
            data_source_caption(col1, "Yahoo Finance (^VIX9D)", "delayed")
            metric_status_caption(col1, status_tracker.get_source("vix9d"))
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
            status_tracker.update("vix9d", DataStatus.UNAVAILABLE, age_hours=snapshot_age_hours or 0.0)
            data_source_caption(col1, "Yahoo Finance (^VIX9D)", "delayed")
            metric_status_caption(col1, status_tracker.get_source("vix9d"))

    with col2:
        vvix = fresh_cboe.get("vvix") if fresh_cboe else None
        # VVIX thresholds from config
        vvix_strong_buy = cfg.get('volatility.vvix.strong_buy_threshold', 120)
        vvix_buy_alert = cfg.get('volatility.vvix.buy_alert_threshold', 110)
        vvix_normal_min = cfg.get('volatility.vvix.normal_min', 80)

        if vvix is not None:
            if vvix >= vvix_strong_buy:
                st.metric("🎯 VVIX", f"{vvix:.1f}", delta="BUY", help=get_tooltip('vvix'))
                vvix_status, vvix_age = status_from_timestamp(fresh_cboe.get("timestamp"))
                status_tracker.update("vvix", vvix_status, age_hours=vvix_age or 0.0)
                data_source_caption(col2, "Yahoo Finance (^VVIX)", "delayed")
                metric_status_caption(col2, status_tracker.get_source("vvix"))
                st.caption("🟢 Historic buy zone")
            elif vvix >= vvix_buy_alert:
                st.metric("VVIX", f"{vvix:.1f}", help=get_tooltip('vvix'))
                vvix_status, vvix_age = status_from_timestamp(fresh_cboe.get("timestamp"))
                status_tracker.update("vvix", vvix_status, age_hours=vvix_age or 0.0)
                data_source_caption(col2, "Yahoo Finance (^VVIX)", "delayed")
                metric_status_caption(col2, status_tracker.get_source("vvix"))
                st.caption("🟡 Elevated")
            elif vvix >= vvix_normal_min:
                st.metric("VVIX", f"{vvix:.1f}", help=get_tooltip('vvix'))
                vvix_status, vvix_age = status_from_timestamp(fresh_cboe.get("timestamp"))
                status_tracker.update("vvix", vvix_status, age_hours=vvix_age or 0.0)
                data_source_caption(col2, "Yahoo Finance (^VVIX)", "delayed")
                metric_status_caption(col2, status_tracker.get_source("vvix"))
                st.caption("Normal")
            else:
                st.metric("VVIX", f"{vvix:.1f}", help=get_tooltip('vvix'))
                vvix_status, vvix_age = status_from_timestamp(fresh_cboe.get("timestamp"))
                status_tracker.update("vvix", vvix_status, age_hours=vvix_age or 0.0)
                data_source_caption(col2, "Yahoo Finance (^VVIX)", "delayed")
                metric_status_caption(col2, status_tracker.get_source("vvix"))
                st.caption("🟠 Complacent")
        else:
            st.metric("VVIX", "N/A")
            status_tracker.update("vvix", DataStatus.UNAVAILABLE, age_hours=snapshot_age_hours or 0.0)
            data_source_caption(col2, "Yahoo Finance (^VVIX)", "delayed")
            metric_status_caption(col2, status_tracker.get_source("vvix"))

    with col3:
        skew = snapshot.get("skew")
        if skew is not None:
            st.metric("SKEW", f"{skew:.0f}", help=get_tooltip('skew'))
            status_tracker.update("skew", snapshot_status, age_hours=snapshot_age_hours or 0.0)
            data_source_caption(col3, "Yahoo Finance (^SKEW)", "delayed")
            metric_status_caption(col3, status_tracker.get_source("skew"))
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
            status_tracker.update("skew", DataStatus.UNAVAILABLE, age_hours=snapshot_age_hours or 0.0)
            data_source_caption(col3, "Yahoo Finance (^SKEW)", "delayed")
            metric_status_caption(col3, status_tracker.get_source("skew"))

    with col4:
        vrp = snapshot.get("vrp")
        if vrp is not None:
            st.metric("VRP", f"{vrp:+.1f}", help=get_tooltip('vrp'))
            status_tracker.update("vrp", snapshot_status, age_hours=snapshot_age_hours or 0.0)
            data_source_caption(col4, "Derived (VIX & SPY realized)", "varies")
            metric_status_caption(col4, status_tracker.get_source("vrp"))
            if vrp > 5:
                st.caption("🟢 Options expensive")
            elif vrp > 0:
                st.caption("Positive premium")
            else:
                st.caption("🔴 Negative (risk)")
        else:
            st.metric("VRP", "N/A")
            status_tracker.update("vrp", DataStatus.UNAVAILABLE, age_hours=snapshot_age_hours or 0.0)
            data_source_caption(col4, "Derived (VIX & SPY realized)", "varies")
            metric_status_caption(col4, status_tracker.get_source("vrp"))

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
            if is_vix3m_est:
                slope_status, slope_age = DataStatus.ESTIMATED, 0.0
            else:
                slope_status, slope_age = status_from_timestamp(fresh_cboe.get("timestamp"))
            status_tracker.update("term_slope", slope_status, age_hours=slope_age or 0.0)
            data_source_caption(col5, "Derived (VIX vs VIX3M)", "varies")
            metric_status_caption(col5, status_tracker.get_source("term_slope"))
            contango_steep = cfg.get('volatility.vix_term.contango_steep_threshold', 0.05)
            if slope_per_day > contango_steep:
                st.caption("🟢 Steep contango")
            elif slope_per_day > 0:
                st.caption("Normal")
            else:
                st.caption("🔴 Inverted")
        else:
            st.metric("Term Slope", "N/A")
            status_tracker.update("term_slope", DataStatus.UNAVAILABLE, age_hours=snapshot_age_hours or 0.0)
            data_source_caption(col5, "Derived (VIX vs VIX3M)", "varies")
            metric_status_caption(col5, status_tracker.get_source("term_slope"))

    # SPY Put/Call - Institutional Hedging Gauge (separate from CBOE PCCE)
    pc_ratios = fresh_cboe.get("put_call_ratios", {})
    spy_pc_vol = pc_ratios.get("spy_pc")  # Volume-based (CBOE-comparable)
    spy_pc_oi = pc_ratios.get("spy_pc_oi")  # OI-based (positioning)
    spy_put_vol = pc_ratios.get("spy_put_volume")
    spy_call_vol = pc_ratios.get("spy_call_volume")
    spy_put_oi = pc_ratios.get("spy_put_oi")
    spy_call_oi = pc_ratios.get("spy_call_oi")

    if spy_pc_vol is not None or spy_pc_oi is not None:
        st.subheader("SPY Options Flow")
        st.caption("*SPY options data - best free proxy for market sentiment (not identical to CBOE PCCE)*")

        # Show both volume and OI metrics
        spy_cols = st.columns(4)

        with spy_cols[0]:
            if spy_pc_vol is not None:
                st.metric(
                    "P/C (Volume)",
                    f"{spy_pc_vol:.3f}",
                    help="Daily volume ratio - comparable to CBOE methodology"
                )
                # Volume-based interpretation
                if spy_pc_vol > 1.2:
                    st.caption("🔴 Heavy put buying")
                elif spy_pc_vol > 0.9:
                    st.caption("🟡 Elevated puts")
                elif spy_pc_vol > 0.6:
                    st.caption("Normal")
                else:
                    st.caption("🟢 Call-heavy day")
            else:
                st.metric("P/C (Volume)", "N/A")

        with spy_cols[1]:
            if spy_pc_oi is not None:
                st.metric(
                    "P/C (OI)",
                    f"{spy_pc_oi:.3f}",
                    help="Open interest ratio - positioning/inventory view"
                )
                # OI-based interpretation (typically higher than volume)
                if spy_pc_oi > 1.5:
                    st.caption("🔴 Heavy hedging")
                elif spy_pc_oi > 1.0:
                    st.caption("🟡 Put-heavy")
                elif spy_pc_oi > 0.7:
                    st.caption("Normal")
                else:
                    st.caption("🟢 Bullish positioning")
            else:
                st.metric("P/C (OI)", "N/A")

        with spy_cols[2]:
            if spy_put_vol and spy_call_vol:
                st.metric("Put Vol", f"{spy_put_vol:,}")
                st.metric("Call Vol", f"{spy_call_vol:,}")
            elif spy_put_oi:
                st.metric("Put OI", f"{spy_put_oi:,}")

        with spy_cols[3]:
            if spy_put_vol and spy_call_vol:
                st.metric("Put OI", f"{spy_put_oi:,}" if spy_put_oi else "N/A")
                st.metric("Call OI", f"{spy_call_oi:,}" if spy_call_oi else "N/A")
            elif spy_call_oi:
                st.metric("Call OI", f"{spy_call_oi:,}")

        spy_status, spy_age = status_from_timestamp(fresh_cboe.get("timestamp"))
        status_tracker.update("spy_put_call", spy_status, age_hours=spy_age or 0.0)
        data_source_caption(st, "Yahoo Finance (SPY front-month options)", "delayed")

    st.divider()
    signal = snapshot.get("left_signal")
    if signal:
        st.subheader(f"Current Signal: {signal}")
