"""Credit & Liquidity page."""
import logging
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.core.helpers import format_large_number
from dashboard.ui_helpers import section_header_with_timestamp

logger = logging.getLogger(__name__)


def render(components):
    # Get latest data timestamp
    _snapshot = components["db"].get_latest_snapshot()
    _snapshot_ts = _snapshot.get("date") if _snapshot else None
    if isinstance(_snapshot_ts, str):
        try:
            _snapshot_ts = datetime.fromisoformat(_snapshot_ts.replace('Z', '+00:00'))
        except (ValueError, TypeError):
            _snapshot_ts = None

    section_header_with_timestamp("💳 Credit Spreads & Liquidity", _snapshot_ts)

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
        fed_bs_collector = components.get("fed_bs")
        fed_bs_snapshot = fed_bs_collector.get_full_snapshot() if fed_bs_collector else None
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
                liquidity_analyzer = components.get("liquidity_analyzer")
                if liquidity_analyzer:
                    net_liq_signal = liquidity_analyzer.analyze(
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
        # HYG/LQD CREDIT ETF FLOWS
        # =======================
        st.divider()
        st.subheader("📈 Credit ETF Sentiment (HYG vs LQD)")

        st.markdown("""
        **HYG** (High Yield) vs **LQD** (Investment Grade) relative performance shows credit risk appetite.
        When HYG outperforms → Risk-on. When LQD outperforms → Flight to quality.
        """)

        with st.spinner("Fetching credit ETF data..."):
            try:
                credit_flows = components["yahoo"].get_credit_etf_flows()

                if credit_flows:
                    # Signal banner
                    signal = credit_flows.get('signal', 'UNKNOWN')
                    description = credit_flows.get('description', '')

                    if signal == "RISK_ON":
                        st.success(f"🟢 **{signal}**: {description}")
                    elif signal == "RISK_OFF":
                        st.error(f"🔴 **{signal}**: {description}")
                    else:
                        st.info(f"⚪ **{signal}**: {description}")

                    # Metrics row
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        hyg_price = credit_flows.get('hyg_price')
                        hyg_5d = credit_flows.get('hyg_5d_pct')
                        st.metric(
                            "HYG (High Yield)",
                            f"${hyg_price:.2f}" if hyg_price else "N/A",
                            delta=f"{hyg_5d:+.2f}% (5d)" if hyg_5d else None,
                            help="iShares High Yield Corporate Bond ETF - junk bonds"
                        )

                    with col2:
                        lqd_price = credit_flows.get('lqd_price')
                        lqd_5d = credit_flows.get('lqd_5d_pct')
                        st.metric(
                            "LQD (Inv Grade)",
                            f"${lqd_price:.2f}" if lqd_price else "N/A",
                            delta=f"{lqd_5d:+.2f}% (5d)" if lqd_5d else None,
                            help="iShares Investment Grade Corporate Bond ETF"
                        )

                    with col3:
                        ratio = credit_flows.get('hyg_lqd_ratio')
                        ratio_change = credit_flows.get('ratio_20d_change_pct')
                        st.metric(
                            "HYG/LQD Ratio",
                            f"{ratio:.3f}" if ratio else "N/A",
                            delta=f"{ratio_change:+.2f}% (20d)" if ratio_change else None,
                            help="Higher ratio = more risk appetite"
                        )

                    with col4:
                        rel_5d = credit_flows.get('relative_5d')
                        if rel_5d is not None:
                            if rel_5d > 0:
                                st.metric("5D Relative", f"HYG +{rel_5d:.2f}%", delta="Risk-On")
                            else:
                                st.metric("5D Relative", f"LQD +{abs(rel_5d):.2f}%", delta="Risk-Off", delta_color="inverse")
                        else:
                            st.metric("5D Relative", "N/A")

                    # Performance comparison table
                    with st.expander("📊 Detailed Performance"):
                        # Safely get values with fallback to 0 for None
                        def safe_pct(val):
                            return val if val is not None else 0

                        perf_data = {
                            'Period': ['1 Day', '5 Day', '20 Day'],
                            'HYG': [
                                f"{safe_pct(credit_flows.get('hyg_1d_pct')):+.2f}%",
                                f"{safe_pct(credit_flows.get('hyg_5d_pct')):+.2f}%",
                                f"{safe_pct(credit_flows.get('hyg_20d_pct')):+.2f}%"
                            ],
                            'LQD': [
                                f"{safe_pct(credit_flows.get('lqd_1d_pct')):+.2f}%",
                                f"{safe_pct(credit_flows.get('lqd_5d_pct')):+.2f}%",
                                f"{safe_pct(credit_flows.get('lqd_20d_pct')):+.2f}%"
                            ],
                            'HYG-LQD (Relative)': [
                                f"{safe_pct(credit_flows.get('relative_1d')):+.2f}%",
                                f"{safe_pct(credit_flows.get('relative_5d')):+.2f}%",
                                f"{safe_pct(credit_flows.get('relative_20d')):+.2f}%"
                            ]
                        }
                        st.dataframe(pd.DataFrame(perf_data), hide_index=True, width='stretch')

                        st.caption("""
                        **Interpretation:**
                        - Positive relative = HYG outperforming (risk appetite)
                        - Negative relative = LQD outperforming (flight to quality)
                        - Watch for divergence from equity markets as leading indicator
                        """)
                else:
                    st.warning("Credit ETF data unavailable")

            except Exception as e:
                logger.error(f"Error fetching credit ETF flows: {e}")
                st.warning(f"Could not fetch credit ETF data: {e}")

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
            st.plotly_chart(fig, width='stretch')

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
            st.plotly_chart(fig, width='stretch')

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
