"""Volatility & VRP page."""
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.core.data import get_vrp_analysis_cached, get_vrp_history_cached
from dashboard.ui_helpers import section_header_with_timestamp


def render(components):
    section_header_with_timestamp("📊 Volatility Risk Premium Analysis", datetime.now())

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
                    st.plotly_chart(vvix_chart, width='stretch')

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
