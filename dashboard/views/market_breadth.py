"""Market Breadth page."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.core.helpers import get_breadth_mode, get_mcclellan_scale_factor
from processors.breadth_enhanced import EnhancedBreadthAnalyzer, get_new_highs_lows


def render(components):
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

                    # Get current values for dynamic scaling
                    mc_data = breadth_history['mcclellan']
                    mc_max_val = mc_data.max()
                    mc_min_val = mc_data.min()
                    mc_range = max(abs(mc_max_val), abs(mc_min_val), 50)  # At least show -50 to +50

                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                        x=breadth_history['date'],
                        y=mc_data,
                        name='McClellan',
                        line=dict(color='#4ECDC4', width=3),
                        fill='tozeroy',
                        fillcolor='rgba(78, 205, 196, 0.2)',
                        hovertemplate='McClellan: %{y:.1f}<extra></extra>'
                    ))

                    # Reference lines
                    fig.add_hline(y=0, line_dash="solid", line_color="white", opacity=0.5)
                    fig.add_hline(y=50, line_dash="dash", line_color="#4CAF50", opacity=0.5,
                                 annotation_text="Overbought (+50)", annotation_position="right")
                    fig.add_hline(y=-50, line_dash="dash", line_color="#F44336", opacity=0.5,
                                 annotation_text="Oversold (-50)", annotation_position="right")

                    fig.update_layout(
                        title="McClellan Oscillator (90 Days)",
                        xaxis_title="Date",
                        yaxis_title="Oscillator Value",
                        height=500,
                        hovermode='x unified',
                        yaxis=dict(
                            range=[-mc_range * 1.2, mc_range * 1.2],  # Dynamic range
                            zeroline=True,
                            zerolinecolor='gray',
                        )
                    )

                    st.plotly_chart(fig, width='stretch')

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        current_mc = mc_data.iloc[-1]
                        st.metric("Current Value", f"{current_mc:+.1f}")
                    with col2:
                        st.metric("90-Day High", f"{mc_max_val:+.1f}")
                    with col3:
                        st.metric("90-Day Low", f"{mc_min_val:+.1f}")
                    with col4:
                        # Trend direction
                        mc_5d_ago = mc_data.iloc[-5] if len(mc_data) >= 5 else mc_data.iloc[0]
                        mc_trend = "Improving" if current_mc > mc_5d_ago else "Weakening"
                        st.metric("5-Day Trend", mc_trend)

                    st.info("""
                    **McClellan Oscillator Interpretation:**
                    - **> +50**: Overbought - potential pullback ahead
                    - **0 to +50**: Positive momentum - bulls in control
                    - **-50 to 0**: Negative momentum - bears in control
                    - **< -50**: Oversold - potential bounce ahead
                    - **Note**: Values depend on breadth mode (S&P 100 fast mode vs S&P 500 full mode)
                    """)

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

                    # Add Summation Index Chart
                    if summation.get('history'):
                        st.markdown("##### Summation Index History")
                        sum_history = pd.DataFrame(summation['history'])

                        fig_sum = go.Figure()

                        # Summation line
                        fig_sum.add_trace(go.Scatter(
                            x=sum_history['date'],
                            y=sum_history['summation'],
                            name='Summation Index',
                            line=dict(color='#2196F3', width=2),
                            fill='tozeroy',
                            fillcolor='rgba(33, 150, 243, 0.15)',
                            hovertemplate='Summation: %{y:+,.0f}<extra></extra>'
                        ))

                        # Key levels
                        fig_sum.add_hline(y=0, line_dash="solid", line_color="white", opacity=0.5)
                        fig_sum.add_hline(y=1000, line_dash="dash", line_color="#4CAF50", opacity=0.5,
                                         annotation_text="Strong Bull (+1000)", annotation_position="right")
                        fig_sum.add_hline(y=500, line_dash="dot", line_color="#8BC34A", opacity=0.3)
                        fig_sum.add_hline(y=-500, line_dash="dot", line_color="#FF9800", opacity=0.3)
                        fig_sum.add_hline(y=-1000, line_dash="dash", line_color="#F44336", opacity=0.5,
                                         annotation_text="Strong Bear (-1000)", annotation_position="right")

                        fig_sum.update_layout(
                            title="McClellan Summation Index (90 Days)",
                            xaxis_title="Date",
                            yaxis_title="Summation Index",
                            height=400,
                            hovermode='x unified',
                            yaxis=dict(
                                zeroline=True,
                                zerolinecolor='gray',
                                zerolinewidth=1,
                            )
                        )

                        st.plotly_chart(fig_sum, width='stretch')
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
