"""Fed Watch page."""
import logging

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_collectors.fed_watch_collector import FedWatchCollector

logger = logging.getLogger(__name__)


def render(components):
    st.markdown("<h1 class='main-header'>Fed Watch - Rate Probabilities</h1>", unsafe_allow_html=True)

    st.markdown("""
    **Institutional-grade** market-implied probabilities for Fed rate decisions using CME FedWatch methodology.
    Data sourced from 30-Day Fed Funds Futures (ZQ contracts) and FRED API.
    """)

    @st.cache_resource(ttl=60)
    def get_fed_watch_collector():
        return FedWatchCollector()

    fed_watch = get_fed_watch_collector()

    try:
        # Get Fed Watch summary
        summary = fed_watch.get_fed_watch_summary()

        if summary.get('status') == 'unavailable':
            st.warning("Fed Watch data temporarily unavailable")
        elif summary.get('status') == 'error':
            st.error(f"Error: {summary.get('error', 'Unknown error')}")
        else:
            # --- DATA SOURCE INDICATOR ---
            source = summary.get('rate_source', 'unknown')
            data_source = summary.get('data_source', 'unknown')
            source_color = '#4CAF50' if source == 'FRED' else '#FF9800'

            col_src1, col_src2, col_src3 = st.columns([2, 2, 2])
            with col_src1:
                st.markdown(
                    f"<div style='padding:8px; background-color:{source_color}20; border-radius:5px; border-left:4px solid {source_color};'>"
                    f"<small>Rate Source: <strong>{source}</strong></small>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            with col_src2:
                st.markdown(
                    f"<div style='padding:8px; background-color:#2196F320; border-radius:5px; border-left:4px solid #2196F3;'>"
                    f"<small>Probability Source: <strong>Fed Funds Futures</strong></small>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            with col_src3:
                as_of = summary.get('rate_as_of', 'N/A')
                st.markdown(
                    f"<div style='padding:8px; background-color:#9E9E9E20; border-radius:5px; border-left:4px solid #9E9E9E;'>"
                    f"<small>Data as of: <strong>{as_of}</strong></small>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            st.divider()

            # --- CURRENT RATE & NEXT MEETING ---
            st.subheader("Current Federal Funds Rate")

            next_meeting = summary.get('next_meeting', {})
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.markdown(
                    f"<div style='text-align:center; padding:15px; background-color:#1e1e1e; border-radius:10px; border:1px solid #333;'>"
                    f"<small style='color:#888;'>Target Range</small><br/>"
                    f"<strong style='font-size:1.4em; color:#2196F3;'>{summary.get('current_rate', 'N/A')}</strong>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            with col2:
                effr = summary.get('effr', 'N/A')
                effr_str = f"{effr:.2f}%" if isinstance(effr, (int, float)) else effr
                st.markdown(
                    f"<div style='text-align:center; padding:15px; background-color:#1e1e1e; border-radius:10px; border:1px solid #333;'>"
                    f"<small style='color:#888;'>Effective Rate (EFFR)</small><br/>"
                    f"<strong style='font-size:1.4em; color:#FF9800;'>{effr_str}</strong>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            with col3:
                st.markdown(
                    f"<div style='text-align:center; padding:15px; background-color:#1e1e1e; border-radius:10px; border:1px solid #333;'>"
                    f"<small style='color:#888;'>Next FOMC Meeting</small><br/>"
                    f"<strong style='font-size:1.4em; color:#FFF;'>{next_meeting.get('date_str', 'N/A')}</strong>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            with col4:
                days = next_meeting.get('days_until', 'N/A')
                st.markdown(
                    f"<div style='text-align:center; padding:15px; background-color:#1e1e1e; border-radius:10px; border:1px solid #333;'>"
                    f"<small style='color:#888;'>Days Until</small><br/>"
                    f"<strong style='font-size:1.4em; color:#FFF;'>{days}</strong>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            with col5:
                has_sep = next_meeting.get('has_sep', False)
                sep_text = "Yes (Dot Plot)" if has_sep else "No"
                sep_color = "#4CAF50" if has_sep else "#9E9E9E"
                st.markdown(
                    f"<div style='text-align:center; padding:15px; background-color:#1e1e1e; border-radius:10px; border:1px solid #333;'>"
                    f"<small style='color:#888;'>SEP Release</small><br/>"
                    f"<strong style='font-size:1.4em; color:{sep_color};'>{sep_text}</strong>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            st.divider()

            # --- MARKET BIAS INDICATOR ---
            bias = summary.get('market_bias', 'Neutral')
            bias_color = summary.get('bias_color', '#9E9E9E')
            implied_rate = summary.get('implied_rate')
            implied_change = summary.get('implied_change_bps', 0)

            col_bias1, col_bias2 = st.columns([1, 2])

            with col_bias1:
                st.markdown(
                    f"<div style='text-align:center; padding:25px; background-color:{bias_color}30; border-radius:15px; border:3px solid {bias_color};'>"
                    f"<small style='color:#888;'>MARKET EXPECTATION</small><br/>"
                    f"<strong style='font-size:2em; color:{bias_color};'>{bias}</strong><br/>"
                    f"<small style='color:#888;'>Implied Change: {implied_change:+.1f} bps</small>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            with col_bias2:
                # Create probability gauge chart
                probs = summary.get('probabilities', {})
                cut_prob = summary.get('cut_probability', 0)
                hold_prob = summary.get('hold_probability', 0)
                hike_prob = summary.get('hike_probability', 0)

                fig_gauge = go.Figure()

                # Stacked horizontal bar showing cut/hold/hike distribution
                fig_gauge.add_trace(go.Bar(
                    y=['Probability'],
                    x=[cut_prob],
                    name=f'Cut ({cut_prob:.1f}%)',
                    orientation='h',
                    marker_color='#4CAF50',
                    text=f'{cut_prob:.0f}%' if cut_prob > 5 else '',
                    textposition='inside',
                ))
                fig_gauge.add_trace(go.Bar(
                    y=['Probability'],
                    x=[hold_prob],
                    name=f'Hold ({hold_prob:.1f}%)',
                    orientation='h',
                    marker_color='#9E9E9E',
                    text=f'{hold_prob:.0f}%' if hold_prob > 5 else '',
                    textposition='inside',
                ))
                fig_gauge.add_trace(go.Bar(
                    y=['Probability'],
                    x=[hike_prob],
                    name=f'Hike ({hike_prob:.1f}%)',
                    orientation='h',
                    marker_color='#F44336',
                    text=f'{hike_prob:.0f}%' if hike_prob > 5 else '',
                    textposition='inside',
                ))

                fig_gauge.update_layout(
                    barmode='stack',
                    height=120,
                    margin=dict(l=0, r=0, t=30, b=0),
                    template='plotly_dark',
                    showlegend=True,
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                    xaxis=dict(range=[0, 100], showticklabels=True, title='Probability (%)'),
                    yaxis=dict(showticklabels=False),
                )

                st.plotly_chart(fig_gauge, width='stretch')

            st.divider()

            # --- RATE PROBABILITIES DETAIL ---
            st.subheader("Rate Decision Probabilities")

            col1, col2, col3, col4, col5 = st.columns(5)
            prob_cols = [col1, col2, col3, col4, col5]

            # Ensure consistent ordering
            prob_order = ['Cut 50bp', 'Cut 25bp', 'No Change', 'Hike 25bp', 'Hike 50bp']

            for i, decision in enumerate(prob_order):
                prob = probs.get(decision, 0)
                with prob_cols[i]:
                    # Color based on decision type
                    if 'Cut' in decision:
                        color = '#4CAF50'
                        icon = '📉'
                    elif 'Hike' in decision:
                        color = '#F44336'
                        icon = '📈'
                    else:
                        color = '#9E9E9E'
                        icon = '➡️'

                    # Highlight the most likely outcome
                    is_most_likely = decision == summary.get('most_likely', '')
                    border_width = '3px' if is_most_likely else '1px'
                    glow = f'box-shadow: 0 0 15px {color}60;' if is_most_likely else ''

                    st.markdown(
                        f"<div style='text-align:center; padding:15px; background-color:{color}15; border-radius:10px; border:{border_width} solid {color}; {glow}'>"
                        f"<span style='font-size:1.2em;'>{icon}</span><br/>"
                        f"<small style='color:#888;'>{decision}</small><br/>"
                        f"<strong style='font-size:1.8em; color:{color};'>{prob:.1f}%</strong>"
                        f"{'<br/><small style=\"color:#4CAF50;\">★ Most Likely</small>' if is_most_likely else ''}"
                        f"</div>",
                        unsafe_allow_html=True
                    )

            st.divider()

            # --- IMPLIED RATE DATA ---
            if implied_rate:
                col_impl1, col_impl2, col_impl3 = st.columns(3)
                with col_impl1:
                    st.metric(
                        "Implied Rate (Futures)",
                        f"{implied_rate:.3f}%",
                        delta=f"{implied_change:+.1f} bps" if implied_change else None,
                        delta_color="inverse"
                    )
                with col_impl2:
                    current_mid = summary.get('current_rate_mid', 0)
                    st.metric(
                        "Current Target Midpoint",
                        f"{current_mid:.3f}%"
                    )
                with col_impl3:
                    terminal = summary.get('terminal_rate', 0)
                    total_change = summary.get('total_change_bps', 0)
                    exp_cuts = summary.get('expected_cuts', 0)
                    exp_hikes = summary.get('expected_hikes', 0)
                    if exp_cuts > 0:
                        outlook = f"{exp_cuts} cuts expected"
                    elif exp_hikes > 0:
                        outlook = f"{exp_hikes} hikes expected"
                    else:
                        outlook = "No change expected"
                    st.metric(
                        "Terminal Rate (8 meetings)",
                        f"{terminal:.3f}%",
                        delta=outlook
                    )

                st.divider()

            # --- RATE PATH EXPECTATIONS ---
            st.subheader("Expected Rate Path")

            rate_path = summary.get('rate_path', [])
            if rate_path:
                # Enhanced table with more data
                path_data = []
                for p in rate_path:
                    change_bps = p.get('change_bps', 0)
                    if change_bps < 0:
                        change_str = f"{change_bps:+d} bps"
                        change_color = "🟢"
                    elif change_bps > 0:
                        change_str = f"{change_bps:+d} bps"
                        change_color = "🔴"
                    else:
                        change_str = "No change"
                        change_color = "⚪"

                    sep_marker = "📊" if p.get('has_sep', False) else ""

                    implied = p.get('implied_rate')
                    implied_str = f"{implied:.3f}%" if implied else "N/A"

                    path_data.append({
                        'Meeting': f"{p.get('meeting', '')} {sep_marker}",
                        'Days': p.get('days_until', 0),
                        'Implied Rate': implied_str,
                        'Expected Rate': f"{p.get('expected_rate', 0):.3f}%",
                        'Δ from Current': f"{change_color} {change_str}",
                    })

                path_df = pd.DataFrame(path_data)
                st.dataframe(path_df, width='stretch', hide_index=True)

                st.caption("📊 = Meeting includes Summary of Economic Projections (Dot Plot)")

                # Create enhanced rate path chart
                fig = go.Figure()

                dates = [p.get('meeting', '') for p in rate_path]
                rates = [p.get('expected_rate', 0) for p in rate_path]
                implied_rates = [p.get('implied_rate') for p in rate_path]
                has_seps = [p.get('has_sep', False) for p in rate_path]

                # Implied rate line
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=[r if r else None for r in implied_rates],
                    mode='lines',
                    name='Implied Rate (Futures)',
                    line=dict(color='#FF9800', width=2, dash='dot'),
                ))

                # Expected rate line with markers
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=rates,
                    mode='lines+markers',
                    name='Expected Rate',
                    line=dict(color='#2196F3', width=3),
                    marker=dict(
                        size=[15 if sep else 10 for sep in has_seps],
                        symbol=['star' if sep else 'circle' for sep in has_seps],
                        color='#2196F3',
                        line=dict(width=2, color='white')
                    )
                ))

                # Add current rate band
                current_upper = summary.get('current_rate_upper', 0)
                current_lower = summary.get('current_rate_lower', 0)

                fig.add_hrect(
                    y0=current_lower,
                    y1=current_upper,
                    line_width=0,
                    fillcolor="#4CAF50",
                    opacity=0.15,
                    annotation_text="Current Target Range",
                    annotation_position="top right"
                )

                # Add current midpoint line
                current_mid = summary.get('current_rate_mid', 0)
                fig.add_hline(
                    y=current_mid,
                    line_dash="solid",
                    line_color="#4CAF50",
                    line_width=2,
                    annotation_text=f"Current: {current_mid:.2f}%",
                    annotation_position="right"
                )

                fig.update_layout(
                    title="Market-Implied Fed Funds Rate Path",
                    yaxis_title="Rate (%)",
                    xaxis_title="FOMC Meeting",
                    height=400,
                    template="plotly_dark",
                    hovermode='x unified',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
                )

                st.plotly_chart(fig, width='stretch')

            st.divider()

            # --- INTERPRETATION GUIDE ---
            with st.expander("Understanding Fed Watch Probabilities"):
                st.markdown("""
                ### CME FedWatch Methodology

                This dashboard implements the **CME FedWatch Tool methodology** using:
                - **30-Day Fed Funds Futures (ZQ)**: Direct market pricing of expected EFFR
                - **FRED API**: Real-time Federal Funds Target Rate (upper/lower bounds)
                - **Linear interpolation**: Between bracketing rate levels

                ### Formula
                ```
                Implied Rate = 100 - Futures Price
                Probability = (Implied Change) / 0.25%
                ```

                ### Interpreting Probabilities

                | Probability | Interpretation | Trading Implication |
                |-------------|----------------|---------------------|
                | >90% | Near certainty | Fully priced in, focus on statement/guidance |
                | 70-90% | Strong expectation | Largely priced in, surprise unlikely |
                | 50-70% | Leaning one way | Market has a view but uncertainty remains |
                | 30-50% | Uncertain | Coin flip, high event volatility expected |
                | <30% | Unlikely | Surprise would cause significant moves |

                ### Market Bias Guide

                - **Hold Expected**: >60% probability of no change
                - **Strongly Dovish**: >70% combined cut probability
                - **Dovish**: Cut probability exceeds hike probability
                - **Hawkish**: Hike probability exceeds cut probability
                - **Strongly Hawkish**: >70% combined hike probability
                - **Uncertain**: No clear direction, probabilities roughly balanced

                ### Data Quality Notes

                - **Source**: Fed Funds Futures from Yahoo Finance + FRED API
                - **Update Frequency**: Real-time during market hours (1-minute cache)
                - **Methodology**: Matches CME Group's FedWatch Tool
                - **Limitation**: End-of-day settlement prices used for futures
                """)

            # Data source note with timestamp
            timestamp = summary.get('timestamp', 'N/A')
            st.caption(f"Data source: {data_source} | Rate source: {source} | Last updated: {timestamp}")

    except Exception as e:
        st.error(f"Error loading Fed Watch data: {e}")
        logger.error(f"Fed Watch error: {e}")
        import traceback
        st.code(traceback.format_exc())
