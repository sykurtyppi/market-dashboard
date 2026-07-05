"""Options Flow page."""
import logging

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_collectors.options_flow_collector import OptionsFlowCollector, get_market_status

logger = logging.getLogger(__name__)


def render(components):
    st.markdown("<h1 class='main-header'>Options Flow Scanner</h1>", unsafe_allow_html=True)

    # Get market status from the collector (centralized logic)
    market_status_info = get_market_status()
    market_open = market_status_info['is_open']
    status_text = f"{'🟢' if market_open else '🔴'} Market {market_status_info['status']}"
    status_color = market_status_info['status_color']

    # Header with market status
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown("""
        **Institutional-grade** unusual options activity scanner across 35+ tickers.
        Tracks premium flow, volume/OI ratios, and sentiment signals.
        """)
    with col_h2:
        st.markdown(
            f"<div style='text-align:right; padding:8px; background-color:{status_color}20; border-radius:8px; border:1px solid {status_color};'>"
            f"<strong style='color:{status_color};'>{status_text}</strong><br/>"
            f"<small style='color:#888;'>{market_status_info['current_time']}</small>"
            f"</div>",
            unsafe_allow_html=True
        )

    @st.cache_resource(ttl=120)
    def get_options_flow_collector():
        return OptionsFlowCollector()

    options_flow = get_options_flow_collector()

    try:
        with st.spinner("Scanning 35+ tickers for unusual options activity..."):
            flow_summary = options_flow.get_options_flow_summary()

        # --- OVERALL FLOW DASHBOARD ---
        st.subheader("Flow Dashboard")

        # Main metrics row
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        signal = flow_summary.get('overall_signal', 'MIXED FLOW')
        signal_color = flow_summary.get('signal_color', '#FF9800')
        bullish = flow_summary.get('bullish_signals', 0)
        bearish = flow_summary.get('bearish_signals', 0)
        neutral = flow_summary.get('neutral_signals', 0)
        unique_tickers = flow_summary.get('unique_tickers', 0)
        total_premium = flow_summary.get('total_premium') or 0

        with col1:
            st.markdown(
                f"<div style='text-align:center; padding:12px; background-color:{signal_color}25; border-radius:10px; border:2px solid {signal_color};'>"
                f"<small style='color:#888;'>OVERALL</small><br/>"
                f"<strong style='font-size:1em; color:{signal_color};'>{signal}</strong>"
                f"</div>",
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f"<div style='text-align:center; padding:12px; background-color:#4CAF5015; border-radius:10px;'>"
                f"<small style='color:#888;'>BULLISH</small><br/>"
                f"<strong style='font-size:1.8em; color:#4CAF50;'>{bullish}</strong>"
                f"</div>",
                unsafe_allow_html=True
            )

        with col3:
            st.markdown(
                f"<div style='text-align:center; padding:12px; background-color:#F4433615; border-radius:10px;'>"
                f"<small style='color:#888;'>BEARISH</small><br/>"
                f"<strong style='font-size:1.8em; color:#F44336;'>{bearish}</strong>"
                f"</div>",
                unsafe_allow_html=True
            )

        with col4:
            st.markdown(
                f"<div style='text-align:center; padding:12px; background-color:#9E9E9E15; border-radius:10px;'>"
                f"<small style='color:#888;'>NEUTRAL</small><br/>"
                f"<strong style='font-size:1.8em; color:#9E9E9E;'>{neutral}</strong>"
                f"</div>",
                unsafe_allow_html=True
            )

        with col5:
            st.markdown(
                f"<div style='text-align:center; padding:12px; background-color:#2196F315; border-radius:10px;'>"
                f"<small style='color:#888;'>TICKERS</small><br/>"
                f"<strong style='font-size:1.8em; color:#2196F3;'>{unique_tickers}</strong>"
                f"</div>",
                unsafe_allow_html=True
            )

        with col6:
            premium_str = f"${total_premium/1e6:.1f}M" if total_premium >= 1e6 else f"${total_premium/1e3:.0f}K"
            st.markdown(
                f"<div style='text-align:center; padding:12px; background-color:#FF980015; border-radius:10px;'>"
                f"<small style='color:#888;'>PREMIUM</small><br/>"
                f"<strong style='font-size:1.4em; color:#FF9800;'>{premium_str}</strong>"
                f"</div>",
                unsafe_allow_html=True
            )

        st.divider()

        # --- INDEX SENTIMENT WITH CHARTS ---
        st.subheader("Index Options Sentiment")

        spy_summary = flow_summary.get('spy_summary', {})
        qqq_summary = flow_summary.get('qqq_summary', {})
        iwm_summary = flow_summary.get('iwm_summary', {})

        col1, col2, col3 = st.columns(3)

        for col, summary, name in [(col1, spy_summary, 'SPY'), (col2, qqq_summary, 'QQQ'), (col3, iwm_summary, 'IWM')]:
            with col:
                if summary and summary.get('status') == 'ok':
                    pc_ratio = summary.get('put_call_ratio', 0)
                    sentiment = summary.get('sentiment', 'NEUTRAL')
                    sentiment_color = summary.get('color', '#9E9E9E')
                    call_vol = summary.get('total_call_volume', 0)
                    put_vol = summary.get('total_put_volume', 0)
                    current_price = summary.get('current_price', 0)
                    call_oi = summary.get('total_call_oi', 0)
                    put_oi = summary.get('total_put_oi', 0)

                    # Calculate percentages
                    total_vol = call_vol + put_vol
                    call_pct = (call_vol / total_vol * 100) if total_vol > 0 else 50
                    put_pct = (put_vol / total_vol * 100) if total_vol > 0 else 50

                    st.markdown(
                        f"<div style='padding:15px; background-color:#1e1e1e; border-radius:10px; border-left:4px solid {sentiment_color};'>"
                        f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
                        f"<span style='font-size:1.3em; font-weight:bold;'>{name}</span>"
                        f"<span style='color:{sentiment_color}; font-weight:bold;'>{sentiment}</span>"
                        f"</div>"
                        f"<div style='margin-top:8px; color:#888;'>${current_price:.2f}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                    # Volume bar visualization
                    fig_bar = go.Figure()
                    fig_bar.add_trace(go.Bar(
                        x=[call_vol], y=['Volume'], orientation='h',
                        name='Calls', marker_color='#4CAF50',
                        text=f'{call_vol:,.0f}', textposition='inside'
                    ))
                    fig_bar.add_trace(go.Bar(
                        x=[put_vol], y=['Volume'], orientation='h',
                        name='Puts', marker_color='#F44336',
                        text=f'{put_vol:,.0f}', textposition='inside'
                    ))
                    fig_bar.update_layout(
                        barmode='stack', height=60, margin=dict(l=0, r=0, t=0, b=0),
                        showlegend=False, template='plotly_dark',
                        xaxis=dict(showticklabels=False, showgrid=False),
                        yaxis=dict(showticklabels=False),
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_bar, width='stretch', key=f"bar_{name}")

                    # P/C Ratio display
                    ratio_color = '#F44336' if pc_ratio > 1.2 else ('#4CAF50' if pc_ratio < 0.8 else '#9E9E9E')
                    st.markdown(
                        f"<div style='display:flex; justify-content:space-between; padding:5px 0;'>"
                        f"<span style='color:#888;'>P/C Ratio</span>"
                        f"<strong style='color:{ratio_color};'>{pc_ratio:.2f}</strong>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.warning(f"{name} unavailable")

        st.divider()

        # --- TOP UNUSUAL ACTIVITY - CARD VIEW ---
        st.subheader("Top Unusual Activity")

        unusual = flow_summary.get('unusual_activity', [])

        if unusual:
            # Display as professional cards
            for i in range(0, min(len(unusual), 12), 3):
                cols = st.columns(3)
                for j, col in enumerate(cols):
                    if i + j < len(unusual):
                        activity = unusual[i + j]
                        ticker = activity.get('ticker', '')
                        opt_type = activity.get('type', '')
                        strike = activity.get('strike')
                        expiry = activity.get('expiry', '')
                        dte_label = activity.get('dte_label', '')
                        volume = activity.get('volume') or 0
                        oi = activity.get('open_interest') or 0
                        vol_oi = activity.get('vol_oi_ratio')
                        signal = activity.get('signal', '')
                        sentiment = activity.get('sentiment', '')
                        score = activity.get('score', 0)
                        premium = activity.get('premium_value') or 0
                        stock_price = activity.get('stock_price', 0)
                        moneyness = activity.get('moneyness_pct')

                        if strike is None or pd.isna(strike):
                            strike = 0
                        if vol_oi is None or pd.isna(vol_oi):
                            vol_oi = 0
                        if moneyness is None or pd.isna(moneyness):
                            moneyness = 0

                        # Colors and emojis
                        if sentiment == 'BULLISH':
                            sent_color = '#4CAF50'
                            emoji = '📈'
                            border_color = '#4CAF50'
                        elif sentiment == 'BEARISH':
                            sent_color = '#F44336'
                            emoji = '📉'
                            border_color = '#F44336'
                        else:
                            sent_color = '#9E9E9E'
                            emoji = '⚖️'
                            border_color = '#9E9E9E'

                        # Format values
                        premium_str = f"${premium/1e6:.1f}M" if premium >= 1e6 else f"${premium/1e3:.0f}K"
                        vol_str = f"{volume:,}"
                        oi_str = f"{oi:,}" if oi else "—"
                        vol_oi_str = f"{vol_oi:.1f}x" if vol_oi else "—"
                        strike_str = f"${strike:.0f}" if strike else "—"
                        moneyness_str = f"{moneyness:+.1f}%" if moneyness else "ATM"
                        expiry_display = f"{expiry} ({dte_label})" if dte_label else expiry

                        with col:
                            st.markdown(
                                f"""<div style='padding:15px; background-color:#1a1a1a; border-radius:12px;
                                    border:1px solid #333; border-left:4px solid {border_color}; margin-bottom:10px;'>
                                <div style='display:flex; justify-content:space-between; align-items:center;'>
                                    <span style='font-size:1.4em; font-weight:bold;'>{ticker}</span>
                                    <span style='background-color:{sent_color}25; color:{sent_color};
                                        padding:4px 10px; border-radius:15px; font-size:0.85em;'>
                                        {emoji} {opt_type}
                                    </span>
                                </div>
                                <div style='margin-top:10px; display:grid; grid-template-columns:1fr 1fr; gap:8px;'>
                                    <div>
                                        <small style='color:#666;'>Strike</small><br/>
                                        <strong>{strike_str}</strong> <small style='color:#888;'>({moneyness_str})</small>
                                    </div>
                                    <div>
                                        <small style='color:#666;'>Expiry</small><br/>
                                        <strong>{expiry_display}</strong>
                                    </div>
                                    <div>
                                        <small style='color:#666;'>Volume</small><br/>
                                        <strong style='color:#2196F3;'>{vol_str}</strong>
                                    </div>
                                    <div>
                                        <small style='color:#666;'>Vol/OI</small><br/>
                                        <strong style='color:#FF9800;'>{vol_oi_str}</strong>
                                    </div>
                                </div>
                                <div style='margin-top:12px; padding-top:10px; border-top:1px solid #333;'>
                                    <div style='display:flex; justify-content:space-between; align-items:center;'>
                                        <span style='color:#888; font-size:0.85em;'>{signal}</span>
                                        <span style='color:#FF9800; font-weight:bold;'>{premium_str}</span>
                                    </div>
                                </div>
                                </div>""",
                                unsafe_allow_html=True
                            )

            st.divider()

            # --- FLOW BY TICKER CHART ---
            st.subheader("Flow Distribution")

            col1, col2 = st.columns(2)

            with col1:
                # Premium by ticker
                ticker_premium = {}
                for u in unusual:
                    t = u.get('ticker', 'Unknown')
                    p = u.get('premium_value', 0) or 0
                    ticker_premium[t] = ticker_premium.get(t, 0) + p

                sorted_premium = sorted(ticker_premium.items(), key=lambda x: -x[1])[:8]

                if sorted_premium:
                    fig_premium = go.Figure()
                    fig_premium.add_trace(go.Bar(
                        x=[t[0] for t in sorted_premium],
                        y=[t[1]/1e6 for t in sorted_premium],
                        marker_color='#FF9800',
                        text=[f"${t[1]/1e6:.1f}M" for t in sorted_premium],
                        textposition='outside'
                    ))
                    fig_premium.update_layout(
                        title="Premium by Ticker ($M)",
                        height=300,
                        template='plotly_dark',
                        showlegend=False,
                        margin=dict(l=40, r=20, t=40, b=40),
                        yaxis_title="Premium ($M)"
                    )
                    st.plotly_chart(fig_premium, width='stretch')

            with col2:
                # Sentiment distribution pie
                fig_sentiment = go.Figure()
                fig_sentiment.add_trace(go.Pie(
                    labels=['Bullish', 'Bearish', 'Neutral'],
                    values=[bullish, bearish, neutral],
                    marker_colors=['#4CAF50', '#F44336', '#9E9E9E'],
                    hole=0.5,
                    textinfo='label+percent',
                    textposition='outside'
                ))
                fig_sentiment.update_layout(
                    title="Sentiment Distribution",
                    height=300,
                    template='plotly_dark',
                    margin=dict(l=20, r=20, t=40, b=20),
                    showlegend=False
                )
                st.plotly_chart(fig_sentiment, width='stretch')

            st.divider()

            # --- ACTIVITY BREAKDOWN ---
            st.subheader("Activity Breakdown")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**By Ticker**")
                ticker_counts = {}
                for u in unusual:
                    t = u.get('ticker', 'Unknown')
                    ticker_counts[t] = ticker_counts.get(t, 0) + 1

                for t, c in sorted(ticker_counts.items(), key=lambda x: -x[1])[:6]:
                    st.markdown(f"<span style='color:#2196F3;'>{t}</span>: {c} signals", unsafe_allow_html=True)

            with col2:
                st.markdown("**By Category**")
                cat_counts = {}
                for u in unusual:
                    c = u.get('category', 'unknown')
                    cat_counts[c] = cat_counts.get(c, 0) + 1

                for c, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
                    st.markdown(f"<span style='color:#FF9800;'>{c}</span>: {n} signals", unsafe_allow_html=True)

            with col3:
                st.markdown("**By Signal Type**")
                signal_counts = {}
                for u in unusual:
                    s = u.get('signal_type', 'unknown')
                    signal_counts[s] = signal_counts.get(s, 0) + 1

                for s, n in sorted(signal_counts.items(), key=lambda x: -x[1]):
                    label = s.replace('_', ' ').title()
                    st.markdown(f"<span style='color:#9E9E9E;'>{label}</span>: {n}", unsafe_allow_html=True)

        else:
            st.info("No unusual options activity detected. This may occur outside market hours or on low-volume days.")

        st.divider()

        # --- STRADDLE SETUPS ---
        straddles = flow_summary.get('straddle_setups', [])
        if straddles:
            st.subheader("Straddle Setups (Rangebound Signals)")

            st.markdown("""
            <small style='color:#888;'>Balanced ATM call + put volume suggests expected rangebound action.
            The implied range shows the expected move based on straddle pricing.</small>
            """, unsafe_allow_html=True)

            for s in straddles[:3]:
                col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1.5])

                with col1:
                    st.markdown(
                        f"<div style='padding:10px; background:#1a1a1a; border-radius:8px; border-left:4px solid #9E9E9E;'>"
                        f"<strong style='font-size:1.2em;'>{s.get('ticker', '')}</strong> @ ${s.get('strike', 0):.0f}"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                with col2:
                    implied = s.get('implied_range_pct', 0)
                    st.metric("Implied Range", f"±{implied:.1f}%")

                with col3:
                    cost = s.get('straddle_cost', 0)
                    st.metric("Straddle Cost", f"${cost:.2f}")

                with col4:
                    call_v = s.get('call_volume', 0)
                    put_v = s.get('put_volume', 0)
                    st.markdown(f"Calls: **{call_v:,}** | Puts: **{put_v:,}**")

            st.divider()

        # --- INTERPRETATION GUIDE ---
        with st.expander("Understanding Options Flow"):
            st.markdown("""
            ### Signal Types

            | Signal | Description | Implication |
            |--------|-------------|-------------|
            | **High Vol/OI** | Volume >> Open Interest | New positions being opened |
            | **Large Premium** | $500K+ premium on single contract | Institutional-size bet |
            | **ATM Activity** | High volume near current price | Higher conviction play |
            | **Straddle** | Balanced ATM call + put volume | Hedging or rangebound expected |
            | **Put Heavy** | P/C ratio > 2.0 | Bearish or hedging |
            | **Call Heavy** | P/C ratio < 0.4 | Bullish sentiment |

            ### Categories Scanned (35+ Tickers)

            - **Indices:** SPY, QQQ, IWM, DIA, VIX
            - **Mega-Cap:** AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA
            - **Financials:** JPM, BAC, GS, MS, C
            - **Tech:** AMD, INTC, CRM, ORCL, ADBE
            - **Semis:** AVGO, QCOM, MU, AMAT, LRCX
            - **Consumer:** NFLX, DIS, SBUX, MCD, NKE
            - **Speculative:** COIN, GME, AMC, MARA, RIOT

            ### Scoring System

            - **Volume/OI Ratio:** Higher ratio = higher score (new interest)
            - **Relative Volume:** What % of chain volume is this contract?
            - **Premium Size:** Larger premium = more institutional interest
            - **Per-Ticker Limit:** Max 3 findings per ticker for diversity

            ### Important Caveats

            1. **Delayed Data**: Not real-time (Yahoo Finance)
            2. **No Direction**: Can't distinguish buy vs sell
            3. **No Sweep Detection**: Can't identify aggressive sweeps
            4. **Context Matters**: Put buying could be hedging, not bearish

            ### Straddle/Rangebound Interpretation

            When ATM calls AND puts both have high volume:
            - **Implied Range** = Straddle Cost / Stock Price
            - If implied range is small (1-2%), market expects rangebound
            - If implied range is large (5%+), market expects big move
            """)

        # Freshness indicator with styling
        freshness_msg = flow_summary.get('freshness_msg', 'Data status unknown')
        freshness_color = flow_summary.get('freshness_color', '#9E9E9E')
        data_note = flow_summary.get('data_note', '')

        st.markdown(
            f"<div style='padding:10px; background-color:#1a1a1a; border-radius:8px; margin-top:10px;'>"
            f"<span style='color:{freshness_color}; font-weight:bold;'>📊 {freshness_msg}</span>"
            f"<br/><small style='color:#666;'>Tickers scanned: 35+ | {data_note}</small>"
            f"</div>",
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"Error scanning options flow: {e}")
        logger.error(f"Options flow error: {e}")
        import traceback
        st.code(traceback.format_exc())
