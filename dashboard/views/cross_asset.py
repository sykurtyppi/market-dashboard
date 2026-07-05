"""Cross-Asset page."""
import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_collectors.cross_asset_collector import CrossAssetCollector

logger = logging.getLogger(__name__)


def render(components):
    st.markdown("<h1 class='main-header'>Cross-Asset Analysis</h1>", unsafe_allow_html=True)

    st.markdown("""
    Track correlations between major asset classes to identify regime changes.
    Key relationships reveal risk-on/risk-off dynamics and market stress.
    """)

    @st.cache_resource
    def get_cross_asset_collector():
        return CrossAssetCollector()

    cross_asset = get_cross_asset_collector()

    try:
        # --- REGIME SIGNAL ---
        st.subheader("Current Market Regime")

        regime = cross_asset.get_regime_signal(period='1mo')

        col1, col2 = st.columns([1, 2])

        with col1:
            regime_name = regime.get('regime', 'UNKNOWN')
            regime_color = regime.get('color', '#9E9E9E')
            regime_emoji = regime.get('emoji', '')
            confidence = regime.get('confidence', 0)

            st.markdown(
                f"<div style='text-align:center; padding:30px; background-color:{regime_color}30; border-radius:15px; border: 3px solid {regime_color};'>"
                f"<h1 style='margin:0; font-size:3em;'>{regime_emoji}</h1>"
                f"<h2 style='margin:10px 0; color:{regime_color};'>{regime_name}</h2>"
                f"<p style='margin:0;'>Confidence: {confidence}%</p>"
                f"</div>",
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(f"**{regime.get('description', '')}**")

            # Show asset returns driving the regime
            returns = regime.get('returns', {})
            if returns:
                st.markdown("**30-Day Asset Performance:**")

                ret_cols = st.columns(5)
                assets = ['SPY', 'TLT', 'GLD', 'UUP', 'VIX']
                asset_names = ['S&P 500', 'Bonds', 'Gold', 'Dollar', 'VIX']

                for i, (asset, name) in enumerate(zip(assets, asset_names)):
                    if asset in returns:
                        ret = returns[asset]
                        color = '#4CAF50' if ret > 0 else '#F44336'
                        with ret_cols[i]:
                            st.markdown(
                                f"<div style='text-align:center; padding:10px; background-color:#1e1e1e; border-radius:8px;'>"
                                f"<small>{name}</small><br/>"
                                f"<strong style='color:{color};'>{ret:+.1f}%</strong>"
                                f"</div>",
                                unsafe_allow_html=True
                            )

        st.divider()

        # --- KEY CORRELATIONS ---
        st.subheader("Key Asset Correlations (3-Month)")

        correlations = cross_asset.get_key_correlations(period='3mo')

        if correlations:
            for corr in correlations:
                col1, col2, col3 = st.columns([1, 1, 3])

                with col1:
                    st.markdown(f"**{corr['pair']}**")
                    st.caption(f"{corr['ticker1']} vs {corr['ticker2']}")

                with col2:
                    corr_val = corr['correlation']
                    corr_color = corr['color']
                    st.markdown(
                        f"<div style='text-align:center; padding:10px; background-color:{corr_color}30; border-radius:8px;'>"
                        f"<strong style='font-size:1.3em; color:{corr_color};'>{corr_val:.2f}</strong><br/>"
                        f"<small>{corr['strength']}</small>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                with col3:
                    st.markdown(f"*{corr['interpretation']}*")

                st.markdown("---")

        st.divider()

        # --- CORRELATION MATRIX ---
        st.subheader("Full Correlation Matrix")

        period_select = st.selectbox(
            "Lookback Period",
            ['1mo', '3mo', '6mo'],
            index=1,
            key='corr_period'
        )

        corr_matrix = cross_asset.get_correlation_matrix(period=period_select)

        if corr_matrix is not None and not corr_matrix.empty:
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu_r',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                hovertemplate='%{x} vs %{y}: %{z:.2f}<extra></extra>'
            ))

            fig.update_layout(
                title=f"Asset Correlation Matrix ({period_select})",
                height=500,
                template="plotly_dark"
            )

            st.plotly_chart(fig, width='stretch')
        else:
            st.warning("Could not load correlation matrix")

        st.divider()

        # --- ASSET PERFORMANCE ---
        st.subheader("Cross-Asset Performance Summary")

        perf = cross_asset.get_asset_performance_summary(period='1mo')

        if perf:
            perf_data = []
            for ticker, data in perf.items():
                perf_data.append({
                    'Asset': data.get('name', ticker),
                    'Ticker': ticker,
                    'Price': f"${data.get('price', 0):.2f}",
                    'Change': f"{data.get('change_pct', 0):+.2f}%",
                })

            perf_df = pd.DataFrame(perf_data)

            def color_change(val):
                if '+' in str(val):
                    return 'color: #4CAF50'
                elif '-' in str(val):
                    return 'color: #F44336'
                return ''

            styled_perf = perf_df.style.map(
                color_change,
                subset=['Change']
            )

            st.dataframe(styled_perf, width='stretch', hide_index=True)

        st.divider()

        # --- REGIME GUIDE ---
        with st.expander("Understanding Market Regimes"):
            st.markdown("""
            ### Regime Types

            | Regime | Description | Favored Assets |
            |--------|-------------|----------------|
            | **RISK_ON** | Equities rallying, bonds selling, weak dollar | Stocks, High Yield, EM |
            | **RISK_OFF** | Flight to safety, VIX spiking | Bonds, Gold, Dollar |
            | **INFLATION** | Both stocks & bonds down, commodities up | Commodities, TIPS, Gold |
            | **DEFLATION** | Everything down, dollar strong | Cash, Short-term Bonds |
            | **GOLDILOCKS** | Steady growth, low volatility | Balanced, Quality Growth |

            ### Key Correlation Signals

            **Stock-Bond (SPY-TLT):**
            - Negative = Normal (diversification works)
            - Positive = Stress (2022-style correlation breakdown)

            **Stock-Gold (SPY-GLD):**
            - Negative = Gold as hedge working
            - Positive = Both inflation hedges

            **Stock-Dollar (SPY-UUP):**
            - Negative = Risk-on environment
            - Positive = Unusual, watch closely

            **Credit-Stock (HYG-SPY):**
            - Positive = Normal (credit follows stocks)
            - Divergence = Credit stress warning
            """)

    except Exception as e:
        st.error(f"Error loading cross-asset data: {e}")
        logger.error(f"Cross-asset error: {e}")
