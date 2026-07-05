"""CTA Flow Tracker page."""
import logging

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

logger = logging.getLogger(__name__)


def render(components):
    st.markdown("<h1 class='main-header'> CTA Flow Tracker</h1>", unsafe_allow_html=True)

    st.markdown("""
    Track systematic trend-following flows. CTAs control $400B+ and create self-reinforcing
    momentum at key technical levels. **Flip levels** show where CTAs likely adjust positions.
    """)

    # Use cloud-compatible CTA collector (no database needed)
    @st.cache_data(ttl=900)  # Cache for 15 minutes
    def get_cloud_cta_result():
        """Get CTA analysis using cloud-compatible collector"""
        try:
            from data_collectors.cta_collector_cloud import CTACollectorCloud
            collector = CTACollectorCloud()
            return collector.get_cta_analysis(period="2y")
        except Exception as e:
            logger.error(f"CTA analysis failed: {e}")
            return None

    # Refresh button
    if st.button("🔄 Refresh Analysis", help="Recompute CTA signals with latest data"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner("Running CTA analysis (fetching 2 years of price data)..."):
        result = get_cloud_cta_result()

    if result is None:
        st.warning("⚠️ Could not fetch CTA data. Please try again later.")
        st.info("This may be due to Yahoo Finance rate limiting. Wait a moment and click Refresh.")
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
        st.subheader("📍 CTA Flip Levels")
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
            st.dataframe(df_display, width='stretch')

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
            st.plotly_chart(fig, width='stretch')

        st.divider()

        # Signal interpretation
        st.subheader("📈 Trading Implications")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🟢 Strong Longs (>0.3)**")
            strong_longs = result.latest_exposure[result.latest_exposure > 0.3].sort_values(ascending=False)
            if not strong_longs.empty:
                for sym, exp in strong_longs.items():
                    st.markdown(f"- **{sym}**: {exp:.2f}")
            else:
                st.markdown("*None*")

        with col2:
            st.markdown("**🔴 Strong Shorts (<-0.3)**")
            strong_shorts = result.latest_exposure[result.latest_exposure < -0.3].sort_values()
            if not strong_shorts.empty:
                for sym, exp in strong_shorts.items():
                    st.markdown(f"- **{sym}**: {exp:.2f}")
            else:
                st.markdown("*None*")

        st.info("💡 **Strategy:** Watch price action near flip levels. Breaking above = CTA buying. Breaking below = CTA selling.")
