"""LEFT Strategy page."""
import plotly.graph_objects as go
import streamlit as st


def render(components):
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
