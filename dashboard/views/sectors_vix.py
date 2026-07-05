"""Sectors & VIX page."""
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.core.data import get_sector_comparison_chart, get_sector_performance, get_vix_term_structure


def render(components):
    try:
        st.header("Sector Rotation & VIX Analysis")

        st.subheader("VIX Term Structure")
        vix_term = get_vix_term_structure()

        if not vix_term.empty:
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=vix_term["Maturity"],
                    y=vix_term["VIX Level"],
                    mode="lines+markers",
                    name="VIX Term Structure",
                    line=dict(color="royalblue", width=3),
                    marker=dict(size=12, color="royalblue"),
                    text=[f"{v:.2f}" for v in vix_term["VIX Level"]],
                    textposition="top center",
                    hovertemplate="<b>%{x}</b><br>VIX: %{y:.2f}<extra></extra>"
                )
            )

            # Use VIX3M vs VIX as the primary contango definition.
            # This matches the core dashboard methodology and avoids
            # misleading "front/back" values when 1D and 1Y are present.
            level_map = {row["Maturity"]: float(row["VIX Level"]) for _, row in vix_term.iterrows()}
            vix_spot_level = level_map.get("VIX")
            vix3m_level = level_map.get("3M")

            if (
                vix_spot_level is not None
                and vix3m_level is not None
                and vix_spot_level > 0
            ):
                spread = vix3m_level - vix_spot_level
                spread_pct = (spread / vix_spot_level) * 100
                spread_label = "VIX3M - VIX"
                spread_basis = "based on VIX3M vs spot VIX"
            else:
                front = float(vix_term["VIX Level"].iloc[0])
                back = float(vix_term["VIX Level"].iloc[-1])
                spread = back - front
                spread_pct = (spread / front) * 100 if front > 0 else 0
                spread_label = "Back - Front"
                spread_basis = "fallback: first vs last curve point"

            if spread > 0:
                contango_text = "Contango"
                structure_desc = "bullish - normal market conditions"
                color = "green"
            else:
                contango_text = "Backwardation"
                structure_desc = "risk-off - elevated near-term fear"
                color = "red"

            fig.update_layout(
                title=f"VIX Term Structure - {contango_text} ({spread_pct:+.1f}%)",
                xaxis_title="Maturity",
                yaxis_title="VIX Level",
                height=400,
                hovermode="x unified",
                annotations=[
                    dict(
                        text=f"{contango_text}: {structure_desc} ({spread_basis})",
                        x=0.5,
                        y=1.08,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        font=dict(size=12, color=color),
                    )
                ],
            )

            st.plotly_chart(fig, width='stretch')

            # Show summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "VIX Spot (30D)",
                    f"{vix_spot_level:.2f}" if vix_spot_level is not None else "N/A",
                )
            with col2:
                st.metric(
                    "VIX 3M (90D)",
                    f"{vix3m_level:.2f}" if vix3m_level is not None else "N/A",
                )
            with col3:
                delta_color = "normal" if spread > 0 else "inverse"
                st.metric(
                    spread_label,
                    f"{spread:+.2f}",
                    delta=f"{spread_pct:+.1f}%",
                    delta_color=delta_color,
                )

            if vix_spot_level is None or vix3m_level is None:
                st.caption("Contango uses fallback curve slope because VIX or VIX3M is unavailable.")
        else:
            st.warning("VIX term structure data unavailable. Unable to fetch VIX indices.")

        st.divider()

        st.subheader("Sector Performance Analysis")

        col1, col2 = st.columns([3, 1])

        with col1:
            period_options = {
                "1 Day": "1d",
                "1 Week": "5d",
                "1 Month": "1mo",
                "3 Months": "3mo",
                "6 Months": "6mo",
                "YTD": "ytd",
                "1 Year": "1y",
                "5 Years": "5y",
            }

            selected_period_label = st.radio(
                "Select Time Period:",
                options=list(period_options.keys()),
                horizontal=True,
                index=2,
            )

        with col2:
            view_mode = st.selectbox("View:", ["Bar Chart", "Line Chart", "Table Only"])

        period_code = period_options[selected_period_label]
        
        try:
            sectors = get_sector_performance(period_code)
        except Exception as e:
            st.error(f"Error fetching sector data: {e}")
            st.stop()

        if not sectors.empty:
            if view_mode == "Bar Chart":
                fig = px.bar(
                    sectors,
                    x="Sector",
                    y="Change %",
                    color="Change %",
                    color_continuous_scale=["red", "yellow", "green"],
                    title=f"Sector Returns - {selected_period_label}",
                    text="Change %",
                )

                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig.update_layout(height=450, showlegend=False)
                st.plotly_chart(fig, width='stretch')

            elif view_mode == "Line Chart":
                comparison_data = get_sector_comparison_chart(period_code)

                if not comparison_data.empty:
                    fig = go.Figure()

                    for col in comparison_data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=comparison_data.index,
                                y=comparison_data[col],
                                mode="lines",
                                name=col,
                                line=dict(width=2),
                            )
                        )

                    fig.update_layout(
                        title=f"Sector Performance Comparison - {selected_period_label} (Base 100)",
                        xaxis_title="Date",
                        yaxis_title="Performance (Base 100)",
                        height=500,
                        hovermode="x unified",
                    )

                    st.plotly_chart(fig, width='stretch')

            st.divider()

            st.subheader("Detailed Sector Data")

            sectors["Rank"] = range(1, len(sectors) + 1)
            display_df = sectors[["Rank", "Sector", "Ticker", "Change %", "Price", "Volatility"]]

            st.dataframe(
                display_df.style.format(
                    {
                        "Change %": "{:+.2f}%",
                        "Price": "${:.2f}",
                        "Volatility": "{:.2f}%",
                    }
                ),
                width='stretch',
                hide_index=True,
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Best Performer",
                    sectors.iloc[0]["Sector"],
                    f"{sectors.iloc[0]['Change %']:+.2f}%",
                )
            with col2:
                st.metric(
                    "Worst Performer",
                    sectors.iloc[-1]["Sector"],
                    f"{sectors.iloc[-1]['Change %']:+.2f}%",
                )
            with col3:
                st.metric("Average Return", f"{sectors['Change %'].mean():+.2f}%")
            with col4:
                advancing = (sectors["Change %"] > 0).sum()
                st.metric("Sectors Advancing", f"{advancing}/{len(sectors)}")
        else:
            st.error("Unable to fetch sector data.")
    except Exception as e:
        st.error(f"Error loading Sectors page: {e}")
        import traceback
        st.code(traceback.format_exc())
