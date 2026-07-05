"""Sentiment page."""
import plotly.graph_objects as go
import streamlit as st


def render(components):
    st.header("Market Sentiment")

    try:
        fg_data = components["fear_greed"].get_fear_greed_score()

        if fg_data:
            score = fg_data["score"]

            if score < 25:
                color, label = "red", "EXTREME FEAR"
            elif score < 45:
                color, label = "orange", "FEAR"
            elif score < 55:
                color, label = "yellow", "NEUTRAL"
            elif score < 75:
                color, label = "lightgreen", "GREED"
            else:
                color, label = "green", "EXTREME GREED"

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=score,
                    title={"text": f"Fear & Greed<br>{label}"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": color},
                        "steps": [
                            {"range": [0, 25], "color": "rgba(255,0,0,0.2)"},
                            {"range": [25, 45], "color": "rgba(255,165,0,0.2)"},
                            {"range": [45, 55], "color": "rgba(255,255,0,0.2)"},
                            {"range": [55, 75], "color": "rgba(144,238,144,0.2)"},
                            {"range": [75, 100], "color": "rgba(0,128,0,0.2)"},
                        ],
                    },
                )
            )

            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current", f"{score:.0f}")
            with col2:
                if fg_data.get("previous_close") is not None:
                    st.metric("Yesterday", f"{fg_data['previous_close']:.0f}")
            with col3:
                if fg_data.get("one_week_ago") is not None:
                    st.metric("Last Week", f"{fg_data['one_week_ago']:.0f}")
        else:
            st.warning("No Fear & Greed data available.")
    except Exception as e:
        st.error(f"Error: {e}")

    # ============================================================
    # PUT/CALL RATIOS
    # ============================================================
    st.divider()
    st.subheader("📊 Put/Call Ratios")
    
    try:
        # Get CBOE data
        cboe = components["cboe"]
        cboe_data = cboe.get_all_data()
        
        # Check for manual override
        use_manual = st.session_state.get('use_manual_equity_pc', False)
        total_pc = None

        if use_manual:
            equity_pc = st.session_state.get('manual_equity_pc', 1.0)
            st.info(f"📝 Using manual equity P/C ratio: {equity_pc:.3f}")
        else:
            # Get nested put_call_ratios dict
            pc_ratios = cboe_data.get("put_call_ratios", {})
            equity_pc = pc_ratios.get("equity_pc")
            total_pc = pc_ratios.get("total_pc")
        
        if equity_pc is not None or total_pc is not None:
            col1, col2 = st.columns(2)
            
            # Equity P/C Ratio
            with col1:
                if equity_pc is not None:
                    # Determine sentiment
                    if equity_pc > 1.2:
                        color, label = "#f44336", "VERY BEARISH"
                    elif equity_pc > 1.0:
                        color, label = "#ff9800", "BEARISH"
                    elif equity_pc > 0.8:
                        color, label = "#9e9e9e", "NEUTRAL"
                    elif equity_pc > 0.6:
                        color, label = "#8bc34a", "BULLISH"
                    else:
                        color, label = "#4caf50", "VERY BULLISH"
                    
                    st.markdown(
                        f"<div style='text-align: center; padding: 1.5rem; background: {color}20; border-radius: 0.5rem; border: 2px solid {color};'>"
                        f"<h3 style='margin:0;'>Equity P/C Ratio</h3>"
                        f"<h1 style='margin:0.5rem 0; color: {color};'>{equity_pc:.3f}</h1>"
                        f"<p style='margin:0; font-size: 1.1rem; font-weight: bold;'>{label}</p>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    
                    st.caption("**Interpretation:** >1.0 = More puts than calls (bearish), <1.0 = More calls than puts (bullish)")
                else:
                    st.warning("Equity P/C ratio unavailable")
            
            # Total P/C Ratio
            with col2:
                if total_pc is not None:
                    # Determine sentiment
                    if total_pc > 1.1:
                        color, label = "#4caf50", "EXTREME FEAR (Contrarian Buy)"
                    elif total_pc > 0.90:
                        color, label = "#8bc34a", "NORMAL/HEALTHY"
                    elif total_pc > 0.80:
                        color, label = "#9e9e9e", "NEUTRAL"
                    elif total_pc > 0.70:
                        color, label = "#ff9800", "LOW (Complacent)"
                    else:
                        color, label = "#f44336", "EXTREME LOW (Danger)"
                    
                    st.markdown(
                        f"<div style='text-align: center; padding: 1.5rem; background: {color}20; border-radius: 0.5rem; border: 2px solid {color};'>"
                        f"<h3 style='margin:0;'>Total P/C Ratio</h3>"
                        f"<h1 style='margin:0.5rem 0; color: {color};'>{total_pc:.3f}</h1>"
                        f"<p style='margin:0; font-size: 1.1rem; font-weight: bold;'>{label}</p>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    
                    st.caption("**Interpretation:** Includes index + equity options (broader sentiment)")
                else:
                    st.warning("Total P/C ratio unavailable")
            
            # Explanation
            st.markdown("""
            ---
            **📖 What are Put/Call Ratios?**
            
            Put/Call ratios measure options trading activity:
            - **Equity P/C**: Individual stock options only (pure directional bets)
            - **Total P/C**: All options including indices (broader sentiment)
            
            **How to interpret:**
            - **> 1.0**: More put buying than calls → **Bearish/protective sentiment**
            - **< 1.0**: More call buying than puts → **Bullish sentiment**
            - **Extreme readings** (>1.5 or <0.5) often mark sentiment extremes and potential reversals
            
            **Note:** Can be manually overridden in Settings page if data is unavailable.
            """)
        else:
            st.warning("⚠️ Put/Call ratio data unavailable. You can set manual values in Settings.")
            
    except Exception as e:
        st.error(f"Error loading P/C ratios: {e}")
        import traceback
        st.code(traceback.format_exc())
