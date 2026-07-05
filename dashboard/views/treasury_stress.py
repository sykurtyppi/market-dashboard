"""Treasury Stress (MOVE) page."""
import logging

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.core.helpers import get_status_color

logger = logging.getLogger(__name__)


def render(components):
    st.markdown(
        "<h1 class='main-header'> Treasury Market Stress</h1>",
        unsafe_allow_html=True,
    )
    
    try:
        move_snapshot = components["move"].get_full_snapshot()
        
        if not move_snapshot or 'move_index' not in move_snapshot:
            st.error("MOVE Index data unavailable. Please check data collection.")
            st.stop()
        
        move_df = move_snapshot.get('move_df')
        
        # Get VIX history
        try:
            vix_df = components["market"].get_vix_history(lookback_days=365)
        except Exception as e:
            logger.debug(f"VIX history fetch failed: {e}")
            vix_df = pd.DataFrame()
        
        if move_df is not None and not move_df.empty and not vix_df.empty:
            vix_series = vix_df['close'] if 'close' in vix_df.columns else vix_df.iloc[:, 0]
            treasury_signal = components["treasury_analyzer"].analyze(move_df, vix_series)
        else:
            treasury_signal = None
    
    except Exception as e:
        st.error(f"Error loading Treasury data: {e}")
        logger.error(f"Error in treasury_stress page: {e}")
        st.stop()
    
    st.subheader(" Current Treasury Market Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        move_value = move_snapshot['move_index']
        stress_level = move_snapshot['stress_level']
        color = get_status_color(stress_level)
        
        st.markdown(
            f"<div style='text-align: center; padding: 1.5rem; background: {color}20; border-radius: 0.5rem;'>"
            f"<h4 style='margin:0;'>MOVE Index</h4>"
            f"<h1 style='margin:0.5rem 0; color: {color};'>{move_value:.1f}</h1>"
            f"<h3 style='margin:0; color: {color};'>{stress_level}</h3>"
            f"</div>",
            unsafe_allow_html=True,
        )
    
    with col2:
        percentile = move_snapshot.get('percentile', 50)
        
        if percentile < 25:
            desc = "Very Calm"
            color = "#4CAF50"
        elif percentile < 50:
            desc = "Calm"
            color = "#8BC34A"
        elif percentile < 75:
            desc = "Active"
            color = "#FF9800"
        else:
            desc = "Elevated"
            color = "#F44336"
        
        st.markdown(
            f"<div style='text-align: center; padding: 1.5rem; background: {color}20; border-radius: 0.5rem;'>"
            f"<h4 style='margin:0;'>Historical Percentile</h4>"
            f"<h1 style='margin:0.5rem 0; color: {color};'>{percentile:.0f}th</h1>"
            f"<h3 style='margin:0; color: {color};'>{desc}</h3>"
            f"</div>",
            unsafe_allow_html=True,
        )
    
    with col3:
        if treasury_signal:
            stress_regime = treasury_signal.stress_level
            strength = treasury_signal.strength
            color = get_status_color(stress_regime)
            
            st.markdown(
                f"<div style='text-align: center; padding: 1.5rem; background: {color}20; border-radius: 0.5rem;'>"
                f"<h4 style='margin:0;'>Treasury Regime</h4>"
                f"<h1 style='margin:0.5rem 0; color: {color};'>{stress_regime}</h1>"
                f"<p style='margin:0;'>Strength: {strength:.0f}/100</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.warning("Analysis unavailable")
    
    with col4:
        if treasury_signal and hasattr(treasury_signal, 'divergence_type') and treasury_signal.divergence_type:
            div_type = treasury_signal.divergence_type
            
            if "Leading" in div_type:
                color = "#FF9800"
                icon = "⚠️"
            else:
                color = "#4CAF50"
                icon = "✓"
            
            st.markdown(
                f"<div style='text-align: center; padding: 1.5rem; background: {color}20; border-radius: 0.5rem;'>"
                f"<h4 style='margin:0;'>MOVE-VIX Divergence</h4>"
                f"<h2 style='margin:0.5rem 0; color: {color};'>{icon}</h2>"
                f"<p style='margin:0; font-size: 0.9rem;'>{div_type}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("Divergence analysis unavailable")
    
    st.markdown("---")
    
    with st.expander("ℹ️ What is the MOVE Index?", expanded=False):
        st.markdown("""
        **The MOVE Index** (Merrill Option Volatility Estimate) is like the VIX but for Treasury bonds:
        
        - **< 80**: Low stress - calm Treasury market  
        - **80-120**: Normal stress - typical market conditions  
        - **120-150**: Elevated stress - increased uncertainty  
        - **> 150**: High stress - crisis territory
        
        Treasury stress often **precedes** equity stress.
        """)
    
    tab1, tab2, tab3 = st.tabs([" MOVE History", " MOVE vs VIX", "Treasury Stress Regime"])
    
    with tab1:
        st.subheader("MOVE Index Historical Chart")
        
        if move_df is not None and not move_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=move_df['date'],
                y=move_df['move'],
                name='MOVE Index',
                line=dict(color='#1f77b4', width=2),
                fill='tozeroy',
                fillcolor='rgba(31, 119, 180, 0.2)'
            ))
            fig.add_hline(y=80, line_dash="dash", line_color="green",
                         annotation_text="Normal Lower", annotation_position="right")
            fig.add_hline(y=120, line_dash="dash", line_color="orange",
                         annotation_text="Elevated", annotation_position="right")
            fig.add_hline(y=150, line_dash="dash", line_color="red",
                         annotation_text="High Stress", annotation_position="right")
            fig.update_layout(
                title="MOVE Index - Last 2 Years",
                xaxis_title="Date",
                yaxis_title="MOVE Index",
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig, width='stretch')
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current MOVE", f"{move_df['move'].iloc[-1]:.1f}")
            with col2:
                st.metric("30-Day Average", f"{move_df['move'].tail(30).mean():.1f}")
            with col3:
                st.metric("1-Year High", f"{move_df['move'].max():.1f}")
        else:
            st.warning("MOVE historical data unavailable")
    
    with tab2:
        st.subheader("MOVE vs VIX Divergence Analysis")
        
        if move_df is not None and not move_df.empty and not vix_df.empty:
            merged = move_df.merge(vix_df[['date', 'close']], on='date', how='inner')
            merged.rename(columns={'close': 'vix'}, inplace=True)
            
            merged['move_norm'] = (merged['move'] - merged['move'].min()) / (merged['move'].max() - merged['move'].min()) * 100
            merged['vix_norm'] = (merged['vix'] - merged['vix'].min()) / (merged['vix'].max() - merged['vix'].min()) * 100
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=merged['date'],
                y=merged['move_norm'],
                name='MOVE (normalized)',
                line=dict(color='#1f77b4', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=merged['date'],
                y=merged['vix_norm'],
                name='VIX (normalized)',
                line=dict(color='#F44336', width=2)
            ))
            fig.update_layout(
                title="MOVE vs VIX - Normalized Comparison",
                xaxis_title="Date",
                yaxis_title="Normalized Value (0-100)",
                hovermode='x unified',
                height=500,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig, width='stretch')
            
            if treasury_signal and hasattr(treasury_signal, 'divergence_type') and treasury_signal.divergence_type:
                st.info(f"**Current Divergence:** {treasury_signal.divergence_type}")
                if hasattr(treasury_signal, 'description'):
                    st.write(treasury_signal.description)
        else:
            st.warning("Insufficient data for divergence analysis")
    
    with tab3:
        st.subheader("Treasury Stress Regime Classification")
        
        if treasury_signal and move_df is not None and not move_df.empty:
            regimes = []
            for _, row in move_df.iterrows():
                if row['move'] < 80:
                    regimes.append(('LOW', '#4CAF50'))
                elif row['move'] < 120:
                    regimes.append(('NORMAL', '#8BC34A'))
                elif row['move'] < 150:
                    regimes.append(('ELEVATED', '#FF9800'))
                else:
                    regimes.append(('HIGH', '#F44336'))
            
            move_df['regime'] = [r[0] for r in regimes]
            move_df['regime_color'] = [r[1] for r in regimes]
            
            fig = go.Figure()
            for regime_name, color in [('LOW', '#4CAF50'), ('NORMAL', '#8BC34A'),
                                       ('ELEVATED', '#FF9800'), ('HIGH', '#F44336')]:
                regime_data = move_df[move_df['regime'] == regime_name]
                if not regime_data.empty:
                    fig.add_trace(go.Scatter(
                        x=regime_data['date'],
                        y=[1] * len(regime_data),
                        name=regime_name,
                        mode='markers',
                        marker=dict(color=color, size=8, symbol='square')
                    ))
            fig.update_layout(
                title="Treasury Stress Regime Over Time",
                xaxis_title="Date",
                yaxis_title="",
                yaxis=dict(showticklabels=False),
                hovermode='x unified',
                height=200,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.warning("Regime classification unavailable")
