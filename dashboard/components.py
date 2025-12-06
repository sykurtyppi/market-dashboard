"""
Reusable UI components for the dashboard
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd

def metric_card(label: str, value: str, delta: str = None, delta_color: str = "normal"):
    """Display a metric card"""
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)

def signal_badge(signal: str) -> str:
    """Return colored badge for signal"""
    colors = {
        'BUY': '🟢',
        'SELL': '🔴',
        'NEUTRAL': '🟡',
        'OVERSOLD': '🟢',
        'OVERBOUGHT': '🔴',
        'BULLISH': '🟢',
        'BEARISH': '🔴'
    }
    return f"{colors.get(signal, '⚪')} **{signal}**"

def strength_gauge(value: float, title: str = "Strength"):
    """Create a gauge chart for signal strength"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {'text': title},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 33], 'color': "lightgray"},
                {'range': [33, 66], 'color': "gray"},
                {'range': [66, 100], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 75
            }
        }
    ))
    
    fig.update_layout(height=250)
    return fig

def line_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str):
    """Create a simple line chart"""
    fig = px.line(df, x=x_col, y=y_col, title=title)
    fig.update_layout(
        hovermode='x unified',
        showlegend=False
    )
    return fig

def credit_spread_chart(df: pd.DataFrame, ema_df: pd.DataFrame = None):
    """Create credit spread chart with EMA"""
    fig = go.Figure()
    
    # Main spread line
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df[df.columns[1]],
        mode='lines',
        name='HYG OAS',
        line=dict(color='royalblue', width=2)
    ))
    
    # EMA line if provided
    if ema_df is not None and 'ema_330' in ema_df.columns:
        fig.add_trace(go.Scatter(
            x=ema_df['date'],
            y=ema_df['ema_330'],
            mode='lines',
            name='330-Day EMA',
            line=dict(color='orange', width=2, dash='dash')
        ))
    
    fig.update_layout(
        title='High Yield Credit Spreads (HYG OAS)',
        xaxis_title='Date',
        yaxis_title='Spread (%)',
        hovermode='x unified',
        height=400
    )
    
    return fig

def fear_greed_gauge(score: float):
    """Create Fear & Greed gauge"""
    # Determine color based on score
    if score < 25:
        color = "red"
        label = "EXTREME FEAR"
    elif score < 45:
        color = "orange"
        label = "FEAR"
    elif score < 55:
        color = "yellow"
        label = "NEUTRAL"
    elif score < 75:
        color = "lightgreen"
        label = "GREED"
    else:
        color = "green"
        label = "EXTREME GREED"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        title = {'text': f"Fear & Greed Index<br><span style='font-size:0.8em'>{label}</span>"},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 25], 'color': "rgba(255,0,0,0.3)"},
                {'range': [25, 45], 'color': "rgba(255,165,0,0.3)"},
                {'range': [45, 55], 'color': "rgba(255,255,0,0.3)"},
                {'range': [55, 75], 'color': "rgba(144,238,144,0.3)"},
                {'range': [75, 100], 'color': "rgba(0,128,0,0.3)"}
            ]
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def summary_table(data: dict):
    """Create summary table of key metrics"""
    df = pd.DataFrame([
        {"Indicator": k, "Value": v} 
        for k, v in data.items()
    ])
    
    st.dataframe(df, width='stretch', hide_index=True)