"""
Chart Builder for PDF Reports
Creates optimized charts for PDF export
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime


def create_vrp_chart(vrp_history: pd.DataFrame) -> go.Figure:
    """Create VRP history chart optimized for PDF"""
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # VIX line
    fig.add_trace(
        go.Scatter(
            x=vrp_history['date'],
            y=vrp_history['vix'],
            name='VIX (Implied)',
            line=dict(color='#FF6B6B', width=3),
            hovertemplate='VIX: %{y:.2f}<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Realized Vol line
    fig.add_trace(
        go.Scatter(
            x=vrp_history['date'],
            y=vrp_history['realized_vol'],
            name='Realized Vol (21d)',
            line=dict(color='#4ECDC4', width=3, dash='dash'),
            hovertemplate='RVol: %{y:.2f}<extra></extra>'
        ),
        secondary_y=False
    )
    
    # VRP area
    fig.add_trace(
        go.Scatter(
            x=vrp_history['date'],
            y=vrp_history['vrp'],
            name='VRP Spread',
            line=dict(color='#95E1D3', width=2),
            fill='tozeroy',
            fillcolor='rgba(149, 225, 211, 0.3)',
            hovertemplate='VRP: %{y:.2f}<extra></extra>'
        ),
        secondary_y=True
    )
    
    # Calculate and add Relative VRP (normalized by realized vol)
    vrp_history['relative_vrp'] = (vrp_history['vrp'] / vrp_history['realized_vol']) * 100
    vrp_history['relative_vrp'] = vrp_history['relative_vrp'].fillna(0)
    
    # Add Relative VRP line (shows if VRP is meaningful given vol regime)
    fig.add_trace(
        go.Scatter(
            x=vrp_history['date'],
            y=vrp_history['relative_vrp'],
            name='Relative VRP (%)',
            line=dict(color='#9C27B0', width=2, dash='dot'),
            yaxis='y3',
            hovertemplate='Rel. VRP: %{y:.1f}%<extra></extra>'
        )
    )
    
    # Styling for PDF with third y-axis for Relative VRP
    fig.update_layout(
        title={
            'text': 'Volatility Risk Premium (VRP) Analysis',
            'font': {'size': 18, 'color': '#1f77b4', 'family': 'Arial'}
        },
        xaxis_title="Date",
        yaxis_title="Volatility (%)",
        yaxis2_title="VRP (pts)",
        yaxis3=dict(
            title=dict(text='Relative VRP (%)', font=dict(color='#9C27B0', size=10)),
            overlaying='y',
            side='right',
            position=0.95,
            tickfont=dict(color='#9C27B0', size=10),
            showgrid=False
        ),
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        hovermode='x unified'
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0', secondary_y=False)
    fig.update_yaxes(showgrid=False, secondary_y=True)
    
    return fig


def create_credit_spreads_chart(hy_history: pd.DataFrame, ig_history: pd.DataFrame) -> go.Figure:
    """Create credit spreads chart"""
    
    fig = go.Figure()
    
    # High Yield line
    if not hy_history.empty:
        fig.add_trace(
            go.Scatter(
                x=hy_history['date'],
                y=hy_history['value'] * 100,
                name='High Yield (HYG)',
                line=dict(color='#FF6B6B', width=3),
                hovertemplate='HYG: %{y:.0f} bps<extra></extra>'
            )
        )
    
    # Investment Grade line
    if not ig_history.empty:
        fig.add_trace(
            go.Scatter(
                x=ig_history['date'],
                y=ig_history['value'] * 100,
                name='Investment Grade (LQD)',
                line=dict(color='#4ECDC4', width=3),
                hovertemplate='LQD: %{y:.0f} bps<extra></extra>'
            )
        )
    
    # Styling
    fig.update_layout(
        title={
            'text': 'Credit Spreads - HYG vs LQD',
            'font': {'size': 18, 'color': '#1f77b4', 'family': 'Arial'}
        },
        xaxis_title="Date",
        yaxis_title="Spread (basis points)",
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        hovermode='x unified'
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    
    return fig


def create_liquidity_chart(liquidity_df: pd.DataFrame) -> go.Figure:
    """Create net liquidity chart"""
    
    fig = go.Figure()
    
    # Calculate net liquidity
    if 'rrp_on' in liquidity_df.columns and 'tga' in liquidity_df.columns:
        liquidity_df = liquidity_df.copy()
        liquidity_df['net_liq'] = -(liquidity_df['rrp_on'] + liquidity_df['tga'])
        
        fig.add_trace(
            go.Scatter(
                x=liquidity_df['date'],
                y=liquidity_df['net_liq'],
                name='Net Liquidity',
                line=dict(color='#00D9FF', width=3),
                fill='tozeroy',
                fillcolor='rgba(0, 217, 255, 0.2)',
                hovertemplate='Net Liq: $%{y:,.0f}B<extra></extra>'
            )
        )
        
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray",
            opacity=0.5
        )
    
    # Styling
    fig.update_layout(
        title={
            'text': 'Net Liquidity (-(RRP + TGA))',
            'font': {'size': 18, 'color': '#1f77b4', 'family': 'Arial'}
        },
        xaxis_title="Date",
        yaxis_title="Net Liquidity (Billions USD)",
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        hovermode='x unified'
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    
    return fig


def create_breadth_gauge(breadth_pct: float) -> go.Figure:
    """Create market breadth gauge chart"""
    
    # Determine color
    if breadth_pct > 60:
        color = "#4CAF50"
        status = "Strong"
    elif breadth_pct > 40:
        color = "#FF9800"
        status = "Neutral"
    else:
        color = "#F44336"
        status = "Weak"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=breadth_pct,
        title={'text': f"Market Breadth<br><span style='font-size:14px'>{status}</span>", 'font': {'size': 18}},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': 'rgba(244, 67, 54, 0.2)'},
                {'range': [40, 60], 'color': 'rgba(255, 152, 0, 0.2)'},
                {'range': [60, 100], 'color': 'rgba(76, 175, 80, 0.2)'},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='white',
        font={'family': 'Arial'}
    )
    
    return fig


def create_fear_greed_gauge(score: float) -> go.Figure:
    """Create Fear & Greed gauge"""
    
    # Determine color and label
    if score < 25:
        color = "#F44336"
        label = "EXTREME FEAR"
    elif score < 45:
        color = "#FF9800"
        label = "FEAR"
    elif score < 55:
        color = "#FFC107"
        label = "NEUTRAL"
    elif score < 75:
        color = "#8BC34A"
        label = "GREED"
    else:
        color = "#4CAF50"
        label = "EXTREME GREED"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': f"Fear & Greed Index<br><span style='font-size:14px'>{label}</span>", 'font': {'size': 18}},
        number={'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 25], 'color': 'rgba(244, 67, 54, 0.2)'},
                {'range': [25, 45], 'color': 'rgba(255, 152, 0, 0.2)'},
                {'range': [45, 55], 'color': 'rgba(255, 193, 7, 0.2)'},
                {'range': [55, 75], 'color': 'rgba(139, 195, 74, 0.2)'},
                {'range': [75, 100], 'color': 'rgba(76, 175, 80, 0.2)'},
            ]
        }
    ))
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='white',
        font={'family': 'Arial'}
    )
    
    return fig

def create_adline_chart(breadth_history):
    """Create Advance-Decline Line chart for PDF"""
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # A/D Line
    fig.add_trace(
        go.Scatter(
            x=breadth_history['date'],
            y=breadth_history['ad_line'],
            name='A/D Line',
            line=dict(color='#1f77b4', width=3),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.2)',
            hovertemplate='A/D Line: %{y:,.0f}<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Breadth percentage
    fig.add_trace(
        go.Scatter(
            x=breadth_history['date'],
            y=breadth_history['breadth_pct'],
            name='Breadth %',
            line=dict(color='#FF6B6B', width=2, dash='dash'),
            hovertemplate='Breadth: %{y:.1f}%<extra></extra>'
        ),
        secondary_y=True
    )
    
    # Add 50% reference line
    fig.add_hline(y=50, line_dash="dot", line_color="gray", 
                  opacity=0.5, secondary_y=True)
    
    # Styling
    fig.update_layout(
        title={
            'text': 'Advance-Decline Line & Breadth',
            'font': {'size': 18, 'color': '#1f77b4', 'family': 'Arial'}
        },
        xaxis_title="Date",
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        hovermode='x unified'
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    fig.update_yaxes(
        title_text="A/D Line", 
        secondary_y=False,
        showgrid=True, 
        gridwidth=1, 
        gridcolor='#f0f0f0'
    )
    fig.update_yaxes(
        title_text="Breadth %", 
        secondary_y=True,
        showgrid=False
    )
    
    return fig


def create_mcclellan_chart(breadth_history):
    """Create McClellan Oscillator chart for PDF"""
    
    if 'mcclellan' not in breadth_history.columns:
        # Calculate it
        breadth_history['ema19'] = breadth_history['ad_diff'].ewm(span=19, adjust=False).mean()
        breadth_history['ema39'] = breadth_history['ad_diff'].ewm(span=39, adjust=False).mean()
        breadth_history['mcclellan'] = breadth_history['ema19'] - breadth_history['ema39']
    
    fig = go.Figure()
    
    # McClellan line
    fig.add_trace(
        go.Scatter(
            x=breadth_history['date'],
            y=breadth_history['mcclellan'],
            name='McClellan Oscillator',
            line=dict(color='#4ECDC4', width=3),
            fill='tozeroy',
            fillcolor='rgba(78, 205, 196, 0.2)',
            hovertemplate='McClellan: %{y:.1f}<extra></extra>'
        )
    )
    
    # Add reference lines
    fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5)
    fig.add_hline(y=50, line_dash="dash", line_color="green", opacity=0.3)
    fig.add_hline(y=-50, line_dash="dash", line_color="red", opacity=0.3)
    
    # Styling
    fig.update_layout(
        title={
            'text': 'McClellan Oscillator',
            'font': {'size': 18, 'color': '#1f77b4', 'family': 'Arial'}
        },
        xaxis_title="Date",
        yaxis_title="Oscillator Value",
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        hovermode='x unified'
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    
    return fig


def create_vix_term_structure_chart(vix_data: dict) -> go.Figure:
    """
    Create VIX term structure chart showing the volatility curve
    Now shows available data with helpful context instead of "unavailable"
    
    Args:
        vix_data: Dict with keys: vix, vix9d, vix3m, vix6m
    """
    import plotly.graph_objects as go
    
    # VIX term structure points (days to expiration)
    all_terms = {
        'VIX9D': {'days': 9, 'value': vix_data.get('vix9d'), 'label': 'VIX9D (9d)'},
        'VIX': {'days': 30, 'value': vix_data.get('vix'), 'label': 'VIX (30d)'},
        'VIX3M': {'days': 93, 'value': vix_data.get('vix3m'), 'label': 'VIX3M (93d)'},
        'VIX6M': {'days': 186, 'value': vix_data.get('vix6m'), 'label': 'VIX6M (186d)'}
    }
    
    # Filter out None values
    valid_terms = [(k, v) for k, v in all_terms.items() if v['value'] is not None]
    
    # If we have less than 2 points, can't draw a curve
    if len(valid_terms) < 2:
        return None  # PDF generator will skip this chart
    
    # Sort by days
    sorted_terms = sorted(valid_terms, key=lambda x: x[1]['days'])
    
    labels = [v['label'] for k, v in sorted_terms]
    days = [v['days'] for k, v in sorted_terms]
    values = [v['value'] for k, v in sorted_terms]
    
    # Determine regime and calculate slope
    slope_absolute = values[-1] - values[0]
    slope_pct = (slope_absolute / values[0]) * 100 if values[0] > 0 else 0
    
    # Calculate M1→M2 slope (annualized steepness)
    days_diff = days[-1] - days[0]
    slope_per_day = slope_absolute / days_diff if days_diff > 0 else 0
    slope_annualized = slope_per_day * 252  # Trading days per year
    
    if slope_absolute > 0:
        curve_color = '#4CAF50'  # Green = Contango (normal)
        regime = 'Contango'
        if slope_pct > 20:
            regime_desc = 'Steep curve - Strong risk-on'
        elif slope_pct > 10:
            regime_desc = 'Normal upward slope'
        else:
            regime_desc = 'Flat curve - Watch for reversal'
    else:
        curve_color = '#F44336'  # Red = Backwardation (stress)
        regime = 'Backwardation'
        regime_desc = 'INVERTED - High stress/crisis mode'
    
    contango_pct = slope_pct
    
    fig = go.Figure()
    
    # Add the term structure line
    # Create smooth curve with realistic shape
    import numpy as np
    
    # For only 2 points, add a middle calculated point to create natural curve
    if len(days) == 2:
        # Add middle point with slight convexity (VIX curves typically have this shape)
        mid_day = (days[0] + days[1]) / 2
        mid_value = (values[0] + values[1]) / 2
        
        # Add slight upward bias to middle (term structures usually have convexity)
        convexity_adjustment = (values[1] - values[0]) * 0.15  # 15% of slope
        mid_value += convexity_adjustment
        
        # Insert middle point
        days = [days[0], mid_day, days[1]]
        values = [values[0], mid_value, values[1]]
    
    # Now interpolate with spline for smooth curve
    days_smooth = np.linspace(min(days), max(days), 100)
    
    if len(days) >= 3:
        from scipy import interpolate
        f = interpolate.interp1d(days, values, kind='quadratic', fill_value='extrapolate')
        values_smooth = f(days_smooth)
    else:
        days_smooth = np.array(days)
        values_smooth = np.array(values)
    
    # Add smooth curve
    fig.add_trace(go.Scatter(
        x=days_smooth,
        y=values_smooth,
        mode='lines',
        name='VIX Term Structure',
        line=dict(color=curve_color, width=3, shape='spline'),
        hoverinfo='skip',
        showlegend=False
    ))
    
    # Add actual data points on top
    fig.add_trace(go.Scatter(
        x=days,
        y=values,
        mode='markers',
        name='VIX Points',
        marker=dict(size=12, color=curve_color, symbol='circle',
                   line=dict(width=2, color='white')),
        hovertemplate='%{text}<br>%{y:.2f}%<extra></extra>',
        text=labels,
        showlegend=False
    ))
    
    # Add value labels at each point
    for label, day, value in zip(labels, days, values):
        fig.add_annotation(
            x=day,
            y=value,
            text=f"<b>{value:.1f}</b>",
            showarrow=False,
            yshift=18,
            font=dict(size=12, color=curve_color, family='Arial Black')
        )
    
    # Add regime indicator box with slope
    fig.add_annotation(
        text=f"<b>{regime}</b> ({contango_pct:+.1f}%)<br>{regime_desc}<br><i>Slope: {slope_per_day:+.3f}/day</i>",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        bgcolor='rgba(76, 175, 80, 0.2)' if regime == 'Contango' else 'rgba(244, 67, 54, 0.2)',
        bordercolor=curve_color,
        borderwidth=2,
        borderpad=8,
        font=dict(size=11, color=curve_color),
        align='left',
        xanchor='left',
        yanchor='top'
    )
    
    # Add note about data availability if we're missing some points
    missing_points = 4 - len(valid_terms)
    if missing_points > 0:
        available_str = ', '.join([k for k, v in sorted_terms])
        fig.add_annotation(
            text=f"<i>Showing available data: {available_str}</i>",
            xref="paper", yref="paper",
            x=0.5, y=-0.12,
            showarrow=False,
            font=dict(size=9, color='gray'),
            xanchor='center'
        )
    
    fig.update_layout(
        title="<b>VIX Term Structure</b><br><sub>Implied volatility across time horizons</sub>",
        xaxis_title="Days to Expiration",
        yaxis_title="Implied Volatility (%)",
        height=450,
        template='plotly_white',
        plot_bgcolor='#F8F9FA',
        hovermode='closest',
        margin=dict(b=80),
        xaxis=dict(
            gridcolor='#E0E0E0',
            showgrid=True
        ),
        yaxis=dict(
            gridcolor='#E0E0E0',
            showgrid=True
        )
    )
    
    return fig


def create_treasury_stress_chart(move_history: pd.DataFrame) -> go.Figure:
    """
    Create MOVE Index (Treasury volatility) chart
    
    Args:
        move_history: DataFrame with columns: date, value
    """
    import plotly.graph_objects as go
    
    if move_history.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="MOVE Index data unavailable",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='gray')
        )
        fig.update_layout(title="Treasury Stress (MOVE Index)", height=400)
        return fig
    
    fig = go.Figure()
    
    # MOVE Index line
    fig.add_trace(go.Scatter(
        x=move_history['date'],
        y=move_history['value'],
        mode='lines',
        name='MOVE Index',
        line=dict(color='#FF6B6B', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(255, 107, 107, 0.2)',
        hovertemplate='%{y:.1f}<extra></extra>'
    ))
    
    # Stress levels
    fig.add_hline(y=100, line_dash="dash", line_color="#FF9800", 
                  annotation_text="Elevated Stress (100)", annotation_position="right")
    fig.add_hline(y=150, line_dash="dash", line_color="#F44336",
                  annotation_text="High Stress (150)", annotation_position="right")
    
    current_move = move_history['value'].iloc[-1]
    
    fig.update_layout(
        title={
            'text': f"Treasury Market Stress - MOVE Index (Current: {current_move:.1f})",
            'font': {'size': 18, 'family': 'Arial Black'}
        },
        xaxis_title="Date",
        yaxis_title="MOVE Index",
        height=400,
        showlegend=False,
        hovermode='x unified',
        paper_bgcolor='white',
        plot_bgcolor='#F8F9FA',
        xaxis=dict(gridcolor='#E0E0E0', showgrid=True),
        yaxis=dict(gridcolor='#E0E0E0', showgrid=True),
        margin=dict(l=60, r=40, t=60, b=60)
    )
    
    return fig


def create_sector_rotation_chart(sector_performance: dict, period: str = '3M') -> go.Figure:
    """
    Create sector rotation chart showing relative performance
    
    Args:
        sector_performance: Dict with sector names as keys and returns as values
        period: '1M' or '3M' for display
    """
    import plotly.graph_objects as go
    
    if not sector_performance:
        fig = go.Figure()
        fig.add_annotation(
            text="Sector performance data unavailable",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='gray')
        )
        fig.update_layout(title=f"Sector Rotation ({period})", height=400)
        return fig
    
    # Sort sectors by performance
    sorted_sectors = sorted(sector_performance.items(), key=lambda x: x[1], reverse=True)
    sectors = [s[0] for s in sorted_sectors]
    returns = [s[1] for s in sorted_sectors]
    
    # Color based on positive/negative
    colors = ['#4CAF50' if r > 0 else '#F44336' for r in returns]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=sectors,
        x=returns,
        orientation='h',
        marker=dict(color=colors),
        text=[f"{r:+.1f}%" for r in returns],
        textposition='outside',
        hovertemplate='%{y}: %{x:+.1f}%<extra></extra>'
    ))
    
    # Add zero line
    fig.add_vline(x=0, line_color='gray', line_width=1)
    
    fig.update_layout(
        title={
            'text': f"Sector Performance - {period} Returns",
            'font': {'size': 18, 'family': 'Arial Black'}
        },
        xaxis_title="Return (%)",
        yaxis_title="",
        height=450,
        showlegend=False,
        paper_bgcolor='white',
        plot_bgcolor='#F8F9FA',
        xaxis=dict(gridcolor='#E0E0E0', showgrid=True, zeroline=False),
        yaxis=dict(gridcolor='#E0E0E0', showgrid=False),
        margin=dict(l=150, r=40, t=60, b=60)
    )
    
    return fig


def create_skew_history_chart(skew_df):
    """Create SKEW history chart with regime bands"""
    if skew_df.empty:
        return None
    
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    # Main SKEW line
    fig.add_trace(go.Scatter(
        x=skew_df['date'],
        y=skew_df['skew'],
        name='SKEW',
        line=dict(color='#9C27B0', width=2.5),
        hovertemplate='%{y:.1f}<extra></extra>'
    ))
    
    # Regime bands
    fig.add_hrect(y0=100, y1=130, fillcolor='rgba(76, 175, 80, 0.1)', 
                  line_width=0, annotation_text="Complacency", annotation_position="left")
    fig.add_hrect(y0=130, y1=145, fillcolor='rgba(255, 193, 7, 0.1)', 
                  line_width=0, annotation_text="Normal", annotation_position="left")
    fig.add_hrect(y0=145, y1=160, fillcolor='rgba(255, 152, 0, 0.1)', 
                  line_width=0, annotation_text="Elevated", annotation_position="left")
    fig.add_hrect(y0=160, y1=200, fillcolor='rgba(244, 67, 54, 0.1)', 
                  line_width=0, annotation_text="Extreme", annotation_position="left")
    
    # Current value annotation
    latest = skew_df.iloc[-1]
    fig.add_annotation(
        x=latest['date'],
        y=latest['skew'],
        text=f"<b>{latest['skew']:.1f}</b>",
        showarrow=True,
        arrowhead=2,
        arrowcolor='#9C27B0',
        bgcolor='white',
        bordercolor='#9C27B0',
        font=dict(color="black")      # 👈 add this line (or use '#9C27B0')
    )
    
    fig.update_layout(
        title='<b>CBOE SKEW Index</b><br><sub>Tail Risk Premium (OTM Put Skew)</sub>',
        xaxis_title='Date',
        yaxis_title='SKEW Level',
        template='plotly_white',
        hovermode='x unified',
        height=400,
        showlegend=False
    )
    
    return fig



def create_vix9d_spread_chart(vix9d_df, vix_df):
    """Create VIX9D vs VIX30D spread chart"""
    if vix9d_df.empty or vix_df.empty:
        return None
    
    import plotly.graph_objects as go
    
    # Merge dataframes
    merged = pd.merge(vix9d_df, vix_df, on='date', how='inner')
    merged['spread'] = merged['vix9d'] - merged['vix']
    merged['spread_pct'] = (merged['spread'] / merged['vix']) * 100
    
    fig = go.Figure()
    
    # Spread line
    fig.add_trace(go.Scatter(
        x=merged['date'],
        y=merged['spread_pct'],
        name='9D-30D Spread',
        line=dict(color='#2196F3', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(33, 150, 243, 0.2)',
        hovertemplate='%{y:+.1f}%<extra></extra>'
    ))
    
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Regime zones
    fig.add_hrect(y0=-50, y1=-10, fillcolor='rgba(76, 175, 80, 0.1)', 
                  line_width=0, annotation_text="Calm Near-Term", annotation_position="left")
    fig.add_hrect(y0=10, y1=50, fillcolor='rgba(244, 67, 54, 0.1)', 
                  line_width=0, annotation_text="Event Risk", annotation_position="left")
    
    # Current value annotation
    latest = merged.iloc[-1]
    fig.add_annotation(
        x=latest['date'],
        y=latest['spread_pct'],
        text=f"<b>{latest['spread_pct']:+.1f}%</b>",
        showarrow=True,
        arrowhead=2,
        arrowcolor='#2196F3',
        bgcolor='white',
        bordercolor='#2196F3',
        font=dict(color="black")   # 👈 ensures number is visible on white box
    )
    
    fig.update_layout(
        title='<b>VIX9D vs VIX (30d) Spread</b><br><sub>Near-Term vs 1-Month Volatility Expectation</sub>',
        xaxis_title='Date',
        yaxis_title='Spread (%)',
        template='plotly_white',
        hovermode='x unified',
        height=400,
        showlegend=False
    )
    
    return fig
