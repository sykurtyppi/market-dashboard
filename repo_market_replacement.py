def render_repo_market_replacement(page, st, components, logger, get_status_color, pd, go):
    """Legacy Repo Market page snippet retained as a callable helper."""
    if page != "Repo Market (SOFR)":
        return

    st.markdown(
        "<h1 class='main-header'>💰 Repo Market & Liquidity Stress</h1>",
        unsafe_allow_html=True,
    )
    
    try:
        repo_snapshot = components["repo"].get_full_snapshot()
        
        if not repo_snapshot or 'sofr' not in repo_snapshot:
            st.error("Repo market data unavailable. Please check data collection.")
            st.stop()
        
        repo_df = repo_snapshot.get('repo_df')
        
        if repo_df is not None and not repo_df.empty:
            repo_signal = components["repo_analyzer"].analyze(repo_df)
        else:
            repo_signal = None
            
    except Exception as e:
        st.error(f"Error loading Repo data: {e}")
        logger.error(f"Error in repo_stress page: {e}")
        st.stop()
    
    st.subheader("📊 Current Repo Market Status")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        sofr = repo_snapshot['sofr']
        color = "#1f77b4"
        
        st.markdown(
            f"<div style='text-align: center; padding: 1.5rem; background: {color}20; border-radius: 0.5rem;'>"
            f"<h4 style='margin:0;'>SOFR Rate</h4>"
            f"<h1 style='margin:0.5rem 0; color: {color};'>{sofr:.2f}%</h1>"
            f"<p style='margin:0; font-size:0.9em;'>Secured Overnight</p>"
            f"</div>",
            unsafe_allow_html=True,
        )
    
    with col2:
        if 'iorb' in repo_snapshot:
            iorb = repo_snapshot['iorb']
            color = "#9C27B0"
            
            st.markdown(
                f"<div style='text-align: center; padding: 1.5rem; background: {color}20; border-radius: 0.5rem;'>"
                f"<h4 style='margin:0;'>IORB Rate</h4>"
                f"<h1 style='margin:0.5rem 0; color: {color};'>{iorb:.2f}%</h1>"
                f"<p style='margin:0; font-size:0.9em;'>Fed Floor Rate</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.warning("IORB unavailable")
    
    with col3:
        if 'sofr_iorb_spread' in repo_snapshot:
            spread = repo_snapshot['sofr_iorb_spread']
            status = repo_snapshot.get('liquidity_status', 'UNKNOWN')
            color = repo_snapshot.get('liquidity_color', '#9E9E9E')
            
            st.markdown(
                f"<div style='text-align: center; padding: 1.5rem; background: {color}20; border-radius: 0.5rem;'>"
                f"<h4 style='margin:0;'>SOFR-IORB Spread</h4>"
                f"<h1 style='margin:0.5rem 0; color: {color};'>{spread:+.1f} bps</h1>"
                f"<p style='margin:0; font-size:0.9em;'>{status}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.warning("Spread unavailable")
    
    with col4:
        if repo_signal:
            stress_level = repo_signal.stress_level
            strength = repo_signal.strength
            color = get_status_color(stress_level)
            
            st.markdown(
                f"<div style='text-align: center; padding: 1.5rem; background: {color}20; border-radius: 0.5rem;'>"
                f"<h4 style='margin:0;'>Funding Stress</h4>"
                f"<h1 style='margin:0.5rem 0; color: {color};'>{stress_level}</h1>"
                f"<p style='margin:0; font-size:0.9em;'>Strength: {strength:.0f}/100</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.warning("Stress analysis unavailable")
    
    with col5:
        rrp_volume = repo_snapshot.get('rrp_volume', 0)
        
        if rrp_volume < 50:
            color = "#F44336"
            desc = "Depleted"
        elif rrp_volume < 500:
            color = "#FF9800"
            desc = "Low"
        else:
            color = "#4CAF50"
            desc = "Elevated"
        
        st.markdown(
            f"<div style='text-align: center; padding: 1.5rem; background: {color}20; border-radius: 0.5rem;'>"
            f"<h4 style='margin:0;'>RRP Volume</h4>"
            f"<h1 style='margin:0.5rem 0; color: {color};'>${rrp_volume:.0f}B</h1>"
            f"<p style='margin:0; font-size:0.9em;'>{desc}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )
    
    st.markdown("---")
    
    with st.expander("ℹ️ What is the Repo Market, SOFR & IORB?", expanded=False):
        st.markdown("""
        **The Repo Market** is where the financial system gets overnight funding:
        
        **SOFR (Secured Overnight Financing Rate):**
        - Benchmark rate for overnight Treasury repo transactions  
        - Replaced LIBOR as primary U.S. rate benchmark  
        - Reflects actual market funding conditions
        
        **IORB (Interest on Reserve Balances):**
        - Rate the Fed pays banks on reserves held at the Fed
        - Acts as a "floor" for overnight rates
        - Changed via Fed policy (FOMC decisions)
        
        **SOFR-IORB Spread = KEY LIQUIDITY INDICATOR:**
        - **Normal**: 5-15 bps positive spread
        - **Abundant**: SOFR ≤ IORB or 0-5 bps (plenty of cash)
        - **Tightening**: 15-30 bps (liquidity getting scarce)
        - **Stress**: >30 bps (funding pressure building)
        - **Crisis**: >50-100 bps (September 2019: 390 bps!)
        
        **Why it matters:**
        - Wide spread = Banks prefer lending in repo over holding reserves
        - Signals declining liquidity and potential funding stress
        - Fed monitors closely to gauge reserve adequacy
        """)
    
    tab1, tab2, tab3, tab4 = st.tabs([
        " SOFR vs IORB", 
        " SOFR-IORB Spread", 
        " Z-Score & Stress", 
        " RRP Volume"
    ])
    
    with tab1:
        st.subheader("SOFR vs IORB - Funding Rate Comparison")
        
        if repo_df is not None and not repo_df.empty and 'iorb' in repo_df.columns:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=repo_df['date'],
                y=repo_df['sofr'],
                name='SOFR (Market Rate)',
                line=dict(color='#1f77b4', width=2),
            ))
            
            fig.add_trace(go.Scatter(
                x=repo_df['date'],
                y=repo_df['iorb'],
                name='IORB (Fed Floor)',
                line=dict(color='#9C27B0', width=2, dash='dash'),
            ))
            
            fig.update_layout(
                title="SOFR vs IORB - Last 2 Years",
                xaxis_title="Date",
                yaxis_title="Rate (%)",
                hovermode='x unified',
                height=500,
                legend=dict(x=0.01, y=0.99)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current SOFR", f"{repo_df['sofr'].iloc[-1]:.2f}%")
            with col2:
                st.metric("Current IORB", f"{repo_df['iorb'].iloc[-1]:.2f}%")
            with col3:
                current_spread = repo_df['sofr_iorb_spread'].iloc[-1]
                st.metric("Spread", f"{current_spread:+.1f} bps")
            with col4:
                avg_spread = repo_df['sofr_iorb_spread'].tail(252).mean()
                st.metric("1Y Avg Spread", f"{avg_spread:+.1f} bps")
            
            if current_spread <= 5:
                st.success("✅ **ABUNDANT LIQUIDITY**: SOFR trading at or below IORB - funding conditions very easy")
            elif current_spread <= 15:
                st.info("➡️ **NORMAL CONDITIONS**: SOFR slightly above IORB - healthy funding market")
            elif current_spread <= 30:
                st.warning("⚠️ **TIGHTENING**: SOFR-IORB spread widening - liquidity starting to thin")
            else:
                st.error("🚨 **FUNDING STRESS**: Wide spread indicates significant liquidity pressure")
        else:
            st.warning("IORB data unavailable for comparison")
    
    with tab2:
        st.subheader("SOFR-IORB Spread History (Liquidity Indicator)")
        
        if repo_df is not None and not repo_df.empty and 'sofr_iorb_spread' in repo_df.columns:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=repo_df['date'],
                y=repo_df['sofr_iorb_spread'],
                name='SOFR-IORB Spread',
                line=dict(color='#2E86AB', width=2),
                fill='tozeroy',
                fillcolor='rgba(46, 134, 171, 0.3)'
            ))
            
            fig.add_hrect(y0=-5, y1=5, fillcolor="green", opacity=0.1,
                         annotation_text="ABUNDANT", annotation_position="left")
            fig.add_hrect(y0=5, y1=15, fillcolor="lightgreen", opacity=0.1,
                         annotation_text="NORMAL", annotation_position="left")
            fig.add_hrect(y0=15, y1=30, fillcolor="orange", opacity=0.1,
                         annotation_text="TIGHTENING", annotation_position="left")
            fig.add_hrect(y0=30, y1=100, fillcolor="red", opacity=0.1,
                         annotation_text="STRESS", annotation_position="left")
            
            fig.add_hline(y=0, line_dash="solid", line_color="gray", 
                         annotation_text="IORB Floor")
            
            fig.update_layout(
                title="SOFR-IORB Spread with Liquidity Bands",
                xaxis_title="Date",
                yaxis_title="Spread (basis points)",
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                current = repo_df['sofr_iorb_spread'].iloc[-1]
                st.metric("Current Spread", f"{current:+.1f} bps")
            with col2:
                max_spread = repo_df['sofr_iorb_spread'].max()
                st.metric("2Y Maximum", f"{max_spread:+.1f} bps")
            with col3:
                days_above_15 = (repo_df['sofr_iorb_spread'] > 15).sum()
                pct = (days_above_15 / len(repo_df)) * 100
                st.metric("Days >15 bps", f"{days_above_15} ({pct:.1f}%)")
            
            st.info("""
            **Interpretation Guide:**
            - **Negative or 0-5 bps**: Abundant liquidity, SOFR at or below floor
            - **5-15 bps**: Normal conditions, typical small premium
            - **15-30 bps**: Liquidity tightening, monitor for escalation
            - **>30 bps**: Funding stress, potential liquidity shortage
            - **>50 bps**: Severe stress (see September 2019: 390 bps spike!)
            """)
        else:
            st.warning("Spread data unavailable")
    
    with tab3:
        st.subheader("SOFR Z-Score & Stress Bands")
        
        if repo_df is not None and not repo_df.empty and 'sofr_z_score' in repo_df.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=repo_df['date'],
                y=repo_df['sofr_z_score'],
                name='SOFR Z-Score',
                line=dict(color='#1f77b4', width=2)
            ))
            fig.add_hrect(y0=-1, y1=1, fillcolor="green", opacity=0.1,
                         annotation_text="NORMAL", annotation_position="left")
            fig.add_hrect(y0=1, y1=2, fillcolor="orange", opacity=0.1,
                         annotation_text="ELEVATED", annotation_position="left")
            fig.add_hrect(y0=2, y1=5, fillcolor="red", opacity=0.1,
                         annotation_text="STRESS", annotation_position="left")
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(
                title="SOFR Z-Score with Stress Bands",
                xaxis_title="Date",
                yaxis_title="Standard Deviations (σ)",
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            if repo_signal and hasattr(repo_signal, 'description'):
                st.info(f"**Current Status:** {repo_signal.description}")
        else:
            st.warning("Z-score data unavailable")
    
    with tab4:
        st.subheader("Overnight RRP Volume")
        
        if repo_df is not None and not repo_df.empty and 'rrp_on' in repo_df.columns:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=repo_df['date'],
                y=repo_df['rrp_on'],
                name='RRP Volume',
                marker_color='#F44336'
            ))
            fig.update_layout(
                title="Overnight Reverse Repo (RRP) Volume - Last 2 Years",
                xaxis_title="Date",
                yaxis_title="Volume ($B)",
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            current_rrp = repo_df['rrp_on'].iloc[-1]
            peak_rrp = repo_df['rrp_on'].max()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Current RRP", f"${current_rrp:.0f}B")
            with col2:
                st.metric(
                    "Peak RRP",
                    f"${peak_rrp:.0f}B",
                    f"{((current_rrp - peak_rrp) / peak_rrp * 100):.1f}% vs peak"
                )
        else:
            st.warning("RRP volume data unavailable")
