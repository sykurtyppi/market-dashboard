"""Settings page."""
import os

import streamlit as st

from dashboard.core.settings_utils import render_api_settings_lock, save_to_env


def render(components):
    st.markdown("<h1 class='main-header'> Settings</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    Configure API keys and dashboard preferences. Changes are saved to `.env` file.
    """)
    
    # API Keys Section
    st.subheader(" API Keys")

    api_settings_unlocked = render_api_settings_lock()

    if api_settings_unlocked:
        # FRED API Key
        with st.expander("FRED API Key (Required)", expanded=False):
            st.markdown("Get your free API key at: https://fred.stlouisfed.org/docs/api/api_key.html")

            current_fred_key = os.getenv('FRED_API_KEY', '')
            fred_key = st.text_input(
                "FRED API Key",
                value=current_fred_key,
                type="password",
                key="fred_api_key"
            )

            if st.button("Save FRED API Key"):
                if fred_key:
                    save_to_env('FRED_API_KEY', fred_key)
                    st.success("✓ FRED API key saved! Restart dashboard to apply.")
                else:
                    st.error("Please enter an API key")

        # Alpha Vantage API Key
        with st.expander("Alpha Vantage API Key (Optional)", expanded=False):
            st.markdown("Get your free API key at: https://www.alphavantage.co/support/#api-key")
            st.info("Used for: Alternative market data, economic indicators")

            current_alpha_key = os.getenv('ALPHA_VANTAGE_API_KEY', '')
            alpha_key = st.text_input(
                "Alpha Vantage API Key",
                value=current_alpha_key,
                type="password",
                key="alpha_api_key"
            )

            if st.button("Save Alpha Vantage API Key"):
                if alpha_key:
                    save_to_env('ALPHA_VANTAGE_API_KEY', alpha_key)
                    st.success("✓ Alpha Vantage API key saved! Restart dashboard to apply.")
                else:
                    st.error("Please enter an API key")

        # Polygon API Key
        with st.expander("Polygon.io API Key (Optional)", expanded=False):
            st.markdown("Get your API key at: https://polygon.io/")
            st.info("Used for: Real-time market data, options flow")

            current_polygon_key = os.getenv('POLYGON_API_KEY', '')
            polygon_key = st.text_input(
                "Polygon API Key",
                value=current_polygon_key,
                type="password",
                key="polygon_api_key"
            )

            if st.button("Save Polygon API Key"):
                if polygon_key:
                    save_to_env('POLYGON_API_KEY', polygon_key)
                    st.success("✓ Polygon API key saved! Restart dashboard to apply.")
                else:
                    st.error("Please enter an API key")

        # Nasdaq Data Link API Key (for COT data)
        with st.expander("Nasdaq Data Link API Key (Recommended)", expanded=False):
            st.markdown("Get your **free** API key at: [data.nasdaq.com](https://data.nasdaq.com)")
            st.info("🎯 **Used for:** CFTC Commitments of Traders (COT) positioning data. Without this key, COT data uses slow CFTC file downloads.")

            current_nasdaq_key = os.getenv('NASDAQ_DATA_LINK_KEY', '')
            nasdaq_key = st.text_input(
                "Nasdaq Data Link API Key",
                value=current_nasdaq_key,
                type="password",
                key="nasdaq_api_key",
                help="Sign up at data.nasdaq.com (free), then go to Account Settings to get your API key"
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Save Nasdaq API Key"):
                    if nasdaq_key:
                        save_to_env('NASDAQ_DATA_LINK_KEY', nasdaq_key)
                        st.success("✓ Nasdaq Data Link API key saved! Restart dashboard to apply.")
                    else:
                        st.error("Please enter an API key")

            with col2:
                if st.button("Test Nasdaq API Key"):
                    if nasdaq_key:
                        with st.spinner("Testing API key..."):
                            try:
                                import requests
                                # Test with a simple dataset request
                                url = f"https://data.nasdaq.com/api/v3/datasets/CFTC/13874A_F_L_ALL.json?api_key={nasdaq_key}&rows=1"
                                response = requests.get(url, timeout=10)
                                if response.status_code == 200:
                                    st.success("✅ API key is valid! COT data will use fast Nasdaq API.")
                                elif response.status_code == 400:
                                    st.error("❌ Invalid API key. Please check your key.")
                                elif response.status_code == 429:
                                    st.warning("⚠️ Rate limited. Key is valid but too many requests.")
                                else:
                                    st.warning(f"⚠️ Unexpected response: {response.status_code}")
                            except Exception as e:
                                st.error(f"❌ Test failed: {str(e)}")
                    else:
                        st.warning("Enter an API key first")
    else:
        st.info("API key controls are hidden until you unlock this section.")

    st.divider()
    
    # Manual Overrides Section
    st.subheader(" Manual Data Overrides")
    
    with st.expander("Equity Put/Call Ratio Override", expanded=False):
        st.markdown("""
        Manually set the equity put/call ratio when automatic data is unavailable or incorrect.
        """)
        
        use_manual = st.checkbox("Use Manual Equity P/C Ratio", key="use_manual_pc")
        
        if use_manual:
            manual_pc = st.number_input(
                "Manual Equity P/C Ratio",
                min_value=0.0,
                max_value=5.0,
                value=1.0,
                step=0.01,
                help="Typical range: 0.5-2.0. Above 1.0 = bearish, below 1.0 = bullish"
            )
            
            if st.button("Save Manual P/C Ratio"):
                st.session_state['manual_equity_pc'] = manual_pc
                st.session_state['use_manual_equity_pc'] = True
                st.success(f"✓ Manual equity P/C ratio set to {manual_pc:.2f}")
        else:
            if 'use_manual_equity_pc' in st.session_state:
                st.session_state['use_manual_equity_pc'] = False
            st.info("Using automatic equity P/C ratio from CBOE")

    st.divider()

    # Market Breadth Settings
    st.subheader("📊 Market Breadth Settings")

    st.markdown("""
    **Stock Sample Size for Breadth Calculation**

    Choose between speed and precision for market breadth analysis:
    """)

    # Get current setting from env
    current_breadth_mode = os.getenv('BREADTH_MODE', 'fast').lower()

    breadth_mode = st.radio(
        "Breadth Calculation Mode",
        options=['fast', 'full'],
        index=0 if current_breadth_mode == 'fast' else 1,
        format_func=lambda x: {
            'fast': '⚡ Fast Mode (100 stocks) - Recommended',
            'full': '🎯 Full Mode (500 stocks) - More precise'
        }[x],
        key="breadth_mode_setting",
        help="Fast mode uses 100 representative stocks (~30-60 sec). Full mode uses all 500 S&P stocks (~3-5 min)."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **⚡ Fast Mode (100 stocks)**
        - ~30-60 seconds refresh
        - Representative sample
        - McClellan scaled to NYSE standard
        """)
    with col2:
        st.markdown("""
        **🎯 Full Mode (500 stocks)**
        - ~3-5 minutes refresh
        - True S&P 500 breadth
        - No scaling needed
        """)

    if st.button("Save Breadth Mode", key="save_breadth_mode_btn"):
        save_to_env('BREADTH_MODE', breadth_mode)
        st.success(f"✅ Breadth mode saved: {breadth_mode.upper()}")
        st.info("🔄 Go to Market Breadth page and click 'Refresh Breadth Data' to apply.")

    # Show current status
    mode_icon = "⚡" if current_breadth_mode == 'fast' else "🎯"
    stock_count = "100" if current_breadth_mode == 'fast' else "500"
    st.caption(f"Current mode: {mode_icon} {current_breadth_mode.upper()} ({stock_count} stocks)")

    st.divider()

    # Dashboard Preferences
    st.subheader(" Dashboard Preferences")
    
    st.markdown("""
    **Streamlit Built-in Settings** (accessible via ☰ menu in top right):
    - Theme (Light/Dark mode)
    - Wide mode toggle
    - Run on save
    - Developer options
    """)
    
    st.divider()
    
    # About Section
    st.subheader("About")
    
    st.markdown("""
    **Market Dashboard** - Institutional-grade market analysis
    
    **Core Features:**
    -  VIX & Volatility Risk Premium tracking
    -  Fed liquidity monitoring (RRP, TGA, Balance Sheet)
    -  CTA systematic flow tracking
    -  Credit spreads & repo stress
    -  Market breadth & sentiment
    
    **Data Sources:**
    - Federal Reserve Economic Data (FRED)
    - CBOE (VIX, SKEW, Options)
    - Yahoo Finance (Market data)
    - CNN Fear & Greed Index
    - Alpha Vantage (Optional)
    - Polygon.io (Optional)
    
    **Version:** 2.0 | **Last Updated:** December 2024
    """)
