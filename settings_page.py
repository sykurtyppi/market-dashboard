"""
Settings Page - API Key Management
Allows users to configure API keys through Streamlit GUI
"""

import streamlit as st
import os
import logging
from pathlib import Path
from dotenv import load_dotenv, set_key, find_dotenv
import requests
from datetime import datetime


class SettingsManager:
    """Manage API keys and settings through Streamlit UI"""
    
    def __init__(self):
        self.env_file = find_dotenv() or '.env'
        load_dotenv(self.env_file)
        
        # Create .env if it doesn't exist
        if not Path(self.env_file).exists():
            Path(self.env_file).touch()
    
    def get_api_key(self, key_name: str) -> str:
        """Get API key from environment"""
        return os.getenv(key_name, '')
    
    def save_api_key(self, key_name: str, value: str) -> bool:
        """Save API key to .env file"""
        try:
            set_key(self.env_file, key_name, value)
            os.environ[key_name] = value  # Update current session
            return True
        except Exception as e:
            st.error(f"Error saving {key_name}: {e}")
            return False
    
    def test_fred_api(self, api_key: str) -> bool:
        """Test FRED API key"""
        try:
            url = "https://api.stlouisfed.org/fred/series"
            params = {
                'series_id': 'BAMLH0A0HYM2',
                'api_key': api_key,
                'file_type': 'json'
            }
            response = requests.get(url, params=params, timeout=5)
            return response.status_code == 200
        except (requests.RequestException, ValueError, ConnectionError) as e:
            logging.warning(f"FRED API test failed: {e}")
            return False

    def test_alpha_vantage_api(self, api_key: str) -> bool:
        """Test Alpha Vantage API key"""
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_INTRADAY',
                'symbol': 'SPY',
                'interval': '5min',
                'apikey': api_key
            }
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            return 'Time Series (5min)' in data or 'Meta Data' in data
        except (requests.RequestException, ValueError, ConnectionError) as e:
            logging.warning(f"Alpha Vantage API test failed: {e}")
            return False

    def test_polygon_api(self, api_key: str) -> bool:
        """Test Polygon.io API key"""
        try:
            url = f"https://api.polygon.io/v2/aggs/ticker/SPY/prev"
            params = {'apiKey': api_key}
            response = requests.get(url, params=params, timeout=5)
            return response.status_code == 200
        except (requests.RequestException, ValueError, ConnectionError) as e:
            logging.warning(f"Polygon API test failed: {e}")
            return False


def render_settings_page():
    """Render the settings page in Streamlit"""
    
    st.title("⚙️ Dashboard Settings")
    st.markdown("Configure your API keys and data sources here.")
    
    settings = SettingsManager()
    
    st.markdown("---")
    
    # FRED API Section
    st.header(" FRED API (Federal Reserve Economic Data)")
    st.markdown("""
    **Get your free API key:** [https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)
    
    **Used for:**
    - Credit spreads (HY, IG)
    - Treasury yields
    - Economic indicators
    """)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        fred_key = st.text_input(
            "FRED API Key",
            value=settings.get_api_key('FRED_API_KEY'),
            type="password",
            key="fred_input"
        )
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        if st.button("Test", key="test_fred"):
            if fred_key:
                with st.spinner("Testing..."):
                    if settings.test_fred_api(fred_key):
                        st.success("✅ Valid!")
                    else:
                        st.error("❌ Invalid")
            else:
                st.warning("Enter key first")
    
    if st.button("Save FRED Key", key="save_fred"):
        if fred_key:
            if settings.save_api_key('FRED_API_KEY', fred_key):
                st.success("✅ FRED API key saved!")
        else:
            st.warning("Please enter an API key")
    
    st.markdown("---")
    
    # Alpha Vantage Section
    st.header(" Alpha Vantage API")
    st.markdown("""
    **Get your free API key:** [https://www.alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key)
    
    **Used for:**
    - Real-time stock quotes
    - Market breadth data
    - Backup for Yahoo Finance
    
    **Free tier:** 500 API calls/day
    """)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        av_key = st.text_input(
            "Alpha Vantage API Key",
            value=settings.get_api_key('ALPHA_VANTAGE_API_KEY'),
            type="password",
            key="av_input"
        )
    with col2:
        st.write("")
        st.write("")
        if st.button("Test", key="test_av"):
            if av_key:
                with st.spinner("Testing..."):
                    if settings.test_alpha_vantage_api(av_key):
                        st.success("✅ Valid!")
                    else:
                        st.error("❌ Invalid")
            else:
                st.warning("Enter key first")
    
    if st.button("Save Alpha Vantage Key", key="save_av"):
        if av_key:
            if settings.save_api_key('ALPHA_VANTAGE_API_KEY', av_key):
                st.success("✅ Alpha Vantage API key saved!")
        else:
            st.warning("Please enter an API key")
    
    st.markdown("---")
    
    # Polygon.io Section
    st.header(" Polygon.io API (Optional)")
    st.markdown("""
    **Get your free API key:** [https://polygon.io/](https://polygon.io/)
    
    **Used for:**
    - Options flow data
    - Equity put/call ratios
    - High-quality market data
    
    **Free tier:** 5 API calls/minute
    """)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        polygon_key = st.text_input(
            "Polygon API Key",
            value=settings.get_api_key('POLYGON_API_KEY'),
            type="password",
            key="polygon_input"
        )
    with col2:
        st.write("")
        st.write("")
        if st.button("Test", key="test_polygon"):
            if polygon_key:
                with st.spinner("Testing..."):
                    if settings.test_polygon_api(polygon_key):
                        st.success("✅ Valid!")
                    else:
                        st.error("❌ Invalid")
            else:
                st.warning("Enter key first")
    
    if st.button("Save Polygon Key", key="save_polygon"):
        if polygon_key:
            if settings.save_api_key('POLYGON_API_KEY', polygon_key):
                st.success("✅ Polygon API key saved!")
        else:
            st.warning("Please enter an API key")
    
    st.markdown("---")
    
    # Manual PCCE Input Section
    st.header("📊 Manual Data Overrides")
    st.markdown("""
    **CBOE Equity Put/Call Ratio (PCCE)**
    
    Real CBOE data requires a subscription. Enter today's PCCE value from your trading platform:
    - ThinkorSwim: $PCCE
    - TradingView: CBOE:PCCE
    - Your charts: PCCE ticker
    
    This will override the estimated value on the Overview page.
    """)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        manual_pcce = st.number_input(
            "PCCE Value (e.g., 0.718)",
            min_value=0.0,
            max_value=5.0,
            value=float(settings.get_api_key('MANUAL_PCCE') or 0.0),
            step=0.001,
            format="%.3f",
            key="pcce_input",
            help="Enter the PCCE value from your trading platform"
        )
    with col2:
        st.write("")
        st.write("")
        pcce_date = st.text_input(
            "Date",
            value=datetime.now().strftime("%Y-%m-%d"),
            key="pcce_date",
            help="Date of this P/C value"
        )
    
    if st.button("Save PCCE Override", key="save_pcce"):
        if manual_pcce > 0:
            # Save PCCE value
            if settings.save_api_key('MANUAL_PCCE', str(manual_pcce)):
                # Save date
                settings.save_api_key('MANUAL_PCCE_DATE', pcce_date)
                st.success(f"✅ PCCE override saved: {manual_pcce:.3f} (Date: {pcce_date})")
                st.info(" Overview page will now show this value instead of estimated P/C")
        else:
            st.warning("Please enter a valid PCCE value (typically 0.5 - 2.0)")
    
    # Show current override status
    current_pcce = settings.get_api_key('MANUAL_PCCE')
    current_date = settings.get_api_key('MANUAL_PCCE_DATE')
    
    if current_pcce and float(current_pcce) > 0:
        st.info(f"✅ Current override: PCCE = {float(current_pcce):.3f} (Date: {current_date or 'Not set'})")
        
        if st.button("Clear PCCE Override", key="clear_pcce"):
            settings.save_api_key('MANUAL_PCCE', '0.0')
            settings.save_api_key('MANUAL_PCCE_DATE', '')
            st.success("Override cleared - will use estimated value")
            st.rerun()
    else:
        st.caption("No manual override set - using estimated P/C from VIX/VXV")
    
    st.markdown("---")

    # Market Breadth Settings
    st.header("📊 Market Breadth Settings")
    st.markdown("""
    **Stock Sample Size for Breadth Calculation**

    Choose between speed and precision for market breadth analysis:
    """)

    # Get current setting
    current_mode = settings.get_api_key('BREADTH_MODE') or 'fast'

    breadth_mode = st.radio(
        "Breadth Calculation Mode",
        options=['fast', 'full'],
        index=0 if current_mode == 'fast' else 1,
        format_func=lambda x: {
            'fast': '⚡ Fast Mode (100 stocks) - Recommended',
            'full': '🎯 Full Mode (500 stocks) - More precise'
        }[x],
        key="breadth_mode_radio",
        help="Fast mode uses 100 representative stocks and completes in 30-60 seconds. Full mode uses all 500 S&P stocks but takes 3-5 minutes."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **⚡ Fast Mode (100 stocks)**
        - ~30-60 seconds refresh time
        - Representative sample across all sectors
        - McClellan values scaled to match NYSE standard
        - Lower risk of API rate limits
        """)
    with col2:
        st.markdown("""
        **🎯 Full Mode (500 stocks)**
        - ~3-5 minutes refresh time
        - True S&P 500 breadth calculation
        - No scaling needed
        - Higher precision during unusual markets
        """)

    if st.button("Save Breadth Settings", key="save_breadth_mode"):
        if settings.save_api_key('BREADTH_MODE', breadth_mode):
            st.success(f"✅ Breadth mode saved: {breadth_mode.upper()}")
            st.info("🔄 Click 'Refresh Breadth Data' on the Market Breadth page to apply.")

    # Show current status
    if current_mode:
        mode_display = "⚡ Fast (100 stocks)" if current_mode == 'fast' else "🎯 Full (500 stocks)"
        st.caption(f"Current mode: {mode_display}")

    st.markdown("---")

    # Current Status
    st.header("🔌 API Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fred_status = "✅ Configured" if settings.get_api_key('FRED_API_KEY') else "❌ Not Set"
        st.metric("FRED", fred_status)
    
    with col2:
        av_status = "✅ Configured" if settings.get_api_key('ALPHA_VANTAGE_API_KEY') else "❌ Not Set"
        st.metric("Alpha Vantage", av_status)
    
    with col3:
        polygon_status = "✅ Configured" if settings.get_api_key('POLYGON_API_KEY') else "⚪ Optional"
        st.metric("Polygon", polygon_status)
    
    st.markdown("---")
    
    # Help Section
    with st.expander("❓ Help & Tips"):
        st.markdown("""
        **Where are my API keys stored?**
        - Keys are saved to a `.env` file in your project directory
        - This file is not committed to Git (should be in `.gitignore`)
        - Keys are loaded automatically when the dashboard starts
        
        **Do I need all these APIs?**
        - **FRED:** Highly recommended for credit spreads and economic data
        - **Alpha Vantage:** Recommended as Yahoo Finance backup
        - **Polygon:** Optional, for advanced options data
        
        **Free tier limits:**
        - FRED: Unlimited (with key)
        - Alpha Vantage: 500 calls/day
        - Polygon: 5 calls/minute
        
        **Security:**
        - Keys are stored locally on your machine
        - Never share your `.env` file
        - Use password-type inputs to hide keys
        """)


if __name__ == "__main__":
    render_settings_page()