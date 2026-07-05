"""System Health page."""
import streamlit as st

from config import cfg


def render(components):
    st.markdown("<h1 class='main-header'>🏥 System Health Check</h1>", unsafe_allow_html=True)

    st.markdown("""
    Monitor the status of all data collectors and API connections.
    Use this page to diagnose issues with data fetching.
    """)

    st.divider()

    # --- API Key Status ---
    st.subheader("🔑 API Key Status")

    from utils.secrets_helper import get_secret

    api_keys = {
        'FRED_API_KEY': {'name': 'FRED (Federal Reserve)', 'required': True},
        'NASDAQ_DATA_LINK_KEY': {'name': 'Nasdaq Data Link', 'required': False},
        'ALPHA_VANTAGE_API_KEY': {'name': 'Alpha Vantage', 'required': False},
        'POLYGON_API_KEY': {'name': 'Polygon.io', 'required': False},
        'FINRA_API_KEY': {'name': 'FINRA', 'required': False},
    }

    api_cols = st.columns(len(api_keys))
    for i, (key, info) in enumerate(api_keys.items()):
        with api_cols[i]:
            value = get_secret(key)
            if value:
                st.metric(info['name'], "✅ Set")
            elif info['required']:
                st.metric(info['name'], "❌ Missing", delta="Required", delta_color="inverse")
            else:
                st.metric(info['name'], "⚪ Not set", delta="Optional")

    st.divider()

    # --- Collector Health Check ---
    st.subheader("📡 Data Collector Status")

    if st.button("🔄 Run Health Check", type="primary"):
        with st.spinner("Testing all collectors..."):
            health_results = []

            # Test CBOE Collector
            try:
                cboe = components.get("cboe")
                if cboe:
                    vix = cboe.get_vix()
                    if vix is not None:
                        health_results.append(("CBOE (VIX)", "✅ Working", f"VIX: {vix:.2f}", "success"))
                    else:
                        health_results.append(("CBOE (VIX)", "⚠️ No Data", "VIX returned None", "warning"))
                else:
                    health_results.append(("CBOE (VIX)", "❌ Not Loaded", "Collector not initialized", "error"))
            except Exception as e:
                health_results.append(("CBOE (VIX)", "❌ Error", str(e)[:50], "error"))

            # Test Yahoo Finance
            try:
                yahoo = components.get("yahoo")
                if yahoo:
                    spy_data = yahoo.get_spy_price()
                    if spy_data is not None:
                        health_results.append(("Yahoo Finance", "✅ Working", f"SPY: ${spy_data:.2f}", "success"))
                    else:
                        health_results.append(("Yahoo Finance", "⚠️ Partial", "Some data unavailable", "warning"))
                else:
                    health_results.append(("Yahoo Finance", "❌ Not Loaded", "Collector not initialized", "error"))
            except Exception as e:
                health_results.append(("Yahoo Finance", "❌ Error", str(e)[:50], "error"))

            # Test FRED (if key exists)
            fred_key = get_secret('FRED_API_KEY')
            if fred_key:
                try:
                    fed_bs = components.get("fed_bs")
                    if fed_bs and not getattr(fed_bs, '_disabled', False):
                        snapshot = fed_bs.get_full_snapshot()
                        if snapshot and 'total_assets' in snapshot:
                            health_results.append(("FRED API", "✅ Working", f"Fed BS: ${snapshot['total_assets']/1e6:.1f}T", "success"))
                        else:
                            health_results.append(("FRED API", "⚠️ Partial", "Some data missing", "warning"))
                    else:
                        health_results.append(("FRED API", "⚠️ Disabled", "API key issue", "warning"))
                except Exception as e:
                    health_results.append(("FRED API", "❌ Error", str(e)[:50], "error"))
            else:
                health_results.append(("FRED API", "⚪ Skipped", "No API key", "info"))

            # Test Fear & Greed
            try:
                fg = components.get("fear_greed")
                if fg:
                    fg_data = fg.get_fear_greed()
                    if fg_data and 'score' in fg_data:
                        health_results.append(("CNN Fear & Greed", "✅ Working", f"Score: {fg_data['score']}", "success"))
                    else:
                        health_results.append(("CNN Fear & Greed", "⚠️ No Data", "Score unavailable", "warning"))
                else:
                    health_results.append(("CNN Fear & Greed", "❌ Not Loaded", "Collector not initialized", "error"))
            except Exception as e:
                health_results.append(("CNN Fear & Greed", "❌ Error", str(e)[:50], "error"))

            # Test Database
            try:
                db = components.get("db")
                if db:
                    snapshot = db.get_latest_snapshot()
                    if snapshot:
                        snapshot_date = snapshot.get('date', 'Unknown')
                        health_results.append(("Database", "✅ Working", f"Last: {snapshot_date}", "success"))
                    else:
                        health_results.append(("Database", "⚠️ Empty", "No snapshots", "warning"))
                else:
                    health_results.append(("Database", "❌ Not Loaded", "DB not initialized", "error"))
            except Exception as e:
                health_results.append(("Database", "❌ Error", str(e)[:50], "error"))

            # Display results as table
            st.markdown("### Health Check Results")

            for name, status, details, status_type in health_results:
                if status_type == "success":
                    st.success(f"**{name}**: {status} - {details}")
                elif status_type == "warning":
                    st.warning(f"**{name}**: {status} - {details}")
                elif status_type == "error":
                    st.error(f"**{name}**: {status} - {details}")
                else:
                    st.info(f"**{name}**: {status} - {details}")

            # Summary
            success_count = sum(1 for r in health_results if r[3] == "success")
            warning_count = sum(1 for r in health_results if r[3] == "warning")
            error_count = sum(1 for r in health_results if r[3] == "error")

            st.divider()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Working", success_count, delta="collectors")
            with col2:
                st.metric("Warnings", warning_count)
            with col3:
                st.metric("Errors", error_count, delta_color="inverse" if error_count > 0 else "off")

    st.divider()

    # --- Cache Status ---
    st.subheader("💾 Cache & Session Status")

    cache_info = {
        "Streamlit Cache": len(st.session_state) if hasattr(st, 'session_state') else 0,
        "Database Path": components.get("db").db_path if components.get("db") else "N/A",
    }

    for key, value in cache_info.items():
        st.text(f"{key}: {value}")

    # Clear cache button
    if st.button("🗑️ Clear All Caches"):
        st.cache_data.clear()
        st.success("✅ All caches cleared! Page will refresh.")
        st.rerun()

    st.divider()

    # --- Rate Limiter Status ---
    st.subheader("⏱️ Rate Limiter Status")

    try:
        from utils.retry_utils import RateLimiter
        limiters = RateLimiter._instances

        if limiters:
            for name, limiter in limiters.items():
                st.text(f"{name}: {limiter.rate} req/s, burst={limiter.burst}, tokens={limiter.tokens:.1f}")
        else:
            st.info("No rate limiters active")
    except ImportError:
        st.info("Rate limiter module not available")

    st.divider()

    # --- Config Status ---
    st.subheader("⚙️ Configuration Status")

    try:
        st.json({
            "VVIX Strong Buy": cfg.get('volatility.vvix.strong_buy_threshold', 'N/A'),
            "VRP Lookback Days": cfg.get('volatility.vrp.lookback_days', 'N/A'),
            "Put/Call Bearish": cfg.get('options.equity_put_call.bearish_threshold', 'N/A'),
            "Retry Max Attempts": cfg.get('data_collection.retry.max_retries', 'N/A'),
        })
    except Exception as e:
        st.error(f"Could not load config: {e}")
