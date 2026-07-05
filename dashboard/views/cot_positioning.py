"""COT Positioning page."""
import streamlit as st

from data_collectors.cot_collector import COTCollector


def render(components):
    st.markdown("<h1 class='main-header'>📊 CFTC Commitments of Traders</h1>", unsafe_allow_html=True)

    st.markdown("""
    **Institutional futures positioning** from CFTC weekly reports. Shows how hedge funds (speculators)
    and commercial hedgers are positioned. Extreme positioning often precedes reversals.
    """)

    st.info("""
    **How to read COT data:**
    - **Speculators (Non-Commercial)**: Hedge funds, CTAs - trend followers
    - **Commercials**: Hedgers (producers/consumers) - usually fade extremes
    - **Extreme Long**: Speculators max long → potential top (contrarian sell)
    - **Extreme Short**: Speculators max short → potential bottom (contrarian buy)
    """)

    st.divider()

    # Initialize COT collector
    @st.cache_resource
    def get_cot_collector():
        return COTCollector()

    cot = get_cot_collector()

    # Fetch positioning extremes
    with st.spinner("Analyzing institutional positioning..."):
        try:
            extremes = cot.get_all_extremes(weeks_back=52)
        except Exception as e:
            st.error(f"Error fetching COT data: {e}")
            extremes = []

    if not extremes:
        st.warning("""
        ⚠️ **COT data unavailable**

        CFTC data requires downloading annual ZIP files. This may take a moment on first load.
        If this persists, the CFTC website may be temporarily unavailable.
        """)
    else:
        # Summary of extreme positions
        st.subheader("🎯 Positioning Extremes")

        extreme_signals = [e for e in extremes if 'EXTREME' in e.get('signal', '')]

        if extreme_signals:
            st.warning(f"**{len(extreme_signals)} market(s) at extreme positioning levels**")

            for ext in extreme_signals:
                pct = ext['percentile']
                if ext['signal'] == 'EXTREME_LONG':
                    st.error(f"🔴 **{ext['name']}**: Speculators {pct:.0f}th percentile LONG - Contrarian SELL signal")
                else:
                    st.success(f"🟢 **{ext['name']}**: Speculators {pct:.0f}th percentile SHORT - Contrarian BUY signal")
        else:
            st.success("✅ No extreme positioning detected across tracked markets")

        st.divider()

        # Detailed positioning table
        st.subheader("📋 Current Positioning by Market")

        # Group by category
        categories = {}
        for ext in extremes:
            cat = cot.CONTRACTS.get(ext['symbol'], {}).get('category', 'other')
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(ext)

        # Display by category
        for category, items in categories.items():
            st.markdown(f"**{category.upper()}**")

            cols = st.columns(len(items)) if len(items) <= 4 else st.columns(4)

            for i, item in enumerate(items[:4]):
                with cols[i % 4]:
                    pct = item['percentile']

                    # Color based on percentile
                    if pct >= 80:
                        color = "#F44336"  # Red - extreme long
                        status = "Crowded Long"
                    elif pct >= 60:
                        color = "#FF9800"  # Orange
                        status = "Long"
                    elif pct <= 20:
                        color = "#4CAF50"  # Green - extreme short
                        status = "Crowded Short"
                    elif pct <= 40:
                        color = "#8BC34A"  # Light green
                        status = "Short"
                    else:
                        color = "#9E9E9E"  # Gray
                        status = "Neutral"

                    st.markdown(
                        f"""<div style='text-align: center; padding: 1rem;
                        background: {color}20; border-radius: 0.5rem; border-left: 4px solid {color};'>
                        <h4 style='margin:0;'>{item['symbol']}</h4>
                        <p style='margin:0.2rem 0; font-size: 0.85em;'>{item['name']}</p>
                        <h2 style='margin:0.5rem 0; color: {color};'>{pct:.0f}th</h2>
                        <p style='margin:0; color: {color};'>{status}</p>
                        </div>""",
                        unsafe_allow_html=True
                    )

                    # Net position info
                    if 'current_net' in item:
                        net = item['current_net']
                        st.caption(f"Net: {net:+,} contracts")

            st.markdown("")  # Spacing

        st.divider()

        # Historical context
        st.subheader("📈 How to Use This Data")

        st.markdown("""
        **Contrarian Signals:**
        - When speculators hit **90th+ percentile** long, they're "all in" → less buying power left
        - When speculators hit **10th percentile** short, they're max bearish → short covering rally potential

        **Confirmation Signals:**
        - Extreme positioning + price reversal = higher probability trade
        - COT data is **weekly** (released Fridays) - use for swing/position trades, not day trading

        **Key Markets to Watch:**
        - **ES (S&P 500)**: Overall equity sentiment
        - **TY (10Y Treasury)**: Bond market positioning
        - **GC (Gold)**: Safe haven flows
        - **DX (Dollar)**: Currency/risk sentiment
        - **VX (VIX)**: Volatility positioning
        """)
