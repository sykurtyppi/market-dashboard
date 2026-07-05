"""Institutional Flow page."""
import streamlit as st

from data_collectors.dark_pool_collector import DarkPoolCollector
from data_collectors.insider_trading_collector import InsiderTradingCollector
from data_collectors.sector_collector import SectorCollector
from data_collectors.treasury_auction_collector import TreasuryAuctionCollector


def render(components):
    st.markdown("<h1 class='main-header'>Institutional Flow Analysis</h1>", unsafe_allow_html=True)

    st.markdown("""
    Track where smart money is positioning through dark pools, insider transactions, and Treasury auction demand.
    """)

    # Initialize collectors
    @st.cache_resource
    def get_institutional_collectors():
        return {
            'dark_pool': DarkPoolCollector(),
            'insider': InsiderTradingCollector(),
            'treasury': TreasuryAuctionCollector(),
            'sector': SectorCollector(),
        }

    inst_collectors = get_institutional_collectors()

    # --- SECTOR ROTATION SIGNAL ---
    st.subheader("Sector Rotation Signal")

    try:
        rotation = inst_collectors['sector'].get_rotation_signal()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Rotation Signal",
                rotation.get('signal', 'Unknown'),
                delta=f"{rotation.get('weighted_spread', 0):+.1f}% spread"
            )
        with col2:
            leading = rotation.get('leading_sectors', [])
            st.markdown("**Leading Sectors:**")
            for s in leading[:3]:
                st.markdown(f"- {s}")
        with col3:
            lagging = rotation.get('lagging_sectors', [])
            st.markdown("**Lagging Sectors:**")
            for s in lagging[:3]:
                st.markdown(f"- {s}")

        with st.expander("Rotation Interpretation"):
            st.info(rotation.get('interpretation', 'No interpretation available'))

    except Exception as e:
        st.warning(f"Could not load sector rotation: {e}")

    st.divider()

    # --- DARK POOL ACTIVITY ---
    st.subheader("Dark Pool Activity")

    try:
        dp_summary = inst_collectors['dark_pool'].get_dark_pool_summary()

        if dp_summary.get('status') != 'unavailable':
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Avg Dark Pool %",
                    f"{dp_summary.get('avg_dark_pool_pct', 0):.1f}%",
                    help="Percentage of trading through dark pools"
                )
            with col2:
                st.metric(
                    "Sentiment",
                    dp_summary.get('sentiment', 'Unknown'),
                )
            with col3:
                status = dp_summary.get('status', 'unknown')
                if status == 'estimated':
                    st.info("Data: Estimated (FINRA API requires registration)")
                else:
                    st.success("Data: Live from FINRA")

            with st.expander("Dark Pool Interpretation"):
                st.markdown(f"""
                **What is Dark Pool Trading?**

                Dark pools are private exchanges where institutional investors trade large blocks
                without revealing their intentions to public markets.

                **Current Reading:** {dp_summary.get('interpretation', 'N/A')}

                **Key Levels:**
                - **> 45%**: Unusually high institutional activity
                - **35-45%**: Normal institutional participation
                - **< 35%**: Retail-dominated trading
                """)
        else:
            st.warning("Dark pool data not available")

    except Exception as e:
        st.warning(f"Could not load dark pool data: {e}")

    st.divider()

    # --- INSIDER TRADING ---
    st.subheader("Insider Trading Activity")

    try:
        insider_summary = inst_collectors['insider'].get_insider_summary()

        if insider_summary.get('status') != 'unavailable':
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Buy/Sell Ratio", insider_summary.get('buy_sell_ratio', 'N/A'))
            with col2:
                st.metric("Buys", insider_summary.get('buy_count', 0))
            with col3:
                st.metric("Sells", insider_summary.get('sell_count', 0))
            with col4:
                signal = insider_summary.get('signal', 'NEUTRAL')
                color = insider_summary.get('color', '#9E9E9E')
                # Add emoji for accessibility (not color-only)
                signal_emoji = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}.get(signal, "⚪")
                st.markdown(f"<h3 style='color:{color}'>{signal_emoji} {signal}</h3>", unsafe_allow_html=True)

            # Notable transactions
            notable = inst_collectors['insider'].get_notable_transactions()
            if notable:
                with st.expander("Notable Insider Transactions (>$1M)"):
                    for t in notable[:5]:
                        emoji = "" if t.get('transaction_code') == 'P' else ""
                        st.markdown(f"""
                        {emoji} **{t.get('ticker', 'N/A')}** - {t.get('insider_name', 'Unknown')} ({t.get('title', 'N/A')})
                        - {t.get('transaction_type', 'Unknown')}: {t.get('shares', 0):,} shares @ ${t.get('price', 0):.2f}
                        - Value: ${t.get('value', 0):,.0f}
                        - Date: {t.get('date', 'N/A')}
                        """)

            with st.expander("Insider Trading Interpretation"):
                st.markdown(f"""
                **Insider Activity Signal:** {insider_summary.get('sentiment', 'N/A')}

                **Why Track Insider Trading?**

                Corporate insiders (CEOs, CFOs, Directors) have deep knowledge of their companies.
                Their personal trades often signal future prospects.

                **Key Signals:**
                - **Cluster buying** (multiple insiders buying): Strong bullish signal
                - **CEO/CFO purchases**: Most significant - they know the business best
                - **Planned sales (10b5-1)**: Less significant - often pre-scheduled

                Note: Data shown is sample/delayed. Real SEC EDGAR data has 2-day reporting requirement.
                """)
        else:
            st.warning("Insider trading data not available")

    except Exception as e:
        st.warning(f"Could not load insider data: {e}")

    st.divider()

    # --- TREASURY AUCTIONS ---
    st.subheader("Treasury Auction Demand")

    try:
        auction_summary = inst_collectors['treasury'].get_auction_summary()

        if auction_summary:
            col1, col2, col3 = st.columns(3)
            with col1:
                btc = auction_summary.get('avg_bid_to_cover', 0)
                st.metric(
                    "Avg Bid-to-Cover",
                    f"{btc:.2f}" if btc else "N/A",
                    help="Higher = More demand for Treasuries"
                )
            with col2:
                st.metric(
                    "Auction Health",
                    auction_summary.get('health', 'Unknown'),
                )
            with col3:
                st.metric(
                    "Strong/Weak",
                    f"{auction_summary.get('strong_auctions', 0)} / {auction_summary.get('weak_auctions', 0)}",
                    help="Strong (BTC>2.5) vs Weak (BTC<2.2) auctions"
                )

            # Key auction results
            key_auctions = inst_collectors['treasury'].get_key_auction_results()
            if key_auctions:
                with st.expander("Key Auction Results"):
                    for term, data in key_auctions.items():
                        if data:
                            st.markdown(f"""
                            **{term}** (auctioned {data.get('date', 'N/A').strftime('%Y-%m-%d') if data.get('date') else 'N/A'})
                            - Yield: {data.get('yield', 'N/A')}%
                            - Bid-to-Cover: {data.get('bid_to_cover', 'N/A')} ({data.get('demand_rating', 'N/A')})
                            - Indirect: {data.get('indirect_pct', 'N/A')}% | Direct: {data.get('direct_pct', 'N/A')}%
                            """)

            with st.expander("Treasury Auction Interpretation"):
                st.markdown(f"""
                **Why Track Treasury Auctions?**

                Treasury auctions show real demand for US government debt.
                Weak auctions can signal:
                - Rising rates ahead
                - Foreign buyer concerns
                - Potential market stress

                **Key Metrics:**
                - **Bid-to-Cover > 2.5**: Strong demand
                - **Bid-to-Cover < 2.0**: Weak demand (warning sign)
                - **Indirect Bidders**: Foreign/institutional demand
                - **Direct Bidders**: Domestic institutional demand
                """)
        else:
            st.warning("Treasury auction data not available")

    except Exception as e:
        st.warning(f"Could not load treasury auction data: {e}")
