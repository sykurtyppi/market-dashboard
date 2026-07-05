"""Component initialization: collectors, analyzers, and database handles."""
import streamlit as st

from dashboard.core.helpers import get_breadth_mode
from data_collectors.cboe_collector import CBOECollector
from data_collectors.fear_greed_collector import FearGreedCollector
from data_collectors.fed_balance_sheet_collector import FedBalanceSheetCollector
from data_collectors.fred_collector import FREDCollector
from data_collectors.liquidity_collector import LiquidityCollector
from data_collectors.market_data import MarketDataCollector
from data_collectors.move_collector import MOVECollector
from data_collectors.repo_collector_enhanced import RepoCollector
from data_collectors.sp500_adline_calculator import SP500ADLineCalculator
from data_collectors.yahoo_collector import YahooCollector
from database.db_manager import DatabaseManager
from database.health_check import HealthCheckSystem
from processors.left_strategy import LEFTStrategy
from processors.liquidity_signals import LiquidityAnalyzer
from processors.qt_analyzer import QTAnalyzer
from processors.repo_analyzer import RepoAnalyzer
from processors.treasury_liquidity_analyzer import TreasuryLiquidityAnalyzer
from processors.vrp_module import VRPAnalyzer


@st.cache_resource
def init_components():
    try:
        # FRED-dependent collectors are optional (require FRED_API_KEY)
        fred_collector = None
        liquidity_collector = None
        fed_bs_collector = None

        try:
            fred_collector = FREDCollector()
            liquidity_collector = LiquidityCollector()
            fed_bs_collector = FedBalanceSheetCollector()
        except Exception as fred_error:
            st.warning(f"FRED API not configured: {fred_error}")
            st.info("You can add your FRED API key in Settings → Secrets to enable liquidity data")

        components = {
            # Core DB / health
            "db": DatabaseManager(),
            "health": HealthCheckSystem(),

            # Phase 1 collectors
            "fred": fred_collector,
            "fear_greed": FearGreedCollector(),
            "cboe": CBOECollector(),
            "breadth": SP500ADLineCalculator(mode=get_breadth_mode()),
            "yahoo": YahooCollector(),
            "left_strategy": LEFTStrategy(),
            "vrp": VRPAnalyzer(lookback_days=21),

            # Liquidity collectors & analyzers (FRED-dependent)
            "liquidity": liquidity_collector,
            "liquidity_analyzer": LiquidityAnalyzer() if liquidity_collector else None,

            # Phase 2 collectors & processors
            "market": MarketDataCollector(),
            "fed_bs": fed_bs_collector,
            "move": MOVECollector(),
            "repo": RepoCollector(),
            "qt_analyzer": QTAnalyzer() if fed_bs_collector else None,
            "treasury_analyzer": TreasuryLiquidityAnalyzer() if fed_bs_collector else None,
            "repo_analyzer": RepoAnalyzer(),
        }

        return components
    except Exception as e:
        st.error(f"Failed to initialize: {e}")
        return None
