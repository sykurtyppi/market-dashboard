"""
Daily market data update script
Fetches all data sources, calculates VRP, and updates the database.
Run from project root with:

    python -m scheduler.daily_update

or

    python scheduler/daily_update.py

Phase 1: FRED, Fear/Greed, CBOE, Breadth, Yahoo, VRP
Phase 2: Fed Balance Sheet, MOVE Index, Repo/SOFR rates
"""

from datetime import datetime
from pathlib import Path
import sys
import logging
import pandas as pd

# Configure logging for scheduled tasks
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Path setup – add project root to sys.path so imports work when run as a script
# --------------------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# --------------------------------------------------------------------------------------
# Imports - Phase 1
# --------------------------------------------------------------------------------------
from data_collectors.fred_collector import FREDCollector
from data_collectors.fear_greed_collector import FearGreedCollector
from data_collectors.cboe_collector import CBOECollector
from data_collectors.breadth_collector import SP500ADLineCalculator
from data_collectors.yahoo_collector import YahooCollector
from database.db_manager import DatabaseManager
from processors.left_strategy import LEFTStrategy
from processors.vrp_module import VRPAnalyzer

# --------------------------------------------------------------------------------------
# Imports - Phase 2 (Fed Balance Sheet, Treasury Stress, Repo Markets)
# --------------------------------------------------------------------------------------
from data_collectors.fed_balance_sheet_collector import FedBalanceSheetCollector
from data_collectors.move_collector import MOVECollector
from data_collectors.repo_collector_enhanced import RepoCollector

class MarketDataUpdater:
    """Main class for daily market data updates."""

    def __init__(self):
        # Phase 1 collectors - FRED is optional (requires API key)
        try:
            self.fred = FREDCollector()
        except ValueError as e:
            logger.warning(f"FRED collector disabled: {e}")
            self.fred = None

        self.fear_greed = FearGreedCollector()
        self.cboe = CBOECollector()
        self.breadth = SP500ADLineCalculator()
        self.yahoo = YahooCollector()

        # Phase 2 collectors - also optional
        try:
            self.fed_bs = FedBalanceSheetCollector()
        except Exception as e:
            logger.warning(f"Fed Balance Sheet collector disabled: {e}")
            self.fed_bs = None

        self.move = MOVECollector()

        try:
            self.repo = RepoCollector()
        except Exception as e:
            logger.warning(f"Repo collector disabled: {e}")
            self.repo = None

        # DB & processors
        self.db = DatabaseManager()
        self.left_strategy = LEFTStrategy()
        self.vrp_analyzer = VRPAnalyzer(lookback_days=21)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_get_latest_fred(fred_dict, key):
        """Safely extract the latest value from a FRED DataFrame dict."""
        if key in fred_dict and fred_dict[key] is not None and not fred_dict[key].empty:
            df = fred_dict[key]
            # second column is the series id
            series_id = df.columns[1]
            return float(df.iloc[-1][series_id])
        return None

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------
    def run_full_update(self):
        """Run the complete market data update pipeline."""
        logger.info("=" * 60)
        logger.info(f"MARKET DATA UPDATE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)

        # ------------------------------------------------------------------
        # 1. FRED data (optional - requires API key)
        # ------------------------------------------------------------------
        fred_data = None
        if self.fred:
            logger.info("Fetching FRED data...")
            fred_data = self.fred.get_all_indicators()

            if fred_data:
                logger.info(f"Fetched {len(fred_data)} FRED indicator series")
                self.db.save_indicators_batch(fred_data)
                logger.info("Saved FRED indicators to database")
            else:
                logger.warning("No FRED data returned")
        else:
            logger.warning("Skipping FRED data (no API key configured)")

        # ------------------------------------------------------------------
        # 2. LEFT strategy signal (uses HY credit spreads from FRED)
        # ------------------------------------------------------------------
        logger.info("Calculating LEFT strategy signal...")

        left_signal_data = {"signal": "NEUTRAL"}
        try:
            if fred_data and "credit_spread_hy" in fred_data and not fred_data["credit_spread_hy"].empty:
                left_signal_data = self.left_strategy.calculate_signal(
                    fred_data["credit_spread_hy"]
                )

                if left_signal_data.get("signal") != "INSUFFICIENT_DATA":
                    logger.info(f"LEFT Signal: {left_signal_data['signal']}")
                    logger.info(f"  Strength: {left_signal_data['strength']:.1f}/100")
                    logger.info(f"  Distance from EMA: {left_signal_data['pct_from_ema']:+.2f}%")

                    self.db.save_signal(
                        "LEFT",
                        left_signal_data["signal"],
                        left_signal_data["strength"],
                        left_signal_data,
                    )
                else:
                    logger.warning(f"{left_signal_data.get('reason', 'Insufficient data')}")
                    left_signal_data = {"signal": "NEUTRAL"}
            else:
                logger.warning("No credit_spread_hy data available for LEFT signal")
        except Exception as e:
            logger.error(f"Error calculating LEFT signal: {e}")
            left_signal_data = {"signal": "NEUTRAL"}

        # ------------------------------------------------------------------
        # 3. Fear & Greed Index
        # ------------------------------------------------------------------
        logger.info("Fetching Fear & Greed Index...")
        fear_greed_data = self.fear_greed.get_fear_greed_score() or {}

        if "score" in fear_greed_data and fear_greed_data["score"] is not None:
            logger.info(
                f"Fear & Greed Score: "
                f"{fear_greed_data['score']:.0f} ({fear_greed_data.get('rating', '')})"
            )
        else:
            logger.error("Failed to fetch Fear & Greed data")
            fear_greed_data = {"score": None}

        # ------------------------------------------------------------------
        # 4. CBOE data (VIX, contango, put/call, etc.)
        # ------------------------------------------------------------------
        logger.info("Fetching CBOE volatility & options data...")
        cboe_data = self.cboe.get_all_data() or {}

        vix_spot_cboe = cboe_data.get("vix_spot")
        vix_contango_cboe = cboe_data.get("vix_contango")
        total_pc_cboe = cboe_data.get("put_call_ratios", {}).get("total_pc")

        if vix_spot_cboe:
            logger.info(f"VIX Spot (CBOE): {vix_spot_cboe:.2f}")
            if vix_contango_cboe is not None:
                logger.info(f"  VIX Contango (CBOE): {vix_contango_cboe:+.2f}%")
        else:
            logger.warning("VIX spot from CBOE unavailable")

        # ------------------------------------------------------------------
        # 5. Yahoo Finance data (fallbacks/proxies)
        # ------------------------------------------------------------------
        logger.info("Fetching Yahoo Finance proxies...")
        yahoo_data = self.yahoo.get_all_data() or {}

        if yahoo_data.get("vix") is not None:
            logger.info(f"VIX (Yahoo): {yahoo_data['vix']:.2f}")
        if yahoo_data.get("vix_contango_proxy") is not None:
            logger.info(f"VIX Contango Proxy (Yahoo): {yahoo_data['vix_contango_proxy']:+.2f}%")
        if yahoo_data.get("market_breadth_proxy") is not None:
            logger.info(
                f"Market Breadth Proxy (Yahoo): "
                f"{yahoo_data['market_breadth_proxy']*100:.1f}% advancing"
            )
        if yahoo_data.get("put_call_proxy") is not None:
            logger.info(f"Put/Call Proxy (Yahoo): {yahoo_data['put_call_proxy']:.3f}")

        # ------------------------------------------------------------------
        # 6. Market breadth (NYSE breadth collector)
        # ------------------------------------------------------------------
        logger.info("Fetching NYSE market breadth...")
        nyse_breadth_ratio = None
        try:
            breadth_df = self.breadth.get_breadth_history(days=90)
            if not breadth_df.empty:
                self.db.save_breadth_data(breadth_df)
                latest_breadth = breadth_df.iloc[-1]['breadth_pct']
                nyse_breadth_ratio = latest_breadth / 100.0  # Convert to ratio
                logger.info(f"NYSE Breadth: {latest_breadth:.1f}% advancing")
                logger.info(f"Saved {len(breadth_df)} days of breadth data")
            else:
                logger.warning("No breadth data available")
        except Exception as e:
            logger.warning(f"Breadth calculation failed: {e}")

        # ------------------------------------------------------------------
        # 7. VRP (Volatility Risk Premium) analysis
        # ------------------------------------------------------------------
        logger.info("Calculating Volatility Risk Premium (VRP)...")

        # Prefer CBOE VIX; fall back to Yahoo VIX if needed
        vix_for_vrp = vix_spot_cboe or yahoo_data.get("vix")
        vrp_analysis = None

        if vix_for_vrp is None:
            logger.error("No VIX value available for VRP calculation")
        else:
            vrp_analysis = self.vrp_analyzer.get_complete_analysis(vix=vix_for_vrp)

            if "error" in vrp_analysis:
                logger.error(f"VRP analysis failed: {vrp_analysis['error']}")
                vrp_analysis = None
            else:
                logger.info(
                    f"VRP: {vrp_analysis['vrp']:+.2f} | "
                    f"Realized Vol (21d): {vrp_analysis['realized_vol']:.2f} | "
                    f"Regime: {vrp_analysis['regime']} "
                    f"({vrp_analysis['vix_range']})"
                )
                logger.info(
                    f"  Expected 6M SPX return for this regime: "
                    f"{vrp_analysis['expected_6m_return']:.1f}%"
                )

                # Store VRP and components into the generic indicators table
                today = datetime.now()
                try:
                    self.db.save_indicator(
                        "vrp_21d", today, vrp_analysis["vrp"], series_id="VRP_21D"
                    )
                    self.db.save_indicator(
                        "realized_vol_21d",
                        today,
                        vrp_analysis["realized_vol"],
                        series_id="SPY_RV_21D",
                    )
                    self.db.save_indicator(
                        "vix_spot_vrp", today, vrp_analysis["vix"], series_id="^VIX"
                    )
                except Exception as e:
                    logger.warning(f"Failed to save VRP indicators: {e}")

                # If DatabaseManager has a dedicated helper, use it as well
                if hasattr(self.db, "save_vrp_data"):
                    try:
                        self.db.save_vrp_data(vrp_analysis)
                        logger.info("VRP snapshot saved via save_vrp_data()")
                    except Exception as e:
                        logger.warning(f"save_vrp_data() failed: {e}")

        # ------------------------------------------------------------------
        # 8. PHASE 2: Fed Balance Sheet (for Net Liquidity)
        # ------------------------------------------------------------------
        logger.info("Fetching Fed Balance Sheet data (Phase 2)...")
        fed_bs_data = None
        try:
            if not self.fed_bs:
                logger.warning("Fed Balance Sheet collector not available (requires FRED API)")
                raise Exception("Collector not initialized")
            fed_bs_data = self.fed_bs.get_balance_sheet_df()
            if fed_bs_data is not None and not fed_bs_data.empty:
                latest_assets = fed_bs_data['total_assets'].iloc[-1] / 1e6  # Convert to trillions
                logger.info(f"Fed Total Assets: ${latest_assets:.2f}T")
                logger.info(f"Fetched {len(fed_bs_data)} observations")

                # Save to database if method exists
                if hasattr(self.db, "save_fed_balance_sheet"):
                    self.db.save_fed_balance_sheet(fed_bs_data)
                    logger.info("Fed Balance Sheet saved to database")
            else:
                logger.warning("No Fed Balance Sheet data available")
        except Exception as e:
            logger.error(f"Fed Balance Sheet fetch failed: {e}")

        # ------------------------------------------------------------------
        # 9. PHASE 2: MOVE Index (Treasury Volatility)
        # ------------------------------------------------------------------
        logger.info("Fetching MOVE Index data (Phase 2)...")
        move_data = None
        move_value = None
        try:
            move_snapshot = self.move.get_full_snapshot()
            if move_snapshot:
                move_value = move_snapshot.get('move')
                stress_level = move_snapshot.get('stress_level', 'UNKNOWN')
                percentile = move_snapshot.get('percentile', 0)

                logger.info(f"MOVE Index: {move_value:.1f}")
                logger.info(f"  Stress Level: {stress_level} ({percentile:.0f}th percentile)")

                # Save to database if method exists
                if hasattr(self.db, "save_move_data"):
                    move_df = move_snapshot.get('move_df')
                    if move_df is not None and not move_df.empty:
                        self.db.save_move_data(move_df)
                        logger.info("MOVE data saved to database")
                    else:
                        # Fallback: save only latest observation
                        move_df = pd.DataFrame([{
                            'date': move_snapshot.get('latest_date'),
                            'move': move_snapshot.get('move'),
                            'percentile': move_snapshot.get('percentile'),
                            'source': 'snapshot'
                        }])
                        self.db.save_move_data(move_df)
                        logger.info("MOVE latest value saved to database")
            else:
                logger.warning("No MOVE data available")
        except Exception as e:
            logger.error(f"MOVE Index fetch failed: {e}")

        # ------------------------------------------------------------------
        # 10. PHASE 2: Repo Market Data (SOFR, IORB, RRP)
        # ------------------------------------------------------------------
        logger.info("Fetching Repo Market data (Phase 2)...")
        repo_data = None
        sofr_value = None
        rrp_value = None
        try:
            if not self.repo:
                logger.warning("Repo collector not available (requires FRED API)")
                raise Exception("Collector not initialized")
            repo_df = self.repo.get_repo_history(days_back=90)
            if repo_df is not None and not repo_df.empty:
                latest = repo_df.iloc[-1]
                sofr_value = latest.get('sofr')
                iorb_value = latest.get('iorb')
                rrp_value = latest.get('rrp_on')
                spread = latest.get('sofr_iorb_spread')

                if sofr_value is not None:
                    logger.info(f"SOFR: {sofr_value:.2f}%")
                if iorb_value is not None:
                    logger.info(f"  IORB: {iorb_value:.2f}%")
                if spread is not None:
                    logger.info(f"  SOFR-IORB Spread: {spread:+.2f} bps")
                if rrp_value is not None:
                    logger.info(f"  RRP Volume: ${rrp_value:.0f}B")

                # Save to database if method exists
                if hasattr(self.db, "save_repo_data"):
                    self.db.save_repo_data(repo_df)
                    logger.info("Repo data saved to database")
            else:
                logger.warning("No Repo market data available")
        except Exception as e:
            logger.error(f"Repo data fetch failed: {e}")

        # ------------------------------------------------------------------
        # 11. Build and save daily dashboard snapshot
        # ------------------------------------------------------------------
        logger.info("Saving daily snapshot for dashboard...")

        # Choose best available sources (CBOE first, then Yahoo proxies)
        vix_spot = vix_spot_cboe or yahoo_data.get("vix")
        vix_contango = vix_contango_cboe if vix_contango_cboe is not None else yahoo_data.get(
            "vix_contango_proxy"
        )
        put_call_ratio = (
            total_pc_cboe
            if total_pc_cboe is not None
            else yahoo_data.get("put_call_proxy")
        )
        market_breadth = (
            nyse_breadth_ratio
            if nyse_breadth_ratio is not None
            else yahoo_data.get("market_breadth_proxy")
        )

        # Fetch VIX9D, VVIX, and SKEW
        vix9d = cboe_data.get('vix9d')
        vvix = cboe_data.get('vvix')
        vvix_signal_data = cboe_data.get('vvix_signal', {})
        vvix_signal = vvix_signal_data.get('signal') if vvix_signal_data else None
        skew = cboe_data.get('skew')

        # Log VVIX buy signal if active
        if vvix_signal == 'STRONG BUY':
            logger.info(f"VVIX BUY SIGNAL ACTIVE! VVIX at {vvix:.1f} - Historic turning point")
        elif vvix_signal == 'BUY ALERT':
            logger.info(f"VVIX elevated at {vvix:.1f} - Watch for 120+ spike")

        # Get fallback values from Yahoo if FRED unavailable
        fred_hy_spread = self._safe_get_latest_fred(fred_data, "credit_spread_hy")
        fred_treasury_10y = self._safe_get_latest_fred(fred_data, "treasury_10y")

        # Use Yahoo fallbacks if FRED data is missing
        treasury_10y_final = fred_treasury_10y
        hy_spread_final = fred_hy_spread

        if treasury_10y_final is None:
            yahoo_treasury = yahoo_data.get("treasury_10y")
            if yahoo_treasury is not None:
                treasury_10y_final = yahoo_treasury
                logger.info(f"Using Yahoo fallback for 10Y Treasury: {treasury_10y_final:.2f}%")

        if hy_spread_final is None:
            yahoo_hy = yahoo_data.get("hy_spread_proxy")
            if yahoo_hy is not None:
                hy_spread_final = yahoo_hy
                logger.info(f"Using Yahoo fallback for HY Spread: {hy_spread_final:.2f}%")

        snapshot = {
            # Core identifiers
            "date": datetime.now().strftime("%Y-%m-%d"),
            # Phase 1: Credit & Rates (with Yahoo fallbacks)
            "credit_spread_hy": hy_spread_final,
            "credit_spread_ig": self._safe_get_latest_fred(fred_data, "credit_spread_ig"),
            "treasury_10y": treasury_10y_final,
            "fed_funds": self._safe_get_latest_fred(fred_data, "fed_funds"),
            # Phase 1: Volatility
            "vix_spot": vix_spot,
            "vix_contango": vix_contango,
            "vix9d": vix9d,
            "vvix": vvix,
            "vvix_signal": vvix_signal,
            "skew": skew,
            "vrp": vrp_analysis.get("vrp") if vrp_analysis else None,
            # Phase 1: Sentiment & Breadth
            "put_call_ratio": put_call_ratio,
            "fear_greed_score": fear_greed_data.get("score"),
            "market_breadth": market_breadth,
            "left_signal": left_signal_data.get("signal"),
            # Phase 2: Treasury Stress
            "move_index": move_value,
            # Phase 2: Repo Market
            "sofr": sofr_value,
            "rrp_volume": rrp_value,
        }

        self.db.save_daily_snapshot(snapshot)
        logger.info("Daily snapshot saved to database")

        logger.info("=" * 60)
        logger.info("UPDATE COMPLETE")
        logger.info("=" * 60)


if __name__ == "__main__":
    updater = MarketDataUpdater()
    updater.run_full_update()
