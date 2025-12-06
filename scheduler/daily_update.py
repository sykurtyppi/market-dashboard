"""
Daily market data update script
Fetches all data sources, calculates VRP, and updates the database.
Run from project root with:

    python -m scheduler.daily_update

or

    python scheduler/daily_update.py
"""

from datetime import datetime
from pathlib import Path
import sys

# --------------------------------------------------------------------------------------
# Path setup – add project root to sys.path so imports work when run as a script
# --------------------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# --------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------
from data_collectors.fred_collector import FREDCollector
from data_collectors.fear_greed_collector import FearGreedCollector
from data_collectors.cboe_collector import CBOECollector
from data_collectors.breadth_collector import SP500ADLineCalculator
from data_collectors.yahoo_collector import YahooCollector
from database.db_manager import DatabaseManager
from processors.left_strategy import LEFTStrategy

# VRP lives in the dashboard package (dashboard/vrp_module.py)
from dashboard.vrp_module import VRPAnalyzer



from data_collectors.sp500_adline_calculator import SP500ADLineCalculator

class MarketDataUpdater:
    """Main class for daily market data updates."""

    def __init__(self):
        # Core collectors
        self.fred = FREDCollector()
        self.fear_greed = FearGreedCollector()
        self.cboe = CBOECollector()
        self.breadth = SP500ADLineCalculator()
        self.yahoo = YahooCollector()

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
        print("\n" + "=" * 80)
        print(f"MARKET DATA UPDATE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # ------------------------------------------------------------------
        # 1. FRED data
        # ------------------------------------------------------------------
        print("\n📊 Fetching FRED data...")
        fred_data = self.fred.get_all_indicators()

        if fred_data:
            print(f"✓ Fetched {len(fred_data)} FRED indicator series")
            self.db.save_indicators_batch(fred_data)
            print("✓ Saved FRED indicators to database")
        else:
            print("✗ Failed to fetch FRED data")

        # ------------------------------------------------------------------
        # 2. LEFT strategy signal (uses HY credit spreads from FRED)
        # ------------------------------------------------------------------
        print("\n📈 Calculating LEFT strategy signal...")

        left_signal_data = {"signal": "NEUTRAL"}
        try:
            if fred_data and "credit_spread_hy" in fred_data and not fred_data["credit_spread_hy"].empty:
                left_signal_data = self.left_strategy.calculate_signal(
                    fred_data["credit_spread_hy"]
                )

                if left_signal_data.get("signal") != "INSUFFICIENT_DATA":
                    print(f"✓ LEFT Signal: {left_signal_data['signal']}")
                    print(f"  Strength: {left_signal_data['strength']:.1f}/100")
                    print(f"  Distance from EMA: {left_signal_data['pct_from_ema']:+.2f}%")

                    self.db.save_signal(
                        "LEFT",
                        left_signal_data["signal"],
                        left_signal_data["strength"],
                        left_signal_data,
                    )
                else:
                    print(f"⚠ {left_signal_data.get('reason', 'Insufficient data')}")
                    left_signal_data = {"signal": "NEUTRAL"}
            else:
                print("⚠ No credit_spread_hy data available for LEFT signal")
        except Exception as e:
            print(f"✗ Error calculating LEFT signal: {e}")
            left_signal_data = {"signal": "NEUTRAL"}

        # ------------------------------------------------------------------
        # 3. Fear & Greed Index
        # ------------------------------------------------------------------
        print("\n😱 Fetching Fear & Greed Index...")
        fear_greed_data = self.fear_greed.get_fear_greed_score() or {}

        if "score" in fear_greed_data and fear_greed_data["score"] is not None:
            print(
                f"✓ Fear & Greed Score: "
                f"{fear_greed_data['score']:.0f} ({fear_greed_data.get('rating', '')})"
            )
        else:
            print("✗ Failed to fetch Fear & Greed data")
            fear_greed_data = {"score": None}

        # ------------------------------------------------------------------
        # 4. CBOE data (VIX, contango, put/call, etc.)
        # ------------------------------------------------------------------
        print("\n📉 Fetching CBOE volatility & options data...")
        cboe_data = self.cboe.get_all_data() or {}

        vix_spot_cboe = cboe_data.get("vix_spot")
        vix_contango_cboe = cboe_data.get("vix_contango")
        total_pc_cboe = cboe_data.get("put_call_ratios", {}).get("total_pc")

        if vix_spot_cboe:
            print(f"✓ VIX Spot (CBOE): {vix_spot_cboe:.2f}")
            if vix_contango_cboe is not None:
                print(f"  VIX Contango (CBOE): {vix_contango_cboe:+.2f}%")
        else:
            print("⚠ VIX spot from CBOE unavailable")

        # ------------------------------------------------------------------
        # 5. Yahoo Finance data (fallbacks/proxies)
        # ------------------------------------------------------------------
        print("\n🛰  Fetching Yahoo Finance proxies...")
        yahoo_data = self.yahoo.get_all_data() or {}

        if yahoo_data.get("vix") is not None:
            print(f"✓ VIX (Yahoo): {yahoo_data['vix']:.2f}")
        if yahoo_data.get("vix_contango_proxy") is not None:
            print(f"✓ VIX Contango Proxy (Yahoo): {yahoo_data['vix_contango_proxy']:+.2f}%")
        if yahoo_data.get("market_breadth_proxy") is not None:
            print(
                f"✓ Market Breadth Proxy (Yahoo): "
                f"{yahoo_data['market_breadth_proxy']*100:.1f}% advancing"
            )
        if yahoo_data.get("put_call_proxy") is not None:
            print(f"✓ Put/Call Proxy (Yahoo): {yahoo_data['put_call_proxy']:.3f}")

        # ------------------------------------------------------------------
        # 6. Market breadth (NYSE breadth collector)
        # ------------------------------------------------------------------
        print("\n📊 Fetching NYSE market breadth...")
        nyse_breadth_ratio = None
        try:
            breadth_df = self.breadth.get_breadth_history(days=90)
            if not breadth_df.empty:
                self.db.save_breadth_data(breadth_df)
                latest_breadth = breadth_df.iloc[-1]['breadth_pct']
                nyse_breadth_ratio = latest_breadth / 100.0  # Convert to ratio
                print(f"✓ NYSE Breadth: {latest_breadth:.1f}% advancing")
                print(f"✓ Saved {len(breadth_df)} days of breadth data")
            else:
                print("⚠ No breadth data available")
        except Exception as e:
            print(f"⚠ Breadth calculation failed: {e}")

        # ------------------------------------------------------------------
        # 7. VRP (Volatility Risk Premium) analysis
        # ------------------------------------------------------------------
        print("\n🔍 Calculating Volatility Risk Premium (VRP)...")

        # Prefer CBOE VIX; fall back to Yahoo VIX if needed
        vix_for_vrp = vix_spot_cboe or yahoo_data.get("vix")
        vrp_analysis = None

        if vix_for_vrp is None:
            print("✗ No VIX value available for VRP calculation")
        else:
            vrp_analysis = self.vrp_analyzer.get_complete_analysis(vix=vix_for_vrp)

            if "error" in vrp_analysis:
                print(f"✗ VRP analysis failed: {vrp_analysis['error']}")
                vrp_analysis = None
            else:
                print(
                    f"✓ VRP: {vrp_analysis['vrp']:+.2f} | "
                    f"Realized Vol (21d): {vrp_analysis['realized_vol']:.2f} | "
                    f"Regime: {vrp_analysis['regime']} "
                    f"({vrp_analysis['vix_range']})"
                )
                print(
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
                    print(f"⚠ Failed to save VRP indicators: {e}")

                # If DatabaseManager has a dedicated helper, use it as well
                if hasattr(self.db, "save_vrp_data"):
                    try:
                        self.db.save_vrp_data(vrp_analysis)
                        print("✓ VRP snapshot saved via save_vrp_data()")
                    except Exception as e:
                        print(f"⚠ save_vrp_data() failed: {e}")

        # ------------------------------------------------------------------
        # 8. Build and save daily dashboard snapshot
        # ------------------------------------------------------------------
        print("\n💾 Saving daily snapshot for dashboard...")

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

        # Fetch VIX9D and SKEW
        vix9d = cboe_data.get('vix9d')
        skew = cboe_data.get('skew')
        
        snapshot = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "credit_spread_hy": self._safe_get_latest_fred(fred_data, "credit_spread_hy"),
            "credit_spread_ig": self._safe_get_latest_fred(fred_data, "credit_spread_ig"),
            "treasury_10y": self._safe_get_latest_fred(fred_data, "treasury_10y"),
            "fed_funds": self._safe_get_latest_fred(fred_data, "fed_funds"),
            "vix_spot": vix_spot,
            "vix_contango": vix_contango,
            "vix9d": vix9d,
            "skew": skew,
            "put_call_ratio": put_call_ratio,
            "fear_greed_score": fear_greed_data.get("score"),
            "market_breadth": market_breadth,
            "left_signal": left_signal_data.get("signal"),
        }

        self.db.save_daily_snapshot(snapshot)
        print("✓ Daily snapshot saved to database")

        print("\n" + "=" * 80)
        print("✓ UPDATE COMPLETE")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    updater = MarketDataUpdater()
    updater.run_full_update()
