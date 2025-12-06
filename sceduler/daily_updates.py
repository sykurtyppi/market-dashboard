"""
Daily market data update script - INTEGRATED VERSION
Phase 1 + Phase 2 Complete
"""

import sys
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Phase 1 modules
try:
    from yahoo_collector import YahooCollector
    from vrp_module import VRPAnalyzer
    from db_manager import DatabaseManager
    from health_check import HealthCheckSystem
except ImportError as e:
    logger.error(f"Import error (Phase 1): {e}")
    logger.error("Make sure all Phase 1 files are in the same directory")
    sys.exit(1)

# Import Phase 2 modules
try:
    from data_collectors.fed_balance_sheet_collector import FedBalanceSheetCollector
    from data_collectors.move_collector import MOVECollector
    from data_collectors.repo_collector import RepoCollector
except ImportError as e:
    logger.warning(f"Import error (Phase 2): {e}")
    logger.warning("Phase 2 modules not found - will skip Phase 2 updates")
    PHASE2_AVAILABLE = False
else:
    PHASE2_AVAILABLE = True


# ============================================================================
# PHASE 2 COLLECTION FUNCTIONS
# ============================================================================

def collect_fed_balance_sheet(db: DatabaseManager, lookback_days: int = 730) -> bool:
    """
    Collect Fed Balance Sheet data
    
    Args:
        db: DatabaseManager instance
        lookback_days: Days of history to fetch
    
    Returns:
        bool: True if successful
    """
    logger.info("Starting Fed Balance Sheet collection...")
    
    try:
        collector = FedBalanceSheetCollector()
        
        # Get data
        df = collector.get_balance_sheet_history(lookback_days=lookback_days)
        
        if df.empty:
            logger.warning("No Fed Balance Sheet data returned")
            return False
        
        # Calculate QT metrics
        df_with_qt = collector.calculate_qt_metrics(df)
        
        # Save to database
        success = db.save_fed_balance_sheet(df_with_qt)
        
        if success:
            logger.info(f"Successfully saved {len(df_with_qt)} Fed BS records")
            latest = df_with_qt.iloc[-1]
            logger.info(f"Latest Fed BS: ${latest['total_assets']:.0f}B, QT Pace: ${latest.get('qt_monthly_pace', 0):.0f}B/mo")
        
        return success
        
    except Exception as e:
        logger.error(f"Error collecting Fed Balance Sheet: {e}")
        return False


def collect_move_index(db: DatabaseManager, lookback_days: int = 730) -> bool:
    """
    Collect MOVE Index data
    
    Args:
        db: DatabaseManager instance
        lookback_days: Days of history to fetch
    
    Returns:
        bool: True if successful
    """
    logger.info("Starting MOVE Index collection...")
    
    try:
        collector = MOVECollector()
        
        # Try FRED first, fallback to Yahoo
        df = collector.get_move_history(lookback_days=lookback_days, source="auto")
        
        if df.empty:
            logger.warning("No MOVE Index data returned")
            return False
        
        # Save to database
        success = db.save_move_data(df)
        
        if success:
            logger.info(f"Successfully saved {len(df)} MOVE records")
            latest = df.iloc[-1]
            logger.info(f"Latest MOVE: {latest['move']:.1f} ({latest.get('source', 'unknown')} source)")
        
        return success
        
    except Exception as e:
        logger.error(f"Error collecting MOVE Index: {e}")
        return False


def collect_repo_market(db: DatabaseManager, lookback_days: int = 730) -> bool:
    """
    Collect Repo Market data (SOFR, RRP, etc.)
    
    Args:
        db: DatabaseManager instance
        lookback_days: Days of history to fetch
    
    Returns:
        bool: True if successful
    """
    logger.info("Starting Repo Market collection...")
    
    try:
        collector = RepoCollector()
        
        # Get data
        df = collector.get_repo_history(lookback_days=lookback_days)
        
        if df.empty:
            logger.warning("No Repo Market data returned")
            return False
        
        # Calculate stress metrics
        df_with_metrics = collector.calculate_stress_metrics(df)
        
        # Save to database
        success = db.save_repo_data(df_with_metrics)
        
        if success:
            logger.info(f"Successfully saved {len(df_with_metrics)} Repo records")
            latest = df_with_metrics.iloc[-1]
            logger.info(f"Latest SOFR: {latest['sofr']:.2f}%, Z-score: {latest['sofr_z_score']:.2f}")
        
        return success
        
    except Exception as e:
        logger.error(f"Error collecting Repo Market data: {e}")
        return False


def run_phase2_updates(db: DatabaseManager) -> dict:
    """
    Run all Phase 2 data collection tasks
    
    Args:
        db: DatabaseManager instance
    
    Returns:
        dict: Results of each collection task
    """
    logger.info("=" * 70)
    logger.info("STARTING PHASE 2 DATA COLLECTION")
    logger.info("=" * 70)
    
    results = {
        "fed_balance_sheet": False,
        "move_index": False,
        "repo_market": False,
        "timestamp": datetime.now().isoformat()
    }
    
    # 1. Fed Balance Sheet (weekly data, but safe to run daily)
    logger.info("\n[1/3] Fed Balance Sheet...")
    results["fed_balance_sheet"] = collect_fed_balance_sheet(db)
    
    # 2. MOVE Index (daily data)
    logger.info("\n[2/3] MOVE Index...")
    results["move_index"] = collect_move_index(db)
    
    # 3. Repo Market (daily data)
    logger.info("\n[3/3] Repo Market...")
    results["repo_market"] = collect_repo_market(db)
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2 COLLECTION SUMMARY")
    logger.info("=" * 70)
    
    success_count = sum(1 for v in results.values() if isinstance(v, bool) and v)
    total_count = 3
    
    logger.info(f"Successful: {success_count}/{total_count}")
    logger.info(f"Fed BS: {'✅' if results['fed_balance_sheet'] else '❌'}")
    logger.info(f"MOVE:   {'✅' if results['move_index'] else '❌'}")
    logger.info(f"Repo:   {'✅' if results['repo_market'] else '❌'}")
    
    # Health check
    try:
        health = db.check_phase2_health()
        logger.info("\nDatabase Health:")
        for module, status in health.items():
            logger.info(f"  {module}: {status['status']} ({status['record_count']} records)")
    except AttributeError:
        logger.warning("Phase 2 health check not available (old db_manager.py)")
    
    logger.info("=" * 70)
    
    return results


# ============================================================================
# PHASE 1 UPDATER (EXISTING)
# ============================================================================

class MarketDataUpdater:
    """Market data updater for Phase 1 + Phase 2"""
    
    def __init__(self):
        self.yahoo = YahooCollector()
        self.vrp = VRPAnalyzer(lookback_days=21)
        self.db = DatabaseManager()
        self.health = HealthCheckSystem(self.db)
    
    def run_full_update(self):
        """Run complete market data update"""
        logger.info("=" * 80)
        logger.info(f"MARKET DATA UPDATE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        
        try:
            # ========================================
            # PHASE 1 UPDATES
            # ========================================
            
            # 1. Fetch Yahoo data
            logger.info("\n Fetching Yahoo Finance data...")
            yahoo_data = self.yahoo.get_all_data()
            
            if yahoo_data.get('vix'):
                logger.info(f"✓ VIX: {yahoo_data['vix']:.2f}")
                logger.info(f"✓ VIX Contango Proxy: {yahoo_data.get('vix_contango_proxy', 'N/A')}")
                logger.info(f"✓ Put/Call Proxy: {yahoo_data.get('put_call_proxy', 'N/A')}")
                logger.info(f"✓ Breadth Proxy: {yahoo_data.get('market_breadth_proxy', 'N/A')}")
            else:
                logger.warning("⚠ Failed to fetch VIX data")
            
            # 2. Calculate VRP
            logger.info("\n Calculating VRP...")
            vix = yahoo_data.get('vix')
            
            if vix:
                vrp_analysis = self.vrp.get_complete_analysis(vix=vix)
                
                if 'error' not in vrp_analysis:
                    logger.info(f"✓ VRP: {vrp_analysis['vrp']:+.2f}")
                    logger.info(f"✓ Regime: {vrp_analysis['regime']}")
                    logger.info(f"✓ Expected 6M Return: {vrp_analysis['expected_6m_return']:.1f}%")
                    
                    # Save VRP data
                    self.db.save_vrp_data(vrp_analysis)
                    logger.info("✓ VRP data saved to database")
                else:
                    logger.warning(f"⚠ VRP calculation error: {vrp_analysis.get('error')}")
                    vrp_analysis = None
            else:
                logger.warning("⚠ Skipping VRP calculation (no VIX data)")
                vrp_analysis = None
            
            # 3. Create daily snapshot
            logger.info("\n Saving daily snapshot...")
            snapshot = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'vix_spot': yahoo_data.get('vix'),
                'vix_contango': yahoo_data.get('vix_contango_proxy'),
                'put_call_ratio': yahoo_data.get('put_call_proxy'),
                'market_breadth': yahoo_data.get('market_breadth_proxy'),
                # Add VRP data if available
                'vrp': vrp_analysis.get('vrp') if vrp_analysis else None,
                'vol_regime': vrp_analysis.get('regime') if vrp_analysis else None,
            }
            
            self.db.save_daily_snapshot(snapshot)
            logger.info("✓ Daily snapshot saved")
            
            # 4. Run health check
            logger.info("\n Running health check...")
            health_summary = self.health.get_health_summary()
            overall_status = health_summary.get('overall_status', 'UNKNOWN')
            
            logger.info(f"✓ Overall health: {overall_status}")
            
            # Show any degraded sources
            for source_name, source_data in health_summary.get('sources', {}).items():
                status = source_data.get('status', 'UNKNOWN')
                if status not in ['HEALTHY', 'STALE']:
                    logger.warning(f"  ⚠ {source_name}: {status}")
            
            # ========================================
            # PHASE 2 UPDATES (NEW!)
            # ========================================
            
            if PHASE2_AVAILABLE:
                logger.info("\n" + "=" * 80)
                logger.info(" RUNNING PHASE 2 UPDATES")
                logger.info("=" * 80)
                
                phase2_results = run_phase2_updates(self.db)
                
                # Log summary
                success_count = sum(1 for v in phase2_results.values() if isinstance(v, bool) and v)
                logger.info(f"\n✓ Phase 2 updates: {success_count}/3 successful")
            else:
                logger.info("\n⚠ Phase 2 modules not available - skipping Phase 2 updates")
                logger.info("  To enable: Install Phase 2 data collectors in data_collectors/")
            
            logger.info("\n" + "=" * 80)
            logger.info("✓ UPDATE COMPLETE")
            logger.info("=" * 80 + "\n")
            
            return True
            
        except Exception as e:
            logger.error(f"\n❌ UPDATE FAILED: {e}", exc_info=True)
            return False


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    try:
        updater = MarketDataUpdater()
        success = updater.run_full_update()
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("\n⚠ Update interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()