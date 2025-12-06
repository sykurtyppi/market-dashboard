"""
Test Script for New Macro Modules

Run this to verify all 3 new modules are working correctly:
1. Fed Balance Sheet & QT
2. MOVE Index & Treasury Liquidity
3. Repo Market Stress

Usage:
    python3 test_new_modules.py
"""

import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent
sys.path.insert(0, str(parent_dir))

# Load environment
from dotenv import load_dotenv
load_dotenv()

print("=" * 70)
print("TESTING NEW MACRO MODULES")
print("=" * 70)

# Test 1: Fed Balance Sheet
print("\n[1/3] Testing Fed Balance Sheet & QT Module...")
try:
    from data_collectors.fed_balance_sheet_collector import FedBalanceSheetCollector
    from processors.qt_analyzer import QTAnalyzer
    
    fed = FedBalanceSheetCollector()
    snapshot = fed.get_full_snapshot(lookback_days=365)
    
    if "error" in snapshot:
        print(f"   ❌ ERROR: {snapshot['error']}")
    else:
        print(f"   ✅ Fed Balance Sheet: ${snapshot['total_assets_billions']:,.0f}B")
        print(f"   ✅ QT Cumulative: ${snapshot['qt_cumulative_billions']:,.0f}B")
        print(f"   ✅ QT Monthly Pace: ${snapshot['qt_monthly_pace_billions']:,.0f}B/month")
        print(f"   ✅ Data points: {len(snapshot['history'])}")
        
        # Test QT Analyzer
        print("\n   Testing QT Analyzer...")
        analyzer = QTAnalyzer()
        # Note: Need TGA and RRP data to test fully
        print("   ✅ QT Analyzer initialized")

except Exception as e:
    print(f"   ❌ ERROR: {e}")

# Test 2: MOVE Index
print("\n[2/3] Testing MOVE Index & Treasury Liquidity Module...")
try:
    from data_collectors.move_collector import MOVECollector
    from processors.treasury_liquidity_analyzer import TreasuryLiquidityAnalyzer
    
    move = MOVECollector()
    snapshot = move.get_full_snapshot(lookback_days=365)
    
    if "error" in snapshot:
        print(f"   ⚠️  WARNING: {snapshot['error']}")
        print("   Note: MOVE may not be available from FRED/Yahoo")
    else:
        print(f"   ✅ MOVE Index: {snapshot['move']:.1f}")
        print(f"   ✅ Percentile: {snapshot['move_percentile']:.0f}th")
        print(f"   ✅ Stress Level: {snapshot['stress_level']}")
        print(f"   ✅ Data points: {len(snapshot['history'])}")
        
        # Test Treasury Analyzer
        print("\n   Testing Treasury Liquidity Analyzer...")
        analyzer = TreasuryLiquidityAnalyzer()
        signal = analyzer.analyze(snapshot['history'])
        print(f"   ✅ Stress Signal: {signal.stress_level}")
        print(f"   ✅ Strength: {signal.strength:.0f}/100")

except Exception as e:
    print(f"   ❌ ERROR: {e}")

# Test 3: Repo Market
print("\n[3/3] Testing Repo Market Stress Module...")
try:
    from data_collectors.repo_collector import RepoCollector
    from processors.repo_analyzer import RepoAnalyzer
    
    repo = RepoCollector()
    snapshot = repo.get_full_snapshot(lookback_days=365)
    
    if "error" in snapshot:
        print(f"   ❌ ERROR: {snapshot['error']}")
    else:
        print(f"   ✅ SOFR: {snapshot['sofr']:.2f}%")
        print(f"   ✅ SOFR Z-Score: {snapshot['sofr_z_score']:+.2f}")
        print(f"   ✅ Stress Level: {snapshot['stress_level']}")
        if snapshot['rrp_billions']:
            print(f"   ✅ RRP Volume: ${snapshot['rrp_billions']:,.0f}B")
        print(f"   ✅ Data points: {len(snapshot['history'])}")
        
        # Test Repo Analyzer
        print("\n   Testing Repo Analyzer...")
        analyzer = RepoAnalyzer()
        signal = analyzer.analyze(snapshot['history'])
        print(f"   ✅ Stress Signal: {signal.stress_level}")
        print(f"   ✅ Strength: {signal.strength:.0f}/100")

except Exception as e:
    print(f"   ❌ ERROR: {e}")

# Summary
print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
print("\nIf you see ✅ for all modules, everything is working!")
print("If you see ❌, check:")
print("  1. FRED_API_KEY is set in .env")
print("  2. Files are in correct directories")
print("  3. All dependencies are installed (requests, pandas, yfinance)")
print("\nNext step: Integrate into dashboard UI")
print("=" * 70)