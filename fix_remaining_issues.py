#!/usr/bin/env python3
"""
Fix remaining dashboard issues:
1. Repo collector not returning repo_df properly
2. Liquidity NaN values
3. Fed BS display issue
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path.home() / "Market Dashboard"
sys.path.insert(0, str(project_root))

print("=" * 80)
print("DIAGNOSING DASHBOARD ISSUES")
print("=" * 80)

# Test 1: Repo Collector
print("\n1. Testing Repo Collector...")
try:
    from data_collectors.repo_collector import RepoCollector
    
    repo = RepoCollector()
    snapshot = repo.get_full_snapshot()
    
    print(f"   Snapshot keys: {list(snapshot.keys())}")
    print(f"   SOFR: {snapshot.get('sofr')}")
    print(f"   RRP Volume: {snapshot.get('rrp_volume')}")
    print(f"   Has repo_df: {'repo_df' in snapshot}")
    
    if 'repo_df' in snapshot:
        df = snapshot['repo_df']
        print(f"   ✅ repo_df exists: {len(df)} rows")
        print(f"   ✅ Columns: {list(df.columns)}")
    else:
        print("   ❌ repo_df is MISSING!")
        print("   → This is why charts don't work")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Liquidity Collector
print("\n2. Testing Liquidity Collector...")
try:
    from data_collectors.liquidity_collector import LiquidityCollector
    
    liq = LiquidityCollector()
    df = liq.get_liquidity_history(lookback_days=365)
    
    print(f"   Total rows: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    
    if 'rrp_on' in df.columns:
        rrp_nans = df['rrp_on'].isna().sum()
        rrp_valid = df['rrp_on'].notna().sum()
        print(f"   RRP: {rrp_valid} valid, {rrp_nans} NaN ({rrp_nans/len(df)*100:.1f}% missing)")
    
    if 'tga' in df.columns:
        tga_nans = df['tga'].isna().sum()
        tga_valid = df['tga'].notna().sum()
        print(f"   TGA: {tga_valid} valid, {tga_nans} NaN ({tga_nans/len(df)*100:.1f}% missing)")
    
    if rrp_nans > 100 or tga_nans > 100:
        print("   ❌ Too many NaN values!")
        print("   → This causes 'N/A' display")
    else:
        print("   ✅ Data quality looks good")
        
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 3: Fed Balance Sheet
print("\n3. Testing Fed Balance Sheet Collector...")
try:
    from data_collectors.fed_balance_sheet_collector import FedBalanceSheetCollector
    
    fed_bs = FedBalanceSheetCollector()
    snapshot = fed_bs.get_full_snapshot()
    
    print(f"   Snapshot keys: {list(snapshot.keys())}")
    print(f"   Total assets: ${snapshot.get('total_assets'):.0f}B")
    print(f"   QT pace: {snapshot.get('qt_pace_billions_month')}B/mo")
    print(f"   Has balance_sheet_df: {'balance_sheet_df' in snapshot}")
    
    if 'balance_sheet_df' in snapshot:
        df = snapshot['balance_sheet_df']
        print(f"   ✅ balance_sheet_df exists: {len(df)} rows")
    else:
        print("   ❌ balance_sheet_df is MISSING!")
        
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 4: QT Analyzer
print("\n4. Testing QT Analyzer...")
try:
    from processors.qt_analyzer import QTAnalyzer
    from data_collectors.fed_balance_sheet_collector import FedBalanceSheetCollector
    from data_collectors.liquidity_collector import LiquidityCollector
    
    fed_bs = FedBalanceSheetCollector()
    fed_snapshot = fed_bs.get_full_snapshot()
    
    liq = LiquidityCollector()
    liq_df = liq.get_liquidity_history(lookback_days=365)
    
    qt = QTAnalyzer()
    
    if 'balance_sheet_df' in fed_snapshot:
        signal = qt.analyze(fed_snapshot['balance_sheet_df'], liq_df)
        
        if signal:
            print(f"   ✅ QT Signal generated")
            print(f"   Net liquidity: ${signal.net_liquidity_billions:.1f}B")
            print(f"   Regime: {signal.regime}")
        else:
            print("   ❌ QT Signal is None")
    else:
        print("   ❌ Cannot test - balance_sheet_df missing")
        
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)
print("\nSummary of issues found above.")
print("Scroll up to see specific problems marked with ❌")