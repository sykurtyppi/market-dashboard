"""
Test script for all data collectors
Run this to verify everything is working before building the dashboard
"""

import os
from dotenv import load_dotenv
from data_collectors.fred_collector import FREDCollector
from data_collectors.fear_greed_collector import FearGreedCollector

load_dotenv()

def test_fred():
    """Test FRED API collector"""
    print("\n" + "="*80)
    print("TESTING FRED COLLECTOR")
    print("="*80)
    
    if not os.getenv('FRED_API_KEY'):
        print("\n⚠ WARNING: No FRED API key found in .env file")
        print("Get your free API key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("Then add it to .env file as: FRED_API_KEY=your_key_here")
        return False
    
    try:
        collector = FREDCollector()
        
        # Test basic data fetch
        latest = collector.get_latest_values()
        
        if latest:
            print("\n✓ Successfully fetched latest values")
            print(f"  Total indicators: {len(latest)}")
            
            # Show credit spread
            if 'credit_spread_hy' in latest:
                print(f"  HYG OAS: {latest['credit_spread_hy']['value']:.4f}%")
        
        # Test LEFT strategy
        signals = collector.calculate_credit_spread_signals()
        
        if 'signal' in signals:
            print(f"\n✓ LEFT Strategy calculated")
            print(f"  Signal: {signals['signal']}")
            print(f"  Distance from EMA: {signals['pct_from_ema']:+.2f}%")
        
        return True
        
    except Exception as e:
        print(f"\n✗ FRED Collector failed: {e}")
        return False

def test_fear_greed():
    """Test Fear & Greed collector"""
    print("\n" + "="*80)
    print("TESTING FEAR & GREED COLLECTOR")
    print("="*80)
    
    try:
        collector = FearGreedCollector()
        data = collector.get_fear_greed_score()
        
        if data and 'score' in data:
            print(f"\n✓ Successfully fetched Fear & Greed Index")
            print(f"  Score: {data['score']}")
            print(f"  Rating: {data['rating']}")
            return True
        else:
            print("\n✗ No data returned from Fear & Greed API")
            return False
            
    except Exception as e:
        print(f"\n✗ Fear & Greed Collector failed: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("MARKET DASHBOARD - DATA COLLECTOR TESTS")
    print("="*80)
    
    results = {
        'FRED API': test_fred(),
        'Fear & Greed Index': test_fear_greed(),
    }
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:30s} {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Ready to build the dashboard.")
    else:
        print("\n⚠ Some tests failed. Fix the issues before proceeding.")
        print("\nCommon fixes:")
        print("1. Make sure you have a FRED API key in your .env file")
        print("2. Check your internet connection")
        print("3. Make sure all dependencies are installed: pip install -r requirements.txt")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    main()