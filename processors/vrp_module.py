"""
Volatility Risk Premium (VRP) & Regime Classification Module
Calculates realized volatility, VRP, and classifies volatility regimes
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import logging
from utils.validators import validate_vix, validate_realized_vol

# Set up logger
logger = logging.getLogger(__name__)


class VRPAnalyzer:
    """Analyzes Volatility Risk Premium and classifies volatility regimes"""
    
    # Historical VIX-to-forward-return mapping (based on empirical data)
    VIX_FORWARD_RETURNS = {
        (0, 12): {"6m_return": 15.2, "label": "Complacent"},
        (12, 16): {"6m_return": 12.8, "label": "Normal"},
        (16, 20): {"6m_return": 10.5, "label": "Elevated"},
        (20, 30): {"6m_return": 8.2, "label": "Fearful"},
        (30, 40): {"6m_return": 18.5, "label": "Panic"},
        (40, 100): {"6m_return": 25.0, "label": "Extreme Panic"},
    }
    
    def __init__(self, lookback_days: int = 21):
        """
        Initialize VRP Analyzer
        
        Args:
            lookback_days: Number of days for realized volatility calculation (default 21)
        """
        self.lookback_days = lookback_days
    
    def calculate_realized_volatility(self, ticker: str = "SPY") -> Optional[float]:
        """
        Calculate realized volatility over lookback period
        
        Args:
            ticker: Ticker symbol (default SPY)
        
        Returns:
            Annualized realized volatility as percentage, or None if calculation fails
        """
        try:
            # Fetch data with buffer (+10 for weekends/holidays, +1 for log returns)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days + 15)
            
            spy = yf.Ticker(ticker)
            data = spy.history(start=start_date, end=end_date)
            
            if len(data) < self.lookback_days + 1:  # Need +1 for log returns
                logger.warning(f"Insufficient data: got {len(data)} days, need {self.lookback_days + 1}")
                return None
            
            # Use last N+1 days (extra for log returns calculation)
            data = data.tail(self.lookback_days + 1)
            
            # Calculate log returns
            log_returns = np.log(data['Close'] / data['Close'].shift(1))
            
            # Annualized standard deviation (252 trading days)
            realized_vol = log_returns.std() * np.sqrt(252) * 100
            
            return float(realized_vol)
            
        except Exception as e:
            logger.error(f"Error calculating realized volatility: {e}")
            return None
    
    def calculate_vrp(self, vix: float, realized_vol: float) -> Optional[float]:
        """
        Calculate Volatility Risk Premium
        
        Args:
            vix: Current VIX level
            realized_vol: Realized volatility
        
        Returns:
            VRP = VIX - Realized Vol, or None if inputs invalid
        """
        # Validate inputs
        vix = validate_vix(vix)
        realized_vol = validate_realized_vol(realized_vol)
        
        if vix is None or realized_vol is None:
            logger.error("Invalid inputs for VRP calculation")
            return None
        
        return vix - realized_vol
    
    def classify_vol_regime(self, vix: float) -> Dict:
        """
        Classify volatility regime based on VIX level
        
        Args:
            vix: Current VIX level
        
        Returns:
            Dict with regime label and expected 6-month forward return
        """
        for (low, high), data in self.VIX_FORWARD_RETURNS.items():
            if low <= vix < high:
                return {
                    "regime": data["label"],
                    "vix_range": f"{low}-{high}",
                    "expected_6m_return": data["6m_return"],
                    "vix_level": vix
                }
        
        # Default to highest bucket
        return {
            "regime": "Extreme Panic",
            "vix_range": "40+",
            "expected_6m_return": 25.0,
            "vix_level": vix
        }
    
    def get_regime_color(self, regime: str) -> str:
        """Get color code for regime visualization"""
        colors = {
            "Complacent": "#90EE90",      # Light green
            "Normal": "#4CAF50",          # Green
            "Elevated": "#FFC107",        # Amber
            "Fearful": "#FF9800",         # Orange
            "Panic": "#F44336",           # Red
            "Extreme Panic": "#B71C1C",   # Dark red
        }
        return colors.get(regime, "#9E9E9E")
    
    def get_vrp_interpretation(self, vrp: float) -> Dict:
        """
        Interpret VRP value
        
        Args:
            vrp: Volatility Risk Premium value
        
        Returns:
            Dict with interpretation and trading implications
        """
        if vrp > 8:
            return {
                "level": "Very High",
                "interpretation": "Options are rich vs realized volatility",
                "implication": "Historically supportive for equities; consider selling vol",
                "color": "#4CAF50"
            }
        elif vrp > 4:
            return {
                "level": "High",
                "interpretation": "Options moderately expensive",
                "implication": "Neutral to slightly bullish for equities",
                "color": "#8BC34A"
            }
        elif vrp > 0:
            return {
                "level": "Positive",
                "interpretation": "Options fairly priced",
                "implication": "Balanced risk/reward",
                "color": "#FFC107"
            }
        elif vrp > -4:
            return {
                "level": "Negative",
                "interpretation": "Realized vol exceeding implied",
                "implication": "Options may be cheap; caution warranted",
                "color": "#FF9800"
            }
        else:
            return {
                "level": "Very Negative",
                "interpretation": "Realized vol significantly above implied",
                "implication": "Market stress; consider hedging",
                "color": "#F44336"
            }
    
    def get_complete_analysis(self, vix: float = None) -> Dict:
        """
        Get complete VRP analysis
        
        Args:
            vix: Current VIX level. If None, will fetch from yfinance.
                 Recommended: pass VIX from YahooCollector to reuse retry logic.
        
        Returns:
            Complete analysis dict with all metrics
        """
        try:
            # Fetch VIX if not provided
            if vix is None:
                logger.info("VIX not provided, fetching directly from yfinance")
                vix_ticker = yf.Ticker("^VIX")
                vix_data = vix_ticker.history(period="1d")
                
                if vix_data.empty:
                    logger.error("Could not fetch VIX data")
                    return {"error": "Could not fetch VIX data"}
                
                vix = float(vix_data['Close'].iloc[-1])
            
            # Calculate realized volatility
            realized_vol = self.calculate_realized_volatility()
            
            if realized_vol is None:
                logger.error("Could not calculate realized volatility")
                return {"error": "Could not calculate realized volatility"}
            
            # Calculate VRP
            vrp = self.calculate_vrp(vix, realized_vol)
            
            # Get regime classification
            regime = self.classify_vol_regime(vix)
            
            # Get VRP interpretation
            vrp_interp = self.get_vrp_interpretation(vrp)
            
            return {
                "vix": round(vix, 2),
                "realized_vol": round(realized_vol, 2),
                "vrp": round(vrp, 2),
                "regime": regime["regime"],
                "vix_range": regime["vix_range"],
                "expected_6m_return": regime["expected_6m_return"],
                "vrp_level": vrp_interp["level"],
                "vrp_interpretation": vrp_interp["interpretation"],
                "vrp_implication": vrp_interp["implication"],
                "regime_color": self.get_regime_color(regime["regime"]),
                "vrp_color": vrp_interp["color"],
                "lookback_days": self.lookback_days,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"VRP analysis failed: {e}", exc_info=True)
            return {"error": f"Analysis failed: {str(e)}"}
    
    def get_historical_vrp(self, days: int = 252) -> pd.DataFrame:
        """
        Calculate historical VRP time series with proper date alignment
        
        Args:
            days: Number of days of history to fetch
        
        Returns:
            DataFrame with VIX, Realized Vol (21d & 50d), and VRP, properly aligned by date
        """
        try:
            logger.info(f"Starting get_historical_vrp for {days} days")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 60)  # Extra buffer for 50-day calc
            
            # Fetch VIX
            logger.info("Fetching VIX data...")
            vix_ticker = yf.Ticker("^VIX")
            vix_data = vix_ticker.history(start=start_date, end=end_date)
            logger.info(f"VIX data: {len(vix_data)} rows")
            
            # Fetch SPY
            logger.info("Fetching SPY data...")
            spy = yf.Ticker("SPY")
            spy_data = spy.history(start=start_date, end=end_date)
            logger.info(f"SPY data: {len(spy_data)} rows")
            
            if vix_data.empty or spy_data.empty:
                logger.error(f"Empty data! VIX: {len(vix_data)}, SPY: {len(spy_data)}")
                return pd.DataFrame()
            
            # Calculate log returns
            logger.info("Calculating log returns...")
            log_returns = np.log(spy_data['Close'] / spy_data['Close'].shift(1))
            
            # Calculate rolling realized volatility - both 21-day and 50-day
            logger.info("Calculating realized volatility...")
            realized_vol_21d = log_returns.rolling(window=self.lookback_days).std() * np.sqrt(252) * 100
            realized_vol_50d = log_returns.rolling(window=50).std() * np.sqrt(252) * 100
            
            # Create DataFrame with both realized vols (has SPY dates)
            spy_df = pd.DataFrame({
                'realized_vol': realized_vol_21d,
                'realized_vol_50d': realized_vol_50d
            }, index=spy_data.index)
            
            # Create DataFrame with VIX
            vix_df = pd.DataFrame({
                'vix': vix_data['Close']
            }, index=vix_data.index)
            
            # CRITICAL FIX: Remove timezone and normalize to date-only
            # VIX has America/Chicago, SPY has America/New_York - they won't match!
            # Solution: Convert to timezone-naive dates
            logger.info("Normalizing date indices...")
            spy_df.index = pd.to_datetime(spy_df.index.date)  # Remove time AND timezone
            vix_df.index = pd.to_datetime(vix_df.index.date)  # Remove time AND timezone
            
            logger.info(f"SPY index sample: {spy_df.index[:3].tolist()}")
            logger.info(f"VIX index sample: {vix_df.index[:3].tolist()}")
            
            # Align by date index (handles holidays/misalignment)
            logger.info("Aligning data...")
            combined = vix_df.join(spy_df, how='inner')
            logger.info(f"After join: {len(combined)} rows")
            combined = combined.dropna()
            logger.info(f"After dropna: {len(combined)} rows")
            
            if combined.empty:
                logger.error("Combined DataFrame is empty after join/dropna!")
                return pd.DataFrame()
            
            # Calculate VRP (using 21-day as primary)
            combined['vrp'] = combined['vix'] - combined['realized_vol']
            
            # Reset index to make date a column
            result = combined.reset_index()
            # Rename the index column to 'date' regardless of its original name
            result.columns = ['date'] + list(result.columns[1:])
            
            logger.info(f"Successfully created VRP history with {len(result)} rows")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating historical VRP: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()


# Test function
if __name__ == "__main__":
    print("\n" + "="*80)
    print("VRP & VOLATILITY REGIME ANALYZER - TEST")
    print("="*80)
    
    analyzer = VRPAnalyzer(lookback_days=21)
    
    print("\n Running complete VRP analysis...")
    analysis = analyzer.get_complete_analysis()
    
    if "error" in analysis:
        print(f"❌ Error: {analysis['error']}")
    else:
        print(f"\n✅ CURRENT METRICS:")
        print(f"   VIX: {analysis['vix']}")
        print(f"   Realized Vol (21d): {analysis['realized_vol']:.2f}")
        print(f"   VRP: {analysis['vrp']:.2f}")
        
        print(f"\n REGIME CLASSIFICATION:")
        print(f"   Regime: {analysis['regime']}")
        print(f"   VIX Range: {analysis['vix_range']}")
        print(f"   Expected 6M Return: {analysis['expected_6m_return']:.1f}%")
        
        print(f"\n VRP INTERPRETATION:")
        print(f"   Level: {analysis['vrp_level']}")
        print(f"   {analysis['vrp_interpretation']}")
        print(f"   Implication: {analysis['vrp_implication']}")
    
    print("\n Fetching historical VRP (last 90 days)...")
    history = analyzer.get_historical_vrp(days=90)
    
    if not history.empty:
        print(f"✅ Retrieved {len(history)} days of historical data")
        print(f"   VRP Range: {history['vrp'].min():.2f} to {history['vrp'].max():.2f}")
        print(f"   Current VRP Percentile: {(history['vrp'] < analysis['vrp']).mean() * 100:.1f}%")
    else:
        print("❌ Could not retrieve historical data")
    
    print("\n" + "="*80)
    print("✓ TEST COMPLETE")
    print("="*80 + "\n")