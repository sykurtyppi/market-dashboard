"""
Volatility Risk Premium (VRP) & Regime Classification Module
Calculates realized volatility, VRP, and classifies volatility regimes

Parameters loaded from config/parameters.yaml
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import logging
from utils.validators import validate_vix, validate_realized_vol

from config import cfg

# Set up logger
logger = logging.getLogger(__name__)


class VRPAnalyzer:
    """Analyzes Volatility Risk Premium and classifies volatility regimes"""

    # Default regimes fallback for Streamlit Cloud compatibility
    DEFAULT_REGIMES = {
        'complacent': {'vix_max': 12, 'expected_6m_return': 15.2, 'color': '#4CAF50'},
        'normal': {'vix_min': 12, 'vix_max': 16, 'expected_6m_return': 12.8, 'color': '#8BC34A'},
        'elevated': {'vix_min': 16, 'vix_max': 20, 'expected_6m_return': 10.5, 'color': '#FFC107'},
        'fearful': {'vix_min': 20, 'vix_max': 30, 'expected_6m_return': 8.2, 'color': '#FF9800'},
        'panic': {'vix_min': 30, 'vix_max': 40, 'expected_6m_return': 18.5, 'color': '#F44336'},
        'extreme_panic': {'vix_min': 40, 'expected_6m_return': 25.0, 'color': '#9C27B0'},
    }

    def __init__(self, lookback_days: Optional[int] = None):
        """
        Initialize VRP Analyzer

        Args:
            lookback_days: Number of days for realized volatility calculation (default from config)
        """
        # Load from config with fallbacks
        try:
            vrp_cfg = cfg.volatility.vrp
            self.lookback_days = lookback_days or vrp_cfg.lookback_days
            self._regimes = vrp_cfg.regimes if hasattr(vrp_cfg, 'regimes') else self.DEFAULT_REGIMES
        except AttributeError:
            # Fallback if config not fully loaded
            self.lookback_days = lookback_days or 21
            self._regimes = self.DEFAULT_REGIMES
            logger.info("Using default VRP regimes")
    
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
        # Use config-based regime thresholds
        complacent = self._regimes.complacent
        normal = self._regimes.normal
        elevated = self._regimes.elevated
        fearful = self._regimes.fearful
        panic = self._regimes.panic
        extreme = self._regimes.extreme_panic

        if vix < complacent.vix_max:
            return {
                "regime": "Complacent",
                "vix_range": f"0-{complacent.vix_max}",
                "expected_6m_return": complacent.expected_6m_return,
                "vix_level": vix,
                "color": complacent.color
            }
        elif vix < normal.vix_max:
            return {
                "regime": "Normal",
                "vix_range": f"{normal.vix_min}-{normal.vix_max}",
                "expected_6m_return": normal.expected_6m_return,
                "vix_level": vix,
                "color": normal.color
            }
        elif vix < elevated.vix_max:
            return {
                "regime": "Elevated",
                "vix_range": f"{elevated.vix_min}-{elevated.vix_max}",
                "expected_6m_return": elevated.expected_6m_return,
                "vix_level": vix,
                "color": elevated.color
            }
        elif vix < fearful.vix_max:
            return {
                "regime": "Fearful",
                "vix_range": f"{fearful.vix_min}-{fearful.vix_max}",
                "expected_6m_return": fearful.expected_6m_return,
                "vix_level": vix,
                "color": fearful.color
            }
        elif vix < panic.vix_max:
            return {
                "regime": "Panic",
                "vix_range": f"{panic.vix_min}-{panic.vix_max}",
                "expected_6m_return": panic.expected_6m_return,
                "vix_level": vix,
                "color": panic.color
            }
        else:
            return {
                "regime": "Extreme Panic",
                "vix_range": f"{extreme.vix_min}+",
                "expected_6m_return": extreme.expected_6m_return,
                "vix_level": vix,
                "color": extreme.color
            }
    
    def get_regime_color(self, regime: str) -> str:
        """Get color code for regime visualization"""
        # Use config-based colors
        color_map = {
            "Complacent": self._regimes.complacent.color,
            "Normal": self._regimes.normal.color,
            "Elevated": self._regimes.elevated.color,
            "Fearful": self._regimes.fearful.color,
            "Panic": self._regimes.panic.color,
            "Extreme Panic": self._regimes.extreme_panic.color,
        }
        return color_map.get(regime, "#9E9E9E")
    
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
