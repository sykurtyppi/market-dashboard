"""
VRP Calculator - Phase 2
Enhanced VRP calculation module (optional - not currently used in dashboard)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class VRPCalculator:
    """
    Enhanced VRP (Volatility Risk Premium) Calculator
    This is a Phase 2 component that provides additional VRP analysis
    
    Note: The main dashboard uses vrp_module.py (VRPAnalyzer)
    This class is imported for future enhancements but not currently used
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_vrp(self, vix: float, realized_vol: float) -> dict:
        """
        Calculate basic VRP
        
        Args:
            vix: VIX index value (implied volatility)
            realized_vol: Realized volatility
            
        Returns:
            dict with VRP analysis
        """
        try:
            vrp = vix - realized_vol
            
            return {
                'vix': vix,
                'realized_vol': realized_vol,
                'vrp': vrp,
                'vrp_ratio': vix / realized_vol if realized_vol > 0 else None,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating VRP: {e}")
            return {}
    
    def analyze_vrp_regime(self, vrp: float) -> str:
        """
        Classify VRP regime
        
        Args:
            vrp: VRP value
            
        Returns:
            Regime classification string
        """
        if vrp > 8:
            return "EXPENSIVE"
        elif vrp > 4:
            return "ELEVATED"
        elif vrp > 0:
            return "POSITIVE"
        elif vrp > -4:
            return "NEGATIVE"
        else:
            return "CHEAP"
    
    def get_trading_signal(self, vrp: float) -> dict:
        """
        Generate trading signal from VRP
        
        Args:
            vrp: VRP value
            
        Returns:
            dict with signal information
        """
        regime = self.analyze_vrp_regime(vrp)
        
        if vrp > 8:
            signal = "SELL_VOL"
            description = "Options expensive - consider selling volatility"
        elif vrp > 4:
            signal = "NEUTRAL_BEARISH"
            description = "Options moderately expensive"
        elif vrp > 0:
            signal = "NEUTRAL"
            description = "Options fairly priced"
        elif vrp > -4:
            signal = "NEUTRAL_BULLISH"
            description = "Options slightly cheap"
        else:
            signal = "BUY_VOL"
            description = "Options cheap - consider buying protection"
        
        return {
            'signal': signal,
            'regime': regime,
            'description': description,
            'vrp': vrp
        }


# For backward compatibility
def calculate_vrp(vix: float, realized_vol: float) -> float:
    """Legacy function - returns simple VRP value"""
    return vix - realized_vol