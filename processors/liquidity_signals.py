"""
Liquidity Signals Module
Provides liquidity analysis and signal generation for Phase 1 & Phase 2
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class LiquiditySignal:
    """Data class for liquidity signals"""
    signal: str  # "SUPPORTIVE", "NEUTRAL", "DRAINING"
    regime: str  # Same as signal for backward compatibility
    strength: float  # 0-100
    net_liquidity_billions: float
    z_score: float
    details: dict
    timestamp: datetime


class LiquidityAnalyzer:
    """
    Phase 1 Liquidity Analyzer
    Analyzes RRP + TGA based liquidity (old formula)
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze(self, liquidity_df: pd.DataFrame) -> LiquiditySignal:
        """
        Analyze liquidity data and generate signal
        
        Args:
            liquidity_df: DataFrame with columns: date, rrp_on, tga, sofr
            
        Returns:
            LiquiditySignal object
        """
        try:
            if liquidity_df.empty:
                return self._empty_signal()
            
            # Calculate net liquidity (old formula: -(RRP + TGA))
            latest = liquidity_df.iloc[-1]
            rrp = latest.get('rrp_on')  # Don't default to 0 - preserve NaN
            tga = latest.get('tga')      # Don't default to 0 - preserve NaN
            
            # DEBUG: Log actual values
            logger.info(f"DEBUG: RRP type={type(rrp)}, value={rrp}, isna={pd.isna(rrp)}")
            logger.info(f"DEBUG: TGA type={type(tga)}, value={tga}, isna={pd.isna(tga)}")
            
            # Check for NaN BEFORE calculation
            if pd.isna(rrp) or pd.isna(tga):
                logger.error("Liquidity data contains NaN - cannot generate reliable signal")
                logger.error(f"RRP: {rrp}, TGA: {tga}")
                return LiquiditySignal(
                    signal='DATA_UNAVAILABLE',
                    regime='UNKNOWN',
                    strength=0,
                    net_liquidity_billions=None,
                    z_score=None,
                    details={'error': 'RRP or TGA data unavailable (NaN)', 'rrp': str(rrp), 'tga': str(tga)},
                    timestamp=datetime.now()
                )
            
            net_liquidity = -(rrp + tga)
            
            # Calculate z-score for regime classification
            liquidity_df_calc = liquidity_df.copy()
            liquidity_df_calc['net_liquidity'] = -(
                liquidity_df_calc.get('rrp_on', 0).fillna(0) +
                liquidity_df_calc.get('tga', 0).fillna(0)
            )
            
            mean = liquidity_df_calc['net_liquidity'].mean()
            std = liquidity_df_calc['net_liquidity'].std()
            z_score = (net_liquidity - mean) / std if std > 0 else 0
            
            # CRITICAL: Check if calculation produced NaN
            if pd.isna(net_liquidity) or pd.isna(z_score) or pd.isna(mean) or pd.isna(std):
                logger.error(f"Calculation produced NaN: net={net_liquidity}, z={z_score}, mean={mean}, std={std}")
                return LiquiditySignal(
                    signal='DATA_UNAVAILABLE',
                    regime='UNKNOWN',
                    strength=0,
                    net_liquidity_billions=None,
                    z_score=None,
                    details={'error': 'NaN produced during calculation', 'rrp': float(rrp), 'tga': float(tga)},
                    timestamp=datetime.now()
                )
            
# Classify regime
            if z_score > 0.7:
                signal = "SUPPORTIVE"
                strength = min(100, 50 + (z_score * 25))
            elif z_score < -0.7:
                signal = "DRAINING"
                strength = min(100, 50 + (abs(z_score) * 25))
            else:
                signal = "NEUTRAL"
                strength = 50
            
            details = {
                'rrp_on': float(rrp),
                'tga': float(tga),
                'sofr': float(latest.get('sofr', 0)) if 'sofr' in latest else None,
                'lookback_days': len(liquidity_df)
            }
            
            return LiquiditySignal(
                signal=signal,
                regime=signal,
                strength=strength,
                net_liquidity_billions=net_liquidity,
                z_score=z_score,
                details=details,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing liquidity: {e}")
            return self._empty_signal()
    
    def _empty_signal(self) -> LiquiditySignal:
        """Return empty signal for error cases"""
        return LiquiditySignal(
            signal="UNKNOWN",
            regime="UNKNOWN",
            strength=0,
            net_liquidity_billions=0,
            z_score=0,
            details={},
            timestamp=datetime.now()
        )


class LiquiditySignalGenerator:
    """
    Phase 2 Liquidity Signal Generator
    Enhanced liquidity analysis (can work with Fed BS data)
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_signal(self, liquidity_df: pd.DataFrame) -> LiquiditySignal:
        """
        Generate liquidity signal from data
        
        Args:
            liquidity_df: DataFrame with liquidity data
            
        Returns:
            LiquiditySignal object
        """
        try:
            if liquidity_df.empty:
                return self._empty_signal()
            
            # Calculate net liquidity (old formula for compatibility)
            latest = liquidity_df.iloc[-1]
            rrp = latest.get('rrp_on', 0) or 0
            tga = latest.get('tga', 0) or 0
            net_liquidity = -(rrp + tga)
            
            # Calculate historical context
            liquidity_df_calc = liquidity_df.copy()
            liquidity_df_calc['net_liquidity'] = -(
                liquidity_df_calc.get('rrp_on', 0).fillna(0) +
                liquidity_df_calc.get('tga', 0).fillna(0)
            )
            
            mean = liquidity_df_calc['net_liquidity'].mean()
            std = liquidity_df_calc['net_liquidity'].std()
            z_score = (net_liquidity - mean) / std if std > 0 else 0
            
            # Generate signal
            if z_score > 0.7:
                signal = "SUPPORTIVE"
                regime = "SUPPORTIVE"
                strength = min(100, 50 + (z_score * 25))
            elif z_score < -0.7:
                signal = "DRAINING"
                regime = "DRAINING"
                strength = min(100, 50 + (abs(z_score) * 25))
            else:
                signal = "NEUTRAL"
                regime = "NEUTRAL"
                strength = 50
            
            details = {
                'rrp_on': float(rrp),
                'tga': float(tga),
                'sofr': float(latest.get('sofr', 0)) if 'sofr' in latest and pd.notna(latest.get('sofr')) else None,
                'lookback_days': len(liquidity_df),
                'mean': float(mean),
                'std': float(std)
            }
            
            self.logger.info(
                f"Liquidity signal generated: {signal} "
                f"(Net: ${net_liquidity:.0f}B, Z: {z_score:.2f})"
            )
            
            return LiquiditySignal(
                signal=signal,
                regime=regime,
                strength=strength,
                net_liquidity_billions=net_liquidity,
                z_score=z_score,
                details=details,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error generating liquidity signal: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return self._empty_signal()
    
    def _empty_signal(self) -> LiquiditySignal:
        """Return empty signal for error cases"""
        return LiquiditySignal(
            signal="UNKNOWN",
            regime="UNKNOWN",
            strength=0,
            net_liquidity_billions=0,
            z_score=0,
            details={},
            timestamp=datetime.now()
        )


# Backward compatibility functions
def analyze_liquidity(liquidity_df: pd.DataFrame) -> dict:
    """Legacy function - returns dict instead of LiquiditySignal"""
    analyzer = LiquidityAnalyzer()
    signal = analyzer.analyze(liquidity_df)
    
    return {
        'signal': signal.signal,
        'regime': signal.regime,
        'strength': signal.strength,
        'net_liquidity': signal.net_liquidity_billions,
        'z_score': signal.z_score,
        'details': signal.details
    }