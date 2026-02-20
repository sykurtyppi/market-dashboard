"""
Enhanced RRP Analysis for Market Dashboard
Adds rate-of-change tracking and contextual interpretation

Parameters loaded from config/parameters.yaml
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime

from config import cfg

logger = logging.getLogger(__name__)


class RRPAnalyzer:
    """
    Analyze RRP trends and provide market context

    Key insight: RRP draining = money flowing back to banks = BULLISH
    But approaching zero = transition risk (Fed must pause QT)

    Parameters loaded from config/parameters.yaml
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Load thresholds from config
        rrp_cfg = cfg.liquidity.rrp
        self.RRP_PEAK = 2600  # $2.6T peak in 2022 (historical constant)
        self.RRP_CRITICAL = rrp_cfg.critical_billions
        self.RRP_WARNING = rrp_cfg.warning_billions
    
    def analyze(self, rrp_df: pd.DataFrame) -> Dict:
        """
        Comprehensive RRP analysis
        
        Args:
            rrp_df: DataFrame with 'date' and 'rrp_on' columns
            
        Returns:
            Dict with analysis results
        """
        try:
            if rrp_df.empty or len(rrp_df) < 2:
                return self._empty_result()
            
            # Get latest value
            latest = rrp_df.iloc[-1]
            current_rrp = latest['rrp_on']
            
            # Calculate rates of change
            changes = self._calculate_changes(rrp_df)
            
            # Get phase and interpretation
            phase = self._get_phase(current_rrp, changes)
            interpretation = self._get_interpretation(current_rrp, changes, phase)
            market_impact = self._get_market_impact(phase, changes)
            
            return {
                'current_rrp': current_rrp,
                'date': latest['date'],
                'change_1d': changes['1d'],
                'change_7d': changes['7d'],
                'change_30d': changes['30d'],
                'change_90d': changes['90d'],
                'pct_from_peak': ((current_rrp - self.RRP_PEAK) / self.RRP_PEAK) * 100,
                'phase': phase,
                'interpretation': interpretation,
                'market_impact': market_impact,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing RRP: {e}")
            return self._empty_result()
    
    def _calculate_changes(self, rrp_df: pd.DataFrame) -> Dict:
        """Calculate RRP change over multiple timeframes"""
        changes = {}
        current = rrp_df.iloc[-1]['rrp_on']
        
        # 1-day change
        if len(rrp_df) >= 2:
            prev_1d = rrp_df.iloc[-2]['rrp_on']
            changes['1d'] = current - prev_1d
            changes['1d_pct'] = (changes['1d'] / prev_1d * 100) if prev_1d > 0 else 0
        else:
            changes['1d'] = 0
            changes['1d_pct'] = 0
        
        # 7-day change
        if len(rrp_df) >= 7:
            prev_7d = rrp_df.iloc[-7]['rrp_on']
            changes['7d'] = current - prev_7d
            changes['7d_pct'] = (changes['7d'] / prev_7d * 100) if prev_7d > 0 else 0
        else:
            changes['7d'] = 0
            changes['7d_pct'] = 0
        
        # 30-day change
        if len(rrp_df) >= 30:
            prev_30d = rrp_df.iloc[-30]['rrp_on']
            changes['30d'] = current - prev_30d
            changes['30d_pct'] = (changes['30d'] / prev_30d * 100) if prev_30d > 0 else 0
            changes['30d_rate'] = changes['30d'] / 30  # Daily drain rate
        else:
            changes['30d'] = 0
            changes['30d_pct'] = 0
            changes['30d_rate'] = 0
        
        # 90-day change
        if len(rrp_df) >= 90:
            prev_90d = rrp_df.iloc[-90]['rrp_on']
            changes['90d'] = current - prev_90d
            changes['90d_pct'] = (changes['90d'] / prev_90d * 100) if prev_90d > 0 else 0
        else:
            changes['90d'] = 0
            changes['90d_pct'] = 0
        
        return changes
    
    def _get_phase(self, current_rrp: float, changes: Dict) -> str:
        """
        Determine current RRP phase
        
        Phases:
        - CRITICAL: <$50B, approaching zero
        - LATE_STAGE: <$200B, getting low
        - DRAINING: >$200B, declining
        - STABLE: Little change
        - BUILDING: Increasing (rare post-2022)
        """
        if current_rrp < self.RRP_CRITICAL:
            return "CRITICAL"
        elif current_rrp < self.RRP_WARNING:
            return "LATE_STAGE"
        elif changes['30d'] < -20:  # Draining >$20B/month
            return "DRAINING"
        elif abs(changes['30d']) < 20:  # Stable within +/- $20B
            return "STABLE"
        else:  # Increasing
            return "BUILDING"
    
    def _get_interpretation(self, current_rrp: float, changes: Dict, phase: str) -> str:
        """Get detailed interpretation of RRP status"""
        lines = []
        
        # Current level context
        pct_from_peak = ((current_rrp - self.RRP_PEAK) / self.RRP_PEAK) * 100
        lines.append(f"RRP at ${current_rrp:.1f}B ({pct_from_peak:+.1f}% from $2.6T peak)")
        
        # Phase-specific interpretation
        if phase == "CRITICAL":
            lines.append(" CRITICAL: RRP nearly exhausted - Fed likely to pause/end QT soon")
            lines.append("Liquidity buffer GONE - any further QT hits bank reserves directly")
            if changes['30d_rate'] < 0:
                days_to_zero = abs(current_rrp / changes['30d_rate']) if changes['30d_rate'] != 0 else 999
                if days_to_zero < 60:
                    lines.append(f" Estimated {int(days_to_zero)} days until RRP reaches zero")
        
        elif phase == "LATE_STAGE":
            lines.append("⚠️ LATE STAGE: RRP running low - Fed watching closely")
            lines.append("Market expecting QT to end in coming months")
            if changes['30d_rate'] < 0:
                days_to_critical = abs((current_rrp - self.RRP_CRITICAL) / changes['30d_rate']) if changes['30d_rate'] != 0 else 999
                lines.append(f"~{int(days_to_critical)} days until critical level at current drain rate")
        
        elif phase == "DRAINING":
            lines.append(" DRAINING: Money flowing from RRP back to banks = BULLISH")
            lines.append(f"Draining ${abs(changes['30d']):.1f}B over past 30 days")
            lines.append("This liquidity release has supported the 2023-2024 rally")
        
        elif phase == "STABLE":
            lines.append(" STABLE: RRP relatively unchanged")
            lines.append("Liquidity conditions steady")
        
        elif phase == "BUILDING":
            lines.append(" BUILDING: Money parking at RRP (rare in 2024-2025)")
            lines.append("Flight to safety or technical factors")
        
        # Add rate of change context if available
        if changes['30d'] != 0:
            lines.append(f"30-day change: ${changes['30d']:+.1f}B ({changes['30d_pct']:+.1f}%)")
        
        return " | ".join(lines)
    
    def _get_market_impact(self, phase: str, changes: Dict) -> str:
        """
        Assess market impact
        
        Key: RRP draining = bullish (money flows to markets)
             RRP hitting zero = transition risk (Fed must act)
        """
        if phase == "CRITICAL":
            return " TRANSITION RISK: Fed must pause/end QT → likely BULLISH (signals QE restart approaching)"
        
        elif phase == "LATE_STAGE":
            return " MODERATELY BULLISH: Continued drain supports markets, but watching for Fed action"
        
        elif phase == "DRAINING":
            if changes['30d'] < -50:  # Fast drain
                return "✅ BULLISH: Fast RRP drain = significant liquidity flowing to banks/markets"
            else:
                return "✅ MILDLY BULLISH: Steady RRP drain = gradual liquidity improvement"
        
        elif phase == "STABLE":
            return " NEUTRAL: No major liquidity shifts from RRP"
        
        elif phase == "BUILDING":
            return "⚠️ CAUTION: RRP building = money leaving system (defensive positioning?)"
        
        return "UNKNOWN"
    
    def _empty_result(self) -> Dict:
        """Return empty result on error"""
        return {
            'current_rrp': None,
            'date': None,
            'phase': "UNKNOWN",
            'interpretation': "RRP data unavailable",
            'market_impact': "UNKNOWN",
            'timestamp': datetime.now()
        }
