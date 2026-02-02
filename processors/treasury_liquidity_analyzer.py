"""
Treasury Liquidity Analyzer

Analyzes Treasury market stress using:
1. MOVE Index (Treasury implied volatility)
2. MOVE-VIX divergence
3. Treasury liquidity regime

This provides early warning of systemic stress - Treasury stress often
precedes equity stress.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TreasuryStressSignal:
    """Treasury market stress signal."""
    stress_level: str  # "LOW", "NORMAL", "ELEVATED", "HIGH STRESS"
    move_value: float
    move_percentile: float
    vix_value: Optional[float]
    move_vix_divergence: Optional[str]  # "Treasury Leading", "Equity Leading", "Aligned"
    stress_color: str
    strength: float  # 0-100
    description: str
    details: dict


class TreasuryLiquidityAnalyzer:
    """
    Analyze Treasury market stress and liquidity conditions.
    
    Uses MOVE Index as primary indicator.
    """

    # MOVE thresholds
    MOVE_LOW = 80
    MOVE_NORMAL = 120
    MOVE_ELEVATED = 150

    def __init__(self, lookback_days: int = 252):
        """
        Args:
            lookback_days: Days for historical context
        """
        self.lookback_days = lookback_days

    def classify_stress(self, move_value: float, percentile: float) -> tuple:
        """
        Classify Treasury stress level.
        
        Returns:
            (stress_level, color, strength, description)
        """
        if move_value < self.MOVE_LOW:
            return (
                "LOW",
                "#4CAF50",
                20.0,
                "Treasury volatility extremely low. Calm market conditions."
            )
        elif move_value < self.MOVE_NORMAL:
            return (
                "NORMAL",
                "#8BC34A",
                40.0,
                "Treasury volatility in normal range. Healthy market functioning."
            )
        elif move_value < self.MOVE_ELEVATED:
            return (
                "ELEVATED",
                "#FF9800",
                70.0,
                "Treasury volatility elevated. Increased uncertainty in fixed income markets."
            )
        else:
            # High stress - scale strength by how extreme
            strength = min(100.0, 85.0 + (move_value - self.MOVE_ELEVATED) / 10)
            return (
                "HIGH STRESS",
                "#F44336",
                strength,
                "Treasury volatility in stress territory. Significant market uncertainty. Often precedes broader risk-off."
            )

    def analyze_move_vix_divergence(
        self,
        move_value: float,
        vix_value: Optional[float],
        move_history: pd.DataFrame,
        vix_history: Optional[pd.Series],
    ) -> Optional[str]:
        """
        Analyze MOVE-VIX divergence.
        
        Returns:
            "Treasury Stress Leading", "Equity Stress Leading", "Aligned", or None
        """
        if vix_value is None or vix_history is None or move_history.empty:
            return None

        try:
            # Calculate MOVE/VIX ratio
            current_ratio = move_value / vix_value

            # Historical ratio
            df = move_history.copy()
            df = df.set_index("date") if "date" in df.columns else df
            df = df.join(vix_history.rename("vix"), how="left")
            df["ratio"] = df["move"] / df["vix"]
            df = df.dropna(subset=["ratio"])

            if len(df) < 30:
                return None

            ratio_mean = df["ratio"].mean()
            ratio_std = df["ratio"].std()

            if ratio_std == 0:
                return "Aligned"

            z = (current_ratio - ratio_mean) / ratio_std

            if z > 1.2:
                return "Treasury Stress Leading"
            elif z < -1.2:
                return "Equity Stress Leading"
            else:
                return "Aligned"

        except Exception as e:
            logger.warning(f"Error calculating MOVE-VIX divergence: {e}")
            return None

    def analyze(
        self,
        move_df: pd.DataFrame,
        vix_series: Optional[pd.Series] = None,
    ) -> TreasuryStressSignal:
        """
        Analyze Treasury stress regime.
        
        Args:
            move_df: DataFrame with columns: date, move
            vix_series: Optional Series with VIX (index=date)
        
        Returns:
            TreasuryStressSignal with full analysis
        """
        if move_df.empty:
            return TreasuryStressSignal(
                stress_level="UNKNOWN",
                move_value=0.0,
                move_percentile=0.0,
                vix_value=None,
                move_vix_divergence=None,
                stress_color="#9E9E9E",
                strength=0.0,
                description="No MOVE data available",
                details={"error": "No data"},
            )

        # Get latest values
        latest = move_df.iloc[-1]
        move_value = float(latest["move"])
        latest_date = latest["date"].strftime("%Y-%m-%d") if isinstance(latest["date"], pd.Timestamp) else str(latest["date"])

        # Calculate percentile
        percentile = (move_df["move"] < move_value).sum() / len(move_df) * 100

        # Get VIX if available
        vix_value = None
        if vix_series is not None and not vix_series.empty:
            try:
                vix_value = float(vix_series.iloc[-1])
            except (ValueError, TypeError, IndexError) as e:
                logger.warning(f"Could not extract VIX value: {e}")

        # Classify stress
        stress_level, color, strength, description = self.classify_stress(
            move_value, percentile
        )

        # Check divergence
        divergence = self.analyze_move_vix_divergence(
            move_value, vix_value, move_df, vix_series
        )

        # Add divergence context to description
        if divergence == "Treasury Stress Leading":
            description += " ⚠️ Treasury stress elevated vs equities - watch for spillover."
        elif divergence == "Equity Stress Leading":
            description += " Equity volatility high but Treasuries calm - potential safe haven flow."

        details = {
            "latest_date": latest_date,
            "move": move_value,
            "move_percentile": percentile,
            "vix": vix_value,
            "move_vix_divergence": divergence,
            "thresholds": {
                "low": self.MOVE_LOW,
                "normal": self.MOVE_NORMAL,
                "elevated": self.MOVE_ELEVATED,
            }
        }

        logger.info(
            f"Treasury Stress: {stress_level} (MOVE={move_value:.1f}, {percentile:.0f}th pct)"
        )

        return TreasuryStressSignal(
            stress_level=stress_level,
            move_value=move_value,
            move_percentile=percentile,
            vix_value=vix_value,
            move_vix_divergence=divergence,
            stress_color=color,
            strength=strength,
            description=description,
            details=details,
        )