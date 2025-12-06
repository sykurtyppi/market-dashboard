"""
Repo Market Stress Analyzer

Analyzes funding stress in the repo market using:
1. SOFR levels and volatility
2. SOFR z-scores vs historical norms
3. RRP usage patterns
4. Repo market stress regime

The repo market is where banks and dealers get overnight funding.
Stress here = liquidity/funding crisis potential.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RepoStressSignal:
    """Repo market stress signal."""
    stress_level: str  # "NORMAL", "ELEVATED", "STRESS"
    sofr: float
    sofr_z_score: float
    rrp_billions: Optional[float]
    strength: float  # 0-100
    stress_color: str
    description: str
    details: dict


class RepoAnalyzer:
    """
    Analyze repo market stress and funding conditions.
    
    Uses SOFR as primary indicator of funding stress.
    """

    def __init__(self, lookback_days: int = 252):
        """
        Args:
            lookback_days: Days for z-score calculation
        """
        self.lookback_days = lookback_days

    def classify_stress(
        self,
        sofr_value: float,
        z_score: float,
        rrp_billions: Optional[float],
    ) -> tuple:
        """
        Classify repo market stress.
        
        CRITICAL: Only POSITIVE z-scores (high SOFR) = stress
        Negative z-scores (low SOFR) = normal/stable funding
        
        Returns:
            (stress_level, color, strength, description)
        """
        # STRESS: SOFR spiking above normal (positive z-score > 2)
        if z_score > 2.0:
            strength = min(100.0, 75.0 + z_score * 10)
            
            # Add context about RRP if available
            context = ""
            if rrp_billions is not None and rrp_billions < 50:
                context = " RRP depleted - potential liquidity shortage."
            
            return (
                "STRESS",
                "#F44336",
                strength,
                f"SOFR spiking - significant funding stress.{context} Elevated risk of broader market impact."
            )
        
        # ELEVATED: SOFR moderately high (positive z-score 1-2)
        elif z_score > 1.0:
            return (
                "ELEVATED",
                "#FF9800",
                60.0,
                f"SOFR above normal levels. Moderate funding pressure. Monitor for escalation."
            )
        
        # NORMAL: SOFR in normal range or below (z-score <= 1)
        # Note: Low SOFR (negative z-score) is actually GOOD - stable funding
        else:
            if z_score < -2.0:
                desc = f"Repo market very stable. SOFR well below historical average ({z_score:.1f}σ) - abundant liquidity."
            elif z_score < 0:
                desc = f"Repo market stable. SOFR below average - healthy funding conditions."
            else:
                desc = "Repo market functioning normally. No funding stress detected."
            
            return (
                "NORMAL",
                "#4CAF50",
                max(10.0, 30.0 - abs(z_score) * 5),  # Lower strength = more stable
                desc
            )

    def analyze(
        self,
        repo_df: pd.DataFrame,
    ) -> RepoStressSignal:
        """
        Analyze repo market stress.
        
        Args:
            repo_df: DataFrame from RepoCollector.get_repo_history()
                     Must have columns: date, sofr, sofr_z_score
        
        Returns:
            RepoStressSignal with full analysis
        """
        if repo_df.empty or "sofr" not in repo_df.columns:
            return RepoStressSignal(
                stress_level="UNKNOWN",
                sofr=0.0,
                sofr_z_score=0.0,
                rrp_billions=None,
                strength=0.0,
                stress_color="#9E9E9E",
                description="No repo market data available",
                details={"error": "No data"},
            )

        latest = repo_df.iloc[-1]
        sofr = float(latest["sofr"])
        z_score = float(latest["sofr_z_score"]) if "sofr_z_score" in repo_df.columns else 0.0
        latest_date = latest["date"].strftime("%Y-%m-%d") if isinstance(latest["date"], pd.Timestamp) else str(latest["date"])

        # RRP if available
        rrp_billions = None
        if "rrp" in repo_df.columns:
            rrp_billions = float(latest["rrp"] / 1000.0)

        # GC Repo if available
        gc_repo = None
        if "gc_repo" in repo_df.columns:
            gc_repo = float(latest["gc_repo"])

        # Classify stress
        stress_level, color, strength, description = self.classify_stress(
            sofr, z_score, rrp_billions
        )

        details = {
            "latest_date": latest_date,
            "sofr": sofr,
            "sofr_z_score": z_score,
            "rrp_billions": rrp_billions,
            "gc_repo": gc_repo,
        }

        logger.info(
            f"Repo Stress: {stress_level} (SOFR={sofr:.2f}%, z={z_score:+.2f})"
        )

        return RepoStressSignal(
            stress_level=stress_level,
            sofr=sofr,
            sofr_z_score=z_score,
            rrp_billions=rrp_billions,
            strength=strength,
            stress_color=color,
            description=description,
            details=details,
        )

    def detect_repo_spikes(
        self,
        repo_df: pd.DataFrame,
        spike_threshold: float = 2.0,
    ) -> pd.DataFrame:
        """
        Detect historical repo spikes (like Sep 2019).
        
        Args:
            repo_df: Repo history DataFrame
            spike_threshold: Z-score threshold for "spike"
        
        Returns:
            DataFrame with spike events
        """
        if repo_df.empty or "sofr_z_score" not in repo_df.columns:
            return pd.DataFrame()

        spikes = repo_df[abs(repo_df["sofr_z_score"]) > spike_threshold].copy()
        
        if not spikes.empty:
            spikes["spike_magnitude"] = abs(spikes["sofr_z_score"])
            spikes = spikes.sort_values("spike_magnitude", ascending=False)
            
            logger.info(f"Detected {len(spikes)} repo spike events")
        
        return spikes