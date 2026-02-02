"""
QT (Quantitative Tightening) Analyzer

Combines Fed Balance Sheet, TGA, and RRP to calculate PROPER net liquidity:

    Net Liquidity = Fed Balance Sheet - TGA - RRP

This is the formula used by major macro funds (Crossborder Capital, Brent Donnelly, SpotGamma).

Also provides QT regime classification and liquidity stress signals.

Parameters loaded from config/parameters.yaml
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config import cfg

logger = logging.getLogger(__name__)


@dataclass
class NetLiquiditySignal:
    """Complete net liquidity signal with proper formula."""
    signal: str  # "SUPPORTIVE", "NEUTRAL", "DRAINING"
    strength: float  # 0-100
    net_liquidity_billions: float  # Fed BS - TGA - RRP (in billions)
    z_score: float
    fed_bs_billions: float
    tga_billions: float
    rrp_billions: float
    qt_cumulative_billions: float  # Total QT since peak
    qt_pace_billions_month: float  # Current QT pace
    regime_color: str
    description: str
    details: dict


class QTAnalyzer:
    """
    Analyze Quantitative Tightening and calculate proper net liquidity.

    Combines:
    - Fed Balance Sheet (from FedBalanceSheetCollector)
    - TGA (Treasury General Account)
    - RRP (Overnight Reverse Repo)

    To produce: Net Liquidity = Fed BS - TGA - RRP

    Parameters loaded from config/parameters.yaml
    """

    def __init__(self, lookback_days: Optional[int] = None):
        """
        Args:
            lookback_days: Days to use for z-score calculation (default from config)
        """
        # Load from config
        qt_cfg = cfg.liquidity.qt
        self.lookback_days = lookback_days or qt_cfg.lookback_days
        self.expanding_threshold = qt_cfg.expanding_threshold
        self.contracting_threshold = qt_cfg.contracting_threshold

    def _z_score(self, series: pd.Series) -> Optional[float]:
        """Calculate z-score for latest value vs historical mean."""
        clean_series = series.dropna()
        
        if len(clean_series) < 30:
            logger.warning("Not enough data for z-score")
            return None
        
        mu = clean_series.mean()
        sigma = clean_series.std()
        
        if sigma == 0 or np.isnan(sigma):
            return None
        
        latest = clean_series.iloc[-1]
        return float((latest - mu) / sigma)

    def calculate_net_liquidity(
        self,
        fed_bs_df: pd.DataFrame,
        tga_series: pd.Series,
        rrp_series: pd.Series,
    ) -> pd.DataFrame:
        """
        Calculate proper net liquidity from components.
        
        Args:
            fed_bs_df: DataFrame from FedBalanceSheetCollector.get_balance_sheet_history()
                       Must have columns: date, total_assets
            tga_series: Series with TGA values (index=date, values=millions)
            rrp_series: Series with RRP values (index=date, values=millions)
        
        Returns:
            DataFrame with:
                - date
                - fed_bs (billions)
                - tga (billions)
                - rrp (billions)
                - net_liquidity (billions) = fed_bs - tga - rrp
                - qt_cumulative (billions)
        """
        if fed_bs_df.empty:
            return pd.DataFrame()

        df = fed_bs_df.copy()
        df = df.set_index("date") if "date" in df.columns else df

        # Convert Fed BS to billions
        df["fed_bs"] = df["total_assets"] / 1000.0

        # Merge TGA and RRP
        df = df.join(tga_series.rename("tga_millions"), how="left")
        df = df.join(rrp_series.rename("rrp_millions"), how="left")

        # Forward fill and convert to billions
        df["tga_millions"] = df["tga_millions"].ffill().fillna(0)
        df["rrp_millions"] = df["rrp_millions"].ffill().fillna(0)

        df["tga"] = df["tga_millions"] / 1000.0
        df["rrp"] = df["rrp_millions"] / 1000.0

        # Calculate NET LIQUIDITY (proper formula!)
        df["net_liquidity"] = df["fed_bs"] - df["tga"] - df["rrp"]

        # Add QT metrics if available
        if "qt_cumulative" in df.columns:
            df["qt_cumulative_billions"] = df["qt_cumulative"] / 1000.0
        if "qt_pace_billions_month" in df.columns:
            df["qt_pace"] = df["qt_pace_billions_month"]

        df = df.reset_index()
        
        logger.info(
            f"Net Liquidity: {df['net_liquidity'].iloc[-1]:.1f}B "
            f"(Fed: {df['fed_bs'].iloc[-1]:.0f}B, TGA: {df['tga'].iloc[-1]:.0f}B, RRP: {df['rrp'].iloc[-1]:.0f}B)"
        )

        return df

    def analyze(
        self,
        fed_bs_df: pd.DataFrame,
        tga_series: pd.Series,
        rrp_series: pd.Series,
    ) -> NetLiquiditySignal:
        """
        Analyze net liquidity regime and return signal.
        
        Args:
            fed_bs_df: Fed balance sheet DataFrame
            tga_series: TGA pandas Series (index=date)
            rrp_series: RRP pandas Series (index=date)
        
        Returns:
            NetLiquiditySignal with regime classification
        """
        df = self.calculate_net_liquidity(fed_bs_df, tga_series, rrp_series)

        if df.empty:
            return NetLiquiditySignal(
                signal="NEUTRAL",
                strength=0.0,
                net_liquidity_billions=0.0,
                z_score=0.0,
                fed_bs_billions=0.0,
                tga_billions=0.0,
                rrp_billions=0.0,
                qt_cumulative_billions=0.0,
                qt_pace_billions_month=0.0,
                regime_color="#9E9E9E",
                description="No data available",
                details={"error": "No data"},
            )

        # Take last N days for z-score
        df_recent = df.tail(self.lookback_days)
        
        latest = df.iloc[-1]
        net_liq = latest["net_liquidity"]
        fed_bs = latest["fed_bs"]
        tga = latest["tga"]
        rrp = latest["rrp"]
        
        qt_cumulative = latest.get("qt_cumulative_billions", 0.0)
        qt_pace = latest.get("qt_pace", 0.0)

        # Calculate z-score
        z = self._z_score(df_recent["net_liquidity"])
        z_score = z if z is not None else 0.0

        # Regime classification using config thresholds
        # Higher net liquidity = MORE supportive for risk assets
        if z_score > self.expanding_threshold:
            signal = "SUPPORTIVE"
            color = "#4CAF50"  # Green
            description = "Net liquidity is expanding. Fed balance sheet growth outpacing TGA/RRP drains. Supportive for risk assets."
            strength = min(100.0, 60 + (z_score - self.expanding_threshold) * 20)
        elif z_score < self.contracting_threshold:
            signal = "DRAINING"
            color = "#F44336"  # Red
            description = "Net liquidity is contracting. QT + elevated TGA/RRP draining liquidity from markets. Headwind for risk assets."
            strength = min(100.0, 60 + (-z_score - abs(self.contracting_threshold)) * 20)
        else:
            signal = "NEUTRAL"
            color = "#FFC107"  # Yellow
            description = "Net liquidity near historical average. Neither strong tailwind nor headwind."
            strength = max(0.0, 40 - abs(z_score) * 20)

        details = {
            "latest_date": latest["date"].strftime("%Y-%m-%d") if isinstance(latest["date"], pd.Timestamp) else str(latest["date"]),
            "net_liq_formula": "Fed BS - TGA - RRP",
            "fed_bs_billions": float(fed_bs),
            "tga_billions": float(tga),
            "rrp_billions": float(rrp),
            "qt_since_peak": float(qt_cumulative),
            "qt_monthly_pace": float(qt_pace),
            "z_score": float(z_score),
        }

        logger.info(
            f"Net Liquidity Regime: {signal} (z={z_score:+.2f}, strength={strength:.1f})"
        )

        return NetLiquiditySignal(
            signal=signal,
            strength=float(strength),
            net_liquidity_billions=float(net_liq),
            z_score=float(z_score),
            fed_bs_billions=float(fed_bs),
            tga_billions=float(tga),
            rrp_billions=float(rrp),
            qt_cumulative_billions=float(qt_cumulative),
            qt_pace_billions_month=float(qt_pace),
            regime_color=color,
            description=description,
            details=details,
        )