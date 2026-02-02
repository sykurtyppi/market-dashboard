"""
Liquidity Signals Module - CORRECTED VERSION

Uses the PROPER institutional net liquidity formula:

    Net Liquidity = Fed Balance Sheet - TGA - RRP

This is the formula used by:
- Crossborder Capital
- SpotGamma
- Brent Donnelly / Spectra Markets
- Other major macro funds

The old formula -(RRP + TGA) was INCORRECT as it ignored the Fed Balance Sheet entirely.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from config import cfg

logger = logging.getLogger(__name__)


# ============================================================================
# SIGNAL DATA CLASSES
# ============================================================================

@dataclass
class NetLiquiditySignal:
    """
    Institutional-grade net liquidity signal.

    Formula: Net Liquidity = Fed BS - TGA - RRP

    Attributes:
        signal: Regime classification ("SUPPORTIVE", "NEUTRAL", "DRAINING")
        strength: Signal confidence 0-100
        net_liquidity_billions: Fed BS - TGA - RRP in billions
        z_score: Standard deviations from lookback mean
        fed_bs_billions: Federal Reserve balance sheet total assets
        tga_billions: Treasury General Account balance
        rrp_billions: Overnight Reverse Repo volume
        regime_color: Hex color for UI display
        description: Human-readable regime explanation
        timestamp: When signal was generated
        details: Additional metadata
    """
    signal: str
    strength: float
    net_liquidity_billions: float
    z_score: float
    fed_bs_billions: float
    tga_billions: float
    rrp_billions: float
    regime_color: str
    description: str
    timestamp: datetime
    details: dict

    # Backward compatibility aliases
    @property
    def regime(self) -> str:
        """Alias for signal (backward compatibility)"""
        return self.signal


# Legacy type alias for backward compatibility
LiquiditySignal = NetLiquiditySignal


# ============================================================================
# LIQUIDITY ANALYZER - CORRECTED FORMULA
# ============================================================================

class LiquidityAnalyzer:
    """
    Institutional Net Liquidity Analyzer

    Calculates: Net Liquidity = Fed BS - TGA - RRP

    This replaces the old incorrect formula -(RRP + TGA).

    Parameters loaded from config/parameters.yaml

    Usage:
        analyzer = LiquidityAnalyzer()
        signal = analyzer.analyze(fed_bs_df, tga_series, rrp_series)
    """

    def __init__(self, lookback_days: Optional[int] = None):
        """
        Args:
            lookback_days: Days of history for z-score calculation (default from config)
        """
        # Load from config
        liq_cfg = cfg.liquidity.net_liquidity
        self.lookback_days = lookback_days or liq_cfg.lookback_days
        self.supportive_threshold = liq_cfg.supportive_threshold
        self.draining_threshold = liq_cfg.draining_threshold
        self.min_data_points = liq_cfg.min_data_points
        self._colors = liq_cfg.colors
        self._logger = logging.getLogger(__name__)

    def analyze(
        self,
        fed_bs_df: pd.DataFrame,
        tga_series: Optional[pd.Series] = None,
        rrp_series: Optional[pd.Series] = None,
    ) -> NetLiquiditySignal:
        """
        Analyze net liquidity and generate institutional-grade signal.

        Args:
            fed_bs_df: DataFrame with Fed Balance Sheet data
                       Must have columns: date, total_assets (in millions)
            tga_series: Series with TGA values (index=date, values=millions)
            rrp_series: Series with RRP values (index=date, values=millions)

        Returns:
            NetLiquiditySignal with regime classification
        """
        try:
            # Validate Fed Balance Sheet data
            if fed_bs_df is None or fed_bs_df.empty:
                return self._unavailable_signal("Fed Balance Sheet data unavailable")

            if 'total_assets' not in fed_bs_df.columns:
                return self._unavailable_signal("Fed BS missing 'total_assets' column")

            # Build the net liquidity DataFrame
            df = self._build_liquidity_dataframe(fed_bs_df, tga_series, rrp_series)

            if df.empty:
                return self._unavailable_signal("Could not build liquidity dataframe")

            # Get latest values
            latest = df.iloc[-1]
            fed_bs = latest['fed_bs']
            tga = latest['tga']
            rrp = latest['rrp']
            net_liq = latest['net_liquidity']

            # Calculate z-score using lookback window
            z_score = self._calculate_z_score(df['net_liquidity'])

            if z_score is None:
                z_score = 0.0
                self._logger.warning("Could not calculate z-score, using 0.0")

            # Classify regime
            signal, color, description, strength = self._classify_regime(z_score)

            # Build details dict
            latest_date = latest.get('date', datetime.now())
            if isinstance(latest_date, pd.Timestamp):
                latest_date_str = latest_date.strftime('%Y-%m-%d')
            else:
                latest_date_str = str(latest_date)

            details = {
                'formula': 'Fed BS - TGA - RRP',
                'latest_date': latest_date_str,
                'fed_bs_billions': float(fed_bs),
                'tga_billions': float(tga),
                'rrp_billions': float(rrp),
                'lookback_days': self.lookback_days,
                'data_points': len(df),
            }

            self._logger.info(
                f"Net Liquidity: {net_liq:.1f}B | Signal: {signal} | "
                f"Z-score: {z_score:+.2f} | Strength: {strength:.0f}"
            )

            return NetLiquiditySignal(
                signal=signal,
                strength=strength,
                net_liquidity_billions=float(net_liq),
                z_score=float(z_score),
                fed_bs_billions=float(fed_bs),
                tga_billions=float(tga),
                rrp_billions=float(rrp),
                regime_color=color,
                description=description,
                timestamp=datetime.now(),
                details=details,
            )

        except Exception as e:
            self._logger.error(f"Error in liquidity analysis: {e}")
            import traceback
            self._logger.error(traceback.format_exc())
            return self._unavailable_signal(f"Analysis error: {str(e)}")

    def _build_liquidity_dataframe(
        self,
        fed_bs_df: pd.DataFrame,
        tga_series: Optional[pd.Series],
        rrp_series: Optional[pd.Series],
    ) -> pd.DataFrame:
        """Build combined liquidity DataFrame with all components."""
        df = fed_bs_df.copy()

        # Ensure date is the index
        if 'date' in df.columns:
            df = df.set_index('date')

        # Convert Fed BS to billions (input is in millions)
        df['fed_bs'] = df['total_assets'] / 1000.0

        # Merge TGA (convert from millions to billions)
        if tga_series is not None and not tga_series.empty:
            df = df.join(tga_series.rename('tga_raw'), how='left')
            df['tga'] = df['tga_raw'].ffill().fillna(0) / 1000.0
        else:
            df['tga'] = 0.0
            self._logger.warning("TGA data unavailable, using 0")

        # Merge RRP (convert from millions to billions)
        if rrp_series is not None and not rrp_series.empty:
            df = df.join(rrp_series.rename('rrp_raw'), how='left')
            df['rrp'] = df['rrp_raw'].ffill().fillna(0) / 1000.0
        else:
            df['rrp'] = 0.0
            self._logger.warning("RRP data unavailable, using 0")

        # Calculate NET LIQUIDITY: Fed BS - TGA - RRP
        df['net_liquidity'] = df['fed_bs'] - df['tga'] - df['rrp']

        # Reset index to get date column back
        df = df.reset_index()

        return df

    def _calculate_z_score(self, series: pd.Series) -> Optional[float]:
        """Calculate z-score for latest value vs lookback period."""
        clean = series.dropna()

        if len(clean) < self.min_data_points:
            self._logger.warning(
                f"Insufficient data for z-score: {len(clean)} < {self.min_data_points}"
            )
            return None

        # Use lookback window
        lookback = clean.tail(self.lookback_days)

        mean = lookback.mean()
        std = lookback.std()

        if std == 0 or np.isnan(std):
            return None

        latest = clean.iloc[-1]
        return float((latest - mean) / std)

    def _classify_regime(self, z_score: float) -> tuple:
        """
        Classify liquidity regime based on z-score.

        Returns:
            tuple: (signal, color, description, strength)
        """
        if z_score > self.supportive_threshold:
            signal = "SUPPORTIVE"
            color = self._colors.supportive
            description = (
                "Net liquidity is expanding. Fed balance sheet growth outpacing "
                "TGA/RRP drains. Supportive environment for risk assets."
            )
            # Strength increases with z-score beyond threshold
            strength = min(100.0, 60 + (z_score - self.supportive_threshold) * 25)

        elif z_score < self.draining_threshold:
            signal = "DRAINING"
            color = self._colors.draining
            description = (
                "Net liquidity is contracting. QT + elevated TGA/RRP draining "
                "liquidity from markets. Headwind for risk assets."
            )
            strength = min(100.0, 60 + (-z_score - abs(self.draining_threshold)) * 25)

        else:
            signal = "NEUTRAL"
            color = self._colors.neutral
            description = (
                "Net liquidity near historical average. Neither strong tailwind "
                "nor headwind for markets."
            )
            # Strength decreases as we get closer to neutral (0)
            strength = max(20.0, 50 - abs(z_score) * 30)

        return signal, color, description, strength

    def _unavailable_signal(self, reason: str) -> NetLiquiditySignal:
        """Return signal indicating data unavailable."""
        self._logger.warning(f"Liquidity signal unavailable: {reason}")

        return NetLiquiditySignal(
            signal="UNAVAILABLE",
            strength=0.0,
            net_liquidity_billions=0.0,
            z_score=0.0,
            fed_bs_billions=0.0,
            tga_billions=0.0,
            rrp_billions=0.0,
            regime_color="#9E9E9E",  # Gray
            description=f"Data unavailable: {reason}",
            timestamp=datetime.now(),
            details={'error': reason},
        )


# ============================================================================
# LIQUIDITY SIGNAL GENERATOR - WRAPPER FOR BACKWARD COMPATIBILITY
# ============================================================================

class LiquiditySignalGenerator:
    """
    Backward-compatible wrapper for LiquidityAnalyzer.

    DEPRECATED: Use LiquidityAnalyzer directly instead.

    This class maintains the old interface but now requires Fed BS data
    for accurate calculations.
    """

    def __init__(self, lookback_days: int = 252):
        self._analyzer = LiquidityAnalyzer(lookback_days=lookback_days)
        self._logger = logging.getLogger(__name__)
        self._logger.warning(
            "LiquiditySignalGenerator is deprecated. Use LiquidityAnalyzer directly "
            "with Fed BS, TGA, and RRP data for accurate net liquidity calculations."
        )

    def generate_signal(
        self,
        liquidity_df: pd.DataFrame,
        fed_bs_df: Optional[pd.DataFrame] = None,
    ) -> NetLiquiditySignal:
        """
        Generate liquidity signal.

        Args:
            liquidity_df: DataFrame with 'rrp_on' and 'tga' columns
            fed_bs_df: DataFrame with Fed Balance Sheet data (REQUIRED for accuracy)

        Returns:
            NetLiquiditySignal

        Note:
            If fed_bs_df is not provided, this will return an UNAVAILABLE signal.
            The old formula -(RRP + TGA) has been removed as it was incorrect.
        """
        if fed_bs_df is None or fed_bs_df.empty:
            self._logger.error(
                "Fed Balance Sheet data required for accurate net liquidity. "
                "The old formula -(RRP + TGA) was incorrect and has been removed."
            )
            return self._analyzer._unavailable_signal(
                "Fed Balance Sheet data required. Old formula deprecated."
            )

        # Extract series from liquidity_df
        tga_series = None
        rrp_series = None

        if 'date' in liquidity_df.columns:
            if 'tga' in liquidity_df.columns:
                tga_series = liquidity_df.set_index('date')['tga']
            if 'rrp_on' in liquidity_df.columns:
                rrp_series = liquidity_df.set_index('date')['rrp_on']

        return self._analyzer.analyze(fed_bs_df, tga_series, rrp_series)


# ============================================================================
# LEGACY FUNCTION - DEPRECATED
# ============================================================================

def analyze_liquidity(
    fed_bs_df: pd.DataFrame,
    tga_series: Optional[pd.Series] = None,
    rrp_series: Optional[pd.Series] = None,
) -> dict:
    """
    Legacy function for backward compatibility.

    DEPRECATED: Use LiquidityAnalyzer.analyze() directly.

    Args:
        fed_bs_df: Fed Balance Sheet DataFrame (REQUIRED)
        tga_series: TGA Series (optional)
        rrp_series: RRP Series (optional)

    Returns:
        dict with signal information
    """
    analyzer = LiquidityAnalyzer()
    signal = analyzer.analyze(fed_bs_df, tga_series, rrp_series)

    return {
        'signal': signal.signal,
        'regime': signal.regime,
        'strength': signal.strength,
        'net_liquidity': signal.net_liquidity_billions,
        'net_liquidity_billions': signal.net_liquidity_billions,
        'z_score': signal.z_score,
        'fed_bs_billions': signal.fed_bs_billions,
        'tga_billions': signal.tga_billions,
        'rrp_billions': signal.rrp_billions,
        'regime_color': signal.regime_color,
        'description': signal.description,
        'details': signal.details,
    }
