"""
CTA Collector (Cloud Compatible) - Works without persistent database
Fetches price data directly from Yahoo Finance for Streamlit Cloud deployment.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CTAResultCloud:
    """Simplified CTA result for cloud deployment"""
    exposures: pd.DataFrame       # Historical exposures
    latest_exposure: pd.Series    # Latest exposures per symbol
    latest_state: Dict[str, str]  # LONG/SHORT/FLAT per symbol
    flip_levels: pd.DataFrame     # Flip level table
    summary: Dict                  # Summary metrics


class CTACollectorCloud:
    """
    Cloud-compatible CTA collector that fetches data on-demand.
    No persistent database required - works on Streamlit Cloud.
    """

    # Default CTA universe
    DEFAULT_UNIVERSE = [
        "SPY",   # US Large Cap
        "QQQ",   # US Tech
        "IWM",   # US Small Cap
        "TLT",   # US Treasuries 20Y+
        "GLD",   # Gold
        "UUP",   # US Dollar
        "EEM",   # Emerging Markets
        "HYG",   # High Yield Bonds
    ]

    # Lookback periods for trend calculation (days)
    LOOKBACKS = (21, 63, 126, 252)

    # Vol targeting parameters
    VOL_LOOKBACK = 20
    VOL_TARGET_ANN = 0.15
    MAX_GROSS_LEVERAGE = 2.0
    FLAT_THRESHOLD = 0.03

    def __init__(self, universe: Optional[List[str]] = None):
        self.universe = universe or self.DEFAULT_UNIVERSE
        # Compute lookback weights (shorter horizons get more weight)
        base = np.array([1.0 / np.sqrt(L) for L in self.LOOKBACKS])
        self.weights = base / base.sum()

    def fetch_prices(self, period: str = "2y") -> pd.DataFrame:
        """
        Fetch historical prices from Yahoo Finance.

        Args:
            period: Yahoo Finance period string (e.g., "1y", "2y", "5y")

        Returns:
            DataFrame with adjusted close prices (columns=symbols, index=date)
        """
        logger.info(f"Fetching {period} of price data for {len(self.universe)} symbols...")

        try:
            # Download all symbols at once (more efficient)
            data = yf.download(
                self.universe,
                period=period,
                interval="1d",
                progress=False,
                auto_adjust=True,  # Use adjusted prices
                threads=True
            )

            # Handle single vs multiple symbols
            if len(self.universe) == 1:
                prices = data[['Close']].rename(columns={'Close': self.universe[0]})
            else:
                prices = data['Close']

            # Drop rows with all NaN
            prices = prices.dropna(how='all')

            logger.info(f"Fetched {len(prices)} days of price data")
            return prices

        except Exception as e:
            logger.error(f"Error fetching prices: {e}")
            return pd.DataFrame()

    def calculate_exposures(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate CTA exposures using momentum + vol targeting.

        Returns:
            DataFrame of scaled exposures per symbol
        """
        if prices.empty:
            return pd.DataFrame()

        # Daily returns
        rets = prices.pct_change()

        # Annualized volatility
        vol_ann = rets.rolling(self.VOL_LOOKBACK).std() * np.sqrt(252.0)

        # Continuous trend signal (multi-horizon)
        signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        for w, L in zip(self.weights, self.LOOKBACKS):
            # Log momentum
            mom = np.log(prices / prices.shift(L))

            # Scale by expected move over L days
            denom = vol_ann * np.sqrt(L / 252.0)
            score = mom / denom.replace(0.0, np.nan)
            score = score.replace([np.inf, -np.inf], np.nan).fillna(0.0)

            # Tanh squashing for stability
            signal = signal + w * np.tanh(score)

        signal = signal.clip(-1.0, 1.0)

        # Vol-targeted exposures
        vol_safe = vol_ann.replace(0.0, np.nan)
        exposures = signal * (self.VOL_TARGET_ANN / vol_safe)
        exposures = exposures.fillna(0.0)

        # Cap gross leverage
        gross = exposures.abs().sum(axis=1)
        scale = pd.Series(1.0, index=exposures.index)
        too_big = gross > self.MAX_GROSS_LEVERAGE
        scale.loc[too_big] = self.MAX_GROSS_LEVERAGE / gross[too_big]
        exposures = exposures.mul(scale, axis=0)

        return exposures

    def calculate_flip_levels(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate flip levels - prices where momentum turns.
        """
        if prices.empty:
            return pd.DataFrame()

        latest = prices.index.max()
        idx = prices.index.get_loc(latest)
        out = []

        for sym in prices.columns:
            cur = prices.loc[latest, sym]
            if pd.isna(cur):
                continue

            for L in self.LOOKBACKS:
                if idx - L < 0:
                    continue

                flip = prices.iloc[idx - L][sym]
                if pd.isna(flip) or flip == 0:
                    continue

                dist = (cur / flip - 1.0) * 100.0
                out.append({
                    "symbol": sym,
                    "horizon_days": L,
                    "flip_price": float(flip),
                    "current_price": float(cur),
                    "distance_pct": float(dist),
                })

        return pd.DataFrame(out)

    def get_cta_analysis(self, period: str = "2y") -> Optional[CTAResultCloud]:
        """
        Run full CTA analysis.

        Args:
            period: Historical period to analyze

        Returns:
            CTAResultCloud or None if insufficient data
        """
        # Fetch prices
        prices = self.fetch_prices(period)

        if prices.empty or len(prices) < max(self.LOOKBACKS) + 10:
            logger.warning("Insufficient price data for CTA analysis")
            return None

        # Calculate exposures
        exposures = self.calculate_exposures(prices)

        if exposures.empty:
            return None

        # Latest values
        latest_exposure = exposures.iloc[-1]

        # State labels
        latest_state = {}
        for sym in latest_exposure.index:
            exp = latest_exposure[sym]
            if abs(exp) < self.FLAT_THRESHOLD:
                latest_state[sym] = "FLAT"
            elif exp > 0:
                latest_state[sym] = "LONG"
            else:
                latest_state[sym] = "SHORT"

        # Flip levels
        flip_levels = self.calculate_flip_levels(prices)

        # Summary
        equities = ["SPY", "QQQ", "IWM", "EEM"]
        bonds = ["TLT", "HYG"]
        commodities = ["GLD"]
        fx = ["UUP"]

        def safe_sum(symbols):
            return latest_exposure[[s for s in symbols if s in latest_exposure.index]].sum()

        summary = {
            "equities_exposure": float(safe_sum(equities)),
            "bonds_exposure": float(safe_sum(bonds)),
            "commodities_exposure": float(safe_sum(commodities)),
            "fx_exposure": float(safe_sum(fx)),
            "total_gross_exposure": float(latest_exposure.abs().sum()),
            "latest_date": exposures.index.max().strftime('%Y-%m-%d'),
            "symbols": list(latest_exposure.index),
        }

        logger.info(f"CTA analysis complete. Gross exposure: {summary['total_gross_exposure']:.2f}")

        return CTAResultCloud(
            exposures=exposures,
            latest_exposure=latest_exposure,
            latest_state=latest_state,
            flip_levels=flip_levels,
            summary=summary
        )
