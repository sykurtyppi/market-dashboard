"""
CTA Trend Engine - Continuous signal with vol targeting (institutional-grade)

Parameters loaded from config/parameters.yaml
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import logging

from config import cfg

logger = logging.getLogger(__name__)


@dataclass
class CtaConfig:
    """CTA engine configuration - loads defaults from config/parameters.yaml"""
    lookbacks: Tuple[int, ...] = field(default_factory=lambda: tuple(cfg.cta.lookbacks))
    lookback_weights: Optional[Tuple[float, ...]] = None
    vol_lookback: int = field(default_factory=lambda: cfg.cta.vol_lookback)
    vol_target_ann: float = field(default_factory=lambda: cfg.cta.target_vol)
    max_gross_leverage: float = field(default_factory=lambda: cfg.cta.max_gross_leverage)
    min_exposure_threshold: float = field(default_factory=lambda: cfg.cta.flat_threshold)
    clip_signal: float = 1.0

    def weights(self) -> np.ndarray:
        """Resolve lookback weights"""
        if self.lookback_weights is not None:
            w = np.asarray(self.lookback_weights, dtype=float)
            if len(w) != len(self.lookbacks):
                raise ValueError("lookback_weights must match lookbacks length")
            return w / w.sum()
        # Default: shorter horizons get more weight
        base = np.array([1.0 / np.sqrt(L) for L in self.lookbacks], dtype=float)
        return base / base.sum()


@dataclass
class CtaResult:
    """CTA engine outputs"""
    exposures: pd.DataFrame       # Scaled exposures per symbol
    signal: pd.DataFrame          # Raw trend signal
    vol_ann: pd.DataFrame         # Annualized volatility
    state: pd.DataFrame           # LONG/SHORT/FLAT per symbol
    latest_exposure: pd.Series    # Latest exposures
    latest_state: pd.Series       # Latest states
    flip_levels: pd.DataFrame     # Flip level table


class CtaEngine:
    """
    Multi-horizon CTA engine with continuous signals and vol targeting.
    More accurate than simple sign(momentum).
    """

    def __init__(self, config: Optional[CtaConfig] = None):
        self.cfg = config or CtaConfig()
        self.w = self.cfg.weights()

    def run(self, prices: pd.DataFrame) -> CtaResult:
        """Run CTA engine on daily close prices"""
        prices = self._sanitize(prices)
        rets = prices.pct_change(fill_method=None)

        # Annualized volatility
        vol_ann = rets.rolling(self.cfg.vol_lookback).std() * np.sqrt(252.0)
        
        # Continuous trend signal
        signal = self._continuous_trend_signal(prices, vol_ann)

        # Vol-targeted exposures
        vol_safe = vol_ann.replace(0.0, np.nan)
        exposures = signal * (self.cfg.vol_target_ann / vol_safe)
        exposures = exposures.fillna(0.0)

        # Cap gross leverage
        exposures = self._cap_gross(exposures, self.cfg.max_gross_leverage)

        # State labels
        state = exposures.map(
            lambda x: "FLAT" if abs(x) < self.cfg.min_exposure_threshold 
            else ("LONG" if x > 0 else "SHORT")
        )

        latest_date = exposures.index.max()
        latest_exposure = exposures.loc[latest_date]
        latest_state = state.loc[latest_date]

        flip_levels = self._flip_levels(prices)

        logger.info(f"CTA: Latest gross exposure = {latest_exposure.abs().sum():.2f}")

        return CtaResult(
            exposures=exposures,
            signal=signal,
            vol_ann=vol_ann,
            state=state,
            latest_exposure=latest_exposure,
            latest_state=latest_state,
            flip_levels=flip_levels,
        )

    @staticmethod
    def _sanitize(prices: pd.DataFrame) -> pd.DataFrame:
        """Ensure DatetimeIndex and sorted"""
        if not isinstance(prices.index, pd.DatetimeIndex):
            prices = prices.copy()
            prices.index = pd.to_datetime(prices.index)
        return prices.sort_index()

    def _continuous_trend_signal(
        self, 
        prices: pd.DataFrame, 
        vol_ann: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Continuous signal: log momentum / volatility, aggregated across horizons.
        
        For each lookback L:
          mom_L = log(P_t / P_{t-L})
          score_L = mom_L / (vol_ann * sqrt(L/252))
        
        Weighted sum -> tanh squashing -> clip to [-1, +1]
        """
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        for w, L in zip(self.w, self.cfg.lookbacks):
            # Log momentum
            mom = np.log(prices / prices.shift(L))
            
            # Scale by expected move over L days
            denom = vol_ann * np.sqrt(L / 252.0)
            score = mom / denom.replace(0.0, np.nan)
            score = score.replace([np.inf, -np.inf], np.nan).fillna(0.0)

            # Tanh squashing for stability
            sig = sig + w * np.tanh(score)

        sig = sig.clip(-self.cfg.clip_signal, self.cfg.clip_signal)
        return sig

    @staticmethod
    def _cap_gross(exposures: pd.DataFrame, max_gross: float) -> pd.DataFrame:
        """Cap gross leverage per day"""
        gross = exposures.abs().sum(axis=1)
        scale = pd.Series(1.0, index=exposures.index)
        
        too_big = gross > max_gross
        scale.loc[too_big] = max_gross / gross[too_big]
        
        return exposures.mul(scale, axis=0)

    def _flip_levels(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute flip levels for latest date.
        Flip level for horizon L = price at t-L (momentum zero crossing)
        """
        latest = prices.index.max()
        if pd.isna(latest):
            return pd.DataFrame(columns=[
                "symbol", "horizon_days", "flip_price", 
                "current_price", "distance_pct"
            ])

        idx = prices.index.get_loc(latest)
        out = []
        
        for sym in prices.columns:
            cur = prices.loc[latest, sym]
            if pd.isna(cur):
                continue
                
            for L in self.cfg.lookbacks:
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
