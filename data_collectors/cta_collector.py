"""
CTA Collector - Manages price updates and CTA calculations
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional
import logging

from data_collectors.cta_data_store import PriceDB, PriceDBConfig
from processors.cta_engine import CtaEngine, CtaConfig, CtaResult

logger = logging.getLogger(__name__)


class CTACollector:
    """Collects price data and runs CTA engine"""
    
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
    
    def __init__(self, db_path: str = "data/cta_prices.db", universe: Optional[List[str]] = None):
        self.db = PriceDB(PriceDBConfig(db_path=Path(db_path)))
        self.universe = universe or self.DEFAULT_UNIVERSE
        self.engine = CtaEngine(CtaConfig(
            lookbacks=(21, 63, 126, 252),
            vol_lookback=20,
            vol_target_ann=0.15,
            max_gross_leverage=2.0,
            min_exposure_threshold=0.03,
        ))
    
    def update_prices(self) -> None:
        """
        Update price database with latest data.
        Uses period='5y' for bootstrap, incremental updates after.
        """
        logger.info(f"📈 Updating prices for {len(self.universe)} symbols...")
        
        for symbol in self.universe:
            try:
                last_date = self.db.get_last_date(symbol)
                
                if last_date is None:
                    # Bootstrap: fetch 5 years
                    logger.info(f"  {symbol}: Bootstrapping 5 years...")
                    hist = yf.download(symbol, period="5y", interval="1d", progress=False, auto_adjust=False)
                else:
                    # Incremental: from day after last_date
                    start = pd.to_datetime(last_date) + pd.Timedelta(days=1)
                    
                    if start.date() >= datetime.now().date():
                        logger.info(f"  {symbol}: Already up to date ({last_date})")
                        continue
                    
                    logger.info(f"  {symbol}: Updating since {last_date}...")
                    hist = yf.download(
                        symbol,
                        start=start.strftime('%Y-%m-%d'),
                        interval="1d",
                        progress=False,
                        auto_adjust=False
                    )
                
                if not hist.empty:
                    self.db.upsert_ohlc(symbol, hist)
                else:
                    logger.warning(f"  {symbol}: No new data available")
                    
            except Exception as e:
                logger.error(f"  {symbol}: Error - {e}")
        
        logger.info("✓ Price update complete")
    
    def get_cta_analysis(self, start_date: Optional[str] = None) -> Optional[CtaResult]:
        """
        Run CTA engine on stored prices.
        
        Args:
            start_date: Start date (default: 3 years ago)
        
        Returns:
            CtaResult or None if insufficient data
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=1095)).strftime('%Y-%m-%d')
        
        logger.info("🔍 Running CTA analysis...")
        
        # Fetch from DB
        prices_tall = self.db.get_history(self.universe, start=start_date)
        
        if prices_tall.empty:
            logger.error("No price data available")
            return None
        
        # Pivot to wide: index=date, columns=symbol, values=adj_close
        prices_wide = prices_tall.pivot_table(
            index=prices_tall.index,
            columns="symbol",
            values="adj_close",
        ).sort_index()
        
        logger.info(
            f"Loaded {len(prices_wide)} days × {len(prices_wide.columns)} symbols"
        )
        
        # Run engine
        result = self.engine.run(prices_wide)
        
        logger.info(
            f"✓ CTA complete. Gross exposure: {result.latest_exposure.abs().sum():.2f}"
        )
        
        return result
    
    def get_summary(self) -> dict:
        """Get CTA positioning summary by asset class"""
        result = self.get_cta_analysis()
        
        if result is None:
            return {"error": "No data available"}
        
        # Asset class groupings
        equities = ["SPY", "QQQ", "IWM", "EEM"]
        bonds = ["TLT", "HYG"]
        commodities = ["GLD"]
        fx = ["UUP"]
        
        def safe_sum(symbols):
            return result.latest_exposure[[
                s for s in symbols if s in result.latest_exposure.index
            ]].sum()
        
        return {
            "equities_exposure": float(safe_sum(equities)),
            "bonds_exposure": float(safe_sum(bonds)),
            "commodities_exposure": float(safe_sum(commodities)),
            "fx_exposure": float(safe_sum(fx)),
            "total_gross_exposure": float(result.latest_exposure.abs().sum()),
            "latest_date": result.exposures.index.max().strftime('%Y-%m-%d'),
            "symbols": list(result.latest_exposure.index),
        }
    
    def close(self):
        """Close database connection"""
        self.db.close()
