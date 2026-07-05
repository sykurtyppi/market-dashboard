"""Cached market-data fetch helpers shared across dashboard pages."""
import logging
from datetime import datetime

import pandas as pd
import streamlit as st
import yfinance as yf

from data_collectors.cboe_collector import CBOECollector
from data_collectors.yahoo_collector import YahooCollector
from processors.vrp_module import VRPAnalyzer

logger = logging.getLogger(__name__)


@st.cache_data(ttl=300)
def get_vrp_analysis_cached():
    """Cached wrapper for VRP analysis (5 min TTL)"""
    try:
        yahoo = YahooCollector()
        vix = yahoo.get_vix()
        analyzer = VRPAnalyzer(lookback_days=21)
        
        if vix is not None:
            return analyzer.get_complete_analysis(vix=vix)
        else:
            return analyzer.get_complete_analysis()
    except Exception as e:
        logger.error(f"Failed to get VRP analysis: {e}")
        return {"error": str(e)}
@st.cache_data(ttl=300)
def get_vrp_history_cached(days: int = 180):
    """Cached wrapper for VRP historical data (5 min TTL)"""
    try:
        analyzer = VRPAnalyzer(lookback_days=21)
        history = analyzer.get_historical_vrp(days=days)
        
        if history.empty:
            logger.warning(f"VRP history returned empty for {days} days")
        else:
            logger.info(f"Successfully loaded VRP history: {len(history)} rows")
        
        return history
    except Exception as e:
        logger.error(f"Failed to get VRP history: {e}")
        return pd.DataFrame()
@st.cache_data(ttl=300)
def get_vix_term_structure():
    """
    Build VIX term structure similar to vixcentral.com.
    Uses VIX indices at different tenors for accurate contango/backwardation view.
    Returns multiple points across the volatility curve.
    """
    try:
        term_points = []

        # Get VIX spot from CBOE
        cboe = CBOECollector()
        vix_spot = cboe.get_vix()

        # VIX1D - 1-day implied volatility
        try:
            vix1d_ticker = yf.Ticker("^VIX1D")
            vix1d_data = vix1d_ticker.history(period="5d")
            if not vix1d_data.empty:
                vix1d = float(vix1d_data['Close'].iloc[-1])
                if 5 < vix1d < 150:
                    term_points.append({
                        "Maturity": "1D",
                        "Days": 1,
                        "VIX Level": vix1d
                    })
        except Exception as e:
            logger.debug(f"VIX1D fetch failed (non-critical): {e}")

        # VIX9D - 9-day implied volatility
        try:
            vix9d = cboe.get_vix9d()
            if vix9d is not None:
                term_points.append({
                    "Maturity": "9D",
                    "Days": 9,
                    "VIX Level": float(vix9d)
                })
        except Exception as e:
            logger.debug(f"VIX9D fetch failed (non-critical): {e}")

        # VIX (30-day)
        if vix_spot is not None:
            term_points.append({
                "Maturity": "VIX",
                "Days": 30,
                "VIX Level": float(vix_spot)
            })

        # VIX3M - 3-month implied volatility
        try:
            vix3m = cboe.get_vix3m()
            if vix3m is not None:
                term_points.append({
                    "Maturity": "3M",
                    "Days": 90,
                    "VIX Level": float(vix3m)
                })
        except Exception as e:
            logger.debug(f"VIX3M fetch failed (non-critical): {e}")

        # VIX6M - 6-month implied volatility
        try:
            vix6m_ticker = yf.Ticker("^VIX6M")
            vix6m_data = vix6m_ticker.history(period="5d")
            if not vix6m_data.empty:
                vix6m = float(vix6m_data['Close'].iloc[-1])
                if 5 < vix6m < 100:
                    term_points.append({
                        "Maturity": "6M",
                        "Days": 180,
                        "VIX Level": vix6m
                    })
        except Exception as e:
            logger.debug(f"VIX6M fetch failed (non-critical): {e}")

        # VIX1Y - 1-year implied volatility
        try:
            vix1y_ticker = yf.Ticker("^VIX1Y")
            vix1y_data = vix1y_ticker.history(period="5d")
            if not vix1y_data.empty:
                vix1y = float(vix1y_data['Close'].iloc[-1])
                if 5 < vix1y < 80:
                    term_points.append({
                        "Maturity": "1Y",
                        "Days": 365,
                        "VIX Level": vix1y
                    })
        except Exception as e:
            logger.debug(f"VIX1Y fetch failed (non-critical): {e}")

        if not term_points:
            return pd.DataFrame()

        # Create DataFrame and sort by days
        df = pd.DataFrame(term_points)
        df = df.sort_values("Days").reset_index(drop=True)

        return df[["Maturity", "VIX Level"]]

    except Exception as e:
        logger.warning(f"Error building VIX term structure: {e}")
        return pd.DataFrame()
@st.cache_data(ttl=300)
def get_sector_performance(period: str = "1d") -> pd.DataFrame:
    """Get sector ETF performance for different time periods."""
    sectors = {
        "XLK": "Technology",
        "XLF": "Financials",
        "XLV": "Healthcare",
        "XLE": "Energy",
        "XLY": "Consumer Disc",
        "XLP": "Consumer Staples",
        "XLI": "Industrials",
        "XLB": "Materials",
        "XLU": "Utilities",
        "XLRE": "Real Estate",
        "XLC": "Communication",
    }

    performance = []
    period_days = {
        "1d": 2,
        "5d": 6,
        "1mo": 30,
        "3mo": 90,
        "6mo": 180,
        "1y": 365,
        "5y": 1825,
        "ytd": (datetime.now() - datetime(datetime.now().year, 1, 1)).days + 5,
    }

    days = period_days.get(period, 2)

    for ticker, name in sectors.items():
        try:
            etf = yf.Ticker(ticker)
            hist = etf.history(period=f"{days}d")

            if len(hist) >= 2:
                start_price = hist["Close"].iloc[0]
                end_price = hist["Close"].iloc[-1]
                change = ((end_price / start_price) - 1) * 100
                returns = hist["Close"].pct_change().dropna()
                volatility = returns.std() * 100

                performance.append({
                    "Sector": name,
                    "Ticker": ticker,
                    "Change %": float(change),
                    "Price": float(end_price),
                    "Volatility": float(volatility),
                    "Start Price": float(start_price),
                })
        except Exception as e:
            logger.warning(f"Error fetching {ticker}: {e}")
            continue

    df = pd.DataFrame(performance)
    if not df.empty:
        df = df.sort_values("Change %", ascending=False)
    return df
@st.cache_data(ttl=300)
def get_sector_comparison_chart(period: str = "1y") -> pd.DataFrame:
    """Get historical sector performance for comparison (normalized base 100)."""
    sectors = {
        "XLK": "Technology",
        "XLF": "Financials",
        "XLV": "Healthcare",
        "XLE": "Energy",
        "XLY": "Consumer Disc",
        "XLP": "Consumer Staples",
        "XLI": "Industrials",
        "XLB": "Materials",
        "XLU": "Utilities",
        "XLRE": "Real Estate",
        "XLC": "Communication",
    }

    all_data = {}

    for ticker, name in sectors.items():
        try:
            etf = yf.Ticker(ticker)
            hist = etf.history(period=period)

            if not hist.empty:
                normalized = (hist["Close"] / hist["Close"].iloc[0]) * 100
                all_data[name] = normalized
        except Exception as e:
            logger.debug(f"Failed to fetch {ticker} for credit ETF comparison: {e}")
            continue

    if all_data:
        return pd.DataFrame(all_data)
    return pd.DataFrame()
