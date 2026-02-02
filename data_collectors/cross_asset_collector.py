"""
Cross-Asset Correlation Collector
Tracks correlations between major asset classes to identify regime changes

Key relationships:
- SPY/TLT: Stock-Bond correlation (negative = normal, positive = risk-off)
- SPY/GLD: Stock-Gold (negative = flight to safety)
- SPY/DXY (UUP): Stock-Dollar (negative = risk-on, positive = risk-off)
- TLT/GLD: Bonds-Gold (both safe havens)
- VIX/SPY: Fear gauge (always negative)

Regime detection:
- "Risk-On": Stocks up, bonds down, gold flat, dollar weak
- "Risk-Off": Stocks down, bonds up, gold up, dollar strong
- "Inflation Fear": Stocks down, bonds down, gold up
- "Deflation Fear": Everything down except dollar
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Core cross-asset tickers
CROSS_ASSET_TICKERS = {
    'SPY': 'S&P 500',
    'TLT': 'Long-Term Treasuries',
    'GLD': 'Gold',
    'UUP': 'US Dollar',
    'VIX': 'Volatility',
    'HYG': 'High Yield Bonds',
    'LQD': 'Investment Grade Bonds',
    'EEM': 'Emerging Markets',
    'USO': 'Oil',
    'DBC': 'Commodities',
}

# Key correlation pairs to monitor
KEY_PAIRS = [
    ('SPY', 'TLT', 'Stock-Bond'),
    ('SPY', 'GLD', 'Stock-Gold'),
    ('SPY', 'UUP', 'Stock-Dollar'),
    ('SPY', 'VIX', 'Stock-Fear'),
    ('TLT', 'GLD', 'Bond-Gold'),
    ('HYG', 'SPY', 'Credit-Stock'),
]


class CrossAssetCollector:
    """Tracks cross-asset correlations and regime signals"""

    def __init__(self):
        self.tickers = CROSS_ASSET_TICKERS
        self._cache = {}
        self._cache_time = None
        self._cache_ttl = 300  # 5 minute cache

    def get_cross_asset_data(self, period: str = '3mo') -> Optional[pd.DataFrame]:
        """
        Fetch price data for all cross-asset tickers

        Args:
            period: Lookback period ('1mo', '3mo', '6mo', '1y')

        Returns:
            DataFrame with daily prices
        """
        try:
            tickers = list(self.tickers.keys())

            # Handle VIX separately (use ^VIX)
            download_tickers = [t if t != 'VIX' else '^VIX' for t in tickers]

            data = yf.download(download_tickers, period=period, progress=False)['Close']

            if data.empty:
                return None

            # Rename ^VIX back to VIX
            if '^VIX' in data.columns:
                data = data.rename(columns={'^VIX': 'VIX'})

            return data

        except Exception as e:
            logger.error(f"Error fetching cross-asset data: {e}")
            return None

    def get_correlation_matrix(self, period: str = '3mo') -> Optional[pd.DataFrame]:
        """
        Calculate correlation matrix for all assets

        Returns:
            Correlation matrix DataFrame
        """
        data = self.get_cross_asset_data(period)

        if data is None or data.empty:
            return None

        # Calculate daily returns
        returns = data.pct_change().dropna()

        # Calculate correlation
        corr_matrix = returns.corr()

        return corr_matrix

    def get_key_correlations(self, period: str = '3mo') -> List[Dict]:
        """
        Get correlations for key asset pairs

        Returns:
            List of dicts with pair correlations and interpretations
        """
        data = self.get_cross_asset_data(period)

        if data is None or data.empty:
            return []

        returns = data.pct_change().dropna()
        results = []

        for ticker1, ticker2, pair_name in KEY_PAIRS:
            if ticker1 in returns.columns and ticker2 in returns.columns:
                corr = returns[ticker1].corr(returns[ticker2])

                # Interpret correlation
                interpretation = self._interpret_correlation(ticker1, ticker2, corr)

                results.append({
                    'pair': pair_name,
                    'ticker1': ticker1,
                    'ticker2': ticker2,
                    'correlation': round(corr, 3),
                    'strength': self._correlation_strength(corr),
                    'interpretation': interpretation,
                    'color': self._correlation_color(corr),
                })

        return results

    def _interpret_correlation(self, t1: str, t2: str, corr: float) -> str:
        """Interpret specific pair correlation"""
        pair = f"{t1}-{t2}"

        interpretations = {
            'SPY-TLT': {
                'positive': 'Unusual: Stocks & bonds moving together - potential regime shift',
                'negative': 'Normal: Classic risk-on/risk-off relationship intact',
                'neutral': 'Decoupling: Assets moving independently',
            },
            'SPY-GLD': {
                'positive': 'Inflation hedge mode: Both assets as inflation protection',
                'negative': 'Flight to safety: Gold up when stocks down (normal)',
                'neutral': 'Decoupled: Gold not reacting to equity moves',
            },
            'SPY-UUP': {
                'positive': 'Risk-off: Dollar strengthening with stocks (unusual)',
                'negative': 'Normal: Weak dollar supports stocks (risk-on)',
                'neutral': 'Decoupled: Dollar and stocks independent',
            },
            'SPY-VIX': {
                'positive': 'Broken relationship - very unusual!',
                'negative': 'Normal: VIX rises when stocks fall',
                'neutral': 'Low volatility regime',
            },
            'TLT-GLD': {
                'positive': 'Both safe havens bid: Strong risk-off sentiment',
                'negative': 'Divergence: One safe haven preferred over other',
                'neutral': 'Mixed signals in safe haven assets',
            },
            'HYG-SPY': {
                'positive': 'Normal: Credit and stocks move together',
                'negative': 'Divergence: Credit stress not reflected in stocks (warning)',
                'neutral': 'Credit market sending mixed signals',
            },
        }

        pair_interp = interpretations.get(pair, {
            'positive': 'Positive correlation',
            'negative': 'Negative correlation',
            'neutral': 'No significant correlation',
        })

        if corr > 0.3:
            return pair_interp['positive']
        elif corr < -0.3:
            return pair_interp['negative']
        else:
            return pair_interp['neutral']

    def _correlation_strength(self, corr: float) -> str:
        """Classify correlation strength"""
        abs_corr = abs(corr)
        if abs_corr > 0.7:
            return 'Very Strong'
        elif abs_corr > 0.5:
            return 'Strong'
        elif abs_corr > 0.3:
            return 'Moderate'
        elif abs_corr > 0.1:
            return 'Weak'
        else:
            return 'None'

    def _correlation_color(self, corr: float) -> str:
        """Get color for correlation value"""
        if corr > 0.5:
            return '#4CAF50'  # Strong positive - green
        elif corr > 0.2:
            return '#8BC34A'  # Weak positive - light green
        elif corr > -0.2:
            return '#9E9E9E'  # Neutral - gray
        elif corr > -0.5:
            return '#FF9800'  # Weak negative - orange
        else:
            return '#F44336'  # Strong negative - red

    def get_regime_signal(self, period: str = '1mo') -> Dict:
        """
        Determine current market regime based on cross-asset behavior

        Regimes:
        - RISK_ON: Stocks up, bonds down, dollar weak
        - RISK_OFF: Stocks down, bonds up, dollar strong
        - INFLATION: Stocks/bonds down, commodities up
        - DEFLATION: Everything down, dollar up
        - GOLDILOCKS: Stocks up, bonds flat, low vol

        Returns:
            Dict with regime classification and supporting data
        """
        data = self.get_cross_asset_data(period)

        if data is None or data.empty:
            return {'regime': 'UNKNOWN', 'confidence': 0}

        # Calculate returns over period
        returns = {}
        for ticker in ['SPY', 'TLT', 'GLD', 'UUP', 'VIX']:
            if ticker in data.columns:
                series = data[ticker].dropna()
                if len(series) >= 2:
                    returns[ticker] = (series.iloc[-1] / series.iloc[0] - 1) * 100

        if not returns:
            return {'regime': 'UNKNOWN', 'confidence': 0}

        # Classify regime
        spy_ret = returns.get('SPY', 0)
        tlt_ret = returns.get('TLT', 0)
        gld_ret = returns.get('GLD', 0)
        uup_ret = returns.get('UUP', 0)
        vix_ret = returns.get('VIX', 0)

        # Regime scoring
        scores = {
            'RISK_ON': 0,
            'RISK_OFF': 0,
            'INFLATION': 0,
            'DEFLATION': 0,
            'GOLDILOCKS': 0,
        }

        # Risk-On indicators
        if spy_ret > 2:
            scores['RISK_ON'] += 2
        if tlt_ret < -1:
            scores['RISK_ON'] += 1
        if uup_ret < -1:
            scores['RISK_ON'] += 1
        if vix_ret < -10:
            scores['RISK_ON'] += 1

        # Risk-Off indicators
        if spy_ret < -2:
            scores['RISK_OFF'] += 2
        if tlt_ret > 2:
            scores['RISK_OFF'] += 1
        if gld_ret > 2:
            scores['RISK_OFF'] += 1
        if vix_ret > 20:
            scores['RISK_OFF'] += 1

        # Inflation indicators
        if spy_ret < 0 and tlt_ret < 0:
            scores['INFLATION'] += 2
        if gld_ret > 3:
            scores['INFLATION'] += 2

        # Deflation indicators
        if spy_ret < -3 and tlt_ret < 0 and gld_ret < 0:
            scores['DEFLATION'] += 3
        if uup_ret > 2:
            scores['DEFLATION'] += 1

        # Goldilocks indicators
        if spy_ret > 1 and abs(tlt_ret) < 2 and abs(vix_ret) < 10:
            scores['GOLDILOCKS'] += 3

        # Determine regime
        max_score = max(scores.values())
        if max_score == 0:
            regime = 'MIXED'
            confidence = 0
        else:
            regime = max(scores, key=scores.get)
            confidence = min(100, max_score * 20)

        # Get regime details
        regime_info = {
            'RISK_ON': {
                'color': '#4CAF50',
                'emoji': '🚀',
                'description': 'Risk appetite strong - favor equities over bonds',
            },
            'RISK_OFF': {
                'color': '#F44336',
                'emoji': '🛡️',
                'description': 'Flight to safety - favor bonds and gold',
            },
            'INFLATION': {
                'color': '#FF9800',
                'emoji': '🔥',
                'description': 'Inflation fears - commodities and TIPS favored',
            },
            'DEFLATION': {
                'color': '#9C27B0',
                'emoji': '❄️',
                'description': 'Deflation/crisis mode - cash and short-term bonds',
            },
            'GOLDILOCKS': {
                'color': '#2196F3',
                'emoji': '✨',
                'description': 'Ideal conditions - balanced growth with low volatility',
            },
            'MIXED': {
                'color': '#9E9E9E',
                'emoji': '🤷',
                'description': 'No clear regime - mixed signals across assets',
            },
        }

        info = regime_info.get(regime, regime_info['MIXED'])

        return {
            'regime': regime,
            'confidence': confidence,
            'color': info['color'],
            'emoji': info['emoji'],
            'description': info['description'],
            'returns': returns,
            'scores': scores,
        }

    def get_rolling_correlations(self, ticker1: str, ticker2: str,
                                  window: int = 20, period: str = '6mo') -> Optional[pd.DataFrame]:
        """
        Calculate rolling correlation between two assets

        Useful for spotting regime changes when correlation breaks down

        Returns:
            DataFrame with rolling correlation
        """
        data = self.get_cross_asset_data(period)

        if data is None or ticker1 not in data.columns or ticker2 not in data.columns:
            return None

        returns = data[[ticker1, ticker2]].pct_change().dropna()

        # Calculate rolling correlation
        rolling_corr = returns[ticker1].rolling(window).corr(returns[ticker2])

        result = pd.DataFrame({
            'date': returns.index,
            'correlation': rolling_corr.values,
        })

        return result.dropna()

    def get_asset_performance_summary(self, period: str = '1mo') -> Dict:
        """
        Get performance summary for all tracked assets

        Returns:
            Dict with performance data for dashboard display
        """
        data = self.get_cross_asset_data(period)

        if data is None or data.empty:
            return {}

        summary = {}
        for ticker, name in self.tickers.items():
            if ticker in data.columns:
                series = data[ticker].dropna()
                if len(series) >= 2:
                    current = series.iloc[-1]
                    start = series.iloc[0]
                    pct_change = (current / start - 1) * 100

                    summary[ticker] = {
                        'name': name,
                        'price': round(current, 2),
                        'change_pct': round(pct_change, 2),
                        'color': '#4CAF50' if pct_change > 0 else '#F44336',
                    }

        return summary
