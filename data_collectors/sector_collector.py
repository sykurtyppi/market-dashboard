"""
Sector Performance Collector
Tracks sector ETF performance for rotation analysis

Enhanced features:
- Multi-timeframe performance
- Relative strength vs SPY
- Sector rotation signals (cyclical vs defensive)
- Correlation analysis
- Risk-on/Risk-off indicator
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Sector classification for rotation analysis
CYCLICAL_SECTORS = ['XLK', 'XLY', 'XLF', 'XLI', 'XLB', 'XLC']  # Risk-on
DEFENSIVE_SECTORS = ['XLV', 'XLP', 'XLU', 'XLRE']  # Risk-off
ENERGY_SECTOR = ['XLE']  # Commodity-linked (special case)


class SectorCollector:
    """Collects sector performance data with enhanced rotation analysis"""

    def __init__(self):
        self.sectors = {
            'XLK': 'Technology',
            'XLF': 'Financials',
            'XLV': 'Healthcare',
            'XLE': 'Energy',
            'XLY': 'Consumer Discretionary',
            'XLP': 'Consumer Staples',
            'XLI': 'Industrials',
            'XLB': 'Materials',
            'XLU': 'Utilities',
            'XLRE': 'Real Estate',
            'XLC': 'Communication Services'
        }

        self.sector_categories = {
            'XLK': 'Cyclical',
            'XLF': 'Cyclical',
            'XLY': 'Cyclical',
            'XLI': 'Cyclical',
            'XLB': 'Cyclical',
            'XLC': 'Cyclical',
            'XLV': 'Defensive',
            'XLP': 'Defensive',
            'XLU': 'Defensive',
            'XLRE': 'Defensive',
            'XLE': 'Commodity',
        }

    def get_sector_performance(self, period: str = '1d') -> Dict:
        """
        Get sector performance
        period: '1d', '5d', '1mo', '3mo', 'ytd'
        """
        performance = {}

        for ticker, name in self.sectors.items():
            try:
                etf = yf.Ticker(ticker)
                hist = etf.history(period='5d')

                if len(hist) >= 2:
                    current = hist['Close'].iloc[-1]
                    previous = hist['Close'].iloc[-2]
                    change_pct = ((current - previous) / previous) * 100

                    performance[ticker] = {
                        'name': name,
                        'price': float(current),
                        'change_pct': float(change_pct),
                        'category': self.sector_categories.get(ticker, 'Other')
                    }
            except Exception as e:
                logger.warning(f"Error fetching {ticker}: {e}")
                continue

        return performance

    def get_sector_rankings(self) -> pd.DataFrame:
        """Get sector performance ranked"""
        perf = self.get_sector_performance()

        df = pd.DataFrame.from_dict(perf, orient='index')
        df = df.sort_values('change_pct', ascending=False)

        return df

    def get_multi_timeframe_performance(self) -> pd.DataFrame:
        """
        Get sector performance across multiple timeframes

        Returns:
            DataFrame with 1d, 5d, 1mo, 3mo, 6mo, YTD returns
        """
        tickers = list(self.sectors.keys()) + ['SPY']  # Include SPY for comparison
        periods = {'1d': 2, '5d': 6, '1mo': 25, '3mo': 65, '6mo': 130}

        try:
            # Fetch 6 months of data to cover all periods
            data = yf.download(tickers, period='6mo', progress=False)['Close']

            if data.empty:
                return pd.DataFrame()

            results = []
            for ticker in tickers:
                if ticker not in data.columns:
                    continue

                prices = data[ticker].dropna()
                if len(prices) < 2:
                    continue

                current = prices.iloc[-1]
                row = {
                    'ticker': ticker,
                    'name': self.sectors.get(ticker, 'S&P 500'),
                    'category': self.sector_categories.get(ticker, 'Benchmark'),
                    'price': current,
                }

                for period_name, days in periods.items():
                    if len(prices) >= days:
                        prev_price = prices.iloc[-days]
                        pct_change = ((current - prev_price) / prev_price) * 100
                        row[f'{period_name}_return'] = round(pct_change, 2)
                    else:
                        row[f'{period_name}_return'] = None

                # Calculate YTD
                ytd_start = prices[prices.index.year == datetime.now().year]
                if not ytd_start.empty:
                    ytd_first = ytd_start.iloc[0]
                    row['ytd_return'] = round(((current - ytd_first) / ytd_first) * 100, 2)
                else:
                    row['ytd_return'] = None

                results.append(row)

            df = pd.DataFrame(results)
            return df

        except Exception as e:
            logger.error(f"Error fetching multi-timeframe data: {e}")
            return pd.DataFrame()

    def get_relative_strength(self, period: str = '1mo') -> pd.DataFrame:
        """
        Calculate relative strength vs SPY

        Relative Strength = Sector Return - SPY Return
        Positive = Outperforming
        Negative = Underperforming

        Returns:
            DataFrame with relative strength metrics
        """
        df = self.get_multi_timeframe_performance()

        if df.empty:
            return pd.DataFrame()

        # Get SPY return as benchmark
        spy_row = df[df['ticker'] == 'SPY']
        if spy_row.empty:
            return df

        periods = ['1d_return', '5d_return', '1mo_return', '3mo_return', '6mo_return']

        for period in periods:
            spy_return = spy_row[period].iloc[0] if period in spy_row.columns else 0
            rs_col = f'{period.replace("_return", "")}_rs'
            df[rs_col] = df[period] - spy_return

        # Filter out SPY from results
        df = df[df['ticker'] != 'SPY']

        return df

    def get_rotation_signal(self) -> Dict:
        """
        Determine market rotation signal (Risk-On vs Risk-Off)

        Compares cyclical sector performance to defensive sectors.
        When cyclicals outperform defensives = Risk-On
        When defensives outperform cyclicals = Risk-Off

        Returns:
            Dict with rotation signal and supporting metrics
        """
        df = self.get_multi_timeframe_performance()

        if df.empty:
            return {'signal': 'Unknown', 'confidence': 0}

        # Calculate average returns by category
        cyclical_df = df[df['category'] == 'Cyclical']
        defensive_df = df[df['category'] == 'Defensive']

        signals = {}

        for period in ['1d_return', '5d_return', '1mo_return']:
            if period not in df.columns:
                continue

            cyc_avg = cyclical_df[period].mean() if not cyclical_df.empty else 0
            def_avg = defensive_df[period].mean() if not defensive_df.empty else 0

            diff = cyc_avg - def_avg
            signals[period] = {
                'cyclical_avg': round(cyc_avg, 2),
                'defensive_avg': round(def_avg, 2),
                'spread': round(diff, 2),
                'signal': 'Risk-On' if diff > 0 else 'Risk-Off'
            }

        # Determine overall signal (weighted by timeframe)
        weights = {'1d_return': 0.2, '5d_return': 0.3, '1mo_return': 0.5}
        weighted_spread = sum(
            signals.get(p, {}).get('spread', 0) * w
            for p, w in weights.items()
        )

        if weighted_spread > 1.0:
            overall_signal = 'Strong Risk-On'
            color = '#4CAF50'
        elif weighted_spread > 0:
            overall_signal = 'Mild Risk-On'
            color = '#8BC34A'
        elif weighted_spread > -1.0:
            overall_signal = 'Mild Risk-Off'
            color = '#FF9800'
        else:
            overall_signal = 'Strong Risk-Off'
            color = '#F44336'

        return {
            'signal': overall_signal,
            'color': color,
            'weighted_spread': round(weighted_spread, 2),
            'details': signals,
            'interpretation': self._interpret_rotation(overall_signal),
            'leading_sectors': self._get_leading_sectors(df),
            'lagging_sectors': self._get_lagging_sectors(df),
        }

    def _interpret_rotation(self, signal: str) -> str:
        """Interpret rotation signal"""
        interpretations = {
            'Strong Risk-On': 'Institutions rotating into growth/cyclicals - bullish equity environment',
            'Mild Risk-On': 'Slight preference for cyclicals - cautiously optimistic',
            'Mild Risk-Off': 'Slight preference for defensives - growing caution',
            'Strong Risk-Off': 'Flight to safety underway - reduce risk exposure',
        }
        return interpretations.get(signal, 'Mixed signals')

    def _get_leading_sectors(self, df: pd.DataFrame, n: int = 3) -> List[str]:
        """Get top N performing sectors"""
        if '1mo_return' not in df.columns:
            return []
        sorted_df = df.nlargest(n, '1mo_return')
        return [f"{row['name']} ({row['1mo_return']:+.1f}%)"
                for _, row in sorted_df.iterrows()]

    def _get_lagging_sectors(self, df: pd.DataFrame, n: int = 3) -> List[str]:
        """Get bottom N performing sectors"""
        if '1mo_return' not in df.columns:
            return []
        sorted_df = df.nsmallest(n, '1mo_return')
        return [f"{row['name']} ({row['1mo_return']:+.1f}%)"
                for _, row in sorted_df.iterrows()]

    def get_correlation_matrix(self, period: str = '3mo') -> Optional[pd.DataFrame]:
        """
        Calculate correlation matrix between sectors

        High correlation = Sectors move together
        Low/negative correlation = Diversification benefit

        Returns:
            Correlation matrix DataFrame
        """
        try:
            tickers = list(self.sectors.keys())
            data = yf.download(tickers, period=period, progress=False)['Close']

            if data.empty:
                return None

            # Calculate daily returns
            returns = data.pct_change().dropna()

            # Calculate correlation matrix
            corr_matrix = returns.corr()

            return corr_matrix

        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return None

    def get_sector_momentum_scores(self) -> pd.DataFrame:
        """
        Calculate momentum scores for each sector

        Combines multiple timeframe returns into a single momentum score.
        Higher score = Stronger momentum.

        Returns:
            DataFrame with momentum scores
        """
        df = self.get_multi_timeframe_performance()

        if df.empty:
            return pd.DataFrame()

        # Momentum score = weighted average of returns
        weights = {
            '1d_return': 0.1,
            '5d_return': 0.15,
            '1mo_return': 0.25,
            '3mo_return': 0.25,
            '6mo_return': 0.25,
        }

        df['momentum_score'] = 0
        for col, weight in weights.items():
            if col in df.columns:
                df['momentum_score'] += df[col].fillna(0) * weight

        df['momentum_score'] = df['momentum_score'].round(2)

        # Add momentum rating
        df['momentum_rating'] = df['momentum_score'].apply(self._rate_momentum)

        # Sort by momentum score
        df = df.sort_values('momentum_score', ascending=False)

        return df[df['ticker'] != 'SPY']  # Exclude benchmark

    def _rate_momentum(self, score: float) -> str:
        """Rate momentum score"""
        if score > 15:
            return 'Very Strong'
        elif score > 8:
            return 'Strong'
        elif score > 0:
            return 'Positive'
        elif score > -8:
            return 'Weak'
        else:
            return 'Very Weak'

    def get_sector_summary(self) -> Dict:
        """
        Get comprehensive sector analysis summary

        Returns:
            Dict with all key sector metrics
        """
        rotation = self.get_rotation_signal()
        momentum = self.get_sector_momentum_scores()

        if momentum.empty:
            return {'status': 'unavailable'}

        # Best and worst sectors
        best = momentum.iloc[0] if len(momentum) > 0 else None
        worst = momentum.iloc[-1] if len(momentum) > 0 else None

        # Count positive vs negative momentum
        positive = len(momentum[momentum['momentum_score'] > 0])
        negative = len(momentum[momentum['momentum_score'] <= 0])

        summary = {
            'rotation_signal': rotation['signal'],
            'rotation_color': rotation['color'],
            'rotation_spread': rotation['weighted_spread'],
            'best_sector': {
                'name': best['name'] if best is not None else 'N/A',
                'ticker': best['ticker'] if best is not None else 'N/A',
                'score': best['momentum_score'] if best is not None else 0,
            },
            'worst_sector': {
                'name': worst['name'] if worst is not None else 'N/A',
                'ticker': worst['ticker'] if worst is not None else 'N/A',
                'score': worst['momentum_score'] if worst is not None else 0,
            },
            'sectors_positive': positive,
            'sectors_negative': negative,
            'market_breadth_pct': round(positive / (positive + negative) * 100, 1) if (positive + negative) > 0 else 50,
            'interpretation': rotation['interpretation'],
        }

        return summary
