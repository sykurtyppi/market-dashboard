"""
S&P 500 Breadth PROXY Calculator

IMPORTANT: This is a PROXY measurement using 100 representative stocks,
NOT the true S&P 500 Advance-Decline Line which requires all 500 constituents.

Why this matters:
- True A/D Line measures participation across ALL 500 stocks
- This proxy samples ~20% of the index, biased toward larger caps
- During narrow rallies (mega-cap led), this proxy may show false breadth
- Use for directional guidance, not precise breadth measurement

For true S&P 500 breadth:
- Subscribe to NYSE market data (A/D Line, cumulative A/D)
- Or use $NYAD index from data providers

Limitations documented per data quality audit.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from utils.sp500_constituents import BREADTH_SAMPLE_SIZE, Constituents, load_constituents

logger = logging.getLogger(__name__)

class SP500ADLineCalculator:
    """
    PROXY breadth calculator using 100-stock sample.

    WARNING: This is NOT true S&P 500 breadth!
    - Uses 100 representative stocks (20% sample)
    - Biased toward mega-caps and liquid names
    - May diverge from true A/D Line during narrow rallies

    For true breadth, use NYSE $NYAD or professional data feeds.

    The 100-stock sample provides:
    - Faster calculation (~10x speed vs 500 stocks)
    - Good directional correlation with true breadth

    Membership: REPRESENTATIVE_STOCKS is a curated seed. At startup it is
    reconciled against the tracked constituent list
    (config/sp500_constituents.csv): any seed name no longer in the index is
    replaced by a current constituent from whichever sector the sample
    under-represents most, so the universe follows the index without a code
    edit. The sample size never changes — every downstream scale (McClellan
    thresholds, ad_diff history) depends on it.
    - Adequate for regime detection (risk-on/risk-off)

    NOT suitable for:
    - Precise breadth thrust signals
    - Exact A/D ratio calculations
    - Professional breadth divergence analysis
    """

    # Scaling factor: Sample represents ~20% of S&P 500
    # To estimate true S&P 500 breadth, multiply by 5x
    SAMPLE_SIZE = BREADTH_SAMPLE_SIZE
    INDEX_SIZE = 500
    SCALE_FACTOR = INDEX_SIZE / SAMPLE_SIZE  # 5.0

    # Flag to indicate this is proxy data
    IS_PROXY = True
    PROXY_DISCLAIMER = "100-stock proxy (not true S&P 500 breadth)"

    # 100 stocks across all sectors, market caps, and styles
    REPRESENTATIVE_STOCKS = [
        # Mega caps (20 stocks)
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
        'LLY', 'V', 'JPM', 'UNH', 'XOM', 'MA', 'PG', 'JNJ', 'HD', 'AVGO',
        'COST', 'ABBV',
        
        # Large caps (20 stocks)
        'MRK', 'CVX', 'BAC', 'ADBE', 'CRM', 'ORCL', 'WMT', 'KO', 'PEP',
        'CSCO', 'TMO', 'ACN', 'MCD', 'NFLX', 'ABT', 'DIS', 'AMD', 'INTC',
        'NKE', 'VZ',
        
        # Mid/Large (20 stocks)
        'CMCSA', 'TXN', 'UPS', 'RTX', 'HON', 'PM', 'NEE', 'QCOM', 'AMGN',
        'LOW', 'UNP', 'IBM', 'SPGI', 'GE', 'SBUX', 'CAT', 'AXP', 'BLK',
        'DE', 'MMM',
        
        # Mid caps (20 stocks)
        'GILD', 'MDT', 'TJX', 'CI', 'MDLZ', 'SYK', 'ISRG', 'ADI', 'VRTX',
        'ZTS', 'REGN', 'PLD', 'CB', 'DUK', 'SO', 'BSX', 'EOG', 'CME',
        'LRCX', 'MRSH',
        
        # Smaller/Cyclical (20 stocks)
        'ITW', 'NOC', 'AON', 'SHW', 'APH', 'MCO', 'ICE', 'GD', 'CL',
        'FIS', 'USB', 'PNC', 'TGT', 'F', 'AIG', 'MET', 'HCA', 'PSX',
        'MAR', 'CARR'
    ]
    
    def __init__(self, constituents: "Constituents | None" = None):
        self.constituents = constituents if constituents is not None else load_constituents()
        self.stocks, self.universe_changes = self.build_universe(
            self.REPRESENTATIVE_STOCKS, self.constituents
        )
        self.unresolved: list[str] = []
        if self.universe_changes["reconciled"]:
            logger.info(
                f"Initialized with {len(self.stocks)} stocks reconciled against the "
                f"S&P 500 list as of {self.universe_changes['as_of']}"
            )
            for dropped, added in zip(self.universe_changes["dropped"], self.universe_changes["added"]):
                logger.warning(f"Breadth sample: {dropped} is no longer a constituent; using {added}")
            if self.universe_changes["unreplaced"]:
                # The sample is now smaller than SAMPLE_SIZE, which every
                # downstream scale assumes. Say so loudly; do not hide it in
                # a shorter zip.
                logger.error(
                    f"Breadth sample is {len(self.stocks)} of {self.SAMPLE_SIZE}: no replacement "
                    f"available for {', '.join(self.universe_changes['unreplaced'])}"
                )
        else:
            logger.warning(
                f"Initialized with {len(self.stocks)} stocks from the unreconciled seed list — "
                "no constituent list available (run scripts/update_sp500_constituents.py)"
            )

    @staticmethod
    def build_universe(seed, constituents: "Constituents | None"):
        """Return (universe, changes): the seed with non-members replaced.

        Each dropped name is replaced by the alphabetically-first current
        constituent, not already in the sample, from the sector whose share of
        the sample lags its share of the index most. Deterministic for a given
        seed and list, so two processes agree on the universe.
        """
        seed = list(dict.fromkeys(seed))  # order-preserving dedupe: no name counts twice
        if constituents is None:
            return seed, {"reconciled": False, "dropped": [], "added": [], "unreplaced": [], "as_of": None}

        members = set(constituents.symbols)
        universe = [t for t in seed if t in members]
        dropped = [t for t in seed if t not in members]
        added: list[str] = []

        index_share = Counter(constituents.frame["gics_sector"])
        total_members = sum(index_share.values()) or 1
        for _ in dropped:
            have = Counter(constituents.sector_of(t) for t in universe)
            # Deficit: how many names a sector "should" have in a sample this
            # size versus how many it has. Largest deficit wins; ties by name.
            deficit = {
                sector: index_share[sector] / total_members * len(seed) - have.get(sector, 0)
                for sector in index_share
            }
            for sector in sorted(deficit, key=lambda k: (-deficit[k], k)):
                pool = constituents.frame[constituents.frame["gics_sector"] == sector]["symbol"]
                candidate = next((sym for sym in sorted(pool) if sym not in universe), None)
                if candidate:
                    universe.append(candidate)
                    added.append(candidate)
                    break

        # Replacement is attempted in order, and exhaustion is monotonic, so
        # any drops that found no candidate are the trailing suffix.
        unreplaced = dropped[len(added):]
        return universe, {
            "reconciled": True, "dropped": dropped, "added": added, "unreplaced": unreplaced,
            "as_of": constituents.as_of.isoformat(),
        }
    
    def fetch_stock_data(self, ticker, period='60d'):
        """Fetch data for single stock"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            if hist.empty:
                return None
            
            return {
                'ticker': ticker,
                'data': hist
            }
        except Exception as e:
            logger.warning(f"Failed to fetch {ticker}: {e}")
            return None
    
    def fetch_all_stocks(self, period='60d', max_workers=10):
        """Fetch data for all stocks in parallel"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.fetch_stock_data, ticker, period): ticker
                for ticker in self.stocks
            }
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results[result['ticker']] = result['data']
        
        self.unresolved = sorted(set(self.stocks) - set(results))
        logger.info(f"✅ Fetched {len(results)}/{len(self.stocks)} stocks")
        if self.unresolved:
            # A name that returns no bars is usually a symbol change or a
            # delisting the constituent list has not caught up with yet.
            logger.warning(f"Breadth sample: no price data for {', '.join(self.unresolved)}")
        return results
    
    def calculate_daily_breadth(self, stock_data):
        """
        Calculate advance/decline for each day
        Returns DataFrame with date, advancing, declining, ad_line
        """
        # Get all dates from the data
        all_dates = set()
        for ticker, df in stock_data.items():
            all_dates.update(df.index)
        
        dates = sorted(list(all_dates))
        
        breadth_data = []
        
        for date in dates:
            advancing = 0
            declining = 0
            unchanged = 0
            
            for ticker, df in stock_data.items():
                if date not in df.index:
                    continue
                
                # Get current and previous close
                try:
                    idx = df.index.get_loc(date)
                    if idx == 0:
                        continue  # Skip first day (no previous close)
                    
                    current = df.iloc[idx]['Close']
                    previous = df.iloc[idx - 1]['Close']
                    
                    if current > previous:
                        advancing += 1
                    elif current < previous:
                        declining += 1
                    else:
                        unchanged += 1
                        
                except Exception:
                    continue
            
            total = advancing + declining + unchanged
            
            if total > 0:
                breadth_data.append({
                    'date': date,
                    'advancing': advancing,
                    'declining': declining,
                    'unchanged': unchanged,
                    'total': total,
                    'ad_diff': advancing - declining,
                    'breadth_pct': (advancing / total) * 100
                })
        
        df = pd.DataFrame(breadth_data)
        
        if not df.empty:
            # Calculate cumulative A/D line
            df['ad_line'] = df['ad_diff'].cumsum()
            
            # Normalize to start at 10000 for easier visualization
            if len(df) > 0:
                df['ad_line'] = df['ad_line'] - df['ad_line'].iloc[0] + 10000
        
        return df
    
    def get_current_breadth(self):
        """Get current breadth snapshot"""
        logger.info("📊 Calculating current breadth...")
        
        # Fetch recent data
        stock_data = self.fetch_all_stocks(period='10d')
        
        if len(stock_data) < 50:
            logger.error(f"Only got {len(stock_data)} stocks - insufficient data")
            return None
        
        # Calculate breadth
        breadth_df = self.calculate_daily_breadth(stock_data)
        
        if breadth_df.empty:
            return None
        
        # Get latest day
        latest = breadth_df.iloc[-1]
        prev_week = breadth_df.iloc[-6] if len(breadth_df) >= 6 else breadth_df.iloc[0]
        
        return {
            'date': latest['date'],
            'advancing': int(latest['advancing']),
            'declining': int(latest['declining']),
            'unchanged': int(latest['unchanged']),
            'total': int(latest['total']),
            'breadth_pct': float(latest['breadth_pct']),
            'ad_line': float(latest['ad_line']),
            'week_change': float(latest['ad_line'] - prev_week['ad_line']),
            'trend': 'Improving' if latest['ad_line'] > prev_week['ad_line'] else 'Weakening',
            # Proxy metadata - important for data quality transparency
            'is_proxy': self.IS_PROXY,
            'sample_size': self.SAMPLE_SIZE,
            'disclaimer': self.PROXY_DISCLAIMER,
            # Estimated true S&P 500 values (scaled up from sample)
            'estimated_sp500_advancing': int(latest['advancing'] * self.SCALE_FACTOR),
            'estimated_sp500_declining': int(latest['declining'] * self.SCALE_FACTOR),
        }
    
    def get_breadth_history(self, days=60):
        """Get breadth history for charting"""
        logger.info(f"📈 Calculating {days}-day breadth history...")
        
        stock_data = self.fetch_all_stocks(period=f'{days}d')
        
        if len(stock_data) < 50:
            logger.error(f"Only got {len(stock_data)} stocks")
            return pd.DataFrame()
        
        breadth_df = self.calculate_daily_breadth(stock_data)
        
        return breadth_df
    
    def calculate_mcclellan_oscillator(self, breadth_df=None):
        """Calculate McClellan Oscillator from breadth data"""
        if breadth_df is None or breadth_df.empty:
            breadth_df = self.get_breadth_history(days=90)
        
        if len(breadth_df) < 39:
            logger.warning("Insufficient data for McClellan")
            return None
        
        # McClellan = (19-day EMA - 39-day EMA) of net advances
        breadth_df['ema19'] = breadth_df['ad_diff'].ewm(span=19, adjust=False).mean()
        breadth_df['ema39'] = breadth_df['ad_diff'].ewm(span=39, adjust=False).mean()
        breadth_df['mcclellan'] = breadth_df['ema19'] - breadth_df['ema39']
        
        latest = breadth_df.iloc[-1]
        value = float(latest['mcclellan'])
        
        # Interpretation
        if value > 50:
            signal = "Strong Bullish"
        elif value > 20:
            signal = "Bullish"
        elif value > -20:
            signal = "Neutral"
        elif value > -50:
            signal = "Bearish"
        else:
            signal = "Strong Bearish"
        
        return {
            'value': value,
            'signal': signal,
            'date': latest['date'],
            'history': breadth_df[['date', 'mcclellan']].tail(60)
        }

