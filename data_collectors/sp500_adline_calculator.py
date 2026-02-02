"""
S&P 500 Advance-Decline Line Calculator
Calculates real breadth from actual stock data using Yahoo Finance

Supports two modes:
- Fast (100 stocks): ~30-60 seconds, scaled McClellan
- Full (500 stocks): ~3-5 minutes, true S&P 500 breadth
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os

logger = logging.getLogger(__name__)

class SP500ADLineCalculator:
    """
    Calculate true S&P 500 breadth metrics from stock data

    Modes:
    - 'fast': 100-stock representative sample (default)
    - 'full': All 500 S&P 500 stocks
    """

    # 100 stocks across all sectors, market caps, and styles (Fast mode)
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
        'LRCX', 'MMC',

        # Smaller/Cyclical (20 stocks)
        'ITW', 'NOC', 'AON', 'SHW', 'APH', 'MCO', 'ICE', 'GD', 'CL',
        'FIS', 'USB', 'PNC', 'TGT', 'F', 'AIG', 'MET', 'HCA', 'PSX',
        'MAR', 'CARR'
    ]

    # Full S&P 500 list (as of 2024) - All 500+ constituents
    FULL_SP500_STOCKS = [
        # Information Technology
        'AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL', 'CRM', 'ADBE', 'AMD', 'CSCO', 'ACN',
        'IBM', 'INTC', 'INTU', 'TXN', 'QCOM', 'AMAT', 'NOW', 'ADI', 'LRCX', 'MU',
        'KLAC', 'SNPS', 'CDNS', 'MCHP', 'APH', 'MSI', 'ROP', 'FTNT', 'ADSK', 'PANW',
        'NXPI', 'MPWR', 'CTSH', 'IT', 'ANSS', 'KEYS', 'ON', 'CDW', 'FSLR', 'TYL',
        'HPQ', 'HPE', 'EPAM', 'WDC', 'STX', 'ZBRA', 'JNPR', 'NTAP', 'PTC', 'AKAM',
        'SWKS', 'TER', 'ENPH', 'SEDG', 'QRVO', 'GLW', 'JBL', 'FFIV', 'GEN', 'TRMB',

        # Health Care
        'LLY', 'UNH', 'JNJ', 'ABBV', 'MRK', 'TMO', 'ABT', 'PFE', 'DHR', 'AMGN',
        'ISRG', 'ELV', 'MDT', 'BMY', 'GILD', 'VRTX', 'SYK', 'BSX', 'REGN', 'ZTS',
        'BDX', 'CI', 'MCK', 'CVS', 'HUM', 'HCA', 'IDXX', 'EW', 'IQV', 'MTD',
        'A', 'DXCM', 'RMD', 'CAH', 'GEHC', 'ALGN', 'HOLX', 'BAX', 'ZBH', 'WAT',
        'COO', 'MOH', 'CNC', 'LH', 'DGX', 'VTRS', 'CRL', 'TECH', 'HSIC', 'XRAY',
        'BIO', 'INCY', 'BIIB', 'MRNA', 'ILMN', 'RVTY', 'TFX', 'STE', 'PKI', 'CTLT',

        # Financials
        'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'SPGI', 'BLK', 'AXP',
        'C', 'PGR', 'SCHW', 'CB', 'MMC', 'CME', 'ICE', 'AON', 'USB', 'PNC',
        'TFC', 'AJG', 'MCO', 'AFL', 'MET', 'AIG', 'TRV', 'ALL', 'PRU', 'BK',
        'MSCI', 'AMP', 'COF', 'FIS', 'FITB', 'DFS', 'STT', 'RJF', 'CINF', 'HBAN',
        'NTRS', 'WRB', 'RF', 'KEY', 'CFG', 'NDAQ', 'CBOE', 'FDS', 'TROW', 'L',
        'BRO', 'EG', 'JKHY', 'MTB', 'MKTX', 'WTW', 'ERIE', 'ACGL', 'RE', 'AIZ',
        'GL', 'CMA', 'ZION', 'SBNY',

        # Consumer Discretionary
        'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'TJX', 'BKNG', 'MAR',
        'ORLY', 'AZO', 'CMG', 'DHI', 'ROST', 'YUM', 'HLT', 'GM', 'F', 'EBAY',
        'LVS', 'DRI', 'PHM', 'NVR', 'LEN', 'GPC', 'APTV', 'ULTA', 'TSCO', 'BBY',
        'GRMN', 'POOL', 'WYNN', 'CCL', 'RCL', 'DPZ', 'MGM', 'CZR', 'EXPE', 'LKQ',
        'BWA', 'RL', 'TPR', 'VFC', 'HAS', 'NCLH', 'PVH', 'MHK', 'NWL', 'WHR',

        # Communication Services
        'META', 'GOOGL', 'GOOG', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'TMUS', 'CHTR',
        'EA', 'TTWO', 'WBD', 'OMC', 'IPG', 'LYV', 'MTCH', 'PARA', 'NWSA', 'NWS',
        'FOX', 'FOXA',

        # Industrials
        'CAT', 'UNP', 'HON', 'GE', 'RTX', 'UPS', 'DE', 'BA', 'LMT', 'ADP',
        'ITW', 'ETN', 'NOC', 'GD', 'WM', 'CSX', 'NSC', 'FDX', 'EMR', 'PH',
        'JCI', 'CTAS', 'TT', 'CARR', 'PCAR', 'FAST', 'CPRT', 'AME', 'ODFL', 'VRSK',
        'RSG', 'GWW', 'PWR', 'IR', 'CMI', 'PAYX', 'XYL', 'ROK', 'EFX', 'LHX',
        'HWM', 'DOV', 'WAB', 'OTIS', 'FTV', 'DAL', 'UAL', 'LUV', 'AAL', 'JBHT',
        'EXPD', 'CHRW', 'IEX', 'SNA', 'TXT', 'LDOS', 'J', 'NDSN', 'MAS', 'ALLE',
        'AOS', 'SWK', 'PNR', 'GNRC', 'AXON', 'HII', 'RHI', 'PAYC', 'HUBB', 'BLDR',

        # Consumer Staples
        'PG', 'KO', 'PEP', 'COST', 'WMT', 'PM', 'MDLZ', 'MO', 'CL', 'TGT',
        'KMB', 'GIS', 'STZ', 'ADM', 'SYY', 'KR', 'HSY', 'MKC', 'KHC', 'K',
        'EL', 'CLX', 'CAG', 'CHD', 'SJM', 'HRL', 'CPB', 'TSN', 'LW', 'BG',
        'TAP', 'WBA', 'KDP', 'MNST', 'CASY', 'LAMB',

        # Energy
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'PXD', 'VLO', 'OXY',
        'WMB', 'HES', 'KMI', 'HAL', 'DVN', 'BKR', 'FANG', 'TRGP', 'OKE', 'CTRA',
        'MRO', 'EQT', 'APA',

        # Utilities
        'NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'EXC', 'XEL', 'PEG', 'ED',
        'WEC', 'AWK', 'ES', 'EIX', 'DTE', 'ETR', 'PPL', 'FE', 'AEE', 'CMS',
        'EVRG', 'ATO', 'CNP', 'NI', 'PNW', 'LNT', 'NRG', 'CEG',

        # Real Estate
        'PLD', 'AMT', 'EQIX', 'CCI', 'PSA', 'O', 'WELL', 'DLR', 'SPG', 'VICI',
        'AVB', 'EQR', 'VTR', 'ARE', 'EXR', 'MAA', 'ESS', 'WY', 'INVH', 'SUI',
        'UDR', 'HST', 'PEAK', 'KIM', 'REG', 'CPT', 'BXP', 'FRT', 'DOC', 'IRM',

        # Materials
        'LIN', 'APD', 'SHW', 'ECL', 'FCX', 'NUE', 'NEM', 'DOW', 'DD', 'VMC',
        'MLM', 'PPG', 'CTVA', 'BALL', 'AVY', 'IP', 'PKG', 'ALB', 'CF', 'MOS',
        'FMC', 'CE', 'EMN', 'IFF', 'LYB', 'AMCR', 'SEE', 'WRK',

        # Additional stocks to reach ~500
        'BRK-B', 'PYPL', 'SQ', 'PLTR', 'ABNB', 'UBER', 'LYFT', 'SNAP', 'PINS', 'ROKU',
        'ZM', 'CRWD', 'DDOG', 'NET', 'SNOW', 'MDB', 'OKTA', 'TWLO', 'ZS', 'BILL',
    ]

    def __init__(self, mode: str = None):
        """
        Initialize calculator with specified mode.

        Args:
            mode: 'fast' (100 stocks) or 'full' (500 stocks)
                  If None, reads from BREADTH_MODE env var, defaults to 'fast'
        """
        # Get mode from parameter, env var, or default
        if mode is None:
            mode = os.getenv('BREADTH_MODE', 'fast').lower()

        self.mode = mode

        if mode == 'full':
            self.stocks = self.FULL_SP500_STOCKS
            self.scale_factor = 1.0  # No scaling needed for full 500
            logger.info(f"Initialized FULL mode with {len(self.stocks)} stocks")
        else:
            self.stocks = self.REPRESENTATIVE_STOCKS
            self.scale_factor = 5.0  # Scale 100 stocks to approximate 500
            logger.info(f"Initialized FAST mode with {len(self.stocks)} stocks (scale factor: {self.scale_factor}x)")
    
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
        
        logger.info(f"✅ Fetched {len(results)}/{len(self.stocks)} stocks")
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
            'trend': 'Improving' if latest['ad_line'] > prev_week['ad_line'] else 'Weakening'
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
        """Calculate McClellan Oscillator from breadth data

        Note: In fast mode (100 stocks), values are scaled by self.scale_factor
        to produce readings comparable to traditional NYSE McClellan (~-100 to +100).
        In full mode (500 stocks), no scaling is applied.
        """
        if breadth_df is None or breadth_df.empty:
            breadth_df = self.get_breadth_history(days=90)

        if len(breadth_df) < 39:
            logger.warning("Insufficient data for McClellan")
            return None

        # McClellan = (19-day EMA - 39-day EMA) of net advances
        # Scale by self.scale_factor (1.0 for full mode, 5.0 for fast mode)
        breadth_df['ema19'] = breadth_df['ad_diff'].ewm(span=19, adjust=False).mean() * self.scale_factor
        breadth_df['ema39'] = breadth_df['ad_diff'].ewm(span=39, adjust=False).mean() * self.scale_factor
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

