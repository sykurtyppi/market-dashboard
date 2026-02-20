"""
CFTC Commitments of Traders (COT) Data Collector

Fetches weekly institutional futures positioning data from CFTC.
This data shows how large speculators (hedge funds) and commercials (hedgers)
are positioned in key markets.

Key Markets Tracked:
- S&P 500 E-mini (ES) - Equity positioning
- 10-Year Treasury Notes (TY) - Bond positioning
- Gold (GC) - Safe haven positioning
- Crude Oil (CL) - Energy/risk sentiment
- US Dollar Index (DX) - Currency positioning
- VIX Futures - Volatility positioning

Data Sources (in priority order):
1. cot_reports library - Reliable, handles CFTC format changes
2. Nasdaq Data Link (Quandl) - Fast, requires free API key
3. Direct CFTC downloads - Fallback

Install: pip install cot_reports
"""

import logging
import os
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from io import StringIO
from dotenv import load_dotenv
import re
import json
from pathlib import Path

# Try to import cot_reports library
try:
    import cot_reports as cot
    COT_LIBRARY_AVAILABLE = True
except ImportError:
    COT_LIBRARY_AVAILABLE = False

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class COTCollector:
    """
    Collects CFTC Commitments of Traders data.

    The COT report shows positioning of:
    - Commercial traders (hedgers): Usually fade the trend
    - Non-commercial traders (speculators/hedge funds): Usually follow the trend
    - Net positioning = Long - Short contracts

    Extreme positioning often signals reversals.
    """

    # CFTC data endpoints
    CFTC_BASE_URL = "https://www.cftc.gov/dea/newcot"

    # Alternative: Quandl/Nasdaq Data Link (free with API key)
    QUANDL_BASE_URL = "https://data.nasdaq.com/api/v3/datasets/CFTC"

    # Contract codes for key markets
    CONTRACTS = {
        'ES': {
            'name': 'S&P 500 E-mini',
            'cftc_code': '13874A',  # CME E-mini S&P 500
            'quandl_code': '13874A_F_L_ALL',
            'category': 'equity'
        },
        'NQ': {
            'name': 'Nasdaq 100 E-mini',
            'cftc_code': '20974A',  # CME E-mini Nasdaq
            'quandl_code': '20974A_F_L_ALL',
            'category': 'equity'
        },
        'TY': {
            'name': '10-Year Treasury Note',
            'cftc_code': '043602',  # CBOT 10Y T-Note
            'quandl_code': '043602_F_L_ALL',
            'category': 'bonds'
        },
        'GC': {
            'name': 'Gold',
            'cftc_code': '088691',  # COMEX Gold
            'quandl_code': '088691_F_L_ALL',
            'category': 'metals'
        },
        'CL': {
            'name': 'Crude Oil WTI',
            'cftc_code': '067651',  # NYMEX WTI Crude
            'quandl_code': '067651_F_L_ALL',
            'category': 'energy'
        },
        'DX': {
            'name': 'US Dollar Index',
            'cftc_code': '098662',  # ICE US Dollar Index
            'quandl_code': '098662_F_L_ALL',
            'category': 'currency'
        },
        'VX': {
            'name': 'VIX Futures',
            'cftc_code': '1170E1',  # CFE VIX Futures
            'quandl_code': '1170E1_F_L_ALL',
            'category': 'volatility'
        },
    }

    def __init__(self, quandl_api_key: Optional[str] = None):
        """
        Initialize COT collector.

        Args:
            quandl_api_key: Optional Nasdaq Data Link API key for direct access.
                           If not provided, will check NASDAQ_DATA_LINK_KEY env var,
                           then fall back to CFTC CSV downloads.
        """
        # Priority: explicit arg > env var > None
        self.api_key = quandl_api_key or os.getenv('NASDAQ_DATA_LINK_KEY') or os.getenv('QUANDL_API_KEY')

        if self.api_key:
            logger.info("COT Collector initialized with Nasdaq Data Link API key")
        else:
            logger.warning(
                "No Nasdaq Data Link API key found. COT data will use slow CFTC downloads. "
                "For faster data, sign up at https://data.nasdaq.com (free) and add "
                "NASDAQ_DATA_LINK_KEY to your .env file."
            )

        self.logger = logging.getLogger(__name__)
        self._cache: Dict[str, Dict[str, object]] = {}
        self._cache_ttl = timedelta(hours=6)  # Fresh cache window
        self._persistent_ttl = timedelta(days=14)  # LKG fallback window
        self._cache_dir = Path(__file__).resolve().parents[1] / "data" / "cache" / "cot"
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.debug(f"Could not create COT cache directory: {e}")
        self._quandl_disabled_until: Optional[datetime] = None

    def _is_quandl_enabled(self) -> bool:
        if self._quandl_disabled_until is None:
            return True
        return datetime.now() >= self._quandl_disabled_until

    def _disable_quandl_for(self, hours: int, reason: str):
        self._quandl_disabled_until = datetime.now() + timedelta(hours=hours)
        self.logger.warning(f"Nasdaq Data Link disabled for {hours}h: {reason}")

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [str(c).replace('\ufeff', '').strip() for c in df.columns]
        return df

    @staticmethod
    def _find_column_by_slug(df: pd.DataFrame, slug: str) -> Optional[str]:
        for col in df.columns:
            normalized = re.sub(r'[^a-z0-9]', '', str(col).lower())
            if normalized == slug:
                return col
        return None

    def _normalize_cftc_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._normalize_columns(df)
        slug_map = {
            "cftccontractmarketcode": "CFTC_Contract_Market_Code",
            "asofdateinformyymmdd": "As_of_Date_In_Form_YYMMDD",
            "reportdateasyyyymmdd": "Report_Date_as_YYYY-MM-DD",
            "noncommpositionslongall": "NonComm_Positions_Long_All",
            "noncommpositionsshortall": "NonComm_Positions_Short_All",
            "commpositionslongall": "Comm_Positions_Long_All",
            "commpositionsshortall": "Comm_Positions_Short_All",
            "openinterestall": "Open_Interest_All",
        }
        for slug, target in slug_map.items():
            col = self._find_column_by_slug(df, slug)
            if col and col != target:
                df = df.rename(columns={col: target})
        return df

    def _cache_file(self, symbol: str) -> Path:
        return self._cache_dir / f"{symbol.lower()}_lkg.json"

    def _slice_weeks(self, df: pd.DataFrame, weeks_back: int) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        sliced = df.sort_values("date").tail(weeks_back).copy()
        return sliced.reset_index(drop=True)

    def _get_cached_df(self, symbol: str, weeks_back: int, allow_stale: bool = False) -> Optional[pd.DataFrame]:
        entry = self._cache.get(symbol)
        if not entry:
            return None
        ts = entry.get("ts")
        df = entry.get("df")
        if not isinstance(ts, datetime) or not isinstance(df, pd.DataFrame) or df.empty:
            return None
        age = datetime.now() - ts
        if age <= self._cache_ttl or allow_stale:
            return self._slice_weeks(df, weeks_back)
        return None

    def _set_cached_df(self, symbol: str, df: pd.DataFrame):
        if df is None or df.empty:
            return
        self._cache[symbol] = {"ts": datetime.now(), "df": df.copy()}

    def _save_persisted_df(self, symbol: str, df: pd.DataFrame):
        if df is None or df.empty:
            return
        try:
            payload_df = df.copy()
            if "date" in payload_df.columns:
                payload_df["date"] = pd.to_datetime(payload_df["date"]).dt.strftime("%Y-%m-%d")
            rows = json.loads(payload_df.to_json(orient="records"))
            payload = {
                "saved_at": datetime.now().isoformat(),
                "rows": rows,
            }
            path = self._cache_file(symbol)
            tmp_path = path.with_suffix(".tmp")
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f)
            tmp_path.replace(path)
        except Exception as e:
            self.logger.debug(f"Could not persist COT LKG for {symbol}: {e}")

    def _load_persisted_df(self, symbol: str, weeks_back: int) -> Optional[pd.DataFrame]:
        try:
            path = self._cache_file(symbol)
            if not path.exists():
                return None
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                return None
            saved_at = payload.get("saved_at")
            rows = payload.get("rows")
            if not saved_at or not isinstance(rows, list) or len(rows) == 0:
                return None
            saved_at_dt = datetime.fromisoformat(saved_at)
            if datetime.now() - saved_at_dt > self._persistent_ttl:
                return None
            df = pd.DataFrame(rows)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"])
            if df.empty:
                return None
            return self._slice_weeks(df, weeks_back)
        except Exception as e:
            self.logger.debug(f"Could not load COT LKG for {symbol}: {e}")
            return None

    def fetch_cot_data(self, symbol: str, weeks_back: int = 52) -> Optional[pd.DataFrame]:
        """
        Fetch COT data for a specific contract.

        Args:
            symbol: Contract symbol (ES, TY, GC, CL, DX, VX)
            weeks_back: Number of weeks of history (default 52 = 1 year)

        Returns:
            DataFrame with COT positioning data
        """
        if symbol not in self.CONTRACTS:
            self.logger.error(f"Unknown symbol: {symbol}")
            return None

        contract = self.CONTRACTS[symbol]
        cached_df = self._get_cached_df(symbol, weeks_back)
        if cached_df is not None and not cached_df.empty:
            return cached_df

        try:
            # Method 1: Try cot_reports library first (most reliable)
            if COT_LIBRARY_AVAILABLE:
                df = self._fetch_from_cot_library(contract, weeks_back)
                if df is not None and not df.empty:
                    self._set_cached_df(symbol, df)
                    self._save_persisted_df(symbol, df)
                    return self._slice_weeks(df, weeks_back)

            # Method 2: Try Quandl/Nasdaq Data Link if API key provided
            if self.api_key and self._is_quandl_enabled():
                df = self._fetch_from_quandl(contract['quandl_code'], weeks_back)
                if df is not None and not df.empty:
                    self._set_cached_df(symbol, df)
                    self._save_persisted_df(symbol, df)
                    return self._slice_weeks(df, weeks_back)

            # Method 3: Fallback to CFTC direct (CSV)
            df = self._fetch_from_cftc_csv(contract['cftc_code'], weeks_back)
            if df is not None and not df.empty:
                self._set_cached_df(symbol, df)
                self._save_persisted_df(symbol, df)
                return self._slice_weeks(df, weeks_back)

        except Exception as e:
            self.logger.error(f"Error fetching COT data for {symbol}: {e}")

        # Fallback chain: stale in-memory cache, then last-known-good on disk
        stale_df = self._get_cached_df(symbol, weeks_back, allow_stale=True)
        if stale_df is not None and not stale_df.empty:
            self.logger.warning(f"Using stale in-memory COT cache for {symbol}")
            return stale_df

        persisted_df = self._load_persisted_df(symbol, weeks_back)
        if persisted_df is not None and not persisted_df.empty:
            self.logger.warning(f"Using last-known-good persisted COT data for {symbol}")
            self._set_cached_df(symbol, persisted_df)
            return persisted_df

        return None

    def _fetch_from_cot_library(self, contract: Dict, weeks_back: int) -> Optional[pd.DataFrame]:
        """
        Fetch using cot_reports library - most reliable method.
        Handles CFTC format changes automatically.
        """
        import sys
        from io import StringIO

        try:
            # Get contract name for searching
            contract_name = contract['name']
            cftc_code = contract['cftc_code']

            # Fetch legacy futures-only report (most commonly used)
            # cot.cot_year fetches data for a specific year
            current_year = datetime.now().year

            all_data = []
            for year in [current_year, current_year - 1]:
                try:
                    # Suppress cot_reports verbose stdout output
                    old_stdout = sys.stdout
                    sys.stdout = StringIO()
                    try:
                        df = cot.cot_year(year=year, cot_report_type='legacy_fut')
                    finally:
                        sys.stdout = old_stdout

                    if df is not None and not df.empty:
                        # Filter for our contract by CFTC code or name
                        if 'CFTC Contract Market Code' in df.columns:
                            filtered = df[df['CFTC Contract Market Code'].astype(str).str.contains(cftc_code, na=False)]
                        elif 'Contract Market Name' in df.columns:
                            # Fallback to name matching
                            filtered = df[df['Contract Market Name'].str.contains(contract_name.split()[0], case=False, na=False)]
                        else:
                            continue

                        if not filtered.empty:
                            all_data.append(filtered)
                except Exception as e:
                    self.logger.debug(f"cot_reports year {year} failed: {e}")
                    continue

            if not all_data:
                return None

            combined = pd.concat(all_data, ignore_index=True)
            return self._process_cot_library_data(combined, weeks_back)

        except Exception as e:
            self.logger.warning(f"cot_reports library fetch failed: {e}")
            return None
        finally:
            # Clean up temp file created by cot_reports library
            try:
                if os.path.exists('annual.txt'):
                    os.remove('annual.txt')
            except OSError as e:
                self.logger.debug(f"Could not remove temp file annual.txt: {e}")

    def _process_cot_library_data(self, df: pd.DataFrame, weeks_back: int) -> pd.DataFrame:
        """Process data from cot_reports library into standardized format."""
        try:
            processed = pd.DataFrame()

            # Date column - try multiple formats
            date_cols = ['As of Date in Form YYYY-MM-DD', 'Report_Date_as_YYYY-MM-DD', 'As_of_Date_In_Form_YYMMDD']
            for col in date_cols:
                if col in df.columns:
                    if 'YYMMDD' in col:
                        processed['date'] = pd.to_datetime(df[col], format='%y%m%d')
                    else:
                        processed['date'] = pd.to_datetime(df[col])
                    break

            if 'date' not in processed.columns:
                self.logger.warning("No date column found in COT data")
                return pd.DataFrame()

            # Non-commercial positions (speculators)
            noncomm_long_cols = ['Noncommercial Positions-Long (All)', 'NonComm_Positions_Long_All']
            noncomm_short_cols = ['Noncommercial Positions-Short (All)', 'NonComm_Positions_Short_All']

            for col in noncomm_long_cols:
                if col in df.columns:
                    processed['spec_long'] = pd.to_numeric(df[col], errors='coerce')
                    break

            for col in noncomm_short_cols:
                if col in df.columns:
                    processed['spec_short'] = pd.to_numeric(df[col], errors='coerce')
                    break

            # Commercial positions (hedgers)
            comm_long_cols = ['Commercial Positions-Long (All)', 'Comm_Positions_Long_All']
            comm_short_cols = ['Commercial Positions-Short (All)', 'Comm_Positions_Short_All']

            for col in comm_long_cols:
                if col in df.columns:
                    processed['comm_long'] = pd.to_numeric(df[col], errors='coerce')
                    break

            for col in comm_short_cols:
                if col in df.columns:
                    processed['comm_short'] = pd.to_numeric(df[col], errors='coerce')
                    break

            # Open interest
            oi_cols = ['Open Interest (All)', 'Open_Interest_All']
            for col in oi_cols:
                if col in df.columns:
                    processed['open_interest'] = pd.to_numeric(df[col], errors='coerce')
                    break

            # Calculate net positions
            if 'spec_long' in processed.columns and 'spec_short' in processed.columns:
                processed['spec_net'] = processed['spec_long'] - processed['spec_short']

            if 'comm_long' in processed.columns and 'comm_short' in processed.columns:
                processed['comm_net'] = processed['comm_long'] - processed['comm_short']

            # Sort and limit
            processed = processed.dropna(subset=['date'])
            processed = processed.sort_values('date', ascending=False).head(weeks_back)
            processed = processed.sort_values('date')

            return processed

        except Exception as e:
            self.logger.error(f"Error processing cot_reports data: {e}")
            return pd.DataFrame()

    def _fetch_from_quandl(self, quandl_code: str, weeks_back: int) -> Optional[pd.DataFrame]:
        """Fetch from Nasdaq Data Link (formerly Quandl)."""
        try:
            url = f"{self.QUANDL_BASE_URL}/{quandl_code}.csv"
            params = {
                'api_key': self.api_key,
                'rows': weeks_back
            }

            response = requests.get(url, params=params, timeout=15)
            if response.status_code in (401, 403):
                self._disable_quandl_for(24, f"HTTP {response.status_code}")
                return None
            if response.status_code == 429:
                self._disable_quandl_for(6, "Rate limited (429)")
                return None
            response.raise_for_status()

            df = pd.read_csv(StringIO(response.text), parse_dates=['Date'])
            df = df.rename(columns={'Date': 'date'})

            return self._process_quandl_data(df)

        except Exception as e:
            self.logger.warning(f"Quandl fetch failed: {e}")
            return None

    def _fetch_from_cftc_csv(self, cftc_code: str, weeks_back: int) -> Optional[pd.DataFrame]:
        """
        Fetch from CFTC public CSV files.
        Uses the legacy format which is freely available.
        """
        try:
            # CFTC provides annual CSV files
            current_year = datetime.now().year

            all_data = []

            # Fetch current year and previous year for sufficient history
            for year in [current_year, current_year - 1]:
                url = f"https://www.cftc.gov/files/dea/history/deacot{year}.zip"

                try:
                    response = requests.get(url, timeout=30)
                    if response.status_code == 200:
                        # Process ZIP file containing CSV
                        import zipfile
                        import io

                        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                            # Find the CSV file in the ZIP
                            csv_files = [f for f in z.namelist() if f.endswith('.txt')]
                            if csv_files:
                                with z.open(csv_files[0]) as f:
                                    df = pd.read_csv(f, low_memory=False)
                                    df = self._normalize_cftc_columns(df)
                                    # Filter for our contract
                                    code_col = self._find_column_by_slug(df, "cftccontractmarketcode")
                                    if not code_col:
                                        self.logger.warning(
                                            f"CFTC CSV missing contract code column. Columns: {list(df.columns)[:6]}"
                                        )
                                        continue
                                    codes = df[code_col].astype(str).str.strip()
                                    target = str(cftc_code).lstrip('0')
                                    df_filtered = df[codes.str.lstrip('0') == target]
                                    if not df_filtered.empty:
                                        all_data.append(df_filtered)
                except Exception as e:
                    self.logger.warning(f"Could not fetch CFTC data for {year}: {e}")
                    continue

            if not all_data:
                return None

            # Combine all years
            combined = pd.concat(all_data, ignore_index=True)

            return self._process_cftc_data(combined, weeks_back)

        except Exception as e:
            self.logger.error(f"CFTC CSV fetch failed: {e}")
            return None

    def _process_quandl_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process Quandl format data into standardized format."""
        # Quandl columns vary, but typically include:
        # - Non-commercial long/short
        # - Commercial long/short
        # - Total open interest

        processed = pd.DataFrame()
        processed['date'] = df['date']

        # Try to extract key columns (names vary by dataset)
        try:
            # Non-commercial (speculators/hedge funds)
            if 'Noncommercial Long' in df.columns:
                processed['spec_long'] = df['Noncommercial Long']
                processed['spec_short'] = df['Noncommercial Short']
            elif 'Asset Mgr Longs' in df.columns:
                processed['spec_long'] = df['Asset Mgr Longs']
                processed['spec_short'] = df['Asset Mgr Shorts']

            # Commercial (hedgers)
            if 'Commercial Long' in df.columns:
                processed['comm_long'] = df['Commercial Long']
                processed['comm_short'] = df['Commercial Short']

            # Open interest
            if 'Open Interest' in df.columns:
                processed['open_interest'] = df['Open Interest']
            elif 'Open Interest (All)' in df.columns:
                processed['open_interest'] = df['Open Interest (All)']

            # Calculate net positions
            if 'spec_long' in processed.columns and 'spec_short' in processed.columns:
                processed['spec_net'] = processed['spec_long'] - processed['spec_short']

            if 'comm_long' in processed.columns and 'comm_short' in processed.columns:
                processed['comm_net'] = processed['comm_long'] - processed['comm_short']

        except Exception as e:
            self.logger.warning(f"Error processing Quandl data: {e}")

        return processed

    def _process_cftc_data(self, df: pd.DataFrame, weeks_back: int) -> pd.DataFrame:
        """Process CFTC legacy format data into standardized format."""
        try:
            # CFTC legacy columns
            processed = pd.DataFrame()

            # Date column
            if 'As_of_Date_In_Form_YYMMDD' in df.columns:
                df['date'] = pd.to_datetime(df['As_of_Date_In_Form_YYMMDD'], format='%y%m%d')
            elif 'Report_Date_as_YYYY-MM-DD' in df.columns:
                df['date'] = pd.to_datetime(df['Report_Date_as_YYYY-MM-DD'])

            processed['date'] = df['date']

            # Non-commercial positions (speculators)
            if 'NonComm_Positions_Long_All' in df.columns:
                processed['spec_long'] = df['NonComm_Positions_Long_All']
                processed['spec_short'] = df['NonComm_Positions_Short_All']

            # Commercial positions (hedgers)
            if 'Comm_Positions_Long_All' in df.columns:
                processed['comm_long'] = df['Comm_Positions_Long_All']
                processed['comm_short'] = df['Comm_Positions_Short_All']

            # Open interest
            if 'Open_Interest_All' in df.columns:
                processed['open_interest'] = df['Open_Interest_All']

            # Calculate net positions
            if 'spec_long' in processed.columns:
                processed['spec_net'] = processed['spec_long'] - processed['spec_short']

            if 'comm_long' in processed.columns:
                processed['comm_net'] = processed['comm_long'] - processed['comm_short']

            # Sort by date and limit to requested weeks
            processed = processed.sort_values('date', ascending=False).head(weeks_back)
            processed = processed.sort_values('date')

            return processed

        except Exception as e:
            self.logger.error(f"Error processing CFTC data: {e}")
            return pd.DataFrame()

    def get_all_positions(self, weeks_back: int = 26) -> Dict[str, pd.DataFrame]:
        """
        Fetch COT data for all tracked contracts.

        Args:
            weeks_back: Weeks of history

        Returns:
            Dict mapping symbol to DataFrame
        """
        positions = {}

        for symbol in self.CONTRACTS:
            df = self.fetch_cot_data(symbol, weeks_back)
            if df is not None and not df.empty:
                positions[symbol] = df
                self.logger.info(f"Fetched {len(df)} weeks of COT data for {symbol}")
            else:
                self.logger.warning(f"No COT data available for {symbol}")

        return positions

    def get_positioning_summary(self) -> Optional[Dict]:
        """
        Get current positioning summary across all markets.

        Returns:
            Dict with current net positions and changes
        """
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'positions': {}
            }

            for symbol, contract in self.CONTRACTS.items():
                df = self.fetch_cot_data(symbol, weeks_back=4)  # Just need recent data

                if df is None or df.empty:
                    continue

                latest = df.iloc[-1]
                prev = df.iloc[-2] if len(df) > 1 else latest

                position_data = {
                    'name': contract['name'],
                    'category': contract['category'],
                    'date': latest['date'].strftime('%Y-%m-%d') if hasattr(latest['date'], 'strftime') else str(latest['date']),
                }

                # Speculator positioning
                if 'spec_net' in latest:
                    position_data['spec_net'] = int(latest['spec_net'])
                    position_data['spec_net_change'] = int(latest['spec_net'] - prev['spec_net'])

                # Commercial positioning
                if 'comm_net' in latest:
                    position_data['comm_net'] = int(latest['comm_net'])
                    position_data['comm_net_change'] = int(latest['comm_net'] - prev['comm_net'])

                # Open interest
                if 'open_interest' in latest:
                    position_data['open_interest'] = int(latest['open_interest'])

                summary['positions'][symbol] = position_data

            return summary

        except Exception as e:
            self.logger.error(f"Error generating positioning summary: {e}")
            return None

    def analyze_positioning_extremes(self, symbol: str, weeks_back: int = 52) -> Optional[Dict]:
        """
        Analyze if current positioning is at historical extremes.

        Extreme positioning often precedes reversals:
        - Max long speculators = potential top
        - Max short speculators = potential bottom

        Args:
            symbol: Contract symbol
            weeks_back: Lookback period for percentile calculation

        Returns:
            Dict with percentile rankings and signals
        """
        df = self.fetch_cot_data(symbol, weeks_back)

        if df is None or df.empty or 'spec_net' not in df.columns:
            return None

        current_net = df.iloc[-1]['spec_net']

        # Calculate percentile rank
        percentile = (df['spec_net'] < current_net).sum() / len(df) * 100

        # Determine signal
        if percentile >= 90:
            signal = "EXTREME_LONG"
            description = "Speculators extremely long - potential contrarian sell"
        elif percentile >= 75:
            signal = "LONG"
            description = "Speculators moderately long"
        elif percentile <= 10:
            signal = "EXTREME_SHORT"
            description = "Speculators extremely short - potential contrarian buy"
        elif percentile <= 25:
            signal = "SHORT"
            description = "Speculators moderately short"
        else:
            signal = "NEUTRAL"
            description = "Positioning within normal range"

        return {
            'symbol': symbol,
            'name': self.CONTRACTS[symbol]['name'],
            'current_net': int(current_net),
            'percentile': round(percentile, 1),
            'signal': signal,
            'description': description,
            'lookback_weeks': weeks_back,
            '52w_high': int(df['spec_net'].max()),
            '52w_low': int(df['spec_net'].min()),
        }

    def get_all_extremes(self, weeks_back: int = 52) -> List[Dict]:
        """
        Analyze positioning extremes across all markets.

        Returns:
            List of extreme analysis dicts, sorted by most extreme first
        """
        extremes = []

        for symbol in self.CONTRACTS:
            analysis = self.analyze_positioning_extremes(symbol, weeks_back)
            if analysis:
                extremes.append(analysis)

        # Sort by how extreme (distance from 50th percentile)
        extremes.sort(key=lambda x: abs(x['percentile'] - 50), reverse=True)

        return extremes
