"""
SQLite database manager for storing historical market data
INTEGRATED VERSION - Phase 1 + Phase 2 Complete

Now includes data validation layer to catch malformed data before database saves.
"""

import sqlite3
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import json
from pathlib import Path
import logging

from utils.data_validator import DataValidator, ValidationResult, validate_before_save

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages SQLite database for market data"""
    
    def __init__(self, db_path: str = "data/market_data.db"):
        self.db_path = db_path
        
        # Create data directory if it doesn't exist
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Indicators table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    indicator_name TEXT NOT NULL,
                    series_id TEXT,
                    date DATE NOT NULL,
                    value REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(indicator_name, date)
                )
            """)
            
            # Signals table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_type TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    strength REAL,
                    metadata TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Daily snapshots table (for dashboard)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL UNIQUE,
                    credit_spread_hy REAL,
                    credit_spread_ig REAL,
                    treasury_10y REAL,
                    fed_funds REAL,
                    vix_spot REAL,
                    vix9d REAL,
                    vvix REAL,
                    vvix_signal TEXT,
                    skew REAL,
                    vrp REAL,
                    vix_contango REAL,
                    put_call_ratio REAL,
                    fear_greed_score REAL,
                    market_breadth REAL,
                    left_signal TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # VRP table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vrp_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL UNIQUE,
                    vix REAL NOT NULL,
                    realized_vol REAL NOT NULL,
                    vrp REAL NOT NULL,
                    regime TEXT,
                    expected_6m_return REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # PHASE 2: Fed Balance Sheet table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fed_balance_sheet (
                    date TEXT PRIMARY KEY,
                    total_assets REAL,
                    reserve_balances REAL,
                    loans REAL,
                    qt_cumulative REAL,
                    qt_monthly_pace REAL,
                    updated_at TEXT
                )
            """)
            
            # PHASE 2: MOVE Index table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS move_index (
                    date TEXT PRIMARY KEY,
                    move REAL,
                    percentile REAL,
                    stress_level TEXT,
                    source TEXT,
                    updated_at TEXT
                )
            """)
            
            # PHASE 2: Repo Market table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS repo_market (
                    date TEXT PRIMARY KEY,
                    sofr REAL,
                    gc_repo REAL,
                    rrp_on REAL,
                    triparty_repo REAL,
                    sofr_z_score REAL,
                    stress_level TEXT,
                    updated_at TEXT
                )
            """)

            # Breadth History table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS breadth_history (
                    date TEXT PRIMARY KEY,
                    advancing INTEGER NOT NULL,
                    declining INTEGER NOT NULL,
                    unchanged INTEGER DEFAULT 0,
                    total INTEGER NOT NULL,
                    breadth_pct REAL NOT NULL,
                    ad_line REAL,
                    ad_diff INTEGER,
                    mcclellan REAL,
                    ema19 REAL,
                    ema39 REAL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_indicators_date ON indicators(date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_indicators_name ON indicators(indicator_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_vrp_date ON vrp_data(date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_fed_bs_date ON fed_balance_sheet(date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_move_date ON move_index(date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_repo_date ON repo_market(date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_breadth_date ON breadth_history(date)")

            conn.commit()

            # Run migrations for existing databases
            self._run_migrations(conn)

    # Allowlist of valid column names and types for migrations (prevents SQL injection)
    VALID_MIGRATION_COLUMNS = {
        'vix9d': 'REAL',
        'vvix': 'REAL',
        'vvix_signal': 'TEXT',
        'skew': 'REAL',
    }
    VALID_SQL_TYPES = {'REAL', 'TEXT', 'INTEGER', 'BLOB', 'NULL'}

    def _run_migrations(self, conn):
        """Add missing columns to existing tables (safe migrations)"""
        cursor = conn.cursor()

        # Get existing columns in daily_snapshots
        cursor.execute("PRAGMA table_info(daily_snapshots)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        # Add missing columns using allowlisted values only
        for col_name, col_type in self.VALID_MIGRATION_COLUMNS.items():
            # Validate column type is in allowlist (defense in depth)
            if col_type not in self.VALID_SQL_TYPES:
                logger.warning(f"Invalid column type '{col_type}' for '{col_name}' - skipping")
                continue

            if col_name not in existing_columns:
                try:
                    # Safe because col_name and col_type come from class constants
                    cursor.execute(f"ALTER TABLE daily_snapshots ADD COLUMN {col_name} {col_type}")
                    logger.info(f"Added missing column '{col_name}' to daily_snapshots")
                except sqlite3.OperationalError as e:
                    # Column might already exist or other error
                    logger.debug(f"Migration note for {col_name}: {e}")

        conn.commit()
    
    # ========================================
    # EXISTING METHODS (Phase 1)
    # ========================================
    
    def save_indicator(self, indicator_name: str, date: datetime, value: float, series_id: Optional[str] = None):
        """Save a single indicator value"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO indicators (indicator_name, series_id, date, value)
                VALUES (?, ?, ?, ?)
            """, (indicator_name, series_id, date.strftime('%Y-%m-%d'), value))
            
            conn.commit()
    
    def save_indicators_batch(self, data: Dict[str, pd.DataFrame]):
        """Save multiple indicators from FRED data"""
        with sqlite3.connect(self.db_path) as conn:
            for indicator_name, df in data.items():
                if df.empty:
                    continue
                
                series_id = df.columns[1]  # Second column is the series_id
                
                for _, row in df.iterrows():
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO indicators (indicator_name, series_id, date, value)
                        VALUES (?, ?, ?, ?)
                    """, (indicator_name, series_id, row['date'].strftime('%Y-%m-%d'), float(row[series_id])))
            
            conn.commit()
    
    def save_signal(self, signal_type: str, signal: str, strength: float = 50.0, metadata: Optional[Dict] = None):
        """Save a trading signal"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            metadata_json = json.dumps(metadata) if metadata else None
            
            cursor.execute("""
                INSERT INTO signals (signal_type, signal, strength, metadata)
                VALUES (?, ?, ?, ?)
            """, (signal_type, signal, strength, metadata_json))
            
            conn.commit()
    
    def save_daily_snapshot(self, snapshot: Dict) -> Tuple[bool, ValidationResult]:
        """
        Save complete daily market snapshot with validation.

        Args:
            snapshot: Dictionary with market data

        Returns:
            Tuple of (success: bool, validation_result: ValidationResult)
        """
        # Validate data before saving
        result = validate_before_save(snapshot, 'daily_snapshot')

        if not result.is_valid:
            logger.error(f"Daily snapshot validation failed: {result.errors}")
            return False, result

        if result.warnings:
            for warning in result.warnings:
                logger.warning(f"Daily snapshot warning: {warning}")

        if result.estimated_fields:
            logger.info(f"Estimated fields in snapshot: {result.estimated_fields}")

        # Save validated data
        validated = result.data
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO daily_snapshots
                (date, credit_spread_hy, credit_spread_ig, treasury_10y, fed_funds,
                 vix_spot, vix9d, vvix, vvix_signal, skew, vrp, vix_contango, put_call_ratio,
                 fear_greed_score, market_breadth, left_signal)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                validated.get('date'),
                validated.get('credit_spread_hy'),
                validated.get('credit_spread_ig'),
                validated.get('treasury_10y'),
                validated.get('fed_funds'),
                validated.get('vix_spot'),
                validated.get('vix9d'),
                validated.get('vvix'),
                validated.get('vvix_signal'),
                validated.get('skew'),
                validated.get('vrp'),
                validated.get('vix_contango'),
                validated.get('put_call_ratio'),
                validated.get('fear_greed_score'),
                validated.get('market_breadth'),
                validated.get('left_signal')
            ))

            conn.commit()

        logger.info(f"Saved daily snapshot for {validated.get('date')}")
        return True, result
    
    def get_indicator_history(self, indicator_name: str, days: int = 365) -> pd.DataFrame:
        """Get historical data for an indicator"""
        query = """
            SELECT date, value
            FROM indicators
            WHERE indicator_name = ?
            AND date >= date('now', '-{} days')
            ORDER BY date
        """.format(days)
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=(indicator_name,))
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
            return df
    
    def get_latest_snapshot(self, include_age: bool = False) -> Optional[Dict]:
        """
        Get most recent daily snapshot with optional age tracking.

        Args:
            include_age: If True, adds '_age_hours' and '_status' fields

        Returns:
            Dict with snapshot data, or None if no data exists.
            If include_age=True, also includes:
                - _age_hours: float (hours since data was recorded)
                - _age_string: str (human-readable age like "2h ago")
                - _status: str ("fresh", "stale", "very_stale")
                - _is_fresh: bool (True if < 4 hours old)
        """
        query = """
            SELECT * FROM daily_snapshots
            ORDER BY date DESC
            LIMIT 1
        """

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query)

            row = cursor.fetchone()
            if not row:
                return None

            columns = [description[0] for description in cursor.description]
            result = dict(zip(columns, row))

            if include_age:
                # Calculate data age
                try:
                    data_date = datetime.strptime(result.get('date', ''), '%Y-%m-%d')
                    # Check for updated_at timestamp if available
                    if 'updated_at' in result and result['updated_at']:
                        try:
                            data_date = datetime.fromisoformat(result['updated_at'].replace('Z', '+00:00'))
                            data_date = data_date.replace(tzinfo=None)  # Make naive for comparison
                        except (ValueError, AttributeError):
                            pass  # Use date field instead

                    now = datetime.now()
                    age_delta = now - data_date
                    age_hours = age_delta.total_seconds() / 3600

                    # Determine status
                    if age_hours < 4:
                        status = "fresh"
                        age_string = f"{int(age_hours * 60)}m ago" if age_hours < 1 else f"{int(age_hours)}h ago"
                    elif age_hours < 24:
                        status = "stale"
                        age_string = f"{int(age_hours)}h ago"
                    elif age_hours < 168:  # 1 week
                        status = "very_stale"
                        age_string = f"{int(age_hours / 24)}d ago"
                    else:
                        status = "very_stale"
                        age_string = f"{int(age_hours / 168)}w ago"

                    result['_age_hours'] = round(age_hours, 2)
                    result['_age_string'] = age_string
                    result['_status'] = status
                    result['_is_fresh'] = age_hours < 4

                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not calculate data age: {e}")
                    result['_age_hours'] = None
                    result['_age_string'] = "unknown"
                    result['_status'] = "unknown"
                    result['_is_fresh'] = False

            return result
    
    def get_recent_signals(self, signal_type: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """Get recent signals"""
        if signal_type:
            query = """
                SELECT * FROM signals
                WHERE signal_type = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """
            params = (signal_type, limit)
        else:
            query = """
                SELECT * FROM signals
                ORDER BY timestamp DESC
                LIMIT ?
            """
            params = (limit,)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def save_vrp_data(self, vrp_analysis: Dict) -> Tuple[bool, ValidationResult]:
        """
        Save VRP analysis to database with validation.

        Args:
            vrp_analysis: Dictionary with VRP data

        Returns:
            Tuple of (success: bool, validation_result: ValidationResult)
        """
        # Add date if not present
        if 'date' not in vrp_analysis:
            vrp_analysis['date'] = datetime.now().strftime('%Y-%m-%d')

        # Validate
        result = validate_before_save(vrp_analysis, 'vrp')

        if not result.is_valid:
            logger.error(f"VRP validation failed: {result.errors}")
            return False, result

        if result.warnings:
            for warning in result.warnings:
                logger.warning(f"VRP warning: {warning}")

        validated = result.data
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO vrp_data
                (date, vix, realized_vol, vrp, regime, expected_6m_return)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                validated.get('date'),
                validated.get('vix'),
                validated.get('realized_vol'),
                validated.get('vrp'),
                validated.get('regime'),
                validated.get('expected_6m_return')
            ))

            conn.commit()

        logger.info(f"Saved VRP data for {validated.get('date')}")
        return True, result
    
    def get_vrp_history(self, days: int = 365) -> pd.DataFrame:
        """Get historical VRP data"""
        query = """
            SELECT date, vix, realized_vol, vrp, regime
            FROM vrp_data
            WHERE date >= date('now', '-{} days')
            ORDER BY date
        """.format(days)
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
            return df
    
    def get_latest_vrp(self) -> Optional[Dict]:
        """Get most recent VRP analysis"""
        query = """
            SELECT * FROM vrp_data
            ORDER BY date DESC
            LIMIT 1
        """
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            
            row = cursor.fetchone()
            if not row:
                return None
            
            columns = [description[0] for description in cursor.description]
            return dict(zip(columns, row))
    
    # ========================================
    # LIQUIDITY DATA METHODS (Phase 1)
    # ========================================
    
    def save_liquidity_history(self, df: pd.DataFrame):
        """
        Save liquidity history to liquidity_history table.
        
        Args:
            df: DataFrame with columns: date, rrp_on, tga, sofr
        """
        if df.empty:
            logger.warning("Empty liquidity DataFrame, skipping save")
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for _, row in df.iterrows():
                    # Calculate net liquidity using CORRECT formula: Fed BS - TGA - RRP
                    # Note: If fed_bs is not in this row, we can't compute proper net liquidity
                    # The old formula -(RRP + TGA) was WRONG - it inverted the relationship
                    net_liq = None
                    fed_bs = row.get('fed_bs') or row.get('fed_balance_sheet')
                    rrp = row.get('rrp_on') or row.get('rrp')
                    tga = row.get('tga')

                    if pd.notna(fed_bs) and pd.notna(rrp) and pd.notna(tga):
                        # Correct institutional formula: Net Liquidity = Fed BS - TGA - RRP
                        net_liq = float(fed_bs) - float(tga) - float(rrp)
                    elif pd.notna(rrp) and pd.notna(tga):
                        # Fallback: store negative drain components (less accurate)
                        # This at least shows the drain direction correctly
                        net_liq = -(float(rrp) + float(tga))
                    
                    cursor.execute("""
                        INSERT OR REPLACE INTO liquidity_history
                        (date, rrp_on, tga, sofr, net_liquidity)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date']),
                        float(row['rrp_on']) if pd.notna(row.get('rrp_on')) else None,
                        float(row['tga']) if pd.notna(row.get('tga')) else None,
                        float(row['sofr']) if pd.notna(row.get('sofr')) else None,
                        net_liq
                    ))
                
                conn.commit()
            
            logger.info(f"Saved {len(df)} rows of liquidity data to liquidity_history table")
            
        except Exception as e:
            logger.error(f"Error saving liquidity history: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def get_liquidity_history(self, days: int = 365) -> pd.DataFrame:
        """
        Get historical liquidity data from database.
        
        Args:
            days: Number of days of history
            
        Returns:
            DataFrame with columns: date, rrp_on, tga, sofr, net_liquidity
        """
        try:
            rrp = self.get_indicator_history('liquidity_rrp', days)
            tga = self.get_indicator_history('liquidity_tga', days)
            sofr = self.get_indicator_history('liquidity_sofr', days)
            net_liq = self.get_indicator_history('liquidity_net', days)
            
            # Merge all series
            df = pd.DataFrame()
            
            if not rrp.empty:
                df = rrp.rename(columns={'value': 'rrp_on'})
            
            if not tga.empty:
                tga_renamed = tga.rename(columns={'value': 'tga'})
                if df.empty:
                    df = tga_renamed
                else:
                    df = df.merge(tga_renamed, on='date', how='outer')
            
            if not sofr.empty:
                sofr_renamed = sofr.rename(columns={'value': 'sofr'})
                if df.empty:
                    df = sofr_renamed
                else:
                    df = df.merge(sofr_renamed, on='date', how='outer')
            
            if not net_liq.empty:
                net_liq_renamed = net_liq.rename(columns={'value': 'net_liquidity'})
                if df.empty:
                    df = net_liq_renamed
                else:
                    df = df.merge(net_liq_renamed, on='date', how='outer')
            
            if not df.empty:
                df = df.sort_values('date').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting liquidity history: {e}")
            return pd.DataFrame()
    
    def get_latest_liquidity(self) -> dict:
        """
        Get latest liquidity values.
        
        Returns:
            Dict with keys: rrp_on, tga, sofr, net_liquidity
        """
        try:
            # Get latest values for each indicator
            rrp_df = self.get_indicator_history('liquidity_rrp', days=30)
            tga_df = self.get_indicator_history('liquidity_tga', days=30)
            sofr_df = self.get_indicator_history('liquidity_sofr', days=30)
            net_liq_df = self.get_indicator_history('liquidity_net', days=30)
            
            return {
                'rrp_on': rrp_df['value'].iloc[-1] if not rrp_df.empty else None,
                'tga': tga_df['value'].iloc[-1] if not tga_df.empty else None,
                'sofr': sofr_df['value'].iloc[-1] if not sofr_df.empty else None,
                'net_liquidity': net_liq_df['value'].iloc[-1] if not net_liq_df.empty else None
            }
        except Exception as e:
            logger.error(f"Error getting latest liquidity: {e}")
            return {}
    
    # ========================================
    # PHASE 2: FED BALANCE SHEET METHODS
    # ========================================
    
    def save_fed_balance_sheet(self, df: pd.DataFrame) -> bool:
        """
        Save Fed Balance Sheet data to database
        
        Args:
            df: DataFrame with columns: date, total_assets, reserve_balances, loans
        
        Returns:
            bool: True if successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Insert or replace data
                for _, row in df.iterrows():
                    cursor.execute("""
                        INSERT OR REPLACE INTO fed_balance_sheet 
                        (date, total_assets, reserve_balances, loans, qt_cumulative, qt_monthly_pace, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row['date'].strftime('%Y-%m-%d'),
                        float(row['total_assets']),
                        float(row.get('reserve_balances', 0)),
                        float(row.get('loans', 0)),
                        float(row.get('qt_cumulative', 0)),
                        float(row.get('qt_monthly_pace', 0)),
                        datetime.now().isoformat()
                    ))
                
                conn.commit()
                logger.info(f"Saved {len(df)} Fed Balance Sheet records")
                return True
                
        except Exception as e:
            logger.error(f"Error saving Fed Balance Sheet data: {e}")
            return False
    
    def get_fed_balance_sheet_history(self, days: int = 365) -> pd.DataFrame:
        """
        Get Fed Balance Sheet history
        
        Args:
            days: Number of days of history
        
        Returns:
            DataFrame with Fed BS data
        """
        try:
            query = """
            SELECT * FROM fed_balance_sheet
            WHERE date >= date('now', '-' || ? || ' days')
            ORDER BY date DESC
            """
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=(days,))
                
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                
                return df
                
        except Exception as e:
            logger.error(f"Error fetching Fed Balance Sheet history: {e}")
            return pd.DataFrame()
    
    def get_latest_fed_balance_sheet(self) -> dict:
        """Get latest Fed Balance Sheet snapshot"""
        try:
            query = """
            SELECT * FROM fed_balance_sheet
            ORDER BY date DESC
            LIMIT 1
            """
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                result = cursor.execute(query).fetchone()
                
                if result:
                    columns = [desc[0] for desc in cursor.description]
                    return dict(zip(columns, result))
                
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching latest Fed Balance Sheet: {e}")
            return {}
    
    # ========================================
    # PHASE 2: MOVE INDEX METHODS
    # ========================================
    
    def save_move_data(self, df: pd.DataFrame) -> bool:
        """
        Save MOVE Index data to database
        
        Args:
            df: DataFrame with columns: date, move, percentile
        
        Returns:
            bool: True if successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Insert or replace data
                for _, row in df.iterrows():
                    # Classify stress level
                    move_value = float(row['move'])
                    if move_value < 80:
                        stress_level = "LOW"
                    elif move_value < 120:
                        stress_level = "NORMAL"
                    elif move_value < 150:
                        stress_level = "ELEVATED"
                    else:
                        stress_level = "HIGH"
                    
                    cursor.execute("""
                        INSERT OR REPLACE INTO move_index 
                        (date, move, percentile, stress_level, source, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        row['date'].strftime('%Y-%m-%d'),
                        move_value,
                        float(row.get('percentile', 50)),
                        stress_level,
                        row.get('source', 'unknown'),
                        datetime.now().isoformat()
                    ))
                
                conn.commit()
                logger.info(f"Saved {len(df)} MOVE Index records")
                return True
                
        except Exception as e:
            logger.error(f"Error saving MOVE data: {e}")
            return False
    
    def get_move_history(self, days: int = 365) -> pd.DataFrame:
        """
        Get MOVE Index history
        
        Args:
            days: Number of days of history
        
        Returns:
            DataFrame with MOVE data
        """
        try:
            query = """
            SELECT * FROM move_index
            WHERE date >= date('now', '-' || ? || ' days')
            ORDER BY date DESC
            """
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=(days,))
                
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                
                return df
                
        except Exception as e:
            logger.error(f"Error fetching MOVE history: {e}")
            return pd.DataFrame()
    
    def get_latest_move(self) -> dict:
        """Get latest MOVE Index value"""
        try:
            query = """
            SELECT * FROM move_index
            ORDER BY date DESC
            LIMIT 1
            """
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                result = cursor.execute(query).fetchone()
                
                if result:
                    columns = [desc[0] for desc in cursor.description]
                    return dict(zip(columns, result))
                
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching latest MOVE: {e}")
            return {}
    
    # ========================================
    # PHASE 2: REPO MARKET METHODS
    # ========================================
    
    def save_repo_data(self, df: pd.DataFrame) -> bool:
        """
        Save Repo market data to database
        
        Args:
            df: DataFrame with columns: date, sofr, gc_repo, rrp_on, sofr_z_score
        
        Returns:
            bool: True if successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Insert or replace data
                for _, row in df.iterrows():
                    # Classify stress level based on z-score
                    z_score = float(row.get('sofr_z_score', 0))
                    if z_score > 2.0:
                        stress_level = "STRESS"
                    elif z_score > 1.0:
                        stress_level = "ELEVATED"
                    else:
                        stress_level = "NORMAL"
                    
                    cursor.execute("""
                        INSERT OR REPLACE INTO repo_market 
                        (date, sofr, gc_repo, rrp_on, triparty_repo, sofr_z_score, stress_level, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row['date'].strftime('%Y-%m-%d'),
                        float(row['sofr']),
                        float(row.get('gc_repo', 0)),
                        float(row.get('rrp_on', 0)),
                        float(row.get('triparty_repo', 0)),
                        z_score,
                        stress_level,
                        datetime.now().isoformat()
                    ))
                
                conn.commit()
                logger.info(f"Saved {len(df)} Repo market records")
                return True
                
        except Exception as e:
            logger.error(f"Error saving Repo data: {e}")
            return False
    
    def get_repo_history(self, days: int = 365) -> pd.DataFrame:
        """
        Get Repo market history
        
        Args:
            days: Number of days of history
        
        Returns:
            DataFrame with Repo data
        """
        try:
            query = """
            SELECT * FROM repo_market
            WHERE date >= date('now', '-' || ? || ' days')
            ORDER BY date DESC
            """
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=(days,))
                
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                
                return df
                
        except Exception as e:
            logger.error(f"Error fetching Repo history: {e}")
            return pd.DataFrame()
    
    def get_latest_repo(self) -> dict:
        """Get latest Repo market snapshot"""
        try:
            query = """
            SELECT * FROM repo_market
            ORDER BY date DESC
            LIMIT 1
            """
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                result = cursor.execute(query).fetchone()
                
                if result:
                    columns = [desc[0] for desc in cursor.description]
                    return dict(zip(columns, result))
                
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching latest Repo data: {e}")
            return {}
    
    # ========================================
    # PHASE 2: HEALTH CHECK
    # ========================================
    
    def check_phase2_health(self) -> dict:
        """
        Health check for Phase 2 modules
        
        Returns:
            dict: Health status for each module
        """
        health = {
            "fed_balance_sheet": {"status": "unknown", "last_update": None, "record_count": 0},
            "move_index": {"status": "unknown", "last_update": None, "record_count": 0},
            "repo_market": {"status": "unknown", "last_update": None, "record_count": 0},
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check Fed Balance Sheet
                fed_bs = self.get_latest_fed_balance_sheet()
                if fed_bs:
                    health["fed_balance_sheet"]["status"] = "healthy"
                    health["fed_balance_sheet"]["last_update"] = fed_bs.get('updated_at')
                    
                    count = cursor.execute("SELECT COUNT(*) FROM fed_balance_sheet").fetchone()[0]
                    health["fed_balance_sheet"]["record_count"] = count
                else:
                    health["fed_balance_sheet"]["status"] = "no_data"
                
                # Check MOVE Index
                move = self.get_latest_move()
                if move:
                    health["move_index"]["status"] = "healthy"
                    health["move_index"]["last_update"] = move.get('updated_at')
                    
                    count = cursor.execute("SELECT COUNT(*) FROM move_index").fetchone()[0]
                    health["move_index"]["record_count"] = count
                else:
                    health["move_index"]["status"] = "no_data"
                
                # Check Repo Market
                repo = self.get_latest_repo()
                if repo:
                    health["repo_market"]["status"] = "healthy"
                    health["repo_market"]["last_update"] = repo.get('updated_at')
                    
                    count = cursor.execute("SELECT COUNT(*) FROM repo_market").fetchone()[0]
                    health["repo_market"]["record_count"] = count
                else:
                    health["repo_market"]["status"] = "no_data"
                    
        except Exception as e:
            logger.error(f"Error in Phase 2 health check: {e}")
        
        return health

    def save_breadth_data(self, breadth_df) -> Tuple[int, List[str]]:
        """
        Save breadth history data with validation.

        Args:
            breadth_df: DataFrame with breadth data

        Returns:
            Tuple of (rows_saved: int, errors: List[str])
        """
        if breadth_df.empty:
            return 0, []

        validator = DataValidator()
        errors = []
        rows_saved = 0

        try:
            breadth_df = breadth_df.copy()
            # Handle date conversion
            if hasattr(breadth_df['date'].iloc[0], 'strftime'):
                breadth_df['date'] = breadth_df['date'].dt.strftime('%Y-%m-%d')

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                for idx, row in breadth_df.iterrows():
                    # Validate each row
                    result = validator.validate_breadth_data(row.to_dict())

                    if not result.is_valid:
                        errors.extend([f"Row {idx}: {e}" for e in result.errors])
                        continue

                    validated = result.data
                    cursor.execute("""
                        INSERT OR REPLACE INTO breadth_history
                        (date, advancing, declining, unchanged, total, breadth_pct, ad_line, ad_diff, mcclellan)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        validated.get('date'),
                        validated.get('advancing', 0),
                        validated.get('declining', 0),
                        validated.get('unchanged', 0),
                        validated.get('total', 0),
                        validated.get('breadth_pct'),
                        validated.get('ad_line', 0),
                        validated.get('ad_diff', 0),
                        validated.get('mcclellan', 0)
                    ))
                    rows_saved += 1

                conn.commit()

            logger.info(f"Saved {rows_saved} breadth records ({len(errors)} validation errors)")

        except Exception as e:
            logger.error(f"Error saving breadth data: {e}")
            errors.append(str(e))

        return rows_saved, errors
    
    def get_breadth_history(self, days=90):
        """Get breadth history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(
                    "SELECT * FROM breadth_history ORDER BY date DESC LIMIT ?",
                    conn,
                    params=(days,)
                )
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
            return df
        except Exception as e:
            logger.error(f"Error getting breadth history: {e}")
            return pd.DataFrame()
    
    def get_latest_breadth(self):
        """Get latest breadth"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT * FROM breadth_history ORDER BY date DESC LIMIT 1")
                row = cursor.fetchone()
            if row:
                return dict(zip([d[0] for d in cursor.description], row))
            return None
        except Exception as e:
            logger.error(f"Error getting latest breadth: {e}")
            return None
