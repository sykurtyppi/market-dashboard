"""
SQLite database manager for storing historical market data
INTEGRATED VERSION - Phase 1 + Phase 2 Complete
"""

import sqlite3
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import json
from pathlib import Path
import logging

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
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_indicators_date ON indicators(date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_indicators_name ON indicators(indicator_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_vrp_date ON vrp_data(date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_fed_bs_date ON fed_balance_sheet(date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_move_date ON move_index(date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_repo_date ON repo_market(date)")
            
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
    
    def save_daily_snapshot(self, snapshot: Dict):
        """Save complete daily market snapshot"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO daily_snapshots 
                (date, credit_spread_hy, credit_spread_ig, treasury_10y, fed_funds,
                 vix_spot, vix_contango, put_call_ratio, fear_greed_score, 
                 market_breadth, left_signal)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot.get('date'),
                snapshot.get('credit_spread_hy'),
                snapshot.get('credit_spread_ig'),
                snapshot.get('treasury_10y'),
                snapshot.get('fed_funds'),
                snapshot.get('vix_spot'),
                snapshot.get('vix_contango'),
                snapshot.get('put_call_ratio'),
                snapshot.get('fear_greed_score'),
                snapshot.get('market_breadth'),
                snapshot.get('left_signal')
            ))
            
            conn.commit()
    
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
    
    def get_latest_snapshot(self) -> Optional[Dict]:
        """Get most recent daily snapshot"""
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
            return dict(zip(columns, row))
    
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
    
    def save_vrp_data(self, vrp_analysis: Dict):
        """Save VRP analysis to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            date = datetime.now().strftime('%Y-%m-%d')
            
            cursor.execute("""
                INSERT OR REPLACE INTO vrp_data 
                (date, vix, realized_vol, vrp, regime, expected_6m_return)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                date,
                vrp_analysis.get('vix'),
                vrp_analysis.get('realized_vol'),
                vrp_analysis.get('vrp'),
                vrp_analysis.get('regime'),
                vrp_analysis.get('expected_6m_return')
            ))
            
            conn.commit()
    
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
        Save liquidity history (RRP, TGA, SOFR) to indicators table.
        
        Args:
            df: DataFrame with columns: date, rrp_on, tga, sofr
        """
        if df.empty:
            logger.warning("Empty liquidity DataFrame, skipping save")
            return
        
        try:
            # Save RRP
            if 'rrp_on' in df.columns:
                rrp_df = df[['date', 'rrp_on']].dropna()
                for _, row in rrp_df.iterrows():
                    self.save_indicator('liquidity_rrp', row['date'], float(row['rrp_on']))
            
            # Save TGA
            if 'tga' in df.columns:
                tga_df = df[['date', 'tga']].dropna()
                for _, row in tga_df.iterrows():
                    self.save_indicator('liquidity_tga', row['date'], float(row['tga']))
            
            # Save SOFR
            if 'sofr' in df.columns:
                sofr_df = df[['date', 'sofr']].dropna()
                for _, row in sofr_df.iterrows():
                    self.save_indicator('liquidity_sofr', row['date'], float(row['sofr']))
            
            # Calculate and save net liquidity
            if 'rrp_on' in df.columns and 'tga' in df.columns:
                net_liq_df = df[['date', 'rrp_on', 'tga']].dropna()
                for _, row in net_liq_df.iterrows():
                    net_liq = -(row['rrp_on'] + row['tga'])
                    self.save_indicator('liquidity_net', row['date'], float(net_liq))
            
            logger.info(f"Saved {len(df)} rows of liquidity data")
            
        except Exception as e:
            logger.error(f"Error saving liquidity history: {e}")
    
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

    def save_breadth_data(self, breadth_df):
        """Save breadth history data"""
        if breadth_df.empty:
            return
        
        try:
            breadth_df = breadth_df.copy()
            breadth_df['date'] = breadth_df['date'].dt.strftime('%Y-%m-%d')
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                for _, row in breadth_df.iterrows():
                    cursor.execute("""
                        INSERT OR REPLACE INTO breadth_history 
                        (date, advancing, declining, unchanged, total, breadth_pct, ad_line, ad_diff, mcclellan)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row['date'],
                        int(row['advancing']),
                        int(row['declining']),
                        int(row['unchanged']),
                        int(row['total']),
                        float(row['breadth_pct']),
                        float(row['ad_line']),
                        int(row['ad_diff']),
                        float(row.get('mcclellan', 0))
                    ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving breadth data: {e}")
    
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
