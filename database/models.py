"""
SQLite database manager for storing historical market data
"""

import sqlite3
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import json
from pathlib import Path

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
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_indicators_date ON indicators(date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_indicators_name ON indicators(indicator_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)")
            
            conn.commit()
    
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


# Test function
if __name__ == "__main__":
    print("\n" + "="*80)
    print("DATABASE MANAGER TEST")
    print("="*80)
    
    db = DatabaseManager("data/test_market_data.db")
    
    # Test 1: Save indicator
    print("\n Test 1: Save indicator")
    db.save_indicator('test_indicator', datetime.now(), 3.09, 'TEST_SERIES')
    print("✓ Indicator saved")
    
    # Test 2: Save signal
    print("\n Test 2: Save signal")
    db.save_signal('LEFT', 'BUY', 75.0, {'reason': 'Credit spreads below threshold'})
    print("✓ Signal saved")
    
    # Test 3: Save daily snapshot
    print("\n Test 3: Save daily snapshot")
    snapshot = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'credit_spread_hy': 3.09,
        'treasury_10y': 4.45,
        'fear_greed_score': 21.0,
        'left_signal': 'NEUTRAL'
    }
    db.save_daily_snapshot(snapshot)
    print("✓ Snapshot saved")
    
    # Test 4: Retrieve data
    print("\n Test 4: Retrieve latest snapshot")
    latest = db.get_latest_snapshot()
    if latest:
        print(f"✓ Retrieved snapshot from {latest['date']}")
        print(f"  HYG OAS: {latest['credit_spread_hy']}")
        print(f"  LEFT Signal: {latest['left_signal']}")
    
    print("\n" + "="*80)
    print("✓ ALL DATABASE TESTS PASSED")
    print("="*80 + "\n")
# Breadth Snapshot Table
class BreadthSnapshot(Base):
    __tablename__ = "breadth_snapshots"

    id = Column(Integer, primary_key=True)
    timestamp = Column(Date, index=True)

    ma50_pct = Column(Float)
    ma200_pct = Column(Float)
    pct_highs = Column(Float)
    pct_lows = Column(Float)
    ad_ratio = Column(Float)
    breadth_thrust = Column(Float)
    up_volume_pct = Column(Float)
    volume_ratio = Column(Float)

