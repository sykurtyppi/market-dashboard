"""
CTA Price Database - SQLite-backed OHLCV store (Thread-safe, robust column handling)
"""

from __future__ import annotations
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import logging

logger = logging.getLogger(__name__)

DDL_PRICES = """
CREATE TABLE IF NOT EXISTS prices (
    symbol      TEXT NOT NULL,
    date        TEXT NOT NULL,
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    adj_close   REAL,
    volume      REAL,
    PRIMARY KEY (symbol, date)
);
"""

DDL_META = """
CREATE TABLE IF NOT EXISTS symbol_meta (
    symbol          TEXT PRIMARY KEY,
    first_date      TEXT,
    last_date       TEXT,
    last_updated_at TEXT
);
"""


@dataclass
class PriceDBConfig:
    db_path: Path
    pragma_journal_mode: str = "WAL"
    pragma_synchronous: str = "NORMAL"


class PriceDB:
    """Thread-safe SQLite OHLCV store with robust column normalization"""

    def __init__(self, config: PriceDBConfig):
        self.config = config
        self._conn = sqlite3.connect(
            str(config.db_path),
            check_same_thread=False,  # Thread-safe for dashboard workers
        )
        self._conn.execute(f"PRAGMA journal_mode={config.pragma_journal_mode};")
        self._conn.execute(f"PRAGMA synchronous={config.pragma_synchronous};")
        self._conn.execute(DDL_PRICES)
        self._conn.execute(DDL_META)
        self._conn.commit()

    @staticmethod
    def _normalize_ohlc_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Map vendor columns to canonical: open, high, low, close, adj_close, volume
        Handles MultiIndex columns from yfinance: ('Adj Close', 'SPY') -> 'adj_close'
        """
        df = df.copy()

        # Handle MultiIndex columns (yfinance returns these for single symbols)
        if isinstance(df.columns, pd.MultiIndex):
            # Extract first level: ('Adj Close', 'SPY') -> 'Adj Close'
            df.columns = df.columns.get_level_values(0)

        # Normalize all column names to lowercase
        df.columns = [str(c).strip().lower() for c in df.columns]

        # Handle Yahoo's "adj close" (with space)
        if "adj close" in df.columns and "adj_close" not in df.columns:
            df.rename(columns={"adj close": "adj_close"}, inplace=True)

        # If adj_close still missing, use close as fallback
        if "adj_close" not in df.columns and "close" in df.columns:
            df["adj_close"] = df["close"]

        return df

    def upsert_ohlc(self, symbol: str, df: pd.DataFrame) -> None:
        """Upsert OHLCV data with robust error handling"""
        if df is None or df.empty:
            logger.warning(f"{symbol}: Empty DataFrame, skipping")
            return

        df = self._normalize_ohlc_columns(df)

        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        df["date"] = df.index.date.astype(str)

        # Validate required columns
        required = ["open", "high", "low", "close", "adj_close"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"{symbol}: missing required column '{col}'")

        vol_present = "volume" in df.columns

        rows = []
        for _, r in df.iterrows():
            rows.append(
                (
                    symbol,
                    r["date"],
                    float(r["open"]) if pd.notna(r["open"]) else None,
                    float(r["high"]) if pd.notna(r["high"]) else None,
                    float(r["low"]) if pd.notna(r["low"]) else None,
                    float(r["close"]) if pd.notna(r["close"]) else None,
                    float(r["adj_close"]) if pd.notna(r["adj_close"]) else None,
                    float(r["volume"]) if vol_present and pd.notna(r["volume"]) else None,
                )
            )

        with self._conn:
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO prices
                    (symbol, date, open, high, low, close, adj_close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                rows,
            )

        first_date = min(x[1] for x in rows)
        last_date = max(x[1] for x in rows)

        with self._conn:
            self._conn.execute(
                """
                INSERT INTO symbol_meta(symbol, first_date, last_date, last_updated_at)
                VALUES (?, ?, ?, datetime('now'))
                ON CONFLICT(symbol) DO UPDATE SET
                    first_date = MIN(first_date, excluded.first_date),
                    last_date  = MAX(last_date,  excluded.last_date),
                    last_updated_at = excluded.last_updated_at;
                """,
                (symbol, first_date, last_date),
            )

        logger.info(f"{symbol}: upserted {len(rows)} rows ({first_date} → {last_date})")

    def get_history(
        self,
        symbols: Sequence[str],
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV (tall format)"""
        if not symbols:
            return pd.DataFrame()

        placeholders = ",".join("?" for _ in symbols)
        params = list(symbols)

        where = [f"symbol IN ({placeholders})"]
        if start:
            where.append("date >= ?")
            params.append(start)
        if end:
            where.append("date <= ?")
            params.append(end)

        sql = f"""
        SELECT symbol, date, open, high, low, close, adj_close, volume
        FROM prices
        WHERE {' AND '.join(where)}
        ORDER BY date ASC, symbol ASC;
        """

        df = pd.read_sql_query(sql, self._conn, params=params)
        if df.empty:
            return df

        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        return df

    def get_last_date(self, symbol: str) -> Optional[str]:
        """Return last available date (YYYY-MM-DD) or None"""
        cur = self._conn.execute(
            "SELECT last_date FROM symbol_meta WHERE symbol = ?;",
            (symbol,),
        )
        row = cur.fetchone()
        return row[0] if row and row[0] else None

    def list_symbols(self) -> list[str]:
        cur = self._conn.execute("SELECT symbol FROM symbol_meta ORDER BY symbol ASC;")
        return [r[0] for r in cur.fetchall()]

    def close(self) -> None:
        self._conn.close()
