"""S&P 500 constituent list: a tracked snapshot, refreshed by a script.

The breadth sample is drawn from this list, so membership drift — removals,
mergers, symbol changes — is a data-maintenance task (run
scripts/update_sp500_constituents.py) rather than a code edit. Symbols use
Yahoo's form (BRK-B, not BRK.B).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

CONSTITUENTS_PATH = Path(__file__).resolve().parent.parent / "config" / "sp500_constituents.csv"
COLUMNS = ("symbol", "security", "gics_sector", "gics_sub_industry", "date_added", "as_of")
# The index reconstitutes quarterly; a list that has missed two of those is stale.
STALE_AFTER_DAYS = 120


def normalize_symbol(symbol: str) -> str:
    """Wikipedia writes class shares as BRK.B; Yahoo wants BRK-B."""
    return str(symbol).strip().upper().replace(".", "-")


@dataclass(frozen=True)
class Constituents:
    frame: pd.DataFrame
    as_of: date

    @property
    def symbols(self) -> list[str]:
        return self.frame["symbol"].tolist()

    def sector_of(self, symbol: str) -> Optional[str]:
        hit = self.frame.loc[self.frame["symbol"] == symbol, "gics_sector"]
        return None if hit.empty else str(hit.iloc[0])

    def age_days(self, today: Optional[date] = None) -> int:
        return ((today or datetime.now().date()) - self.as_of).days

    def is_stale(self, today: Optional[date] = None) -> bool:
        return self.age_days(today) > STALE_AFTER_DAYS


def load_constituents(path: Path = CONSTITUENTS_PATH) -> Optional[Constituents]:
    """The tracked list, or None (logged) if it is missing or malformed."""
    try:
        frame = pd.read_csv(path, dtype=str).fillna("")
    except FileNotFoundError:
        logger.warning(f"S&P 500 constituent list not found at {path}")
        return None
    except Exception as e:  # malformed file: unreadable is as good as missing
        logger.warning(f"S&P 500 constituent list at {path} could not be read: {e}")
        return None

    missing = [c for c in ("symbol", "gics_sector", "as_of") if c not in frame.columns]
    if missing or frame.empty:
        logger.warning(f"S&P 500 constituent list at {path} is missing columns {missing} or empty")
        return None
    try:
        as_of = date.fromisoformat(frame["as_of"].iloc[0])
    except ValueError:
        logger.warning(f"S&P 500 constituent list has an unparseable as_of: {frame['as_of'].iloc[0]!r}")
        return None

    frame = frame.assign(symbol=frame["symbol"].map(normalize_symbol)).drop_duplicates("symbol")
    return Constituents(frame=frame.reset_index(drop=True), as_of=as_of)
