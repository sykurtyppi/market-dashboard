#!/usr/bin/env python
"""Refresh config/sp500_constituents.csv from Wikipedia's S&P 500 table.

Run it whenever the index reconstitutes (quarterly, plus ad-hoc changes); the
breadth page warns when the file is older than utils.sp500_constituents.
STALE_AFTER_DAYS. Prints what changed. Exit code is non-zero if the fetch
fails or the result looks wrong, so a cron cannot quietly write garbage.

    python scripts/update_sp500_constituents.py
"""
from __future__ import annotations

import io
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.sp500_constituents import (  # noqa: E402
    COLUMNS, CONSTITUENTS_PATH, PLAUSIBLE_ROWS, load_constituents, normalize_symbol,
)

SOURCE_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
USER_AGENT = "market-dashboard/1.0 (open-source research dashboard; github.com/sykurtyppi/market-dashboard)"


def fetch_html(url: str = SOURCE_URL) -> str:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    return response.text


def parse_constituents(html: str, as_of: str) -> pd.DataFrame:
    """The first table on the page is the current constituent list.

    That position is an assumption about Wikipedia's layout. If it changes,
    the column check below raises and main() exits non-zero without writing,
    so the failure is loud rather than a wrong file.
    """
    table = pd.read_html(io.StringIO(html))[0]
    wanted = {
        "Symbol": "symbol", "Security": "security", "GICS Sector": "gics_sector",
        "GICS Sub-Industry": "gics_sub_industry", "Date added": "date_added",
    }
    missing = [c for c in wanted if c not in table.columns]
    if missing:
        raise ValueError(f"constituent table is missing columns {missing}; page layout changed?")
    frame = table[list(wanted)].rename(columns=wanted).astype(str)
    frame["symbol"] = frame["symbol"].map(normalize_symbol)
    frame["as_of"] = as_of
    frame = frame.drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)
    return frame[list(COLUMNS)]


def main() -> int:
    as_of = datetime.now().strftime("%Y-%m-%d")
    try:
        frame = parse_constituents(fetch_html(), as_of)
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1
    if len(frame) not in PLAUSIBLE_ROWS:
        print(f"FAILED: parsed {len(frame)} rows, expected ~503 — not writing", file=sys.stderr)
        return 1

    before = load_constituents(CONSTITUENTS_PATH)
    old = set(before.symbols) if before else set()
    new = set(frame["symbol"])

    CONSTITUENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(CONSTITUENTS_PATH, index=False)

    print(f"wrote {len(frame)} constituents as of {as_of} -> {CONSTITUENTS_PATH}")
    if before:
        print(f"previous list as of {before.as_of}: +{len(new - old)} added, -{len(old - new)} removed")
        for s in sorted(new - old):
            print(f"  + {s}")
        for s in sorted(old - new):
            print(f"  - {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
