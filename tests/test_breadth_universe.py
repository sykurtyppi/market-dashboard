"""Guards for the hardcoded breadth universes.

The refresh's A/D line runs over breadth_collector.REPRESENTATIVE_STOCKS; a
retired ticker silently shrinks the sample (Yahoo returns no bars and the
collector logs "possibly delisted"). 2026-01-14: MMC became MRSH.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_collectors.breadth_collector import SP500ADLineCalculator as Live  # noqa: E402
from data_collectors.sp500_adline_calculator import SP500ADLineCalculator as Legacy  # noqa: E402

RETIRED = {"MMC": "MRSH"}  # old symbol -> current symbol
TICKER = re.compile(r"^[A-Z][A-Z0-9.\-]{0,5}$")

UNIVERSES = {
    "live representative": Live.REPRESENTATIVE_STOCKS,
    "legacy representative": Legacy.REPRESENTATIVE_STOCKS,
    "legacy full": Legacy.FULL_SP500_STOCKS,
}


def test_live_universe_is_100_unique_well_formed_tickers():
    stocks = Live.REPRESENTATIVE_STOCKS
    assert len(stocks) == 100
    assert len(set(stocks)) == 100
    bad = [t for t in stocks if not TICKER.match(t)]
    assert not bad, bad


def test_no_universe_contains_a_retired_symbol():
    for name, stocks in UNIVERSES.items():
        stale = set(stocks) & set(RETIRED)
        assert not stale, f"{name}: {stale} retired — use {[RETIRED[s] for s in stale]}"


def test_no_universe_has_duplicates():
    for name, stocks in UNIVERSES.items():
        dupes = {t for t in stocks if stocks.count(t) > 1}
        assert not dupes, f"{name}: {dupes}"
