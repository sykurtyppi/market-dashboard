"""The breadth universe follows a tracked S&P 500 constituent list.

The 100-stock sample used to be a hand-edited 2024 list: a symbol change made
a name vanish from the A/D line silently, and names removed from the index
kept trading and were never noticed. The seed is now reconciled at startup
against config/sp500_constituents.csv, refreshed by a script, and the breadth
page says when the list is stale or the sample did not fully price.
"""
import importlib.util
import logging
import re
import sys
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from utils.sp500_constituents import (  # noqa: E402
    BREADTH_SAMPLE_SIZE, COLUMNS, PLAUSIBLE_ROWS, STALE_AFTER_DAYS, Constituents, load_constituents, normalize_symbol,
)
from data_collectors.breadth_collector import SP500ADLineCalculator  # noqa: E402

_spec = importlib.util.spec_from_file_location("update_sp500", REPO / "scripts" / "update_sp500_constituents.py")
update_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(update_script)

TICKER = re.compile(r"^[A-Z][A-Z0-9\-]{0,5}$")


def synthetic(as_of=date(2026, 9, 28)) -> Constituents:
    """A tiny index: 6 tech, 2 energy, 2 health — shares 0.6 / 0.2 / 0.2."""
    rows = [(f"T{i}", "Tech") for i in range(1, 7)] + [("E1", "Energy"), ("E2", "Energy"), ("H1", "Health"), ("H2", "Health")]
    frame = pd.DataFrame(rows, columns=["symbol", "gics_sector"])
    return Constituents(frame=frame, as_of=as_of)


class TestLoader:
    def test_normalize_symbol_uses_yahoo_form(self):
        assert normalize_symbol("BRK.B") == "BRK-B"
        assert normalize_symbol(" aapl ") == "AAPL"

    def test_tracked_list_loads_and_is_well_formed(self):
        c = load_constituents()
        assert c is not None
        assert 480 <= len(c.symbols) <= 530
        assert isinstance(c.as_of, date)
        bad = [s for s in c.symbols if not TICKER.match(s)]
        assert not bad, bad
        assert "BRK-B" in c.symbols and not any("." in s for s in c.symbols)
        assert c.frame["gics_sector"].nunique() == 11
        assert c.sector_of("AAPL") == "Information Technology"
        assert c.sector_of("NOT-A-SYMBOL") is None

    def test_missing_or_malformed_file_is_none(self, tmp_path):
        assert load_constituents(tmp_path / "nope.csv") is None
        (tmp_path / "bad.csv").write_text("symbol,security\nAAPL,Apple\n")   # no gics_sector / as_of
        assert load_constituents(tmp_path / "bad.csv") is None

    def test_implausibly_small_file_is_ignored(self, tmp_path, caplog):
        # A truncated list would shrink the sample silently; refuse it instead.
        path = tmp_path / "short.csv"
        path.write_text("symbol,gics_sector,as_of\n" + "\n".join(f"S{i},Tech,2026-09-28" for i in range(5)) + "\n")
        with caplog.at_level(logging.WARNING):
            assert load_constituents(path) is None
        assert "expected ~503" in caplog.text
        assert 503 in PLAUSIBLE_ROWS

    def test_staleness_is_judged_against_a_given_today(self):
        c = synthetic(as_of=date(2026, 1, 1))
        assert c.age_days(today=date(2026, 9, 28)) == 270
        assert c.is_stale(today=date(2026, 9, 28))
        assert not c.is_stale(today=date(2026, 1, 1) + timedelta(days=STALE_AFTER_DAYS))


class TestBuildUniverse:
    def test_no_list_means_the_seed_unchanged(self):
        universe, changes = SP500ADLineCalculator.build_universe(["A", "B"], None)
        assert universe == ["A", "B"]
        assert changes == {"reconciled": False, "dropped": [], "added": [], "unreplaced": [], "as_of": None}

    def test_members_are_kept_in_seed_order(self):
        universe, changes = SP500ADLineCalculator.build_universe(["T2", "E1", "T1"], synthetic())
        assert universe == ["T2", "E1", "T1"]
        assert changes["dropped"] == [] and changes["added"] == []
        assert changes["reconciled"] is True and changes["as_of"] == "2026-09-28"

    def test_dropped_names_are_replaced_from_the_most_under_represented_sector(self):
        # Sample of 5 from a 60/20/20 index should hold ~3 tech, 1 energy,
        # 1 health. After keeping T1, T2, E1 the deficits are tech 1, health 1
        # (tie -> alphabetical sector) then tech 1: replacements H1 then T3.
        seed = ["T1", "T2", "E1", "GONE1", "GONE2"]
        universe, changes = SP500ADLineCalculator.build_universe(seed, synthetic())
        assert universe == ["T1", "T2", "E1", "H1", "T3"]
        assert changes["dropped"] == ["GONE1", "GONE2"]
        assert changes["added"] == ["H1", "T3"]

    def test_duplicate_seed_names_are_collapsed(self):
        universe, _ = SP500ADLineCalculator.build_universe(["T1", "T1", "E1"], synthetic())
        assert universe == ["T1", "E1"]

    def test_exhausted_pools_are_reported_not_hidden(self, caplog):
        # Seed larger than the whole index: replacements run out. The shrink
        # must be visible in the changes and logged at ERROR by the collector.
        tiny = Constituents(pd.DataFrame([("T1", "Tech"), ("T2", "Tech"), ("E1", "Energy")], columns=["symbol", "gics_sector"]), date(2026, 9, 28))
        universe, changes = SP500ADLineCalculator.build_universe(["T1", "GONE1", "GONE2", "GONE3"], tiny)
        # 2/3 Tech index, 4-name sample: Tech deficit 1.67 beats Energy 1.33, so
        # T2 first; then Energy 1.33 beats Tech 0.67, so E1; then nothing left.
        assert universe == ["T1", "T2", "E1"]
        assert changes["added"] == ["T2", "E1"]
        assert changes["unreplaced"] == ["GONE3"]
        with caplog.at_level(logging.ERROR):
            c = SP500ADLineCalculator(constituents=tiny)   # 100-name seed vs a 3-name index
        assert len(c.stocks) == 3
        assert len(c.universe_changes["unreplaced"]) == BREADTH_SAMPLE_SIZE - 3
        assert "no replacement available" in caplog.text

    def test_size_is_preserved_and_deterministic(self):
        seed = ["T1", "GONE1", "GONE2", "GONE3"]
        first = SP500ADLineCalculator.build_universe(seed, synthetic())
        second = SP500ADLineCalculator.build_universe(seed, synthetic())
        assert first == second
        assert len(first[0]) == len(seed) and len(set(first[0])) == len(seed)


class TestLiveUniverse:
    def test_collector_universe_is_drawn_from_the_tracked_list(self):
        c = SP500ADLineCalculator()
        members = set(load_constituents().symbols)
        assert len(c.stocks) == c.SAMPLE_SIZE
        assert set(c.stocks) <= members
        assert c.universe_changes["reconciled"] is True

    def test_a_seed_name_that_leaves_the_index_is_replaced_at_startup(self):
        full = load_constituents()
        without_aapl = Constituents(frame=full.frame[full.frame["symbol"] != "AAPL"].reset_index(drop=True), as_of=full.as_of)
        c = SP500ADLineCalculator(constituents=without_aapl)
        assert "AAPL" not in c.stocks
        assert len(c.stocks) == c.SAMPLE_SIZE
        assert c.universe_changes["dropped"] == ["AAPL"]
        assert len(c.universe_changes["added"]) == 1 and c.universe_changes["added"][0] in without_aapl.symbols


class TestUpdateScript:
    HTML = """<html><body><table>
    <tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>GICS Sub-Industry</th>
        <th>Headquarters Location</th><th>Date added</th><th>CIK</th><th>Founded</th></tr>
    <tr><td>MMM</td><td>3M</td><td>Industrials</td><td>Conglomerates</td><td>St Paul</td><td>1957-03-04</td><td>66740</td><td>1902</td></tr>
    <tr><td>BRK.B</td><td>Berkshire</td><td>Financials</td><td>Multi-Sector Holdings</td><td>Omaha</td><td>2010-02-16</td><td>1067983</td><td>1839</td></tr>
    <tr><td>AAPL</td><td>Apple</td><td>Information Technology</td><td>Hardware</td><td>Cupertino</td><td>1982-11-30</td><td>320193</td><td>1977</td></tr>
    </table></body></html>"""

    def test_parses_normalizes_and_sorts(self):
        frame = update_script.parse_constituents(self.HTML, as_of="2026-09-28")
        assert list(frame.columns) == list(COLUMNS)
        assert frame["symbol"].tolist() == ["AAPL", "BRK-B", "MMM"]
        assert set(frame["as_of"]) == {"2026-09-28"}

    def test_layout_change_is_an_error_not_a_bad_file(self):
        with pytest.raises(ValueError):
            update_script.parse_constituents("<table><tr><th>Ticker</th></tr><tr><td>AAPL</td></tr></table>", "2026-09-28")


def breadth_history(total: int, days: int = 45) -> pd.DataFrame:
    dates = pd.bdate_range(end="2026-09-25", periods=days).strftime("%Y-%m-%d")
    adv = [int(total * 0.6)] * days
    return pd.DataFrame({
        "date": dates, "advancing": adv, "declining": [total - a for a in adv], "unchanged": 0,
        "total": total, "breadth_pct": 60.0, "ad_line": 10000.0, "ad_diff": [2 * a - total for a in adv], "mcclellan": 0.0,
    })


class TestBreadthPageDisclosure:
    def _warnings(self, total, constituents):
        from api.pages_service import build_breadth
        fake_db = SimpleNamespace(get_breadth_history=lambda days=120: breadth_history(total))
        with patch("api.pages_service.get_db", return_value=fake_db), \
             patch("api.pages_service.load_constituents", return_value=constituents):
            return build_breadth()["warnings"]

    def test_full_coverage_and_fresh_list_add_no_warnings(self):
        assert self._warnings(100, synthetic(as_of=date.today())) == []

    def test_partial_coverage_is_disclosed(self):
        w = self._warnings(85, synthetic(as_of=date.today()))
        assert any("85 of 100 sampled stocks" in x for x in w)

    def test_stale_list_is_disclosed(self):
        w = self._warnings(100, synthetic(as_of=date(2025, 1, 1)))
        assert any("constituent list is" in x and "update_sp500_constituents" in x for x in w)

    def test_missing_list_is_disclosed(self):
        w = self._warnings(100, None)
        assert any("constituent list missing" in x for x in w)
