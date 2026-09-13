"""Unit tests for the Fed Watch probability pipeline (data_collectors/fed_watch_collector.py).

All network inputs (Fed funds futures quotes, FRED rates, the meeting schedule
clock) are stubbed, so these run offline and deterministically.
"""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_collectors.fed_watch_collector import FedWatchCollector  # noqa: E402

SEP_16_2026 = datetime(2026, 9, 16)


def _meeting(date: datetime, has_sep: bool = True) -> dict:
    return {
        "date": date,
        "date_str": date.strftime("%b %d, %Y"),
        "days_until": 3,
        "has_sep": has_sep,
        "month_code": "U",
        "year": date.year,
        "month": date.month,
    }


def make_collector(contracts: dict, effr: float = 3.63, mid: float = 3.625,
                   meeting: datetime = SEP_16_2026) -> FedWatchCollector:
    """Collector wired to fake inputs. `contracts` maps (year, month) -> implied rate;
    a missing key behaves like an expired / no-data contract."""
    c = FedWatchCollector()
    # The calculator holds the same futures object, so this patches both.
    c.futures.get_implied_rate = lambda y, m: contracts.get((y, m))
    c.get_current_rate = lambda: {
        "upper": mid + 0.125, "lower": mid - 0.125, "mid": mid,
        "range_str": f"{mid - 0.125:.2f}% - {mid + 0.125:.2f}%",
        "effr": effr, "effr_source": "FRED", "as_of": "2026-09-11", "source": "FRED",
    }
    c.get_upcoming_meetings = lambda n=8: [_meeting(meeting)][:n]
    return c


class TestAnchorDisclosure:
    def test_missing_anchor_contract_is_flagged_degraded(self):
        # August contract absent (expired) — the exact 2026-09-11 production case.
        probs = make_collector({(2026, 9): 3.7275}).get_rate_probabilities()
        assert probs["degraded"] is True
        assert any("ZQQ26.CBT" in w for w in probs["warnings"])

    def test_missing_meeting_contract_is_flagged_degraded(self):
        probs = make_collector({(2026, 8): 3.63}).get_rate_probabilities()
        assert probs["degraded"] is True
        assert probs["data_source"] == "fallback"
        assert any("ZQU26.CBT" in w for w in probs["warnings"])

    def test_summary_carries_warnings_through(self):
        summary = make_collector({(2026, 9): 3.7275}).get_fed_watch_summary()
        assert summary["degraded"] is True
        assert summary["warnings"]

    def test_all_contracts_live_is_not_degraded(self):
        probs = make_collector({(2026, 8): 3.63, (2026, 9): 3.7275}).get_rate_probabilities()
        assert probs["degraded"] is False
        assert probs["warnings"] == []
