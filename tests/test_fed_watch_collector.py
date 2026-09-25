"""Unit tests for the Fed Watch probability pipeline (data_collectors/fed_watch_collector.py).

All network inputs (Fed funds futures quotes, FRED rates, the clock) are
stubbed, so these run offline and deterministically. The Sep 16, 2026 numbers
are the real inputs observed on 2026-09-13 (ZQU26 96.2725, EFFR 3.63).
"""
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_collectors.fed_watch_collector import (  # noqa: E402
    FedWatchCalculator,
    FedWatchCollector,
)

SEP_16_2026 = datetime(2026, 9, 16)
OCT_28_2026 = datetime(2026, 10, 28)
IN_SEP = datetime(2026, 9, 13)   # inside the meeting month: Aug contract expired
IN_AUG = datetime(2026, 8, 20)   # before the meeting month: Aug contract trading


def _meeting(date: datetime) -> dict:
    return {
        "date": date, "date_str": date.strftime("%b %d, %Y"), "days_until": 3,
        "has_sep": True, "month_code": "U", "year": date.year, "month": date.month,
    }


def make_collector(contracts: dict, *, now: datetime = IN_SEP, effr: float = 3.63,
                   effr_source: str = "FRED", mid: float = 3.625,
                   meeting: datetime = SEP_16_2026,
                   meetings: list | None = None) -> FedWatchCollector:
    """Collector wired to fake inputs. `contracts` maps (year, month) -> implied
    rate; a missing key behaves like an expired / no-data contract."""
    c = FedWatchCollector()
    # The calculator holds the same futures object, so this patches both.
    c.futures.get_implied_rate = lambda y, m: contracts.get((y, m))
    c.get_current_rate = lambda: {
        "upper": mid + 0.125, "lower": mid - 0.125, "mid": mid,
        "range_str": f"{mid - 0.125:.2f}% - {mid + 0.125:.2f}%",
        "effr": effr, "effr_source": effr_source, "as_of": "2026-09-11", "source": "FRED",
    }
    schedule = [_meeting(d) for d in (meetings or [meeting])]
    c.get_upcoming_meetings = lambda n=8: schedule[:n]
    c._now = lambda: now
    return c


class TestPreMeetingRate:
    def test_expired_prior_contract_inside_meeting_month_uses_effr_not_degraded(self):
        # The 2026-09-11 production case. Inside September the August contract
        # has expired by design; EFFR is the correct pre-meeting rate.
        probs = make_collector({(2026, 9): 3.7275}).get_rate_probabilities()
        assert probs["start_rate_source"] == "effr"
        assert probs["anchor_rate"] == pytest.approx(3.63)
        assert probs["degraded"] is False
        assert probs["warnings"] == []

    def test_prior_contract_missing_while_it_should_trade_is_degraded(self):
        probs = make_collector({(2026, 9): 3.7275}, now=IN_AUG).get_rate_probabilities()
        assert probs["start_rate_source"] == "missing_prior_contract_effr"
        assert probs["degraded"] is True
        assert any("ZQQ26.CBT" in w for w in probs["warnings"])

    def test_live_prior_contract_is_used_before_meeting_month(self):
        probs = make_collector({(2026, 8): 3.64, (2026, 9): 3.7275},
                               now=IN_AUG).get_rate_probabilities()
        assert probs["start_rate_source"] == "prior_month_contract"
        assert probs["anchor_rate"] == pytest.approx(3.64)
        assert probs["degraded"] is False

    def test_effr_unavailable_falls_back_to_midpoint_and_is_degraded(self):
        probs = make_collector({(2026, 9): 3.7275},
                               effr_source="target_midpoint").get_rate_probabilities()
        assert probs["start_rate_source"] == "target_midpoint"
        assert probs["anchor_rate"] == pytest.approx(3.625)
        assert probs["degraded"] is True
        assert any("EFFR" in w for w in probs["warnings"])


class TestMidMonthWeighting:
    def test_post_meeting_rate_known_example(self):
        # 16 days at 3.63 + 14 days at x must average 3.7275 over 30 days.
        end = FedWatchCalculator.post_meeting_rate(3.7275, 3.63, meeting_day=16, days_in_month=30)
        assert end == pytest.approx((3.7275 * 30 - 3.63 * 16) / 14)
        assert (3.63 * 16 + end * 14) / 30 == pytest.approx(3.7275)

    def test_no_post_meeting_days_returns_none(self):
        assert FedWatchCalculator.post_meeting_rate(3.7, 3.6, meeting_day=30, days_in_month=30) is None

    def test_sep_16_2026_matches_cme_scale_not_the_diluted_42pct(self):
        probs = make_collector({(2026, 9): 3.7275}).get_rate_probabilities()
        assert probs["end_rate_source"] == "meeting_month_contract"
        assert probs["implied_rate"] == pytest.approx(3.8389, abs=1e-4)
        assert probs["probabilities"]["Hike 25bp"] == pytest.approx(83.6)
        assert probs["probabilities"]["No Change"] == pytest.approx(16.4)
        assert probs["most_likely"] == "Hike 25bp"


class TestEndRateSource:
    def test_next_month_contract_used_when_it_has_no_meeting(self):
        # Oct 28 meeting, no November meeting: the Nov average IS the post-meeting
        # rate, so the (noisy, 3-post-day) October back-out is not needed at all.
        c = make_collector({(2026, 11): 3.96}, now=datetime(2026, 9, 20),
                           effr=3.88, meeting=OCT_28_2026)
        probs = c.get_rate_probabilities()
        assert probs["end_rate_source"] == "next_month_contract"
        assert probs["probabilities"]["Hike 25bp"] == pytest.approx(32.0)
        assert probs["probabilities"]["No Change"] == pytest.approx(68.0)
        assert probs["warnings"] == []

    def test_late_month_back_out_is_flagged_but_not_degraded(self):
        c = make_collector({(2026, 10): 3.8875}, now=datetime(2026, 9, 20),
                           effr=3.88, meeting=OCT_28_2026)
        probs = c.get_rate_probabilities()
        assert probs["end_rate_source"] == "meeting_month_contract"
        assert any("only 3 days" in w for w in probs["warnings"])
        assert probs["degraded"] is False

    def test_missing_meeting_contract_is_neutral_placeholder_and_degraded(self):
        probs = make_collector({}).get_rate_probabilities()
        assert probs["degraded"] is True
        assert probs["data_source"] == "fallback"
        assert any("ZQU26.CBT" in w for w in probs["warnings"])


class TestSummaryPassThrough:
    def test_summary_carries_degraded_and_warnings(self):
        summary = make_collector({}).get_fed_watch_summary()
        assert summary["degraded"] is True
        assert summary["warnings"]

    def test_clean_summary_is_not_degraded(self):
        summary = make_collector({(2026, 9): 3.7275}).get_fed_watch_summary()
        assert summary["degraded"] is False
        assert summary["warnings"] == []
        assert summary["implied_rate"] == pytest.approx(3.8389, abs=1e-4)


class TestRatePath:
    """The path chains meeting to meeting; each step must use the weighted
    post-meeting rate, not the raw meeting-month average."""

    SCHEDULE = [SEP_16_2026, OCT_28_2026]
    # Sep avg 3.7275 (ZQU26), Nov 3.96 — November has no FOMC meeting, so its
    # average IS the rate after the Oct 28 decision.
    CONTRACTS = {(2026, 9): 3.7275, (2026, 11): 3.96}

    def _path(self, **kw):
        c = make_collector(kw.pop("contracts", self.CONTRACTS),
                           meetings=self.SCHEDULE, **kw)
        return c.get_rate_path_expectations()

    def test_first_step_uses_weighted_rate_not_raw_monthly_average(self):
        first = self._path()["path"][0]
        assert first["source"] == "meeting_month_contract"
        assert first["implied_rate"] == pytest.approx(3.839, abs=1e-3)
        assert first["implied_rate"] != pytest.approx(3.7275)  # the old, diluted value

    def test_path_chains_each_meeting_onto_the_previous(self):
        path = self._path()["path"]
        second = path[1]
        assert second["source"] == "next_month_contract"
        assert second["implied_rate"] == pytest.approx(3.96)
        # Move attributed to THIS meeting, measured from the prior meeting's
        # post-meeting rate rather than from today's rate.
        assert second["change_from_prior"] == pytest.approx(3.96 - 3.8389, abs=1e-3)

    def test_terminal_rate_comes_from_the_end_of_the_chain(self):
        res = self._path()
        assert res["terminal_rate"] == pytest.approx(4.0)  # 3.96 -> nearest 12.5bp
        assert res["expected_hikes"] >= 1
        assert res["warnings"] == []

    def test_missing_quotes_carry_forward_flat_and_are_disclosed(self):
        res = self._path(contracts={})
        assert [p["source"] for p in res["path"]] == ["carried_forward"] * 2
        assert all(p["implied_rate"] is None for p in res["path"])
        assert res["terminal_rate"] == pytest.approx(3.625)  # EFFR 3.63 -> 12.5bp grid
        assert any("carried forward flat" in w for w in res["warnings"])

    def test_partial_curve_names_the_first_missing_meeting(self):
        res = self._path(contracts={(2026, 9): 3.7275})
        assert res["path"][0]["source"] == "meeting_month_contract"
        assert res["path"][1]["source"] == "carried_forward"
        assert any("Oct 28, 2026" in w for w in res["warnings"])

    def test_summary_merges_path_warnings_without_marking_page_degraded(self):
        # A far-dated gap must not label the next-meeting probability panel a
        # fallback — those probabilities are fine.
        c = make_collector({(2026, 9): 3.7275}, meetings=self.SCHEDULE)
        summary = c.get_fed_watch_summary()
        assert summary["degraded"] is False
        assert any("carried forward flat" in w for w in summary["warnings"])
