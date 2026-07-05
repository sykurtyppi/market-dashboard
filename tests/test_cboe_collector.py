"""
Test Suite for CBOECollector (data_collectors/cboe_collector.py)

Covers:
- VIX / VIX3M fetching with mocked yfinance (no network access)
- Memory cache, fetch-miss cooldown, rate-limit cooldown, last-known-good fallback
- Persistent LKG store redirected to tmp_path (never touches data/cache)
- VIX3M estimation via smooth multiplier interpolation
- Contango calculation and term structure regime classification
- Put/call ratio parsing (SPY options chain, CBOE HTML scrape) and fallbacks
- VVIX signal thresholds and put/call sentiment interpretation

Run with: python -m pytest tests/test_cboe_collector.py -v
"""

import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import Mock, patch

import pandas as pd
import pytest
from yfinance.exceptions import YFRateLimitError

from data_collectors.cboe_collector import CBOECollector

YF_TICKER = "data_collectors.cboe_collector.yf.Ticker"


def make_history_df(close_value: float) -> pd.DataFrame:
    """Build a minimal yfinance-style history DataFrame."""
    index = pd.DatetimeIndex([pd.Timestamp("2026-07-01"), pd.Timestamp("2026-07-02")])
    return pd.DataFrame(
        {"Close": [close_value - 0.5, close_value], "Volume": [0, 0]},
        index=index,
    )


def make_ticker_mock(close_value: float) -> Mock:
    """Build a yf.Ticker mock whose history() returns a valid DataFrame."""
    ticker = Mock()
    ticker.history.return_value = make_history_df(close_value)
    return ticker


@pytest.fixture(autouse=True)
def isolate_collector_state(tmp_path, monkeypatch):
    """
    Reset class-level caches and redirect the persistent LKG store to tmp_path
    so tests never touch data/cache and never depend on execution order.
    """
    monkeypatch.setattr(CBOECollector, "_LKG_PATH", tmp_path / "cboe_lkg.json")
    CBOECollector._CACHE.clear()
    CBOECollector._MISS_CACHE.clear()
    CBOECollector._LAST_RATE_LIMIT_AT = None
    yield
    CBOECollector._CACHE.clear()
    CBOECollector._MISS_CACHE.clear()
    CBOECollector._LAST_RATE_LIMIT_AT = None


@pytest.fixture
def collector():
    return CBOECollector()


class TestGetVix:
    """VIX fetching, caching, and fallback behavior"""

    def test_returns_latest_close_from_yfinance(self, collector):
        # Arrange
        with patch(YF_TICKER, return_value=make_ticker_mock(18.5)) as ticker_mock:
            # Act
            result = collector.get_vix()

        # Assert
        assert result == pytest.approx(18.5)
        ticker_mock.assert_called_once_with("^VIX")

    def test_returns_none_when_history_empty_and_no_cache(self, collector):
        # Arrange
        empty_ticker = Mock()
        empty_ticker.history.return_value = pd.DataFrame()

        # Act
        with patch(YF_TICKER, return_value=empty_ticker):
            result = collector.get_vix()

        # Assert
        assert result is None

    def test_returns_none_on_fetch_exception_with_no_cache(self, collector):
        # Arrange
        broken_ticker = Mock()
        broken_ticker.history.side_effect = ConnectionError("network down")

        # Act
        with patch(YF_TICKER, return_value=broken_ticker):
            result = collector.get_vix()

        # Assert
        assert result is None

    def test_second_call_served_from_memory_cache_without_refetch(self, collector):
        # Arrange - first call populates the cache
        with patch(YF_TICKER, return_value=make_ticker_mock(21.0)):
            first = collector.get_vix()

        # Act - second call must not hit yfinance at all
        with patch(YF_TICKER) as ticker_mock:
            second = collector.get_vix()

        # Assert
        assert first == pytest.approx(21.0)
        assert second == pytest.approx(21.0)
        ticker_mock.assert_not_called()

    def test_falls_back_to_stale_cache_on_fetch_error(self, collector):
        # Arrange - seed a stale in-memory entry (older than fresh TTL, within stale TTL)
        CBOECollector._CACHE["vix"] = {"ts": time.time() - 300, "value": 33.0}
        broken_ticker = Mock()
        broken_ticker.history.side_effect = RuntimeError("boom")

        # Act
        with patch(YF_TICKER, return_value=broken_ticker):
            result = collector.get_vix()

        # Assert - last-known-good is used
        assert result == pytest.approx(33.0)

    def test_rate_limit_error_sets_cooldown_and_uses_stale_value(self, collector):
        # Arrange
        CBOECollector._CACHE["vix"] = {"ts": time.time() - 300, "value": 27.5}
        limited_ticker = Mock()
        limited_ticker.history.side_effect = YFRateLimitError()

        # Act
        with patch(YF_TICKER, return_value=limited_ticker):
            result = collector.get_vix()

        # Assert
        assert result == pytest.approx(27.5)
        assert CBOECollector._LAST_RATE_LIMIT_AT is not None
        assert collector._rate_limited_recently() is True

    def test_fetch_miss_cooldown_prevents_immediate_refetch(self, collector):
        # Arrange - first call misses (empty data)
        empty_ticker = Mock()
        empty_ticker.history.return_value = pd.DataFrame()
        with patch(YF_TICKER, return_value=empty_ticker):
            assert collector.get_vix() is None

        # Act - second call within miss cooldown must not call yfinance
        with patch(YF_TICKER) as ticker_mock:
            result = collector.get_vix()

        # Assert
        assert result is None
        ticker_mock.assert_not_called()

    def test_persistent_lkg_store_survives_memory_cache_clear(self, collector):
        # Arrange - populate memory + persistent store, then wipe memory cache
        with patch(YF_TICKER, return_value=make_ticker_mock(25.0)):
            collector.get_vix()
        CBOECollector._CACHE.clear()
        fresh_collector = CBOECollector()

        # Act - fresh collector should read the persisted value without fetching
        with patch(YF_TICKER) as ticker_mock:
            result = fresh_collector.get_vix()

        # Assert
        assert result == pytest.approx(25.0)
        ticker_mock.assert_not_called()


class TestGetVix3m:
    """VIX3M fetching and estimation fallback"""

    def test_returns_real_vix3m_when_ticker_has_data(self, collector):
        # Arrange
        with patch(YF_TICKER, return_value=make_ticker_mock(19.8)):
            # Act
            result = collector.get_vix3m()

        # Assert - real data, nothing marked as estimated
        assert result == pytest.approx(19.8)
        assert collector.estimated_fields == []

    def test_estimates_from_vix_when_vix3m_unavailable(self, collector):
        # Arrange - ^VIX3M empty, ^VIX returns 20 (multiplier 1.01 at anchor)
        def fake_ticker(symbol):
            if symbol == "^VIX":
                return make_ticker_mock(20.0)
            empty = Mock()
            empty.history.return_value = pd.DataFrame()
            return empty

        # Act
        with patch(YF_TICKER, side_effect=fake_ticker):
            result = collector.get_vix3m()

        # Assert - estimated as VIX * 1.01 and tracked
        assert result == pytest.approx(20.0 * 1.01)
        assert any(e["field"] == "vix3m" for e in collector.estimated_fields)

    def test_estimation_not_tracked_when_track_estimation_false(self, collector):
        # Arrange
        def fake_ticker(symbol):
            if symbol == "^VIX":
                return make_ticker_mock(20.0)
            empty = Mock()
            empty.history.return_value = pd.DataFrame()
            return empty

        # Act
        with patch(YF_TICKER, side_effect=fake_ticker):
            result = collector.get_vix3m(track_estimation=False)

        # Assert
        assert result == pytest.approx(20.2)
        assert collector.estimated_fields == []

    def test_returns_none_when_vix_and_vix3m_both_unavailable(self, collector):
        # Arrange
        empty_ticker = Mock()
        empty_ticker.history.return_value = pd.DataFrame()

        # Act
        with patch(YF_TICKER, return_value=empty_ticker):
            result = collector.get_vix3m()

        # Assert
        assert result is None

    def test_cached_estimated_value_marks_estimation_on_new_instance(self, collector):
        # Arrange - first collector caches an estimated VIX3M (dict with estimated=True)
        def fake_ticker(symbol):
            if symbol == "^VIX":
                return make_ticker_mock(20.0)
            empty = Mock()
            empty.history.return_value = pd.DataFrame()
            return empty

        with patch(YF_TICKER, side_effect=fake_ticker):
            collector.get_vix3m()

        other = CBOECollector()

        # Act - served from class-level cache, no fetch
        with patch(YF_TICKER) as ticker_mock:
            result = other.get_vix3m()

        # Assert
        assert result == pytest.approx(20.2)
        assert any(e["field"] == "vix3m" for e in other.estimated_fields)
        ticker_mock.assert_not_called()


class TestInterpolateVix3mMultiplier:
    """Smooth piecewise-linear VIX3M multiplier"""

    def test_clamps_to_strong_contango_below_lowest_anchor(self, collector):
        multiplier, regime = collector._interpolate_vix3m_multiplier(8.0)

        assert multiplier == pytest.approx(1.08)
        assert regime == "contango"

    def test_clamps_to_steep_backwardation_above_highest_anchor(self, collector):
        multiplier, regime = collector._interpolate_vix3m_multiplier(100.0)

        assert multiplier == pytest.approx(0.75)
        assert regime == "steep backwardation"

    def test_interpolates_linearly_between_anchor_points(self, collector):
        # Midpoint between VIX=10 (1.08) and VIX=15 (1.05)
        multiplier, regime = collector._interpolate_vix3m_multiplier(12.5)

        assert multiplier == pytest.approx(1.065)
        assert regime == "contango"

    def test_backwardation_regime_for_elevated_vix(self, collector):
        # Midpoint between VIX=25 (0.97) and VIX=35 (0.90)
        multiplier, regime = collector._interpolate_vix3m_multiplier(30.0)

        assert multiplier == pytest.approx(0.935)
        assert regime == "backwardation"

    def test_anchor_point_returns_exact_multiplier(self, collector):
        multiplier, regime = collector._interpolate_vix3m_multiplier(20.0)

        assert multiplier == pytest.approx(1.01)
        assert regime == "mild contango"


class TestGetRealContango:
    """Contango calculation from VIX / VIX3M"""

    def test_computes_contango_from_real_values(self, collector):
        # Arrange - VIX=20, VIX3M=21 → (21-20)/20*100 = 5%
        with patch.object(collector, "get_vix", return_value=20.0), \
             patch.object(collector, "get_vix3m", return_value=21.0):
            # Act
            result = collector.get_real_contango()

        # Assert
        assert result == pytest.approx(5.0)

    def test_negative_contango_in_backwardation(self, collector):
        # Arrange - VIX=30, VIX3M=27 → -10%
        with patch.object(collector, "get_vix", return_value=30.0), \
             patch.object(collector, "get_vix3m", return_value=27.0):
            result = collector.get_real_contango()

        assert result == pytest.approx(-10.0)

    def test_estimates_contango_from_multiplier_when_vix3m_missing(self, collector):
        # Arrange - VIX=40, no VIX3M → multiplier 0.8733 → -12.67%
        with patch.object(collector, "get_vix", return_value=40.0), \
             patch.object(collector, "get_vix3m", return_value=None):
            # Act
            result = collector.get_real_contango()

        # Assert
        assert result == pytest.approx(-12.67, abs=0.01)
        assert any(e["field"] == "vix_contango" for e in collector.estimated_fields)

    def test_returns_none_when_vix_unavailable(self, collector):
        with patch.object(collector, "get_vix", return_value=None), \
             patch.object(collector, "get_vix3m", return_value=None):
            result = collector.get_real_contango()

        assert result is None


class TestGetTermStructureRegime:
    """Term structure regime classification"""

    @pytest.mark.parametrize("contango,expected_regime,expected_signal", [
        (7.5, "STRONG_CONTANGO", "BULLISH"),
        (2.0, "MILD_CONTANGO", "MILDLY_BULLISH"),
        (-1.5, "FLAT", "NEUTRAL"),
        (-6.0, "BACKWARDATION", "BEARISH"),
        (-15.0, "CRISIS_BACKWARDATION", "EXTREME_FEAR"),
    ])
    def test_classifies_regime_from_contango(self, collector, contango,
                                             expected_regime, expected_signal):
        # Arrange / Act
        with patch.object(collector, "get_real_contango", return_value=contango), \
             patch.object(collector, "get_vix", return_value=20.0):
            result = collector.get_term_structure_regime()

        # Assert
        assert result["regime"] == expected_regime
        assert result["signal"] == expected_signal
        assert result["contango_pct"] == contango
        assert result["vix"] == 20.0
        assert result["is_estimated"] is False

    def test_returns_none_when_contango_unavailable(self, collector):
        with patch.object(collector, "get_real_contango", return_value=None), \
             patch.object(collector, "get_vix", return_value=None):
            result = collector.get_term_structure_regime()

        assert result is None

    def test_flags_estimated_contango(self, collector):
        # Arrange
        collector._mark_estimated("vix_contango", "test")

        # Act
        with patch.object(collector, "get_real_contango", return_value=3.0), \
             patch.object(collector, "get_vix", return_value=18.0):
            result = collector.get_term_structure_regime()

        # Assert
        assert result["is_estimated"] is True


class TestVvixSignal:
    """VVIX buy-signal thresholds (from config/parameters.yaml)"""

    @pytest.mark.parametrize("vvix,expected_signal", [
        (130.0, "STRONG BUY"),   # >= 120
        (115.0, "BUY ALERT"),    # >= 110
        (95.0, "NEUTRAL"),       # 80-110
        (70.0, "CAUTION"),       # < 80
    ])
    def test_signal_matches_vvix_level(self, collector, vvix, expected_signal):
        with patch.object(collector, "get_vvix", return_value=vvix):
            result = collector.get_vvix_signal()

        assert result["signal"] == expected_signal
        assert result["level"] == vvix

    def test_returns_unavailable_when_vvix_missing(self, collector):
        with patch.object(collector, "get_vvix", return_value=None):
            result = collector.get_vvix_signal()

        assert result["signal"] == "UNAVAILABLE"
        assert result["level"] is None
        assert result["strength"] == 0

    def test_strong_buy_strength_capped_at_100(self, collector):
        with patch.object(collector, "get_vvix", return_value=200.0):
            result = collector.get_vvix_signal()

        assert result["strength"] == 100


class TestInterpretPutCall:
    """Put/call ratio sentiment interpretation"""

    @pytest.mark.parametrize("ratio,expected_signal", [
        (1.4, "CONTRARIAN BUY"),   # > 1.3 extreme fear
        (1.2, "BEARISH"),          # > 1.1
        (1.0, "NEUTRAL"),          # > 0.9
        (0.8, "BULLISH"),          # > 0.7
        (0.6, "CONTRARIAN SELL"),  # <= 0.7 extreme greed
    ])
    def test_sentiment_signal_for_ratio(self, collector, ratio, expected_signal):
        result = collector._interpret_put_call(ratio)

        assert result["signal"] == expected_signal
        assert {"reading", "signal", "color", "description"} <= set(result)


class TestScrapeCboePutCall:
    """CBOE daily statistics page scraping"""

    def test_parses_equity_put_call_from_html_table(self, collector):
        # Arrange - minimal but structurally accurate CBOE-style table
        html = """
        <html><body>
        <table>
            <tr><th>Ratio</th><th>Value</th></tr>
            <tr><td>Equity Put/Call Ratio</td><td>0.65</td></tr>
        </table>
        </body></html>
        """
        response = Mock(status_code=200, text=html)

        # Act
        with patch.object(collector.session, "get", return_value=response):
            result = collector._scrape_cboe_put_call()

        # Assert
        assert result == {"equity_pc": 0.65, "source": "CBOE"}

    def test_returns_none_when_page_blocked(self, collector):
        # Arrange
        response = Mock(status_code=403, text="")

        # Act
        with patch.object(collector.session, "get", return_value=response):
            result = collector._scrape_cboe_put_call()

        # Assert
        assert result is None

    def test_returns_none_when_no_put_call_table_present(self, collector):
        # Arrange
        html = "<html><body><table><tr><td>Volume</td><td>123</td></tr></table></body></html>"
        response = Mock(status_code=200, text=html)

        # Act
        with patch.object(collector.session, "get", return_value=response):
            result = collector._scrape_cboe_put_call()

        # Assert
        assert result is None

    def test_returns_none_when_value_cell_not_numeric(self, collector):
        # Arrange - table mentions put/call but the value is malformed
        html = """
        <table>
            <tr><td>Equity Put/Call Ratio</td><td>N/A</td></tr>
        </table>
        """
        response = Mock(status_code=200, text=html)

        # Act
        with patch.object(collector.session, "get", return_value=response):
            result = collector._scrape_cboe_put_call()

        # Assert
        assert result is None

    def test_returns_none_on_request_exception(self, collector):
        with patch.object(collector.session, "get", side_effect=ConnectionError("refused")):
            result = collector._scrape_cboe_put_call()

        assert result is None


class TestSpyPutCallRatio:
    """SPY options chain put/call ratio parsing"""

    @staticmethod
    def make_spy_ticker(put_vols, call_vols, put_ois, call_ois,
                        expiry="2026-07-17") -> Mock:
        chain = Mock()
        chain.puts = pd.DataFrame({"volume": put_vols, "openInterest": put_ois})
        chain.calls = pd.DataFrame({"volume": call_vols, "openInterest": call_ois})
        ticker = Mock()
        ticker.options = (expiry,)
        ticker.option_chain.return_value = chain
        return ticker

    def test_computes_volume_and_oi_ratios_from_chain(self, collector):
        # Arrange - put vol 300 / call vol 200 = 1.5; put OI 900 / call OI 600 = 1.5
        ticker = self.make_spy_ticker(
            put_vols=[100, 200], call_vols=[150, 50],
            put_ois=[400, 500], call_ois=[300, 300],
        )

        # Act
        with patch(YF_TICKER, return_value=ticker):
            result = collector.get_spy_put_call_ratio()

        # Assert
        assert result["volume_ratio"] == pytest.approx(1.5)
        assert result["oi_ratio"] == pytest.approx(1.5)
        assert result["ratio"] == pytest.approx(1.5)
        assert result["put_volume"] == 300
        assert result["call_volume"] == 200
        assert result["expiry"] == "2026-07-17"
        assert result["source"] == "SPY_OPTIONS"

    def test_nan_volumes_treated_as_zero(self, collector):
        # Arrange - NaN volumes must not poison the sums
        ticker = self.make_spy_ticker(
            put_vols=[100, float("nan")], call_vols=[100, float("nan")],
            put_ois=[500, 500], call_ois=[1000, float("nan")],
        )

        # Act
        with patch(YF_TICKER, return_value=ticker):
            result = collector.get_spy_put_call_ratio()

        # Assert
        assert result["volume_ratio"] == pytest.approx(1.0)
        assert result["oi_ratio"] == pytest.approx(1.0)

    def test_falls_back_to_oi_ratio_when_no_call_volume(self, collector):
        # Arrange - zero call volume, valid open interest
        ticker = self.make_spy_ticker(
            put_vols=[50, 50], call_vols=[0, 0],
            put_ois=[600, 600], call_ois=[400, 400],
        )

        # Act
        with patch(YF_TICKER, return_value=ticker):
            result = collector.get_spy_put_call_ratio()

        # Assert
        assert result["volume_ratio"] is None
        assert result["ratio"] == pytest.approx(1.5)  # OI-based 1200/800

    def test_returns_none_when_no_options_dates(self, collector):
        # Arrange
        ticker = Mock()
        ticker.options = ()

        # Act
        with patch(YF_TICKER, return_value=ticker):
            result = collector.get_spy_put_call_ratio()

        # Assert
        assert result is None

    def test_returns_none_on_chain_fetch_error(self, collector):
        # Arrange
        ticker = Mock()
        ticker.options = ("2026-07-17",)
        ticker.option_chain.side_effect = RuntimeError("chain unavailable")

        # Act
        with patch(YF_TICKER, return_value=ticker):
            result = collector.get_spy_put_call_ratio()

        # Assert
        assert result is None

    def test_second_call_served_from_cache(self, collector):
        # Arrange
        ticker = self.make_spy_ticker(
            put_vols=[100], call_vols=[100], put_ois=[100], call_ois=[100],
        )
        with patch(YF_TICKER, return_value=ticker):
            first = collector.get_spy_put_call_ratio()

        # Act
        with patch(YF_TICKER) as ticker_mock:
            second = collector.get_spy_put_call_ratio()

        # Assert
        assert second == first
        ticker_mock.assert_not_called()


class TestEquityPutCallFallbackChain:
    """equity P/C source priority: CBOE PCCE > SPY > VIX proxy"""

    def test_prefers_official_cboe_pcce(self, collector):
        with patch.object(collector, "get_cboe_equity_put_call", return_value=0.72):
            result = collector.get_equity_put_call_ratio()

        assert result == 0.72
        assert collector.estimated_fields == []

    def test_falls_back_to_spy_ratio_and_marks_estimated(self, collector):
        # Arrange
        spy_data = {"ratio": 1.15, "volume_ratio": 1.15}

        # Act
        with patch.object(collector, "get_cboe_equity_put_call", return_value=None), \
             patch.object(collector, "get_spy_put_call_ratio", return_value=spy_data):
            result = collector.get_equity_put_call_ratio()

        # Assert
        assert result == pytest.approx(1.15)
        assert any(e["field"] == "equity_put_call" for e in collector.estimated_fields)

    def test_last_resort_vix_proxy(self, collector):
        # Arrange - VIX=20, VIX3M=21 → 0.5 + (20/21)*0.7
        with patch.object(collector, "get_cboe_equity_put_call", return_value=None), \
             patch.object(collector, "get_spy_put_call_ratio", return_value=None), \
             patch.object(collector, "get_vix", return_value=20.0), \
             patch.object(collector, "get_vix3m", return_value=21.0):
            # Act
            result = collector.get_equity_put_call_ratio()

        # Assert
        assert result == pytest.approx(0.5 + (20.0 / 21.0) * 0.7)
        assert any(e["field"] == "equity_put_call" for e in collector.estimated_fields)

    def test_returns_none_when_all_sources_fail(self, collector):
        with patch.object(collector, "get_cboe_equity_put_call", return_value=None), \
             patch.object(collector, "get_spy_put_call_ratio", return_value=None), \
             patch.object(collector, "get_vix", return_value=None), \
             patch.object(collector, "get_vix3m", return_value=None):
            result = collector.get_equity_put_call_ratio()

        assert result is None


class TestMarkEstimated:
    """Estimated-field tracking"""

    def test_records_field_and_reason(self, collector):
        collector._mark_estimated("vix3m", "Estimated from VIX")

        assert collector.estimated_fields == [
            {"field": "vix3m", "reason": "Estimated from VIX"}
        ]

    def test_does_not_duplicate_same_field(self, collector):
        collector._mark_estimated("vix3m", "reason one")
        collector._mark_estimated("vix3m", "reason two")

        assert len(collector.estimated_fields) == 1


class TestGetAllData:
    """Aggregated snapshot structure"""

    def test_returns_expected_keys_and_resets_estimated_fields(self, collector):
        # Arrange - stale estimation flags from a previous fetch must be reset
        collector._mark_estimated("old_field", "stale flag")

        # Act
        with patch.object(collector, "get_vix", return_value=18.0), \
             patch.object(collector, "get_vix9d", return_value=17.0), \
             patch.object(collector, "get_vix3m", return_value=19.0), \
             patch.object(collector, "get_vvix", return_value=95.0), \
             patch.object(collector, "get_skew", return_value=130.0), \
             patch.object(collector, "get_real_contango", return_value=5.56), \
             patch.object(collector, "get_put_call_ratios", return_value={"equity_pc": 0.8}), \
             patch.object(collector, "get_vvix_signal", return_value={"signal": "NEUTRAL"}):
            result = collector.get_all_data()

        # Assert
        expected_keys = {
            "vix_spot", "vix9d", "vix3m", "vvix", "vvix_signal", "skew",
            "vix_contango", "put_call_ratios", "timestamp",
            "estimated_fields", "has_estimated_data",
        }
        assert expected_keys <= set(result)
        assert result["vix_spot"] == 18.0
        assert result["estimated_fields"] == []
        assert result["has_estimated_data"] is False

    def test_put_call_ratios_defaults_to_empty_dict_when_unavailable(self, collector):
        with patch.object(collector, "get_vix", return_value=None), \
             patch.object(collector, "get_vix9d", return_value=None), \
             patch.object(collector, "get_vix3m", return_value=None), \
             patch.object(collector, "get_vvix", return_value=None), \
             patch.object(collector, "get_skew", return_value=None), \
             patch.object(collector, "get_real_contango", return_value=None), \
             patch.object(collector, "get_put_call_ratios", return_value=None), \
             patch.object(collector, "get_vvix_signal", return_value={"signal": "UNAVAILABLE"}):
            result = collector.get_all_data()

        assert result["put_call_ratios"] == {}
