"""
Test Suite for YahooCollector (data_collectors/yahoo_collector.py)

Tests cover:
- Latest-value extraction from yfinance history DataFrames
- In-memory caching, stale fallback, and persistent last-known-good (LKG) store
- Rate-limit cooldown and fetch-miss cooldown behavior
- Empty-DataFrame and exception handling
- Derived computations (breadth fraction, HY spread proxy, put/call ratio,
  credit ETF flow signals)
- with_retry decorator semantics

No network access: yfinance is replaced with stubs at the module boundary.
The persistent LKG store is redirected to tmp_path so the real data/ directory
is never touched.

Run with: python -m pytest tests/test_yahoo_collector.py -v
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import pytest
from yfinance.exceptions import YFRateLimitError

import data_collectors.yahoo_collector as yahoo_module
from data_collectors.yahoo_collector import RetryConfig, YahooCollector, with_retry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_history(closes):
    """Build a DataFrame shaped like yf.Ticker().history() output."""
    closes = [float(c) for c in closes]
    index = pd.date_range(
        end=datetime.now(), periods=len(closes), freq="B", tz="America/New_York"
    )
    close = pd.Series(closes, index=index)
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.002,
            "Low": close * 0.997,
            "Close": close,
            "Volume": [1_000_000] * len(closes),
        },
        index=index,
    )


class FakeTicker:
    """Stub for yf.Ticker supporting history(), info, options, option_chain()."""

    def __init__(self, history_df=None, history_exc=None, info=None,
                 options=(), option_chain_result=None):
        self._history_df = history_df if history_df is not None else pd.DataFrame()
        self._history_exc = history_exc
        self.info = info or {}
        self.options = tuple(options)
        self._option_chain_result = option_chain_result
        self.history_calls = 0

    def history(self, *args, **kwargs):
        self.history_calls += 1
        if self._history_exc is not None:
            raise self._history_exc
        return self._history_df

    def option_chain(self, expiry):
        return self._option_chain_result


SECTOR_TICKERS = ["XLK", "XLF", "XLV", "XLE", "XLY", "XLP", "XLI", "XLB", "XLU"]


def install_tickers(monkeypatch, tickers):
    """Replace the module's yf namespace with a stub Ticker factory.

    Returns the list of symbols requested, so tests can assert on
    whether/how often the network boundary was hit.
    """
    requested = []

    def fake_ticker(symbol):
        requested.append(symbol)
        if symbol not in tickers:
            raise KeyError(f"Unexpected ticker requested: {symbol}")
        return tickers[symbol]

    monkeypatch.setattr(yahoo_module, "yf", SimpleNamespace(Ticker=fake_ticker))
    return requested


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolate_collector_state(tmp_path, monkeypatch):
    """Reset class-level caches and redirect the persistent LKG store.

    YahooCollector keeps shared state on the class (_CACHE, _MISS_CACHE,
    _LAST_RATE_LIMIT_AT) and persists values to data/cache/yahoo_lkg.json.
    Both must be isolated so tests are order-independent and never touch
    the real data/ directory.
    """
    monkeypatch.setattr(YahooCollector, "_CACHE", {})
    monkeypatch.setattr(YahooCollector, "_MISS_CACHE", {})
    monkeypatch.setattr(YahooCollector, "_LAST_RATE_LIMIT_AT", None)
    monkeypatch.setattr(YahooCollector, "_LKG_PATH", tmp_path / "yahoo_lkg.json")
    # Disable the centralized rate limiter and retry backoff sleeps
    monkeypatch.setattr(yahoo_module, "YAHOO_LIMITER", None)
    monkeypatch.setattr(yahoo_module.time, "sleep", lambda seconds: None)


@pytest.fixture
def collector():
    """Collector with fast retry settings (no real backoff delay)."""
    return YahooCollector(
        retry_config=RetryConfig(max_retries=2, initial_backoff=0.0, max_backoff=0.0)
    )


# ---------------------------------------------------------------------------
# with_retry decorator
# ---------------------------------------------------------------------------

class TestWithRetry:
    class _Dummy:
        def __init__(self, failures, exc_type=ValueError):
            self.retry_config = RetryConfig(
                max_retries=3, initial_backoff=0.0, max_backoff=0.0
            )
            self.failures = failures
            self.exc_type = exc_type
            self.attempts = 0

        @with_retry
        def fetch(self):
            self.attempts += 1
            if self.attempts <= self.failures:
                raise self.exc_type("boom")
            return 42

    def test_returns_value_on_first_success(self):
        # Arrange
        dummy = self._Dummy(failures=0)

        # Act
        result = dummy.fetch()

        # Assert
        assert result == 42
        assert dummy.attempts == 1

    def test_retries_on_value_error_then_succeeds(self):
        # Arrange
        dummy = self._Dummy(failures=2, exc_type=ValueError)

        # Act
        result = dummy.fetch()

        # Assert
        assert result == 42
        assert dummy.attempts == 3

    def test_returns_none_when_all_retries_exhausted(self):
        # Arrange
        dummy = self._Dummy(failures=99, exc_type=ConnectionError)

        # Act
        result = dummy.fetch()

        # Assert
        assert result is None
        assert dummy.attempts == 3

    def test_does_not_retry_on_unexpected_error(self):
        # Arrange: KeyError is not in the retryable exception set
        dummy = self._Dummy(failures=99, exc_type=KeyError)

        # Act
        result = dummy.fetch()

        # Assert
        assert result is None
        assert dummy.attempts == 1


# ---------------------------------------------------------------------------
# get_vix
# ---------------------------------------------------------------------------

class TestGetVix:
    def test_returns_latest_close_from_history(self, collector, monkeypatch):
        # Arrange
        install_tickers(
            monkeypatch, {"^VIX": FakeTicker(history_df=make_history([15.2, 16.8, 17.4]))}
        )

        # Act
        value = collector.get_vix()

        # Assert
        assert value == pytest.approx(17.4)

    def test_second_call_served_from_cache_without_fetch(self, collector, monkeypatch):
        # Arrange
        ticker = FakeTicker(history_df=make_history([18.0]))
        install_tickers(monkeypatch, {"^VIX": ticker})

        # Act
        first = collector.get_vix()
        second = collector.get_vix()

        # Assert
        assert first == second == pytest.approx(18.0)
        assert ticker.history_calls == 1

    def test_empty_history_returns_none(self, collector, monkeypatch):
        # Arrange
        install_tickers(monkeypatch, {"^VIX": FakeTicker(history_df=pd.DataFrame())})

        # Act
        value = collector.get_vix()

        # Assert
        assert value is None

    def test_fetch_error_without_lkg_returns_none(self, collector, monkeypatch):
        # Arrange
        install_tickers(
            monkeypatch, {"^VIX": FakeTicker(history_exc=RuntimeError("network down"))}
        )

        # Act
        value = collector.get_vix()

        # Assert
        assert value is None

    def test_fetch_error_falls_back_to_stale_cache_value(self, collector, monkeypatch):
        # Arrange: stale in-memory value (older than fresh TTL, within stale TTL)
        YahooCollector._CACHE["vix"] = {"ts": time.time() - 600, "value": 21.0}
        install_tickers(
            monkeypatch, {"^VIX": FakeTicker(history_exc=RuntimeError("network down"))}
        )

        # Act
        value = collector.get_vix()

        # Assert
        assert value == pytest.approx(21.0)

    def test_rate_limit_error_sets_cooldown_and_returns_stale(self, collector, monkeypatch):
        # Arrange
        YahooCollector._CACHE["vix"] = {"ts": time.time() - 600, "value": 19.0}
        install_tickers(
            monkeypatch, {"^VIX": FakeTicker(history_exc=YFRateLimitError())}
        )

        # Act
        value = collector.get_vix()

        # Assert
        assert value == pytest.approx(19.0)
        assert YahooCollector._LAST_RATE_LIMIT_AT is not None

    def test_rate_limit_cooldown_skips_fetch_and_serves_stale(self, collector, monkeypatch):
        # Arrange: recent rate limit + stale value available
        YahooCollector._LAST_RATE_LIMIT_AT = time.time()
        YahooCollector._CACHE["vix"] = {"ts": time.time() - 600, "value": 22.5}
        requested = install_tickers(monkeypatch, {})

        # Act
        value = collector.get_vix()

        # Assert: no network call was attempted
        assert value == pytest.approx(22.5)
        assert requested == []

    def test_recent_fetch_miss_skips_refetch(self, collector, monkeypatch):
        # Arrange: first call misses (empty data)
        ticker = FakeTicker(history_df=pd.DataFrame())
        install_tickers(monkeypatch, {"^VIX": ticker})

        # Act
        first = collector.get_vix()
        second = collector.get_vix()

        # Assert: second call short-circuits on the miss cooldown
        assert first is None
        assert second is None
        assert ticker.history_calls == 1

    def test_value_is_persisted_to_lkg_store(self, collector, monkeypatch):
        # Arrange
        install_tickers(monkeypatch, {"^VIX": FakeTicker(history_df=make_history([17.5]))})

        # Act
        collector.get_vix()

        # Assert
        store = json.loads(YahooCollector._LKG_PATH.read_text())
        assert store["vix"]["value"] == pytest.approx(17.5)

    def test_persisted_lkg_survives_in_memory_cache_reset(self, collector, monkeypatch):
        # Arrange: persist a value, then wipe the in-memory cache
        collector._set_cached("vix", 17.5)
        YahooCollector._CACHE.clear()
        requested = install_tickers(monkeypatch, {})
        fresh = YahooCollector(
            retry_config=RetryConfig(max_retries=1, initial_backoff=0.0)
        )

        # Act
        value = fresh.get_vix()

        # Assert: served from the persistent store, no fetch attempted
        assert value == pytest.approx(17.5)
        assert requested == []


# ---------------------------------------------------------------------------
# get_vix_futures_proxy (deprecated)
# ---------------------------------------------------------------------------

def test_vix_futures_proxy_is_disabled_and_returns_none(collector, monkeypatch):
    # Arrange: no tickers installed - the method must not fetch anything
    requested = install_tickers(monkeypatch, {})

    # Act
    result = collector.get_vix_futures_proxy()

    # Assert
    assert result is None
    assert requested == []


# ---------------------------------------------------------------------------
# get_market_breadth_proxy
# ---------------------------------------------------------------------------

class TestMarketBreadthProxy:
    def test_all_sectors_advancing_returns_one(self, collector, monkeypatch):
        # Arrange
        tickers = {t: FakeTicker(history_df=make_history([100.0, 101.0]))
                   for t in SECTOR_TICKERS}
        install_tickers(monkeypatch, tickers)

        # Act
        breadth = collector.get_market_breadth_proxy()

        # Assert
        assert breadth == pytest.approx(1.0)

    def test_mixed_sectors_returns_advancing_fraction(self, collector, monkeypatch):
        # Arrange: 6 of 9 sectors advancing
        advancing = SECTOR_TICKERS[:6]
        tickers = {}
        for t in SECTOR_TICKERS:
            closes = [100.0, 101.0] if t in advancing else [100.0, 99.0]
            tickers[t] = FakeTicker(history_df=make_history(closes))
        install_tickers(monkeypatch, tickers)

        # Act
        breadth = collector.get_market_breadth_proxy()

        # Assert
        assert breadth == pytest.approx(6 / 9)

    def test_failing_tickers_are_skipped_in_calculation(self, collector, monkeypatch):
        # Arrange: 3 tickers error out, of the remaining 6 exactly 3 advance
        tickers = {}
        for i, t in enumerate(SECTOR_TICKERS):
            if i < 3:
                tickers[t] = FakeTicker(history_exc=RuntimeError("no data"))
            elif i < 6:
                tickers[t] = FakeTicker(history_df=make_history([100.0, 101.0]))
            else:
                tickers[t] = FakeTicker(history_df=make_history([100.0, 99.0]))
        install_tickers(monkeypatch, tickers)

        # Act
        breadth = collector.get_market_breadth_proxy()

        # Assert
        assert breadth == pytest.approx(0.5)

    def test_no_sector_data_returns_none_after_retries(self, collector, monkeypatch):
        # Arrange: every sector returns an empty DataFrame
        tickers = {t: FakeTicker(history_df=pd.DataFrame()) for t in SECTOR_TICKERS}
        install_tickers(monkeypatch, tickers)

        # Act
        breadth = collector.get_market_breadth_proxy()

        # Assert: the internal ValueError is retried, then swallowed to None
        assert breadth is None

    def test_result_is_cached_across_calls(self, collector, monkeypatch):
        # Arrange
        tickers = {t: FakeTicker(history_df=make_history([100.0, 101.0]))
                   for t in SECTOR_TICKERS}
        requested = install_tickers(monkeypatch, tickers)

        # Act
        collector.get_market_breadth_proxy()
        first_fetch_count = len(requested)
        collector.get_market_breadth_proxy()

        # Assert: no additional fetches on the second call
        assert len(requested) == first_fetch_count


# ---------------------------------------------------------------------------
# get_put_call_ratio_proxy
# ---------------------------------------------------------------------------

class TestPutCallRatioProxy:
    def test_computes_put_to_call_volume_ratio(self, collector, monkeypatch):
        # Arrange
        chain = SimpleNamespace(
            puts=pd.DataFrame({"volume": [500, 250]}),
            calls=pd.DataFrame({"volume": [300, 200]}),
        )
        spy = FakeTicker(options=("2026-07-17",), option_chain_result=chain)
        install_tickers(monkeypatch, {"SPY": spy})

        # Act
        ratio = collector.get_put_call_ratio_proxy()

        # Assert: 750 puts / 500 calls
        assert ratio == pytest.approx(1.5)

    def test_no_expirations_returns_none(self, collector, monkeypatch):
        # Arrange
        install_tickers(monkeypatch, {"SPY": FakeTicker(options=())})

        # Act
        ratio = collector.get_put_call_ratio_proxy()

        # Assert
        assert ratio is None

    def test_zero_call_volume_returns_none(self, collector, monkeypatch):
        # Arrange
        chain = SimpleNamespace(
            puts=pd.DataFrame({"volume": [100]}),
            calls=pd.DataFrame({"volume": [0]}),
        )
        spy = FakeTicker(options=("2026-07-17",), option_chain_result=chain)
        install_tickers(monkeypatch, {"SPY": spy})

        # Act
        ratio = collector.get_put_call_ratio_proxy()

        # Assert
        assert ratio is None


# ---------------------------------------------------------------------------
# get_treasury_10y
# ---------------------------------------------------------------------------

class TestTreasury10Y:
    def test_returns_latest_close_as_percentage(self, collector, monkeypatch):
        # Arrange
        install_tickers(
            monkeypatch, {"^TNX": FakeTicker(history_df=make_history([4.1, 4.2, 4.25]))}
        )

        # Act
        value = collector.get_treasury_10y()

        # Assert
        assert value == pytest.approx(4.25)

    def test_empty_history_returns_none(self, collector, monkeypatch):
        # Arrange
        install_tickers(monkeypatch, {"^TNX": FakeTicker(history_df=pd.DataFrame())})

        # Act
        value = collector.get_treasury_10y()

        # Assert
        assert value is None


# ---------------------------------------------------------------------------
# get_hy_spread_proxy
# ---------------------------------------------------------------------------

class TestHYSpreadProxy:
    def test_spread_computed_from_hyg_yield_minus_treasury(self, collector, monkeypatch):
        # Arrange: HYG SEC yield 8% (decimal form), 10Y treasury 4%
        collector._set_cached("treasury_10y", 4.0)
        install_tickers(monkeypatch, {"HYG": FakeTicker(info={"yield": 0.08})})

        # Act
        spread = collector.get_hy_spread_proxy()

        # Assert: 8.0 - 4.0 = 4.0 (within the 1-15 sanity range)
        assert spread == pytest.approx(4.0)

    def test_out_of_range_spread_falls_back_to_default(self, collector, monkeypatch):
        # Arrange: 25% yield gives spread of 21, outside the 1-15 sanity range
        collector._set_cached("treasury_10y", 4.0)
        install_tickers(monkeypatch, {"HYG": FakeTicker(info={"yield": 0.25})})

        # Act
        spread = collector.get_hy_spread_proxy()

        # Assert: hardcoded "normal" fallback estimate
        assert spread == pytest.approx(3.5)

    def test_missing_yield_uses_dividend_yield_fallback(self, collector, monkeypatch):
        # Arrange: no 'yield' key, dividendYield of 7% (decimal form)
        collector._set_cached("treasury_10y", 4.0)
        install_tickers(monkeypatch, {"HYG": FakeTicker(info={"dividendYield": 0.07})})

        # Act
        spread = collector.get_hy_spread_proxy()

        # Assert: 7.0 - 4.0 = 3.0
        assert spread == pytest.approx(3.0)

    def test_missing_treasury_returns_none(self, collector, monkeypatch):
        # Arrange: treasury fetch yields no data, so spread cannot be computed
        install_tickers(
            monkeypatch,
            {
                "HYG": FakeTicker(info={"yield": 0.08}),
                "^TNX": FakeTicker(history_df=pd.DataFrame()),
            },
        )

        # Act
        spread = collector.get_hy_spread_proxy()

        # Assert
        assert spread is None


# ---------------------------------------------------------------------------
# get_credit_etf_flows
# ---------------------------------------------------------------------------

class TestCreditETFFlows:
    @staticmethod
    def _install_credit_tickers(monkeypatch, hyg_closes, lqd_closes):
        install_tickers(
            monkeypatch,
            {
                "HYG": FakeTicker(history_df=make_history(hyg_closes)),
                "LQD": FakeTicker(history_df=make_history(lqd_closes)),
            },
        )

    def test_risk_on_signal_when_hyg_outperforms(self, collector, monkeypatch):
        # Arrange: HYG rising steadily, LQD drifting slightly lower
        hyg = [80.0 + 0.1 * i for i in range(22)]
        lqd = [110.0 - 0.02 * i for i in range(22)]
        self._install_credit_tickers(monkeypatch, hyg, lqd)

        # Act
        result = collector.get_credit_etf_flows()

        # Assert
        assert result["signal"] == "RISK_ON"
        assert result["hyg_price"] == pytest.approx(hyg[-1])
        assert result["lqd_price"] == pytest.approx(lqd[-1])
        assert result["hyg_lqd_ratio"] == pytest.approx(hyg[-1] / lqd[-1], abs=1e-4)
        assert result["relative_5d"] > 0.5

    def test_risk_off_signal_when_lqd_outperforms(self, collector, monkeypatch):
        # Arrange: HYG falling, LQD drifting slightly higher
        hyg = [80.0 - 0.1 * i for i in range(22)]
        lqd = [110.0 + 0.02 * i for i in range(22)]
        self._install_credit_tickers(monkeypatch, hyg, lqd)

        # Act
        result = collector.get_credit_etf_flows()

        # Assert
        assert result["signal"] == "RISK_OFF"
        assert result["relative_5d"] < -0.5

    def test_flat_lqd_yields_unknown_signal(self, collector, monkeypatch):
        # Arrange: LQD perfectly flat. Its 5d return is exactly 0.0, which the
        # collector's truthiness check (`if hyg_5d and lqd_5d`) treats as
        # missing data. Documents existing behavior as-is.
        hyg = [80.0 + 0.1 * i for i in range(22)]
        lqd = [110.0] * 22
        self._install_credit_tickers(monkeypatch, hyg, lqd)

        # Act
        result = collector.get_credit_etf_flows()

        # Assert
        assert result["signal"] == "UNKNOWN"
        assert result["relative_5d"] is None

    def test_empty_history_returns_none(self, collector, monkeypatch):
        # Arrange
        install_tickers(
            monkeypatch,
            {
                "HYG": FakeTicker(history_df=pd.DataFrame()),
                "LQD": FakeTicker(history_df=make_history([110.0] * 22)),
            },
        )

        # Act
        result = collector.get_credit_etf_flows()

        # Assert
        assert result is None


# ---------------------------------------------------------------------------
# get_all_data / get_health_check aggregation
# ---------------------------------------------------------------------------

class TestAggregation:
    def test_get_all_data_aggregates_individual_getters(self, collector, monkeypatch):
        # Arrange
        monkeypatch.setattr(collector, "get_vix", lambda: 18.0)
        monkeypatch.setattr(collector, "get_market_breadth_proxy", lambda: 0.6)
        monkeypatch.setattr(collector, "get_put_call_ratio_proxy", lambda: 0.9)
        monkeypatch.setattr(collector, "get_treasury_10y", lambda: 4.2)
        monkeypatch.setattr(collector, "get_hy_spread_proxy", lambda: 3.4)

        # Act
        data = collector.get_all_data()

        # Assert
        assert data["vix"] == 18.0
        assert data["market_breadth_proxy"] == 0.6
        assert data["put_call_proxy"] == 0.9
        assert data["treasury_10y"] == 4.2
        assert data["hy_spread_proxy"] == 3.4
        assert "timestamp" in data
        assert "vix_contango_proxy" not in data  # intentionally removed

    def test_health_check_reports_ok_and_failed_sources(self, collector, monkeypatch):
        # Arrange
        monkeypatch.setattr(collector, "get_vix", lambda: 18.0)
        monkeypatch.setattr(collector, "get_market_breadth_proxy", lambda: None)
        monkeypatch.setattr(collector, "get_put_call_ratio_proxy", lambda: 0.9)

        # Act
        health = collector.get_health_check()

        # Assert
        assert health["vix"] == "ok"
        assert health["breadth"] == "failed"
        assert health["put_call"] == "ok"
        # Deprecated futures proxy always returns None -> failed
        assert health["vix_futures"] == "failed"
        assert "timestamp" in health
