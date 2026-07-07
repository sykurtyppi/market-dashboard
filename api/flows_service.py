"""Positioning & flow pages — live endpoints (COT, Options Flow).

Both hit collectors live, so each is behind a process-level TTL cache. Failed /
empty fetches are never cached, so a transient outage can't pin a blank page.
"""
import time
from typing import Any, Callable, Dict

from api.overview_service import _num

_TTL_SECONDS = 300
_cache: Dict[str, tuple[float, Any]] = {}
_EXPECTED_ETFS = ("spy", "qqq", "iwm")


def _cached(key: str, fn: Callable[[], Any], ok: Callable[[Any], bool]) -> Any:
    now = time.monotonic()
    hit = _cache.get(key)
    if hit and (now - hit[0]) < _TTL_SECONDS:
        return hit[1]
    data = fn()
    if ok(data):
        _cache[key] = (now, data)
    return data


# ---------------- COT Positioning ----------------

def _build_cot() -> Dict[str, Any]:
    from data_collectors.cot_collector import COTCollector

    warnings: list[str] = []
    try:
        data = COTCollector().get_positioning_summary() or {}
    except Exception:
        data = {}

    raw = data.get("positions") or {}
    if not raw:
        warnings.append("COT positioning data unavailable (CFTC).")

    positions = [
        {
            "symbol": symbol,
            "name": p.get("name"),
            "category": p.get("category"),
            "date": p.get("date"),
            "spec_net": _num(p.get("spec_net")),
            "spec_net_change": _num(p.get("spec_net_change")),
            "comm_net": _num(p.get("comm_net")),
            "open_interest": _num(p.get("open_interest")),
        }
        for symbol, p in raw.items()
    ]

    return {"as_of": data.get("timestamp"), "positions": positions, "warnings": warnings}


def build_cot() -> Dict[str, Any]:
    return _cached("cot", _build_cot, lambda d: bool(d.get("positions")))


# ---------------- Options Flow ----------------

def _sentiment_state(sentiment: str | None) -> str:
    s = (sentiment or "").upper()
    if s == "BULLISH":
        return "good"
    if s == "BEARISH":
        return "crit"
    return "neutral"


def _signal_state(signal: str | None, sentiment: str | None = None) -> str:
    """Map a machine-readable collector `signal` to a semantic state.

    Institutional collectors expose a clean `signal` enum (BULLISH / BEARISH /
    NEUTRAL / NORMAL / ELEVATED / …) alongside descriptive prose in `sentiment`.
    Prefer the signal; fall back to sentiment only if no signal is present.
    """
    s = (signal or "").upper()
    if s in ("BULLISH", "STRONG"):
        return "good"
    if s in ("BEARISH", "WEAK", "STRESS"):
        return "crit"
    if s in ("ELEVATED", "CAUTION", "CAUTIOUS", "TIGHTENING"):
        return "warn"
    if s in ("NEUTRAL", "NORMAL", "LOW", "MIXED"):
        return "neutral"
    return _sentiment_state(sentiment)


def _build_options_flow() -> Dict[str, Any]:
    from data_collectors.options_flow_collector import OptionsFlowCollector

    warnings: list[str] = []
    try:
        summary = OptionsFlowCollector().get_market_options_summary() or {}
    except Exception:
        summary = {}

    etfs = []
    missing: list[str] = []
    for key in _EXPECTED_ETFS:
        d = summary.get(key)
        if not d or d.get("status") != "ok":
            missing.append(key.upper())
            continue
        etfs.append({
            "ticker": d.get("ticker", key.upper()),
            "price": _num(d.get("current_price")),
            "expiry": d.get("expiry"),
            "dte": d.get("dte"),
            "put_call_ratio": _num(d.get("put_call_ratio")),
            "call_volume": _num(d.get("total_call_volume")),
            "put_volume": _num(d.get("total_put_volume")),
            "sentiment": d.get("sentiment"),
            "state": _sentiment_state(d.get("sentiment")),
        })

    if not etfs:
        warnings.append("Options flow data unavailable (Yahoo Finance).")
    elif missing:
        # Partial failure — name the missing ETFs rather than silently show a
        # normal-looking page with the important ones (SPY/QQQ) absent.
        warnings.append(f"Options flow partially unavailable: {', '.join(missing)}.")

    return {"as_of": summary.get("timestamp"), "etfs": etfs, "warnings": warnings}


def build_options_flow() -> Dict[str, Any]:
    # Only cache a complete fetch (all expected ETFs present); partial/empty
    # results are retried on the next request rather than pinned for the TTL.
    return _cached(
        "options_flow",
        _build_options_flow,
        lambda d: len(d.get("etfs", [])) == len(_EXPECTED_ETFS),
    )


# ---------------- Institutional Flow (dark pool + insider + auctions) ----------------

def _health_state(v: str | None) -> str:
    s = (v or "").upper()
    if "STRONG" in s or "HEALTHY" in s:
        return "good"
    if "WEAK" in s or "STRESS" in s:
        return "crit"
    return "neutral"


def _build_institutional() -> Dict[str, Any]:
    from data_collectors.dark_pool_collector import DarkPoolCollector
    from data_collectors.insider_trading_collector import InsiderTradingCollector
    from data_collectors.treasury_auction_collector import TreasuryAuctionCollector

    warnings: list[str] = []
    missing: list[str] = []

    def _safe(fn):
        try:
            return fn() or {}
        except Exception:
            return {}

    dp = _safe(DarkPoolCollector().get_dark_pool_summary)
    ins = _safe(InsiderTradingCollector().get_insider_summary)
    au = _safe(TreasuryAuctionCollector().get_auction_summary)

    dark_pool = None
    if dp:
        dark_pool = {
            "avg_pct": _num(dp.get("avg_dark_pool_pct")),
            "etf_pct": _num(dp.get("etf_avg_pct")),
            "stock_pct": _num(dp.get("stock_avg_pct")),
            "sentiment": dp.get("sentiment"),
            "state": _signal_state(dp.get("signal"), dp.get("sentiment")),
            "interpretation": dp.get("interpretation"),
            "week_ending": dp.get("week_ending"),
        }
    else:
        missing.append("Dark Pool")

    insider = None
    if ins:
        buys = _num(ins.get("buy_count"))
        sells = _num(ins.get("sell_count"))
        ratio = _num(ins.get("buy_sell_ratio"))
        # With zero classified buys AND sells there is no ratio — the collector's
        # 0/0 placeholder (1.00) would render as a real, neutral-looking reading.
        if not buys and not sells:
            ratio = None
        insider = {
            "total_transactions": _num(ins.get("total_transactions")),
            "buy_count": buys,
            "sell_count": sells,
            "buy_sell_ratio": ratio,
            "sentiment": ins.get("sentiment"),
            "state": _signal_state(ins.get("signal"), ins.get("sentiment")),
            "period_days": _num(ins.get("period_days")),
        }
    else:
        missing.append("Insider")

    auctions = None
    if au:
        auctions = {
            "avg_bid_to_cover": _num(au.get("avg_bid_to_cover")),
            "avg_indirect_pct": _num(au.get("avg_indirect_pct")),
            "avg_direct_pct": _num(au.get("avg_direct_pct")),
            "auction_count": _num(au.get("auction_count")),
            "weak_auctions": _num(au.get("weak_auctions")),
            "strong_auctions": _num(au.get("strong_auctions")),
            "health": au.get("health"),
            "state": _health_state(au.get("health")),
        }
    else:
        missing.append("Treasury Auctions")

    if not (dark_pool or insider or auctions):
        warnings.append("Institutional flow data unavailable.")
    elif missing:
        warnings.append(f"Institutional data partially unavailable: {', '.join(missing)}.")

    return {
        "as_of": dp.get("last_updated") or ins.get("last_updated"),
        "dark_pool": dark_pool,
        "insider": insider,
        "auctions": auctions,
        "warnings": warnings,
    }


def build_institutional() -> Dict[str, Any]:
    # Cache only a complete fetch (all three sections present); partial results
    # retry on the next request.
    return _cached(
        "institutional",
        _build_institutional,
        lambda d: all(d.get(k) for k in ("dark_pool", "insider", "auctions")),
    )
