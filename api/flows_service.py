"""Positioning & flow pages — live endpoints (COT, Options Flow).

Both hit collectors live, so each is behind a process-level TTL cache. Failed /
empty fetches are never cached, so a transient outage can't pin a blank page.
"""
import time
from typing import Any, Callable, Dict

from api.overview_service import _num

_TTL_SECONDS = 300
_cache: Dict[str, tuple[float, Any]] = {}


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


def _build_options_flow() -> Dict[str, Any]:
    from data_collectors.options_flow_collector import OptionsFlowCollector

    warnings: list[str] = []
    try:
        summary = OptionsFlowCollector().get_market_options_summary() or {}
    except Exception:
        summary = {}

    etfs = []
    for key in ("spy", "qqq", "iwm"):
        d = summary.get(key)
        if not d or d.get("status") != "ok":
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

    return {"as_of": summary.get("timestamp"), "etfs": etfs, "warnings": warnings}


def build_options_flow() -> Dict[str, Any]:
    return _cached("options_flow", _build_options_flow, lambda d: bool(d.get("etfs")))
