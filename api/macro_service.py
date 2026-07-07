"""Macro pages — live endpoints (Fed Watch, Cross-Asset).

Both hit collectors live (Fed funds futures / cross-asset prices), so each is
behind a process-level TTL cache. Failed/empty fetches are never cached, so a
transient outage can't pin a blank page for the whole TTL window.
"""
import time
from datetime import datetime
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
    if ok(data):  # only cache a successful fetch
        _cache[key] = (now, data)
    return data


def _change_state(change: float | None) -> str:
    if change is None:
        return "neutral"
    if change > 0.25:
        return "good"
    if change < -0.25:
        return "crit"
    return "neutral"


# ---------------- Fed Watch ----------------

def _bias_state(bias: str | None) -> str:
    if not bias:
        return "neutral"
    b = bias.lower()
    if "dovish" in b or "cut" in b:
        return "good"
    if "hawkish" in b or "hike" in b:
        return "warn"
    return "neutral"


def _build_fed_watch() -> Dict[str, Any]:
    from data_collectors.fed_watch_collector import FedWatchCollector

    warnings: list[str] = []
    try:
        fw = FedWatchCollector().get_fed_watch_summary() or {}
    except Exception:
        fw = {}

    # Provenance: the collector falls back to neutral/derived numbers when the
    # fed funds futures feed is down. Surface that so the probabilities aren't
    # mistaken for real market-implied data.
    degraded = False
    if not fw.get("current_rate"):
        warnings.append("Fed funds data unavailable.")
        degraded = True
    elif fw.get("rate_source") == "fallback" or fw.get("implied_rate") is None:
        warnings.append(
            "Fed Watch is using fallback rate/probability estimates; live futures data unavailable."
        )
        degraded = True

    probs_raw = fw.get("probabilities") or {}
    probabilities = [
        {"outcome": outcome, "pct": _num(pct)}
        for outcome, pct in probs_raw.items()
    ]

    nm = fw.get("next_meeting") or {}
    metrics = [
        {"key": "effr", "label": "Effective Rate (EFFR)", "value": _num(fw.get("effr")),
         "unit": "%", "state": "neutral", "source": "FRED (EFFR)"},
        {"key": "rate_mid", "label": "Target Midpoint", "value": _num(fw.get("current_rate_mid")),
         "unit": "%", "state": "neutral", "source": fw.get("rate_source") or "FRED"},
        {"key": "implied", "label": "Implied Rate", "value": _num(fw.get("implied_rate")),
         "unit": "%", "state": "neutral", "source": "Fed funds futures"},
        {"key": "terminal", "label": "Terminal Rate", "value": _num(fw.get("terminal_rate")),
         "unit": "%", "state": "neutral", "source": "Rate path"},
    ]

    return {
        "as_of": fw.get("rate_as_of") or fw.get("timestamp"),
        "current_rate": fw.get("current_rate"),
        "degraded": degraded,
        "next_meeting": {"date": nm.get("date_str"), "days_until": nm.get("days_until")},
        "most_likely": {"outcome": fw.get("most_likely"), "pct": _num(fw.get("most_likely_prob"))},
        "market_bias": fw.get("market_bias"),
        "bias_state": _bias_state(fw.get("market_bias")),
        "probabilities": probabilities,
        "metrics": metrics,
        "warnings": warnings,
    }


def build_fed_watch() -> Dict[str, Any]:
    # Only cache a genuine (non-degraded) fetch, so fallback data is retried
    # rather than pinned for the whole TTL window.
    return _cached(
        "fed_watch",
        _build_fed_watch,
        lambda d: bool(d.get("current_rate")) and not d.get("degraded"),
    )


# ---------------- Cross-Asset ----------------

def _regime_state(color: str | None, regime: str | None) -> str:
    r = (regime or "").lower()
    if "risk-on" in r or "risk on" in r:
        return "good"
    if "risk-off" in r or "risk off" in r:
        return "crit"
    return "neutral"


def _build_cross_asset() -> Dict[str, Any]:
    from data_collectors.cross_asset_collector import CrossAssetCollector

    warnings: list[str] = []
    ca = CrossAssetCollector()

    try:
        reg = ca.get_regime_signal() or {}
    except Exception:
        reg = {}
    try:
        perf = ca.get_asset_performance_summary() or {}
    except Exception:
        perf = {}
    try:
        corrs = ca.get_key_correlations() or []
    except Exception:
        corrs = []

    if not perf:
        warnings.append("Cross-asset price data unavailable (Yahoo Finance).")

    # The collector's change_pct spans its whole fetch window (1mo default) —
    # it was previously surfaced under a "1d" column header. Emit both horizons
    # explicitly so the frontend can't mislabel them; state colors on the true
    # daily move.
    assets = [
        {
            "ticker": ticker,
            "name": d.get("name"),
            "change_1d_pct": _num(d.get("change_1d_pct")),
            "change_1m_pct": _num(d.get("change_pct")),
            "state": _change_state(_num(d.get("change_1d_pct"))),
        }
        for ticker, d in perf.items()
    ]

    correlations = [
        {
            "pair": c.get("pair"),
            "correlation": _num(c.get("correlation")),
            "strength": c.get("strength"),
            "interpretation": c.get("interpretation"),
        }
        for c in corrs
    ]

    return {
        "as_of": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "regime": {
            "signal": reg.get("regime"),
            "state": _regime_state(reg.get("color"), reg.get("regime")),
            "description": reg.get("description"),
            "confidence": _num(reg.get("confidence")),
        },
        "assets": assets,
        "correlations": correlations,
        "warnings": warnings,
    }


def build_cross_asset() -> Dict[str, Any]:
    return _cached("cross_asset", _build_cross_asset, lambda d: bool(d.get("assets")))


# ---------------- Economic Calendar ----------------

def _field(e: Any, key: str) -> Any:
    return e.get(key) if isinstance(e, dict) else getattr(e, key, None)


def _build_economic_calendar() -> Dict[str, Any]:
    from data_collectors.economic_calendar_collector import EconomicCalendarCollector

    warnings: list[str] = []
    try:
        raw = EconomicCalendarCollector().get_upcoming_events() or []
    except Exception:
        raw = []
    if not raw:
        warnings.append("Economic calendar unavailable.")

    events = []
    for e in raw:
        imp = _field(e, "importance")
        imp_str = getattr(imp, "value", None) or (str(imp) if imp is not None else None)
        d = _field(e, "date")
        date_str = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else (str(d) if d else None)
        days_until = None
        if hasattr(d, "date"):
            try:
                days_until = (d.date() - datetime.now().date()).days
            except Exception:
                days_until = None
        events.append({
            "name": _field(e, "name"),
            "date": date_str,
            "days_until": days_until,
            "importance": imp_str,
            "category": _field(e, "category"),
            "actual": _num(_field(e, "actual")),
            "forecast": _num(_field(e, "forecast")),
            "previous": _num(_field(e, "previous")),
            "yoy_change": _num(_field(e, "yoy_change")),
            "unit": _field(e, "unit"),
        })

    return {"as_of": datetime.now().strftime("%Y-%m-%d %H:%M"), "events": events, "warnings": warnings}


def build_economic_calendar() -> Dict[str, Any]:
    return _cached("economic_calendar", _build_economic_calendar, lambda d: bool(d.get("events")))
