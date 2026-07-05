"""Sectors & VIX payload — the one live endpoint (sector ETFs + VIX tenors).

Sector performance and VIX term structure aren't stored in SQLite, so this hits
the collectors live. A small process-level TTL cache keeps repeated page loads
from hammering Yahoo/CBOE.
"""
import time
from datetime import datetime
from typing import Any, Dict

from api.overview_service import _num

_TTL_SECONDS = 300
_cache: Dict[str, tuple[float, Any]] = {}


def _rotation_state(signal: str | None) -> str:
    if not signal:
        return "neutral"
    s = signal.lower()
    if "risk-off" in s:
        return "crit" if "strong" in s else "warn"
    if "risk-on" in s:
        return "good"
    return "neutral"


def _sector_state(change: float | None) -> str:
    if change is None:
        return "neutral"
    if change > 0.5:
        return "good"
    if change < -0.5:
        return "crit"
    return "neutral"


def _build_live() -> Dict[str, Any]:
    from data_collectors.cboe_collector import CBOECollector
    from data_collectors.sector_collector import SectorCollector

    warnings: list[str] = []

    sc = SectorCollector()
    try:
        perf = sc.get_sector_performance(period="1d") or {}
    except Exception:
        perf = {}
    if not perf:
        warnings.append("Sector ETF data unavailable (Yahoo Finance) — showing VIX only.")
    sectors = [
        {
            "ticker": ticker,
            "name": d.get("name"),
            "category": d.get("category"),
            "change_pct": _num(d.get("change_pct")),
            "price": _num(d.get("price")),
            "state": _sector_state(_num(d.get("change_pct"))),
        }
        for ticker, d in perf.items()
    ]
    sectors.sort(key=lambda s: (s["change_pct"] is not None, s["change_pct"] or 0), reverse=True)

    try:
        rot = sc.get_rotation_signal() or {}
    except Exception:
        rot = {}
    rotation = {
        "signal": rot.get("signal"),
        "state": _rotation_state(rot.get("signal")),
        "interpretation": rot.get("interpretation"),
        "leading_sectors": list(rot.get("leading_sectors", []))[:3],
    }

    cboe = CBOECollector()
    tenors = [("9D", 9, cboe.get_vix9d), ("VIX (30D)", 30, cboe.get_vix), ("3M", 90, cboe.get_vix3m)]
    vix_term = []
    for label, days, fn in tenors:
        try:
            v = fn()
        except Exception:
            v = None
        if v is not None:
            vix_term.append({"maturity": label, "days": days, "value": float(v)})

    structure = None
    if len(vix_term) >= 2:
        near, far = vix_term[0]["value"], vix_term[-1]["value"]
        structure = "Contango" if far > near else "Backwardation"
    if not vix_term:
        warnings.append("VIX term-structure data unavailable (CBOE).")

    return {
        "as_of": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "sectors": sectors,
        "rotation": rotation,
        "vix_term": vix_term,
        "vix_structure": structure,
        "warnings": warnings,
    }


def build_sectors() -> Dict[str, Any]:
    """Return sectors data, TTL-cached — but never cache a failed (empty) fetch,
    so one Yahoo hiccup can't pin an empty page for the whole TTL window."""
    now = time.monotonic()
    hit = _cache.get("sectors")
    if hit and (now - hit[0]) < _TTL_SECONDS:
        return hit[1]
    data = _build_live()
    if data.get("sectors"):  # only cache a successful fetch
        _cache["sectors"] = (now, data)
    return data
