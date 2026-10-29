"""Sectors & VIX payload — the one live endpoint (sector ETFs + VIX tenors).

Sector performance and VIX term structure aren't stored in SQLite, so this hits
the collectors live. A small process-level TTL cache keeps repeated page loads
from hammering Yahoo/CBOE.
"""
import time
from datetime import datetime
from typing import Any, Dict

from api.deps import get_db

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
    tenors = [
        ("9D", 9, "VIX9D", cboe.get_vix9d, False),
        ("1M", 30, "VIX", cboe.get_vix, False),
        ("3M", 90, "VIX3M", cboe.get_vix3m, True),
        ("6M", 180, "VIX6M", cboe.get_vix6m, False),
        ("1Y", 365, "VIX1Y", cboe.get_vix1y, False),
    ]
    raw_curve = []
    raw_values: dict[str, float | None] = {}
    for label, days, symbol, fn, may_estimate in tenors:
        try:
            v = fn()
        except Exception:
            v = None
        raw_value = float(v) if v is not None else None
        raw_values[symbol] = raw_value
        if raw_value is not None:
            estimated = may_estimate and any(
                isinstance(item, dict) and item.get("field") == "vix3m"
                for item in getattr(cboe, "estimated_fields", [])
            )
            raw_curve.append((label, days, symbol, round(raw_value, 2), estimated))
        else:
            warnings.append(f"{symbol} unavailable; adjacent curve comparison suppressed.")

    vix_term = []
    curve_by_symbol = {point[2]: point for point in raw_curve}
    previous_value = None
    for _label, _days, symbol, _fn, _may_estimate in tenors:
        point = curve_by_symbol.get(symbol)
        if point is None:
            previous_value = None
            continue
        label, days, _, value, estimated = point
        change = round(value - previous_value, 2) if previous_value is not None else None
        change_pct = (
            round((value / previous_value - 1) * 100, 2)
            if previous_value not in (None, 0)
            else None
        )
        vix_term.append(
            {
                "maturity": label,
                "days": days,
                "value": value,
                "symbol": symbol,
                "change": change,
                "change_pct": change_pct,
                "estimated": estimated,
            }
        )
        previous_value = value

    structure = None
    near, far = raw_values.get("VIX9D"), raw_values.get("VIX1Y")
    if near is not None and far is not None:
        slope = far - near
        if abs(slope) <= 0.01:
            structure = "Flat"
        else:
            structure = "Contango" if slope > 0 else "Backwardation"
    stale_fields = getattr(cboe, "stale_fields", [])
    if isinstance(stale_fields, list) and stale_fields:
        warnings.append(
            "Cached last-known-good VIX closes used for: " + ", ".join(stale_fields) + "."
        )
    if not vix_term:
        warnings.append("VIX term-structure data unavailable (CBOE).")

    return {
        "as_of": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "sectors": sectors,
        "rotation": rotation,
        "vix_term": vix_term,
        "vix_structure": structure,
        "vix_source": "Yahoo Finance · Cboe volatility index closes",
        "warnings": warnings,
    }


def build_sectors() -> Dict[str, Any]:
    """Serve only the latest durable snapshot; never collect on an HTTP read."""
    from api.snapshot_service import snapshot_response

    return snapshot_response(
        get_db(),
        "sectors",
        {
            "as_of": None,
            "sectors": [],
            "rotation": {
                "signal": "Unavailable",
                "state": "neutral",
                "interpretation": "Waiting for the first scheduled snapshot.",
                "leading_sectors": [],
            },
            "vix_term": [],
            "vix_structure": None,
            "vix_source": "Yahoo Finance · Cboe volatility index closes",
            "warnings": ["Sectors snapshot is not available yet."],
        },
        stale_after_seconds=4 * 60 * 60,
    )
