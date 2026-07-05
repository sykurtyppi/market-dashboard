"""Payload builders for Phase 1 pages (Volatility & VRP, Market Breadth).

Both read cached SQLite history — no live collector calls per request.
"""
from typing import Any, Dict, List

from api.deps import get_db
from api.overview_service import _num, _series


def _vrp_state(vrp: float | None) -> str:
    if vrp is None:
        return "neutral"
    return "good" if vrp > 0 else "warn"


def build_volatility() -> Dict[str, Any]:
    db = get_db()
    latest = db.get_latest_vrp() or {}
    hist = db.get_vrp_history(days=180)

    vix = _num(latest.get("vix"))
    rv = _num(latest.get("realized_vol"))
    vrp = _num(latest.get("vrp"))
    regime = latest.get("regime")
    exp_ret = _num(latest.get("expected_6m_return"))

    metrics = [
        {"key": "vix", "label": "Implied Vol (VIX)", "value": vix, "unit": "",
         "state": "neutral", "source": "CBOE / Yahoo (^VIX)"},
        {"key": "realized_vol", "label": "Realized Vol (21d)", "value": rv, "unit": "",
         "state": "neutral", "source": "SPX returns"},
        {"key": "vrp", "label": "VRP", "value": vrp, "unit": "",
         "state": _vrp_state(vrp), "source": "Implied − realized"},
        {"key": "expected_6m_return", "label": "Expected 6M Return", "value": exp_ret, "unit": "%",
         "state": "good" if (exp_ret is not None and exp_ret > 0) else "neutral",
         "source": "VRP regime model"},
    ]

    return {
        "as_of": latest.get("date"),
        "regime": regime,
        "regime_note": (
            "No VRP data available"
            if vrp is None
            else "Implied above realized — premium is rich (sell-vol favorable)"
            if vrp > 0
            else "Realized above implied — recent turbulence"
        ),
        "metrics": metrics,
        "charts": {
            "vrp_history": _series(hist, value_col="vrp"),
            "vix": _series(hist, value_col="vix"),
            "realized_vol": _series(hist, value_col="realized_vol"),
        },
    }


def _breadth_pct_state(pct: float | None) -> str:
    if pct is None:
        return "neutral"
    if pct >= 55:
        return "good"
    if pct >= 45:
        return "neutral"
    return "warn"


def _mcclellan_state(m: float | None) -> str:
    if m is None:
        return "neutral"
    if m > 0:
        return "good"
    if m > -50:
        return "neutral"
    return "warn"


def build_breadth() -> Dict[str, Any]:
    db = get_db()
    hist = db.get_breadth_history(days=120)

    if hist is None or hist.empty:
        return {"as_of": None, "metrics": [], "charts": {"ad_line": [], "mcclellan": [], "breadth_pct": []}}

    latest = hist.iloc[-1]
    # breadth_pct may be stored 0–1 or 0–100; normalize to a percentage
    raw_pct = _num(latest.get("breadth_pct"))
    pct = (raw_pct * 100) if (raw_pct is not None and raw_pct <= 1.0) else raw_pct
    advancing = _num(latest.get("advancing"))
    declining = _num(latest.get("declining"))
    mcclellan = _num(latest.get("mcclellan"))
    ad_line = _num(latest.get("ad_line"))

    metrics = [
        {"key": "breadth_pct", "label": "Breadth %", "value": pct, "unit": "%",
         "state": _breadth_pct_state(pct), "source": "S&P 500 advancers"},
        {"key": "advancing", "label": "Advancing", "value": advancing, "unit": "",
         "state": "neutral", "source": "S&P 500"},
        {"key": "declining", "label": "Declining", "value": declining, "unit": "",
         "state": "neutral", "source": "S&P 500"},
        {"key": "mcclellan", "label": "McClellan Osc.", "value": mcclellan, "unit": "",
         "state": _mcclellan_state(mcclellan), "source": "EMA(19)−EMA(39) of A/D"},
    ]

    # Normalize the breadth_pct series the same way for the chart
    pct_series = _series(hist, value_col="breadth_pct")
    if pct_series and all(p["value"] <= 1.0 for p in pct_series):
        pct_series = [{"date": p["date"], "value": p["value"] * 100} for p in pct_series]

    return {
        "as_of": str(latest.get("date")),
        "metrics": metrics,
        "charts": {
            "ad_line": _series(hist, value_col="ad_line"),
            "mcclellan": _series(hist, value_col="mcclellan"),
            "breadth_pct": pct_series,
        },
    }
