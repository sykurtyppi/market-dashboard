"""Assemble the Overview page payload from the backend, Streamlit-free.

Reads the latest daily snapshot and a couple of history series from SQLite and
shapes them into a single JSON-friendly dict the frontend renders directly.
Everything is defensive: missing fields degrade to null/neutral rather than
raising, so a partial database still yields a usable page.
"""
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from api.deps import get_db
from processors.composite_risk import calculate_composite_risk_score, put_call_input


def _num(value: Any) -> Optional[float]:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _series(
    df: pd.DataFrame,
    value_col: Optional[str] = None,
    max_points: int = 180,
) -> List[Dict[str, Any]]:
    """Turn a (date, value) DataFrame into [{date, value}, ...].

    Evenly downsamples to at most max_points so a chart of a multi-year table
    stays a lightweight payload (the last point is always kept).
    """
    if df is None or df.empty:
        return []
    date_col = "date" if "date" in df.columns else df.columns[0]
    if value_col is None or value_col not in df.columns:
        candidates = [c for c in df.columns if c != date_col]
        if not candidates:
            return []
        value_col = candidates[0]

    clean = df[[date_col, value_col]].dropna()
    if clean.empty:
        return []
    if len(clean) > max_points:
        step = len(clean) / max_points
        idx = sorted({int(i * step) for i in range(max_points)} | {len(clean) - 1})
        clean = clean.iloc[idx]

    out = []
    for _, row in clean.iterrows():
        y = _num(row[value_col])
        if y is None:
            continue
        d = row[date_col]
        d_str = str(d.date()) if hasattr(d, "date") else str(d)
        out.append({"date": d_str, "value": y})
    return out


# --- state classification (returns one of good|warn|crit|neutral) ---

def _credit_state(hy_pct: Optional[float]) -> str:
    """HY OAS in percent. The old 3.0% "watch" line was the median of the last
    three years locally (2.59–4.61%, median 2.99%), so about half of all days
    read "Elevated" on a benign tape. Bands follow the distribution: under 4%
    is tight-to-normal (locally, 1.6% of days exceeded it, all April 2025);
    4–5.5% is wide (the long-run mean is roughly 5.4%); 5.5%+ is stress, as in
    2016 and 2022."""
    if hy_pct is None:
        return "neutral"
    if hy_pct < 4.0:
        return "good"
    if hy_pct < 5.5:
        return "warn"
    return "crit"


def _sentiment_state(fg: Optional[float]) -> str:
    if fg is None:
        return "neutral"
    if fg < 25 or fg > 78:
        return "warn"   # extreme fear or extreme greed both warrant caution
    if fg < 45:
        return "warn"
    if fg <= 60:
        return "neutral"
    return "good"


def _vix_state(vix: Optional[float]) -> str:
    if vix is None:
        return "neutral"
    if vix < 20:
        return "good"
    if vix < 30:
        return "warn"
    return "crit"


def _breadth_state(b: Optional[float]) -> str:
    if b is None:
        return "neutral"
    if b >= 0.6:
        return "good"
    if b >= 0.4:
        return "neutral"
    return "warn"


def _contango_state(c: Optional[float]) -> str:
    if c is None:
        return "neutral"
    return "good" if c > 0 else "crit"


def _put_call_source(snapshot: Dict) -> str:
    """Which put/call series the stored ratio actually is — one rule, shared with the score."""
    return put_call_input(snapshot)[1]


def _composite(snapshot: Dict, vrp_data: Optional[Dict], move: Optional[float]) -> Tuple[Optional[int], Dict[str, Any]]:
    """Headline score plus the breakdown the page shows under it.

    Imported directly: the old try/except around a dashboard import returned
    None on any failure, blanking the headline with no trace.
    """
    result = calculate_composite_risk_score(snapshot, vrp_data, move_index=move)
    score = result.get("score")
    breakdown = {
        "components": result.get("components", {}),
        "weights": result.get("weights", {}),
        "missing": result.get("missing", []),
        "interpretation": result.get("interpretation"),
        "put_call_source": result.get("put_call_source"),
    }
    return (int(round(score)) if score is not None else None), breakdown


def build_overview() -> Dict[str, Any]:
    db = get_db()
    snapshot = db.get_latest_snapshot(include_age=True) or {}
    vrp_data = db.get_latest_vrp() or {}
    move_row = db.get_latest_move() or {}
    composite, breakdown = _composite(snapshot, vrp_data, _num(move_row.get("move")))

    hy = _num(snapshot.get("credit_spread_hy"))
    ig = _num(snapshot.get("credit_spread_ig"))
    vix = _num(snapshot.get("vix_spot"))
    contango = _num(snapshot.get("vix_contango"))
    fg = _num(snapshot.get("fear_greed_score"))
    breadth = _num(snapshot.get("market_breadth"))
    vrp = _num(snapshot.get("vrp"))
    ten_y = _num(snapshot.get("treasury_10y"))

    metrics = [
        {"key": "vix", "label": "VIX Spot", "value": vix, "unit": "",
         "state": _vix_state(vix), "source": "Yahoo Finance (^VIX)"},
        {"key": "hy", "label": "HY Credit Spread", "value": hy, "unit": "%",
         "state": _credit_state(hy), "source": "FRED (BAMLH0A0HYM2)"},
        {"key": "ten_y", "label": "10Y Treasury", "value": ten_y, "unit": "%",
         "state": "neutral", "source": "FRED (DGS10)"},
        {"key": "fear_greed", "label": "Fear & Greed", "value": fg, "unit": "",
         "state": _sentiment_state(fg), "source": "CNN Fear & Greed (snapshot)"},
        {"key": "vrp", "label": "VRP (21d)", "value": vrp, "unit": "",
         "state": "neutral", "source": "Implied − realized vol"},
        {"key": "contango", "label": "VIX Contango", "value": contango, "unit": "%",
         "state": _contango_state(contango), "source": "VIX / VIX3M"},
    ]

    regime = {
        "composite_risk": composite,
        "breakdown": breakdown,
        "components": [
            {"key": "credit", "label": "Credit", "state": _credit_state(hy),
             "value": {"good": "No stress", "warn": "Watch", "crit": "Stress"}.get(_credit_state(hy), "Watch"),
             "note": f"HY spread {hy:.2f}%" if hy is not None else "n/a"},
            {"key": "volatility", "label": "Volatility", "state": _contango_state(contango),
             "value": "Contango" if _contango_state(contango) == "good" else "Backwardation",
             "note": f"VIX {vix:.1f}" if vix is not None else "n/a"},
            {"key": "sentiment", "label": "Sentiment", "state": _sentiment_state(fg),
             "value": "Fear" if (fg is not None and fg < 45) else ("Greed" if (fg is not None and fg > 60) else "Neutral"),
             "note": f"F&G at {fg:.0f}" if fg is not None else "n/a"},
            {"key": "breadth", "label": "Breadth", "state": _breadth_state(breadth),
             "value": "Healthy" if _breadth_state(breadth) == "good" else "Mixed",
             "note": f"{breadth * 100:.0f}% advancing" if breadth is not None else "n/a"},
        ],
    }

    # Chart series (real history)
    vrp_hist = _series(db.get_vrp_history(days=180))
    hy_hist = _series(db.get_indicator_history("credit_spread_hy", days=365))
    ig_hist = _series(db.get_indicator_history("credit_spread_ig", days=365))

    detail = [
        {"indicator": "VVIX (VIX of VIX)", "value": _num(snapshot.get("vvix")),
         "state": "neutral", "source": "CBOE"},
        {"indicator": "SKEW Index", "value": _num(snapshot.get("skew")),
         "state": "neutral", "source": "CBOE"},
        {"indicator": "VIX9D", "value": _num(snapshot.get("vix9d")),
         "state": _vix_state(_num(snapshot.get("vix9d"))), "source": "CBOE"},
        {"indicator": "Put / Call Ratio", "value": _num(snapshot.get("put_call_ratio")),
         "state": "neutral", "source": _put_call_source(snapshot)},
        {"indicator": "Market Breadth", "value": (breadth * 100) if breadth is not None else None,
         "unit": "%", "state": _breadth_state(breadth), "source": "S&P 500 A/D"},
    ]

    return {
        "as_of": snapshot.get("date"),
        "freshness": {
            "status": snapshot.get("_status", "unknown"),
            "age": snapshot.get("_age_string", "unknown"),
            "is_fresh": bool(snapshot.get("_is_fresh", False)),
        },
        "left_signal": snapshot.get("left_signal"),
        "regime": regime,
        "metrics": metrics,
        "warnings": [],
        "charts": {
            "vrp_history": vrp_hist,
            "credit_spreads": {"hy": hy_hist, "ig": ig_hist},
        },
        "detail": detail,
    }
