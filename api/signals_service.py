"""Final pages — Sentiment, LEFT Strategy, CTA Flow (live, TTL-cached)."""
import time
from typing import Any, Callable, Dict

from api.deps import get_db
from api.overview_service import _num, _series

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


def _snapshot_fallback_series(hist, fields: tuple[str, ...]) -> list[dict[str, Any]]:
    """Build a history series using row-level source priority.

    Daily snapshots can carry several P/C fields. The headline sentiment metric
    prefers CBOE equity > SPY proxy > legacy best-available; the chart should
    use the same priority per date so its label is honest.
    """
    if hist is None or hist.empty:
        return []
    if "date" not in hist.columns:
        return []

    out: list[dict[str, Any]] = []
    for _, row in hist.iterrows():
        value = None
        for field in fields:
            if field in hist.columns:
                value = _num(row.get(field))
                if value is not None:
                    break
        if value is None:
            continue
        d = row["date"]
        d_str = str(d.date()) if hasattr(d, "date") else str(d)
        out.append({"date": d_str, "value": value})
    return out


# ---------------- Sentiment ----------------

def _fg_state(score: float | None) -> str:
    if score is None:
        return "neutral"
    if score < 25 or score > 78:
        return "warn"   # extreme fear or extreme greed both warrant caution
    if score < 45:
        return "warn"
    if score <= 60:
        return "neutral"
    return "good"


def _build_sentiment() -> Dict[str, Any]:
    from data_collectors.fear_greed_collector import FearGreedCollector

    warnings: list[str] = []
    try:
        fg = FearGreedCollector().get_fear_greed_score() or {}
    except Exception:
        fg = {}
    score = _num(fg.get("score"))
    if score is None:
        warnings.append("Fear & Greed data unavailable (CNN).")

    # Put/Call comes from the latest daily snapshot (cached SQLite). Prefer the
    # true CBOE equity ratio, then the SPY proxy, then the legacy best-available
    # field — and report which one so the UI doesn't mislabel it.
    snap = get_db().get_latest_snapshot() or {}
    put_call = None
    put_call_source = None
    for field, label in (
        ("cboe_equity_pc", "CBOE equity"),
        ("spy_put_call", "SPY proxy"),
        ("put_call_ratio", "Best available"),
    ):
        v = _num(snap.get(field))
        if v is not None:
            put_call = v
            put_call_source = label
            break

    # History from the daily snapshots (cached SQLite) so the page can show
    # where sentiment has been, not just where it is. The stored history has
    # known collection gaps — the frontend's GapNotice flags them honestly.
    hist = get_db().get_snapshot_history(days=365)
    fg_series = _series(hist, value_col="fear_greed_score")
    pc_series = _snapshot_fallback_series(hist, ("cboe_equity_pc", "spy_put_call", "put_call_ratio"))

    return {
        "as_of": fg.get("timestamp"),
        "fear_greed": {
            "score": score,
            "rating": fg.get("rating"),
            "state": _fg_state(score),
        },
        "put_call_ratio": put_call,
        "put_call_source": put_call_source,
        "charts": {
            "fear_greed_history": fg_series,
            "put_call_history": pc_series,
        },
        "warnings": warnings,
    }


def build_sentiment() -> Dict[str, Any]:
    return _cached("sentiment", _build_sentiment, lambda d: d["fear_greed"]["score"] is not None)


# ---------------- LEFT Strategy ----------------

def _signal_state(signal: str | None) -> str:
    s = (signal or "").upper()
    if s == "BUY":
        return "good"
    if s == "SELL":
        return "crit"
    return "neutral"


def _build_left() -> Dict[str, Any]:
    from data_collectors.fred_collector import FREDCollector
    from processors.left_strategy import LEFTStrategy

    warnings: list[str] = []
    try:
        hyg = FREDCollector().get_series("BAMLH0A0HYM2", start_date="2023-01-01")
    except Exception:
        hyg = None

    if hyg is None or hyg.empty:
        warnings.append("LEFT signal unavailable — FRED credit-spread data missing.")
        return {"as_of": None, "signal": None, "state": "neutral", "metrics": [],
                "charts": {"spread": [], "ema": []}, "warnings": warnings}

    strat = LEFTStrategy()
    sig = strat.calculate_signal(hyg) or {}
    try:
        hist = strat.get_historical_signals(hyg)
    except Exception:
        hist = None

    signal = sig.get("signal")
    metrics = [
        {"key": "spread", "label": "HYG OAS", "value": _num(sig.get("current_spread")), "unit": "%",
         "state": "neutral", "source": "FRED (BAMLH0A0HYM2)"},
        {"key": "ema", "label": "330d EMA", "value": _num(sig.get("ema_330")), "unit": "%",
         "state": "neutral", "source": "Derived"},
        {"key": "pct_from_ema", "label": "% From EMA", "value": _num(sig.get("pct_from_ema")), "unit": "%",
         "state": "neutral", "source": "Derived"},
        {"key": "strength", "label": "Signal Strength", "value": _num(sig.get("strength")), "unit": "",
         "state": _signal_state(signal), "source": "LEFT model"},
    ]

    spread_series = _series(hist, value_col="spread") if hist is not None else []
    ema_series = _series(hist, value_col="ema_330") if hist is not None else []

    return {
        "as_of": sig.get("date"),
        "signal": signal,
        "state": _signal_state(signal),
        "metrics": metrics,
        "charts": {"spread": spread_series, "ema": ema_series},
        "warnings": warnings,
    }


def build_left() -> Dict[str, Any]:
    return _cached("left", _build_left, lambda d: d.get("signal") is not None)


# ---------------- CTA Flow ----------------

def _cta_state(state: str | None) -> str:
    s = (state or "").upper()
    if s == "LONG":
        return "good"
    if s == "SHORT":
        return "crit"
    return "neutral"


def _build_cta() -> Dict[str, Any]:
    from data_collectors.cta_collector_cloud import CTACollectorCloud

    warnings: list[str] = []
    try:
        result = CTACollectorCloud().get_cta_analysis(period="2y")
    except Exception:
        result = None

    latest_state = getattr(result, "latest_state", None) or {}
    latest_exposure = getattr(result, "latest_exposure", None)
    summary = getattr(result, "summary", None) or {}
    latest_date = summary.get("latest_date")
    as_of = str(latest_date) if latest_date is not None else None

    if not latest_state:
        warnings.append("CTA positioning data unavailable (Yahoo Finance).")
        return {"as_of": None, "positions": [], "long_count": 0, "short_count": 0,
                "flat_count": 0, "warnings": warnings}

    positions = []
    longs = shorts = flats = 0
    for symbol, state in latest_state.items():
        exposure = None
        if latest_exposure is not None:
            try:
                if symbol in latest_exposure.index:
                    exposure = _num(latest_exposure[symbol])
            except Exception:
                exposure = None
        s = (state or "").upper()
        longs += s == "LONG"
        shorts += s == "SHORT"
        flats += s not in ("LONG", "SHORT")
        positions.append({
            "symbol": symbol,
            "position": state,
            "exposure": exposure,
            "state": _cta_state(state),
        })

    return {
        "as_of": as_of,
        "positions": positions,
        "long_count": longs,
        "short_count": shorts,
        "flat_count": flats,
        "warnings": warnings,
    }


def build_cta() -> Dict[str, Any]:
    return _cached("cta", _build_cta, lambda d: bool(d.get("positions")))
