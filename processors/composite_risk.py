"""Composite risk score for the Overview page.

Kept in processors/ (not the API layer) so the arithmetic can be tested on
its own. The API used to import this through a catch-all that blanked the
score on any failure, and the credit term was never contributing (see
normalize()).

Score 0–100: higher = more caution warranted. Each input is normalized to
0–100 against a fixed window, then weighted; weights are renormalized over
the inputs actually present, and the absent ones are reported so a reader
can see what the score is NOT based on.

Design note, not a bug: sentiment and put/call are CONTRARIAN here — extreme
fear and heavy put buying LOWER the score, on the view that they mark
washed-out positioning. That is a return-oriented reading of a "risk" score,
and the Overview's Sentiment cell separately labels fear as "Elevated". Both
are surfaced; neither is hidden.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

# Weight per input. The normalization windows live in normalize().
COMPONENTS: Dict[str, Dict[str, Any]] = {
    "vix":       {"weight": 0.15, "label": "VIX"},
    "credit":    {"weight": 0.20, "label": "HY credit spread"},
    "sentiment": {"weight": 0.15, "label": "Fear & Greed"},
    "vrp":       {"weight": 0.10, "label": "VRP"},
    "breadth":   {"weight": 0.15, "label": "Breadth"},
    "move":      {"weight": 0.10, "label": "MOVE"},
    "skew":      {"weight": 0.10, "label": "SKEW"},
    "put_call":  {"weight": 0.05, "label": "Put/Call"},
}

# SKEW window. Yahoo ^SKEW daily history for the current post-2020 regime
# (2020-01-02 through 2026-10-27, 1,674 readings) has p5=118.74,
# median=140.20, and p95=159.96. Rounded p5/p95 anchors prevent a normal
# reading from appearing near maximum risk while still saturating true tails.
SKEW_WINDOW = (120.0, 160.0)

# CBOE equity-only P/C window from the model's documented thresholds:
# 0.6 is extreme call-heavy complacency and 1.2 is extreme put-heavy fear.
# Proxy or source-unknown ratios are not scored against these CBOE bounds.
PUT_CALL_WINDOW = (0.6, 1.2)

BANDS = [
    (75, "HIGH RISK", "#FF4444", "Multiple stress indicators elevated. Defensive positioning warranted."),
    (60, "ELEVATED", "#FFA500", "Some caution warranted. Monitor for deterioration."),
    (40, "NEUTRAL", "#FFD700", "Mixed signals. Normal market conditions."),
    (25, "LOW RISK", "#90EE90", "Favorable conditions. Risk-on appropriate."),
    (0, "VERY BULLISH", "#00CC00", "Extreme fear/pessimism. Strong contrarian buy signals."),
]


def _clamp(value: float) -> float:
    return min(100.0, max(0.0, value))


def _num(value: Any) -> Optional[float]:
    try:
        if value is None or isinstance(value, bool):
            return None
        out = float(value)
        return None if (out != out or out in (float("inf"), float("-inf"))) else out
    except (TypeError, ValueError, OverflowError):
        return None


def normalize(name: str, value: float) -> float:
    """0–100 risk contribution of one input against its documented window."""
    if name == "vix":
        # VIX 10 -> 0, 60 -> 100
        return _clamp((value - 10.0) / 50.0 * 100.0)
    if name == "credit":
        # HY OAS in PERCENT, as FRED and the snapshot store it: 2% -> 0, 7% -> 100.
        # The original formula subtracted 200 and divided by 500 — basis points —
        # from a percent input, so credit clamped to 0 at every level and the
        # score was understated by up to ~22 points exactly in a credit crisis.
        return _clamp((value - 2.0) / 5.0 * 100.0)
    if name == "sentiment":
        # Fear & Greed 0–100 used directly: greed = risk, fear = contrarian relief.
        return _clamp(value)
    if name == "vrp":
        # VRP +10 -> 0, -10 -> 100 (negative premium = implied below realized = stress)
        return _clamp((10.0 - value) / 20.0 * 100.0)
    if name == "breadth":
        # share advancing 0–1, inverted: 100% advancing -> 0, 0% -> 100
        return _clamp((1.0 - value) * 100.0)
    if name == "move":
        # MOVE 70 -> 0, 170 -> 100
        return _clamp((value - 70.0) / 100.0 * 100.0)
    if name == "skew":
        low, high = SKEW_WINDOW
        return _clamp((value - low) / (high - low) * 100.0)
    if name == "put_call":
        # Contrarian: heavy put buying lowers risk; documented CBOE bounds map
        # 1.2 -> 0 and 0.6 -> 100.
        low, high = PUT_CALL_WINDOW
        return _clamp((high - value) / (high - low) * 100.0)
    raise KeyError(name)


def put_call_observation(snapshot: Dict[str, Any]) -> tuple[Optional[float], str, bool]:
    """Select one put/call observation with its provenance and calibration.

    The selected raw value remains available for display. Only CBOE's
    equity-only series is calibrated for the composite score.
    """
    cboe = _num(snapshot.get("cboe_equity_pc"))
    if cboe is not None:
        return cboe, "CBOE equity P/C (PCCE)", True
    spy = _num(snapshot.get("spy_put_call"))
    if spy is not None:
        return spy, "SPY volume proxy (uncalibrated)", False
    legacy = _num(snapshot.get("put_call_ratio"))
    if legacy is not None:
        return legacy, "Unverified put/call source", False
    return None, "Unavailable", False


def put_call_input(snapshot: Dict[str, Any]) -> tuple[Optional[float], str]:
    """Return only a calibrated scoring input plus selected provenance."""
    value, source, calibrated = put_call_observation(snapshot)
    return (value if calibrated else None), source


def calculate_composite_risk_score(
    snapshot: Dict[str, Any],
    vrp_data: Optional[Dict[str, Any]] = None,
    move_index: Optional[float] = None,
) -> Dict[str, Any]:
    """Score the latest snapshot. Returns score None when nothing is usable.

    Args:
        snapshot: latest daily_snapshots row (vix_spot, credit_spread_hy in
            percent, fear_greed_score, market_breadth as a 0–1 share, skew,
            and the put/call columns).
        vrp_data: latest vrp_data row (vrp).
        move_index: latest MOVE level. It lives in its own table, not the
            snapshot, so callers must pass it — the old code read a snapshot
            key that never existed and the documented 10% weight never applied.
    """
    empty = {"score": None, "components": {}, "weights": {}, "missing": list(COMPONENTS), "put_call_source": None}
    if not snapshot:
        return {**empty, "status": "No data"}

    pc_value, pc_source = put_call_input(snapshot)
    raw = {
        "vix": _num(snapshot.get("vix_spot")),
        "credit": _num(snapshot.get("credit_spread_hy")),
        "sentiment": _num(snapshot.get("fear_greed_score")),
        "vrp": _num((vrp_data or {}).get("vrp")),
        "breadth": _num(snapshot.get("market_breadth")),
        "move": _num(move_index),
        "skew": _num(snapshot.get("skew")),
        "put_call": pc_value,
    }

    scored = {name: normalize(name, value) for name, value in raw.items() if value is not None}
    weights = {name: COMPONENTS[name]["weight"] for name in scored}
    missing = [name for name in COMPONENTS if name not in scored]

    if not scored:
        source = pc_source if pc_source != "Unavailable" else None
        return {**empty, "missing": missing, "put_call_source": source, "status": "Insufficient data"}

    total_weight = sum(weights.values())
    # Score from unrounded contributions; round only what is displayed.
    score = round(sum(scored[k] * weights[k] for k in scored) / total_weight, 1)
    _, interpretation, color, description = next(b for b in BANDS if score >= b[0])

    return {
        "score": score,
        "interpretation": interpretation,
        "color": color,
        "description": description,
        "components": {k: round(v, 1) for k, v in scored.items()},
        "weights": weights,
        "missing": missing,
        "put_call_source": pc_source,
        "component_count": len(scored),
        "timestamp": datetime.now().isoformat(),
    }
