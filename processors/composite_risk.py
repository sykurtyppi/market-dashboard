"""Composite risk score for the Overview page.

Moved out of the legacy Streamlit helpers so the API's headline number does
not depend on that package, and so the arithmetic can be tested. The API
used to import this through a catch-all that blanked the score on any
failure, and the credit term was never contributing (see below).

Score 0–100: higher = more caution warranted. Each component is normalized
to 0–100 against a fixed range, then weighted; weights are renormalized over
the components actually present, and the absent ones are reported so a
reader can see what the score is NOT based on.

Design note, not a bug: sentiment and put/call are CONTRARIAN here — extreme
fear and heavy put buying LOWER the score, on the view that they mark
washed-out positioning. That is a return-oriented reading of a "risk" score,
and the Overview's Sentiment cell separately labels fear as "Elevated". Both
are surfaced; neither is hidden.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

# (weight, documented input range) for each component. Ranges are the
# normalization window: at or below the low end scores 0, at or above the
# high end scores 100.
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
        if value is None:
            return None
        out = float(value)
        return None if out != out else out  # NaN
    except (TypeError, ValueError):
        return None


def normalize(name: str, value: float) -> float:
    """0–100 risk contribution of one input, by the ranges documented here."""
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
        # SKEW 110 -> 0, 150 -> 100
        return _clamp((value - 110.0) / 40.0 * 100.0)
    if name == "put_call":
        # P/C 1.5 -> 0, 0.5 -> 100 (contrarian: heavy put buying lowers risk)
        return _clamp((1.5 - value) / 1.0 * 100.0)
    raise KeyError(name)


def calculate_composite_risk_score(
    snapshot: Dict[str, Any],
    vrp_data: Optional[Dict[str, Any]] = None,
    move_index: Optional[float] = None,
) -> Dict[str, Any]:
    """Score the latest snapshot. Returns score None when nothing is usable.

    Args:
        snapshot: latest daily_snapshots row (vix_spot, credit_spread_hy in
            percent, fear_greed_score, market_breadth as a 0–1 share, skew,
            put_call_ratio).
        vrp_data: latest vrp_data row (vrp).
        move_index: latest MOVE level. It lives in its own table, not the
            snapshot — the old code read snapshot['move_index'], which never
            existed, so the documented 10% MOVE weight never applied.
    """
    if not snapshot:
        return {"score": None, "status": "No data", "components": {}, "weights": {}, "missing": list(COMPONENTS)}

    raw = {
        "vix": _num(snapshot.get("vix_spot")),
        "credit": _num(snapshot.get("credit_spread_hy")),
        "sentiment": _num(snapshot.get("fear_greed_score")),
        "vrp": _num((vrp_data or {}).get("vrp")),
        "breadth": _num(snapshot.get("market_breadth")),
        "move": _num(move_index if move_index is not None else snapshot.get("move_index")),
        "skew": _num(snapshot.get("skew")),
        "put_call": _num(snapshot.get("put_call_ratio")),
    }

    components = {name: round(normalize(name, value), 1) for name, value in raw.items() if value is not None}
    weights = {name: COMPONENTS[name]["weight"] for name in components}
    missing = [name for name in COMPONENTS if name not in components]

    if not components:
        return {"score": None, "status": "Insufficient data", "components": {}, "weights": {}, "missing": missing}

    total_weight = sum(weights.values())
    score = round(sum(components[k] * weights[k] for k in components) / total_weight, 1)

    threshold, interpretation, color, description = next(b for b in BANDS if score >= b[0])
    return {
        "score": score,
        "interpretation": interpretation,
        "color": color,
        "description": description,
        "components": components,
        "weights": weights,
        "missing": missing,
        "component_count": len(components),
        "timestamp": datetime.now().isoformat(),
    }
