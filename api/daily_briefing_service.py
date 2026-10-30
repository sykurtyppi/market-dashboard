"""Rule-based Daily Risk Briefing.

The briefing is deliberately deterministic: it derives posture, drivers and
alert decisions from the same composite model shown on the Overview page. No
LLM or provider call is involved in generation or reads.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from processors.composite_risk import COMPONENTS, calculate_composite_risk_score
from api.deps import get_db

POSTURE_THRESHOLDS = {
    "normal": "score < 40",
    "cautious": "40 <= score < 60",
    "defensive": "60 <= score < 75",
    "stress": "score >= 75",
}

EVIDENCE = {
    "vix": ("Volatility", "/volatility"),
    "credit": ("Credit & Liquidity", "/credit"),
    "sentiment": ("Sentiment", "/sentiment"),
    "vrp": ("Volatility & VRP", "/volatility"),
    "breadth": ("Market Breadth", "/breadth"),
    "move": ("Treasury Stress", "/treasury"),
    "skew": ("Volatility", "/volatility"),
    "put_call": ("Sentiment", "/sentiment"),
}


def posture_for_score(score: Optional[float]) -> str:
    if score is None:
        return "unknown"
    if score >= 75:
        return "stress"
    if score >= 60:
        return "defensive"
    if score >= 40:
        return "cautious"
    return "normal"


def _component_contribution(result: Dict[str, Any], key: str) -> Optional[float]:
    component = result.get("components", {}).get(key)
    if component is None:
        return None
    available_weights = result.get("weights", {})
    total_weight = sum(float(weight) for weight in available_weights.values())
    if total_weight <= 0:
        return None
    return float(component) * float(COMPONENTS[key]["weight"]) / total_weight


def _driver_rows(current: Dict[str, Any], previous: Dict[str, Any]) -> list[Dict[str, Any]]:
    rows = []
    for key, spec in COMPONENTS.items():
        now = _component_contribution(current, key)
        before = _component_contribution(previous, key)
        if now is None and before is None:
            continue
        delta = round((now or 0.0) - (before or 0.0), 2)
        if abs(delta) < 0.01:
            continue
        evidence_label, evidence_href = EVIDENCE[key]
        availability_change = None
        if before is None:
            availability_change = "added"
        elif now is None:
            availability_change = "missing"
        rows.append({
            "key": key,
            "label": spec["label"],
            "risk_delta": delta,
            "direction": "deteriorating" if delta > 0 else "improving",
            "current_component_risk": current.get("components", {}).get(key),
            "previous_component_risk": previous.get("components", {}).get(key),
            "availability_change": availability_change,
            "evidence_label": evidence_label,
            "evidence_href": evidence_href,
        })
    rows.sort(key=lambda row: (-abs(row["risk_delta"]), row["key"]))
    return rows


def _why_it_matters(
    confirming: list[Dict[str, Any]],
    conflicting: list[Dict[str, Any]],
    score_delta: float,
    *,
    has_baseline: bool,
) -> str:
    if not has_baseline:
        return "No prior briefing is available for comparison."
    if not confirming and not conflicting:
        return "The composite is stable; no component moved enough to change today's risk assessment."
    direction = "deteriorating" if score_delta > 0 else "improving" if score_delta < 0 else "mixed"
    confirms = ", ".join(item["label"] for item in confirming) or "no major component"
    if conflicting:
        conflicts = ", ".join(item["label"] for item in conflicting)
        return f"Risk is {direction}, led by {confirms}, while {conflicts} provide conflicting evidence."
    return f"Risk is {direction}, with {confirms} confirming the move and no top driver offsetting it."


def build_briefing(
    current_snapshot: Dict[str, Any],
    previous_snapshot: Optional[Dict[str, Any]],
    *,
    current_vrp: Optional[Dict[str, Any]] = None,
    previous_vrp: Optional[Dict[str, Any]] = None,
    current_move: Optional[float] = None,
    previous_move: Optional[float] = None,
) -> Dict[str, Any]:
    """Build one transparent briefing from stored observations."""
    current = calculate_composite_risk_score(current_snapshot, current_vrp, current_move)
    previous = calculate_composite_risk_score(previous_snapshot or {}, previous_vrp, previous_move)
    current_score = current.get("score")
    previous_score = previous.get("score")
    score_delta = round(current_score - previous_score, 1) if current_score is not None and previous_score is not None else 0.0
    posture = posture_for_score(current_score)
    previous_posture = posture_for_score(previous_score)
    model_comparison_valid = current_score is not None and previous_score is not None
    all_driver_changes = _driver_rows(current, previous) if previous_snapshot else []
    current_inputs = set(current.get("components", {}))
    previous_inputs = set(previous.get("components", {}))
    inputs_comparable = model_comparison_valid and current_inputs == previous_inputs
    drivers = all_driver_changes[:3]
    attribution_residual = round(
        score_delta - sum(row["risk_delta"] for row in all_driver_changes),
        2,
    ) if model_comparison_valid else None

    if score_delta > 0:
        confirming = [row for row in drivers if row["risk_delta"] > 0]
        conflicting = [row for row in drivers if row["risk_delta"] < 0]
    elif score_delta < 0:
        confirming = [row for row in drivers if row["risk_delta"] < 0]
        conflicting = [row for row in drivers if row["risk_delta"] > 0]
    else:
        confirming = []
        conflicting = drivers

    triggers = []
    raw_posture_change = model_comparison_valid and posture != previous_posture
    raw_unusual_move = model_comparison_valid and abs(score_delta) >= 10
    if inputs_comparable and raw_posture_change:
        triggers.append({
            "type": "posture_change",
            "message": f"Posture changed from {previous_posture} to {posture}.",
        })
    if inputs_comparable and raw_unusual_move:
        triggers.append({
            "type": "unusual_score_move",
            "message": f"Composite risk moved {score_delta:+.1f} points in one briefing.",
        })

    missing = current.get("missing", [])
    freshness_status = current_snapshot.get("_status")
    is_fresh = current_snapshot.get("_is_fresh")
    quality_issues = []
    critical_quality_issues = []
    if missing:
        quality_issues.append(f"Missing model inputs: {', '.join(missing)}")
        if len(missing) >= 2:
            critical_quality_issues.append(f"Missing {len(missing)} model inputs: {', '.join(missing)}")
    if model_comparison_valid and not inputs_comparable:
        added_inputs = sorted(current_inputs - previous_inputs)
        removed_inputs = sorted(previous_inputs - current_inputs)
        availability_issue = "Model input availability changed"
        if added_inputs:
            availability_issue += f"; added: {', '.join(added_inputs)}"
        if removed_inputs:
            availability_issue += f"; missing: {', '.join(removed_inputs)}"
        quality_issues.append(availability_issue)
        if raw_posture_change or raw_unusual_move:
            critical_quality_issues.append(
                f"{availability_issue}; market-change alerts suppressed because scores are not comparable"
            )
    if freshness_status in {"stale", "outdated", "missing"} or is_fresh is False:
        issue = f"Snapshot freshness is {freshness_status or 'not fresh'}"
        quality_issues.append(issue)
        critical_quality_issues.append(issue)
    if critical_quality_issues:
        triggers.append({"type": "data_quality", "message": "; ".join(critical_quality_issues)})

    return {
        "as_of": current_snapshot.get("date"),
        "posture": posture,
        "previous_posture": previous_posture if previous_score is not None else None,
        "change": {
            "current_score": current_score,
            "previous_score": previous_score,
            "score_delta": score_delta if previous_score is not None else None,
            "attribution_residual": attribution_residual,
            "inputs_comparable": inputs_comparable,
            "regime_changed": inputs_comparable and posture != previous_posture,
        },
        "drivers": drivers,
        "all_driver_changes": all_driver_changes,
        "confirming": confirming,
        "conflicting": conflicting,
        "why_it_matters": _why_it_matters(
            confirming,
            conflicting,
            score_delta,
            has_baseline=model_comparison_valid,
        ),
        "inspect_next": [
            {
                "label": row["evidence_label"],
                "href": row["evidence_href"],
                "reason": f"{row['label']} is {row['direction']} and contributed {row['risk_delta']:+.2f} risk points.",
            }
            for row in drivers
        ],
        "alert": {"should_alert": bool(triggers), "triggers": triggers},
        "data_quality": {
            "status": "degraded" if quality_issues else "good",
            "issues": quality_issues,
            "missing_inputs": missing,
            "snapshot_status": freshness_status,
        },
        "rules": {
            "posture_thresholds": POSTURE_THRESHOLDS,
            "unusual_score_move": "absolute daily score change >= 10 points",
            "market_alert_comparability": "posture and score-move alerts require identical available model inputs",
            "driver_ranking": "absolute change in each component's normalized share of the available-weight composite; input additions and losses are explicit",
        },
        "outcomes": {"benchmark": "SPY", "1d": None, "5d": None, "20d": None},
    }


def generate_latest_briefing(db=None) -> Optional[Dict[str, Any]]:
    """Generate and persist the latest briefing from local SQLite only."""
    db = db or get_db()
    snapshots = db.get_recent_snapshots(2)
    if not snapshots:
        return None
    current = snapshots[0]
    # Freshness metadata is computed from the same stored snapshot and makes
    # data-quality alerts explicit without calling a provider.
    current_with_age = db.get_latest_snapshot(include_age=True) or current
    previous = snapshots[1] if len(snapshots) > 1 else None
    current_date = current.get("date")
    if not current_date:
        return None
    previous_date = previous.get("date") if previous else None
    briefing = build_briefing(
        current_with_age,
        previous,
        current_vrp=db.get_vrp_for_date(current_date),
        previous_vrp=db.get_vrp_for_date(previous_date) if previous_date else None,
        current_move=db.get_move_for_date(current_date),
        previous_move=db.get_move_for_date(previous_date) if previous_date else None,
    )
    db.save_daily_briefing(briefing)
    db.backfill_daily_briefing_outcomes("SPY")
    return db.get_latest_daily_briefing()


def get_daily_briefing(history_limit: int = 30, db=None) -> Dict[str, Any]:
    """Return the persisted briefing and recent accountability history."""
    db = db or get_db()
    latest = db.get_latest_daily_briefing() or generate_latest_briefing(db)
    if latest is None:
        return {"latest": None, "history": [], "warning": "No daily snapshot is available yet."}
    return {
        "latest": latest,
        "history": db.get_daily_briefing_history(history_limit),
        "warning": None,
    }
