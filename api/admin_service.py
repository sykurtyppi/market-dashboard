"""Admin pages — System Health and Settings.

System Health wraps the existing HealthCheckSystem summary and maps each source
onto the UI's semantic dot. Settings surfaces credential *presence* (never
values) plus non-secret runtime/model configuration. Neither is cached: health
must reflect the current moment, and settings are cheap to assemble.
"""
import os
from pathlib import Path
from typing import Any, Dict

import yaml

from api.deps import get_health

# Map the backend's health vocabulary to the four UI states.
_STATE_BY_STATUS = {
    "healthy": "good",
    "stale": "warn",
    "degraded": "warn",
    "down": "crit",
    "unknown": "neutral",
}


def _state(status: str) -> str:
    return _STATE_BY_STATUS.get((status or "").lower(), "neutral")


def build_system_health() -> Dict[str, Any]:
    summary = get_health().get_health_summary() or {}
    raw_sources = summary.get("sources", {}) or {}

    sources = []
    for key, src in raw_sources.items():
        status = src.get("status", "unknown")
        sources.append({
            "key": key,
            "name": src.get("name", key),
            "status": status,
            "state": _state(status),
            "last_update": src.get("last_update"),
            "age_hours": src.get("age_hours"),
            "message": src.get("message", ""),
        })

    # Stable order: worst first so problems surface at the top of the page.
    severity = {"crit": 0, "warn": 1, "neutral": 2, "good": 3}
    sources.sort(key=lambda s: (severity.get(s["state"], 9), s["name"]))

    overall = summary.get("overall_status", "unknown")
    return {
        "overall_status": overall,
        "overall_state": _state(overall),
        "as_of": summary.get("timestamp"),
        "sources": sources,
        "summary": summary.get("summary", {}),
        "total_sources": summary.get("total_sources", len(sources)),
    }


# Secret-bearing keys in config.yaml that must never be echoed to a client.
_SECRET_CONFIG_KEYS = {"fred_api_key", "smtp_password", "smtp_server", "email_address"}


def _read_config_yaml() -> Dict[str, Any]:
    path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def build_settings() -> Dict[str, Any]:
    from utils.secrets_helper import get_all_secrets

    warnings: list[str] = []
    protected = bool(os.getenv("MARKET_API_TOKEN"))
    if not protected:
        warnings.append(
            "MARKET_API_TOKEN is not set — settings and manual refresh are "
            "unauthenticated on this deployment."
        )

    # Credential presence only. get_all_secrets() returns configured/source and
    # never the secret value itself.
    creds_raw = get_all_secrets()
    credentials = [
        {"name": name, "configured": bool(info.get("configured")), "source": info.get("source")}
        for name, info in creds_raw.items()
    ]

    raw = _read_config_yaml()
    schedule_items = []
    update_time = raw.get("update_time")
    if update_time:
        schedule_items.append({"label": "Daily update time", "value": str(update_time)})
    schedule_items.append({
        "label": "Email alerts",
        "value": "Enabled" if raw.get("email_alerts") else "Disabled",
    })

    # Model parameters come from parameters.yaml via the cfg singleton; degrade
    # gracefully with get() defaults if a key is missing.
    from config import cfg

    def _p(path: str, default: Any = None) -> str:
        v = cfg.get(path, default)
        return "—" if v is None else str(v)

    model_items = [
        {"label": "LEFT EMA period", "value": _p("credit.left_strategy.ema_period", 330)},
        {"label": "LEFT entry threshold", "value": _p("credit.left_strategy.entry_threshold", 0.65)},
        {"label": "LEFT exit threshold", "value": _p("credit.left_strategy.exit_threshold", 1.40)},
        {"label": "Equity P/C bearish", "value": _p("options.equity_put_call.bearish_threshold", 1.0)},
        {"label": "Equity P/C bullish", "value": _p("options.equity_put_call.bullish_threshold", 0.7)},
        {"label": "VRP lookback (days)", "value": _p("volatility.vrp.lookback_days", 21)},
    ]

    origins = os.getenv("FRONTEND_ORIGINS", "http://localhost:3000, http://127.0.0.1:3000")
    runtime_items = [
        {"label": "Settings/refresh auth", "value": "Token required" if protected else "Not enforced"},
        {"label": "Allowed frontend origins", "value": origins},
    ]

    config = [
        {"title": "Runtime & Security", "items": runtime_items},
        {"title": "Update Schedule", "items": schedule_items},
        {"title": "Model Parameters", "items": model_items},
    ]

    return {
        "protected": protected,
        "credentials": credentials,
        "config": config,
        "warnings": warnings,
    }
