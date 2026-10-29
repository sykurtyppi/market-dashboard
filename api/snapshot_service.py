"""Durable snapshots for slow provider-backed dashboard endpoints."""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Callable, Dict

from api.deps import get_db


def _collect_sectors() -> Dict[str, Any]:
    from api.sectors_service import _build_live
    return _build_live()


def _collect_options_flow() -> Dict[str, Any]:
    from api.flows_service import _build_options_flow
    return _build_options_flow()


_collectors: dict[str, tuple[Callable[[], Dict[str, Any]], Callable[[Dict[str, Any]], bool]]] = {
    "sectors": (_collect_sectors, lambda data: bool(data.get("sectors"))),
    "options_flow": (_collect_options_flow, lambda data: len(data.get("etfs", [])) == 3),
}
_locks = {dataset: threading.Lock() for dataset in _collectors}


def available_datasets() -> list[str]:
    return sorted(_collectors)


def refresh_snapshot(dataset: str, db=None) -> Dict[str, Any]:
    """Collect and persist one dataset; reject empty/partial replacements."""
    if dataset not in _collectors:
        raise KeyError(dataset)
    db = db or get_db()
    lock = _locks.setdefault(dataset, threading.Lock())
    if not lock.acquire(blocking=False):
        return {"dataset": dataset, "status": "already_running"}
    try:
        collect, valid = _collectors[dataset]
        try:
            payload = collect()
            if not valid(payload):
                raise ValueError("collector returned empty or incomplete data")
            db.save_api_snapshot(dataset, payload, as_of=payload.get("as_of"))
            return {"dataset": dataset, "status": "ok", "as_of": payload.get("as_of")}
        except Exception as exc:
            db.record_api_snapshot_failure(dataset, str(exc))
            return {"dataset": dataset, "status": "failed"}
    finally:
        lock.release()


def snapshot_response(
    db,
    dataset: str,
    empty_payload: Dict[str, Any],
    *,
    stale_after_seconds: int,
) -> Dict[str, Any]:
    """Merge a persisted payload with freshness metadata, never an error detail."""
    snapshot = db.get_api_snapshot(dataset)
    if not snapshot or not snapshot.get("payload"):
        result = dict(empty_payload)
        status = (snapshot or {}).get("refresh_status") or "missing"
        result.update({
            "stale": True,
            "refresh_status": status,
            "data_age_seconds": None,
        })
        return result

    result = dict(snapshot["payload"])
    refreshed_at = snapshot.get("refreshed_at")
    age = None
    if refreshed_at:
        try:
            parsed = datetime.fromisoformat(refreshed_at.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            age = max(0, int((datetime.now(timezone.utc) - parsed).total_seconds()))
        except (TypeError, ValueError):
            age = None
    status = snapshot.get("refresh_status") or "unknown"
    result.update({
        "stale": status != "ok" or age is None or age > stale_after_seconds,
        "refresh_status": status,
        "data_age_seconds": age,
    })
    return result
