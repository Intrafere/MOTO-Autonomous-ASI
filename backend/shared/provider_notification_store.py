"""Durable non-secret provider/OAuth notifications for frontend recovery."""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from backend.shared.config import system_config
from backend.shared.log_redaction import redact_log_text

logger = logging.getLogger(__name__)

PROVIDER_NOTIFICATIONS_FILENAME = "provider_notifications.json"
MAX_PROVIDER_NOTIFICATIONS = 20
PROVIDER_NOTIFICATION_TTL_SECONDS = 7 * 24 * 60 * 60

_store_lock = threading.Lock()


def _notifications_path() -> Path:
    return Path(system_config.data_dir) / PROVIDER_NOTIFICATIONS_FILENAME


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _coerce_created_at(value: Any) -> str:
    raw = str(value or "").strip()
    return raw or _now_iso()


def _notification_age_seconds(notification: dict[str, Any], now: float) -> float:
    raw = str(notification.get("created_at") or notification.get("timestamp") or "").strip()
    if not raw:
        return 0.0
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max(0.0, now - parsed.timestamp())


def _read_payload() -> dict[str, Any]:
    path = _notifications_path()
    try:
        if not path.exists():
            return {"notifications": []}
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("Ignoring corrupt provider notifications file %s: %s", redact_log_text(path, 240), exc)
        return {"notifications": []}
    except OSError as exc:
        logger.warning("Failed to read provider notifications file %s: %s", redact_log_text(path, 240), exc)
        return {"notifications": []}
    return payload if isinstance(payload, dict) else {"notifications": []}


def _write_payload(payload: dict[str, Any]) -> None:
    path = _notifications_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.tmp")
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp_path.replace(path)
    except OSError as exc:
        logger.warning("Failed to persist provider notifications file %s: %s", redact_log_text(path, 240), exc)


def _clean_notifications(notifications: list[Any]) -> list[dict[str, Any]]:
    now = time.time()
    cleaned: list[dict[str, Any]] = []
    for item in notifications:
        if not isinstance(item, dict):
            continue
        if _notification_age_seconds(item, now) > PROVIDER_NOTIFICATION_TTL_SECONDS:
            continue
        cleaned.append(item)
    return cleaned[-MAX_PROVIDER_NOTIFICATIONS:]


def _safe_string(value: Any, max_chars: int = 700) -> str:
    text = redact_log_text(str(value or "")).strip()
    if max_chars is not None and max_chars >= 0 and len(text) > max_chars:
        if max_chars <= 3:
            return text[:max_chars]
        return text[: max_chars - 3] + "..."
    return text


def _stable_notification_key(provider: str, role_id: str, reason: str, model: str) -> str:
    parts = [provider, role_id, reason, model or "*"]
    return ":".join(part.replace(":", "_") for part in parts)


def record_provider_notification(event_type: str, payload: Dict[str, Any]) -> dict[str, Any]:
    """Persist a recoverable provider/OAuth notification and return the stored payload."""
    provider = _safe_string(payload.get("provider"), 120) or "oauth"
    role_id = _safe_string(payload.get("role_id"), 160) or provider
    reason = _safe_string(payload.get("reason"), 160) or "provider_error"
    model = _safe_string(payload.get("model"), 240)
    created_at = _coerce_created_at(payload.get("created_at") or payload.get("_serverTimestamp"))
    key_model = model
    if reason == "usage_limit_reached" and payload.get("cooldown_until") is not None:
        key_model = f"{model or '*'}@{payload.get('cooldown_until')}"
    notification_key = _stable_notification_key(provider, role_id, reason, key_model)
    notification_id = _safe_string(payload.get("id"), 240) or notification_key

    notification = {
        "id": notification_id,
        "notification_key": notification_key,
        "event_type": _safe_string(event_type, 120),
        "created_at": created_at,
        "provider": provider,
        "provider_label": _safe_string(payload.get("provider_label"), 120),
        "role_id": role_id,
        "model": model,
        "reason": reason,
        "recoverable": bool(payload.get("recoverable", False)),
        "message": _safe_string(payload.get("message"), 700),
        "error_summary": _safe_string(payload.get("error_summary"), 700),
        "oauth_error_message": _safe_string(payload.get("oauth_error_message"), 1800),
    }
    for numeric_key in ("resets_at", "resets_in_seconds", "cooldown_until"):
        raw_value = payload.get(numeric_key)
        if raw_value is None:
            continue
        try:
            notification[numeric_key] = int(raw_value)
        except (TypeError, ValueError):
            continue
    if "fallback_model" in payload:
        notification["fallback_model"] = _safe_string(payload.get("fallback_model"), 240)
    if "plan_type" in payload:
        notification["plan_type"] = _safe_string(payload.get("plan_type"), 120)

    with _store_lock:
        stored_payload = _read_payload()
        notifications = _clean_notifications(stored_payload.get("notifications") or [])
        notifications = [
            item for item in notifications
            if item.get("notification_key") != notification_key and item.get("id") != notification_id
        ]
        notifications.append(notification)
        stored_payload["notifications"] = notifications[-MAX_PROVIDER_NOTIFICATIONS:]
        _write_payload(stored_payload)

    return notification


def list_provider_notifications() -> List[dict[str, Any]]:
    """Return recent provider/OAuth notifications, newest last."""
    with _store_lock:
        payload = _read_payload()
        notifications = _clean_notifications(payload.get("notifications") or [])
        if len(notifications) != len(payload.get("notifications") or []):
            payload["notifications"] = notifications
            _write_payload(payload)
        return list(notifications)
