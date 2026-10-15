"""Canonical provider recovery event construction with stable identities."""
from __future__ import annotations

import hashlib
from typing import Any, Dict

from backend.shared.log_redaction import redact_log_text


def canonical_recovery_event(
    event_type: str,
    payload: Dict[str, Any],
    *,
    recovery_action: str,
    retry_layer: str = "",
) -> Dict[str, Any]:
    provider = redact_log_text(str(payload.get("provider") or "unknown"), 120)
    role_id = redact_log_text(str(payload.get("role_id") or "unknown"), 160)
    model = redact_log_text(str(payload.get("model") or ""), 240)
    reason = redact_log_text(str(payload.get("reason") or event_type), 160)
    reset = str(payload.get("cooldown_until") or payload.get("resets_at") or "")
    explicit_correlation_id = redact_log_text(
        str(
            payload.get("activity_correlation_id")
            or payload.get("correlation_id")
            or ""
        ),
        240,
    ).strip()
    correlation_source = "|".join((provider, role_id, model, reason, reset))
    correlation_id = explicit_correlation_id or hashlib.sha256(
        correlation_source.encode("utf-8")
    ).hexdigest()[:24]
    occurrence = str(payload.get("retry_attempt") or payload.get("lifecycle_generation") or "")
    event_id = f"{event_type}:{correlation_id}"
    if occurrence:
        event_id = f"{event_id}:{occurrence}"
    return {
        **payload,
        "contract_version": 1,
        "event_type": event_type,
        "event_id": event_id,
        "correlation_id": correlation_id,
        "recovery_action": recovery_action,
        "retry_layer": retry_layer,
    }
