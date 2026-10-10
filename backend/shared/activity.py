"""Additive, secret-safe metadata for user-facing activity events."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping

from backend.shared.log_redaction import redact_log_text, redact_sensitive_values
from backend.shared.provider_route_identity import PROVIDER_ERROR_ACTIVITY_EVENTS


ACTIVITY_SCHEMA_VERSION = 1
_TEXT_LIMIT = 2_000
_FEEDBACK_LIMIT = 8_000
_GUIDANCE_LIMIT = 4_000
_IDENTITY_LIMIT = 240
_MAX_DEPTH = 12
_SECRET_KEY = re.compile(
    r"(?:api[_-]?key|authorization|password|secret|access[_-]?token|refresh[_-]?token)",
    re.IGNORECASE,
)
_LEGACY_TEXT_LIMITS = {
    "reasoning": _FEEDBACK_LIMIT,
    "validator_feedback": _FEEDBACK_LIMIT,
    "validation_feedback": _FEEDBACK_LIMIT,
    "feedback": _FEEDBACK_LIMIT,
    "summary": _TEXT_LIMIT,
    "message": _TEXT_LIMIT,
    "error_summary": _TEXT_LIMIT,
    "error_detail": _TEXT_LIMIT,
    "error_message": _TEXT_LIMIT,
    "error": _TEXT_LIMIT,
    "detail": _TEXT_LIMIT,
    "warning": _TEXT_LIMIT,
    "terminal_guidance": _GUIDANCE_LIMIT,
    "guidance": _GUIDANCE_LIMIT,
    "repair_guidance": _GUIDANCE_LIMIT,
}
_PROVIDER_ERROR_OCCURRENCE_KEYS = (
    "task_id",
    "provider_request_id",
    "route_kind",
    "elapsed_minutes",
    "from_model",
    "original_model",
    "fallback_model",
    "cooldown_until",
)


def _redact_value(value: Any, *, depth: int = 0) -> Any:
    """Recursively redact credential-shaped values without changing payload shape."""
    if depth >= _MAX_DEPTH:
        return "[bounded]"
    if isinstance(value, Mapping):
        return {
            str(key): (
                "[redacted]"
                if _SECRET_KEY.search(str(key))
                else _redact_value(item, depth=depth + 1)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_value(item, depth=depth + 1) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(item, depth=depth + 1) for item in value)
    if isinstance(value, str):
        return redact_sensitive_values(value)
    return value


def redact_activity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a recursively redacted copy suitable for WebSocket publication."""
    redacted = _redact_value(payload)

    def bound_known_text(value: Any, *, depth: int = 0) -> Any:
        if depth >= _MAX_DEPTH:
            return value
        if isinstance(value, Mapping):
            return {
                key: (
                    (
                        item[: max(0, _LEGACY_TEXT_LIMITS[str(key).lower()] - 13)]
                        + "...[truncated]"
                    )
                    if (
                        isinstance(item, str)
                        and str(key).lower() in _LEGACY_TEXT_LIMITS
                        and len(item) > _LEGACY_TEXT_LIMITS[str(key).lower()]
                    )
                    else item
                    if isinstance(item, str) and str(key).lower() in _LEGACY_TEXT_LIMITS
                    else bound_known_text(item, depth=depth + 1)
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [bound_known_text(item, depth=depth + 1) for item in value]
        if isinstance(value, tuple):
            return tuple(bound_known_text(item, depth=depth + 1) for item in value)
        return value

    return bound_known_text(redacted)


def _text(value: Any, limit: int) -> str:
    text = redact_log_text(value).strip()
    suffix = "...[truncated]"
    if len(text) > limit:
        return text[: max(0, limit - len(suffix))] + suffix
    return text


def _first(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return value
    return ""


def _identity_digest(identity: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]


def fill_missing_activity_metadata(
    payload: Mapping[str, Any],
    *,
    workflow: str = "",
    run_id: str = "",
    generation: Any = "",
    source_type: str = "",
    source_id: str = "",
    role_id: str = "",
    phase: str = "",
    classification: str = "",
    category: str = "",
    severity: str = "",
    outcome: str = "",
    feedback: Any = "",
    detail: Any = "",
    message: Any = "",
    guidance: Any = "",
) -> dict[str, Any]:
    """Fill producer-known activity fields without replacing legacy or explicit values."""
    result = dict(payload)
    defaults = {
        "activity_workflow": workflow,
        "activity_run_id": run_id,
        "activity_generation": generation,
        "activity_source_type": source_type,
        "activity_source_id": source_id,
        "activity_role_id": role_id,
        "activity_phase": phase,
        "activity_classification": classification,
        "activity_category": category,
        "activity_severity": severity,
        "activity_outcome": outcome,
        "activity_feedback": feedback,
        "activity_detail": detail,
        "activity_message": message,
        "activity_guidance": guidance,
    }
    for key, value in defaults.items():
        if result.get(key) in (None, "") and value not in (None, ""):
            result[key] = value
    return result


def _classify(event_type: str, payload: Mapping[str, Any]) -> tuple[str, str, str, bool]:
    name = event_type.lower()
    explicit = str(payload.get("activity_classification") or "").lower()
    explicit_category = str(payload.get("activity_category") or "").lower()
    explicit_outcome = str(payload.get("activity_outcome") or "").lower()
    validation_stage = str(payload.get("validation_stage") or "").lower()
    provider_markers = ("provider", "oauth", "credit", "rate_limit", "fallback", "retry")
    operational_markers = (
        "error", "failed", "failure", "overflow", "exhaust", "stopped", "cancel",
        "timeout", "pause", "repair", "invalid_json", "parse",
    )
    validator_markers = (
        "accept", "reject", "decline", "validation", "validator", "critique",
    )

    operational_validation_stages = {
        "internal_error",
        "provider_error",
        "parse_error",
        "context_error",
        "pre-validation",
        "pre_validation",
    }
    if validation_stage in operational_validation_stages:
        classification = "operational"
    elif explicit in {"semantic", "operational", "informational"}:
        classification = explicit
    elif any(marker in name for marker in operational_markers) or any(
        marker in name for marker in provider_markers
    ):
        classification = "operational"
    elif any(marker in name for marker in validator_markers):
        classification = "semantic"
    else:
        classification = "informational"

    if explicit_category:
        category = explicit_category
    elif validation_stage in operational_validation_stages:
        category = "provider" if validation_stage == "provider_error" else "workflow"
    elif any(marker in name for marker in provider_markers):
        category = "provider"
    elif classification == "semantic":
        category = "validator"
    elif any(marker in name for marker in operational_markers):
        category = "workflow"
    else:
        category = str(payload.get("activity_category") or "progress")

    if explicit_outcome:
        outcome = explicit_outcome
    elif any(marker in name for marker in ("accepted", "complete", "success", "resumed")):
        outcome = "success"
    elif any(marker in name for marker in ("reject", "decline")):
        outcome = "rejected"
    elif any(marker in name for marker in operational_markers):
        outcome = "error"
    else:
        outcome = str(payload.get("activity_outcome") or "info")

    severity = str(payload.get("activity_severity") or "")
    if not severity:
        severity = "error" if outcome == "error" else "warning" if outcome == "rejected" else "info"
    return category, severity, outcome, classification == "semantic"


def normalize_activity_payload(event_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Add deterministic activity metadata while preserving every legacy field."""
    safe = redact_activity_payload(payload)
    category, severity, outcome, semantic = _classify(event_type, safe)

    workflow = _text(
        _first(safe, "activity_workflow", "workflow", "workflow_type", "workflow_mode", "mode"),
        _IDENTITY_LIMIT,
    )
    run_id = _text(
        _first(safe, "activity_run_id", "run_id", "proof_run_id", "session_id"),
        _IDENTITY_LIMIT,
    )
    source_type = _text(_first(safe, "activity_source_type", "source_type"), _IDENTITY_LIMIT)
    source_id = _text(
        _first(
            safe,
            "activity_source_id",
            "source_id",
            "submission_id",
            "paper_id",
            "topic_id",
            "proof_id",
        ),
        _IDENTITY_LIMIT,
    )
    role_id = _text(
        _first(safe, "activity_role_id", "role_id", "role", "submitter_id"),
        _IDENTITY_LIMIT,
    )
    phase = _text(
        _first(safe, "activity_phase", "phase", "current_phase", "paper_phase"),
        _IDENTITY_LIMIT,
    )
    generation = _text(
        _first(safe, "activity_generation", "lifecycle_generation", "generation"),
        _IDENTITY_LIMIT,
    )

    feedback = _text(
        _first(
            safe,
            "activity_feedback",
            "validator_feedback",
            "feedback",
            "reasoning",
            "reason",
            "summary",
        ),
        _FEEDBACK_LIMIT,
    )
    detail = _text(
        _first(safe, "activity_detail", "error_summary", "error", "detail"),
        _TEXT_LIMIT,
    )
    message = _text(
        _first(safe, "activity_message", "message", "summary", "reasoning", "reason"),
        _TEXT_LIMIT,
    )
    guidance = _text(
        _first(safe, "activity_guidance", "terminal_guidance", "guidance", "repair_guidance"),
        _GUIDANCE_LIMIT,
    )

    identity = {
        "event_type": event_type,
        "workflow": workflow,
        "run_id": run_id,
        "generation": generation,
        "source_type": source_type,
        "source_id": source_id,
        "role_id": role_id,
        "phase": phase,
        "retry_attempt": safe.get("retry_attempt"),
        "decision": safe.get("decision"),
        "attempt": _first(safe, "attempt", "attempt_index", "list_attempt"),
        "round": _first(safe, "proof_round_index", "round_index", "current_round"),
        "subject": _first(
            safe,
            "candidate_id",
            "theorem_id",
            "proof_label",
            "proposal_id",
            "target_hash",
        ),
        "counter": _first(
            safe,
            "submission_number",
            "total_acceptances",
            "total_rejections",
            "acceptance_count",
            "rejection_count",
        ),
    }
    # Per-call discriminators of a provider error. They shape activity_id only,
    # never the correlation digest, so distinct failures of one role and route
    # stay distinct while a redelivered payload keeps its id.
    occurrence: dict[str, Any] = {}
    if event_type in PROVIDER_ERROR_ACTIVITY_EVENTS:
        # Distinct models or hosts failing in one scope are distinct events.
        for key in ("model", "provider", "effective_host_provider", "configured_host_provider"):
            value = safe.get(key)
            if value not in (None, ""):
                identity[key] = _text(value, _IDENTITY_LIMIT)
        for key in _PROVIDER_ERROR_OCCURRENCE_KEYS:
            value = safe.get(key)
            if value not in (None, ""):
                occurrence[key] = _text(value, _IDENTITY_LIMIT)
        correlation = _first(safe, "activity_correlation_id", "correlation_id", "retry_correlation_id")
        if correlation:
            occurrence["correlation"] = _text(correlation, _IDENTITY_LIMIT)
        text = detail or message
        if text:
            occurrence["text"] = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    digest = _identity_digest(identity)
    occurrence_digest = _identity_digest({**identity, "occurrence": occurrence}) if occurrence else digest
    activity_id = _text(safe.get("activity_id") or safe.get("event_id"), _IDENTITY_LIMIT)

    safe.update(
        {
            "activity_schema_version": ACTIVITY_SCHEMA_VERSION,
            "activity_id": activity_id or f"{event_type}:{occurrence_digest}",
            "activity_correlation_id": _text(
                safe.get("activity_correlation_id") or safe.get("correlation_id") or digest,
                _IDENTITY_LIMIT,
            ),
            "activity_category": category,
            "activity_severity": severity,
            "activity_outcome": outcome,
            "activity_semantic": semantic,
            "activity_message": message,
            "activity_detail": detail,
            "activity_guidance": guidance,
            "activity_feedback": feedback,
            "activity_workflow": workflow,
            "activity_run_id": run_id,
            "activity_generation": generation,
            "activity_source_type": source_type,
            "activity_source_id": source_id,
            "activity_role_id": role_id,
            "activity_phase": phase,
            "activity_retryable": bool(safe.get("retryable", False)),
            "activity_recoverable": bool(safe.get("recoverable", False)),
            "activity_fatal": bool(safe.get("fatal", False)),
        }
    )
    return safe
