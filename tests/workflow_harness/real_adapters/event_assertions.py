from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypedDict


class FatalContextOverflowEvent(TypedDict, total=False):
    workflow_mode: str
    role_id: str
    configured_model: str
    configured_provider: str
    effective_model: str
    effective_provider: str
    effective_host_provider: str
    route_kind: str
    reason: str
    message: str
    resolution: str
    fatal: bool


class ProofContextOverflowEvent(FatalContextOverflowEvent, total=False):
    source_type: str
    source_id: str
    theorem_id: str
    overflow_origin: str
    prompt_tokens: int
    max_input_tokens: int


EventRecord = tuple[str, Mapping[str, Any]]


def event_payloads(
    events: Sequence[EventRecord],
    event_type: str,
) -> list[Mapping[str, Any]]:
    return [payload for current_type, payload in events if current_type == event_type]


def assert_event_count(
    events: Sequence[EventRecord],
    event_type: str,
    expected: int,
) -> list[Mapping[str, Any]]:
    payloads = event_payloads(events, event_type)
    assert len(payloads) == expected, (
        f"expected {expected} {event_type!r} event(s), found {len(payloads)}: {payloads!r}"
    )
    return payloads


def assert_no_events(events: Sequence[EventRecord], *event_types: str) -> None:
    for event_type in event_types:
        assert_event_count(events, event_type, 0)


def assert_fatal_context_overflow_event(
    payload: Mapping[str, Any],
    *,
    workflow_mode: str,
    role_id: str,
) -> FatalContextOverflowEvent:
    required = {
        "workflow_mode",
        "role_id",
        "configured_model",
        "configured_provider",
        "reason",
        "message",
        "resolution",
    }
    missing = required.difference(payload)
    assert not missing, f"context_overflow_error missing fields: {sorted(missing)}"
    assert payload["workflow_mode"] == workflow_mode
    assert payload["role_id"] == role_id
    assert payload["reason"] == "context_overflow"
    assert payload.get("fatal", True) is True
    return dict(payload)  # type: ignore[return-value]


def assert_proof_context_overflow_event(
    payload: Mapping[str, Any],
    *,
    workflow_mode: str,
) -> ProofContextOverflowEvent:
    required = {
        "workflow_mode",
        "source_type",
        "source_id",
        "configured_model",
        "configured_provider",
        "overflow_origin",
        "message",
        "fatal",
    }
    missing = required.difference(payload)
    assert not missing, f"proof_context_overflow missing fields: {sorted(missing)}"
    assert payload["workflow_mode"] == workflow_mode
    assert payload["fatal"] is False
    return dict(payload)  # type: ignore[return-value]


def assert_route_identity(
    payload: Mapping[str, Any],
    *,
    configured_model: str,
    configured_provider: str,
    effective_model: str,
    effective_provider: str,
    effective_host_provider: str | None = None,
    route_kind: str | None = None,
) -> None:
    assert payload["configured_model"] == configured_model
    assert payload["configured_provider"] == configured_provider
    assert payload["effective_model"] == effective_model
    assert payload["effective_provider"] == effective_provider
    if effective_host_provider is not None:
        assert payload["effective_host_provider"] == effective_host_provider
    if route_kind is not None:
        assert payload["route_kind"] == route_kind
