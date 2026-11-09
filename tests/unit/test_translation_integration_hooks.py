"""Shared hooks that let Auto Top-p Translate run beside any workflow.

Translator roles never receive Assistant proof memory, provider activity from
translation roles names the ``translation`` workflow even while a top-level
workflow owns execution, and in-process observers see every broadcast event.
"""
from __future__ import annotations

import pytest

from backend.api.routes import websocket as websocket_route
from backend.shared import api_client_manager as api_client_module
from backend.shared.api_client_manager import APIClientManager, _retry_workflow_mode
from backend.shared.provider_route_identity import (
    PROVIDER_ERROR_ACTIVITY_EVENTS,
    VALID_WORKFLOW_MODES,
    route_identity_fields,
    workflow_label,
)
from backend.shared.workflow_start_guard import WorkflowLease, workflow_start_guard


@pytest.mark.parametrize(
    ("role_id", "task_id"),
    [
        ("translation_submitter", "translation_sub_00001"),
        ("translation_validator", "translation_val_00001"),
        ("translation_language_probe_1", "translation_probe_00001"),
    ],
)
def test_translation_roles_never_receive_assistant_memory(role_id: str, task_id: str) -> None:
    assert APIClientManager._assistant_memory_role_is_excluded(role_id, task_id, "Translate this passage.")


def test_translation_task_id_alone_excludes_assistant_memory() -> None:
    assert APIClientManager._assistant_memory_role_is_excluded("", "translation_sub_00001", "")


def test_ordinary_submitter_still_receives_assistant_memory() -> None:
    assert not APIClientManager._assistant_memory_role_is_excluded(
        "aggregator_submitter_1",
        "agg_sub1_001",
        "Propose a new solution step.",
    )


def test_translation_roles_name_their_own_workflow_mode() -> None:
    assert "translation" in VALID_WORKFLOW_MODES
    assert workflow_label("translation") == "Foreign Language Translation w/ Top-p"
    assert "translation_repair_required" in PROVIDER_ERROR_ACTIVITY_EVENTS
    assert _retry_workflow_mode("translation_submitter") == "translation"
    assert _retry_workflow_mode("translation_validator") == "translation"


def test_translation_mode_wins_over_the_active_workflow_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(workflow_start_guard, "_active_lease", WorkflowLease(owner="autonomous", generation=1))
    assert workflow_start_guard.active_owner == "autonomous"
    assert api_client_module._active_notification_workflow_mode() == "autonomous"

    assert _retry_workflow_mode("translation_submitter") == "translation"
    # Ordinary roles still follow the active owner.
    assert _retry_workflow_mode("aggregator_submitter_1") == "autonomous"


def test_translation_route_identity_keeps_mode_and_label() -> None:
    fields = route_identity_fields(
        workflow_mode=_retry_workflow_mode("translation_submitter"),
        role_id="translation_submitter",
        model="vendor/translator",
        provider="openrouter",
        configured_host_provider="",
    )
    assert fields["workflow_mode"] == "translation"
    assert fields["role_label"] == "Translator Submitter"
    assert fields["route_identity"].startswith("[Foreign Language Translation w/ Top-p · Translator Submitter · ")
    assert "vendor/translator via OpenRouter" in fields["route_identity"]


def test_role_identity_for_translation_role_names_translation_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(workflow_start_guard, "_active_lease", WorkflowLease(owner="leanoj", generation=1))
    fields = api_client_module.api_client_manager.route_identity_for_role("translation_validator")
    assert fields["workflow_mode"] == "translation"
    assert fields["route_identity"].startswith("[Foreign Language Translation w/ Top-p · Translator Validator")


@pytest.mark.asyncio
async def test_broadcast_observer_receives_events_and_failures_are_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[tuple[str, dict]] = []

    async def fake_broadcast(event_type: str, data: dict) -> None:
        sent.append((event_type, data))

    monkeypatch.setattr(websocket_route.manager, "broadcast", fake_broadcast)
    observed: list[tuple[str, dict]] = []

    def failing_observer(event_type: str, data: dict) -> None:
        raise RuntimeError("observer failure")

    def recording_observer(event_type: str, data: dict) -> None:
        observed.append((event_type, data))

    websocket_route.register_event_observer(failing_observer)
    websocket_route.register_event_observer(recording_observer)
    websocket_route.register_event_observer(recording_observer)
    try:
        payload = {"paper_id": "paper_1"}
        await websocket_route.broadcast_event("paper_completed", payload)
        assert observed == [("paper_completed", payload)]
        assert sent == [("paper_completed", payload)]

        websocket_route.unregister_event_observer(recording_observer)
        await websocket_route.broadcast_event("proof_verified", {"proof_id": "proof_1"})
        assert observed == [("paper_completed", payload)]
        assert [event for event, _data in sent] == ["paper_completed", "proof_verified"]
    finally:
        websocket_route.unregister_event_observer(failing_observer)
        websocket_route.unregister_event_observer(recording_observer)
    assert failing_observer not in websocket_route._event_observers
    assert recording_observer not in websocket_route._event_observers
