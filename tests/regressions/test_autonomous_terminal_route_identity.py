"""Autonomous terminal events carry route identity for child and parent repairs."""
from unittest.mock import AsyncMock

import pytest

from backend.autonomous.core.autonomous_coordinator import AutonomousCoordinator
from backend.autonomous.memory.research_metadata import research_metadata
from backend.shared.api_client_manager import api_client_manager
from backend.shared.config import system_config
from backend.shared.context_overflow import CONTEXT_OVERFLOW_STOP_MESSAGE
from backend.shared.models import AutonomousTerminalEvent
from backend.shared.provider_errors import ProviderRepairRequiredError
from backend.shared.provider_route_identity import route_identity_fields, with_route_identity

_AUTO_IDENTITY = (
    "[Autonomous · Submitter 2 · vendor/test-free:free via OpenRouter · "
    "host auto → TestHost]"
)


@pytest.fixture(autouse=True)
def _isolated_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(system_config, "data_dir", str(tmp_path))
    monkeypatch.setattr(api_client_manager, "_role_model_configs", {})
    monkeypatch.setattr(api_client_manager, "_role_fallback_state", {})


def _coordinator() -> AutonomousCoordinator:
    coordinator = AutonomousCoordinator()
    coordinator._run_id = "run-route-identity"
    coordinator._lifecycle_generation = 2
    return coordinator


def _aggregator_child_payload() -> dict:
    """A child Aggregator repair payload composed for its own workflow surface."""
    identity = route_identity_fields(
        workflow_mode="aggregator",
        role_id="aggregator_submitter_2",
        model="vendor/test-free:free",
        provider="openrouter",
        configured_host_provider="",
        upstream_host_provider="TestHost",
    )
    return {
        **identity,
        "reason": "spending_limit",
        "recoverable": True,
        "workflow_continues": False,
        "message": with_route_identity(
            "Research stopped because OpenRouter cannot serve the progression-critical "
            "role aggregator_submitter_2.",
            identity,
        ),
        "terminal_guidance": "Raise the provider limit or choose another model, then restart.",
    }


def test_aggregator_child_repair_with_foreign_mode_builds_autonomous_identity():
    coordinator = _coordinator()
    payload = _aggregator_child_payload()
    assert payload["workflow_mode"] == "aggregator"
    assert payload["route_identity"].startswith("[Aggregator · Submitter 2")

    coordinator._mark_terminal_stop(
        reason="provider_repair_required",
        message=payload["message"],
        payload=payload,
    )

    event = coordinator._terminal_event
    assert isinstance(event, AutonomousTerminalEvent)
    assert event.workflow_mode == "autonomous"
    assert event.role_id == "aggregator_submitter_2"
    assert event.role_label == "Submitter 2"
    assert event.host_routing == "auto"
    assert event.configured_host_provider == ""
    assert event.effective_host_provider == "TestHost"
    assert event.route_identity == _AUTO_IDENTITY
    assert event.message.startswith(f"{_AUTO_IDENTITY} Research stopped because")
    assert "[Aggregator" not in event.message


def test_raw_aggregator_child_payload_gains_identity_without_invented_host():
    coordinator = _coordinator()

    coordinator._mark_terminal_stop(
        reason="provider_repair_required",
        message="Research stopped because OpenRouter cannot serve the role.",
        payload={
            "workflow_mode": "aggregator",
            "role_id": "aggregator_submitter_2",
            "provider": "openrouter",
            "provider_label": "OpenRouter",
            "model": "vendor/test-free:free",
            "reason": "spending_limit",
        },
    )

    event = coordinator._terminal_event
    assert event.workflow_mode == "autonomous"
    assert event.route_identity == (
        "[Autonomous · Submitter 2 · vendor/test-free:free via OpenRouter]"
    )
    assert event.host_routing is None
    assert event.configured_host_provider is None
    assert event.message.startswith(event.route_identity)


def test_context_overflow_terminal_drops_foreign_mode_without_identity_prefix():
    coordinator = _coordinator()

    coordinator._mark_context_overflow_stop({
        "workflow_mode": "aggregator",
        "role_id": "aggregator_submitter_2",
        "configured_model": "vendor/test-free:free",
        "configured_provider": "openrouter",
        "configured_host_provider": "TestHost",
        "effective_model": "vendor/test-free:free",
        "effective_provider": "openrouter",
    })

    event = coordinator._terminal_event
    assert event.workflow_mode == "autonomous"
    assert event.configured_host_provider == "TestHost"
    assert event.route_identity is None
    assert event.message == CONTEXT_OVERFLOW_STOP_MESSAGE


@pytest.mark.asyncio
async def test_parent_provider_repair_notification_carries_route_identity(monkeypatch):
    coordinator = _coordinator()
    events = []

    async def capture(event_type, payload):
        events.append((event_type, payload))

    coordinator.set_broadcast_callback(capture)
    monkeypatch.setattr(research_metadata, "get_stats", AsyncMock(return_value={}))
    monkeypatch.setattr(research_metadata, "save_terminal_event", AsyncMock(return_value=True))
    error = ProviderRepairRequiredError(
        provider="openrouter",
        provider_label="OpenRouter",
        role_id="aggregator_submitter_2",
        model="vendor/test-model",
        reason="spending_limit",
        message="OpenRouter reported a spending limit.",
        terminal_guidance="Raise the provider limit, then restart.",
        configured_provider="openrouter",
        configured_model="vendor/test-model",
        configured_host_provider="TestHost",
    )

    coordinator._mark_provider_repair_stop(error)
    await coordinator._broadcast_stopped_once()

    identity = "[Autonomous · Submitter 2 · vendor/test-model via OpenRouter · host TestHost (pinned)]"
    event = coordinator._terminal_event
    assert event.route_identity == identity
    assert event.host_routing == "pinned"
    assert event.configured_host_provider == "TestHost"
    assert event.effective_host_provider == "TestHost"
    assert event.message.startswith(f"{identity} Research stopped because OpenRouter")

    assert len(events) == 1
    event_type, payload = events[0]
    assert event_type == "auto_research_stopped"
    assert payload["notification_kind"] == "model_error"
    assert payload["workflow_mode"] == "autonomous"
    assert payload["route_identity"] == identity
    assert payload["message"].startswith(identity)
    # Route identity stays out of the stable dedupe key.
    assert payload["notification_key"] == (
        "openrouter:aggregator_submitter_2:spending_limit:vendor/test-model"
    )
