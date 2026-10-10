from types import SimpleNamespace
from unittest import mock

import pytest

from backend.aggregator.core import coordinator as coordinator_module
from backend.aggregator.core.coordinator import Coordinator
from backend.shared.api_client_manager import api_client_manager
from backend.shared.config import system_config
from backend.shared.free_model_manager import free_model_manager
from backend.shared.models import ModelConfig
from backend.shared.openrouter_client import FreeModelExhaustedError
from backend.shared.provider_errors import ProviderRepairRequiredError


def _configure_openrouter_role(monkeypatch, role_id: str, *, host=None) -> None:
    monkeypatch.setitem(
        api_client_manager._role_model_configs,
        role_id,
        ModelConfig(
            provider="openrouter",
            model_id="vendor/test-free:free",
            openrouter_model_id="vendor/test-free:free",
            openrouter_provider=host,
            context_window=32000,
            max_output_tokens=4000,
        ),
    )


def _error() -> ProviderRepairRequiredError:
    return ProviderRepairRequiredError(
        provider="xai_grok_oauth",
        provider_label="xAI Grok",
        role_id="aggregator_submitter_1",
        model="grok-4.3",
        reason="spending_limit_reached",
        message="spending limit",
        terminal_guidance="repair provider",
    )


def _submitter(submitter_id: int):
    return SimpleNamespace(
        submitter_id=submitter_id,
        is_running=True,
        state=SimpleNamespace(is_active=True),
        _task=None,
        stop=lambda: None,
    )


@pytest.mark.asyncio
async def test_parallel_lane_disables_without_changing_main_configuration(monkeypatch):
    coordinator = Coordinator()
    first = _submitter(1)
    second = _submitter(2)
    coordinator.submitters = [first, second]
    coordinator.submitter_configs = [
        SimpleNamespace(submitter_id=1, model_id="main-quality-model"),
        SimpleNamespace(submitter_id=2, model_id="other-model"),
    ]
    events = []

    async def broadcast(event, payload):
        events.append((event, payload))

    async def persist(*args, **kwargs):
        return None

    coordinator.websocket_broadcaster = broadcast
    monkeypatch.setattr(coordinator, "_add_persisted_event", persist)

    await coordinator._handle_submitter_provider_repair_required(first, _error())

    assert coordinator.fatal_error_type is None
    assert coordinator.disabled_submitter_ids == {1}
    assert coordinator.submitter_configs[0].model_id == "main-quality-model"
    assert second.is_running is True
    assert events[0][0] == "provider_role_disabled"
    assert events[0][1]["workflow_continues"] is True
    assert events[0][1]["role_label"] == "Submitter 1"
    assert events[0][1]["host_routing"] == "direct"
    assert "Submitter 1 · grok-4.3 via xAI Grok]" in events[0][1]["route_identity"]
    assert events[0][1]["message"].startswith(
        f"{events[0][1]['route_identity']} Submitter 1 was disabled"
    )


@pytest.mark.asyncio
async def test_repair_stop_names_failing_route_in_payload_and_persisted_message(monkeypatch):
    _configure_openrouter_role(monkeypatch, "aggregator_validator", host="TestHost")
    coordinator = Coordinator()
    coordinator.persist_event_log = True
    events = []
    persisted = []

    async def broadcast(event, payload):
        events.append((event, payload))

    async def persist(event_type, message, metadata):
        persisted.append((event_type, message, metadata))

    coordinator.websocket_broadcaster = broadcast
    monkeypatch.setattr(coordinator, "_add_persisted_event", persist)
    monkeypatch.setattr(coordinator, "_notify_top_level_terminal", lambda: None)
    monkeypatch.setattr(coordinator_module.queue_manager, "clear", mock.AsyncMock())
    error = ProviderRepairRequiredError(
        provider="openrouter",
        provider_label="OpenRouter",
        role_id="aggregator_validator",
        model="vendor/test-free:free",
        reason="spending_limit_reached",
        message="spending limit",
        terminal_guidance="repair provider",
    )

    await coordinator._handle_provider_repair_required(
        error,
        role_id="aggregator_single_model",
    )

    identity = "[Aggregator · Validator · vendor/test-free:free via OpenRouter · host TestHost (pinned)]"
    event, payload = events[0]
    assert event == "provider_repair_required"
    assert payload["workflow_mode"] == "aggregator"
    assert payload["role_id"] == "aggregator_single_model"
    assert payload["model_role_id"] == "aggregator_validator"
    assert payload["role_label"] == "Validator"
    assert payload["host_routing"] == "pinned"
    assert payload["configured_host_provider"] == "TestHost"
    assert payload["route_identity"] == identity
    assert payload["message"].startswith(f"{identity} Research stopped because OpenRouter")
    assert persisted[0][1] == payload["message"]
    assert coordinator.fatal_error_payload["route_identity"] == identity
    assert str(error) == "spending limit"


async def _run_single_model_free_model_wait(monkeypatch, tmp_path, error, *, looping=False, auto_selector=False):
    _configure_openrouter_role(monkeypatch, "aggregator_submitter_3", host="TestHost")
    monkeypatch.setattr(system_config, "data_dir", str(tmp_path))
    monkeypatch.setattr(free_model_manager, "looping_enabled", looping)
    monkeypatch.setattr(free_model_manager, "auto_selector_enabled", auto_selector)
    manager_events = []

    async def manager_broadcast(event, payload):
        manager_events.append((event, payload))

    monkeypatch.setattr(api_client_manager, "_broadcast_callback", manager_broadcast)
    monkeypatch.setattr(
        api_client_manager,
        "_sleep_with_optional_stop",
        mock.AsyncMock(return_value=True),
    )
    monkeypatch.setattr(coordinator_module.queue_manager, "prepend_batch", mock.AsyncMock())
    coordinator = Coordinator()
    coordinator.persist_event_log = True
    coordinator.is_running = True
    coordinator_events = []

    async def coordinator_broadcast(event, payload):
        coordinator_events.append((event, payload))

    async def generate_submission():
        coordinator.is_running = False
        raise error

    coordinator.websocket_broadcaster = coordinator_broadcast
    coordinator.submitters = [SimpleNamespace(
        submitter_id=3,
        role_id="aggregator_submitter_3",
        _generate_submission=generate_submission,
    )]
    coordinator.validator = SimpleNamespace(role_id="aggregator_validator")

    await coordinator._single_model_workflow()

    assert coordinator_events[0][0] == "free_models_exhausted"
    retry_event, retry_payload = manager_events[0]
    assert retry_event == "provider_retry_cooldown"
    return coordinator_events[0][1], retry_payload


def _assert_free_model_wait_identity(exhausted, retry_payload, identity):
    assert exhausted["role_id"] == "aggregator_single_model"
    assert exhausted["model_role_id"] == "aggregator_submitter_3"
    assert exhausted["route_identity"] == identity
    assert exhausted["message"] == "All free models exhausted, waiting to retry"
    assert retry_payload["role_id"] == "aggregator_single_model"
    assert retry_payload["route_identity"] == identity
    assert retry_payload["message"].startswith(f"{identity} OpenRouter free models API call will retry")
    assert "free rotation" not in retry_payload["correlation_id"]


@pytest.mark.asyncio
async def test_single_model_free_model_wait_names_in_flight_role(monkeypatch, tmp_path):
    # Looping and the auto-selector are off, so the error came from the role's
    # primary route: its configured model on its pinned host.
    error = FreeModelExhaustedError("All free model options exhausted.")
    exhausted, retry_payload = await _run_single_model_free_model_wait(monkeypatch, tmp_path, error)

    identity = "[Aggregator · Submitter 3 · vendor/test-free:free via OpenRouter · host TestHost (pinned)]"
    _assert_free_model_wait_identity(exhausted, retry_payload, identity)
    assert exhausted["model"] == "vendor/test-free:free"
    assert exhausted.get("route_kind", "primary") == "primary"
    assert exhausted["host_routing"] == "pinned"
    assert exhausted["configured_host_provider"] == "TestHost"
    assert "free rotation" not in exhausted["route_identity"]
    assert str(error) == "All free model options exhausted."


@pytest.mark.asyncio
async def test_free_model_wait_after_rotation_names_auto_host(monkeypatch, tmp_path):
    error = FreeModelExhaustedError("All free model options exhausted.")
    exhausted, retry_payload = await _run_single_model_free_model_wait(
        monkeypatch, tmp_path, error, looping=True,
    )

    identity = (
        "[Aggregator · Submitter 3 (free rotation) · vendor/test-free:free via OpenRouter · "
        "host auto (upstream not reported)]"
    )
    _assert_free_model_wait_identity(exhausted, retry_payload, identity)
    assert exhausted["route_kind"] == "free_rotation"
    assert exhausted["host_routing"] == "auto"


@pytest.mark.asyncio
async def test_account_credit_free_model_wait_names_primary_route_with_rotation_on(monkeypatch, tmp_path):
    # The account-credit short-circuit raises before any rotation leg runs.
    message = "Account free credits exhausted and no LM Studio fallback for role 'aggregator_submitter_3'."
    error = FreeModelExhaustedError(message)
    exhausted, retry_payload = await _run_single_model_free_model_wait(
        monkeypatch, tmp_path, error, looping=True, auto_selector=True,
    )

    identity = "[Aggregator · Submitter 3 · vendor/test-free:free via OpenRouter · host TestHost (pinned)]"
    _assert_free_model_wait_identity(exhausted, retry_payload, identity)
    assert exhausted["host_routing"] == "pinned"
    assert str(error) == message


@pytest.mark.asyncio
async def test_free_model_wait_uses_route_carried_by_error(monkeypatch, tmp_path):
    error = FreeModelExhaustedError("All free model options exhausted.")
    error.role_id = "aggregator_submitter_3"
    error.model = "openrouter/free"
    error.provider = "openrouter"
    error.route_kind = "auto_selector"
    error.configured_host_provider = ""
    exhausted, retry_payload = await _run_single_model_free_model_wait(monkeypatch, tmp_path, error)

    identity = (
        "[Aggregator · Submitter 3 (auto-selector) · openrouter/free via OpenRouter · "
        "host auto (upstream not reported)]"
    )
    _assert_free_model_wait_identity(exhausted, retry_payload, identity)
    assert exhausted["route_kind"] == "auto_selector"
    assert exhausted["model"] == "openrouter/free"
    assert exhausted["host_routing"] == "auto"
    assert str(error) == "All free model options exhausted."
