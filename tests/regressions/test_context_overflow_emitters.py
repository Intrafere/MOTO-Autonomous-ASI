import pytest

from backend.aggregator.core.context_allocator import ContextAllocationError
from backend.aggregator.core.coordinator import Coordinator
from backend.autonomous.core.autonomous_coordinator import AutonomousCoordinator
from backend.autonomous.memory.research_metadata import research_metadata
from backend.compiler.core.compiler_coordinator import CompilerCoordinator
from backend.leanoj.core.leanoj_coordinator import LeanOJConfigurationError, LeanOJCoordinator
from backend.shared.models import ModelConfig
from backend.shared.provider_errors import ProviderContextLengthError, ProviderRouteIdentity


def _config(model: str, provider: str = "openrouter") -> ModelConfig:
    return ModelConfig(
        model_id=model,
        provider=provider,
        context_window=8192,
        max_output_tokens=1024,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role_id", "configured_role", "model"),
    [
        ("aggregator_submitter_1", "aggregator_submitter_1", "submitter-model"),
        ("aggregator_validator", "aggregator_validator", "validator-model"),
        ("aggregator_single_model", "aggregator_validator", "shared-model"),
    ],
)
async def test_aggregator_context_overflow_maps_roles_and_persists_metadata(
    monkeypatch, role_id, configured_role, model
):
    coordinator = Coordinator()
    coordinator.is_running = True
    emitted = []
    persisted = []

    monkeypatch.setattr(
        "backend.aggregator.core.coordinator.api_client_manager.get_role_config",
        lambda requested_role: _config(model) if requested_role == configured_role else None,
    )

    async def capture(event_type, payload):
        emitted.append((event_type, payload))

    async def capture_persisted(event_type, message, metadata):
        persisted.append((event_type, message, metadata))

    async def clear_queue():
        return None

    coordinator.websocket_broadcaster = capture
    monkeypatch.setattr(coordinator, "_add_persisted_event", capture_persisted)
    monkeypatch.setattr("backend.aggregator.core.coordinator.queue_manager.clear", clear_queue)

    await coordinator._handle_context_overflow(
        ContextAllocationError(
            "mandatory direct context overflow",
            required_tokens=9000,
            available_tokens=7000,
            context_window=8192,
            output_reserve=1024,
        ),
        role_id=role_id,
    )

    payload = emitted[0][1]
    assert emitted[0][0] == "context_overflow_error"
    assert payload["workflow_mode"] == "aggregator"
    assert payload["role_id"] == role_id
    assert payload["configured_model"] == model
    assert payload["configured_provider"] == "openrouter"
    assert payload["required_tokens"] == 9000
    assert persisted[0][2] == payload
    assert coordinator.fatal_error_payload == payload


@pytest.mark.asyncio
async def test_aggregator_child_emits_autonomous_mode_without_manual_persistence(monkeypatch):
    coordinator = Coordinator()
    coordinator.persist_event_log = False
    coordinator.is_running = True
    emitted = []

    monkeypatch.setattr(
        "backend.aggregator.core.coordinator.api_client_manager.get_role_config",
        lambda _role: _config("child-model"),
    )

    async def clear_queue():
        return None

    coordinator.websocket_broadcaster = lambda event, payload: _capture(emitted, event, payload)
    monkeypatch.setattr("backend.aggregator.core.coordinator.queue_manager.clear", clear_queue)

    await coordinator._handle_context_overflow(
        ContextAllocationError("context overflow"),
        role_id="aggregator_submitter_1",
    )

    assert emitted[0][1]["workflow_mode"] == "autonomous"


@pytest.mark.asyncio
async def test_aggregator_context_allocation_error_preserves_provider_route(monkeypatch):
    coordinator = Coordinator()
    coordinator.is_running = True
    emitted = []
    route = ProviderRouteIdentity(
        provider="openrouter",
        model="effective-model",
        role_id="aggregator_submitter_1",
        task_id="agg_sub1_001",
        host_provider="effective-host",
        route_kind="free_rotation",
        configured_provider="openrouter",
        configured_model="configured-model",
    )
    provider_error = ProviderContextLengthError("provider context limit", route=route)
    error = ContextAllocationError.from_provider_error(
        provider_error,
        "submitter provider context mismatch",
        required_tokens=9000,
        available_tokens=7000,
    )

    async def clear_queue():
        return None

    coordinator.websocket_broadcaster = lambda event, payload: _capture(emitted, event, payload)
    monkeypatch.setattr("backend.aggregator.core.coordinator.queue_manager.clear", clear_queue)
    monkeypatch.setattr(
        "backend.aggregator.core.coordinator.api_client_manager.get_role_config",
        lambda _role: _config("configured-model"),
    )

    await coordinator._handle_context_overflow(error, role_id="aggregator_submitter_1")

    payload = emitted[0][1]
    assert payload["configured_model"] == "configured-model"
    assert payload["effective_model"] == "effective-model"
    assert payload["effective_provider"] == "openrouter"
    assert payload["effective_host_provider"] == "effective-host"
    assert payload["route_kind"] == "free_rotation"


async def _capture(target, event, payload):
    target.append((event, payload))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role_id", "configured_role", "model"),
    [
        ("compiler_writer", "compiler_writer", "writer-model"),
        ("compiler_validator", "compiler_validator", "validator-model"),
        ("compiler_rigor", "compiler_high_param", "rigor-model"),
    ],
)
async def test_compiler_context_overflow_maps_roles(monkeypatch, role_id, configured_role, model):
    coordinator = CompilerCoordinator()
    emitted = []
    coordinator.websocket_broadcaster = lambda event, payload: _capture(emitted, event, payload)
    monkeypatch.setattr(
        "backend.compiler.core.compiler_coordinator.api_client_manager.get_role_config",
        lambda requested_role: _config(model) if requested_role == configured_role else None,
    )

    await coordinator._handle_context_overflow(
        RuntimeError("prompt context overflow"),
        role_id=role_id,
        mode="review",
    )

    payload = emitted[0][1]
    assert payload["workflow_mode"] == "compiler"
    assert payload["role_id"] == role_id
    assert payload["configured_model"] == model
    assert payload["mode"] == "review"
    assert coordinator.fatal_error_payload == payload


@pytest.mark.asyncio
async def test_autonomous_compiler_marks_child_payload_for_parent(monkeypatch):
    coordinator = CompilerCoordinator()
    coordinator.enable_autonomous_mode()
    emitted = []
    coordinator.websocket_broadcaster = lambda event, payload: _capture(emitted, event, payload)
    monkeypatch.setattr(
        "backend.compiler.core.compiler_coordinator.api_client_manager.get_role_config",
        lambda _role: _config("autonomous-writer"),
    )

    await coordinator._handle_context_overflow(
        RuntimeError("context overflow"),
        role_id="compiler_writer",
    )

    assert emitted[0][1]["workflow_mode"] == "autonomous"
    assert coordinator.fatal_error_payload["configured_model"] == "autonomous-writer"


@pytest.mark.asyncio
async def test_leanoj_overflow_preserves_phase_role_and_terminal_payload(monkeypatch):
    coordinator = LeanOJCoordinator()
    coordinator._state.phase = "final_solver"
    events = []
    monkeypatch.setattr(
        "backend.leanoj.core.leanoj_coordinator.api_client_manager.get_role_config",
        lambda role: _config("final-model") if role == "leanoj_final_solver" else None,
    )

    async def persist_and_broadcast(event, payload=None):
        events.append((event, payload or {}))

    monkeypatch.setattr(coordinator, "_persist_and_broadcast", persist_and_broadcast)
    await coordinator._handle_context_overflow_stop(
        RuntimeError("context overflow"),
        role_id="leanoj_final_solver",
    )

    payload = events[0][1]
    assert payload["workflow_mode"] == "leanoj"
    assert payload["phase"] == "final_solver"
    assert payload["role_id"] == "leanoj_final_solver"
    assert payload["configured_model"] == "final-model"
    terminal_payload = {
        **coordinator.get_status(),
        **coordinator._fatal_stop_payload,
        "reason": coordinator._fatal_stop_reason,
        "message": coordinator._fatal_stop_message,
    }
    assert terminal_payload["configured_model"] == "final-model"
    assert terminal_payload["phase"] == "final_solver"


@pytest.mark.asyncio
async def test_leanoj_wrapped_provider_overflow_is_idempotent_and_preserves_route(monkeypatch):
    coordinator = LeanOJCoordinator()
    coordinator._state.phase = "initial_brainstorm"
    events = []
    route = ProviderRouteIdentity(
        provider="openrouter",
        model="effective-model",
        role_id="leanoj_brainstorm_validator",
        task_id="leanoj_brainstorm_val_001",
        host_provider="effective-host",
        route_kind="boost",
        configured_provider="openrouter",
        configured_model="configured-model",
    )
    cause = ProviderContextLengthError("provider context limit", route=route)
    error = LeanOJConfigurationError(
        "Proof Solver provider context overflow",
        route=route,
    )
    error.__cause__ = cause

    async def persist_and_broadcast(event, payload=None):
        events.append((event, payload or {}))

    monkeypatch.setattr(coordinator, "_persist_and_broadcast", persist_and_broadcast)
    monkeypatch.setattr(
        "backend.leanoj.core.leanoj_coordinator.api_client_manager.get_role_config",
        lambda _role: _config("configured-model"),
    )

    assert coordinator._is_context_overflow_exception(error)
    await coordinator._handle_context_overflow_stop(error)
    await coordinator._handle_context_overflow_stop(error)

    overflow_events = [item for item in events if item[0] == "context_overflow_error"]
    assert len(overflow_events) == 1
    payload = overflow_events[0][1]
    assert payload["role_id"] == "leanoj_brainstorm_validator"
    assert payload["configured_model"] == "configured-model"
    assert payload["effective_model"] == "effective-model"
    assert payload["effective_host_provider"] == "effective-host"
    assert payload["route_kind"] == "boost"
    terminal_payload = {**coordinator.get_status(), **coordinator._fatal_stop_payload}
    assert terminal_payload["phase"] == "initial_brainstorm"
    assert terminal_payload["effective_model"] == "effective-model"


@pytest.mark.asyncio
async def test_autonomous_parent_propagates_once_and_stale_payload_can_be_reset(monkeypatch):
    coordinator = AutonomousCoordinator()
    events = []
    coordinator.set_broadcast_callback(lambda event, payload: _capture(events, event, payload))

    async def stats():
        return {}

    monkeypatch.setattr(research_metadata, "get_stats", stats)
    payload = {
        "workflow_mode": "autonomous",
        "role_id": "compiler_writer",
        "configured_model": "child-writer",
    }
    coordinator._mark_context_overflow_stop(payload)
    await coordinator._broadcast_stopped_once()
    await coordinator._broadcast_stopped_once()

    assert len(events) == 1
    assert events[0][0] == "auto_research_stopped"
    assert events[0][1]["reason"] == "context_overflow"
    assert events[0][1]["configured_model"] == "child-writer"

    coordinator._fatal_stop_reason = None
    coordinator._fatal_stop_message = ""
    coordinator._fatal_stop_payload = {}
    coordinator._stop_broadcast_sent = False
    await coordinator._broadcast_stopped_once()
    assert "configured_model" not in events[1][1]
    assert "reason" not in events[1][1]
