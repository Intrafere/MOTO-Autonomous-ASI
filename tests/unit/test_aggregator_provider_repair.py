from types import SimpleNamespace

import pytest

from backend.aggregator.core.coordinator import Coordinator
from backend.shared.provider_errors import ProviderRepairRequiredError


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

