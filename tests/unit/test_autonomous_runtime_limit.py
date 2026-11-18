"""The optional user-set maximum runtime for one Autonomous Research Start."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from backend.api.routes import autonomous as autonomous_route
from backend.autonomous.core.autonomous_coordinator import (
    MAX_RUNTIME_STOP_REASON,
    AutonomousCoordinator,
    _format_runtime_limit,
)
from backend.shared import cli_control
from backend.shared.config import system_config
from backend.shared.models import (
    AutonomousResearchStartRequest,
    AutonomousResearchStatusResponse,
    SubmitterConfig,
)


def _start_request(**overrides) -> AutonomousResearchStartRequest:
    fields = {
        "user_research_prompt": "Why?",
        "submitter_configs": [SubmitterConfig(
            submitter_id=1, provider="openrouter", model_id="a/b", context_window=100000, max_output_tokens=8000,
        )],
        "validator_provider": "openrouter",
        "validator_model": "a/b",
        "allow_mathematical_proofs": False,
    }
    fields.update(overrides)
    return AutonomousResearchStartRequest(**fields)


def _running_coordinator() -> tuple[AutonomousCoordinator, list[str]]:
    coordinator = AutonomousCoordinator()
    coordinator._running = True
    stops: list[str] = []

    async def stop_like_the_route() -> None:
        stops.append("stopped")
        coordinator._running = False

    coordinator.runtime_limit_stop_callback = stop_like_the_route
    return coordinator, stops


def test_start_request_has_no_limit_by_default_and_rejects_tiny_limits():
    assert _start_request().max_runtime_seconds is None
    assert _start_request(max_runtime_seconds=3600).max_runtime_seconds == 3600
    with pytest.raises(ValidationError):
        _start_request(max_runtime_seconds=5)


def test_runtime_limit_is_described_in_plain_units():
    assert _format_runtime_limit(5400) == "1h 30m"
    assert _format_runtime_limit(86400) == "1d"
    assert _format_runtime_limit(61) == "1m 1s"


@pytest.mark.asyncio
async def test_reaching_the_limit_stops_the_run_with_a_resumable_terminal_reason():
    coordinator, stops = _running_coordinator()
    coordinator.arm_runtime_limit(0.05)

    limit = coordinator.get_runtime_limit()
    assert limit is not None and limit["deadline_at"] > limit["started_at"]
    AutonomousResearchStatusResponse(is_running=True, run_id="r", current_tier="tier1", runtime_limit=limit)

    await asyncio.sleep(0.3)

    assert stops == ["stopped"]
    event = coordinator._terminal_event
    assert event is not None
    assert event.reason == MAX_RUNTIME_STOP_REASON
    assert event.recoverable is True and event.fatal is False
    assert event.notification_kind is None  # a chosen limit is not a model or provider failure
    assert "maximum runtime" in event.message
    assert coordinator.get_runtime_limit() is None


@pytest.mark.asyncio
async def test_a_run_that_already_ended_is_left_alone_at_the_deadline():
    coordinator, stops = _running_coordinator()
    coordinator.arm_runtime_limit(0.05)
    coordinator._running = False  # finished on its own before the limit

    await asyncio.sleep(0.3)

    assert stops == []
    assert coordinator._terminal_event is None


@pytest.mark.asyncio
async def test_a_later_start_without_a_limit_clears_the_earlier_limit():
    coordinator, stops = _running_coordinator()
    coordinator.arm_runtime_limit(0.05)
    coordinator.arm_runtime_limit(None)

    assert coordinator.get_runtime_limit() is None
    await asyncio.sleep(0.3)

    assert stops == []
    assert coordinator._terminal_event is None


@pytest.mark.asyncio
async def test_stop_cancels_the_countdown():
    coordinator, stops = _running_coordinator()
    coordinator.arm_runtime_limit(0.05)
    coordinator._cancel_runtime_limit()

    await asyncio.sleep(0.3)

    assert stops == []


class _FakeCoordinator:
    def __init__(self) -> None:
        self.is_active = False
        self.armed: list = []

    async def initialize(self, **_kwargs) -> None:
        return None

    def set_direct_answer_enabled(self, enabled: bool) -> None:
        self.direct_answer_enabled = enabled

    def start_in_background(self) -> bool:
        self.is_active = True
        return True

    def arm_runtime_limit(self, seconds) -> None:
        self.armed.append(seconds)

    async def stop(self) -> None:
        self.is_active = False


@pytest.mark.asyncio
async def test_start_route_arms_the_limit_but_never_saves_it_for_reuse(monkeypatch, tmp_path):
    fake = _FakeCoordinator()
    monkeypatch.setattr(autonomous_route, "autonomous_coordinator", fake)
    monkeypatch.setattr(autonomous_route, "_get_start_conflict", lambda: None)
    monkeypatch.setattr(autonomous_route, "require_embedding_provider_ready", AsyncMock())
    monkeypatch.setattr(system_config, "data_dir", tmp_path)
    monkeypatch.setattr(system_config, "generic_mode", False)
    template = tmp_path / cli_control.START_TEMPLATE_DIR_NAME / "autonomous.json"
    try:
        result = await autonomous_route.start_autonomous_research(_start_request(max_runtime_seconds=7200))
        assert result["max_runtime_seconds"] == 7200
        assert fake.armed == [7200]
        saved = json.loads(template.read_text(encoding="utf-8"))["request"]
        assert saved["user_research_prompt"] == "Why?"
        assert saved["max_runtime_seconds"] is None

        # A later Start with no limit must not inherit the earlier one.
        fake.is_active = False
        autonomous_route._release_autonomous_workflow_lease()
        await autonomous_route.start_autonomous_research(_start_request())
        assert fake.armed == [7200, None]
    finally:
        autonomous_route._release_autonomous_workflow_lease()
