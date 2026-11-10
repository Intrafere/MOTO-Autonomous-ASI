"""A forced skip-incomplete Tier 3 run is the tracked main task: Stop can end it,
and the coordinator reports itself active while it runs."""
from __future__ import annotations

import asyncio
from importlib import import_module
from unittest.mock import AsyncMock, patch

import pytest

from backend.autonomous.core.autonomous_coordinator import AutonomousCoordinator

module = import_module("backend.autonomous.core.autonomous_coordinator")


@pytest.mark.asyncio
async def test_forced_tier3_is_tracked_and_stoppable():
    coordinator = AutonomousCoordinator()
    coordinator._running = True
    coordinator._state.is_running = True
    coordinator._state.current_tier = "tier1_aggregation"
    coordinator._brainstorm_aggregator = None
    coordinator._paper_compiler = None
    coordinator._broadcast = AsyncMock()
    coordinator._broadcast_stopped_once = AsyncMock()
    coordinator._save_workflow_state = AsyncMock()
    coordinator._stop_active_child_aggregators = AsyncMock()
    tier3_started = asyncio.Event()
    tier3_cancelled = asyncio.Event()

    async def blocking_tier3() -> bool:
        tier3_started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            tier3_cancelled.set()
            raise
        return True

    coordinator._tier3_final_answer_workflow = blocking_tier3  # type: ignore[method-assign]

    with patch.object(module.paper_library, "get_all_papers", AsyncMock(return_value=[{"paper_id": "p1"}])), \
         patch.object(module.api_client_manager, "set_autonomous_logger_callback"), \
         patch.object(module.token_tracker, "stop_timer"):
        force = asyncio.create_task(coordinator.force_tier3_final_answer(mode="skip_incomplete"))
        await asyncio.wait_for(tier3_started.wait(), timeout=5)

        assert coordinator.is_active
        assert coordinator._main_task is not None and not coordinator._main_task.done()

        await coordinator.stop()
        result = await asyncio.wait_for(force, timeout=5)

    assert tier3_cancelled.is_set()
    assert result["success"] is False
    assert result["result"] == "stopped"
    assert not coordinator.is_active
    assert coordinator._ownership_handoff_active is False
