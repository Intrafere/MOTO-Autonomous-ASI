"""Real coordinator handoff regression tests with isolated/faked dependencies."""
import asyncio
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.autonomous.core.autonomous_coordinator import AutonomousCoordinator

module = import_module("backend.autonomous.core.autonomous_coordinator")


@pytest.mark.asyncio
async def test_force_claims_ownership_before_stop_and_proofs_wait_for_drain():
    coordinator = AutonomousCoordinator()
    coordinator._running = True
    coordinator._state.current_tier = "tier1_aggregation"
    coordinator._current_topic_id = "topic-1"
    entered, release = asyncio.Event(), asyncio.Event()

    async def stop():
        assert coordinator._manual_paper_writing_triggered
        entered.set()
        await release.wait()

    child = SimpleNamespace(stop=AsyncMock(side_effect=stop))
    coordinator._brainstorm_aggregator = child
    coordinator._save_workflow_state = AsyncMock()
    coordinator._broadcast = AsyncMock()
    coordinator._recover_brainstorm_acceptance_count = AsyncMock()
    coordinator._run_proof_verification = AsyncMock(return_value="stopped")
    with patch.object(module.brainstorm_memory, "mark_complete", AsyncMock()) as mark, \
         patch.object(module.research_metadata, "mark_brainstorm_complete", AsyncMock()), \
         patch.object(module.brainstorm_memory, "get_metadata", AsyncMock(return_value=None)), \
         patch.object(module.brainstorm_memory, "get_database_content", AsyncMock(return_value="source")):
        request = asyncio.create_task(coordinator.force_paper_writing())
        await entered.wait()
        duplicate = asyncio.create_task(coordinator.force_paper_writing())
        proof = asyncio.create_task(coordinator._run_brainstorm_completion_proofs())
        await asyncio.sleep(0)
        mark.assert_not_awaited()
        coordinator._run_proof_verification.assert_not_awaited()
        assert not proof.done()
        release.set()
        assert await request
        assert await duplicate
        assert await proof == "stopped"
        child.stop.assert_awaited_once()
        mark.assert_awaited_once_with("topic-1")
        coordinator._run_proof_verification.assert_awaited_once()
        assert not coordinator._manual_paper_writing_triggered


@pytest.mark.asyncio
async def test_force_waits_for_child_start_before_stopping():
    coordinator = AutonomousCoordinator()
    coordinator._running = True
    coordinator._state.current_tier = "tier1_aggregation"
    coordinator._current_topic_id = "topic-1"
    child = SimpleNamespace(stop=AsyncMock())
    coordinator._brainstorm_aggregator = child
    coordinator._save_workflow_state = AsyncMock()
    coordinator._broadcast = AsyncMock()
    with patch.object(module.brainstorm_memory, "mark_complete", AsyncMock()), \
         patch.object(module.research_metadata, "mark_brainstorm_complete", AsyncMock()):
        async with coordinator._brainstorm_start_stop_lock:
            request = asyncio.create_task(coordinator.force_paper_writing())
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert coordinator._manual_paper_writing_triggered
            child.stop.assert_not_awaited()
        assert await request
        child.stop.assert_awaited_once()
        coordinator._save_workflow_state.assert_awaited_once_with(
            tier="tier2_paper_writing", phase="brainstorm_proof_verification"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("force_during_init", [False, True])
async def test_real_monitor_honors_force_without_restarting_child(monkeypatch, tmp_path, force_during_init):
    coordinator = AutonomousCoordinator()
    coordinator._running = True
    coordinator._current_topic_id = "topic-1"
    coordinator._get_reference_paper_paths = AsyncMock(return_value=[])
    coordinator._get_reference_brainstorm_contexts = AsyncMock(return_value=[])
    coordinator._get_effective_brainstorm_prompt = AsyncMock(return_value="prompt")
    coordinator._current_brainstorm_available_for_aggregation = AsyncMock(return_value=True)
    coordinator._save_workflow_state = AsyncMock()
    coordinator._broadcast = AsyncMock()
    coordinator._recover_brainstorm_acceptance_count = AsyncMock()
    coordinator._run_proof_verification = AsyncMock(return_value="stopped")
    db = tmp_path / "brainstorm.txt"
    db.touch()
    status = SimpleNamespace(total_acceptances=0, total_rejections=0,
                             removals_executed=0, is_running=True)
    calls = 0

    async def initialize(**kwargs):
        if force_during_init:
            assert await coordinator.force_paper_writing()

    async def get_status():
        nonlocal calls
        calls += 1
        if calls == 2:
            assert await coordinator.force_paper_writing()
            status.is_running = False
        return status

    child = SimpleNamespace(initialize=AsyncMock(side_effect=initialize),
                            start=AsyncMock(), stop=AsyncMock(),
                            get_status=AsyncMock(side_effect=get_status))
    monkeypatch.setattr(module, "AggregatorCoordinator", lambda: child)
    monkeypatch.setattr(module.api_client_manager, "set_model_tracking_callback", lambda _: None)
    monkeypatch.setattr(module.api_client_manager, "set_autonomous_phase", lambda _: None)
    monkeypatch.setattr(module.shared_training_memory, "insights", [])
    monkeypatch.setattr(module.shared_training_memory, "reload_insights_from_current_path", AsyncMock())
    monkeypatch.setattr(module.brainstorm_memory, "_get_database_path", lambda _: db)
    monkeypatch.setattr(module.brainstorm_memory, "get_metadata", AsyncMock(return_value=SimpleNamespace(topic_prompt="prompt")))
    monkeypatch.setattr(module.brainstorm_memory, "get_database_content", AsyncMock(return_value="source"))
    monkeypatch.setattr(module.brainstorm_memory, "mark_complete", AsyncMock())
    monkeypatch.setattr(module.research_metadata, "mark_brainstorm_complete", AsyncMock())
    assert not await coordinator._brainstorm_aggregation_loop()
    child.stop.assert_awaited_once()
    assert child.start.await_count == (0 if force_during_init else 1)
    coordinator._run_proof_verification.assert_awaited_once()


@pytest.mark.asyncio
async def test_force_does_not_publish_handoff_after_parent_stop():
    coordinator = AutonomousCoordinator()
    coordinator._running = True
    coordinator._state.current_tier = "tier1_aggregation"
    coordinator._current_topic_id = "topic-1"

    async def stop():
        coordinator._stop_event.set()
        coordinator._running = False

    coordinator._brainstorm_aggregator = SimpleNamespace(stop=AsyncMock(side_effect=stop))
    coordinator._save_workflow_state = AsyncMock()
    with patch.object(module.brainstorm_memory, "mark_complete", AsyncMock()) as mark:
        assert not await coordinator.force_paper_writing()
        mark.assert_not_awaited()
        coordinator._save_workflow_state.assert_not_awaited()
