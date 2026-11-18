"""A child aggregator must not outlive the Autonomous phase that owns it.

The brainstorm monitor loop does parent work of its own (for example the
completion review model call) while the child aggregator runs. An exception from
that work used to leave the child running after the shared-training memory was
pointed back at the manual database.
"""
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.aggregator.core.coordinator import Coordinator
from backend.autonomous.core.autonomous_coordinator import AutonomousCoordinator

module = import_module("backend.autonomous.core.autonomous_coordinator")


def _prepare_brainstorm_loop(monkeypatch, tmp_path, child):
    coordinator = AutonomousCoordinator()
    coordinator._running = True
    coordinator._current_topic_id = "topic-1"
    coordinator._get_reference_paper_paths = AsyncMock(return_value=[])
    coordinator._get_reference_brainstorm_contexts = AsyncMock(return_value=[])
    coordinator._get_effective_brainstorm_prompt = AsyncMock(return_value="prompt")
    coordinator._save_workflow_state = AsyncMock()
    coordinator._broadcast = AsyncMock()
    coordinator._recover_brainstorm_acceptance_count = AsyncMock()
    db = tmp_path / "brainstorm.txt"
    db.touch()
    monkeypatch.setattr(module, "AggregatorCoordinator", lambda: child)
    monkeypatch.setattr(module.api_client_manager, "set_model_tracking_callback", lambda _: None)
    monkeypatch.setattr(module.api_client_manager, "set_autonomous_phase", lambda _: None)
    monkeypatch.setattr(module.shared_training_memory, "insights", [])
    monkeypatch.setattr(module.shared_training_memory, "reload_insights_from_current_path", AsyncMock())
    # The loop repoints the shared-training singleton at this test's temporary
    # brainstorm file; restore its path state so later tests never inherit it.
    for attribute in ("_file_path", "_uses_default_path", "_root_generation"):
        monkeypatch.setattr(
            module.shared_training_memory,
            attribute,
            getattr(module.shared_training_memory, attribute),
        )
    monkeypatch.setattr(module.system_config, "shared_training_file", module.system_config.shared_training_file)
    monkeypatch.setattr(module.brainstorm_memory, "_get_database_path", lambda _: db)
    monkeypatch.setattr(
        module.brainstorm_memory,
        "get_metadata",
        AsyncMock(return_value=SimpleNamespace(topic_prompt="prompt")),
    )
    return coordinator, db


@pytest.mark.asyncio
async def test_brainstorm_loop_stops_running_child_when_parent_work_raises(monkeypatch, tmp_path):
    original_path = Path(module.shared_training_memory.file_path)
    path_seen_by_stop: list[Path] = []
    status = SimpleNamespace(total_acceptances=0, total_rejections=0, removals_executed=0, is_running=True)

    async def stop():
        path_seen_by_stop.append(Path(module.shared_training_memory.file_path))
        child.has_live_work = False

    child = SimpleNamespace(
        initialize=AsyncMock(),
        start=AsyncMock(),
        stop=AsyncMock(side_effect=stop),
        get_status=AsyncMock(return_value=status),
        has_live_work=True,
    )
    coordinator, db = _prepare_brainstorm_loop(monkeypatch, tmp_path, child)
    coordinator._current_brainstorm_available_for_aggregation = AsyncMock(
        side_effect=RuntimeError("provider failure during parent work")
    )

    with pytest.raises(RuntimeError, match="provider failure during parent work"):
        await coordinator._brainstorm_aggregation_loop()

    child.start.assert_awaited_once()
    child.stop.assert_awaited_once()
    # The child is stopped while the memory still points at its own brainstorm
    # database, so a late acceptance can never be saved into another database.
    assert path_seen_by_stop == [db]
    assert Path(module.shared_training_memory.file_path) == original_path


@pytest.mark.asyncio
async def test_phase_exit_leaves_an_already_stopped_child_alone():
    coordinator = AutonomousCoordinator()
    stopped_child = SimpleNamespace(stop=AsyncMock(), has_live_work=False)

    await coordinator._stop_child_aggregator_on_phase_exit(stopped_child, "brainstorm aggregation")
    await coordinator._stop_child_aggregator_on_phase_exit(None, "brainstorm aggregation")

    stopped_child.stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_phase_exit_restores_state_even_if_child_stop_fails():
    coordinator = AutonomousCoordinator()
    child = SimpleNamespace(stop=AsyncMock(side_effect=RuntimeError("stop failed")), has_live_work=True)

    await coordinator._stop_child_aggregator_on_phase_exit(child, "topic exploration")

    child.stop.assert_awaited_once()


def test_aggregator_reports_live_work_only_while_running_or_tasks_alive():
    aggregator = Coordinator()
    assert not aggregator.has_live_work

    aggregator.is_running = True
    assert aggregator.has_live_work

    aggregator.is_running = False
    aggregator._validator_task = SimpleNamespace(done=lambda: False)
    assert aggregator.has_live_work

    aggregator._validator_task = SimpleNamespace(done=lambda: True)
    assert not aggregator.has_live_work
