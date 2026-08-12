from __future__ import annotations

import pytest

from backend.autonomous.memory.research_metadata import ResearchMetadata


@pytest.mark.asyncio
async def test_terminal_event_is_atomic_and_generation_fenced(tmp_path):
    metadata = ResearchMetadata()
    metadata._workflow_state_path = tmp_path / "workflow_state.json"
    metadata._workflow_state = metadata._get_default_workflow_state()

    generation = await metadata.begin_lifecycle("run-1")
    assert generation == 1

    event = {
        "terminal_event_id": "terminal-1",
        "run_id": "run-1",
        "lifecycle_generation": generation,
        "reason": "context_overflow",
        "message": "Research stopped.",
    }
    assert await metadata.save_terminal_event(
        event,
        expected_run_id="run-1",
        expected_generation=generation,
    )

    reloaded = ResearchMetadata()
    reloaded._workflow_state_path = metadata._workflow_state_path
    assert await reloaded.get_terminal_event() == event

    next_generation = await metadata.begin_lifecycle("run-1")
    assert next_generation == 2
    assert await metadata.get_terminal_event() is None
    assert not await metadata.save_terminal_event(
        event,
        expected_run_id="run-1",
        expected_generation=generation,
    )
    assert await metadata.get_terminal_event() is None


@pytest.mark.asyncio
async def test_unrelated_workflow_save_preserves_terminal_event(tmp_path):
    metadata = ResearchMetadata()
    metadata._workflow_state_path = tmp_path / "workflow_state.json"
    metadata._workflow_state = metadata._get_default_workflow_state()
    generation = await metadata.begin_lifecycle("run-1")
    event = {
        "terminal_event_id": "terminal-1",
        "run_id": "run-1",
        "lifecycle_generation": generation,
        "reason": "context_overflow",
        "message": "Research stopped.",
    }
    await metadata.save_terminal_event(
        event,
        expected_run_id="run-1",
        expected_generation=generation,
    )

    await metadata.save_workflow_state({
        "is_running": False,
        "run_id": "run-1",
        "lifecycle_generation": generation,
        "current_tier": "tier1_aggregation",
    })

    assert await metadata.get_terminal_event() == event
