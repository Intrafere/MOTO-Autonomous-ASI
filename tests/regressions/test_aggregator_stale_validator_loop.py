"""A validator loop must not survive a fatal stop into the next run.

Fatal stops raised from a submitter lane used to leave the validator task alive.
Its in-flight call could return after a later Start, pass the running check, and
keep validating beside the new validator loop.
"""
import asyncio
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.aggregator.core.coordinator import Coordinator

module = import_module("backend.aggregator.core.coordinator")


async def _never_finishes() -> None:
    await asyncio.Event().wait()


def _quiet_coordinator(monkeypatch) -> Coordinator:
    coordinator = Coordinator()
    coordinator.submitters = []
    coordinator._broadcast = AsyncMock()
    coordinator._add_persisted_event = AsyncMock()
    monkeypatch.setattr(module.queue_manager, "clear", AsyncMock())
    return coordinator


@pytest.mark.asyncio
async def test_fatal_stop_from_another_task_cancels_the_validator_loop(monkeypatch):
    coordinator = _quiet_coordinator(monkeypatch)
    coordinator.is_running = True
    validator_task = asyncio.create_task(_never_finishes())
    coordinator._validator_task = validator_task
    await asyncio.sleep(0)
    error = SimpleNamespace(to_repair_event=lambda **_: {"message": "Lean 4 needs repair"})

    await coordinator._handle_lean_workspace_repair_required(error)

    assert validator_task.cancelled()
    assert not coordinator.is_running
    assert not coordinator.has_live_work


@pytest.mark.asyncio
async def test_validator_loop_does_not_cancel_itself(monkeypatch):
    coordinator = _quiet_coordinator(monkeypatch)

    async def validator_loop() -> str:
        await coordinator._cancel_validator_after_fatal_stop()
        return "finished its own fatal handling"

    coordinator._validator_task = asyncio.create_task(validator_loop())

    assert await coordinator._validator_task == "finished its own fatal handling"


@pytest.mark.asyncio
async def test_start_drains_a_loop_left_over_from_an_earlier_run(monkeypatch):
    coordinator = _quiet_coordinator(monkeypatch)
    coordinator.single_model_mode = False
    coordinator.refresh_workflow_predictions = AsyncMock()
    coordinator._validator_loop = _never_finishes
    stale_task = asyncio.create_task(_never_finishes())
    coordinator._validator_task = stale_task
    await asyncio.sleep(0)

    await coordinator.start()
    try:
        assert stale_task.cancelled()
        assert coordinator._validator_task is not stale_task
        assert not coordinator._validator_task.done()
    finally:
        await coordinator.stop()
