"""A failed Supercharge lane must cancel the other in-flight lanes."""
from __future__ import annotations

import asyncio

import pytest

from backend.shared.api_client_manager import APIClientManager
from backend.shared.models import ModelConfig


@pytest.mark.asyncio
async def test_failed_lane_cancels_running_siblings():
    manager = APIClientManager()
    manager.configure_role(
        "agg_sub1",
        ModelConfig(
            provider="openrouter",
            model_id="test/model",
            context_window=4096,
            max_output_tokens=512,
            supercharge_enabled=True,
        ),
    )
    started = asyncio.Event()
    cancelled: list[str] = []
    started_lanes: list[str] = []

    async def fake_once(*, task_id, **_kwargs):
        started_lanes.append(task_id)
        if task_id.endswith("_supercharge_attempt_1"):
            await asyncio.sleep(0)
            raise RuntimeError("boost route rejected for Supercharge")
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.append(task_id)
            raise
        return {"choices": [{"message": {"content": "unused"}}]}

    manager._generate_completion_once = fake_once  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="boost route rejected"):
        await manager._generate_supercharged_completion(
            task_id="agg_sub1_001",
            role_id="agg_sub1",
            model="test/model",
            messages=[{"role": "user", "content": "hello"}],
        )

    assert len(started_lanes) == 4
    assert sorted(cancelled) == sorted(
        f"agg_sub1_001_supercharge_attempt_{index}" for index in (2, 3, 4)
    )
    # No synthesis call was made after the failure.
    assert not any(task_id.endswith("_supercharge_final") for task_id in started_lanes)
