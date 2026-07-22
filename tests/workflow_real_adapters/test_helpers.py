from __future__ import annotations

import pytest

from tests.workflow_harness.real_adapters import EventCollector
from tests.workflow_real_adapters.helpers import CallRecorder, assert_event_sequence


@pytest.mark.asyncio
async def test_call_recorder_captures_sync_and_async_adapter_calls():
    recorder = CallRecorder(result={"accepted": True})

    sync_result = recorder("source", scope="manual")
    async_result = await recorder.async_call("checkpoint", resumed=True)

    assert sync_result == {"accepted": True}
    assert async_result == {"accepted": True}
    assert recorder.count == 2
    assert recorder.calls == [
        (("source",), {"scope": "manual"}),
        (("checkpoint",), {"resumed": True}),
    ]


@pytest.mark.asyncio
async def test_event_sequence_accepts_intervening_events_and_checks_payload():
    collector = EventCollector()
    await collector.broadcast("provider_paused", {"reason": "credit", "scope": "autonomous"})
    await collector.broadcast("checkpoint_saved", {"source_id": "topic-1"})
    await collector.broadcast("provider_resumed", {"reason": "reset", "scope": "autonomous"})

    assert_event_sequence(
        collector,
        "provider_paused",
        "provider_resumed",
        predicate=lambda payload: payload.get("source_id") == "topic-1",
    )


@pytest.mark.asyncio
async def test_event_sequence_rejects_reversed_provider_lifecycle():
    collector = EventCollector()
    await collector.broadcast("provider_resumed", {"reason": "reset"})
    await collector.broadcast("provider_paused", {"reason": "credit"})

    with pytest.raises(AssertionError, match="Expected ordered event"):
        assert_event_sequence(collector, "provider_paused", "provider_resumed")
