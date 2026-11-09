"""Background-service call scope: translation calls stay out of workflow accounting.

Auto Top-p Translate shares the API client manager with the active workflow.
Calls inside ``background_service_call`` must not inflate session token totals
or be credited as paper/Tier 3 authors, while calls outside the scope keep
today's accounting.
"""
from __future__ import annotations

import asyncio

import pytest

from backend.shared.api_client_manager import api_client_manager
from backend.shared.background_call_scope import (
    background_service_call,
    current_background_service,
)
from backend.shared.token_tracker import token_tracker


def _model_totals(model_id: str) -> dict[str, int]:
    return dict(token_tracker.get_stats()["by_model"].get(model_id, {"input": 0, "output": 0}))


def test_scope_sets_and_resets_the_service_marker() -> None:
    assert current_background_service() == ""
    with background_service_call("translation"):
        assert current_background_service() == "translation"
        with background_service_call("other"):
            assert current_background_service() == "other"
        assert current_background_service() == "translation"
    assert current_background_service() == ""


def test_scope_resets_when_the_block_raises() -> None:
    with pytest.raises(RuntimeError):
        with background_service_call("translation"):
            raise RuntimeError("boom")
    assert current_background_service() == ""


@pytest.mark.asyncio
async def test_scope_is_local_to_the_calling_task() -> None:
    seen: list[str] = []
    entered = asyncio.Event()
    release = asyncio.Event()

    async def background() -> None:
        with background_service_call("translation"):
            entered.set()
            await release.wait()

    async def workflow() -> None:
        await entered.wait()
        seen.append(current_background_service())
        release.set()

    await asyncio.gather(background(), workflow())
    assert seen == [""]


def test_token_tracker_ignores_background_calls_and_counts_others() -> None:
    model_id = "test/background-scope-token-model"
    before = token_tracker.get_stats()
    with background_service_call("translation"):
        token_tracker.track(model_id, 1000, 200)
    after_scoped = token_tracker.get_stats()
    assert after_scoped["total_input"] == before["total_input"]
    assert after_scoped["total_output"] == before["total_output"]
    assert model_id not in after_scoped["by_model"]

    token_tracker.track(model_id, 7, 3)
    after = token_tracker.get_stats()
    assert after["total_input"] == before["total_input"] + 7
    assert after["total_output"] == before["total_output"] + 3
    assert _model_totals(model_id) == {"input": 7, "output": 3}


@pytest.mark.asyncio
async def test_model_tracking_callback_skips_background_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    credited: list[str] = []

    async def callback(model_id: str) -> None:
        credited.append(model_id)

    monkeypatch.setattr(api_client_manager, "_model_tracking_callback", callback)

    with background_service_call("translation"):
        await api_client_manager._track_model_usage("vendor/translator-model")
    assert credited == []

    await api_client_manager._track_model_usage("vendor/paper-author-model")
    assert credited == ["vendor/paper-author-model"]
