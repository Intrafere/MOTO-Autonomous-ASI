"""The one-call direct answer that ends an Autonomous Research run started for `moto think`."""
from __future__ import annotations

import asyncio
import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from backend.api.routes import autonomous as autonomous_route
from backend.autonomous.core.autonomous_coordinator import (
    DIRECT_ANSWER_COMPLETE_STOP_REASON,
    DIRECT_ANSWER_FAILED_STOP_REASON,
    MAX_RUNTIME_STOP_REASON,
    AutonomousCoordinator,
)
from backend.autonomous.memory import direct_answer_memory as memory_module
from backend.autonomous.memory.direct_answer_memory import DirectAnswerMemory
from backend.autonomous.prompts.direct_answer_prompts import (
    EXPLORATION_ENDED_EARLY_NOTE,
    FINAL_REMINDER,
    NO_EXPLORATION_RESULTS,
    build_direct_answer_messages,
)
from backend.shared import cli_control
from backend.shared.config import system_config
from backend.shared.models import AutonomousResearchStartRequest, SubmitterConfig

# The package re-exports the coordinator instance under the module's name.
coordinator_module = importlib.import_module("backend.autonomous.core.autonomous_coordinator")

QUESTION = "Which of the two designs should we build first, and why?"


def _submitter() -> SubmitterConfig:
    return SubmitterConfig(
        submitter_id=1, provider="openrouter", model_id="a/b", context_window=100000, max_output_tokens=8000,
    )


def _start_request(**overrides) -> AutonomousResearchStartRequest:
    fields = {
        "user_research_prompt": QUESTION,
        "submitter_configs": [_submitter()],
        "validator_provider": "openrouter",
        "validator_model": "a/b",
        "allow_mathematical_proofs": False,
        "allow_research_papers": False,
    }
    fields.update(overrides)
    return AutonomousResearchStartRequest(**fields)


def _reply(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


@pytest.fixture
def answers(monkeypatch, tmp_path):
    """A direct-answer store on a temp path with no active session."""
    store = DirectAnswerMemory()
    monkeypatch.setattr(store, "_session_path", lambda: None)
    monkeypatch.setattr(DirectAnswerMemory, "_latest_path", staticmethod(lambda: tmp_path / "latest.json"))
    monkeypatch.setattr(memory_module, "direct_answer_memory", store)
    return store


@pytest.fixture
def exploring(monkeypatch, answers):
    """A coordinator that explored one topic, with the model and stores faked."""
    coordinator = AutonomousCoordinator()
    coordinator._submitter_configs = [_submitter()]
    coordinator._base_user_research_prompt = QUESTION
    coordinator._allow_research_papers = False
    coordinator._allow_mathematical_proofs = False
    coordinator._current_topic_id = "t1"
    coordinator._broadcast = AsyncMock()
    calls: list[dict] = []
    replies = ["Build design A first, because it removes the main risk."]

    async def generate_completion(**kwargs):
        calls.append(kwargs)
        return _reply(replies.pop(0) if replies else "")

    monkeypatch.setattr(coordinator_module.api_client_manager, "configure_role", lambda *_a, **_k: None)
    monkeypatch.setattr(coordinator_module.api_client_manager, "generate_completion", generate_completion)
    monkeypatch.setattr(coordinator_module.api_client_manager, "set_autonomous_logger_callback", lambda *_a: None)
    monkeypatch.setattr(coordinator_module.brainstorm_memory, "get_all_brainstorms", AsyncMock(return_value=[
        SimpleNamespace(topic_id="t0", topic_prompt="An older angle", status="complete", submission_count=2),
        SimpleNamespace(topic_id="t1", topic_prompt="What is the main risk?", status="complete", submission_count=3),
        SimpleNamespace(topic_id="t2", topic_prompt="Empty topic", status="in_progress", submission_count=0),
    ]))

    async def brainstorm_context(topic_id, max_tokens, query=""):
        return f"IDEAS OF {topic_id}", False

    monkeypatch.setattr(coordinator_module.autonomous_rag_manager, "get_brainstorm_context", brainstorm_context)
    return coordinator, calls, replies


def test_prompt_keeps_instructions_first_and_the_format_reminder_last():
    messages = build_direct_answer_messages(QUESTION, "IDEA ONE", ended_early=True)
    assert [message["role"] for message in messages] == ["system", "user"]
    user = messages[1]["content"]
    assert user.startswith(f"QUESTION:\n{QUESTION}")
    assert "IDEA ONE" in user and EXPLORATION_ENDED_EARLY_NOTE in user
    assert user.endswith(FINAL_REMINDER)

    finished = build_direct_answer_messages(QUESTION, "  ", ended_early=False)[1]["content"]
    assert NO_EXPLORATION_RESULTS in finished and EXPLORATION_ENDED_EARLY_NOTE not in finished


@pytest.mark.asyncio
async def test_store_keeps_session_answers_and_a_newest_copy(monkeypatch, tmp_path):
    store = DirectAnswerMemory()
    session_file = tmp_path / "session" / "direct_answer.json"
    monkeypatch.setattr(DirectAnswerMemory, "_latest_path", staticmethod(lambda: tmp_path / "latest.json"))
    monkeypatch.setattr(store, "_session_path", lambda: session_file)
    assert await store.latest() is None

    await store.save({"answer_id": "1", "answer": "first"})
    await store.save({"answer_id": "2", "answer": "second"})
    assert (await store.latest())["answer"] == "second"
    assert [item["answer_id"] for item in json.loads(session_file.read_text(encoding="utf-8"))["answers"]] == ["1", "2"]

    # With no active session (a restart after the session was retired) the newest copy is read.
    monkeypatch.setattr(store, "_session_path", lambda: None)
    assert (await store.latest())["answer"] == "second"
    await store.clear_latest()
    assert await store.latest() is None
    assert session_file.exists()


@pytest.mark.asyncio
async def test_one_call_answers_the_question_from_the_explored_ideas(exploring, answers):
    coordinator, calls, _ = exploring

    record = await coordinator._write_direct_answer("exploration_complete")

    assert len(calls) == 1
    call = calls[0]
    assert call["role_id"] == "autonomous_direct_answer" and call["model"] == "a/b"
    assert call["task_id"].startswith("agg_sub1_direct_answer_")
    assert call["temperature"] == 0.0 and call["max_tokens"] == 8000
    user = call["messages"][1]["content"]
    assert QUESTION in user
    # The topic being explored comes first; a topic with no accepted ideas is left out.
    assert user.index("IDEAS OF t1") < user.index("IDEAS OF t0")
    assert "What is the main risk?" in user and "Empty topic" not in user
    assert EXPLORATION_ENDED_EARLY_NOTE not in user

    assert record["answer"] == "Build design A first, because it removes the main risk."
    assert record["question"] == QUESTION and record["exploration_complete"] is True
    assert record["basis"]["topics"] == ["t1", "t0"]
    assert (await answers.latest())["answer_id"] == record["answer_id"]
    events = [call.args[0] for call in coordinator._broadcast.await_args_list]
    assert events == ["direct_answer_started", "direct_answer_ready"]
    ready = coordinator._broadcast.await_args_list[1].args[1]
    assert "answer" not in ready and ready["word_count"] == record["word_count"]
    assert coordinator.direct_answer_writing is False


@pytest.mark.asyncio
async def test_an_empty_reply_is_retried_once_and_then_fails_without_saving(exploring, answers):
    coordinator, calls, replies = exploring
    replies[:] = ["", "A real answer."]
    assert (await coordinator._write_direct_answer("time_limit:2h"))["answer"] == "A real answer."
    assert len(calls) == 2
    assert EXPLORATION_ENDED_EARLY_NOTE in calls[0]["messages"][1]["content"]

    replies[:] = ["", "   "]
    with pytest.raises(RuntimeError):
        await coordinator._write_direct_answer("finish_requested")
    assert (await answers.latest())["answer"] == "A real answer."


@pytest.mark.asyncio
async def test_a_question_too_large_for_the_model_fails_clearly(exploring):
    coordinator, calls, _ = exploring
    coordinator._base_user_research_prompt = "word " * 200_000
    with pytest.raises(ValueError, match="context window"):
        await coordinator._write_direct_answer("exploration_complete")
    assert calls == []


@pytest.mark.asyncio
async def test_answer_is_due_only_after_this_run_finishes_a_topic(exploring, monkeypatch):
    coordinator, _, _ = exploring
    completed = AsyncMock(return_value={"t-old"})
    monkeypatch.setattr(coordinator, "_completed_brainstorm_ids", completed)

    assert await coordinator._direct_answer_due(None) is False  # not a direct-answer run
    coordinator.set_direct_answer_enabled(True)
    coordinator._direct_answer_baseline_topics = {"t-old"}
    assert await coordinator._direct_answer_due(None) is False  # nothing finished in this run yet

    completed.return_value = {"t-old", "t-new"}
    assert await coordinator._direct_answer_due(None) is True
    assert await coordinator._direct_answer_due({"paper_phase": "topic_exploration"}) is True
    # Saved work from the current topic (a proof check or paper) still has to finish first.
    assert await coordinator._direct_answer_due({"paper_phase": "brainstorm_proof_verification"}) is False


@pytest.mark.asyncio
async def test_answer_phase_after_a_time_limit_stops_with_a_resumable_reason(exploring):
    coordinator, _, _ = exploring
    released: list[str] = []
    coordinator.top_level_terminal_callback = lambda: released.append("released")
    coordinator._broadcast_stopped_once = AsyncMock()

    assert coordinator.start_direct_answer_in_background("time_limit:2h") is True
    assert coordinator.is_active is True
    await coordinator._main_task

    event = coordinator._terminal_event
    assert event.reason == MAX_RUNTIME_STOP_REASON and "2h" in event.message
    assert event.recoverable is True and event.notification_kind is None
    assert coordinator._running is False and released == ["released"]
    coordinator._broadcast_stopped_once.assert_awaited_once()


@pytest.mark.asyncio
async def test_answer_phase_reports_a_failed_answer_instead_of_stopping_silently(exploring):
    coordinator, _, replies = exploring
    replies[:] = ["", ""]
    coordinator._broadcast_stopped_once = AsyncMock()

    coordinator.start_direct_answer_in_background("finish_requested")
    await coordinator._main_task

    event = coordinator._terminal_event
    assert event.reason == DIRECT_ANSWER_FAILED_STOP_REASON and event.recoverable is True
    assert "could not be written" in event.message


@pytest.mark.asyncio
async def test_finishing_on_request_is_not_reported_as_a_time_limit(exploring):
    coordinator, _, _ = exploring
    coordinator._broadcast_stopped_once = AsyncMock()
    coordinator.start_direct_answer_in_background("finish_requested")
    await coordinator._main_task
    assert coordinator._terminal_event.reason == DIRECT_ANSWER_COMPLETE_STOP_REASON
    assert coordinator._terminal_event.recoverable is True


@pytest.mark.asyncio
async def test_time_limit_hands_a_direct_answer_run_to_its_answer_instead_of_just_stopping():
    coordinator = AutonomousCoordinator()
    coordinator._running = True
    coordinator.set_direct_answer_enabled(True)
    handed: list[str] = []
    stops: list[str] = []

    async def finish(reason: str) -> bool:
        handed.append(reason)
        coordinator._running = False
        return True

    async def plain_stop() -> None:
        stops.append("stopped")

    coordinator.direct_answer_finish_callback = finish
    coordinator.runtime_limit_stop_callback = plain_stop
    coordinator.arm_runtime_limit(0.05)
    await asyncio.sleep(0.3)

    assert handed == ["time_limit:0s"]
    assert stops == [] and coordinator._terminal_event is None


class _FakeCoordinator:
    def __init__(self) -> None:
        self.is_active = False
        self.direct_answer_enabled = False
        self.direct_answer_writing = False
        self.calls: list = []

    async def initialize(self, **_kwargs) -> None:
        return None

    def set_direct_answer_enabled(self, enabled: bool) -> None:
        self.direct_answer_enabled = enabled

    def start_in_background(self) -> bool:
        self.is_active = True
        return True

    def arm_runtime_limit(self, seconds) -> None:
        self.calls.append(("arm", seconds))

    async def stop(self, announce: bool = True) -> None:
        self.calls.append(("stop", announce))
        self.is_active = False

    def start_direct_answer_in_background(self, reason: str) -> bool:
        self.calls.append(("answer", reason))
        self.is_active = True
        return True


@pytest.fixture
def fake_route(monkeypatch, tmp_path):
    fake = _FakeCoordinator()
    monkeypatch.setattr(autonomous_route, "autonomous_coordinator", fake)
    monkeypatch.setattr(autonomous_route, "_get_start_conflict", lambda: None)
    monkeypatch.setattr(autonomous_route, "require_embedding_provider_ready", AsyncMock())
    monkeypatch.setattr(autonomous_route.assistant_proof_search_coordinator, "stop_all", AsyncMock())
    monkeypatch.setattr(system_config, "data_dir", tmp_path)
    monkeypatch.setattr(system_config, "generic_mode", False)
    yield fake, tmp_path / cli_control.START_TEMPLATE_DIR_NAME
    autonomous_route._release_autonomous_workflow_lease()


@pytest.mark.asyncio
async def test_a_direct_answer_start_needs_no_paper_or_proof_output(fake_route):
    fake, templates = fake_route
    with pytest.raises(HTTPException) as refused:
        await autonomous_route.start_autonomous_research(_start_request())
    assert refused.value.status_code == 400 and not fake.is_active

    result = await autonomous_route.start_autonomous_research(_start_request(direct_answer_enabled=True))
    assert result["direct_answer_enabled"] is True and fake.direct_answer_enabled is True
    # Its settings are saved apart from the ones a plain Autonomous Research start reuses.
    saved = json.loads((templates / "think.json").read_text(encoding="utf-8"))["request"]
    assert saved["direct_answer_enabled"] is True and saved["allow_research_papers"] is False
    assert not (templates / "autonomous.json").exists()


@pytest.mark.asyncio
async def test_finish_stops_the_exploration_quietly_then_starts_the_answer(fake_route):
    fake, _ = fake_route
    with pytest.raises(HTTPException) as idle:
        await autonomous_route.finish_with_direct_answer()
    assert idle.value.status_code == 400

    await autonomous_route.start_autonomous_research(_start_request(allow_research_papers=True))
    with pytest.raises(HTTPException) as plain_run:
        await autonomous_route.finish_with_direct_answer()
    assert plain_run.value.status_code == 400 and ("stop", False) not in fake.calls

    fake.is_active = False
    autonomous_route._release_autonomous_workflow_lease()
    await autonomous_route.start_autonomous_research(_start_request(direct_answer_enabled=True))
    result = await autonomous_route.finish_with_direct_answer()

    assert result["success"] is True
    assert fake.calls[-2:] == [("stop", False), ("answer", "finish_requested")]
    assert fake.is_active is True
    assert autonomous_route.workflow_start_guard.active_owner == autonomous_route.AUTONOMOUS_WORKFLOW_OWNER
