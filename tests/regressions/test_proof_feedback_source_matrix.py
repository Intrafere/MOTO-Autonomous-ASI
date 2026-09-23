"""Bounded source/feedback contracts, not provider or Lean integration coverage.

Source reads are faked; source assembly, formalization fitting, manual round driver,
run lifecycle, and source lookup are real. The stage-result boundary in the lifecycle
test is deliberately faked and does not claim end-to-end overflow propagation.
"""
import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from fastapi import Response

from backend.api.routes import proofs as routes
from backend.autonomous.agents import proof_formalization_agent as formalization
from backend.autonomous.core.proof_run_manager import ProofRunManager
from backend.autonomous.memory.proof_database import ProofDatabase
from backend.shared.models import (
    ProofAttemptFeedback, ProofCheckRequest, ProofRoleConfigSnapshot,
    ProofRuntimeConfigSnapshot, ProofStageResult,
)


SOURCES = [
    pytest.param("brainstorm", "manual_aggregator", id="manual-aggregator"),
    pytest.param("paper", "manual_compiler_current", id="live-compiler"),
    pytest.param("brainstorm", "brainstorm_current", id="autonomous-brainstorm"),
    pytest.param("paper", "paper_current", id="autonomous-paper"),
    pytest.param("paper", "older_session:paper_001", id="history-paper"),
]
MODES = ["one_round", "loop_with_pruning"]
BODY = "MANDATORY source beginning\n" + "Full source evidence.\n" * 40 + "MANDATORY source ending"
GOAL = "The exact user objective"


@pytest_asyncio.fixture
async def source_environment(monkeypatch, tmp_path):
    databases = []
    for name in ("manual", "active", "history"):
        database = ProofDatabase()
        database.set_base_dir(tmp_path / name)
        await database.initialize()
        databases.append(database)
    monkeypatch.setattr(routes, "manual_proof_database", databases[0])
    monkeypatch.setattr(routes, "proof_database", databases[1])
    monkeypatch.setattr(routes, "_history_proof_database_for_session", lambda *a, **kw: databases[2])
    monkeypatch.setattr(routes, "_manual_aggregator_prompt", AsyncMock(return_value=GOAL))
    monkeypatch.setattr(routes.research_metadata, "get_user_prompt", AsyncMock(return_value=GOAL))
    monkeypatch.setattr(routes, "compiler_coordinator", SimpleNamespace(user_prompt=GOAL, paper_title="Draft"))
    monkeypatch.setattr(routes, "_read_manual_aggregator_content", AsyncMock(return_value=BODY))
    monkeypatch.setattr(routes.paper_memory, "get_paper", AsyncMock(return_value=BODY))
    monkeypatch.setattr(routes.outline_memory, "get_outline", AsyncMock(return_value="MANDATORY outline"))
    monkeypatch.setattr(routes.brainstorm_memory, "get_metadata", AsyncMock(return_value=SimpleNamespace(topic_prompt="Topic")))
    monkeypatch.setattr(routes.brainstorm_memory, "get_database_content", AsyncMock(return_value=BODY))
    monkeypatch.setattr(routes.paper_library, "get_metadata", AsyncMock(return_value=SimpleNamespace(title="Paper", source_brainstorm_ids=[])))
    monkeypatch.setattr(routes.paper_library, "get_paper_content", AsyncMock(return_value=BODY))
    monkeypatch.setattr(routes.paper_library, "get_history_paper", AsyncMock(return_value={"content": BODY, "title": "History", "user_prompt": GOAL, "source_brainstorm_ids": []}))
    history_dir = tmp_path / "older_session" / "papers"
    history_dir.mkdir(parents=True)
    (history_dir / "paper_paper_001.txt").write_text(BODY, encoding="utf-8")
    monkeypatch.setattr(routes, "_history_session_dir", lambda _: history_dir.parent)
    monkeypatch.setattr(routes.paper_library, "get_history_papers_dir", lambda _: history_dir)
    monkeypatch.setattr(routes, "_history_brainstorm_memory_for_session", lambda _: None)
    async def prompt_context(prompt, *args, **kwargs):
        return prompt
    monkeypatch.setattr(routes, "_prompt_with_verified_proof_context", prompt_context)
    monkeypatch.setattr(routes, "_prompt_with_history_proof_context", prompt_context)
    return databases


@pytest.mark.asyncio
@pytest.mark.parametrize("source_type,source_id", SOURCES)
@pytest.mark.parametrize("run_mode", MODES)
@pytest.mark.parametrize("newest_overflow", [False, True], ids=["suffix-fits", "newest-overflows"])
async def test_source_adapter_feedback_projection(monkeypatch, source_environment, source_type, source_id, run_mode, newest_overflow):
    request = ProofCheckRequest(source_type=source_type, source_id=source_id, run_mode=run_mode)
    source = await routes._resolve_proof_source_adapter(request)
    assert BODY in source.source_content
    assert source.canonical_user_prompt == GOAL
    expected_store = source_environment[2 if ":" in source_id else 0 if source_id.startswith("manual_") else 1]
    assert source.proof_database is expected_store
    original_source = source.source_content
    history = [ProofAttemptFeedback(attempt=i + 1, theorem_id="target", error_output=f"WHOLE ERROR {i}\nsecond line {i}") for i in range(7)]
    original_history = deepcopy(history)
    built = []
    def builder(*, source_excerpt, prior_attempts, full_source_content, **kwargs):
        text = GOAL + "\n" + full_source_content + "\n" + "\n---\n".join(a.error_output for a in prior_attempts) + "\nSCHEMA"
        built.append((tuple(a.attempt for a in prior_attempts), text))
        return text
    target = builder(source_excerpt="", prior_attempts=history[-1:] if newest_overflow else history[-2:], full_source_content=original_source)
    budget = len(target) - int(newest_overflow)
    monkeypatch.setattr(formalization, "count_tokens", len)
    monkeypatch.setattr("backend.shared.prompt_feedback_budget.count_tokens", len)
    agent = formalization.ProofFormalizationAgent(model_id="fake-model", context_window=32000, max_output_tokens=2000, role_id="matrix-proof-role")
    monkeypatch.setattr(type(formalization.rag_config), "get_available_input_tokens", lambda self, *args: budget)
    fitted = agent._fit_prompt_to_context(builder, min_excerpt_length=0, source_excerpt="", prior_attempts=history, full_source_content=original_source)
    prompt, _, available, tokens, retained = fitted[:5]
    assert [a.attempt for a in retained] == ([7] if newest_overflow else [6, 7])
    assert (tokens > available) is newest_overflow
    assert original_source in prompt and GOAL in prompt and prompt.endswith("SCHEMA")
    for entry in retained:
        assert entry.error_output in prompt
    assert history == original_history
    assert source.source_content == original_source
    # Every rebuilt prompt keeps the mandatory source, even the rejected projections.
    assert all(original_source in text for _, text in built)
    projected = [ids for ids, _ in built if ids and ids[0] >= 3]
    # Ignore the budget-target construction; the real fitter starts with newest five.
    start = projected.index((3, 4, 5, 6, 7))
    assert projected[start:] == ([(3, 4, 5, 6, 7), (4, 5, 6, 7), (5, 6, 7), (6, 7), (7,)] if newest_overflow else [(3, 4, 5, 6, 7), (4, 5, 6, 7), (5, 6, 7), (6, 7)])


@pytest.mark.asyncio
@pytest.mark.parametrize("source_type,source_id", SOURCES)
@pytest.mark.parametrize("run_mode", MODES)
async def test_newest_overflow_route_lifecycle_and_mathematical_proofs_lookup(monkeypatch, source_environment, source_type, source_id, run_mode):
    """Inject a stage overflow result; verify real route/driver/manager ownership."""
    from backend.autonomous.core import proof_run_manager as manager_module
    manager = ProofRunManager()
    monkeypatch.setattr(routes, "proof_run_manager", manager)
    monkeypatch.setattr(manager_module.sleep_inhibitor, "acquire", lambda owner: None)
    monkeypatch.setattr(manager_module.sleep_inhibitor, "release", lambda owner: None)
    monkeypatch.setattr(routes, "_configure_manual_roles", lambda kind, snapshot, **kw: getattr(snapshot, kind if kind == "brainstorm" else "paper"))
    monkeypatch.setattr(routes, "_refresh_manual_assistant_memory", AsyncMock())
    monkeypatch.setattr(routes.assistant_proof_search_coordinator, "stop_all", AsyncMock())
    pruning = SimpleNamespace(restore=AsyncMock(), drain=AsyncMock(), notify_proof_registered=AsyncMock(), notify_context_pressure=AsyncMock(), route_config_fingerprint=lambda _: "test-route")
    monkeypatch.setattr(routes, "ProofPruningCoordinator", lambda **kw: pruning)
    events = []
    async def broadcast(kind, payload):
        events.append((kind, payload))
    monkeypatch.setattr(routes.websocket, "broadcast_event", broadcast)
    monkeypatch.setattr(routes, "_broadcast_manual_aggregator_proof_event", broadcast)
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    async def stage_boundary(**kwargs):
        calls.append(kwargs)
        entered.set()
        await release.wait()
        return ProofStageResult(source_type=source_type, source_id=source_id, total_candidates=1, deferred_candidate_ids=["target"], context_overflow_payload={"overflow_origin": "local_preflight", "configured_model": "fake-model"})
    monkeypatch.setattr(routes, "autonomous_coordinator", SimpleNamespace(_proof_verification_stage=SimpleNamespace(run_manual=stage_boundary)))
    role = ProofRoleConfigSnapshot(model_id="fake-model", context_window=32000, max_output_tokens=2000)
    runtime = ProofRuntimeConfigSnapshot(brainstorm=role, paper=role, validator=role)
    request = ProofCheckRequest(source_type=source_type, source_id=source_id, run_mode=run_mode)
    adapter = await routes._resolve_proof_source_adapter(request)
    queued = await manager.queue(scope=adapter.scope, source_type=source_type, source_id=source_id, proof_store_id=adapter.proof_store_id, run_id="matrix-run", run_mode=run_mode, worker=lambda control: routes._run_manual_proof_check(request, control, runtime), event_callback=broadcast)
    control = manager._runs[queued.proof_run_id]
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        response = Response()
        lookup = await routes.lookup_proof_runs_by_source(response=response, scope=adapter.scope, source_type=source_type, source_id=source_id, limit=20)
        assert "no-store" in response.headers["cache-control"]
        assert lookup.preferred_proof_run_id == queued.proof_run_id
        assert lookup.ambiguous is False
        release.set()
        await asyncio.wait_for(control.task, timeout=5)
        assert len(calls) == 1  # Continuous overflow must not start another round.
        assert calls[0]["content"] == adapter.source_content
        assert calls[0]["canonical_user_prompt"] == GOAL
        assert calls[0]["proof_run_context"]["run_mode"] == run_mode
        fatal = [(kind, data) for kind, data in events if kind == "context_overflow_error"]
        if run_mode == "loop_with_pruning":
            assert control.snapshot.status == "error"
            assert control.snapshot.terminal_reason == routes.CONTEXT_OVERFLOW_STOP_REASON
            assert len(fatal) == 1
            assert fatal[0][1]["proof_run_id"] == queued.proof_run_id
            assert fatal[0][1]["source_id"] == source_id
        else:
            assert control.snapshot.status == "completed"
            assert not fatal
        assert await ProofRunManager().get(queued.proof_run_id) is None
    finally:
        release.set()
        if not control.task.done():
            control.task.cancel()
            await asyncio.gather(control.task, return_exceptions=True)
