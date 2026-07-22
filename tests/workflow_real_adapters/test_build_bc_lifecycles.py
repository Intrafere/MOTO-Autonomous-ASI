from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from backend.api.routes import aggregator as aggregator_route
from backend.api.routes import autonomous as autonomous_route
from backend.api.routes import compiler as compiler_route
from backend.api.routes import leanoj as leanoj_route
from backend.aggregator.core.coordinator import Coordinator
from backend.aggregator.memory.shared_training import SharedTrainingMemory
from backend.autonomous.memory.paper_library import PaperLibrary
from backend.autonomous.core.autonomous_coordinator import AutonomousCoordinator
from backend.compiler.core.compiler_coordinator import CompilerCoordinator
from backend.leanoj.core.leanoj_coordinator import LeanOJCoordinator
from backend.leanoj.core.leanoj_context import leanoj_context_manager
from backend.shared.config import system_config
from backend.shared.models import AutonomousResearchStartRequest, SubmitterConfig
from backend.shared.workflow_start_guard import WorkflowStartGuard
from tests.workflow_harness.real_adapters import (
    LeanOJAdapter,
    ManualAggregatorAdapter,
    ManualCompilerAdapter,
    assert_single_race_winner,
    minimal_aggregator_request,
    minimal_compiler_request,
    minimal_leanoj_request,
    race_starts,
)
from tests.workflow_real_adapters.helpers import inactive_workflow_flags, patch_attributes


class _FakeSolutionPath:
    def __init__(self) -> None:
        self.stopped = 0

    async def stop(self) -> None:
        self.stopped += 1


class _FakeManualAggregatorCoordinator:
    def __init__(self) -> None:
        self.is_running = False
        self.solution_path_manager = _FakeSolutionPath()
        self.initialized: dict = {}
        self.cleared = 0

    async def initialize(self, **kwargs) -> None:
        self.initialized = kwargs

    async def start(self) -> None:
        self.is_running = True

    async def stop(self) -> None:
        self.is_running = False

    async def clear_all_submissions(self) -> None:
        self.cleared += 1

    async def get_results_formatted(self) -> str:
        return "Submission #1\nA rigorous accepted result."


def _autonomous_request() -> AutonomousResearchStartRequest:
    return AutonomousResearchStartRequest(
        user_research_prompt="Research a rigorous solution.",
        submitter_configs=[
            SubmitterConfig(
                submitter_id=1,
                model_id="test-submitter",
                context_window=4096,
                max_output_tokens=512,
            )
        ],
        allow_mathematical_proofs=False,
        allow_research_papers=True,
        validator_model="test-validator",
        validator_context_window=4096,
        validator_max_tokens=512,
        writer_model="test-writer",
        writer_context_window=4096,
        writer_max_tokens=512,
        high_param_model="test-rigor",
        high_param_context_window=4096,
        high_param_max_tokens=512,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("target", "active_mode"),
    [
        (target, active)
        for target in ("aggregator", "compiler", "autonomous", "leanoj")
        for active in ("aggregator", "compiler", "autonomous", "leanoj")
    ],
)
async def test_actual_route_start_conflict_matrix_has_no_start_side_effects(
    monkeypatch, target, active_mode
):
    aggregator = Coordinator()
    compiler = CompilerCoordinator()
    autonomous = AutonomousCoordinator()
    leanoj = LeanOJCoordinator()
    aggregator.is_running = active_mode == "aggregator"
    compiler.is_running = active_mode == "compiler"
    autonomous._state.is_running = active_mode == "autonomous"
    autonomous._main_task = SimpleNamespace(done=lambda: active_mode != "autonomous")
    leanoj._running = active_mode == "leanoj"

    route_modules = (
        aggregator_route,
        compiler_route,
        autonomous_route,
        leanoj_route,
    )
    for module in route_modules:
        patch_attributes(
            monkeypatch,
            module,
            {
                "coordinator": aggregator,
                "compiler_coordinator": compiler,
                "autonomous_coordinator": autonomous,
                "leanoj_coordinator": leanoj,
            },
        )

    forbidden = AsyncMock(side_effect=AssertionError("start side effect ran after conflict"))
    monkeypatch.setattr(aggregator, "initialize", forbidden)
    monkeypatch.setattr(compiler, "initialize", forbidden)
    monkeypatch.setattr(autonomous, "initialize", forbidden)
    monkeypatch.setattr(leanoj, "resume_or_initialize", forbidden)

    starts = {
        "aggregator": lambda: aggregator_route.start_aggregator(minimal_aggregator_request()),
        "compiler": lambda: compiler_route.start_compiler(
            minimal_compiler_request(
                allow_mathematical_proofs=False,
                allow_research_papers=True,
            )
        ),
        "autonomous": lambda: autonomous_route.start_autonomous_research(_autonomous_request()),
        "leanoj": lambda: leanoj_route.start_leanoj(minimal_leanoj_request()),
    }

    with pytest.raises(HTTPException) as exc:
        await starts[target]()

    assert exc.value.status_code == 400
    assert "running" in str(exc.value.detail).lower()
    forbidden.assert_not_awaited()


@pytest.mark.asyncio
async def test_actual_route_start_conflict_counts_pending_autonomous_ownership(monkeypatch):
    aggregator = Coordinator()
    compiler = CompilerCoordinator()
    autonomous = AutonomousCoordinator()
    leanoj = LeanOJCoordinator()
    autonomous._state.is_running = False
    autonomous._main_task = SimpleNamespace(done=lambda: False)
    for module in (aggregator_route, compiler_route, leanoj_route):
        patch_attributes(
            monkeypatch,
            module,
            {
                "coordinator": aggregator,
                "compiler_coordinator": compiler,
                "autonomous_coordinator": autonomous,
                "leanoj_coordinator": leanoj,
            },
        )
    forbidden = AsyncMock(side_effect=AssertionError("pending owner allowed start side effect"))
    monkeypatch.setattr(aggregator, "initialize", forbidden)
    monkeypatch.setattr(compiler, "initialize", forbidden)
    monkeypatch.setattr(leanoj, "resume_or_initialize", forbidden)

    calls = (
        lambda: aggregator_route.start_aggregator(minimal_aggregator_request()),
        lambda: compiler_route.start_compiler(
            minimal_compiler_request(
                allow_mathematical_proofs=False,
                allow_research_papers=True,
            )
        ),
        lambda: leanoj_route.start_leanoj(minimal_leanoj_request()),
    )
    for call in calls:
        with pytest.raises(HTTPException) as exc:
            await call()
        assert exc.value.status_code == 400
        assert "Autonomous Research" in str(exc.value.detail)
    forbidden.assert_not_awaited()


@pytest.mark.asyncio
async def test_manual_aggregator_real_route_start_stop_output_and_temp_root(monkeypatch, tmp_path):
    fake = _FakeManualAggregatorCoordinator()
    flags = inactive_workflow_flags()
    flags["coordinator"] = fake
    patch_attributes(monkeypatch, aggregator_route, flags)
    monkeypatch.setattr(system_config, "data_dir", tmp_path)
    monkeypatch.setattr(system_config, "shared_training_file", tmp_path / "rag_shared_training.txt")
    monkeypatch.setattr(aggregator_route, "save_manual_aggregator_prompt", AsyncMock())
    monkeypatch.setattr(aggregator_route, "require_embedding_provider_ready", AsyncMock())
    monkeypatch.setattr(aggregator_route, "_require_openrouter_host_provider_available", AsyncMock())
    monkeypatch.setattr(aggregator_route.api_client_manager, "configure_role", lambda *_args: None)
    monkeypatch.setattr(aggregator_route.context_allocator, "set_context_windows", lambda *_args: None)
    monkeypatch.setattr(aggregator_route.token_tracker, "reset", lambda: None)
    monkeypatch.setattr(aggregator_route.token_tracker, "start_timer", lambda: None)
    monkeypatch.setattr(aggregator_route.token_tracker, "stop_timer", lambda: None)
    monkeypatch.setattr(aggregator_route.assistant_proof_search_coordinator, "stop_all", AsyncMock())

    class _Registry:
        async def acquire(self, *_args, **_kwargs):
            return fake.solution_path_manager

    import backend.shared.solution_path as solution_path_module

    monkeypatch.setattr(solution_path_module, "solution_path_registry", _Registry())
    adapter = ManualAggregatorAdapter()
    started = await adapter.start(minimal_aggregator_request())
    saved = await adapter.save_results()
    stopped = await adapter.stop()

    assert started["status"] == "started"
    assert fake.initialized["user_prompt"] == "Build a rigorous solution."
    assert (tmp_path / saved["path"]).read_text(encoding="utf-8").startswith("Submission #1")
    assert (tmp_path / saved["path"]).resolve().is_relative_to(tmp_path.resolve())
    assert stopped["status"] == "stopped"
    assert fake.is_running is False


@pytest.mark.asyncio
async def test_manual_aggregator_clear_archives_before_destructive_clear(monkeypatch, tmp_path):
    order: list[str] = []
    fake = _FakeManualAggregatorCoordinator()
    fake.is_running = True
    flags = inactive_workflow_flags()
    flags["coordinator"] = fake
    patch_attributes(monkeypatch, aggregator_route, flags)
    monkeypatch.setattr(system_config, "data_dir", tmp_path)

    @asynccontextmanager
    async def lock():
        yield

    async def archive(*_args, **_kwargs):
        order.append("archive")
        return 2

    async def clear():
        order.append("clear")
        fake.cleared += 1

    fake.clear_all_submissions = clear
    fake.solution_path_manager = None
    monkeypatch.setattr(aggregator_route, "get_manual_proof_context_lock", lock)
    monkeypatch.setattr(aggregator_route, "_manual_proof_clear_blocker", AsyncMock(return_value=None))
    monkeypatch.setattr(aggregator_route.manual_proof_database, "archive_current_run", archive)
    monkeypatch.setattr(aggregator_route, "load_manual_aggregator_prompt", AsyncMock(return_value="prompt"))
    monkeypatch.setattr(aggregator_route, "clear_manual_aggregator_prompt", AsyncMock())
    monkeypatch.setattr(aggregator_route, "_clear_uploaded_files", AsyncMock(return_value=3))
    monkeypatch.setattr(aggregator_route.assistant_proof_search_coordinator, "stop_all", AsyncMock())
    monkeypatch.setattr(aggregator_route.assistant_proof_search_coordinator, "clear_cooldown_state", AsyncMock())

    result = await ManualAggregatorAdapter().clear()

    assert order == ["archive", "clear"]
    assert result["archived_manual_proofs"] == 2
    assert result["deleted_uploads"] == 3


@pytest.mark.asyncio
async def test_manual_aggregator_durable_results_hydrate_from_temp_root(monkeypatch, tmp_path):
    persisted = tmp_path / "rag_shared_training.txt"
    writer = SharedTrainingMemory()
    writer.file_path = persisted
    await writer.initialize()
    await writer.add_accepted_submission("Durable accepted theorem route.")

    reader = SharedTrainingMemory()
    reader.file_path = tmp_path / "stale-other-run.txt"
    coordinator = Coordinator()
    monkeypatch.setattr(system_config, "shared_training_file", persisted)
    monkeypatch.setattr(aggregator_route, "shared_training_memory", reader)
    monkeypatch.setattr(aggregator_route, "coordinator", coordinator)
    monkeypatch.setattr(
        coordinator,
        "get_results_formatted",
        lambda: reader.get_all_content_formatted(),
    )
    monkeypatch.setattr(
        aggregator_route,
        "autonomous_coordinator",
        SimpleNamespace(is_active=False),
    )

    result = await aggregator_route.get_results()

    assert "Durable accepted theorem route." in result["results"]
    assert reader.file_path.resolve() == persisted.resolve()
    assert reader.file_path.resolve().is_relative_to(tmp_path.resolve())


@pytest.mark.asyncio
async def test_manual_aggregator_clear_blocker_has_no_destructive_side_effects(monkeypatch):
    fake = _FakeManualAggregatorCoordinator()
    archive = AsyncMock()
    clear = AsyncMock()
    fake.clear_all_submissions = clear
    flags = inactive_workflow_flags()
    flags["coordinator"] = fake
    patch_attributes(monkeypatch, aggregator_route, flags)

    @asynccontextmanager
    async def lock():
        yield

    monkeypatch.setattr(aggregator_route, "get_manual_proof_context_lock", lock)
    monkeypatch.setattr(
        aggregator_route,
        "_manual_proof_clear_blocker",
        AsyncMock(return_value="proof verification is active"),
    )
    monkeypatch.setattr(aggregator_route.manual_proof_database, "archive_current_run", archive)

    with pytest.raises(HTTPException) as exc:
        await aggregator_route.clear_all_submissions()

    assert exc.value.status_code == 409
    archive.assert_not_awaited()
    clear.assert_not_awaited()


@pytest.mark.asyncio
async def test_manual_compiler_rejects_invalid_output_selection_before_start(monkeypatch):
    flags = inactive_workflow_flags()
    patch_attributes(monkeypatch, compiler_route, flags)
    request = minimal_compiler_request(
        allow_mathematical_proofs=False,
        allow_research_papers=False,
    )

    with pytest.raises(HTTPException) as exc:
        await ManualCompilerAdapter().start(request)

    assert exc.value.status_code == 400
    assert exc.value.detail == "At least one allowed output must be enabled."


@pytest.mark.asyncio
async def test_manual_compiler_real_papers_only_route_start_stop(monkeypatch):
    compiler = CompilerCoordinator()
    aggregator = Coordinator()
    autonomous = SimpleNamespace(
        is_active=False,
        get_state=lambda: SimpleNamespace(is_running=False),
    )
    leanoj = SimpleNamespace(is_active=False)
    patch_attributes(
        monkeypatch,
        compiler_route,
        {
            "compiler_coordinator": compiler,
            "coordinator": aggregator,
            "autonomous_coordinator": autonomous,
            "leanoj_coordinator": leanoj,
        },
    )
    initialized: dict = {}

    async def initialize(**kwargs):
        initialized.update(kwargs)

    async def start():
        compiler.is_running = True

    async def stop():
        compiler.is_running = False

    monkeypatch.setattr(compiler, "initialize", initialize)
    monkeypatch.setattr(compiler, "start", start)
    monkeypatch.setattr(compiler, "stop", stop)
    monkeypatch.setattr(compiler_route, "save_manual_compiler_prompt", AsyncMock())
    monkeypatch.setattr(compiler_route, "require_embedding_provider_ready", AsyncMock())
    monkeypatch.setattr(compiler_route.api_client_manager, "configure_role", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(compiler_route.token_tracker, "reset", lambda: None)
    monkeypatch.setattr(compiler_route.token_tracker, "start_timer", lambda: None)
    monkeypatch.setattr(compiler_route.token_tracker, "stop_timer", lambda: None)
    monkeypatch.setattr(compiler_route.assistant_proof_search_coordinator, "stop_all", AsyncMock())

    import backend.shared.solution_path as solution_path_module

    monkeypatch.setattr(solution_path_module.solution_path_registry, "get", lambda *_args: None)
    adapter = ManualCompilerAdapter()
    result = await adapter.start(
        minimal_compiler_request(
            allow_mathematical_proofs=False,
            allow_research_papers=True,
        )
    )
    stopped = await adapter.stop()

    assert result["status"] == "started"
    assert initialized["allow_mathematical_proofs"] is False
    assert initialized["compiler_prompt"] == "Write a rigorous paper."
    assert stopped["status"] == "stopped"
    assert compiler.is_running is False


@pytest.mark.asyncio
async def test_manual_compiler_proof_only_route_owns_background_task(monkeypatch):
    import asyncio

    compiler = CompilerCoordinator()
    aggregator = Coordinator()
    autonomous = SimpleNamespace(
        is_active=False,
        get_state=lambda: SimpleNamespace(is_running=False),
    )
    leanoj = SimpleNamespace(is_active=False)
    patch_attributes(
        monkeypatch,
        compiler_route,
        {
            "compiler_coordinator": compiler,
            "coordinator": aggregator,
            "autonomous_coordinator": autonomous,
            "leanoj_coordinator": leanoj,
        },
    )
    monkeypatch.setattr(system_config, "generic_mode", False)
    monkeypatch.setattr(system_config, "lean4_enabled", True)
    entered = asyncio.Event()
    release = asyncio.Event()
    class Stage:
        @classmethod
        async def reserve_source(cls, source_type, source_id):
            entered.set()

    async def bounded_check(_request, *, source_reserved=False):
        assert source_reserved is True
        await release.wait()

    monkeypatch.setattr(compiler_route, "ProofVerificationStage", Stage)
    monkeypatch.setattr(compiler_route, "_run_compiler_aggregator_proof_check", bounded_check)
    monkeypatch.setattr(compiler_route, "_compiler_proof_only_task", None)

    result = await compiler_route.start_compiler(
        minimal_compiler_request(
            allow_mathematical_proofs=True,
            allow_research_papers=False,
        )
    )

    assert result["status"] == "proof_check_started"
    assert entered.is_set()
    task = compiler_route._compiler_proof_only_task
    assert task is not None and not task.done()
    assert "proof verification is already running" in compiler_route._get_start_conflict().lower()
    release.set()
    await task
    assert task.done()
    assert compiler_route.workflow_start_guard.active_owner is None


@pytest.mark.asyncio
async def test_manual_compiler_clear_archives_before_paper_clear(monkeypatch, tmp_path):
    compiler = CompilerCoordinator()
    compiler.user_prompt = "Durable compiler prompt"
    compiler.is_running = True
    order: list[str] = []
    monkeypatch.setattr(system_config, "data_dir", tmp_path)
    monkeypatch.setattr(compiler_route, "compiler_coordinator", compiler)

    @asynccontextmanager
    async def lock():
        yield

    async def stop():
        order.append("stop")
        compiler.is_running = False

    async def archive(*args, **_kwargs):
        order.append("archive")
        assert args[0].resolve().is_relative_to(tmp_path.resolve())
        return 1

    async def clear_appendix():
        order.append("appendix")

    async def clear_paper():
        order.append("paper")

    monkeypatch.setattr(compiler_route, "get_manual_proof_context_lock", lock)
    monkeypatch.setattr(compiler_route, "_manual_proof_clear_blocker", AsyncMock(return_value=None))
    monkeypatch.setattr(compiler, "stop", stop)
    monkeypatch.setattr(compiler, "clear_paper", clear_paper)
    monkeypatch.setattr(compiler_route.manual_proof_database, "archive_current_run", archive)
    monkeypatch.setattr(compiler_route, "clear_manual_shared_training_proof_appendix", clear_appendix)
    monkeypatch.setattr(compiler_route, "clear_manual_compiler_prompt", AsyncMock())
    monkeypatch.setattr(compiler_route.assistant_proof_search_coordinator, "stop_all", AsyncMock())
    monkeypatch.setattr(compiler_route.assistant_proof_search_coordinator, "clear_cooldown_state", AsyncMock())
    import backend.shared.critique_memory as critique_module

    monkeypatch.setattr(critique_module, "clear_critiques", AsyncMock())

    result = await ManualCompilerAdapter().clear()

    assert result["archived_manual_proofs"] == 1
    assert order == ["stop", "archive", "appendix", "paper"]


@pytest.mark.asyncio
async def test_leanoj_real_coordinator_skip_force_consume_and_route_clear(monkeypatch, tmp_path):
    monkeypatch.setattr(system_config, "data_dir", tmp_path)
    coordinator = LeanOJCoordinator()
    coordinator.set_broadcast_callback(AsyncMock())
    await coordinator.initialize(minimal_leanoj_request())
    coordinator._state.phase = "recursive_brainstorm"
    monkeypatch.setattr(leanoj_route, "leanoj_coordinator", coordinator)
    monkeypatch.setattr(leanoj_route.assistant_proof_search_coordinator, "stop_all", AsyncMock())
    monkeypatch.setattr(leanoj_route.assistant_proof_search_coordinator, "clear_cooldown_state", AsyncMock())
    monkeypatch.setattr(
        leanoj_context_manager,
        "remove_all_leanoj_rag_sources",
        AsyncMock(),
    )
    adapter = LeanOJAdapter()
    coordinator._running = True
    assert (await adapter.skip_brainstorm())["success"] is True
    assert await coordinator._consume_skip_brainstorm() is True
    assert coordinator.get_state().phase == "final_proof_loop"
    assert coordinator.get_state().user_forced_final_cycle is True
    assert (await adapter.force_brainstorm())["success"] is True
    assert await coordinator._consume_force_brainstorm() is True
    assert coordinator.get_state().phase == "recursive_brainstorm"
    assert coordinator.get_state().user_forced_final_cycle is False
    coordinator._running = False
    assert (await adapter.stop())["success"] is True
    state_path = tmp_path / "leanoj_sessions" / coordinator.get_state().session_id / "state.json"
    assert state_path.exists()
    assert state_path.resolve().is_relative_to(tmp_path.resolve())
    assert (await adapter.clear())["success"] is True
    assert not (tmp_path / "leanoj_sessions").exists()
    assert coordinator.get_state().session_id == ""


@pytest.mark.asyncio
async def test_leanoj_real_intermediate_master_proof_edit_stays_in_temp_root(monkeypatch, tmp_path):
    monkeypatch.setattr(system_config, "data_dir", tmp_path)
    coordinator = LeanOJCoordinator()
    coordinator.set_broadcast_callback(AsyncMock())
    await coordinator.initialize(minimal_leanoj_request())
    coordinator._state.phase = "final_proof_loop"
    adapter = LeanOJAdapter(coordinator=coordinator)
    proof = "import Mathlib\n\nexample : 1 = 1 := by\n  rfl"

    await adapter.write_intermediate_master_proof(proof, summary="bounded intermediate edit")

    proof_path = tmp_path / "leanoj_sessions" / coordinator.get_state().session_id / "master_proof.lean"
    assert proof_path.read_text(encoding="utf-8") == proof
    assert proof_path.resolve().is_relative_to(tmp_path.resolve())
    assert coordinator.get_state().master_proof_version == 1
    assert coordinator.get_state().master_proof_last_edit_summary == "bounded intermediate edit"


def test_generated_proof_appendices_are_stripped_without_truncating_body():
    brainstorm = (
        "Submission #1\nEssential source result.\n\n"
        "=== PROOFS GENERATED FROM THIS BRAINSTORM ===\nprivate generated proof"
    )
    paper = (
        "Abstract\nVisible paper.\n\n"
        "=== PROOFS ATTACHED TO THIS PAPER (Lean 4 Verified) ===\nprivate generated proof"
    )
    assert compiler_route._strip_manual_aggregator_proof_appendix(brainstorm).strip().endswith(
        "Essential source result."
    )
    stripped_paper = PaperLibrary.strip_verified_proofs_from_content(paper)
    assert "Visible paper." in stripped_paper
    assert "private generated proof" not in stripped_paper


@pytest.mark.asyncio
async def test_actual_compiler_and_aggregator_route_start_race_has_one_winner(monkeypatch):
    guard = WorkflowStartGuard()
    monkeypatch.setattr(aggregator_route, "workflow_start_guard", guard)
    monkeypatch.setattr(compiler_route, "workflow_start_guard", guard)
    aggregator = Coordinator()
    compiler = CompilerCoordinator()
    autonomous = SimpleNamespace(
        is_active=False,
        get_state=lambda: SimpleNamespace(is_running=False),
    )
    leanoj = SimpleNamespace(is_active=False)
    for module in (aggregator_route, compiler_route):
        monkeypatch.setattr(module, "coordinator", aggregator)
        monkeypatch.setattr(module, "compiler_coordinator", compiler)
        monkeypatch.setattr(module, "autonomous_coordinator", autonomous)
        monkeypatch.setattr(module, "leanoj_coordinator", leanoj)
    monkeypatch.setattr(aggregator_route, "save_manual_aggregator_prompt", AsyncMock())
    monkeypatch.setattr(compiler_route, "save_manual_compiler_prompt", AsyncMock())
    monkeypatch.setattr(aggregator_route, "require_embedding_provider_ready", AsyncMock())
    monkeypatch.setattr(compiler_route, "require_embedding_provider_ready", AsyncMock())
    monkeypatch.setattr(aggregator_route, "_require_openrouter_host_provider_available", AsyncMock())
    monkeypatch.setattr(aggregator_route.api_client_manager, "configure_role", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(compiler_route.api_client_manager, "configure_role", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(aggregator_route.context_allocator, "set_context_windows", lambda *_args: None)
    monkeypatch.setattr(aggregator_route.token_tracker, "reset", lambda: None)
    monkeypatch.setattr(aggregator_route.token_tracker, "start_timer", lambda: None)
    monkeypatch.setattr(compiler_route.token_tracker, "reset", lambda: None)
    monkeypatch.setattr(compiler_route.token_tracker, "start_timer", lambda: None)

    async def aggregator_initialize(**_kwargs):
        return None

    async def aggregator_start():
        aggregator.is_running = True

    async def compiler_initialize(**_kwargs):
        return None

    async def compiler_start():
        compiler.is_running = True

    monkeypatch.setattr(aggregator, "initialize", aggregator_initialize)
    monkeypatch.setattr(aggregator, "start", aggregator_start)
    monkeypatch.setattr(compiler, "initialize", compiler_initialize)
    monkeypatch.setattr(compiler, "start", compiler_start)

    class Registry:
        async def acquire(self, *_args, **_kwargs):
            return None

        def get(self, *_args):
            return None

    import backend.shared.solution_path as solution_path_module

    monkeypatch.setattr(solution_path_module, "solution_path_registry", Registry())

    outcomes = await race_starts(
        {
            "aggregator": lambda: aggregator_route.start_aggregator(minimal_aggregator_request()),
            "compiler": lambda: compiler_route.start_compiler(
                minimal_compiler_request(
                    allow_mathematical_proofs=False,
                    allow_research_papers=True,
                )
            ),
        }
    )
    winner = assert_single_race_winner(outcomes)
    assert winner.name in {"aggregator", "compiler"}
    assert sum((aggregator.is_running, compiler.is_running)) == 1
