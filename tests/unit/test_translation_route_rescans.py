"""Routes and startup ask Auto Top-p Translate to rescan when displayed outputs change.

Every successful clear or prune route calls ``note_outputs_changed`` exactly
once, so cleared or pruned content stops being targeted; a refused request
calls it never. The LeanOJ restore at startup rescans after the restored run
is loaded, so that run translates as the Proof Solver's current run.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import List
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from backend.api import main as main_module
from backend.api.routes import aggregator as aggregator_route
from backend.api.routes import autonomous as autonomous_route
from backend.api.routes import compiler as compiler_route
from backend.api.routes import leanoj as leanoj_route
from backend.autonomous.memory.brainstorm_memory import BrainstormMemory
from backend.autonomous.memory.paper_library import PaperLibrary
from backend.autonomous.memory.research_metadata import ResearchMetadata
from backend.shared import critique_memory, solution_path
from backend.shared.models import PaperMetadata, PaperPruneBatchRequest
from backend.shared.translation.service import translation_service

pytestmark = pytest.mark.asyncio


@pytest.fixture
def notes(monkeypatch) -> List[str]:
    """The reasons routes pass to ``note_outputs_changed``, in call order."""
    calls: List[str] = []
    monkeypatch.setattr(translation_service, "note_outputs_changed", calls.append)
    return calls


@asynccontextmanager
async def _no_lock():
    yield


def _quiet_assistant(monkeypatch, route) -> None:
    monkeypatch.setattr(route.assistant_proof_search_coordinator, "stop_all", AsyncMock())
    monkeypatch.setattr(route.assistant_proof_search_coordinator, "clear_cooldown_state", AsyncMock())


def _history_scope(monkeypatch, library=None, metadata=None) -> None:
    """Resolve every history session to one scope built from ``library`` and ``metadata``."""
    paths = {"papers_dir": "papers"}
    monkeypatch.setattr(autonomous_route, "_resolve_history_session_paths", lambda _session_id: paths)
    monkeypatch.setattr(autonomous_route, "_build_scoped_paper_library", lambda _paths: library)
    monkeypatch.setattr(autonomous_route, "_build_scoped_brainstorm_memory", lambda _paths: SimpleNamespace())
    monkeypatch.setattr(autonomous_route, "_build_scoped_research_metadata", AsyncMock(return_value=metadata))


# ----------------------------------------------------------------------
# Manual modes and the Proof Solver
# ----------------------------------------------------------------------


def _aggregator_clear_dependencies(monkeypatch, blocker=None) -> None:
    fake = SimpleNamespace(
        is_running=False, stop=AsyncMock(), clear_all_submissions=AsyncMock(), solution_path_manager=None
    )
    monkeypatch.setattr(aggregator_route, "coordinator", fake)
    monkeypatch.setattr(aggregator_route, "compiler_coordinator", SimpleNamespace(is_running=False))
    monkeypatch.setattr(aggregator_route, "get_manual_proof_context_lock", _no_lock)
    monkeypatch.setattr(aggregator_route, "_manual_proof_clear_blocker", AsyncMock(return_value=blocker))
    monkeypatch.setattr(aggregator_route.manual_proof_database, "archive_current_run", AsyncMock(return_value=0))
    monkeypatch.setattr(aggregator_route, "load_manual_aggregator_prompt", AsyncMock(return_value="prompt"))
    monkeypatch.setattr(aggregator_route, "clear_manual_aggregator_prompt", AsyncMock())
    monkeypatch.setattr(aggregator_route, "clear_manual_main_submitter_config", AsyncMock())
    monkeypatch.setattr(aggregator_route, "_clear_uploaded_files", AsyncMock(return_value=0))
    monkeypatch.setattr(solution_path.solution_path_registry, "clear_run", AsyncMock())
    _quiet_assistant(monkeypatch, aggregator_route)


async def test_aggregator_clear_all_rescans(monkeypatch, notes):
    _aggregator_clear_dependencies(monkeypatch)
    assert (await aggregator_route.clear_all_submissions())["status"] == "cleared"
    assert notes == ["aggregator_cleared"]


async def test_a_refused_aggregator_clear_does_not_rescan(monkeypatch, notes):
    _aggregator_clear_dependencies(monkeypatch, blocker="proof verification is active")
    with pytest.raises(HTTPException) as refused:
        await aggregator_route.clear_all_submissions()
    assert refused.value.status_code == 409
    assert notes == []


async def test_compiler_clear_paper_rescans(monkeypatch, notes):
    fake = SimpleNamespace(
        is_running=False, stop=AsyncMock(), clear_paper=AsyncMock(), user_prompt="prompt", solution_path_manager=None
    )
    monkeypatch.setattr(compiler_route, "compiler_coordinator", fake)
    monkeypatch.setattr(compiler_route, "get_manual_proof_context_lock", _no_lock)
    monkeypatch.setattr(compiler_route, "_manual_proof_clear_blocker", AsyncMock(return_value=None))
    monkeypatch.setattr(compiler_route.manual_proof_database, "archive_current_run", AsyncMock(return_value=0))
    monkeypatch.setattr(compiler_route, "clear_manual_shared_training_proof_appendix", AsyncMock())
    monkeypatch.setattr(compiler_route, "clear_manual_compiler_prompt", AsyncMock())
    monkeypatch.setattr(solution_path.solution_path_registry, "clear_run", AsyncMock())
    monkeypatch.setattr(critique_memory, "clear_critiques", AsyncMock())
    _quiet_assistant(monkeypatch, compiler_route)
    assert (await compiler_route.clear_paper(confirm=True))["status"] == "cleared"
    assert notes == ["compiler_cleared"]


async def test_leanoj_clear_rescans(monkeypatch, notes):
    monkeypatch.setattr(
        leanoj_route, "leanoj_coordinator", SimpleNamespace(clear=AsyncMock(), get_status=lambda: {})
    )
    _quiet_assistant(monkeypatch, leanoj_route)
    assert (await leanoj_route.clear_leanoj(confirm=True))["success"] is True
    assert notes == ["leanoj_cleared"]


# ----------------------------------------------------------------------
# Autonomous Research
# ----------------------------------------------------------------------


async def test_autonomous_clear_rescans(monkeypatch, notes):
    monkeypatch.setattr(
        autonomous_route, "autonomous_coordinator", SimpleNamespace(is_active=False, clear_all_data=AsyncMock())
    )
    _quiet_assistant(monkeypatch, autonomous_route)
    assert (await autonomous_route.clear_autonomous_research(confirm=True))["success"] is True
    assert notes == ["autonomous_cleared"]


async def test_an_autonomous_clear_while_research_runs_does_not_rescan(monkeypatch, notes):
    monkeypatch.setattr(
        autonomous_route, "autonomous_coordinator", SimpleNamespace(is_active=True, clear_all_data=AsyncMock())
    )
    with pytest.raises(HTTPException) as refused:
        await autonomous_route.clear_autonomous_research(confirm=True)
    assert refused.value.status_code == 400
    assert notes == []


async def test_tier3_clear_rescans(monkeypatch, notes):
    monkeypatch.setattr(
        autonomous_route,
        "autonomous_coordinator",
        SimpleNamespace(get_state=lambda: SimpleNamespace(is_running=False), _tier3_active=False),
    )
    monkeypatch.setattr(autonomous_route, "final_answer_memory", SimpleNamespace(clear=AsyncMock()))
    monkeypatch.setattr(critique_memory, "clear_critiques", AsyncMock())
    _quiet_assistant(monkeypatch, autonomous_route)
    assert (await autonomous_route.clear_tier3_data(confirm=True))["success"] is True
    assert notes == ["tier3_cleared"]


async def test_deleting_a_brainstorm_rescans(monkeypatch, notes, tmp_path):
    monkeypatch.setattr(
        autonomous_route,
        "autonomous_coordinator",
        SimpleNamespace(_current_topic_id=None, _brainstorm_aggregator=None, is_active=False),
    )
    monkeypatch.setattr(
        autonomous_route,
        "brainstorm_memory",
        SimpleNamespace(
            get_database_path=lambda topic_id: tmp_path / f"brainstorm_{topic_id}.txt",
            get_metadata=AsyncMock(return_value=SimpleNamespace(papers_generated=[])),
            delete_brainstorm=AsyncMock(return_value=True),
        ),
    )
    monkeypatch.setattr(autonomous_route, "shared_training_memory", SimpleNamespace(file_path=tmp_path / "other.txt"))
    monkeypatch.setattr(
        autonomous_route,
        "research_metadata",
        SimpleNamespace(
            delete_brainstorm=AsyncMock(),
            get_stats=AsyncMock(return_value={}),
            set_current_brainstorm=AsyncMock(),
        ),
    )
    assert (await autonomous_route.delete_brainstorm("topic_1", confirm=True))["success"] is True
    assert notes == ["autonomous_brainstorm_deleted"]


async def test_pruning_a_current_paper_rescans(monkeypatch, notes):
    monkeypatch.setattr(
        autonomous_route, "_delete_autonomous_paper_from_scope", AsyncMock(return_value={"success": True})
    )
    assert (await autonomous_route.delete_paper("paper_1", confirm=True)) == {"success": True}
    assert notes == ["autonomous_paper_pruned"]


async def test_pruning_a_history_paper_rescans(monkeypatch, notes):
    _history_scope(monkeypatch)
    monkeypatch.setattr(autonomous_route, "_ensure_history_paper_is_visible", AsyncMock())
    monkeypatch.setattr(
        autonomous_route, "_delete_autonomous_paper_from_scope", AsyncMock(return_value={"success": True})
    )
    assert (await autonomous_route.delete_history_paper("session_1", "paper_1", confirm=True)) == {"success": True}
    assert notes == ["autonomous_paper_pruned"]


async def test_deleting_the_current_pruned_papers_rescans(monkeypatch, notes):
    monkeypatch.setattr(
        autonomous_route,
        "paper_library",
        SimpleNamespace(
            _base_dir="papers",
            _list_pruned_history_papers_from_directory=AsyncMock(return_value=[{"paper_id": "paper_1"}]),
            delete_all_pruned_papers=AsyncMock(return_value=1),
        ),
    )
    monkeypatch.setattr(autonomous_route, "research_metadata", SimpleNamespace(delete_paper=AsyncMock()))
    assert (await autonomous_route.delete_current_pruned_papers(confirm=True))["deleted_count"] == 1
    assert notes == ["autonomous_pruned_papers_deleted"]


async def test_deleting_history_pruned_papers_rescans(monkeypatch, notes):
    library = SimpleNamespace(
        _list_pruned_history_papers_from_directory=AsyncMock(return_value=[{"paper_id": "paper_1"}]),
        delete_all_pruned_papers=AsyncMock(return_value=1),
    )
    _history_scope(monkeypatch, library=library, metadata=SimpleNamespace(delete_paper=AsyncMock()))
    assert (await autonomous_route.delete_pruned_history_papers("session_1", confirm=True))["success"] is True
    assert notes == ["autonomous_pruned_papers_deleted"]


async def _paper_scope(tmp_path, monkeypatch) -> None:
    """One session scope with one complete paper, built from real stores under ``tmp_path``."""
    session_root = tmp_path / "session"
    paths = {
        "papers_dir": session_root / "papers",
        "brainstorms_dir": session_root / "brainstorms",
        "metadata_path": session_root / "session_metadata.json",
        "stats_path": session_root / "session_stats.json",
        "workflow_state_path": session_root / "workflow_state.json",
    }
    paths["papers_dir"].mkdir(parents=True)
    paths["brainstorms_dir"].mkdir()
    library = PaperLibrary()
    library._base_dir = paths["papers_dir"]
    library._archive_dir = paths["papers_dir"] / "archive"
    library._pruned_dir = paths["papers_dir"] / "pruned"
    metadata = PaperMetadata(
        paper_id="paper_one", title="One", abstract="Abstract", source_brainstorm_ids=[], status="complete"
    )
    await library._save_metadata(metadata)
    library._get_paper_path("paper_one").write_text("paper body", encoding="utf-8")
    research = ResearchMetadata()
    research._metadata_path = paths["metadata_path"]
    research._stats_path = paths["stats_path"]
    research._workflow_state_path = paths["workflow_state_path"]
    await research.initialize()
    await research.register_paper(metadata)
    monkeypatch.setattr(autonomous_route, "_resolve_history_session_paths", lambda _session_id: paths)
    monkeypatch.setattr(autonomous_route, "_build_scoped_paper_library", lambda _paths: library)
    monkeypatch.setattr(autonomous_route, "_build_scoped_brainstorm_memory", lambda _paths: BrainstormMemory())
    monkeypatch.setattr(autonomous_route, "_build_scoped_research_metadata", AsyncMock(return_value=research))
    monkeypatch.setattr(autonomous_route, "_delete_autonomous_paper_from_scope", AsyncMock())
    monkeypatch.setattr(autonomous_route, "_publish_paper_batch_commit", AsyncMock())


async def test_a_paper_batch_prune_rescans(monkeypatch, notes, tmp_path):
    await _paper_scope(tmp_path, monkeypatch)
    request = PaperPruneBatchRequest(targets=[{"session_id": "session", "paper_id": "paper_one"}], confirm=True)
    assert (await autonomous_route.prune_papers_batch(request)).pruned_count == 1
    assert notes == ["autonomous_papers_pruned"]


async def test_a_refused_paper_batch_prune_does_not_rescan(notes):
    target = {"session_id": "session", "paper_id": "paper_one"}
    with pytest.raises(HTTPException) as refused:
        await autonomous_route.prune_papers_batch(PaperPruneBatchRequest(targets=[target, target], confirm=True))
    assert refused.value.status_code == 409
    assert notes == []


@pytest.mark.parametrize(
    "call",
    [
        lambda: compiler_route.clear_paper(confirm=False),
        lambda: leanoj_route.clear_leanoj(confirm=False),
        lambda: autonomous_route.clear_autonomous_research(confirm=False),
        lambda: autonomous_route.clear_tier3_data(confirm=False),
        lambda: autonomous_route.delete_brainstorm("topic_1", confirm=False),
        lambda: autonomous_route.delete_paper("paper_1", confirm=False),
        lambda: autonomous_route.delete_history_paper("session_1", "paper_1", confirm=False),
        lambda: autonomous_route.delete_current_pruned_papers(confirm=False),
        lambda: autonomous_route.delete_pruned_history_papers("session_1", confirm=False),
    ],
    ids=[
        "compiler_clear",
        "leanoj_clear",
        "autonomous_clear",
        "tier3_clear",
        "brainstorm_delete",
        "paper_prune",
        "history_paper_prune",
        "current_pruned_delete",
        "history_pruned_delete",
    ],
)
async def test_an_unconfirmed_request_does_not_rescan(call, notes):
    with pytest.raises(HTTPException) as refused:
        await call()
    assert refused.value.status_code == 400
    assert notes == []


# ----------------------------------------------------------------------
# Startup
# ----------------------------------------------------------------------


async def test_the_leanoj_restore_at_startup_rescans_after_the_run_is_loaded(monkeypatch):
    order: List[str] = []

    async def restore_latest_session(*, auto_resume: bool) -> None:
        assert auto_resume is False
        order.append("restore")

    monkeypatch.setattr(
        main_module,
        "leanoj_coordinator",
        SimpleNamespace(restore_latest_session=restore_latest_session, _request=None),
    )
    monkeypatch.setattr(translation_service, "note_outputs_changed", lambda reason: order.append(reason))
    await main_module._restore_leanoj_session()
    assert order == ["restore", "leanoj_restored"]
