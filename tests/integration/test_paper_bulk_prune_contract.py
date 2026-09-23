import asyncio
import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routes import autonomous as autonomous_route
from backend.autonomous.memory.brainstorm_memory import BrainstormMemory
from backend.autonomous.memory.paper_library import PaperLibrary
from backend.autonomous.memory.research_metadata import ResearchMetadata
from backend.shared.models import PaperMetadata, PaperPruneBatchRequest


def test_openapi_exposes_atomic_session_qualified_paper_prune_batch() -> None:
    app = FastAPI()
    app.include_router(autonomous_route.router)
    schema = TestClient(app).get("/openapi.json").json()
    operation = schema["paths"]["/api/auto-research/papers/prune-batch"]["post"]

    assert (
        operation["requestBody"]["content"]["application/json"]["schema"]["$ref"]
        == "#/components/schemas/PaperPruneBatchRequest"
    )
    assert (
        operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
        == "#/components/schemas/PaperPruneBatchResponse"
    )
    target_schema = schema["components"]["schemas"]["PaperPruneBatchTarget"]
    assert set(target_schema["required"]) == {"session_id", "paper_id"}
    request_schema = schema["components"]["schemas"]["PaperPruneBatchRequest"]
    assert set(request_schema["required"]) == {"targets", "confirm"}
    assert request_schema["properties"]["confirm"]["const"] is True


@pytest.mark.asyncio
async def test_paper_batch_rolls_back_selected_files_without_erasing_unrelated_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        paper_id="paper_one",
        title="One",
        abstract="Abstract",
        source_brainstorm_ids=[],
        status="complete",
    )
    await library._save_metadata(metadata)
    library._get_paper_path("paper_one").write_text("paper body", encoding="utf-8")
    research = ResearchMetadata()
    research._metadata_path = paths["metadata_path"]
    research._stats_path = paths["stats_path"]
    research._workflow_state_path = paths["workflow_state_path"]
    await research.initialize()
    await research.register_paper(metadata)

    monkeypatch.setattr(autonomous_route, "_resolve_history_session_paths", lambda _sid: paths)
    monkeypatch.setattr(autonomous_route, "_build_scoped_paper_library", lambda _paths: library)
    monkeypatch.setattr(autonomous_route, "_build_scoped_brainstorm_memory", lambda _paths: BrainstormMemory())

    async def scoped_metadata(_paths):
        return research

    monkeypatch.setattr(autonomous_route, "_build_scoped_research_metadata", scoped_metadata)
    original_delete = autonomous_route._delete_autonomous_paper_from_scope

    async def fail_after_write(**kwargs):
        await original_delete(**kwargs)
        unrelated = paths["papers_dir"] / "unrelated.txt"
        unrelated.write_text("concurrent state", encoding="utf-8")
        raise OSError("forced transaction failure")

    monkeypatch.setattr(autonomous_route, "_delete_autonomous_paper_from_scope", fail_after_write)
    with pytest.raises(Exception, match="Paper batch pruning failed"):
        await autonomous_route.prune_papers_batch(
            PaperPruneBatchRequest(
                targets=[{"session_id": "session", "paper_id": "paper_one"}],
                confirm=True,
            )
        )

    assert library._get_paper_path("paper_one").read_text(encoding="utf-8") == "paper body"
    assert (paths["papers_dir"] / "unrelated.txt").read_text(encoding="utf-8") == "concurrent state"


@pytest.mark.asyncio
async def test_paper_batch_rolls_back_when_cancelled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        paper_id="paper_cancel",
        title="Cancel",
        abstract="Abstract",
        source_brainstorm_ids=[],
        status="complete",
    )
    await library._save_metadata(metadata)
    library._get_paper_path("paper_cancel").write_text("paper body", encoding="utf-8")
    research = ResearchMetadata()
    research._metadata_path = paths["metadata_path"]
    research._stats_path = paths["stats_path"]
    research._workflow_state_path = paths["workflow_state_path"]
    await research.initialize()
    await research.register_paper(metadata)

    monkeypatch.setattr(autonomous_route, "_resolve_history_session_paths", lambda _sid: paths)
    monkeypatch.setattr(autonomous_route, "_build_scoped_paper_library", lambda _paths: library)
    monkeypatch.setattr(autonomous_route, "_build_scoped_brainstorm_memory", lambda _paths: BrainstormMemory())

    async def scoped_metadata(_paths):
        return research

    monkeypatch.setattr(autonomous_route, "_build_scoped_research_metadata", scoped_metadata)
    original_delete = autonomous_route._delete_autonomous_paper_from_scope
    mutation_finished = asyncio.Event()
    release = asyncio.Event()

    async def pause_after_write(**kwargs):
        result = await original_delete(**kwargs)
        mutation_finished.set()
        await release.wait()
        return result

    monkeypatch.setattr(autonomous_route, "_delete_autonomous_paper_from_scope", pause_after_write)
    task = asyncio.create_task(
        autonomous_route.prune_papers_batch(
            PaperPruneBatchRequest(
                targets=[{"session_id": "session", "paper_id": "paper_cancel"}],
                confirm=True,
            )
        )
    )
    await mutation_finished.wait()
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert library._get_paper_path("paper_cancel").read_text(encoding="utf-8") == "paper body"
    assert not library._get_pruned_paper_path("paper_cancel").exists()
