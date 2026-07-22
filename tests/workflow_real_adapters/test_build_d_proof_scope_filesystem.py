from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from backend.api.routes import autonomous as autonomous_route
from backend.api.routes import leanoj as leanoj_route
from backend.api.routes import proofs as proofs_route
from backend.autonomous.core.autonomous_rag_manager import autonomous_rag_manager
from backend.autonomous.memory.paper_library import PaperLibrary
from backend.autonomous.memory.proof_database import ProofDatabase
from backend.leanoj.core.leanoj_coordinator import LeanOJCoordinator
from backend.leanoj.core.leanoj_context import leanoj_context_manager
from backend.shared.config import system_config
from backend.shared.models import LeanOJSubproofRecord, PaperMetadata, ProofRecord
from tests.workflow_harness.real_adapters import minimal_leanoj_request
from tests.workflow_harness.real_adapters.filesystem_assertions import (
    assert_files_exist,
    assert_paths_absent,
    assert_paths_within,
)


def _proof(proof_id: str, statement: str, *, run_id: str) -> ProofRecord:
    return ProofRecord(
        proof_id=proof_id,
        theorem_statement=statement,
        theorem_name=f"{proof_id}_theorem",
        source_type="brainstorm",
        source_id=run_id,
        source_title=run_id,
        user_prompt=f"Prompt for {run_id}",
        run_id=run_id,
        lean_code=f"theorem {proof_id}_theorem : True := by trivial",
        solver="Lean 4",
        created_at=datetime(2026, 7, 13),
        novel=True,
        novelty_tier="mathematical_discovery",
        novelty_reasoning="Isolated workflow-scope fixture.",
        attempt_count=1,
    )


async def _database(base_dir: Path, *records: ProofRecord) -> ProofDatabase:
    database = ProofDatabase()
    database.set_base_dir(base_dir)
    await database.initialize()
    for record in records:
        await database.add_proof(record)
    return database


@pytest.mark.asyncio
async def test_autonomous_current_and_library_routes_do_not_read_manual_scope(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    sessions_root = data_root / "auto_sessions"
    active_dir = sessions_root / "active-session" / "proofs"
    history_dir = sessions_root / "history-session" / "proofs"
    manual_dir = data_root / "manual_proofs"
    autonomous = await _database(
        active_dir,
        _proof("proof_autonomous_current", "AUTONOMOUS CURRENT SENTINEL", run_id="active-session"),
    )
    await _database(
        history_dir,
        _proof("proof_autonomous_history", "AUTONOMOUS HISTORY SENTINEL", run_id="history-session"),
    )
    manual = await _database(
        manual_dir,
        _proof("proof_manual_private", "MANUAL PRIVATE SENTINEL", run_id="manual-run"),
    )

    monkeypatch.setattr(system_config, "data_dir", data_root)
    monkeypatch.setattr(system_config, "auto_sessions_base_dir", sessions_root)
    monkeypatch.setattr(proofs_route, "proof_database", autonomous)
    monkeypatch.setattr(proofs_route, "manual_proof_database", manual)

    current = await proofs_route.list_proofs(scope="autonomous")
    library = await proofs_route.get_proof_library(scope="autonomous")
    current_statements = {item["theorem_statement"] for item in current["proofs"]}
    library_statements = {item["theorem_statement"] for item in library["proofs"]}

    assert current_statements == {"AUTONOMOUS CURRENT SENTINEL"}
    assert "AUTONOMOUS HISTORY SENTINEL" in library_statements
    assert "MANUAL PRIVATE SENTINEL" not in current_statements | library_statements
    assert current["scope"] == library["scope"] == "autonomous"
    assert_paths_within(data_root, [active_dir, history_dir, manual_dir])


@pytest.mark.asyncio
async def test_manual_clear_archive_moves_active_proofs_to_history_only(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    active_dir = data_root / "manual_proofs"
    history_root = data_root / "manual_proof_runs"
    manual = await _database(
        active_dir,
        _proof("proof_manual_active", "MANUAL ACTIVE SENTINEL", run_id="manual-active"),
    )
    autonomous = await _database(
        data_root / "autonomous-active",
        _proof("proof_autonomous_private", "AUTONOMOUS PRIVATE SENTINEL", run_id="auto-active"),
    )
    monkeypatch.setattr(system_config, "data_dir", data_root)
    monkeypatch.setattr(proofs_route, "manual_proof_database", manual)
    monkeypatch.setattr(proofs_route, "proof_database", autonomous)

    before = await proofs_route.list_proofs(scope="manual")
    metadata = await manual.archive_current_run(
        history_root,
        user_prompt="Archived manual prompt.",
        reason="build_d_clear",
    )
    after = await proofs_route.list_proofs(scope="manual")
    history = await proofs_route.get_proof_library(scope="manual")

    assert metadata is not None
    assert [proof["theorem_statement"] for proof in before["proofs"]] == ["MANUAL ACTIVE SENTINEL"]
    assert after["proofs"] == []
    assert {proof["theorem_statement"] for proof in history["proofs"]} == {
        "MANUAL ACTIVE SENTINEL"
    }
    assert "AUTONOMOUS PRIVATE SENTINEL" not in json.dumps(history)
    assert_files_exist(
        data_root,
        f"manual_proof_runs/{metadata['session_id']}/session_metadata.json",
        f"manual_proof_runs/{metadata['session_id']}/proofs/proof_proof_manual_active.json",
        "manual_proofs/proofs_index.json",
    )
    assert_paths_absent(data_root, "manual_proofs/proof_proof_manual_active.json")


@pytest.mark.asyncio
async def test_leanoj_current_and_history_routes_are_disjoint_then_clear_removes_root(
    monkeypatch, tmp_path
):
    data_root = tmp_path / "data"
    monkeypatch.setattr(system_config, "data_dir", data_root)
    coordinator = LeanOJCoordinator()
    coordinator.set_broadcast_callback(AsyncMock())
    await coordinator.initialize(minimal_leanoj_request())
    current_session = coordinator.get_state().session_id
    coordinator.get_state().verified_subproofs.append(
        LeanOJSubproofRecord(
            subproof_id="current_subproof",
            request="Current request",
            theorem_or_lemma="LEANOJ CURRENT SENTINEL",
            verified=True,
            lean_code="theorem leanoj_current : True := by trivial",
        )
    )
    await coordinator._persist_state()

    history_session = "history-session"
    history_dir = data_root / "leanoj_sessions" / history_session
    history_dir.mkdir(parents=True)
    history_payload = {
        "session_id": history_session,
        "phase": "verified",
        "updated_at": "2026-07-12T00:00:00",
        "request": {
            "user_prompt": "Historical LeanOJ prompt",
            "lean_template": "example : True := by trivial",
        },
        "verified_subproofs": [
            {
                "subproof_id": "history_subproof",
                "request": "History request",
                "theorem_or_lemma": "LEANOJ HISTORY SENTINEL",
                "verified": True,
                "lean_code": "theorem leanoj_history : True := by trivial",
            }
        ],
    }
    (history_dir / "state.json").write_text(json.dumps(history_payload), encoding="utf-8")
    monkeypatch.setattr(leanoj_route, "leanoj_coordinator", coordinator)
    monkeypatch.setattr(leanoj_route.assistant_proof_search_coordinator, "stop_all", AsyncMock())
    monkeypatch.setattr(
        leanoj_route.assistant_proof_search_coordinator,
        "clear_cooldown_state",
        AsyncMock(),
    )
    monkeypatch.setattr(
        leanoj_context_manager,
        "remove_all_leanoj_rag_sources",
        AsyncMock(),
    )

    current = await leanoj_route.get_leanoj_proofs()
    history = await leanoj_route.get_leanoj_library()

    assert {proof["theorem_statement"] for proof in current["proofs"]} == {
        "LEANOJ CURRENT SENTINEL"
    }
    assert {proof["theorem_statement"] for proof in history["proofs"]} == {
        "LEANOJ HISTORY SENTINEL"
    }
    assert {session["session_id"] for session in history["sessions"]} == {history_session}
    assert current_session not in {session["session_id"] for session in history["sessions"]}
    assert_files_exist(
        data_root,
        f"leanoj_sessions/{current_session}/state.json",
        f"leanoj_sessions/{history_session}/state.json",
    )

    cleared = await leanoj_route.clear_leanoj(confirm=True)

    assert cleared["success"] is True
    assert_paths_absent(data_root, "leanoj_sessions", "leanoj_partial_proofs", "leanoj_artifacts")


@pytest.mark.asyncio
async def test_pruned_paper_moves_to_temp_history_and_route_excludes_active_file(
    monkeypatch, tmp_path
):
    data_root = tmp_path / "data"
    papers_dir = data_root / "auto_sessions" / "session-prune" / "papers"
    papers_dir.mkdir(parents=True)
    library = PaperLibrary()
    library._base_dir = papers_dir
    library._archive_dir = papers_dir / "archive"
    library._pruned_dir = papers_dir / "pruned"
    library._archive_dir.mkdir()
    library._pruned_dir.mkdir()
    metadata = PaperMetadata(
        paper_id="paper_scope",
        title="Pruned Scope Paper",
        status="complete",
        source_brainstorm_ids=["topic-scope"],
    )
    (papers_dir / "paper_paper_scope.txt").write_text("PRUNED PAPER SENTINEL", encoding="utf-8")
    (papers_dir / "paper_paper_scope_metadata.json").write_text(
        json.dumps(metadata.model_dump(mode="json"), default=str),
        encoding="utf-8",
    )
    monkeypatch.setattr(system_config, "data_dir", data_root)
    monkeypatch.setattr(system_config, "auto_sessions_base_dir", data_root / "auto_sessions")
    monkeypatch.setattr(system_config, "auto_papers_dir", data_root / "auto_papers")
    monkeypatch.setattr(autonomous_route, "paper_library", library)

    class ScopedMetadata:
        def __init__(self):
            self.pruned: list[tuple[str, str, str]] = []

        async def prune_paper(self, paper_id, *, reason, pruned_by):
            self.pruned.append((paper_id, reason, pruned_by))

    class ScopedBrainstorms:
        def __init__(self):
            self.removed: list[tuple[str, str]] = []

        async def remove_paper_reference(self, topic_id, paper_id):
            self.removed.append((topic_id, paper_id))

    scoped_metadata = ScopedMetadata()
    scoped_brainstorms = ScopedBrainstorms()
    rag_removals = AsyncMock()
    monkeypatch.setattr(autonomous_rag_manager, "remove_paper_from_rag", rag_removals)

    result = await autonomous_route._delete_autonomous_paper_from_scope(
        session_id="session-prune",
        scoped_paper_library=library,
        scoped_brainstorm_memory=scoped_brainstorms,
        scoped_research_metadata=scoped_metadata,
        paper_id="paper_scope",
    )
    response = await autonomous_route.get_pruned_paper_history()

    assert result["success"] is True
    assert result["pruned"] is True
    assert result["source_brainstorms"] == ["topic-scope"]
    assert response["success"] is True
    assert response["total_count"] == 1
    assert response["papers"][0]["paper_id"] == "paper_scope"
    assert response["papers"][0]["status"] == "pruned"
    assert await library.get_all_papers() == []
    assert await library.get_papers_summary() == []
    assert scoped_metadata.pruned[0][0] == "paper_scope"
    assert scoped_brainstorms.removed == [("topic-scope", "paper_scope")]
    rag_removals.assert_awaited_once_with("paper_scope")
    assert_files_exist(
        data_root,
        "auto_sessions/session-prune/papers/pruned/pruned_paper_paper_scope.txt",
        "auto_sessions/session-prune/papers/pruned/pruned_paper_paper_scope_metadata.json",
    )
    assert_paths_absent(
        data_root,
        "auto_sessions/session-prune/papers/paper_paper_scope.txt",
        "auto_sessions/session-prune/papers/paper_paper_scope_metadata.json",
    )
