"""Translate all's enumeration of older displayed outputs, and viewer identity parity."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from backend.aggregator.memory.shared_training import SharedTrainingMemory
from backend.api.routes import aggregator as aggregator_routes
from backend.autonomous.memory.final_answer_memory import final_answer_memory
from backend.autonomous.memory.paper_library import paper_library
from backend.autonomous.memory.proof_database import manual_proof_database, proof_database
from backend.compiler.memory.outline_memory import outline_memory
from backend.compiler.memory.paper_memory import format_live_paper_display, paper_memory
from backend.shared.config import bind_runtime_roots, system_config
from backend.shared.models import ProofRecord
from backend.shared.proof_search.moto_sources import _record_from_library_entry, normalize_proof_record
from backend.shared.translation import history
from backend.shared.translation.discovery import (
    CurrentRunState,
    DocumentSpec,
    _read_text_if_present,
    manual_aggregator_results_path,
)
from backend.shared.translation.history import SOURCE_LABELS, enumerate_displayed_outputs, history_sources
from backend.shared.translation.segmentation import PROOF_TRANSLATION_FIELDS
from backend.shared.translation.settings import TRANSLATION_MODES

ALL_MODES = frozenset(TRANSLATION_MODES)
SESSION = "old_session_2026-10-01_10-00_abcd1234"
CURRENT = "current_session_2026-11-02_17-41_49ffb8c9"
BANNER = "PRUNED PAPER - REMOVED FROM MODEL CONTEXT"


@pytest.fixture
def runtime_root(tmp_path):
    original_data = system_config.data_dir
    original_logs = system_config.logs_dir
    bind_runtime_roots(tmp_path / "data", tmp_path / "logs")
    root = Path(system_config.data_dir)
    root.mkdir(parents=True, exist_ok=True)
    yield root
    bind_runtime_roots(original_data, original_logs)


def _state(**overrides: Any) -> CurrentRunState:
    payload = {
        "autonomous_session": CURRENT,
        "aggregator_live": False,
        "compiler_live": False,
        "compiler_writing": False,
        "leanoj_session": "lean-current",
    }
    payload.update(overrides)
    return CurrentRunState(**payload)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _proof_entry(proof_id: str, **fields: Any) -> Dict[str, Any]:
    entry = {
        "proof_id": proof_id,
        "theorem_name": "",
        "theorem_statement": "",
        "formal_sketch": "",
        "novelty_reasoning": "",
        "novelty_group_claim": "",
        "source_type": "brainstorm",
        "source_id": "brainstorm_t1",
        "created_at": "2026-10-01T10:00:00",
    }
    entry.update(fields)
    return entry


def _record(proof_id: str, run_id: str = "run-1", **fields: Any) -> SimpleNamespace:
    return SimpleNamespace(**{**_proof_entry(proof_id, run_id=run_id), **fields})


def _leanoj_state(session_id: str) -> Dict[str, Any]:
    return {
        "session_id": session_id,
        "request": {"user_prompt": "Prove the lattice bound."},
        "final_solution": "theorem t : True := trivial",
        "final_novelty_reasoning": "Not shown in the library view.",
        "final_novelty_group_claim": "",
        "verified_subproofs": [
            {
                "subproof_id": "sub_1",
                "lean_code": "lemma s : True := trivial",
                "theorem_or_lemma": "Every lattice is discrete.",
                "novelty_group_claim": "Together they bound the lattice.",
            }
        ],
    }


@pytest.fixture
def library(runtime_root, monkeypatch):
    """Older outputs of every mode: faked history listings plus real proof, LeanOJ, and manual files."""
    state = {"value": _state()}

    async def fake_state() -> CurrentRunState:
        return state["value"]

    history_papers = {
        (SESSION, "paper_001"): {"title": "Old Paper", "content": "Old paper body.\n\nMore text."},
        (CURRENT, "paper_002"): {"title": "Current Paper", "content": "Current paper body."},
        ("legacy", "paper_003"): {"title": "Blank Paper", "content": "  \n"},
    }
    fetched: List[tuple] = []

    async def list_history_papers() -> List[Dict[str, Any]]:
        return [{"session_id": session, "paper_id": paper} for session, paper in history_papers]

    async def get_history_paper(session_id: str, paper_id: str):
        fetched.append((session_id, paper_id))
        return history_papers.get((session_id, paper_id))

    answers = {
        "legacy": {"metadata": {}, "content": "FINAL ANSWER\nLegacy short form with header and footer."},
        SESSION: {"metadata": {}, "content": "Old volume text."},
    }

    async def list_all_final_answers() -> List[Dict[str, Any]]:
        return [{"answer_id": answer_id, "title": f"Answer {answer_id}"} for answer_id in answers] + [
            {"answer_id": "incomplete", "title": "Writing"}
        ]

    async def get_final_answer_by_id(answer_id: str):
        return answers.get(answer_id)

    async def autonomous_records() -> List[SimpleNamespace]:
        return [
            _record("proof_a", run_id=CURRENT, theorem_statement="A current autonomous proof."),
            _record("proof_l", run_id="lean-old", source_type="leanoj_final", theorem_statement="A LeanOJ proof."),
        ]

    async def manual_records() -> List[SimpleNamespace]:
        return [
            _record("proof_agg", run_id="manual-1", source_id="manual_aggregator", theorem_statement="Aggregator."),
            _record("proof_cmp", run_id="manual-1", source_type="paper", source_id="manual_compiler_current",
                    theorem_statement="Compiler."),
            _record("proof_old", run_id="manual-1", source_type="paper", source_id="paper_001",
                    theorem_statement="Unattributed."),
        ]

    async def get_outline() -> str:
        return "1. Introduction"

    async def get_paper() -> str:
        return "Leftover live paper."

    monkeypatch.setattr(history, "current_run_state", fake_state)
    monkeypatch.setattr(paper_library, "list_history_papers", list_history_papers)
    monkeypatch.setattr(paper_library, "get_history_paper", get_history_paper)
    monkeypatch.setattr(final_answer_memory, "list_all_final_answers", list_all_final_answers)
    monkeypatch.setattr(final_answer_memory, "get_final_answer_by_id", get_final_answer_by_id)
    monkeypatch.setattr(proof_database, "get_all_proofs", autonomous_records)
    monkeypatch.setattr(manual_proof_database, "get_all_proofs", manual_records)
    monkeypatch.setattr(outline_memory, "get_outline", get_outline)
    monkeypatch.setattr(paper_memory, "get_paper", get_paper)

    # Proof library: the legacy store and an older session.
    _write_json(
        runtime_root / "proofs" / "proofs_index.json",
        {"proofs": [_proof_entry("proof_x", theorem_statement="A legacy library proof.", formal_sketch="Sketch.")]},
    )
    _write_json(
        Path(system_config.auto_sessions_base_dir) / SESSION / "proofs" / "proofs_index.json",
        {"proofs": [_proof_entry("proof_y", novelty_reasoning="Why it is new.", novelty_group_claim="  ")]},
    )
    # Archived manual proof runs.
    _write_json(
        runtime_root / "manual_proof_runs" / "manual-run-1" / "proofs" / "proofs_index.json",
        {
            "proofs": [
                _proof_entry("proof_m1", source_id="manual_aggregator", theorem_statement="Archived aggregator."),
                _proof_entry("proof_m2", source_type="paper", source_id="compiler_manual_00ff",
                             theorem_statement="Archived compiler."),
                _proof_entry("proof_m3", source_type="paper", source_id="paper_004",
                             theorem_statement="Archived unattributed."),
            ]
        },
    )
    # Saved Proof Solver runs.
    _write_json(runtime_root / "leanoj_sessions" / "lean-old" / "state.json", _leanoj_state("lean-old"))
    _write_json(runtime_root / "leanoj_sessions" / "lean-current" / "state.json", _leanoj_state("lean-current"))
    # The manual Aggregator database.
    (runtime_root / "rag_shared_training.txt").write_text("An accepted idea.\n", encoding="utf-8")
    return SimpleNamespace(state=state, fetched=fetched)


def _by_key(documents) -> Dict[str, DocumentSpec]:
    return {document.doc_key: document for document in documents}


async def _documents(modes=ALL_MODES) -> Dict[str, DocumentSpec]:
    result = await enumerate_displayed_outputs(modes)
    assert result.failed_sources == ()
    return _by_key(result.documents)


@pytest.mark.asyncio
async def test_every_older_output_matches_its_viewer_recipe(library):
    documents = await _documents()
    assert {key: spec.source for key, spec in documents.items()} == {
        f"paper:{SESSION}:paper_001": "history_papers",
        "final:legacy": "history_final_answers",
        f"final:{SESSION}": "history_final_answers",
        "proof:library:legacy:proof_x": "history_proof_library",
        f"proof:library:{SESSION}:proof_y": "history_proof_library",
        "proof:autonomous:lean-old:proof_l": "autonomous_store_other_proofs",
        "proof:manual:manual-1:proof_old": "manual_store_other_proofs",
        "proof:manual:manual-1:proof_agg": "manual_store_other_proofs",
        "proof:manual:manual-1:proof_cmp": "manual_store_other_proofs",
        "proof:library:manual-run-1:proof_m1": "manual_proof_history",
        "proof:library:manual-run-1:proof_m2": "manual_proof_history",
        "proof:library:manual-run-1:proof_m3": "manual_proof_history",
        "aggregator:results": "aggregator_results_inactive",
        "compiler:live-paper": "compiler_live_paper_inactive",
        "proof:leanoj:lean-old:final_solution": "leanoj_library",
        "proof:leanoj:lean-old:sub_1": "leanoj_library",
    }

    # Stage 2 history: the paper body, without the outline shown above it.
    paper = documents[f"paper:{SESSION}:paper_001"]
    assert (paper.kind, paper.profile, paper.title) == ("paper", "document", "Old Paper")
    assert paper.parts == (("content", "Old paper body.\n\nMore text."),)
    # The current run's history row belongs to current-run discovery and is never read here.
    assert (CURRENT, "paper_002") not in library.fetched

    # Final answers library: the library content, header and footer included.
    final = documents["final:legacy"]
    assert (final.kind, final.profile, final.title) == ("final_answer", "document", "Answer legacy")
    assert final.parts == (("content", "FINAL ANSWER\nLegacy short form with header and footer."),)

    # Proof library entries: the listed prose fields, blank ones dropped.
    assert documents["proof:library:legacy:proof_x"].parts == (
        ("formal_sketch", "Sketch."),
        ("theorem_statement", "A legacy library proof."),
    )
    assert documents[f"proof:library:{SESSION}:proof_y"].parts == (("novelty_reasoning", "Why it is new."),)
    assert documents["proof:library:manual-run-1:proof_m1"].profile == "fields"

    # Proof Solver library: statement and group claim only, with no prompt exclusion.
    assert documents["proof:leanoj:lean-old:final_solution"].parts == (
        ("theorem_statement", "Prove the lattice bound."),
    )
    assert documents["proof:leanoj:lean-old:sub_1"].parts == (
        ("novelty_group_claim", "Together they bound the lattice."),
        ("theorem_statement", "Every lattice is discrete."),
    )

    assert documents["aggregator:results"].parts == (("content", "An accepted idea.\n"),)
    assert documents["compiler:live-paper"].parts == (
        ("content", format_live_paper_display("1. Introduction", "Leftover live paper.")),
    )


@pytest.mark.asyncio
async def test_current_runs_are_left_to_current_run_discovery(library):
    library.state["value"] = _state(aggregator_live=True, compiler_live=True)
    documents = await _documents()
    # Live manual runs keep their proofs and results current; unattributed proofs stay here.
    assert "aggregator:results" not in documents and "compiler:live-paper" not in documents
    assert "proof:manual:manual-1:proof_agg" not in documents
    assert "proof:manual:manual-1:proof_cmp" not in documents
    assert "proof:manual:manual-1:proof_old" in documents
    # The loaded Proof Solver run is current; only older runs are enumerated.
    assert not any(key.startswith("proof:leanoj:lean-current:") for key in documents)
    # The live paper is not enumerated while a compiler writes it.
    library.state["value"] = _state(compiler_writing=True)
    assert "compiler:live-paper" not in await _documents()

    # Without an active session the legacy proof store is not current, so all of it is older output.
    library.state["value"] = _state(autonomous_session="")
    library.fetched.clear()
    documents = await _documents({"autonomous"})
    assert {"proof:autonomous:lean-old:proof_l", f"proof:autonomous:{CURRENT}:proof_a"} <= set(documents)
    # The legacy roots are then the current run, so their history rows are skipped.
    assert ("legacy", "paper_003") not in library.fetched


def test_every_source_is_named_for_messages_and_listed_by_mode():
    assert set(SOURCE_LABELS) == set(history_sources(ALL_MODES))
    assert all(label.strip() for label in SOURCE_LABELS.values())
    assert history_sources({"compiler"}) == (
        "manual_store_other_proofs",
        "manual_proof_history",
        "compiler_live_paper_inactive",
    )
    assert history_sources(()) == ()


@pytest.mark.asyncio
async def test_modes_filter_the_sources_and_manual_records(library):
    assert {spec.source for spec in (await _documents({"leanoj"})).values()} == {"leanoj_library"}
    autonomous = await _documents({"autonomous"})
    assert {spec.source for spec in autonomous.values()} == {
        "history_papers",
        "history_final_answers",
        "history_proof_library",
        "autonomous_store_other_proofs",
    }
    aggregator = await _documents({"aggregator"})
    # Aggregator proofs and unattributed manual proofs, which the manual workspace shows.
    assert sorted(key for key in aggregator if key.startswith("proof:")) == [
        "proof:library:manual-run-1:proof_m1",
        "proof:library:manual-run-1:proof_m3",
        "proof:manual:manual-1:proof_agg",
        "proof:manual:manual-1:proof_old",
    ]
    assert "aggregator:results" in aggregator and "compiler:live-paper" not in aggregator
    compiler = await _documents({"compiler"})
    assert sorted(key for key in compiler if key.startswith("proof:")) == [
        "proof:library:manual-run-1:proof_m2",
        "proof:library:manual-run-1:proof_m3",
        "proof:manual:manual-1:proof_cmp",
        "proof:manual:manual-1:proof_old",
    ]
    assert await _documents(frozenset()) == {}


@pytest.mark.asyncio
async def test_cleared_content_is_not_enumerated(library, runtime_root):
    (runtime_root / "rag_shared_training.txt").write_text("", encoding="utf-8")
    for state_file in (runtime_root / "leanoj_sessions").glob("*/state.json"):
        state_file.unlink()
    documents = await _documents()
    assert "aggregator:results" not in documents
    assert not any(key.startswith("proof:leanoj:") for key in documents)


@pytest.mark.asyncio
async def test_pruned_papers_use_the_banner_content_and_drop_out_once_deleted(runtime_root, monkeypatch, caplog):
    async def fake_state() -> CurrentRunState:
        return _state()

    # Pruned papers are read from real files.
    monkeypatch.setattr(history, "current_run_state", fake_state)
    pruned_dir = Path(system_config.auto_sessions_base_dir) / SESSION / "papers" / "pruned"
    _write_json(
        pruned_dir / "pruned_paper_paper_005_metadata.json",
        {"paper_id": "paper_005", "title": "Pruned", "status": "pruned", "created_at": "2026-10-01T10:00:00",
         "pruned_at": "2026-10-02T10:00:00", "pruned_by": "user"},
    )
    (pruned_dir / "pruned_paper_paper_005.txt").write_text(
        f"{BANNER}\n\nThe user removed this paper.\n{'=' * 80}\n\nPruned body.", encoding="utf-8"
    )
    archive_dir = Path(system_config.auto_papers_dir) / "archive"
    _write_json(
        archive_dir / "paper_paper_006_metadata.json",
        {"paper_id": "paper_006", "title": "Archived", "created_at": "2026-09-01T10:00:00"},
    )
    (archive_dir / "paper_paper_006.txt").write_text("Archived body.", encoding="utf-8")
    # No stored time: its banner would carry the time of each read.
    _write_json(archive_dir / "paper_paper_007_metadata.json", {"paper_id": "paper_007", "title": "Timeless"})
    (archive_dir / "paper_paper_007.txt").write_text("Timeless body.", encoding="utf-8")

    with caplog.at_level(logging.INFO, logger=history.__name__):
        documents = await _documents({"autonomous"})
        await _documents({"autonomous"})
    pruned = documents[f"paper:{SESSION}:paper_005"]
    assert pruned.source == "history_pruned_papers"
    expected = await paper_library.get_pruned_history_paper(SESSION, "paper_005")
    assert pruned.parts == (("content", expected["content"]),)
    archived = documents["paper:legacy:paper_006"]
    assert archived.parts[0][1].startswith(BANNER) and archived.parts[0][1].endswith("Archived body.")
    archived_view = await paper_library.get_pruned_history_paper("legacy", "paper_006")
    assert archived.parts == (("content", archived_view["content"]),)
    assert "paper:legacy:paper_007" not in documents
    assert sum("paper_007" in record.getMessage() for record in caplog.records) == 1

    (pruned_dir / "pruned_paper_paper_005_metadata.json").unlink()
    assert f"paper:{SESSION}:paper_005" not in await _documents({"autonomous"})


@pytest.mark.asyncio
async def test_cleared_autonomous_sessions_stay_in_the_libraries(library):
    # Autonomous Clear marks sessions cleared but keeps listing them in Completed Works.
    session_dir = Path(system_config.auto_sessions_base_dir) / SESSION
    _write_json(
        session_dir / "session_metadata.json",
        {"session_id": SESSION, "status": "cleared", "resume_disabled": True},
    )
    documents = await _documents({"autonomous"})
    assert f"proof:library:{SESSION}:proof_y" in documents
    assert f"paper:{SESSION}:paper_001" in documents


@pytest.mark.asyncio
async def test_a_failing_source_is_isolated(library, monkeypatch):
    async def broken(_scope) -> List[DocumentSpec]:
        raise RuntimeError("history unavailable")

    monkeypatch.setattr(history, "_enumerate_history_final_answers", broken)
    result = await enumerate_displayed_outputs(ALL_MODES)
    assert result.failed_sources == ("history_final_answers",)
    keys = {document.doc_key for document in result.documents}
    assert "final:legacy" not in keys and f"paper:{SESSION}:paper_001" in keys

    async def unreadable() -> CurrentRunState:
        raise OSError("prompt file unreadable")

    monkeypatch.setattr(history, "current_run_state", unreadable)
    result = await enumerate_displayed_outputs({"compiler", "leanoj"})
    assert result.documents == ()
    assert result.failed_sources == (
        "manual_store_other_proofs",
        "manual_proof_history",
        "compiler_live_paper_inactive",
        "leanoj_library",
    )


# ----------------------------------------------------------------------
# Viewer identity parity
# ----------------------------------------------------------------------


def _proof_record(**fields: Any) -> ProofRecord:
    payload = {
        "proof_id": "proof_1",
        "theorem_name": "Lemma",
        "theorem_statement": "  Every lattice is discrete.\n",
        "formal_sketch": "Use compactness.",
        "novelty_reasoning": "No reference states it.",
        "novelty_group_claim": "Together they bound it.",
        "source_type": "brainstorm",
        "source_id": "brainstorm_t1",
        "lean_code": "theorem t : True := trivial",
    }
    payload.update(fields)
    return ProofRecord(**payload)


def test_proof_search_hydration_keeps_the_prose_fields_byte_for_byte():
    record = _proof_record()
    for hydrated in (
        normalize_proof_record(record, session_id="s1"),
        _record_from_library_entry(record.model_dump(mode="json"), default_corpus="moto"),
    ):
        assert {field: getattr(hydrated, field) for field in PROOF_TRANSLATION_FIELDS} == {
            field: getattr(record, field) for field in PROOF_TRANSLATION_FIELDS
        }


@pytest.mark.asyncio
async def test_live_results_text_equals_the_saved_aggregator_file(runtime_root):
    memory = SharedTrainingMemory()
    memory.file_path = manual_aggregator_results_path()
    await memory.initialize()
    await memory.add_accepted_submission("First accepted idea.")
    await memory.add_accepted_submission("Second idea with $x^2$ math.")
    formatted = await memory.get_all_content_formatted()
    assert formatted == _read_text_if_present(manual_aggregator_results_path())


@pytest.mark.asyncio
async def test_aggregator_results_report_whether_they_read_the_manual_database(runtime_root, monkeypatch):
    reader = SharedTrainingMemory()

    async def formatted() -> str:
        return await reader.get_all_content_formatted()

    monkeypatch.setattr(aggregator_routes, "shared_training_memory", reader)
    monkeypatch.setattr(
        aggregator_routes, "coordinator", SimpleNamespace(is_running=False, get_results_formatted=formatted)
    )
    monkeypatch.setattr(aggregator_routes, "autonomous_coordinator", SimpleNamespace(is_active=False))
    with monkeypatch.context() as patch:
        patch.setattr(system_config, "shared_training_file", str(manual_aggregator_results_path()))
        assert (await aggregator_routes.get_results())["manual_database"] is True

    # During an autonomous run the memory may hold a child database, which the route leaves in place.
    reader.file_path = runtime_root / "auto_brainstorms" / "brainstorm_t1.txt"
    monkeypatch.setattr(aggregator_routes, "autonomous_coordinator", SimpleNamespace(is_active=True))
    result = await aggregator_routes.get_results()
    assert result["manual_database"] is False
    assert set(result) == {"results", "manual_database"}
