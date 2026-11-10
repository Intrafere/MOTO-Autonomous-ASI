"""Discovery of the checked modes' current-run outputs for Auto Top-p Translate."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from backend.api.routes import compiler as compiler_routes
from backend.autonomous.core.autonomous_coordinator import autonomous_coordinator
from backend.autonomous.memory.brainstorm_memory import brainstorm_memory
from backend.autonomous.memory.final_answer_memory import final_answer_memory
from backend.autonomous.memory.paper_library import paper_library
from backend.autonomous.memory.proof_database import manual_proof_database, proof_database
from backend.compiler.core.compiler_coordinator import compiler_coordinator
from backend.compiler.memory.outline_memory import outline_memory
from backend.compiler.memory.paper_memory import format_live_paper_display, paper_memory
from backend.leanoj.core.leanoj_coordinator import leanoj_coordinator
from backend.leanoj.core.leanoj_proof_views import _extract_leanoj_proofs
from backend.shared.config import bind_runtime_roots, system_config
from backend.shared.models import FinalAnswerState, VolumeOrganization
from backend.shared.translation import discovery
from backend.shared.translation.discovery import (
    LEANOJ_TRANSLATION_FIELDS,
    CurrentRunState,
    DocumentSpec,
    canonical_parts,
    current_run_state,
    discover_current_outputs,
    js_trim,
    leanoj_proof_specs,
    make_document_spec,
    manual_proof_mode,
    proof_document_specs,
)
from backend.shared.translation.settings import TRANSLATION_MODES

ALL_MODES = frozenset(TRANSLATION_MODES)
LIVE_PAPER = "Live paper body.\n\nSecond paragraph."
LIVE_OUTLINE = "1. Introduction\n2. Results"


@pytest.fixture
def runtime_root(tmp_path):
    original_data = system_config.data_dir
    original_logs = system_config.logs_dir
    bind_runtime_roots(tmp_path / "data", tmp_path / "logs")
    root = Path(system_config.data_dir)
    root.mkdir(parents=True, exist_ok=True)
    yield root
    bind_runtime_roots(original_data, original_logs)


def _proof(proof_id: str, run_id: str = "run-1", **fields: Any) -> SimpleNamespace:
    payload = {
        "proof_id": proof_id,
        "run_id": run_id,
        "source_type": "",
        "source_id": "",
        "theorem_name": "",
        "theorem_statement": "",
        "formal_sketch": "",
        "novelty_reasoning": "",
        "novelty_group_claim": "",
    }
    payload.update(fields)
    return SimpleNamespace(**payload)


def _state(**overrides: Any) -> CurrentRunState:
    payload = {
        "autonomous_session": "sess-1",
        "aggregator_live": True,
        "compiler_live": True,
        "compiler_writing": False,
        "leanoj_session": "lean-1",
    }
    payload.update(overrides)
    return CurrentRunState(**payload)


def _leanoj_status() -> Dict[str, Any]:
    return {
        "session_id": "lean-1",
        "request": {"user_prompt": "Prove the lattice bound.", "lean_template": "theorem t : True := by sorry"},
        "final_solution": "theorem t : True := trivial",
        "final_novelty_reasoning": "No reference proves this bound.",
        "verified_subproofs": [
            {
                "subproof_id": "sub_1",
                "lean_code": "lemma s : True := trivial",
                "theorem_or_lemma": "Every lattice is discrete.",
                "novelty_reasoning": "",
                "novelty_group_claim": "Together they bound the lattice.",
            },
        ],
    }


@pytest.fixture
def sources(runtime_root, monkeypatch):
    """Fake every memory source with a small, fully known output set."""
    papers = {
        "p1": ("Paper One", "Paper one body.\n\nLean 4 proof:\ntheorem x : True := trivial\n---\n"),
        "p2": ("Empty Paper", "   "),
    }
    state = {"value": _state()}

    async def fake_state() -> CurrentRunState:
        return state["value"]

    async def get_all_papers(**_kwargs: Any) -> List[SimpleNamespace]:
        return [SimpleNamespace(paper_id=paper_id, title=title) for paper_id, (title, _text) in papers.items()]

    async def get_paper_content(paper_id: str, **_kwargs: Any) -> str:
        return papers[paper_id][1] if paper_id in papers else ""

    async def get_metadata(paper_id: str, **_kwargs: Any):
        return SimpleNamespace(title=papers[paper_id][0]) if paper_id in papers else None

    async def get_final_volume() -> str:
        return "Final volume text."

    async def autonomous_proofs() -> List[SimpleNamespace]:
        return [
            _proof("proof_1", theorem_name="Main lemma", theorem_statement="Every ring is a ring.",
                   formal_sketch="Use the definition.", novelty_reasoning="New.", novelty_group_claim=""),
            _proof("proof_2", run_id=""),
            _proof("proof_3", run_id="lean-1", source_type="leanoj_subproof", theorem_statement="A LeanOJ lemma."),
        ]

    async def manual_proofs() -> List[SimpleNamespace]:
        return [
            _proof("proof_9", run_id="manual-abc", source_type="brainstorm", source_id="manual_aggregator",
                   theorem_statement="Two plus two is four.\nDetails."),
            _proof("proof_10", run_id="manual-abc", source_type="paper", source_id="manual_compiler_current",
                   theorem_statement="A compiler theorem."),
            _proof("proof_11", run_id="manual-abc", source_type="paper", source_id="paper_001",
                   theorem_statement="An unattributed theorem."),
        ]

    async def get_all_brainstorms() -> List[SimpleNamespace]:
        return [
            SimpleNamespace(topic_id="t1", status="complete", topic_prompt="Topic one"),
            SimpleNamespace(topic_id="t2", status="in_progress", topic_prompt="Topic two"),
        ]

    async def get_database_content(topic_id: str, **_kwargs: Any) -> str:
        return f"Brainstorm content for {topic_id}."

    async def get_outline() -> str:
        return LIVE_OUTLINE

    async def get_paper() -> str:
        return LIVE_PAPER

    monkeypatch.setattr(discovery, "current_run_state", fake_state)
    monkeypatch.setattr(paper_library, "get_all_papers", get_all_papers)
    monkeypatch.setattr(paper_library, "get_paper_content", get_paper_content)
    monkeypatch.setattr(paper_library, "get_metadata", get_metadata)
    monkeypatch.setattr(
        final_answer_memory,
        "get_state",
        lambda: FinalAnswerState(
            status="complete",
            answer_format="long_form",
            volume_organization=VolumeOrganization(volume_title="The Volume"),
        ),
    )
    monkeypatch.setattr(final_answer_memory, "get_final_volume", get_final_volume)
    monkeypatch.setattr(proof_database, "get_all_proofs", autonomous_proofs)
    monkeypatch.setattr(manual_proof_database, "get_all_proofs", manual_proofs)
    monkeypatch.setattr(brainstorm_memory, "get_all_brainstorms", get_all_brainstorms)
    monkeypatch.setattr(brainstorm_memory, "get_database_content", get_database_content)
    monkeypatch.setattr(outline_memory, "get_outline", get_outline)
    monkeypatch.setattr(paper_memory, "get_paper", get_paper)
    monkeypatch.setattr(leanoj_coordinator, "get_status", _leanoj_status)
    (runtime_root / "rag_shared_training.txt").write_text(
        "=====\nSUBMISSION #1 | Accepted: now\n=====\n\nAn accepted idea.\n",
        encoding="utf-8",
        newline="\n",
    )
    return state


def _by_key(documents) -> Dict[str, DocumentSpec]:
    return {document.doc_key: document for document in documents}


async def _keys(modes=ALL_MODES) -> List[str]:
    return [document.doc_key for document in (await discover_current_outputs(modes)).documents]


@pytest.mark.asyncio
async def test_discovery_builds_documented_doc_keys_parts_and_sources(sources):
    result = await discover_current_outputs(ALL_MODES)
    assert result.failed_sources == ()
    documents = _by_key(result.documents)
    assert list(documents) == [
        "paper:sess-1:p1",
        "final_answer:sess-1",
        "proof:autonomous:run-1:proof_1",
        "brainstorm:current:t1",
        "aggregator:results",
        "proof:manual:manual-abc:proof_9",
        "compiler:live-paper",
        "proof:manual:manual-abc:proof_10",
        "proof:leanoj:lean-1:final_solution",
        "proof:leanoj:lean-1:sub_1",
    ]
    assert {key: spec.source for key, spec in documents.items()} == {
        "paper:sess-1:p1": "autonomous_papers",
        "final_answer:sess-1": "autonomous_final_answer",
        "proof:autonomous:run-1:proof_1": "autonomous_proofs",
        "brainstorm:current:t1": "autonomous_brainstorms",
        "aggregator:results": "aggregator_results",
        "proof:manual:manual-abc:proof_9": "aggregator_proofs",
        "compiler:live-paper": "compiler_live_paper",
        "proof:manual:manual-abc:proof_10": "compiler_proofs",
        "proof:leanoj:lean-1:final_solution": "leanoj_proofs",
        "proof:leanoj:lean-1:sub_1": "leanoj_proofs",
    }

    paper = documents["paper:sess-1:p1"]
    assert (paper.kind, paper.profile, paper.title) == ("paper", "document", "Paper One")
    assert paper.parts == (("content", "Paper one body.\n\nLean 4 proof:\ntheorem x : True := trivial\n---\n"),)

    final = documents["final_answer:sess-1"]
    assert (final.kind, final.title, final.parts) == ("final_answer", "The Volume", (("content", "Final volume text."),))

    proof = documents["proof:autonomous:run-1:proof_1"]
    assert (proof.kind, proof.profile, proof.title) == ("proof", "fields", "Main lemma")
    # Only non-empty natural-language fields, sorted by name like viewer registrations.
    assert proof.parts == (
        ("formal_sketch", "Use the definition."),
        ("novelty_reasoning", "New."),
        ("theorem_statement", "Every ring is a ring."),
    )
    assert documents["proof:manual:manual-abc:proof_9"].title == "Two plus two is four."

    brainstorm = documents["brainstorm:current:t1"]
    assert (brainstorm.kind, brainstorm.title) == ("brainstorm", "Topic one")
    assert brainstorm.parts == (("content", "Brainstorm content for t1."),)

    aggregator = documents["aggregator:results"]
    assert aggregator.kind == "aggregator_results"
    assert "An accepted idea." in aggregator.parts[0][1]

    live = documents["compiler:live-paper"]
    assert (live.kind, live.profile) == ("live_paper", "document")
    assert live.parts == (("content", format_live_paper_display(LIVE_OUTLINE, LIVE_PAPER)),)


@pytest.mark.asyncio
async def test_only_the_checked_modes_are_scanned(sources):
    assert await _keys({"autonomous"}) == [
        "paper:sess-1:p1",
        "final_answer:sess-1",
        "proof:autonomous:run-1:proof_1",
        "brainstorm:current:t1",
    ]
    assert await _keys({"aggregator"}) == ["aggregator:results", "proof:manual:manual-abc:proof_9"]
    assert await _keys({"compiler"}) == ["compiler:live-paper", "proof:manual:manual-abc:proof_10"]
    assert await _keys({"leanoj"}) == ["proof:leanoj:lean-1:final_solution", "proof:leanoj:lean-1:sub_1"]
    result = await discover_current_outputs(frozenset())
    assert (result.documents, result.failed_sources) == ((), ())


@pytest.mark.asyncio
async def test_manual_modes_without_a_current_run_have_no_current_outputs(sources):
    sources["value"] = _state(aggregator_live=False, compiler_live=False)
    assert await _keys({"aggregator", "compiler"}) == []
    sources["value"] = _state(aggregator_live=False)
    assert await _keys({"aggregator", "compiler"}) == ["compiler:live-paper", "proof:manual:manual-abc:proof_10"]


@pytest.mark.asyncio
async def test_the_live_paper_is_left_out_while_a_compiler_writes_it(sources):
    sources["value"] = _state(compiler_writing=True)
    # Its proofs are complete records, so they stay current.
    assert await _keys({"compiler"}) == ["proof:manual:manual-abc:proof_10"]


@pytest.mark.asyncio
async def test_autonomous_proofs_exclude_proof_solver_records_and_need_a_session(sources):
    keys = await _keys({"autonomous"})
    assert "proof:autonomous:lean-1:proof_3" not in keys
    sources["value"] = _state(autonomous_session="")
    keys = await _keys({"autonomous"})
    # The legacy paper and brainstorm roots are current, but the legacy proof store is not.
    assert keys == ["paper:legacy:p1", "brainstorm:current:t1"]


@pytest.mark.asyncio
async def test_proof_solver_needs_a_loaded_run(sources):
    sources["value"] = _state(leanoj_session="")
    assert await _keys({"leanoj"}) == []


@pytest.mark.asyncio
async def test_final_answer_requires_a_complete_answer_in_an_active_session(sources, monkeypatch):
    monkeypatch.setattr(final_answer_memory, "get_state", lambda: FinalAnswerState(status="writing", is_active=True))
    assert "final_answer:sess-1" not in await _keys({"autonomous"})

    monkeypatch.setattr(
        final_answer_memory,
        "get_state",
        lambda: FinalAnswerState(status="complete", answer_format="short_form", short_form_paper_id="p1"),
    )
    documents = _by_key((await discover_current_outputs({"autonomous"})).documents)
    assert documents["final_answer:sess-1"].title == "Paper One"
    assert documents["final_answer:sess-1"].parts[0][1].startswith("Paper one body.")


@pytest.mark.asyncio
async def test_failing_source_is_skipped_and_reported(sources, monkeypatch):
    async def broken(_state: CurrentRunState) -> List[DocumentSpec]:
        raise RuntimeError("brainstorm store unavailable")

    monkeypatch.setattr(discovery, "_scan_autonomous_brainstorms", broken)
    result = await discover_current_outputs(ALL_MODES)
    assert result.failed_sources == ("autonomous_brainstorms",)
    keys = [document.doc_key for document in result.documents]
    assert "brainstorm:current:t1" not in keys
    assert "paper:sess-1:p1" in keys and "aggregator:results" in keys


@pytest.mark.asyncio
async def test_unreadable_current_runs_fail_every_checked_source(sources, monkeypatch):
    async def broken() -> CurrentRunState:
        raise OSError("prompt file unreadable")

    monkeypatch.setattr(discovery, "current_run_state", broken)
    result = await discover_current_outputs({"aggregator", "leanoj"})
    assert result.documents == ()
    assert result.failed_sources == ("aggregator_results", "aggregator_proofs", "leanoj_proofs")


@pytest.mark.asyncio
async def test_aggregator_results_come_from_the_manual_file_in_the_data_root(sources, monkeypatch, tmp_path):
    other = tmp_path / "autonomous_override.txt"
    other.write_text("Autonomous brainstorm database.", encoding="utf-8")
    # Autonomous runs temporarily repoint the shared-training file. The patch is
    # scoped so it is undone before the runtime-root fixture rebinds the roots;
    # a function-level undo would run after that rebind and leave a stale path.
    with monkeypatch.context() as patch:
        patch.setattr(system_config, "shared_training_file", str(other))
        documents = _by_key((await discover_current_outputs({"aggregator"})).documents)
    assert "An accepted idea." in documents["aggregator:results"].parts[0][1]
    assert "Autonomous brainstorm database." not in documents["aggregator:results"].parts[0][1]

    (Path(system_config.data_dir) / "rag_shared_training.txt").unlink()
    assert "aggregator:results" not in await _keys({"aggregator"})


@pytest.mark.asyncio
async def test_current_run_state_reads_the_manual_prompts_and_run_owners(runtime_root, monkeypatch):
    monkeypatch.setattr(discovery, "_active_session_id", lambda: "sess-9")
    monkeypatch.setattr(compiler_coordinator, "is_running", False)
    monkeypatch.setattr(type(autonomous_coordinator), "is_active", property(lambda _self: False))
    monkeypatch.setattr(leanoj_coordinator, "get_state", lambda: SimpleNamespace(session_id=" lean-7 "))
    state = await current_run_state()
    assert state == CurrentRunState(
        autonomous_session="sess-9",
        aggregator_live=False,
        compiler_live=False,
        compiler_writing=False,
        leanoj_session="lean-7",
    )
    assert state.autonomous_run == "sess-9"

    (runtime_root / "manual_aggregator_prompt.txt").write_text("Aggregator prompt", encoding="utf-8")
    (runtime_root / "manual_compiler_prompt.txt").write_text("   \n", encoding="utf-8")
    state = await current_run_state()
    assert (state.aggregator_live, state.compiler_live) == (True, False)
    (runtime_root / "manual_compiler_prompt.txt").write_text("Compiler prompt", encoding="utf-8")
    assert (await current_run_state()).compiler_live is True

    monkeypatch.setattr(compiler_coordinator, "is_running", True)
    assert (await current_run_state()).compiler_writing is True
    monkeypatch.setattr(compiler_coordinator, "is_running", False)
    # Autonomous child compilers write the same compiler paper.
    monkeypatch.setattr(type(autonomous_coordinator), "is_active", property(lambda _self: True))
    assert (await current_run_state()).compiler_writing is True

    monkeypatch.setattr(discovery, "_active_session_id", lambda: "")
    assert (await current_run_state()).autonomous_run == "legacy"


def test_manual_proof_mode_follows_the_proof_source():
    def mode(source_type: str, source_id: str) -> str:
        return manual_proof_mode(_proof("p", source_type=source_type, source_id=source_id))

    assert mode("brainstorm", "manual_aggregator") == "aggregator"
    assert mode("paper", "manual_compiler_current") == "compiler"
    assert mode("paper", "manual_compiler_standalone_0123456789ab") == "compiler"
    assert mode("paper", "compiler_manual_0123456789abcdef") == "compiler"
    assert mode("paper", "paper_001") == ""
    assert mode("brainstorm", "brainstorm_t1") == ""
    assert mode("paper", "manual_aggregator") == ""
    assert manual_proof_mode({"source_type": "brainstorm", "source_id": "manual_aggregator"}) == "aggregator"


def test_js_trim_matches_javascript_string_trim():
    bom, file_separator, next_line = chr(0xFEFF), chr(0x1C), chr(0x85)
    no_break, line_separator = chr(0xA0), chr(0x2028)
    # JavaScript trims a byte order mark; Python's strip keeps it.
    assert js_trim(f"{bom} text {bom}") == "text"
    assert f"{bom}".strip() == bom
    # Python strips these separators; JavaScript keeps them.
    assert js_trim(f"{file_separator}text{next_line}") == f"{file_separator}text{next_line}"
    assert f"{file_separator}text{next_line}".strip() == "text"
    assert js_trim(f"\t{no_break}{line_separator} text \n") == "text"


def test_blank_checks_use_javascript_trim():
    bom = chr(0xFEFF)
    [spec] = proof_document_specs(
        [_proof("p1", theorem_statement="Statement.", formal_sketch=f" {bom} ", novelty_reasoning=chr(0x1C))],
        "autonomous",
    )
    # A part only JavaScript considers blank is dropped; one only Python considers blank is kept.
    assert spec.parts == (("novelty_reasoning", chr(0x1C)), ("theorem_statement", "Statement."))


def test_proof_specs_skip_records_without_text_and_default_the_run_id():
    specs = proof_document_specs(
        [
            _proof("a", run_id="", theorem_statement="Statement A."),
            _proof("b"),
            _proof("", theorem_statement="No identifier."),
        ],
        "autonomous",
    )
    assert [spec.doc_key for spec in specs] == ["proof:autonomous:current:a"]


def test_leanoj_parts_match_the_proof_solver_proof_view():
    proofs = _extract_leanoj_proofs(_leanoj_status())
    specs = _by_key(leanoj_proof_specs(proofs, LEANOJ_TRANSLATION_FIELDS, hide_prompt_statement=True))
    # The final proof's statement repeats the run prompt, so the view never shows or sends it.
    assert specs["proof:leanoj:lean-1:final_solution"].parts == (
        ("novelty_reasoning", "No reference proves this bound."),
    )
    assert specs["proof:leanoj:lean-1:sub_1"].parts == (
        ("novelty_group_claim", "Together they bound the lattice."),
        ("theorem_statement", "Every lattice is discrete."),
    )
    # The statement is compared after a JavaScript trim of both sides.
    bom = chr(0xFEFF)
    padded = {**proofs[0], "theorem_statement": f"{bom}Prove the lattice bound.\n"}
    [spec] = leanoj_proof_specs([padded], LEANOJ_TRANSLATION_FIELDS, hide_prompt_statement=True)
    assert [name for name, _text in spec.parts] == ["novelty_reasoning"]
    [kept] = leanoj_proof_specs([padded], LEANOJ_TRANSLATION_FIELDS, hide_prompt_statement=False)
    assert [name for name, _text in kept.parts] == ["novelty_reasoning", "theorem_statement"]


@pytest.mark.asyncio
async def test_live_paper_text_matches_the_compiler_paper_route(sources, monkeypatch):
    async def word_count() -> int:
        return 4

    monkeypatch.setattr(paper_memory, "get_word_count", word_count)
    route = await compiler_routes.get_paper()
    documents = _by_key((await discover_current_outputs({"compiler"})).documents)
    assert documents["compiler:live-paper"].parts == (("content", route["paper"]),)
    assert format_live_paper_display("", LIVE_PAPER) == LIVE_PAPER


def test_spec_validation():
    spec = make_document_spec(
        doc_key="paper:s:p", kind="paper", title="  A\n title  ", profile="document", parts=[("content", "x")]
    )
    assert spec.title == "A title"
    assert spec.source == ""
    for bad in (
        {"doc_key": "", "kind": "paper", "profile": "document", "parts": []},
        {"doc_key": "k", "kind": "novel", "profile": "document", "parts": []},
        {"doc_key": "k", "kind": "paper", "profile": "slides", "parts": []},
        {"doc_key": "k", "kind": "paper", "profile": "document", "parts": [("a", "x"), ("a", "y")]},
        {"doc_key": "k\x00", "kind": "paper", "profile": "document", "parts": []},
    ):
        with pytest.raises(ValueError):
            make_document_spec(title="", **bad)


def test_part_order_is_canonical_so_viewer_and_discovery_specs_match():
    record = _proof(
        "p1",
        theorem_statement="Every ring is a ring.",
        formal_sketch="Use the definition.",
        novelty_reasoning="New.",
    )
    [discovered] = proof_document_specs([record], "autonomous")
    # A viewer lists the same fields in another order.
    viewer = make_document_spec(
        doc_key=discovered.doc_key,
        kind="proof",
        title="",
        profile="fields",
        parts=[
            ("novelty_reasoning", "New."),
            ("theorem_statement", "Every ring is a ring."),
            ("formal_sketch", "Use the definition."),
        ],
    )
    assert viewer.parts == discovered.parts
    assert [name for name, _text in discovered.parts] == ["formal_sketch", "novelty_reasoning", "theorem_statement"]
    assert canonical_parts([("b", "2"), ("a", "1")]) == (("a", "1"), ("b", "2"))
