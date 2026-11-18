"""Master proof expansion wired through the router, prompt sites, stage, and export."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from backend.aggregator.core.context_allocator import ContextAllocator
from backend.api.routes.proofs import _build_master_proof_bundle
from backend.autonomous.core.master_proof import MasterProofState, MasterProofStore, master_proof_key
from backend.autonomous.core.proof_verification_stage import ProofVerificationStage
from backend.autonomous.prompts.master_proof_prompts import build_master_proof_scaffold_prompt
from backend.autonomous.prompts.proof_prompts import build_proof_identification_prompt
from backend.compiler.prompts.construction_prompts import build_construction_prompt
from backend.shared.api_client_manager import APIClientManager
from backend.shared.master_proof_context import (
    MASTER_PROOF_LIST_NOTICE,
    PROOFS_BLOCK_END,
    PROOFS_BLOCK_START,
    master_proof_handle,
    noted_context_parts,
    offered_handles,
    render_master_proof_entry,
)
from backend.shared.master_proof_expansion import MASTER_PROOF_EXPANSION_EVENT
from backend.shared.models import MasterProofStep, MasterProofSummary, ModelConfig

MASTER_CODE = """import Mathlib

def weight (n : Nat) : Nat := n + 1

lemma step_one (n : Nat) : n + 0 = n := by
  simp

theorem master (n : Nat) : n + 0 = n := by
  exact step_one n"""
SUMMARY = MasterProofSummary(
    master_key="key",
    root_theorem_name="master",
    total_attempts=4,
    steps=[MasterProofStep(name="step_one", signature="lemma step_one (n : Nat) : n + 0 = n", failed_attempts=1)],
)
HANDLE = master_proof_handle(MASTER_CODE)


def _prompt() -> str:
    entry = render_master_proof_entry(
        heading="PROOF 1 [Mathematical Discovery]",
        proof_id="proof_001",
        theorem_name="master",
        theorem_statement="Adding zero is the identity.",
        lean_code=MASTER_CODE,
        summary=SUMMARY,
    )
    block = "\n".join([PROOFS_BLOCK_START, MASTER_PROOF_LIST_NOTICE, *entry, "---", PROOFS_BLOCK_END])
    return f"ROLE TASK\n{block}\nRespond with JSON."


def _response(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


def _manager(replies: list[str], seen: list[dict]) -> APIClientManager:
    manager = APIClientManager()
    manager.configure_role(
        "aggregator_submitter_1",
        ModelConfig(
            provider="openrouter", model_id="test/model", context_window=64_000, max_output_tokens=2_000
        ),
    )

    async def fake_once(*, task_id, messages, tools=None, **_kwargs):
        seen.append({"task_id": task_id, "messages": messages, "tools": tools})
        return _response(replies.pop(0))

    manager._generate_completion_once = fake_once  # type: ignore[method-assign]
    return manager


@pytest.mark.asyncio
async def test_router_serves_an_expansion_request_then_returns_the_role_answer():
    seen: list[dict] = []
    events: list[tuple[str, dict]] = []
    manager = _manager(
        [
            json.dumps({"expand_master_proof": HANDLE}),
            '{"keep_declarations": ["step_one"]}',
            '{"submission": "built on step_one"}',
        ],
        seen,
    )

    async def broadcast(event, payload):
        events.append((event, payload))

    manager.set_broadcast_callback(broadcast)
    response = await manager.generate_completion(
        task_id="agg_sub1_001",
        role_id="aggregator_submitter_1",
        model="test/model",
        messages=[{"role": "user", "content": _prompt()}],
    )

    assert response["choices"][0]["message"]["content"] == '{"submission": "built on step_one"}'
    assert len(seen) == 3
    assert "MASTER PROOF EXPANSION REVIEW" in seen[1]["messages"][0]["content"]
    assert "def weight (n : Nat) : Nat := n + 1" in seen[1]["messages"][0]["content"]
    reissued = seen[2]["messages"][0]["content"]
    assert "lemma step_one (n : Nat) : n + 0 = n := by\n  simp" in reissued
    assert "def weight" not in reissued
    assert offered_handles(reissued) == []
    expansion = [payload for event, payload in events if event == MASTER_PROOF_EXPANSION_EVENT]
    assert [payload["outcome"] for payload in expansion] == ["carried"]
    assert expansion[0]["role_id"] == "aggregator_submitter_1"


@pytest.mark.asyncio
async def test_router_leaves_ordinary_answers_and_tool_calls_alone():
    seen: list[dict] = []
    manager = _manager(['{"submission": "plain"}', json.dumps({"expand_master_proof": HANDLE})], seen)

    plain = await manager.generate_completion(
        task_id="agg_sub1_001",
        role_id="aggregator_submitter_1",
        model="test/model",
        messages=[{"role": "user", "content": _prompt()}],
    )
    assert plain["choices"][0]["message"]["content"] == '{"submission": "plain"}'

    # A tool-call conversation keeps its exact turns, so expansion is not offered there
    # and a request that arrives anyway is returned as-is.
    with_tools = await manager.generate_completion(
        task_id="agg_sub1_002",
        role_id="aggregator_submitter_1",
        model="test/model",
        messages=[{"role": "user", "content": _prompt()}],
        tools=[{"type": "function", "function": {"name": "search_lean_proofs", "parameters": {}}}],
    )
    assert "expand_master_proof" in with_tools["choices"][0]["message"]["content"]
    assert len(seen) == 2
    tool_prompt = seen[1]["messages"][0]["content"]
    assert "expand_master_proof" not in tool_prompt
    assert offered_handles(tool_prompt) == []
    assert "Solved master proof template" in tool_prompt


def test_prompt_sites_record_the_context_that_can_make_room():
    brainstorm = "[SHARED TRAINING]\n" + "accepted submission text. " * 40
    files = "[FILE: paper_one.txt]\n" + "reference paper text. " * 40
    ContextAllocator._note_droppable_direct_parts([brainstorm, "[REJECTION LOG]\n" + "x" * 400, files])

    source = "brainstorm source content. " * 40
    build_proof_identification_prompt(
        user_prompt="prompt", source_type="brainstorm", source_id="topic_001", source_content=source
    )
    paper_source = "paper source content. " * 40
    build_master_proof_scaffold_prompt(
        user_prompt="prompt",
        source_type="paper",
        source_title="",
        theorem_statement="statement",
        formal_sketch="",
        full_source_content=paper_source,
        relevant_lemmas=[],
        prior_attempts=[],
    )
    noted = {(part.kind, part.text) for part in noted_context_parts()}
    assert ("brainstorms", brainstorm) in noted
    assert ("papers", files) in noted
    assert ("brainstorms", source) in noted
    assert ("papers", paper_source) in noted
    assert not any(text.startswith("[REJECTION LOG]") for _kind, text in noted)


@pytest.mark.asyncio
async def test_compiler_construction_records_the_paper_and_brainstorm():
    paper = "paper in progress. " * 40
    brainstorm = "brainstorm database entry. " * 40
    prompt = await build_construction_prompt(
        user_prompt="prompt",
        current_outline="I. Introduction",
        current_paper=paper,
        rag_evidence="",
        brainstorm_content=brainstorm,
    )
    noted = {(part.kind, part.text) for part in noted_context_parts()}
    assert ("papers", paper) in noted and ("brainstorms", brainstorm) in noted
    # Recorded text is the exact text in the prompt, so it can be located later.
    assert paper in prompt and brainstorm in prompt


@pytest.mark.asyncio
async def test_stage_reads_the_build_summary_from_the_saved_scaffold_state():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp) / "master_proofs"

        class Database:
            def get_master_proofs_dir(self):
                return directory

        key = master_proof_key(
            run_id="run-1", source_type="brainstorm", source_id="topic_001", statement="Statement."
        )
        state = MasterProofState(
            master_key=key,
            theorem_id="thm_1",
            statement="Statement.",
            status="verified",
            root_name="master",
            scaffold_code=MASTER_CODE,
            scaffold_revision=2,
            total_attempts=5,
        )
        await MasterProofStore(directory).save(state)

        summary = await ProofVerificationStage._load_master_proof_summary(
            Database(), run_id="run-1", source_type="brainstorm", source_id="topic_001", statement="Statement."
        )
        assert (summary.master_key, summary.root_theorem_name, summary.total_attempts) == (key, "master", 5)
        missing = await ProofVerificationStage._load_master_proof_summary(
            Database(), run_id="run-2", source_type="brainstorm", source_id="topic_001", statement="Statement."
        )
        assert missing.steps == [] and missing.master_key


def test_bundle_holds_the_whole_master_proof_in_one_text_export():
    bundle = _build_master_proof_bundle(
        {
            "proof_id": "proof_001",
            "run_id": "run-1",
            "theorem_name": "master",
            "theorem_statement": "Adding zero is the identity.",
            "novelty_tier": "mathematical_discovery",
            "proof_kind": "master_proof",
            "master_proof": SUMMARY.model_dump(mode="json"),
        },
        MASTER_CODE,
    )
    assert bundle.lean_code == MASTER_CODE
    assert "sorry  -- SOLVED" in bundle.solved_template
    assert [(step.name, step.failed_attempts) for step in bundle.steps] == [("step_one", 1)]
    assert bundle.steps[0].lean_code == "lemma step_one (n : Nat) : n + 0 = n := by\n  simp"
    assert [item.name for item in bundle.supporting_declarations] == ["weight"]
    assert bundle.master_proof.total_attempts == 4


@pytest.mark.asyncio
async def test_displaced_context_is_retrieved_through_a_one_off_source_and_removed(monkeypatch):
    """The real retrieval pipeline serves the excerpt and leaves no source behind."""
    import uuid

    import chromadb

    import backend.aggregator.core.rag_manager as rag_module
    from backend.aggregator.core.rag_manager import RAGManager
    from backend.shared.master_proof_context import EPHEMERAL_RAG_SOURCE_PREFIX
    from backend.shared.master_proof_expansion import retrieve_context_evidence

    class FakeEmbeddingRoute:
        async def get_embeddings_in_space(self, texts, *, embedding_space=None, model=None):
            vectors = [
                [float(len(text) % 7 + 1), float("renewal" in text), 1.0, 0.5] for text in texts
            ]
            return vectors, embedding_space or "lm_studio:test-model"

    monkeypatch.setattr(rag_module, "api_client_manager", FakeEmbeddingRoute())
    client = chromadb.EphemeralClient()
    manager = RAGManager()
    manager.collections = {
        size: client.get_or_create_collection(
            name=f"chunks_{size}_{uuid.uuid4().hex}", metadata={"chunk_size": size}
        )
        for size in (256, 512, 768, 1024)
    }

    async def opened():
        return None

    monkeypatch.setattr(manager, "ensure_initialized", opened)
    monkeypatch.setattr(manager, "_ensure_initialized_locked", lambda: None)
    monkeypatch.setattr(rag_module, "rag_manager", manager)

    paper = "\n\n".join(
        f"Section {index}. The renewal histogram converse bounds the archive in case {index}."
        for index in range(30)
    )
    evidence = await retrieve_context_evidence("papers", paper, 400, "renewal histogram converse")

    assert "renewal histogram converse" in evidence
    assert not any(
        name.startswith(EPHEMERAL_RAG_SOURCE_PREFIX) for name in manager.document_access_order
    )
    assert not any(
        chunk.source_file.startswith(EPHEMERAL_RAG_SOURCE_PREFIX)
        for chunks in manager.chunks_by_size.values()
        for chunk in chunks
    )
    assert await retrieve_context_evidence("papers", "   ", 400, "query") == ""
    assert await retrieve_context_evidence("papers", paper, 0, "query") == ""


@pytest.mark.asyncio
async def test_certificate_marks_a_master_proof():
    from backend.api.routes.proofs import _certificate_response

    record = {
        "proof_id": "proof_001",
        "theorem_statement": "Adding zero is the identity.",
        "theorem_name": "master",
        "source_type": "brainstorm",
        "source_id": "topic_001",
        "novelty_tier": "mathematical_discovery",
        "proof_kind": "master_proof",
        "master_proof": SUMMARY.model_dump(mode="json"),
    }
    payload = json.loads((await _certificate_response(record, MASTER_CODE)).body)
    assert payload["proof_kind"] == "master_proof"
    assert payload["master_proof"]["steps"][0]["name"] == "step_one"

    individual = json.loads(
        (await _certificate_response({**record, "proof_kind": "individual"}, MASTER_CODE)).body
    )
    assert individual["proof_kind"] == "individual" and individual["master_proof"] is None
