from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from backend.aggregator.core.context_allocator import ContextAllocator
from backend.compiler.agents.high_param_submitter import HighParamSubmitter
from backend.compiler.agents.writer_submitter import WritingSubmitter
from backend.shared.config import system_config
from tests.workflow_harness.real_adapters.source_tagged_rag import SourceTaggedRagIndex


@pytest.mark.asyncio
async def test_build_e_aggregator_mixed_direct_and_offloaded_sources_are_excluded(
    monkeypatch,
):
    index = SourceTaggedRagIndex()
    index.add("rag_shared_training.txt", "DUPLICATE SHARED")
    index.add("upload.txt", "DUPLICATE UPLOAD")
    index.add("local_training.txt", "OFFLOADED LOCAL EVIDENCE")
    monkeypatch.setattr(
        "backend.aggregator.core.context_allocator.rag_manager",
        index,
    )
    monkeypatch.setattr(
        ContextAllocator,
        "_get_shared_training_rag_sources",
        lambda _self: ["rag_shared_training.txt"],
    )
    monkeypatch.setattr(
        "backend.aggregator.core.context_allocator.count_tokens",
        lambda text: len(text.split()),
    )

    allocation = await ContextAllocator().allocate_submitter_context(
        user_prompt="solve the target",
        json_schema="{}",
        system_prompt="return JSON",
        shared_training_content="DIRECT SHARED",
        local_training_content="local " * 100_000,
        rejection_log_content="",
        user_files_content={"upload.txt": "DIRECT UPLOAD"},
        chunk_size=256,
        context_window=12_000,
        max_output_tokens=512,
    )

    assert "DIRECT SHARED" in allocation["direct"]
    assert "DIRECT UPLOAD" in allocation["direct"]
    assert "OFFLOADED LOCAL EVIDENCE" in allocation["rag_context"].text
    assert "DUPLICATE SHARED" not in allocation["rag_context"].text
    assert "DUPLICATE UPLOAD" not in allocation["rag_context"].text
    assert set(index.calls[-1]["exclude_sources"]) == {
        "rag_shared_training.txt",
        "upload.txt",
    }


@pytest.mark.asyncio
async def test_build_e_aggregator_source_becomes_retrievable_when_offloaded(
    monkeypatch,
):
    index = SourceTaggedRagIndex()
    index.add("rag_shared_training.txt", "SHARED SOURCE RETRIEVED")
    monkeypatch.setattr(
        "backend.aggregator.core.context_allocator.rag_manager",
        index,
    )
    monkeypatch.setattr(
        ContextAllocator,
        "_get_shared_training_rag_sources",
        lambda _self: ["rag_shared_training.txt"],
    )
    monkeypatch.setattr(
        "backend.aggregator.core.context_allocator.count_tokens",
        lambda text: len(text.split()),
    )

    allocation = await ContextAllocator().allocate_submitter_context(
        user_prompt="solve the target",
        json_schema="{}",
        system_prompt="return JSON",
        shared_training_content="shared " * 100_000,
        local_training_content="",
        rejection_log_content="",
        user_files_content={},
        chunk_size=256,
        context_window=12_000,
        max_output_tokens=512,
    )

    assert "SHARED SOURCE RETRIEVED" in allocation["rag_context"].text
    assert "rag_shared_training.txt" not in index.calls[-1]["exclude_sources"]


@pytest.mark.asyncio
async def test_build_e_compiler_construction_excludes_direct_sources_and_keeps_references(
    monkeypatch,
):
    index = SourceTaggedRagIndex()
    index.add("compiler_outline.txt", "DUPLICATE OUTLINE")
    index.add("compiler_paper.txt", "DUPLICATE PAPER")
    index.add("brainstorm_topic.txt", "DUPLICATE BRAINSTORM")
    index.add("reference_paper.txt", "ELIGIBLE REFERENCE EVIDENCE")
    captured: dict[str, str] = {}

    monkeypatch.setattr(
        "backend.compiler.agents.writer_submitter.compiler_rag_manager",
        index,
    )
    monkeypatch.setattr(
        "backend.compiler.agents.writer_submitter.outline_memory.get_outline",
        AsyncMock(return_value="DIRECT OUTLINE"),
    )
    monkeypatch.setattr(
        "backend.compiler.agents.writer_submitter.paper_memory.get_paper",
        AsyncMock(return_value="DIRECT PAPER"),
    )
    monkeypatch.setattr(
        "backend.compiler.agents.writer_submitter.api_client_manager.prewarm_assistant_memory_context",
        AsyncMock(),
    )

    async def generate_completion(**kwargs):
        captured["prompt"] = kwargs["messages"][0]["content"]
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"needs_construction": false, "section_complete": false, '
                            '"reasoning": "No edit."}'
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(
        "backend.compiler.agents.writer_submitter.api_client_manager.generate_completion",
        generate_completion,
    )
    monkeypatch.setattr(system_config, "compiler_writer_context_window", 32_000)
    monkeypatch.setattr(system_config, "compiler_writer_max_output_tokens", 1_000)

    submitter = WritingSubmitter("fake-model", "solve the target")
    await submitter.submit_construction(
        section_phase="body",
        brainstorm_content="DIRECT BRAINSTORM",
        brainstorm_source_name="brainstorm_topic.txt",
    )

    assert "DIRECT OUTLINE" in captured["prompt"]
    assert "DIRECT PAPER" in captured["prompt"]
    assert "DIRECT BRAINSTORM" in captured["prompt"]
    assert "ELIGIBLE REFERENCE EVIDENCE" in captured["prompt"]
    assert "DUPLICATE OUTLINE" not in captured["prompt"]
    assert "DUPLICATE PAPER" not in captured["prompt"]
    assert "DUPLICATE BRAINSTORM" not in captured["prompt"]
    assert set(index.calls[-1]["exclude_sources"]) == {
        "compiler_outline.txt",
        "compiler_paper.txt",
        "brainstorm_topic.txt",
    }


@pytest.mark.asyncio
async def test_build_e_compiler_rigor_excludes_direct_outline_and_paper(monkeypatch):
    index = SourceTaggedRagIndex()
    index.add("compiler_outline.txt", "DUPLICATE RIGOR OUTLINE")
    index.add("compiler_paper.txt", "DUPLICATE RIGOR PAPER")
    index.add("reference_paper.txt", "RIGOR REFERENCE EVIDENCE")
    monkeypatch.setattr(
        "backend.compiler.agents.high_param_submitter.compiler_rag_manager",
        index,
    )
    monkeypatch.setattr(system_config, "compiler_high_param_context_window", 16_000)
    monkeypatch.setattr(system_config, "compiler_high_param_max_output_tokens", 1_000)

    submitter = HighParamSubmitter("fake-model", "solve the target")
    submitter.context_window = 16_000
    submitter.max_output_tokens = 1_000
    evidence = await submitter._build_rigor_rag_context(
        query_seed="DIRECT OUTLINE DIRECT PAPER",
        reserved_tokens=2_000,
    )

    assert "RIGOR REFERENCE EVIDENCE" in evidence
    assert "DUPLICATE RIGOR OUTLINE" not in evidence
    assert "DUPLICATE RIGOR PAPER" not in evidence
    assert set(index.calls[-1]["exclude_sources"]) == {
        "compiler_outline.txt",
        "compiler_paper.txt",
    }
