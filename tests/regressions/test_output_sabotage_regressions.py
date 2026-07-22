import pytest

from backend.aggregator.core.context_allocator import ContextAllocator
from backend.aggregator.core import context_allocator as context_allocator_module
from backend.aggregator.core.rag_manager import rag_manager
from backend.compiler.agents.high_param_submitter import HighParamSubmitter
from backend.shared.config import system_config
from backend.shared.models import ContextPack
from backend.shared.openrouter_client import CreditExhaustionError


@pytest.mark.asyncio
async def test_submitter_direct_injects_shared_memory_that_fits(monkeypatch):
    async def fail_retrieve(*args, **kwargs):
        raise AssertionError("RAG should not run when accepted memory fits directly")

    monkeypatch.setattr(context_allocator_module.rag_manager, "retrieve", fail_retrieve)

    allocator = ContextAllocator()
    shared_memory = "accepted insight " * 4000

    result = await allocator.allocate_submitter_context(
        user_prompt="solve the problem",
        json_schema='{"submission": "string"}',
        system_prompt="submit useful work",
        shared_training_content=shared_memory,
        local_training_content="",
        rejection_log_content="",
        user_files_content={},
        chunk_size=512,
        context_window=10000,
        max_output_tokens=1000,
    )

    assert shared_memory in result["direct"]
    assert result["rag_context"] is None


@pytest.mark.asyncio
async def test_submitter_frees_rag_budget_by_offloading_direct_shared_memory(monkeypatch):
    captured = {}

    def fake_count_tokens(text, *args, **kwargs):
        if str(text).startswith("[SHARED TRAINING]"):
            return 3000
        if str(text).startswith("[FILE:"):
            return 2000
        if "RETRIEVED EVIDENCE" in str(text):
            return 10
        return 10

    async def fake_retrieve(*, query, chunk_size, max_tokens, exclude_sources=None):
        captured["max_tokens"] = max_tokens
        captured["exclude_sources"] = exclude_sources
        return ContextPack(text="retrieved accepted memory", evidence=[])

    monkeypatch.setattr(context_allocator_module, "count_tokens", fake_count_tokens)
    monkeypatch.setattr(context_allocator_module.rag_manager, "retrieve", fake_retrieve)

    allocator = ContextAllocator()
    result = await allocator.allocate_submitter_context(
        user_prompt="solve",
        json_schema="{}",
        system_prompt="submit",
        shared_training_content="accepted memory that fits directly",
        local_training_content="",
        rejection_log_content="",
        user_files_content={"large.txt": "large file"},
        chunk_size=512,
        context_window=4000,
        max_output_tokens=100,
    )

    assert "[SHARED TRAINING]" not in result["direct"]
    assert captured["max_tokens"] > 0
    assert captured["exclude_sources"] is None


@pytest.mark.asyncio
async def test_rag_excludes_direct_sources_before_hybrid_recall(monkeypatch):
    captured = {}

    async def fake_recall(queries, chunk_size, exclude_sources=None, include_sources=None, include_source_prefixes=None):
        captured["exclude_sources"] = exclude_sources
        return []

    monkeypatch.setattr(rag_manager, "_hybrid_recall", fake_recall)

    await rag_manager.retrieve(
        query="find relevant context",
        chunk_size=512,
        max_tokens=1000,
        exclude_sources=["rag_shared_training.txt"],
    )

    assert captured["exclude_sources"] == ["rag_shared_training.txt"]


def test_rigor_proof_source_preserves_full_source_context(monkeypatch):
    monkeypatch.setattr(system_config, "compiler_high_param_context_window", 10000)
    monkeypatch.setattr(system_config, "compiler_high_param_max_output_tokens", 1000)
    submitter = HighParamSubmitter(
        model_name="test-model",
        user_prompt="prove something",
        validator_context_window=10000,
        validator_max_tokens=1000,
    )
    source = "start " + ("middle evidence " * 6000) + "end"
    submitter.set_source_material_context(source, "source database")

    proof_source = submitter._get_paper_proof_source_content("Current paper.")

    assert "middle evidence" in proof_source
    assert source in proof_source
    assert "direct source context truncated" not in proof_source


@pytest.mark.asyncio
async def test_rigor_llm_provider_failure_is_not_converted_to_decline(monkeypatch):
    monkeypatch.setattr(system_config, "compiler_high_param_context_window", 10000)
    monkeypatch.setattr(system_config, "compiler_high_param_max_output_tokens", 1000)
    submitter = HighParamSubmitter(
        model_name="test-model",
        user_prompt="prove something",
        validator_context_window=10000,
        validator_max_tokens=1000,
    )

    async def no_op_cache(*args, **kwargs):
        return None

    async def raise_credit_exhaustion(*args, **kwargs):
        raise CreditExhaustionError("credits exhausted")

    monkeypatch.setattr(
        "backend.compiler.agents.high_param_submitter.lm_studio_client.cache_model_load_config",
        no_op_cache,
    )
    monkeypatch.setattr(
        "backend.compiler.agents.high_param_submitter.api_client_manager.generate_completion",
        raise_credit_exhaustion,
    )

    with pytest.raises(CreditExhaustionError):
        await submitter._call_llm_and_parse(prompt="{}", task_label="rigor_discovery")


@pytest.mark.asyncio
async def test_rigor_wrapped_provider_failure_is_not_converted_to_decline(monkeypatch):
    monkeypatch.setattr(system_config, "compiler_high_param_context_window", 10000)
    monkeypatch.setattr(system_config, "compiler_high_param_max_output_tokens", 1000)
    submitter = HighParamSubmitter(
        model_name="test-model",
        user_prompt="prove something",
        validator_context_window=10000,
        validator_max_tokens=1000,
    )

    async def no_op_cache(*args, **kwargs):
        return None

    async def raise_wrapped_transient(*args, **kwargs):
        raise RuntimeError("TRANSIENT PROVIDER ERROR: provider connection failed before usable proof output.")

    monkeypatch.setattr(
        "backend.compiler.agents.high_param_submitter.lm_studio_client.cache_model_load_config",
        no_op_cache,
    )
    monkeypatch.setattr(
        "backend.compiler.agents.high_param_submitter.api_client_manager.generate_completion",
        raise_wrapped_transient,
    )

    with pytest.raises(RuntimeError, match="TRANSIENT PROVIDER ERROR"):
        await submitter._call_llm_and_parse(prompt="{}", task_label="rigor_discovery")
