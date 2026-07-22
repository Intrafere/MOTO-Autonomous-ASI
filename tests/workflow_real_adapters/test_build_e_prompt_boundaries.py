from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.autonomous.agents import proof_formalization_agent as formalization_module
from backend.autonomous.agents.proof_formalization_agent import ProofFormalizationAgent
from backend.autonomous.memory.paper_library import PaperLibrary
from backend.compiler.agents import high_param_submitter as high_param_module
from backend.compiler.agents.high_param_submitter import HighParamSubmitter
from backend.compiler.memory.paper_memory import (
    THEOREMS_APPENDIX_END,
    THEOREMS_APPENDIX_START,
)
from backend.shared.models import ProofCandidate


class _ProofStore:
    def __init__(self, proofs=()):
        self._proofs = list(proofs)

    async def get_all_proofs(self):
        return list(self._proofs)

    async def get_recent_failure_hints(self, *_args, **_kwargs):
        return []


def _completion(content: str) -> dict:
    return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}


@pytest.mark.asyncio
async def test_real_proof_formalization_message_keeps_source_boundaries_and_verified_library(
    monkeypatch,
):
    source_head = "ORDINARY SOURCE HEAD SENTINEL"
    source_tail = "ORDINARY SOURCE TAIL SENTINEL"
    generated_proof = "GENERATED PAPER PROOF MUST NOT BE MODEL VISIBLE"
    raw_source = (
        f"{source_head}\n"
        + ("ordinary mathematical source material\n" * 80)
        + f"\n{THEOREMS_APPENDIX_START}\n"
        + generated_proof
        + f"\n{THEOREMS_APPENDIX_END}\n"
        + source_tail
    )
    # Real source readers apply this boundary before formalization.
    source_for_model = PaperLibrary.strip_verified_proofs_from_content(raw_source)

    verified_library = "SEPARATELY INJECTED VERIFIED PROOF LIBRARY SENTINEL"
    captured_messages = []

    async def fake_completion(**kwargs):
        captured_messages.extend(kwargs["messages"])
        return _completion(
            '{"theorem_name":"boundary_test","lean_code":'
            '"import Mathlib\\n\\ntheorem boundary_test : True := by trivial",'
            '"reasoning":"test"}'
        )

    async def fake_check(_code, **_kwargs):
        return SimpleNamespace(success=True, error_output="", goal_states="")

    monkeypatch.setattr(
        formalization_module,
        "_latest_assistant_pack_for_lean_attempts",
        lambda: ("pack-hash", verified_library, []),
    )
    monkeypatch.setattr(
        formalization_module.api_client_manager,
        "generate_completion",
        fake_completion,
    )
    monkeypatch.setattr(
        formalization_module,
        "get_lean4_client",
        lambda: SimpleNamespace(check_proof=fake_check),
    )
    monkeypatch.setattr(
        formalization_module.assistant_proof_search_coordinator,
        "mark_pack_consumed_by_solver",
        lambda *_args, **_kwargs: None,
    )

    agent = ProofFormalizationAgent(
        model_id="fake-model",
        context_window=16_000,
        max_output_tokens=1_000,
        role_id="autonomous_proof_paper",
    )
    success, _, _, _ = await agent.prove_candidate(
        user_research_prompt="Prove the prompt.",
        source_type="paper",
        theorem_candidate=ProofCandidate(
            theorem_id="boundary-test",
            statement="True",
            formal_sketch="By triviality.",
            expected_novelty_tier="novel_variant",
            prompt_relevance_rationale="Directly tests the requested claim.",
            novelty_rationale="Boundary test.",
            why_not_standard_known_result="Test fixture candidate.",
        ),
        source_content=source_for_model,
        max_attempts=1,
        source_title="Boundary Paper",
    )

    assert success is True
    assert len(captured_messages) == 1
    model_visible = captured_messages[0]["content"]
    assert source_head in model_visible
    assert source_tail in model_visible
    assert generated_proof not in model_visible
    assert verified_library in model_visible


@pytest.mark.asyncio
async def test_real_compiler_rigor_message_strips_theorem_appendix_and_keeps_proof_summary(
    monkeypatch,
):
    paper_head = "ORDINARY PAPER HEAD SENTINEL"
    paper_tail = "ORDINARY PAPER TAIL SENTINEL"
    generated_theorem = "GENERATED THEOREM APPENDIX MUST NOT BE MODEL VISIBLE"
    paper = (
        f"{paper_head}\n"
        + ("ordinary paper body\n" * 40)
        + f"{THEOREMS_APPENDIX_START}\n"
        + generated_theorem
        + f"\n{THEOREMS_APPENDIX_END}\n"
        + f"{paper_tail}"
    )
    verified_summary = "SEPARATELY INJECTED VERIFIED RIGOR PROOF SENTINEL"
    proof_record = SimpleNamespace(
        proof_id="proof-boundary",
        novel=True,
        theorem_statement=verified_summary,
    )
    captured_messages = []

    async def fake_completion(**kwargs):
        captured_messages.extend(kwargs["messages"])
        return _completion('{"needs_theorem_work":false,"reasoning":"test"}')

    monkeypatch.setattr(high_param_module.outline_memory, "get_outline", AsyncMock(return_value="I. Body"))
    monkeypatch.setattr(high_param_module.paper_memory, "get_paper", AsyncMock(return_value=paper))
    monkeypatch.setattr(
        high_param_module.compiler_rag_manager,
        "retrieve_for_mode",
        AsyncMock(return_value=SimpleNamespace(text="")),
    )
    monkeypatch.setattr(
        high_param_module.api_client_manager,
        "generate_completion",
        fake_completion,
    )
    monkeypatch.setattr(
        high_param_module.lm_studio_client,
        "cache_model_load_config",
        AsyncMock(),
    )
    monkeypatch.setattr(high_param_module.system_config, "compiler_high_param_context_window", 16_000)
    monkeypatch.setattr(high_param_module.system_config, "compiler_high_param_max_output_tokens", 1_000)

    submitter = HighParamSubmitter(
        "fake-model",
        "Advance the target problem.",
        validator_context_window=8_000,
        validator_max_tokens=500,
        proof_database_store=_ProofStore([proof_record]),
    )
    submitter.context_window = 16_000
    submitter.max_output_tokens = 1_000
    submitter.available_input_tokens = 14_000
    submitter.set_source_material_context(
        "SOURCE MATERIAL HEAD\n" + ("source body\n" * 30) + "SOURCE MATERIAL TAIL",
        "Source brainstorm",
    )

    result = await submitter._step_discovery()

    assert result is None
    assert len(captured_messages) == 1
    model_visible = captured_messages[0]["content"]
    assert paper_head in model_visible
    assert paper_tail in model_visible
    assert "SOURCE MATERIAL HEAD" in model_visible
    assert "SOURCE MATERIAL TAIL" in model_visible
    assert generated_theorem not in model_visible
    assert verified_summary in model_visible


@pytest.mark.asyncio
async def test_real_compiler_rigor_mandatory_full_source_overflow_is_visible_and_pre_llm(
    monkeypatch,
):
    source = "MANDATORY SOURCE HEAD\n" + ("large mandatory source token " * 5_000) + "\nMANDATORY SOURCE TAIL"
    paper = "CURRENT PAPER HEAD\nordinary paper body\nCURRENT PAPER TAIL"
    model_call = AsyncMock()
    rag_call = AsyncMock()

    monkeypatch.setattr(high_param_module.outline_memory, "get_outline", AsyncMock(return_value="I. Body"))
    monkeypatch.setattr(high_param_module.paper_memory, "get_paper", AsyncMock(return_value=paper))
    monkeypatch.setattr(high_param_module.api_client_manager, "generate_completion", model_call)
    monkeypatch.setattr(high_param_module.compiler_rag_manager, "retrieve_for_mode", rag_call)
    monkeypatch.setattr(high_param_module.system_config, "compiler_high_param_context_window", 2_000)
    monkeypatch.setattr(high_param_module.system_config, "compiler_high_param_max_output_tokens", 200)

    submitter = HighParamSubmitter(
        "fake-model",
        "Advance the target problem.",
        validator_context_window=2_000,
        validator_max_tokens=200,
        proof_database_store=_ProofStore(),
    )
    submitter.context_window = 2_000
    submitter.max_output_tokens = 200
    submitter.available_input_tokens = 800
    submitter.set_source_material_context(source, "Mandatory source brainstorm")

    with pytest.raises(ValueError, match="mandatory full source context"):
        await submitter._step_discovery()

    model_call.assert_not_awaited()
    rag_call.assert_not_awaited()
    assert submitter._source_material_context == source
    assert submitter._source_material_context.startswith("MANDATORY SOURCE HEAD")
    assert submitter._source_material_context.endswith("MANDATORY SOURCE TAIL")
