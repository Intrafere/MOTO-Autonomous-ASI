"""Provider context retries preserve whole feedback and locally fitted context."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.autonomous.agents import proof_formalization_agent as module
from backend.shared.models import ProofAttemptFeedback, ProofCandidate
from backend.shared.provider_errors import ProviderContextLengthError, ProviderRouteIdentity


@pytest.mark.asyncio
@pytest.mark.parametrize("tactic", [False, True])
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("omit_pack", [False, True])
@pytest.mark.parametrize("always_reject", [False, True])
async def test_provider_feedback_suffix_retries(monkeypatch, tactic, compact, omit_pack, always_reject):
    history = [
        ProofAttemptFeedback(
            attempt=i, theorem_id="target", error_output=f"WHOLE_FEEDBACK_{i}",
            failure_kind="output_truncated" if compact else "lean_rejected",
        )
        for i in range(1, 4)
    ]
    original = [entry.model_dump() for entry in history]
    def builder(**kwargs):
        return json.dumps({
            "source": kwargs["full_source_content"],
            "pack": kwargs["retrieved_proofs_context"],
            "feedback": [entry.error_output for entry in kwargs["prior_attempts"]],
        })

    builder_name = (
        "build_compact_proof_tactic_script_prompt" if compact and tactic else
        "build_compact_proof_formalization_prompt" if compact else
        "build_proof_tactic_script_prompt" if tactic else "build_proof_formalization_prompt"
    )
    monkeypatch.setattr(module, builder_name, builder)
    pack = "PACK" * (1000 if omit_pack else 1)
    monkeypatch.setattr(module, "_latest_assistant_pack_for_lean_attempts", lambda: ("", pack, []))
    monkeypatch.setattr(type(module.rag_config), "get_available_input_tokens", lambda *args: 500)
    monkeypatch.setattr(module, "count_tokens", len)
    monkeypatch.setattr("backend.shared.prompt_feedback_budget.count_tokens", len)
    route = ProviderRouteIdentity(provider="openrouter", model="test-model")
    sent = []

    async def generate(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        assert len(prompt) <= 500
        sent.append((json.loads(prompt), kwargs))
        if always_reject or len(sent) < 3:
            raise ProviderContextLengthError("context_length_exceeded", route=route)
        return {"choices": [{"message": {"content": json.dumps({
            "theorem_name": "target", "lean_code": "theorem target : True := by trivial",
            "theorem_header": "theorem target : True", "tactics": ["trivial"],
            "reasoning": "Done",
        })}}]}

    monkeypatch.setattr(module.api_client_manager, "generate_completion", generate)
    lean = SimpleNamespace(success=True, error_output="", goal_states="", tactic_error_slice="")
    checker = SimpleNamespace(check_proof=AsyncMock(return_value=lean), check_tactic_script=AsyncMock(return_value=lean))
    monkeypatch.setattr(module, "get_lean4_client", lambda: checker)
    agent = module.ProofFormalizationAgent(model_id="test-model", context_window=32000, max_output_tokens=2000, role_id="proof-test")
    callback = AsyncMock()
    method = agent.prove_candidate_tactic_script if tactic else agent.prove_candidate
    success, _, _, attempts = await method(
        user_research_prompt="Prove the target", source_type="paper",
        theorem_candidate=ProofCandidate(theorem_id="target", statement="True"),
        source_content="MANDATORY SOURCE", prior_attempts=history, max_attempts=1,
        attempt_callback=callback,
    )
    assert success is not always_reject
    assert [item[0]["feedback"] for item in sent] == [
        ["WHOLE_FEEDBACK_1", "WHOLE_FEEDBACK_2", "WHOLE_FEEDBACK_3"],
        ["WHOLE_FEEDBACK_2", "WHOLE_FEEDBACK_3"],
        ["WHOLE_FEEDBACK_3"],
    ]
    expected_pack = module._PROOF_SEARCH_CONTEXT_OMITTED if omit_pack else pack
    assert all(item[0]["pack"] == expected_pack for item in sent)
    assert all(item[0]["source"] == "MANDATORY SOURCE" for item in sent)
    assert len({item[1]["task_id"] for item in sent}) == 1
    assert all(item[1]["_moto_disable_supercharge"] == compact for item in sent)
    assert [entry.model_dump() for entry in history] == original
    assert len(attempts) == 4
    assert attempts[-1].attempt == 4
    callback.assert_awaited_once()
    if always_reject:
        checker.check_proof.assert_not_awaited()
        checker.check_tactic_script.assert_not_awaited()
