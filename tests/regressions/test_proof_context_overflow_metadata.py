from unittest.mock import AsyncMock

import pytest

from backend.autonomous.agents import proof_formalization_agent as formalization_module
from backend.autonomous.agents.proof_formalization_agent import (
    ProofFormalizationAgent,
    ProofFormalizationContextOverflowError,
)
from backend.autonomous.core.proof_verification_stage import (
    ProofVerificationStage,
    _LeanVerificationOutcome,
)
from backend.compiler.core.compiler_coordinator import CompilerCoordinator
from backend.shared.models import ProofAttemptFeedback, ProofCandidate
from backend.shared.provider_errors import ProviderContextLengthError, ProviderRouteIdentity


def _candidate() -> ProofCandidate:
    return ProofCandidate(theorem_id="candidate-1", statement="True")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "strategy"),
    [
        ("prove_candidate", "full_script"),
        ("prove_candidate_tactic_script", "tactic_script"),
    ],
)
async def test_typed_provider_overflow_route_survives_formalization(
    monkeypatch,
    method_name,
    strategy,
):
    route = ProviderRouteIdentity(
        provider="lm_studio",
        model="fallback-model:2",
        configured_model="configured/model",
        configured_provider="openrouter",
    )
    monkeypatch.setattr(
        formalization_module.api_client_manager,
        "generate_completion",
        AsyncMock(
            side_effect=ProviderContextLengthError(
                "context_length_exceeded",
                route=route,
            )
        ),
    )

    agent = ProofFormalizationAgent(
        model_id="configured/model",
        context_window=32_000,
        max_output_tokens=2_000,
        role_id="autonomous_proof_formalization_paper",
    )
    method = getattr(agent, method_name)
    success, _, _, attempts = await method(
        user_research_prompt="Prove the claim.",
        source_type="paper",
        theorem_candidate=_candidate(),
        source_content="A short source.",
        max_attempts=1,
    )

    assert success is False
    assert len(attempts) == 1
    feedback = attempts[0]
    assert feedback.strategy == strategy
    assert feedback.configured_model == "configured/model"
    assert feedback.configured_provider == "openrouter"
    assert feedback.effective_model == "fallback-model:2"
    assert feedback.effective_provider == "lm_studio"
    assert feedback.overflow_origin == "provider"
    assert feedback.prompt_tokens is None
    assert feedback.max_input_tokens is None
    assert "context_length_exceeded" in feedback.error_output


def test_proof_overflow_workflow_scope_is_explicit():
    assert ProofVerificationStage._proof_workflow_mode("automatic") == "autonomous"
    assert ProofVerificationStage._proof_workflow_mode("manual") == "manual_proof_check"
    assert ProofVerificationStage._proof_workflow_mode("manual_compiler_save") == "compiler"
    assert ProofVerificationStage._proof_workflow_mode("manual_compiler_aggregator") == "aggregator"


@pytest.mark.asyncio
async def test_proof_overflow_event_is_scoped_nonfatal_and_uses_effective_route(monkeypatch):
    feedback = ProofAttemptFeedback(
        attempt=1,
        theorem_id="candidate-1",
        error_output="MANDATORY FULL SOURCE CONTEXT OVERFLOW",
        configured_model="configured/model",
        configured_provider="openrouter",
        effective_model="fallback-model:2",
        effective_provider="lm_studio",
        overflow_origin="provider",
    )

    async def fake_prove(self, *args, attempt_callback=None, **kwargs):
        await attempt_callback(feedback)
        return False, "", "", [feedback]

    monkeypatch.setattr(ProofFormalizationAgent, "prove_candidate", fake_prove)
    stage = ProofVerificationStage()
    monkeypatch.setattr(stage, "_prepare_candidate", AsyncMock(return_value=_candidate()))
    monkeypatch.setattr(stage, "_run_smt_check", AsyncMock(return_value=None))
    broadcast = AsyncMock()

    await stage._run_lean_pipeline_for_candidate(
        theorem_candidate=_candidate(),
        base_event={"source_type": "paper", "source_id": "paper-1"},
        proof_label="A",
        user_prompt="Prove the claim.",
        source_type="paper",
        source_id="paper-1",
        source_content="A short source.",
        source_title="Paper",
        submitter_model="configured/model",
        submitter_context=32_000,
        submitter_max_tokens=2_000,
        role_suffix="paper",
        trigger="manual_compiler_save",
        novel_proofs_db=None,
        broadcast_fn=broadcast,
    )

    overflow_calls = [
        call for call in broadcast.await_args_list if call.args[0] == "proof_context_overflow"
    ]
    assert len(overflow_calls) == 1
    payload = overflow_calls[0].args[1]
    assert payload["workflow_mode"] == "compiler"
    assert payload["fatal"] is False
    assert payload["effective_model"] == "fallback-model:2"
    assert payload["effective_provider"] == "lm_studio"
    assert payload["overflow_origin"] == "provider"
    assert "provider rejected" in payload["message"].lower()
    assert not any(call.args[0] == "proof_attempt_failed" for call in broadcast.await_args_list)


@pytest.mark.asyncio
async def test_local_preflight_overflow_reports_token_budget(monkeypatch):
    feedback = ProofAttemptFeedback(
        attempt=1,
        theorem_id="candidate-1",
        error_output="MANDATORY FULL SOURCE CONTEXT OVERFLOW",
        overflow_origin="local_preflight",
        prompt_tokens=12_345,
        max_input_tokens=10_000,
    )

    async def fake_prove(self, *args, attempt_callback=None, **kwargs):
        await attempt_callback(feedback)
        return False, "", "", [feedback]

    monkeypatch.setattr(ProofFormalizationAgent, "prove_candidate", fake_prove)
    stage = ProofVerificationStage()
    monkeypatch.setattr(stage, "_prepare_candidate", AsyncMock(return_value=_candidate()))
    monkeypatch.setattr(stage, "_run_smt_check", AsyncMock(return_value=None))
    broadcast = AsyncMock()

    await stage._run_lean_pipeline_for_candidate(
        theorem_candidate=_candidate(),
        base_event={"source_type": "paper", "source_id": "paper-1"},
        proof_label="A",
        user_prompt="Prove the claim.",
        source_type="paper",
        source_id="paper-1",
        source_content="A short source.",
        source_title="Paper",
        submitter_model="configured/model",
        submitter_context=32_000,
        submitter_max_tokens=2_000,
        role_suffix="paper",
        trigger="automatic",
        novel_proofs_db=None,
        broadcast_fn=broadcast,
    )

    overflow_call = next(
        call for call in broadcast.await_args_list if call.args[0] == "proof_context_overflow"
    )
    payload = overflow_call.args[1]
    assert payload["overflow_origin"] == "local_preflight"
    assert payload["prompt_tokens"] == 12_345
    assert payload["max_input_tokens"] == 10_000
    assert "12,345" in payload["message"]
    assert "10,000" in payload["message"]
    assert "before provider invocation" in payload["message"]


@pytest.mark.asyncio
async def test_overflow_candidate_is_deferred_while_sibling_continues(monkeypatch):
    overflow_candidate = ProofCandidate(theorem_id="overflow", statement="True")
    sibling_candidate = ProofCandidate(theorem_id="sibling", statement="False")
    overflow_feedback = ProofAttemptFeedback(
        attempt=1,
        theorem_id="overflow",
        error_output="MANDATORY FULL SOURCE CONTEXT OVERFLOW",
        overflow_origin="local_preflight",
        prompt_tokens=11_000,
        max_input_tokens=10_000,
    )
    sibling_feedback = ProofAttemptFeedback(
        attempt=1,
        theorem_id="sibling",
        error_output="ordinary Lean failure",
    )
    calls = []

    async def fake_resolve(**kwargs):
        return [overflow_candidate, sibling_candidate]

    async def fake_pipeline(**kwargs):
        candidate = kwargs["theorem_candidate"]
        calls.append(candidate.theorem_id)
        feedback = overflow_feedback if candidate.theorem_id == "overflow" else sibling_feedback
        await kwargs["attempt_checkpoint_callback"](candidate, [feedback])
        return _LeanVerificationOutcome(
            candidate=candidate,
            proof_label=kwargs["proof_label"],
            success=False,
            theorem_name="",
            lean_code="",
            attempts=[feedback],
        )

    old_enabled = formalization_module.system_config.lean4_enabled
    formalization_module.system_config.lean4_enabled = True
    stage = ProofVerificationStage()
    monkeypatch.setattr(stage, "_resolve_candidates", fake_resolve)
    monkeypatch.setattr(stage, "_run_lean_pipeline_for_candidate", fake_pipeline)
    checkpoints = []

    async def save_checkpoint(payload):
        checkpoints.append(payload)

    try:
        result = await stage.run(
            content="Source",
            source_type="paper",
            source_id="paper-1",
            user_prompt="Prompt",
            submitter_model="model",
            submitter_context=20_000,
            submitter_max_tokens=2_000,
            validator_model="validator",
            validator_context=20_000,
            validator_max_tokens=2_000,
            broadcast_fn=AsyncMock(),
            novel_proofs_db=AsyncMock(),
            checkpoint_callback=save_checkpoint,
        )
    finally:
        formalization_module.system_config.lean4_enabled = old_enabled

    assert set(calls) == {"overflow", "sibling"}
    assert result.had_error is False
    assert result.deferred_candidate_ids == ["overflow"]
    assert [item.theorem_id for item in result.results] == ["sibling"]
    assert checkpoints[-1]["status"] == "deferred"
    assert checkpoints[-1]["processed_candidate_ids"] == ["sibling"]


@pytest.mark.asyncio
async def test_compiler_context_overflow_preserves_feedback_route():
    coordinator = object.__new__(CompilerCoordinator)
    coordinator.autonomous_mode = False
    coordinator.current_mode = "rigor"
    coordinator.is_running = True
    coordinator._broadcast = AsyncMock()

    feedback = ProofAttemptFeedback(
        attempt=1,
        theorem_id="candidate-1",
        error_output="MANDATORY FULL SOURCE CONTEXT OVERFLOW",
        configured_model="configured/model",
        configured_provider="openrouter",
        effective_model="fallback-model:2",
        effective_provider="lm_studio",
    )
    error = ProofFormalizationContextOverflowError(feedback)

    await coordinator._handle_context_overflow(error, role_id="compiler_rigor")

    event_name, payload = coordinator._broadcast.await_args.args
    assert event_name == "context_overflow_error"
    assert payload["configured_model"] == "configured/model"
    assert payload["configured_provider"] == "openrouter"
    assert payload["effective_model"] == "fallback-model:2"
    assert payload["effective_provider"] == "lm_studio"
