import asyncio
from unittest.mock import AsyncMock
from types import SimpleNamespace

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
from backend.autonomous.agents.proof_candidate_list_validator import (
    ProofCandidateListValidator,
)
from backend.autonomous.core import proof_verification_stage as stage_module
from backend.shared.lean_proof_integrity import LeanProofIntegrityResult
from backend.compiler.core.compiler_coordinator import CompilerCoordinator
from backend.shared.models import (
    ProofAttemptFeedback,
    ProofCandidate,
    ProofCandidateListValidation,
    ProofCandidateNoveltyDecision,
)
from backend.shared.provider_errors import ProviderContextLengthError, ProviderRouteIdentity


def _candidate() -> ProofCandidate:
    return ProofCandidate(theorem_id="candidate-1", statement="True")


@pytest.fixture(autouse=True)
def _approve_candidate_lists(monkeypatch):
    async def approve(_self, *, candidates, **_kwargs):
        return ProofCandidateListValidation(
            results=[
                ProofCandidateNoveltyDecision(
                    theorem_id=candidate.theorem_id,
                    decision="approve_novel",
                    reasoning="Approved for this proof-stage regression fixture.",
                )
                for candidate in candidates
            ],
            feedback="All fixture candidates are approved.",
        )

    monkeypatch.setattr(ProofCandidateListValidator, "validate", approve)


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
async def test_overflow_pressure_callback_uses_run_id_without_candidate_error(monkeypatch):
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

    class FakeProofDatabase:
        def __init__(self):
            self.requested_run_ids = []

        def get_novel_proofs_for_injection(self, run_id=""):
            self.requested_run_ids.append(run_id)
            return "prior proof context"

        async def get_proof_set_revision(self):
            return 42

    monkeypatch.setattr(ProofFormalizationAgent, "prove_candidate", fake_prove)
    stage = ProofVerificationStage()
    monkeypatch.setattr(stage, "_prepare_candidate", AsyncMock(return_value=_candidate()))
    monkeypatch.setattr(stage, "_run_smt_check", AsyncMock(return_value=None))
    broadcast = AsyncMock()
    proof_db = FakeProofDatabase()
    pressures = []

    async def pressure_callback(pressure, **kwargs):
        pressures.append((pressure, kwargs))

    await stage._run_lean_pipeline_for_candidate(
        theorem_candidate=_candidate(),
        base_event={"source_type": "paper", "source_id": "paper-1", "run_id": "run-123"},
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
        novel_proofs_db=proof_db,
        broadcast_fn=broadcast,
        proof_pruning_pressure_callback=pressure_callback,
        run_id="run-123",
    )

    assert proof_db.requested_run_ids == ["run-123"]
    assert len(pressures) == 1
    pressure, kwargs = pressures[0]
    assert pressure.proof_set_revision == 42
    assert kwargs["proof_set_revision"] == 42


def _overflow_feedback(theorem_id: str, attempt: int = 1) -> ProofAttemptFeedback:
    return ProofAttemptFeedback(
        attempt=attempt,
        theorem_id=theorem_id,
        error_output="MANDATORY FULL SOURCE CONTEXT OVERFLOW",
        overflow_origin="local_preflight",
        prompt_tokens=11_000,
        max_input_tokens=10_000,
    )


def _lean_failure(theorem_id: str, attempt: int = 1) -> ProofAttemptFeedback:
    return ProofAttemptFeedback(
        attempt=attempt,
        theorem_id=theorem_id,
        lean_code="theorem t : False := by simp",
        error_output="t.lean:1:0: error: simp made no progress",
    )


def _failed_outcome(kwargs, attempts) -> _LeanVerificationOutcome:
    overflowed = bool(attempts) and ProofFormalizationAgent.is_context_overflow_feedback(attempts[-1])
    return _LeanVerificationOutcome(
        candidate=kwargs["theorem_candidate"],
        proof_label=kwargs["proof_label"],
        success=False,
        theorem_name="",
        lean_code="",
        attempts=list(attempts),
        context_overflow_payload=(
            {
                "role_id": "autonomous_proof_formalization_paper",
                "configured_model": "model",
                "configured_provider": "openrouter",
            }
            if overflowed
            else {}
        ),
    )


async def _run_overflow_stage(
    monkeypatch,
    candidates,
    pipeline,
    *,
    source_type="paper",
    proof_run_context=None,
    novel_proofs_db=None,
):
    async def fake_resolve(**kwargs):
        return list(candidates)

    stage = ProofVerificationStage()
    monkeypatch.setattr(stage, "_resolve_candidates", fake_resolve)
    monkeypatch.setattr(stage, "_run_lean_pipeline_for_candidate", pipeline)
    monkeypatch.setattr(formalization_module.system_config, "lean4_enabled", True)
    checkpoints = []

    async def save_checkpoint(payload):
        checkpoints.append(payload)

    result = await stage.run(
        content="Source",
        source_type=source_type,
        source_id="source-1",
        user_prompt="Prompt",
        submitter_model="model",
        submitter_context=20_000,
        submitter_max_tokens=2_000,
        validator_model="validator",
        validator_context=20_000,
        validator_max_tokens=2_000,
        broadcast_fn=AsyncMock(),
        novel_proofs_db=novel_proofs_db or AsyncMock(),
        checkpoint_callback=save_checkpoint,
        proof_run_context=proof_run_context,
    )
    return result, checkpoints


_AUTONOMOUS_OWNER_CONTEXT = {
    "scope": "autonomous",
    "owner": stage_module.AUTONOMOUS_WORKFLOW_PROOF_OWNER,
    "proof_run_id": "autonomous-run",
    "lifecycle_generation": 3,
}


@pytest.mark.asyncio
async def test_overflow_candidate_settles_as_open_target_while_sibling_continues(monkeypatch):
    candidates = [
        ProofCandidate(theorem_id="overflow", statement="True"),
        ProofCandidate(theorem_id="sibling", statement="False"),
    ]
    calls = []

    async def pipeline(**kwargs):
        theorem_id = kwargs["theorem_candidate"].theorem_id
        calls.append(theorem_id)
        attempts = (
            [_overflow_feedback("overflow")]
            if theorem_id == "overflow"
            else [_lean_failure("sibling")]
        )
        await kwargs["attempt_checkpoint_callback"](kwargs["theorem_candidate"], attempts)
        return _failed_outcome(kwargs, attempts)

    proof_db = AsyncMock()
    result, checkpoints = await _run_overflow_stage(
        monkeypatch,
        candidates,
        pipeline,
        source_type="brainstorm",
        novel_proofs_db=proof_db,
    )

    assert set(calls) == {"overflow", "sibling"}
    assert result.had_error is False
    assert result.context_overflow_blocked is False
    assert result.fatal_stop_reason == ""
    assert result.deferred_candidate_ids == ["overflow"]
    assert [item.theorem_id for item in result.results] == ["sibling", "overflow"]
    recorded = [
        call.args[1].theorem_id
        for call in proof_db.record_failed_candidate.await_args_list
    ]
    assert recorded == ["sibling", "overflow"]
    assert checkpoints[-1]["status"] == "complete"
    assert sorted(checkpoints[-1]["processed_candidate_ids"]) == ["overflow", "sibling"]


@pytest.mark.asyncio
async def test_autonomous_overflow_lets_running_sibling_finish(monkeypatch):
    candidates = [
        ProofCandidate(theorem_id="overflow", statement="True"),
        ProofCandidate(theorem_id="sibling", statement="False"),
    ]
    overflow_returned = asyncio.Event()
    sibling_cancelled = asyncio.Event()

    async def pipeline(**kwargs):
        theorem_id = kwargs["theorem_candidate"].theorem_id
        if theorem_id == "overflow":
            overflow_returned.set()
            return _failed_outcome(kwargs, [_overflow_feedback("overflow")])
        try:
            await overflow_returned.wait()
            # Let the overflow outcome be fully processed while this sibling runs.
            for _ in range(20):
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            sibling_cancelled.set()
            raise
        return _failed_outcome(kwargs, [_lean_failure("sibling")])

    result, checkpoints = await _run_overflow_stage(
        monkeypatch,
        candidates,
        pipeline,
        proof_run_context=dict(_AUTONOMOUS_OWNER_CONTEXT),
    )

    assert not sibling_cancelled.is_set()
    assert result.fatal_stop_reason == ""
    assert result.context_overflow_blocked is False
    assert [item.theorem_id for item in result.results] == ["sibling", "overflow"]
    assert checkpoints[-1]["status"] == "complete"


@pytest.mark.asyncio
async def test_overflow_does_not_block_queued_later_batch_candidate(monkeypatch):
    candidates = [
        ProofCandidate(theorem_id="overflow", statement="True"),
        ProofCandidate(theorem_id="queued", statement="False"),
    ]
    calls = []

    async def pipeline(**kwargs):
        theorem_id = kwargs["theorem_candidate"].theorem_id
        calls.append(theorem_id)
        if theorem_id == "overflow":
            return _failed_outcome(kwargs, [_overflow_feedback("overflow")])
        return _failed_outcome(kwargs, [_lean_failure("queued")])

    monkeypatch.setattr(stage_module.system_config, "proof_max_parallel_candidates", 1)
    result, _checkpoints = await _run_overflow_stage(
        monkeypatch,
        candidates,
        pipeline,
        proof_run_context=dict(_AUTONOMOUS_OWNER_CONTEXT),
    )

    assert calls == ["overflow", "queued"]
    assert result.fatal_stop_reason == ""
    assert result.context_overflow_blocked is False


@pytest.mark.asyncio
@pytest.mark.parametrize("owned_by_autonomous_workflow", [True, False])
async def test_round_is_blocked_only_when_no_candidate_reached_the_model(
    monkeypatch,
    owned_by_autonomous_workflow,
):
    candidates = [
        ProofCandidate(theorem_id="first", statement="True"),
        ProofCandidate(theorem_id="second", statement="False"),
    ]

    async def pipeline(**kwargs):
        theorem_id = kwargs["theorem_candidate"].theorem_id
        return _failed_outcome(kwargs, [_overflow_feedback(theorem_id)])

    context = (
        dict(_AUTONOMOUS_OWNER_CONTEXT)
        if owned_by_autonomous_workflow
        # A manual check of an Autonomous source shares the storage scope only.
        else {"scope": "autonomous", "run_mode": "loop_with_pruning"}
    )
    proof_db = AsyncMock()
    result, checkpoints = await _run_overflow_stage(
        monkeypatch,
        candidates,
        pipeline,
        source_type="brainstorm",
        proof_run_context=context,
        novel_proofs_db=proof_db,
    )

    assert result.context_overflow_blocked is True
    assert sorted(result.deferred_candidate_ids) == ["first", "second"]
    assert result.results == []
    proof_db.record_failed_candidate.assert_not_awaited()
    assert checkpoints[-1]["status"] == "fatal_context_overflow"
    if owned_by_autonomous_workflow:
        assert result.fatal_stop_reason == "context_overflow"
        assert result.fatal_stop_payload["fatal"] is True
        assert result.fatal_stop_payload["notification_kind"] == "model_error"
        assert checkpoints[-1]["fatal_stop_reason"] == "context_overflow"
    else:
        assert result.fatal_stop_reason == ""


@pytest.mark.asyncio
async def test_overflow_after_real_attempts_does_not_block_the_round(monkeypatch):
    candidates = [ProofCandidate(theorem_id="grown", statement="True")]

    async def pipeline(**kwargs):
        attempts = [_lean_failure("grown", 1), _overflow_feedback("grown", 2)]
        await kwargs["attempt_checkpoint_callback"](kwargs["theorem_candidate"], attempts)
        return _failed_outcome(kwargs, attempts)

    result, checkpoints = await _run_overflow_stage(
        monkeypatch,
        candidates,
        pipeline,
        proof_run_context=dict(_AUTONOMOUS_OWNER_CONTEXT),
    )

    assert result.context_overflow_blocked is False
    assert result.fatal_stop_reason == ""
    settled = result.results[0]
    assert settled.theorem_id == "grown"
    assert settled.attempts_used == 1
    assert "simp made no progress" in settled.error_summary
    assert checkpoints[-1]["status"] == "complete"


@pytest.mark.asyncio
async def test_manual_novelty_provider_failure_preserves_lean_verified_artifact(monkeypatch):
    candidate = ProofCandidate(
        theorem_id="verified-before-novelty",
        statement="True",
        formal_sketch="Trivial verified theorem.",
    )
    lean_code = "import Mathlib\n\ntheorem verified_before_novelty : True := by trivial\n"
    feedback = ProofAttemptFeedback(
        attempt=1,
        theorem_id=candidate.theorem_id,
        lean_code=lean_code,
        success=True,
    )
    events = []
    classifications = []

    async def fake_resolve(**_kwargs):
        return [candidate]

    async def fake_pipeline(**kwargs):
        return _LeanVerificationOutcome(
            candidate=candidate,
            proof_label=kwargs["proof_label"],
            success=True,
            theorem_name="verified_before_novelty",
            lean_code=lean_code,
            attempts=[feedback],
        )

    async def fake_integrity(**_kwargs):
        return LeanProofIntegrityResult(
            valid=True,
            actual_theorem_statement="True",
            actual_theorem_name="verified_before_novelty",
        )

    class Stored:
        proof_id = "proof-preserved"
        novel = False
        novelty_tier = "not_novel"
        novelty_reasoning = "classification unavailable"

    async def fake_register(**kwargs):
        classifications.append(kwargs.get("novelty_classification"))
        if kwargs.get("novelty_classification") is None:
            raise stage_module.RetryableProviderError(
                provider="openrouter",
                provider_label="OpenRouter",
                role_id="autonomous_proof_novelty",
                model="validator",
                reason="transient_provider_error",
                message="upstream timeout",
            )
        return SimpleNamespace(record=Stored(), duplicate=False)

    async def broadcast(event_type, payload):
        events.append((event_type, payload))

    old_enabled = stage_module.system_config.lean4_enabled
    stage_module.system_config.lean4_enabled = True
    stage = ProofVerificationStage()
    monkeypatch.setattr(stage, "_resolve_candidates", fake_resolve)
    monkeypatch.setattr(stage, "_run_lean_pipeline_for_candidate", fake_pipeline)
    monkeypatch.setattr(stage_module, "validate_full_lean_proof_integrity", fake_integrity)
    monkeypatch.setattr(stage_module, "register_verified_lean_proof", fake_register)
    monkeypatch.setattr(stage._dependency_extractor, "extract_dependencies", AsyncMock(return_value=[]))
    try:
        result = await stage.run(
            content="Source",
            source_type="brainstorm",
            source_id="topic-1",
            user_prompt="Prove True",
            submitter_model="model",
            submitter_context=20_000,
            submitter_max_tokens=2_000,
            validator_model="validator",
            validator_context=20_000,
            validator_max_tokens=2_000,
            broadcast_fn=broadcast,
            novel_proofs_db=AsyncMock(),
            theorem_candidates=[candidate],
            trigger="manual",
            append_to_source=False,
        )
    finally:
        stage_module.system_config.lean4_enabled = old_enabled

    assert result.verified_count == 1
    assert result.novel_count == 0
    assert result.results[0].lean_code == lean_code
    assert result.results[0].proof_id == "proof-preserved"
    assert classifications[0] is None
    assert classifications[1][0] == "not_novel"
    assert "transient failure" in classifications[1][1]
    assert any(event == "proof_verified" for event, _payload in events)
    completion = next(payload for event, payload in events if event == "proof_check_complete")
    assert completion["verified_count"] == 1


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
