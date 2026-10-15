"""A broken Lean workspace stops workflows instead of looping on unchecked proofs."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from backend.autonomous.agents import proof_formalization_agent as formalization_module
from backend.autonomous.agents.proof_candidate_list_validator import ProofCandidateListValidator
from backend.autonomous.agents.proof_formalization_agent import ProofFormalizationAgent
from backend.autonomous.core import proof_verification_stage as stage_module
from backend.autonomous.core.proof_verification_stage import (
    LEAN_WORKSPACE_BLOCKED_STATUS,
    ProofVerificationStage,
)
from backend.autonomous.memory.proof_database import _without_workspace_error_summary
from backend.autonomous.prompts.proof_prompts import _format_attempt_history
from backend.compiler.agents import high_param_submitter as high_param_module
from backend.compiler.agents.high_param_submitter import HighParamSubmitter
from backend.compiler.core.compiler_coordinator import CompilerCoordinator
from backend.leanoj.core import leanoj_coordinator as leanoj_module
from backend.leanoj.core.leanoj_coordinator import LeanOJCoordinator
from backend.shared import brainstorm_proof_gate as gate_module
from backend.shared.config import system_config
from backend.shared.lean4_client import (
    LEAN_WORKSPACE_REPAIR_REQUIRED_REASON,
    LEAN_WORKSPACE_SETUP_FAILED,
    LeanWorkspaceRepairRequiredError,
)
from backend.shared.models import (
    FailedProofCandidate,
    ProofAttemptFeedback,
    ProofCandidate,
    ProofCandidateListValidation,
    ProofCandidateNoveltyDecision,
)

WORKSPACE_OUTPUT = "LEAN 4 WORKSPACE ERROR: Mathlib cache/workspace repair failed."


def _repair_error() -> LeanWorkspaceRepairRequiredError:
    return LeanWorkspaceRepairRequiredError(
        failure_kind=LEAN_WORKSPACE_SETUP_FAILED,
        message="Lean 4 could not prepare its Mathlib workspace.",
        terminal_guidance="Fix Lean, then press Start again.",
    )


def _workspace_lean_result():
    return SimpleNamespace(
        success=False,
        error_output=WORKSPACE_OUTPUT,
        goal_states="",
        tactic_error_slice=WORKSPACE_OUTPUT,
    )


def _formalization_response() -> dict:
    return {"choices": [{"message": {"content": json.dumps({
        "theorem_name": "target",
        "lean_code": "theorem target : True := by trivial",
        "theorem_header": "theorem target : True",
        "tactics": ["trivial"],
        "reasoning": "Done",
    })}}]}


@pytest.fixture(autouse=True)
def _approve_candidate_lists(monkeypatch):
    async def approve(_self, *, candidates, **_kwargs):
        return ProofCandidateListValidation(
            results=[
                ProofCandidateNoveltyDecision(
                    theorem_id=candidate.theorem_id,
                    decision="approve_novel",
                    reasoning="Approved for this regression fixture.",
                )
                for candidate in candidates
            ],
            feedback="All fixture candidates are approved.",
        )

    monkeypatch.setattr(ProofCandidateListValidator, "validate", approve)


@pytest.fixture
def lean_enabled(monkeypatch):
    monkeypatch.setattr(system_config, "lean4_enabled", True)


@pytest.mark.asyncio
@pytest.mark.parametrize("method_name", ["prove_candidate", "prove_candidate_tactic_script"])
async def test_formalization_makes_no_model_call_when_lean_is_unavailable(monkeypatch, method_name):
    checker = SimpleNamespace(check_proof=AsyncMock(), check_tactic_script=AsyncMock())
    guard = Mock(side_effect=_repair_error())
    generate = AsyncMock()
    monkeypatch.setattr(formalization_module, "get_lean4_client", lambda: checker)
    monkeypatch.setattr(formalization_module, "require_lean_workspace", guard)
    monkeypatch.setattr(formalization_module.api_client_manager, "generate_completion", generate)
    agent = ProofFormalizationAgent(
        model_id="model", context_window=32000, max_output_tokens=2000, role_id="proof-test",
    )

    with pytest.raises(LeanWorkspaceRepairRequiredError):
        await getattr(agent, method_name)(
            user_research_prompt="Prove the target",
            source_type="paper",
            theorem_candidate=ProofCandidate(theorem_id="target", statement="True"),
            source_content="Source",
        )

    guard.assert_called_once_with(checker)
    generate.assert_not_awaited()
    checker.check_proof.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("method_name", ["prove_candidate", "prove_candidate_tactic_script"])
async def test_workspace_result_is_not_recorded_as_a_proof_attempt(monkeypatch, method_name):
    checker = SimpleNamespace(
        check_proof=AsyncMock(return_value=_workspace_lean_result()),
        check_tactic_script=AsyncMock(return_value=_workspace_lean_result()),
    )
    monkeypatch.setattr(formalization_module, "get_lean4_client", lambda: checker)
    monkeypatch.setattr(formalization_module, "require_lean_workspace", Mock())
    monkeypatch.setattr(
        formalization_module.api_client_manager,
        "generate_completion",
        AsyncMock(return_value=_formalization_response()),
    )
    monkeypatch.setattr(
        formalization_module, "_latest_assistant_pack_for_lean_attempts", lambda: ("", "", []),
    )
    agent = ProofFormalizationAgent(
        model_id="model", context_window=32000, max_output_tokens=2000, role_id="proof-test",
    )
    attempt_callback = AsyncMock()

    with pytest.raises(LeanWorkspaceRepairRequiredError):
        await getattr(agent, method_name)(
            user_research_prompt="Prove the target",
            source_type="paper",
            theorem_candidate=ProofCandidate(theorem_id="target", statement="True"),
            source_content="Source",
            attempt_callback=attempt_callback,
        )

    attempt_callback.assert_not_awaited()


async def _run_stage(stage, checkpoints, novel_proofs_db):
    async def save_checkpoint(payload):
        checkpoints.append(payload)

    return await stage.run(
        content="Source",
        source_type="brainstorm",
        source_id="topic_001",
        user_prompt="Prompt",
        submitter_model="model",
        submitter_context=20_000,
        submitter_max_tokens=2_000,
        validator_model="validator",
        validator_context=20_000,
        validator_max_tokens=2_000,
        broadcast_fn=AsyncMock(),
        novel_proofs_db=novel_proofs_db,
        checkpoint_callback=save_checkpoint,
        proof_run_context={"scope": "autonomous", "proof_run_id": "run", "lifecycle_generation": 1},
    )


@pytest.mark.asyncio
async def test_stage_keeps_candidates_unprocessed_and_stops_siblings(monkeypatch, lean_enabled):
    blocked = ProofCandidate(theorem_id="blocked", statement="True")
    sibling = ProofCandidate(theorem_id="sibling", statement="False")
    sibling_cancelled = asyncio.Event()

    async def fake_resolve(**_kwargs):
        return [blocked, sibling]

    async def fake_pipeline(**kwargs):
        if kwargs["theorem_candidate"].theorem_id == "sibling":
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                sibling_cancelled.set()
                raise
        raise _repair_error()

    stage = ProofVerificationStage()
    monkeypatch.setattr(stage, "_resolve_candidates", fake_resolve)
    monkeypatch.setattr(stage, "_run_lean_pipeline_for_candidate", fake_pipeline)
    monkeypatch.setattr(stage_module, "recover_lean_workspace", AsyncMock())
    checkpoints = []
    novel_proofs_db = AsyncMock()

    with pytest.raises(LeanWorkspaceRepairRequiredError):
        await _run_stage(stage, checkpoints, novel_proofs_db)

    assert sibling_cancelled.is_set()
    novel_proofs_db.record_failed_candidate.assert_not_awaited()
    final = checkpoints[-1]
    assert final["status"] == LEAN_WORKSPACE_BLOCKED_STATUS
    assert final["processed_candidate_ids"] == []
    assert not final["fatal_stop_reason"]
    assert [item["candidate"]["theorem_id"] for item in final["candidates"]] == ["blocked", "sibling"]


@pytest.mark.asyncio
async def test_stage_skips_identification_when_lean_is_known_broken(monkeypatch, lean_enabled):
    stage = ProofVerificationStage()
    resolve = AsyncMock()
    monkeypatch.setattr(stage, "_resolve_candidates", resolve)
    guard = AsyncMock(side_effect=_repair_error())
    monkeypatch.setattr(stage_module, "recover_lean_workspace", guard)
    checkpoints = []

    with pytest.raises(LeanWorkspaceRepairRequiredError):
        await _run_stage(stage, checkpoints, AsyncMock())

    guard.assert_awaited_once_with()
    resolve.assert_not_awaited()
    assert checkpoints == []


def test_stored_workspace_failures_never_reach_model_feedback():
    attempts = [
        ProofAttemptFeedback(attempt=1, theorem_id="t", error_output=WORKSPACE_OUTPUT),
        ProofAttemptFeedback(attempt=2, theorem_id="t", error_output="unsolved goals"),
    ]
    history = _format_attempt_history(attempts)
    assert "WORKSPACE" not in history
    assert "unsolved goals" in history

    hint = _without_workspace_error_summary(
        FailedProofCandidate(
            source_brainstorm_id="topic_001",
            theorem_id="t",
            theorem_statement="True",
            error_summary=WORKSPACE_OUTPUT,
        )
    )
    assert hint.error_summary == "Not yet checked by Lean (the Lean workspace was unavailable)."


@pytest.mark.asyncio
async def test_brainstorm_gate_spends_no_repair_calls_when_lean_cannot_run(monkeypatch, lean_enabled):
    checker = SimpleNamespace(check_proof=AsyncMock(return_value=_workspace_lean_result()))
    generate = AsyncMock()
    monkeypatch.setattr(gate_module, "get_lean4_client", lambda: checker)
    monkeypatch.setattr(gate_module.api_client_manager, "generate_completion", generate)

    with pytest.raises(LeanWorkspaceRepairRequiredError):
        await gate_module.verify_brainstorm_proof_candidate(
            parsed={
                "theorem_statement": "True",
                "lean_code": "theorem t : True := by trivial",
                "expected_novelty_tier": "novel_variant",
                "prompt_relevance_rationale": "Directly advances the prompt.",
                "novelty_rationale": "Not in Mathlib.",
                "why_not_standard_known_result": "No standard reference states it.",
            },
            user_prompt="Prompt",
            source_context="Source",
            model_id="model",
            role_id="aggregator_submitter_1",
            task_id_prefix="agg_sub1_000_lean",
            max_tokens=2000,
            validator_model="validator",
            validator_context=20_000,
            validator_max_tokens=2000,
            validator_role_id="aggregator_validator",
        )

    checker.check_proof.assert_awaited_once()
    generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_rigor_discovery_is_skipped_when_lean_is_known_broken(monkeypatch, lean_enabled):
    monkeypatch.setattr(system_config, "compiler_high_param_context_window", 10000)
    monkeypatch.setattr(system_config, "compiler_high_param_max_output_tokens", 1000)
    submitter = HighParamSubmitter(
        model_name="model",
        user_prompt="Prove something",
        validator_context_window=10000,
        validator_max_tokens=1000,
    )
    discovery = AsyncMock()
    monkeypatch.setattr(submitter, "_step_discovery", discovery)
    monkeypatch.setattr(
        high_param_module, "recover_lean_workspace", AsyncMock(side_effect=_repair_error()),
    )

    with pytest.raises(LeanWorkspaceRepairRequiredError):
        await submitter.submit_rigor_lean_theorem()

    discovery.assert_not_awaited()


async def _capture(target, event, payload):
    target.append((event, payload))


@pytest.mark.asyncio
async def test_compiler_stops_with_lean_repair_instead_of_declining(monkeypatch):
    coordinator = CompilerCoordinator()
    coordinator.enable_autonomous_mode()
    events = []
    coordinator.set_websocket_broadcaster(lambda event, payload: _capture(events, event, payload))

    async def fail_workflow():
        raise _repair_error()

    monkeypatch.setattr(coordinator, "_outline_creation_loop", fail_workflow)
    monkeypatch.setattr(
        "backend.compiler.core.compiler_coordinator.paper_memory.get_paper",
        AsyncMock(return_value=""),
    )
    monkeypatch.setattr(
        "backend.compiler.core.compiler_coordinator.outline_memory.get_outline",
        AsyncMock(return_value=""),
    )
    coordinator.is_running = True
    await coordinator._main_workflow()

    assert coordinator.is_running is False
    assert coordinator.fatal_error_type == LEAN_WORKSPACE_REPAIR_REQUIRED_REASON
    assert coordinator.fatal_error_payload["terminal_guidance"] == "Fix Lean, then press Start again."
    assert events[-1][0] == LEAN_WORKSPACE_REPAIR_REQUIRED_REASON


@pytest.mark.asyncio
async def test_leanoj_lean_check_raises_instead_of_counting_an_attempt(monkeypatch):
    checker = SimpleNamespace(check_proof=AsyncMock(return_value=_workspace_lean_result()))
    monkeypatch.setattr(leanoj_module, "get_lean4_client", lambda: checker)
    coordinator = LeanOJCoordinator()

    with pytest.raises(LeanWorkspaceRepairRequiredError):
        await coordinator._check_proof_and_capture_partial(
            None,
            "theorem t : True := by trivial",
            target="final",
            attempt_number=1,
            proof_request="final",
            reasoning="",
        )


@pytest.mark.asyncio
async def test_leanoj_lean_repair_stop_is_resumable(monkeypatch):
    coordinator = LeanOJCoordinator()
    events = []

    async def persist(event, data=None):
        events.append((event, data))

    monkeypatch.setattr(coordinator, "_persist_and_broadcast", persist)
    coordinator._state.phase = "final_proof"

    await coordinator._handle_lean_workspace_stop(_repair_error())

    assert coordinator._state.phase == "stopped"
    assert coordinator._fatal_stop_reason == LEAN_WORKSPACE_REPAIR_REQUIRED_REASON
    assert coordinator._stop_event.is_set()
    assert "press Start again" in coordinator._state.last_error
    assert events[-1][0] == LEAN_WORKSPACE_REPAIR_REQUIRED_REASON
    assert events[-1][1]["phase"] == "final_proof"
