from __future__ import annotations

import asyncio
from importlib import import_module
from unittest.mock import AsyncMock

import pytest

from backend.api.routes import compiler as compiler_route
from backend.autonomous.core.autonomous_coordinator import AutonomousCoordinator
from backend.autonomous.memory.research_metadata import ResearchMetadata
from backend.leanoj.core.leanoj_coordinator import LeanOJCoordinator
from backend.shared.config import system_config
from backend.shared.models import CompilerStartRequest, LeanOJRoleConfig, LeanOJStartRequest
from backend.shared.models import ProofCandidate
from backend.shared.provider_pause import resume_provider_pauses
from tests.workflow_real_adapters.coverage_records import PROOF_ROUTE_COVERAGE
from tests.workflow_real_adapters.helpers import CallRecorder, assert_event_sequence
from tests.workflow_harness.invariants import (
    assert_observed_invariant,
    assert_proofs_only_never_enters_paper_phase,
)
from tests.workflow_harness.model import WorkflowMode, WorkflowPhase
from tests.workflow_harness.real_adapters import (
    EventCollector,
    FakeResearchMetadata,
    FakeProofStage,
    RealWorkflowObservation,
    RoleConfigCapture,
)

coordinator_module = import_module("backend.autonomous.core.autonomous_coordinator")


def _compiler_request(**overrides) -> CompilerStartRequest:
    data = {
        "compiler_prompt": "Prove something from the manual Aggregator database.",
        "allow_mathematical_proofs": True,
        "allow_research_papers": False,
        "validator_model": "validator-model",
        "validator_context_size": 1000,
        "validator_max_output_tokens": 100,
        "writer_model": "writer-model",
        "writer_context_size": 1000,
        "writer_max_output_tokens": 100,
        "high_param_provider": "openrouter",
        "high_param_model": "rigor-model",
        "high_param_openrouter_provider": "RigorHost",
        "high_param_openrouter_reasoning_effort": "high",
        "high_param_lm_studio_fallback": "rigor-fallback",
        "high_param_context_size": 3000,
        "high_param_max_output_tokens": 300,
        "high_param_supercharge_enabled": True,
    }
    data.update(overrides)
    return CompilerStartRequest(**data)


def _leanoj_request() -> LeanOJStartRequest:
    role = LeanOJRoleConfig(
        model_id="leanoj-test-model",
        context_window=8192,
        max_output_tokens=1024,
    )
    return LeanOJStartRequest(
        user_prompt="Prove one equals one.",
        lean_template="import Mathlib\n\nexample : 1 = 1 := by\n  sorry",
        topic_generator=role,
        topic_validator=role,
        brainstorm_submitters=[role],
        brainstorm_validator=role,
        path_decider=role,
        final_solver=role,
    )


@pytest.mark.asyncio
async def test_manual_compiler_proof_only_uses_rigor_settings_and_manual_scope(monkeypatch, tmp_path):
    events = EventCollector()
    roles = RoleConfigCapture()
    fake_stage = FakeProofStage()
    observation = RealWorkflowObservation(
        runtime_root=tmp_path,
        mode=WorkflowMode.MANUAL_COMPILER,
        phase=WorkflowPhase.MANUAL_PROOF,
        allow_mathematical_proofs=True,
        allow_research_papers=False,
    )
    request = _compiler_request()

    class StageFactory:
        def __call__(self):
            return fake_stage

    class ManualProofDb:
        def inject_into_prompt(self, prompt: str) -> str:
            return f"manual-scope::{prompt}"

    async def append_manual_proof(_proof):
        observation.ingest_event(
            "proof_verified",
            {"proof_id": "manual-proof-1", "scope": "manual", "phase": WorkflowPhase.MANUAL_PROOF.value},
        )

    monkeypatch.setattr(
        compiler_route,
        "_read_manual_aggregator_context",
        AsyncMock(return_value="Accepted manual Aggregator submission."),
        raising=False,
    )
    monkeypatch.setattr(compiler_route, "_release_pre_reserved_source", AsyncMock(), raising=False)
    monkeypatch.setattr(compiler_route.websocket, "broadcast_event", events.broadcast)
    monkeypatch.setattr(compiler_route.api_client_manager, "configure_role", roles.configure)
    monkeypatch.setattr(compiler_route, "ProofVerificationStage", StageFactory())
    monkeypatch.setattr(compiler_route, "manual_proof_database", ManualProofDb())
    monkeypatch.setattr(compiler_route, "append_proof_to_manual_shared_training", append_manual_proof)
    monkeypatch.setattr(compiler_route.assistant_proof_search_coordinator, "stop_all", AsyncMock())
    monkeypatch.setattr(compiler_route.token_tracker, "reset", lambda: None)
    monkeypatch.setattr(compiler_route.token_tracker, "start_timer", lambda: None)
    monkeypatch.setattr(compiler_route.token_tracker, "stop_timer", lambda: None)

    await compiler_route._run_compiler_aggregator_proof_check(request, source_reserved=True)

    proof_role = roles.roles["autonomous_proof_formalization_compiler_aggregator"]
    assert proof_role.provider == "openrouter"
    assert proof_role.model_id == "rigor-model"
    assert proof_role.openrouter_provider == "RigorHost"
    assert proof_role.openrouter_reasoning_effort == "high"
    assert proof_role.lm_studio_fallback_id == "rigor-fallback"
    assert proof_role.context_window == 3000
    assert proof_role.max_output_tokens == 300
    assert proof_role.supercharge_enabled is True
    assert fake_stage.calls[0]["submitter_model"] == "rigor-model"
    assert fake_stage.calls[0]["novel_proofs_db"].inject_into_prompt("x") == "manual-scope::x"
    observation.record("manual_compiler_proof_only")
    assert observation.manual_proofs_active == {"manual-proof-1"}
    observation.observed_invariants.add("proof_scope.manual_not_in_autonomous_current")
    assert_observed_invariant(observation, "proof_scope.manual_not_in_autonomous_current")


@pytest.mark.asyncio
async def test_autonomous_proofs_only_brainstorm_handoff_returns_to_topic_exploration(monkeypatch, tmp_path):
    events = EventCollector()
    coordinator = AutonomousCoordinator()
    coordinator._allow_mathematical_proofs = True
    coordinator._allow_research_papers = False
    coordinator._current_topic_id = "topic-proof-only"
    coordinator._current_paper_id = "stale-paper"
    coordinator._current_paper_title = "Stale paper"
    coordinator._current_reference_papers = ["paper-ref"]
    coordinator._current_reference_brainstorms = ["brainstorm-ref"]
    coordinator._state.current_tier = "tier2_paper_writing"
    coordinator._broadcast = events.broadcast

    saved_states = []

    async def save_state(state):
        saved_states.append(dict(state))

    monkeypatch.setattr(coordinator_module.research_metadata, "set_current_brainstorm", AsyncMock())
    monkeypatch.setattr(coordinator_module.research_metadata, "save_workflow_state", save_state)
    monkeypatch.setattr(coordinator_module.research_metadata, "get_workflow_state", AsyncMock(return_value={}))
    monkeypatch.setattr(
        coordinator_module.final_answer_memory,
        "get_state",
        lambda: type("Tier3State", (), {"is_active": False})(),
    )
    monkeypatch.setattr(coordinator_module.final_answer_memory, "get_answer_format", lambda: None)

    await coordinator._handle_papers_disabled_after_brainstorm()

    observation = RealWorkflowObservation(
        runtime_root=tmp_path,
        mode=WorkflowMode.AUTONOMOUS,
        phase=WorkflowPhase.TOPIC_EXPLORATION,
        allow_mathematical_proofs=True,
        allow_research_papers=False,
    )
    observation.record("autonomous_proofs_only_handoff")
    for event_type, payload in events.events:
        observation.ingest_event(event_type, payload)
    observation.checkpoint = saved_states[-1]

    assert coordinator._current_topic_id is None
    assert coordinator._current_paper_id is None
    assert coordinator._current_reference_papers == []
    assert saved_states[-1]["current_tier"] == "tier1_aggregation"
    assert saved_states[-1]["paper_phase"] == "topic_exploration"
    assert_proofs_only_never_enters_paper_phase(observation)


@pytest.mark.asyncio
async def test_autonomous_provider_credit_pause_preserves_checkpoint_and_resumes(monkeypatch, tmp_path):
    events = EventCollector()
    fake_stage = FakeProofStage(
        pause=True,
        checkpoint={
            "source_type": "brainstorm",
            "source_id": "topic-paused",
            "trigger": "automatic",
            "status": "phase_a_running",
            "candidates": [
                {
                    "index": 1,
                    "candidate": {
                        "theorem_id": "paused_candidate",
                        "statement": "A direct proof target from the source.",
                        "expected_novelty_tier": "novel_variant",
                    },
                },
                {
                    "index": 4,
                    "candidate": {
                        "theorem_id": "already_processed",
                        "statement": "A completed proof target.",
                        "expected_novelty_tier": "novel_variant",
                    },
                }
            ],
            "processed_candidate_ids": ["already_processed"],
            "attempts_by_candidate": {
                "paused_candidate": [
                    {
                        "attempt": 2,
                        "theorem_id": "paused_candidate",
                        "reasoning": "Retry with a more direct route.",
                        "lean_code": "example : True := by trivial",
                        "error_output": "previous tactic failed",
                        "strategy": "full_script",
                        "success": False,
                    }
                ],
                "already_processed": [
                    {
                        "attempt": 1,
                        "theorem_id": "already_processed",
                        "strategy": "full_script",
                        "success": True,
                    }
                ],
            },
            "theorem_names_by_candidate": {
                "paused_candidate": "pausedTheorem",
                "already_processed": "finishedTheorem",
            },
            "proof_round_index": 1,
        },
    )
    fake_metadata = FakeResearchMetadata()
    coordinator = AutonomousCoordinator()
    coordinator._proof_verification_stage = fake_stage
    coordinator._allow_mathematical_proofs = True
    coordinator._allow_research_papers = False
    coordinator._high_param_model = "rigor-model"
    coordinator._high_param_context = 3000
    coordinator._high_param_max_tokens = 300
    coordinator._validator_model = "validator-model"
    coordinator._validator_context = 2000
    coordinator._validator_max_tokens = 200
    coordinator._user_research_prompt = "Solve the prompt."
    coordinator._state.current_tier = "tier2_paper_writing"
    coordinator._broadcast = events.broadcast

    old_lean_enabled = system_config.lean4_enabled
    monkeypatch.setattr(coordinator_module, "research_metadata", fake_metadata)
    paused_calls = CallRecorder()
    monkeypatch.setattr(coordinator_module, "mark_provider_paused", paused_calls)
    monkeypatch.setattr(coordinator_module, "wait_for_provider_resume", AsyncMock(return_value=None))
    monkeypatch.setattr(
        coordinator_module.proof_database,
        "inject_into_prompt",
        lambda prompt, **_kwargs: prompt,
        raising=False,
    )
    monkeypatch.setattr(
        coordinator_module.final_answer_memory,
        "get_state",
        lambda: type("Tier3State", (), {"is_active": False})(),
    )
    monkeypatch.setattr(coordinator_module.final_answer_memory, "get_answer_format", lambda: None)
    system_config.lean4_enabled = True
    try:
        status = await coordinator._run_proof_verification(
            "Brainstorm source content.",
            "brainstorm",
            "topic-paused",
            source_title="Paused topic",
            theorem_candidates=[
                ProofCandidate(
                    theorem_id="initial_candidate",
                    statement="A prompt-relevant theorem.",
                    expected_novelty_tier="novel_variant",
                )
            ],
        )
    finally:
        system_config.lean4_enabled = old_lean_enabled

    observation = RealWorkflowObservation(
        runtime_root=tmp_path,
        mode=WorkflowMode.AUTONOMOUS,
        phase=WorkflowPhase.BRAINSTORM_PROOF,
        allow_mathematical_proofs=True,
        allow_research_papers=False,
    )
    observation.record("autonomous_provider_credit_pause")
    for event_type, payload in events.events:
        observation.ingest_event(event_type, payload)

    assert status == "complete"
    assert fake_stage.pause_count == 1
    assert len(fake_stage.calls) == 2
    assert fake_metadata.saved_proof_checkpoints
    saved_checkpoint = fake_metadata.saved_proof_checkpoints[0]
    assert saved_checkpoint["source_type"] == "brainstorm"
    assert saved_checkpoint["source_id"] == "topic-paused"
    assert saved_checkpoint["trigger"] == "automatic"
    assert saved_checkpoint["candidates"][0]["candidate"]["theorem_id"] == "paused_candidate"
    retry_call = fake_stage.calls[1]
    assert [candidate.theorem_id for candidate in retry_call["theorem_candidates"]] == [
        "paused_candidate"
    ]
    assert retry_call["proof_candidate_indexes"] == {
        "paused_candidate": 1,
        "already_processed": 4,
    }
    assert retry_call["checkpoint_attempts_by_candidate"]["paused_candidate"][0].attempt == 2
    assert retry_call["checkpoint_attempts_by_candidate"]["paused_candidate"][0].error_output == (
        "previous tactic failed"
    )
    assert retry_call["checkpoint_attempts_by_candidate"]["already_processed"][0].success is True
    assert retry_call["checkpoint_theorem_names_by_candidate"] == {
        "paused_candidate": "pausedTheorem",
        "already_processed": "finishedTheorem",
    }
    assert retry_call["trigger"] == "automatic"
    assert fake_metadata.proof_checkpoint is not None
    assert fake_metadata.proof_checkpoint["source_type"] == "brainstorm"
    assert fake_metadata.proof_checkpoint["source_id"] == "topic-paused"
    assert fake_metadata.proof_checkpoint["trigger"] == "automatic"
    assert fake_metadata.proof_checkpoint["status"] == "trigger_complete"
    assert fake_metadata.completed_triggers == ["automatic"]
    assert fake_metadata.workflow_states[-1]["paper_phase"] == "brainstorm_proof_verification"
    paused_payloads = events.payloads("autonomous_proof_provider_paused")
    resumed_payloads = events.payloads("autonomous_proof_provider_resumed")
    assert paused_payloads
    assert resumed_payloads
    assert paused_payloads[0]["reason"] == "openrouter_credit_exhaustion"
    assert paused_calls.count == 1
    assert_event_sequence(
        events,
        "autonomous_proof_provider_paused",
        "autonomous_proof_provider_resumed",
        predicate=lambda payload: payload.get("reason") == "openrouter_credit_exhaustion",
    )
    assert observation.provider.credit_exhausted is False
    observation.observed_invariants.add("provider.pause_preserves_checkpoint")
    assert_observed_invariant(observation, "provider.pause_preserves_checkpoint")


@pytest.mark.asyncio
async def test_research_metadata_proof_checkpoint_roundtrip_preserves_resume_cursor(tmp_path):
    workflow_path = tmp_path / "workflow_state.json"
    checkpoint = {
        "source_type": "brainstorm",
        "source_id": "topic-roundtrip",
        "source_title": "Roundtrip topic",
        "trigger": "manual_retry",
        "status": "phase_a_running",
        "candidates": [
            {
                "index": 7,
                "candidate": {
                    "theorem_id": "roundtrip_candidate",
                    "statement": "A persisted theorem target.",
                    "expected_novelty_tier": "novel_variant",
                },
            }
        ],
        "processed_candidate_ids": ["finished_candidate"],
        "attempts_by_candidate": {
            "roundtrip_candidate": [
                {
                    "attempt": 3,
                    "theorem_id": "roundtrip_candidate",
                    "error_output": "persisted Lean failure",
                    "strategy": "tactic_script",
                    "success": False,
                }
            ]
        },
        "theorem_names_by_candidate": {
            "roundtrip_candidate": "roundtripTheorem",
        },
    }

    metadata = ResearchMetadata()
    metadata._workflow_state_path = workflow_path
    await metadata.save_proof_checkpoint(dict(checkpoint))

    restored = ResearchMetadata()
    restored._workflow_state_path = workflow_path
    loaded = await restored.get_proof_checkpoint(
        "brainstorm",
        "topic-roundtrip",
        "manual_retry",
    )

    assert loaded is not None
    assert loaded["candidates"] == checkpoint["candidates"]
    assert loaded["processed_candidate_ids"] == ["finished_candidate"]
    assert loaded["attempts_by_candidate"] == checkpoint["attempts_by_candidate"]
    assert loaded["theorem_names_by_candidate"] == checkpoint["theorem_names_by_candidate"]
    assert loaded["trigger"] == "manual_retry"
    assert loaded["completed_triggers"] == []
    assert loaded["updated_at"]
    assert await restored.get_proof_checkpoint("paper", "topic-roundtrip") is None

    await restored.mark_proof_checkpoint_trigger_complete(
        "brainstorm",
        "topic-roundtrip",
        "manual_retry",
        "Roundtrip topic",
    )
    reloaded = ResearchMetadata()
    reloaded._workflow_state_path = workflow_path
    completed = await reloaded.get_proof_checkpoint(
        "brainstorm",
        "topic-roundtrip",
        "manual_retry",
    )
    assert completed is not None
    assert completed["status"] == "trigger_complete"
    assert completed["completed_triggers"] == ["manual_retry"]
    assert completed["attempts_by_candidate"] == checkpoint["attempts_by_candidate"]
    assert completed["theorem_names_by_candidate"] == checkpoint["theorem_names_by_candidate"]


@pytest.mark.asyncio
async def test_autonomous_provider_pause_stop_reset_restart_uses_real_event_and_metadata(
    monkeypatch,
    tmp_path,
):
    workflow_path = tmp_path / "workflow_state.json"
    metadata = ResearchMetadata()
    metadata._workflow_state_path = workflow_path
    paused_stage = FakeProofStage(
        pause=True,
        checkpoint={
            "source_type": "brainstorm",
            "source_id": "topic-restart",
            "source_title": "Restart topic",
            "trigger": "restart_trigger",
            "status": "phase_a_running",
            "candidates": [
                {
                    "index": 3,
                    "candidate": {
                        "theorem_id": "restart_candidate",
                        "statement": "A restart-safe proof target.",
                        "expected_novelty_tier": "novel_variant",
                    },
                }
            ],
            "processed_candidate_ids": [],
            "attempts_by_candidate": {
                "restart_candidate": [
                    {
                        "attempt": 1,
                        "theorem_id": "restart_candidate",
                        "error_output": "credit pause after first attempt",
                        "strategy": "full_script",
                        "success": False,
                    }
                ]
            },
            "theorem_names_by_candidate": {"restart_candidate": "restartTheorem"},
        },
    )
    first = AutonomousCoordinator()
    first._proof_verification_stage = paused_stage
    first._allow_mathematical_proofs = True
    first._high_param_model = "rigor-model"
    first._high_param_context = 3000
    first._high_param_max_tokens = 300
    first._validator_model = "validator-model"
    first._validator_context = 2000
    first._validator_max_tokens = 200
    first._user_research_prompt = "Solve the restart prompt."
    first._base_user_research_prompt = "Solve the restart prompt."
    first._state.current_tier = "tier2_paper_writing"
    first._current_topic_id = "topic-restart"
    first._broadcast = EventCollector().broadcast

    monkeypatch.setattr(coordinator_module, "research_metadata", metadata)
    monkeypatch.setattr(
        coordinator_module.proof_database,
        "inject_into_prompt",
        lambda prompt, **_kwargs: prompt,
        raising=False,
    )
    monkeypatch.setattr(
        coordinator_module.final_answer_memory,
        "get_state",
        lambda: type("Tier3State", (), {"is_active": False})(),
    )
    monkeypatch.setattr(coordinator_module.final_answer_memory, "get_answer_format", lambda: None)
    old_lean_enabled = system_config.lean4_enabled
    system_config.lean4_enabled = True
    resume_provider_pauses()
    try:
        paused_task = asyncio.create_task(
            first._run_proof_verification(
                "Persisted brainstorm source.",
                "brainstorm",
                "topic-restart",
                source_title="Restart topic",
                trigger="restart_trigger",
            )
        )
        for _ in range(100):
            if paused_stage.pause_count:
                break
            await asyncio.sleep(0.01)
        assert paused_stage.pause_count == 1

        first._stop_event.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(paused_task, timeout=2.0)
        persisted = await metadata.get_proof_checkpoint(
            "brainstorm",
            "topic-restart",
            "restart_trigger",
        )
        assert persisted is not None
        assert persisted["status"] == "phase_a_running"
        assert persisted["attempts_by_candidate"]["restart_candidate"][0]["attempt"] == 1

        assert resume_provider_pauses() >= 1
        restored_metadata = ResearchMetadata()
        restored_metadata._workflow_state_path = workflow_path
        resumed_stage = FakeProofStage()
        restarted = AutonomousCoordinator()
        restarted._proof_verification_stage = resumed_stage
        restarted._allow_mathematical_proofs = True
        restarted._high_param_model = "rigor-model"
        restarted._high_param_context = 3000
        restarted._high_param_max_tokens = 300
        restarted._validator_model = "validator-model"
        restarted._validator_context = 2000
        restarted._validator_max_tokens = 200
        restarted._user_research_prompt = "Solve the restart prompt."
        restarted._base_user_research_prompt = "Solve the restart prompt."
        restarted._broadcast = EventCollector().broadcast
        monkeypatch.setattr(coordinator_module, "research_metadata", restored_metadata)

        assert await restarted._run_proof_verification(
            "Persisted brainstorm source.",
            "brainstorm",
            "topic-restart",
            source_title="Restart topic",
            trigger="restart_trigger",
        ) == "complete"
        assert len(resumed_stage.calls) == 1
        resumed_call = resumed_stage.calls[0]
        assert [candidate.theorem_id for candidate in resumed_call["theorem_candidates"]] == [
            "restart_candidate"
        ]
        assert resumed_call["proof_candidate_indexes"] == {"restart_candidate": 3}
        assert resumed_call["checkpoint_attempts_by_candidate"]["restart_candidate"][0].attempt == 1
        assert resumed_call["checkpoint_theorem_names_by_candidate"] == {
            "restart_candidate": "restartTheorem"
        }
        assert resumed_call["trigger"] == "restart_trigger"
        completed = await restored_metadata.get_proof_checkpoint(
            "brainstorm",
            "topic-restart",
            "restart_trigger",
        )
        assert completed is not None
        assert completed["status"] == "trigger_complete"
        assert "restart_trigger" in completed["completed_triggers"]
    finally:
        resume_provider_pauses()
        system_config.lean4_enabled = old_lean_enabled


@pytest.mark.asyncio
async def test_leanoj_master_proof_stop_resume_preserves_isolated_state(monkeypatch, tmp_path):
    request = _leanoj_request()
    master_proof = "import Mathlib\n\nexample : 1 = 1 := by\n  rfl"
    autonomous_proof_store_accessed = False

    class AutonomousProofStoreGuard:
        def __getattr__(self, name):
            nonlocal autonomous_proof_store_accessed
            autonomous_proof_store_accessed = True
            raise AssertionError(f"LeanOJ stop/resume must not access autonomous proof storage: {name}")

    async def no_broadcast(*_args, **_kwargs):
        return None

    monkeypatch.setattr(system_config, "data_dir", tmp_path)
    monkeypatch.setattr(coordinator_module, "proof_database", AutonomousProofStoreGuard())

    coordinator = LeanOJCoordinator()
    monkeypatch.setattr(coordinator, "_broadcast", no_broadcast)
    await coordinator.initialize(request)
    coordinator._state.phase = "final_proof_loop"
    await coordinator._write_master_proof(master_proof, summary="durable final-proof draft")
    await coordinator.stop()

    restored = LeanOJCoordinator()
    monkeypatch.setattr(restored, "_broadcast", no_broadcast)
    resumed = await restored.resume_or_initialize(request)

    observation = RealWorkflowObservation(
        runtime_root=tmp_path,
        mode=WorkflowMode.LEANOJ,
        phase=WorkflowPhase.PAPER_WRITING,
    )
    observation.record("leanoj_master_proof_stop_resume")

    assert resumed is True
    assert restored.get_state().phase == "final_proof_loop"
    assert restored.get_state().master_proof_initialized is True
    assert await restored._read_master_proof() == master_proof
    assert autonomous_proof_store_accessed is False
    assert observation.autonomous_proofs == set()


@pytest.mark.asyncio
async def test_leanoj_persisted_provider_pause_resumes_before_nonfinal_workflow(
    monkeypatch,
    tmp_path,
):
    request = _leanoj_request()
    events = EventCollector()
    monkeypatch.setattr(system_config, "data_dir", tmp_path)
    resume_provider_pauses()

    coordinator = LeanOJCoordinator()
    coordinator._broadcast = events.broadcast
    await coordinator.initialize(request)
    coordinator._state.phase = "initial_brainstorm"
    coordinator._state.last_active_phase = "initial_brainstorm"
    coordinator._state.provider_paused = True
    coordinator._state.provider_pause_reason = "openrouter_credit_exhaustion"
    coordinator._state.provider_pause_role_id = "leanoj_brainstorm_sub1"
    coordinator._state.provider_pause_message = "OpenRouter credits exhausted"
    await coordinator._persist_state()

    restored = LeanOJCoordinator()
    restored._broadcast = events.broadcast
    assert await restored.resume_or_initialize(request) is True
    assert restored.get_state().phase == "initial_brainstorm"
    assert restored.get_state().provider_paused is True

    entered_phases: list[str] = []

    async def nonfinal_workflow(_request):
        entered_phases.append(restored.get_state().phase)

    monkeypatch.setattr(restored, "_run_workflow", nonfinal_workflow)
    start_task = asyncio.create_task(restored.start())
    try:
        for _ in range(100):
            if events.payloads("leanoj_provider_paused"):
                break
            await asyncio.sleep(0.01)
        assert events.payloads("leanoj_provider_paused")
        assert entered_phases == []
        assert resume_provider_pauses() >= 1
        await asyncio.wait_for(start_task, timeout=2.0)
    finally:
        resume_provider_pauses()
        if not start_task.done():
            start_task.cancel()
            await asyncio.gather(start_task, return_exceptions=True)

    assert entered_phases == ["initial_brainstorm"]
    assert restored.get_state().provider_paused is False
    assert restored.get_state().phase != "final_proof_loop"
    assert_event_sequence(
        events,
        "leanoj_provider_paused",
        "leanoj_provider_resumed",
        predicate=lambda payload: payload.get("reason") == "openrouter_credit_exhaustion",
    )
