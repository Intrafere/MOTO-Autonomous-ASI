from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from backend.aggregator.core.context_allocator import ContextAllocationError
from backend.aggregator.core.coordinator import Coordinator
from backend.aggregator.memory.event_log import EventLog, event_log
from backend.autonomous.agents.proof_formalization_agent import ProofFormalizationAgent
from backend.autonomous.core.autonomous_coordinator import AutonomousCoordinator
from backend.autonomous.core.proof_verification_stage import ProofVerificationStage
from backend.autonomous.memory.research_metadata import research_metadata
from backend.shared.models import ModelConfig, ProofAttemptFeedback, ProofCandidate
from backend.shared.provider_errors import ProviderContextLengthError, ProviderRouteIdentity
from tests.workflow_harness.real_adapters import (
    EventCollector,
    assert_event_count,
    assert_fatal_context_overflow_event,
    assert_no_events,
    assert_proof_context_overflow_event,
    assert_route_identity,
)


def _model_config(model_id: str) -> ModelConfig:
    return ModelConfig(
        model_id=model_id,
        provider="openrouter",
        context_window=8192,
        max_output_tokens=1024,
    )


def _candidate() -> ProofCandidate:
    return ProofCandidate(theorem_id="candidate-1", statement="True")


@pytest.mark.asyncio
async def test_manual_aggregator_fatal_overflow_preserves_route_and_reloads(
    monkeypatch,
    tmp_path,
):
    coordinator = Coordinator()
    coordinator.is_running = True
    collector = EventCollector()
    coordinator.websocket_broadcaster = collector.broadcast
    event_log.file_path = tmp_path / "aggregator_event_log.txt"
    event_log.events = []
    await event_log.initialize()

    route = ProviderRouteIdentity(
        provider="lm_studio",
        model="fallback-model:2",
        role_id="aggregator_submitter_1",
        task_id="agg_sub1_001",
        host_provider="local-sibling",
        route_kind="lm_studio_fallback",
        configured_provider="openrouter",
        configured_model="configured/model",
    )
    provider_error = ProviderContextLengthError("provider context limit", route=route)
    overflow = ContextAllocationError.from_provider_error(
        provider_error,
        "mandatory direct context overflow",
        required_tokens=9000,
        available_tokens=7000,
    )
    monkeypatch.setattr(
        "backend.aggregator.core.coordinator.api_client_manager.get_role_config",
        lambda _role: _model_config("configured/model"),
    )
    monkeypatch.setattr(
        "backend.aggregator.core.coordinator.queue_manager.clear",
        AsyncMock(),
    )

    await coordinator._handle_context_overflow(
        overflow,
        role_id="aggregator_submitter_1",
    )

    payload = assert_event_count(
        collector.events,
        "context_overflow_error",
        1,
    )[0]
    assert_fatal_context_overflow_event(
        payload,
        workflow_mode="aggregator",
        role_id="aggregator_submitter_1",
    )
    assert_route_identity(
        payload,
        configured_model="configured/model",
        configured_provider="openrouter",
        effective_model="fallback-model:2",
        effective_provider="lm_studio",
        effective_host_provider="local-sibling",
        route_kind="lm_studio_fallback",
    )
    assert coordinator.is_running is False
    assert coordinator.fatal_error_payload == payload
    assert event_log.file_path.resolve().is_relative_to(tmp_path.resolve())

    reloaded = EventLog()
    reloaded.file_path = event_log.file_path
    await reloaded.initialize()
    persisted = await reloaded.get_all_events()
    assert len(persisted) == 1
    assert persisted[0]["type"] == "context_overflow_error"
    assert persisted[0]["metadata"] == payload


@pytest.mark.asyncio
async def test_autonomous_fatal_overflow_terminal_stop_emits_exactly_once(monkeypatch):
    coordinator = AutonomousCoordinator()
    collector = EventCollector()
    coordinator.set_broadcast_callback(collector.broadcast)
    monkeypatch.setattr(research_metadata, "get_stats", AsyncMock(return_value={}))
    payload = {
        "workflow_mode": "autonomous",
        "role_id": "compiler_writer",
        "configured_model": "configured/model",
        "configured_provider": "openrouter",
        "effective_model": "fallback-model:2",
        "effective_provider": "lm_studio",
        "reason": "context_overflow",
        "message": "Mandatory direct context exceeded the configured model window.",
        "resolution": "Increase the configured context window or reduce mandatory context.",
    }

    coordinator._mark_context_overflow_stop(payload)
    await coordinator._broadcast_stopped_once()
    await coordinator._broadcast_stopped_once()

    terminal = assert_event_count(
        collector.events,
        "auto_research_stopped",
        1,
    )[0]
    assert_fatal_context_overflow_event(
        terminal,
        workflow_mode="autonomous",
        role_id="compiler_writer",
    )
    assert_route_identity(
        terminal,
        configured_model="configured/model",
        configured_provider="openrouter",
        effective_model="fallback-model:2",
        effective_provider="lm_studio",
    )


@pytest.mark.asyncio
async def test_nonfatal_proof_overflow_emits_once_without_parent_stop(monkeypatch):
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
    collector = EventCollector()

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
        broadcast_fn=collector.broadcast,
    )

    payload = assert_event_count(
        collector.events,
        "proof_context_overflow",
        1,
    )[0]
    assert_proof_context_overflow_event(payload, workflow_mode="autonomous")
    assert_route_identity(
        payload,
        configured_model="configured/model",
        configured_provider="openrouter",
        effective_model="fallback-model:2",
        effective_provider="lm_studio",
    )
    assert_no_events(
        collector.events,
        "context_overflow_error",
        "auto_research_stopped",
        "compiler_stopped",
        "aggregator_stopped",
        "leanoj_stopped",
        "proof_attempt_failed",
    )
