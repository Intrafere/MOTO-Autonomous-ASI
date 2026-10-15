"""Proof agents keep the failed provider route's identity on workflow retries.

A transient provider failure inside proof formalization, proof-list review or
proof pruning must reach Live Activity with the real role, model, provider and
host instead of an ``unknown`` placeholder, while exception text stays generic.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from backend.autonomous.agents import proof_formalization_agent as formalization_module
from backend.autonomous.agents.proof_candidate_list_validator import (
    ProofCandidateListValidator,
)
from backend.autonomous.core import proof_verification_stage as stage_module
from backend.autonomous.core.proof_pruning_coordinator import ProofPruningCoordinator
from backend.autonomous.core.proof_verification_stage import (
    ProofVerificationStage,
    _LeanVerificationOutcome,
)
from backend.shared.api_client_manager import RetryableProviderError, api_client_manager
from backend.shared.config import system_config
from backend.shared.lean_proof_integrity import LeanProofIntegrityResult
from backend.shared.models import (
    ProofAttemptFeedback,
    ProofCandidate,
    ProofCandidateListValidation,
    ProofCandidateNoveltyDecision,
    ProofRoleConfigSnapshot,
    ProofRuntimeConfigSnapshot,
)
from backend.shared.openrouter_client import CreditExhaustionError
from backend.shared.provider_errors import (
    ProviderRepairRequiredError,
    ProviderRouteError,
    ProviderRouteIdentity,
)

TEST_MODEL = "vendor/test-free:free"
TEST_HOST = "TestHost"
PINNED_HOST = "PinnedHost"
REQUEST_ID = "gen-test-0001"
FORMALIZATION_ROLE = "autonomous_proof_formalization_manual_paper_testrun"
VALIDATOR_ROLE = "autonomous_proof_candidate_list_validator_manual_paper_testrun"
PRUNE_ROLE = "autonomous_proof_prune_validator_manual_testrun"


def _route_error(role_id: str, *, host_provider: str = "") -> ProviderRouteError:
    return ProviderRouteError(
        "OpenRouter request timed out",
        route=ProviderRouteIdentity(
            provider="openrouter",
            model=TEST_MODEL,
            role_id=role_id,
            task_id="proof_form_001",
            host_provider=host_provider,
            configured_provider="openrouter",
            configured_model=TEST_MODEL,
            upstream_host_provider=TEST_HOST,
            provider_request_id=REQUEST_ID,
        ),
        cause=httpx.ReadTimeout("upstream timed out"),
    )


def _formalization_agent(monkeypatch, error: BaseException):
    monkeypatch.setattr(
        formalization_module,
        "_latest_assistant_pack_for_lean_attempts",
        lambda: ("", "", []),
    )
    monkeypatch.setattr(
        formalization_module.api_client_manager,
        "generate_completion",
        AsyncMock(side_effect=error),
    )
    return formalization_module.ProofFormalizationAgent(
        model_id="agent-model",
        context_window=32000,
        max_output_tokens=2000,
        role_id="agent-display-role",
    )


async def _prove(agent, *, tactic: bool):
    method = agent.prove_candidate_tactic_script if tactic else agent.prove_candidate
    return await method(
        user_research_prompt="Prove the target.",
        source_type="paper",
        theorem_candidate=ProofCandidate(theorem_id="target", statement="True"),
        source_content="Mandatory source.",
        max_attempts=1,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("tactic", [False, True])
@pytest.mark.parametrize("pinned_host", ["", PINNED_HOST])
async def test_transient_route_error_keeps_openrouter_identity(monkeypatch, tactic, pinned_host):
    route_error = _route_error(FORMALIZATION_ROLE, host_provider=pinned_host)
    agent = _formalization_agent(monkeypatch, route_error)

    with pytest.raises(RetryableProviderError) as raised:
        await _prove(agent, tactic=tactic)

    error = raised.value
    assert error.__cause__ is route_error
    assert error.provider == "openrouter"
    assert error.provider_label == "OpenRouter"
    assert error.model == TEST_MODEL
    assert error.role_id == FORMALIZATION_ROLE
    assert error.reason == "transient_provider_error"
    assert error.task_id == "proof_form_001"
    assert error.configured_provider == "openrouter"
    assert error.configured_model == TEST_MODEL
    assert error.configured_host_provider == pinned_host
    assert error.upstream_host_provider == TEST_HOST
    assert error.provider_request_id == REQUEST_ID
    # Identity is structured-only: the message keeps its transient wording and
    # never gains the upstream host OpenRouter reported.
    assert str(error).startswith("TRANSIENT PROVIDER ERROR")
    assert TEST_HOST not in str(error)
    assert TEST_HOST not in error.error_summary

    fields = api_client_manager.route_identity_for_role(
        error.role_id,
        workflow_mode="manual_proof_check",
        error=error,
    )
    assert fields["provider"] == "openrouter"
    assert fields["model"] == TEST_MODEL
    assert fields["effective_host_provider"] == TEST_HOST
    if pinned_host:
        assert fields["host_routing"] == "pinned"
        assert f"host {PINNED_HOST} (pinned) → {TEST_HOST}" in fields["route_identity"]
    else:
        assert fields["host_routing"] == "auto"
        assert f"host auto → {TEST_HOST}" in fields["route_identity"]


@pytest.mark.asyncio
@pytest.mark.parametrize("tactic", [False, True])
async def test_existing_retryable_provider_error_is_reraised_unchanged(monkeypatch, tactic):
    original = RetryableProviderError(
        provider="openrouter",
        provider_label="OpenRouter",
        role_id=FORMALIZATION_ROLE,
        model=TEST_MODEL,
        reason="transient_provider_error",
        message="OpenRouter transient provider failure",
        configured_host_provider="",
        upstream_host_provider=TEST_HOST,
    )
    agent = _formalization_agent(monkeypatch, original)

    with pytest.raises(RetryableProviderError) as raised:
        await _prove(agent, tactic=tactic)

    assert raised.value is original


@pytest.mark.asyncio
async def test_untyped_transient_error_does_not_fabricate_route(monkeypatch):
    agent = _formalization_agent(
        monkeypatch,
        RuntimeError("OpenRouter request failed: HTTP 502 Bad Gateway"),
    )

    with pytest.raises(RetryableProviderError) as raised:
        await _prove(agent, tactic=False)

    error = raised.value
    assert error.provider == "unknown"
    assert error.provider_label == "Inference provider"
    assert error.role_id == "agent-display-role"
    assert error.model == "agent-model"
    assert error.configured_host_provider is None
    assert error.upstream_host_provider == ""


@pytest.mark.asyncio
async def test_candidate_list_transient_error_keeps_validator_route_identity():
    events = []

    async def broadcast(event, payload):
        events.append((event, payload))

    route_error = _route_error(VALIDATOR_ROLE)
    with patch.object(system_config, "lean4_enabled", True), patch(
        "backend.autonomous.core.proof_verification_stage."
        "ProofCandidateListValidator.validate",
        new=AsyncMock(side_effect=route_error),
    ), pytest.raises(RetryableProviderError) as raised:
        await ProofVerificationStage().run(
            content="Mandatory source.",
            source_type="paper",
            source_id="paper-1",
            source_title="Paper",
            user_prompt="Solve the objective.",
            submitter_model="submitter",
            submitter_context=8000,
            submitter_max_tokens=1000,
            validator_model="validator",
            validator_context=8000,
            validator_max_tokens=1000,
            broadcast_fn=broadcast,
            novel_proofs_db=object(),
            theorem_candidates=[ProofCandidate(theorem_id="thm-0", statement="Novel statement.")],
            source_reserved=True,
            release_source_on_exit=False,
            source_reservation_token="reservation",
            should_stop=lambda: False,
            trigger="manual",
            run_id="run-1",
        )

    error = raised.value
    assert error.provider == "openrouter"
    assert error.model == TEST_MODEL
    assert error.role_id == VALIDATOR_ROLE
    assert error.upstream_host_provider == TEST_HOST

    interrupted = next(
        payload
        for event, payload in events
        if event == "proof_candidate_list_review_interrupted"
    )
    assert interrupted["error_kind"] == "transient_provider_error"
    assert interrupted["workflow_mode"] == "manual_proof_check"
    assert interrupted["role_id"] == VALIDATOR_ROLE
    assert interrupted["provider"] == "openrouter"
    assert interrupted["model"] == TEST_MODEL
    assert interrupted["effective_host_provider"] == TEST_HOST
    assert interrupted["host_routing"] == "auto"
    assert interrupted["route_identity"].startswith("[Manual proof check · ")
    assert interrupted["message"].startswith("Validator proof-list review was interrupted")


def _approve_candidate_lists(monkeypatch):
    async def approve(_self, *, candidates, **_kwargs):
        return ProofCandidateListValidation(
            results=[
                ProofCandidateNoveltyDecision(
                    theorem_id=candidate.theorem_id,
                    decision="approve_novel",
                    reasoning="Approved for this route-identity fixture.",
                )
                for candidate in candidates
            ],
            feedback="All fixture candidates are approved.",
        )

    monkeypatch.setattr(ProofCandidateListValidator, "validate", approve)


def _transient_error(role_id: str) -> RetryableProviderError:
    return RetryableProviderError(
        provider="openrouter",
        provider_label="OpenRouter",
        role_id=role_id,
        model=TEST_MODEL,
        reason="transient_provider_error",
        message="OpenRouter transient provider failure",
        configured_host_provider="",
        upstream_host_provider=TEST_HOST,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("trigger", "role_suffix_override", "expected_mode", "expected_label"),
    [
        ("manual", None, "manual_proof_check", "Manual proof check"),
        (
            "manual_compiler_aggregator",
            "compiler_aggregator",
            "manual_proof_check",
            "Manual proof check",
        ),
        ("manual_compiler_save", "compiler_manual_paper", "compiler", "Compiler"),
    ],
)
async def test_post_lean_retry_waits_name_stage_workflow_over_active_owner(
    monkeypatch,
    tmp_path,
    trigger,
    role_suffix_override,
    expected_mode,
    expected_label,
):
    """User-triggered checks keep their own workflow while Autonomous owns the session."""
    candidate = ProofCandidate(theorem_id="verified-1", statement="True")
    lean_code = "import Mathlib\n\ntheorem verified_one : True := by trivial\n"
    novelty_role = f"autonomous_proof_novelty_{role_suffix_override or 'paper'}"
    integrity_calls = []
    register_calls = []

    async def fake_resolve(**_kwargs):
        return [candidate]

    async def fake_pipeline(**kwargs):
        return _LeanVerificationOutcome(
            candidate=candidate,
            proof_label=kwargs["proof_label"],
            success=True,
            theorem_name="verified_one",
            lean_code=lean_code,
            attempts=[
                ProofAttemptFeedback(
                    attempt=1,
                    theorem_id=candidate.theorem_id,
                    lean_code=lean_code,
                    success=True,
                )
            ],
        )

    async def fake_integrity(**_kwargs):
        integrity_calls.append(1)
        if len(integrity_calls) == 1:
            raise _transient_error(novelty_role)
        return LeanProofIntegrityResult(
            valid=True,
            actual_theorem_statement="True",
            actual_theorem_name="verified_one",
        )

    async def fake_register(**_kwargs):
        register_calls.append(1)
        if len(register_calls) == 1:
            raise _transient_error(novelty_role)
        return SimpleNamespace(
            record=SimpleNamespace(
                proof_id="proof-1",
                novel=False,
                novelty_tier="not_novel",
                novelty_reasoning="Fixture classification.",
            ),
            duplicate=False,
        )

    async def broadcast(_event, _payload):
        return None

    _approve_candidate_lists(monkeypatch)
    stage = ProofVerificationStage()
    monkeypatch.setattr(stage, "_resolve_candidates", fake_resolve)
    monkeypatch.setattr(stage, "_run_lean_pipeline_for_candidate", fake_pipeline)
    monkeypatch.setattr(stage_module, "validate_full_lean_proof_integrity", fake_integrity)
    monkeypatch.setattr(stage_module, "register_verified_lean_proof", fake_register)
    monkeypatch.setattr(
        stage._dependency_extractor,
        "extract_dependencies",
        AsyncMock(return_value=[]),
    )
    manager_broadcast = AsyncMock()
    with patch.object(system_config, "lean4_enabled", True), patch.object(
        system_config, "data_dir", str(tmp_path)
    ), patch(
        "backend.shared.api_client_manager._active_notification_workflow_mode",
        return_value="autonomous",
    ), patch.object(api_client_manager, "_broadcast", manager_broadcast), patch.object(
        api_client_manager,
        "_sleep_with_optional_stop",
        AsyncMock(return_value=False),
    ), patch.dict(api_client_manager._retryable_provider_backoff_state, clear=True):
        result = await stage.run(
            content="Source",
            source_type="paper",
            source_id="paper-1",
            user_prompt="Prove True",
            submitter_model="submitter",
            submitter_context=20_000,
            submitter_max_tokens=2_000,
            validator_model="validator",
            validator_context=20_000,
            validator_max_tokens=2_000,
            broadcast_fn=broadcast,
            novel_proofs_db=AsyncMock(),
            theorem_candidates=[candidate],
            role_suffix_override=role_suffix_override,
            trigger=trigger,
            append_to_source=False,
        )

    assert result.verified_count == 1
    retries = [
        call.args[1]
        for call in manager_broadcast.await_args_list
        if call.args[0] == "provider_retry_cooldown"
    ]
    # Manual checks conservatively store a novelty-provider failure as not
    # novel, so only their integrity check waits; other triggers also wait
    # before retrying registration.
    assert len(retries) == (1 if trigger == "manual" else 2)
    for payload in retries:
        assert payload["workflow_mode"] == expected_mode
        assert payload["role_id"] == novelty_role
        assert payload["model"] == TEST_MODEL
        assert payload["route_identity"].startswith(f"[{expected_label} · ")
        assert payload["message"].startswith(payload["route_identity"])


@pytest.mark.asyncio
async def test_competitor_unavailable_names_manual_check_workflow(monkeypatch, tmp_path):
    """A user-triggered check of an Autonomous source labels its secondary route manual."""
    candidate = ProofCandidate(theorem_id="contested-1", statement="True")
    secondary = {
        "provider": "openrouter",
        "model_id": "vendor/secondary-model",
        "openrouter_provider": PINNED_HOST,
        "context_window": 8_000,
        "max_output_tokens": 1_000,
    }
    events = []

    async def fake_resolve(**_kwargs):
        return [candidate]

    async def fake_pipeline(**kwargs):
        if "_competition_" in kwargs["role_suffix"]:
            raise _transient_error(
                f"autonomous_proof_formalization_{kwargs['role_suffix']}"
            )
        return _LeanVerificationOutcome(
            candidate=candidate,
            proof_label=kwargs["proof_label"],
            success=False,
            theorem_name="",
            lean_code="",
            attempts=[
                ProofAttemptFeedback(
                    attempt=attempt,
                    theorem_id=candidate.theorem_id,
                    lean_code="theorem contested_one : True := by simp",
                    error_output="error: unsolved goals",
                    success=False,
                )
                for attempt in range(1, 6)
            ],
        )

    async def broadcast(event, payload):
        events.append((event, payload))

    proofs_db = AsyncMock()
    proofs_db._base_dir = str(tmp_path)
    _approve_candidate_lists(monkeypatch)
    stage = ProofVerificationStage()
    monkeypatch.setattr(stage, "_resolve_candidates", fake_resolve)
    monkeypatch.setattr(stage, "_run_lean_pipeline_for_candidate", fake_pipeline)
    with patch.object(system_config, "lean4_enabled", True), patch.object(
        system_config, "data_dir", str(tmp_path)
    ), patch.object(api_client_manager, "configure_role", Mock()):
        await stage.run(
            content="Source",
            source_type="paper",
            source_id="paper-1",
            user_prompt="Prove True",
            canonical_user_prompt="Prove True",
            submitter_model="submitter",
            submitter_context=20_000,
            submitter_max_tokens=2_000,
            validator_model="validator",
            validator_context=20_000,
            validator_max_tokens=2_000,
            broadcast_fn=broadcast,
            novel_proofs_db=proofs_db,
            theorem_candidates=[candidate],
            trigger="manual",
            append_to_source=False,
            proof_run_context={"scope": "autonomous"},
            competition_config={"enabled": True, "secondaries": [secondary]},
        )

    unavailable = next(
        payload for event, payload in events if event == "proof_competitor_unavailable"
    )
    assert unavailable["workflow_mode"] == "manual_proof_check"
    assert unavailable["model"] == "vendor/secondary-model"
    assert unavailable["provider"] == "openrouter"
    assert unavailable["route_identity"].startswith("[Manual proof check · ")


def _pruning_coordinator(events):
    role = ProofRoleConfigSnapshot(
        provider="openrouter",
        model_id=TEST_MODEL,
        context_window=8192,
        max_output_tokens=1024,
    )

    async def broadcast(event, payload):
        events.append((event, payload))

    return ProofPruningCoordinator(
        proof_database=object(),
        runtime_snapshot=ProofRuntimeConfigSnapshot(
            brainstorm=role,
            paper=role,
            validator=role,
            assistant=role,
        ),
        proof_run_id="proof-run-1",
        run_mode="one_round",
        run_id="owning-run",
        lifecycle_generation=1,
        scope="manual",
        source_type="paper",
        source_id="paper-one",
        canonical_user_prompt="Prove the objective.",
        proof_store_id="manual:active",
        broadcast_fn=broadcast,
    )


@pytest.mark.asyncio
async def test_pruning_retry_wait_passes_failed_route_error():
    coordinator = _pruning_coordinator([])
    route_error = _route_error(PRUNE_ROLE)
    emit = AsyncMock(return_value=False)
    with patch.object(api_client_manager, "emit_and_wait_for_retry", emit):
        await coordinator._handle_failure(route_error, ["three_novel_proofs"])

    emit.assert_awaited_once()
    payload = emit.await_args.args[0]
    assert emit.await_args.kwargs["error"] is route_error
    assert emit.await_args.kwargs["role_id"] == "proof_prune"
    assert payload["workflow_mode"] == "manual_proof_check"
    assert payload["provider"] == "unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize("error_role", ["", PRUNE_ROLE])
async def test_pruning_typed_retry_names_own_workflow_over_active_owner(tmp_path, error_role):
    """A manual pruning retry stays manual while another workflow owns the session."""
    coordinator = _pruning_coordinator([])
    manager_broadcast = AsyncMock()
    with patch.object(system_config, "data_dir", str(tmp_path)), patch(
        "backend.shared.api_client_manager._active_notification_workflow_mode",
        return_value="aggregator",
    ), patch.object(api_client_manager, "_broadcast", manager_broadcast), patch.object(
        api_client_manager,
        "_sleep_with_optional_stop",
        AsyncMock(return_value=False),
    ), patch.dict(api_client_manager._retryable_provider_backoff_state, clear=True):
        await coordinator._handle_failure(
            _transient_error(error_role),
            ["three_novel_proofs"],
        )

    event, payload = manager_broadcast.await_args.args
    assert event == "provider_retry_cooldown"
    assert payload["workflow_mode"] == "manual_proof_check"
    assert payload["model"] == TEST_MODEL
    assert payload["effective_host_provider"] == TEST_HOST
    assert payload["route_identity"].startswith("[Manual proof check · ")
    assert payload["message"].startswith(payload["route_identity"])


@pytest.mark.asyncio
async def test_pruning_repair_event_names_failed_route():
    events = []
    coordinator = _pruning_coordinator(events)
    repair = ProviderRepairRequiredError(
        provider="openrouter",
        provider_label="OpenRouter",
        role_id=PRUNE_ROLE,
        model=TEST_MODEL,
        reason="provider_repair_required",
        message="OpenRouter could not serve the pruning Validator.",
        configured_host_provider="",
        upstream_host_provider=TEST_HOST,
    )

    await coordinator._handle_failure(repair, ["three_novel_proofs"])

    event, payload = events[-1]
    assert event == "proof_prune_repair_required"
    assert payload["workflow_mode"] == "manual_proof_check"
    assert payload["role_id"] == PRUNE_ROLE
    assert payload["provider"] == "openrouter"
    assert payload["model"] == TEST_MODEL
    assert payload["effective_host_provider"] == TEST_HOST
    assert payload["route_identity"]
    assert payload["route_identity"] not in payload["message"]


@pytest.mark.asyncio
async def test_pruning_credit_pause_without_route_names_workflow_only():
    events = []
    coordinator = _pruning_coordinator(events)
    with patch(
        "backend.autonomous.core.proof_pruning_coordinator.wait_for_provider_reset",
        new=AsyncMock(),
    ):
        await coordinator._handle_failure(
            CreditExhaustionError("OpenRouter credits exhausted"),
            ["three_novel_proofs"],
        )

    event, payload = next(item for item in events if item[0] == "proof_prune_provider_paused")
    assert event == "proof_prune_provider_paused"
    assert payload["workflow_mode"] == "manual_proof_check"
    assert payload["role_id"] == "proof_prune"
    assert "model" not in payload
    assert "provider" not in payload
    assert "host_routing" not in payload
