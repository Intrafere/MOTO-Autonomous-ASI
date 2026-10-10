from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from backend.autonomous.core.proof_run_manager import ProofRunControl, ProofRunManager
from backend.autonomous.core.proof_verification_stage import (
    ProofVerificationStage,
    _candidate_list_fingerprint,
    _candidate_list_review_scope,
)
from backend.autonomous.memory.research_metadata import ResearchMetadata
from backend.shared.config import system_config
from backend.shared.models import (
    ProofCandidate,
    ProofCandidateListValidation,
    ProofCandidateNoveltyDecision,
    ProofRunSnapshot,
)


def _candidate(index: int) -> ProofCandidate:
    return ProofCandidate(
        theorem_id=f"candidate-{index}",
        statement=f"Novel theorem statement {index}.",
    )


def _validation(
    candidates: list[ProofCandidate],
    *,
    approved_count: int = 3,
) -> ProofCandidateListValidation:
    return ProofCandidateListValidation(
        results=[
            ProofCandidateNoveltyDecision(
                theorem_id=candidate.theorem_id,
                decision=(
                    "approve_novel" if index < approved_count else "reject_not_novel"
                ),
                reasoning=f"Independent decision {index}.",
            )
            for index, candidate in enumerate(candidates)
        ],
        feedback="Replace the rejected target.",
    )


async def _run_gate(
    candidates: list[ProofCandidate],
    *,
    checkpoint_callback=None,
    checkpoint_candidate_list_state=None,
    checkpoint_processed_candidate_ids=None,
    proof_run_context=None,
    trigger: str = "automatic",
    approved_count: int = 3,
) -> tuple[list[tuple[str, dict]], AsyncMock]:
    events: list[tuple[str, dict]] = []
    stopped = False

    async def broadcast(event_type: str, payload: dict) -> None:
        nonlocal stopped
        events.append((event_type, payload))
        if event_type == "proof_check_candidates_found":
            stopped = True

    validate = AsyncMock(
        return_value=_validation(candidates, approved_count=approved_count)
    )
    with (
        patch.object(system_config, "lean4_enabled", True),
        patch(
            "backend.autonomous.core.proof_verification_stage."
            "ProofCandidateListValidator.validate",
            new=validate,
        ),
        patch.object(
            ProofVerificationStage,
            "_run_lean_pipeline_for_candidate",
            new=AsyncMock(
                side_effect=AssertionError(
                    "Candidate-list validation must complete before proof cost."
                )
            ),
        ),
    ):
        await ProofVerificationStage().run(
            content="Mandatory source.",
            source_type="paper",
            source_id="paper-1",
            source_title="Paper",
            user_prompt="Solve the objective.",
            canonical_user_prompt="Solve the objective.",
            submitter_model="submitter",
            submitter_context=8000,
            submitter_max_tokens=1000,
            validator_model="validator",
            validator_context=8000,
            validator_max_tokens=1000,
            broadcast_fn=broadcast,
            novel_proofs_db=object(),
            theorem_candidates=candidates,
            source_reserved=True,
            release_source_on_exit=False,
            source_reservation_token="reservation",
            should_stop=lambda: stopped,
            checkpoint_callback=checkpoint_callback,
            checkpoint_candidate_list_state=checkpoint_candidate_list_state,
            checkpoint_processed_candidate_ids=checkpoint_processed_candidate_ids,
            proof_round_index=1,
            trigger=trigger,
            run_id="run-1",
            proof_run_context=proof_run_context,
        )
    return events, validate


@pytest.mark.asyncio
async def test_real_candidate_list_gate_precedes_phase_a_and_forwards_approved_subset():
    candidates = [_candidate(index) for index in range(4)]
    checkpoints: list[dict] = []

    async def save_checkpoint(payload: dict) -> None:
        checkpoints.append(payload)

    events, validate = await _run_gate(
        candidates,
        checkpoint_callback=save_checkpoint,
        proof_run_context={
            "proof_run_id": "proof-run-gate",
            "lifecycle_generation": 1,
        },
        approved_count=1,
    )

    names = [event_type for event_type, _ in events]
    assert "proof_candidate_list_review_rejected" not in names
    assert "proof_candidate_list_regeneration_started" not in names
    assert names.index("proof_candidate_list_review_started") < names.index(
        "proof_candidate_list_review_accepted"
    )
    assert names.index("proof_candidate_list_review_accepted") < names.index(
        "proof_check_candidates_found"
    )
    accepted = next(
        payload
        for event_type, payload in events
        if event_type == "proof_candidate_list_review_accepted"
    )
    found = next(
        payload
        for event_type, payload in events
        if event_type == "proof_check_candidates_found"
    )
    assert accepted["proposed_count"] == 4
    assert accepted["approved_count"] == 1
    assert accepted["candidate_ids"] == ["candidate-0"]
    assert checkpoints[-1]["candidate_list_review"]["approved_candidate_ids"] == (
        accepted["candidate_ids"]
    )
    assert found["approved_count"] == 1
    validate.assert_awaited_once()


@pytest.mark.asyncio
async def test_real_autonomous_candidate_list_checkpoint_survives_lifecycle_restart(
    tmp_path,
):
    candidates = [_candidate(index) for index in range(4)]
    workflow_path = tmp_path / "workflow_state.json"
    metadata = ResearchMetadata()
    metadata._workflow_state_path = workflow_path
    await metadata.begin_lifecycle("run-1")

    async def save_checkpoint(payload: dict) -> None:
        await metadata.save_proof_checkpoint(payload)

    await _run_gate(
        candidates,
        checkpoint_callback=save_checkpoint,
        proof_run_context={
            "proof_run_id": "durable-autonomous-run",
            "lifecycle_generation": 1,
        },
    )
    persisted = await metadata.get_proof_checkpoint(
        "paper",
        "paper-1",
        "automatic",
    )
    assert persisted is not None
    approved_state = persisted["candidate_list_review"]

    restarted = ResearchMetadata()
    restarted._workflow_state_path = workflow_path
    await restarted.begin_lifecycle("run-1")
    loaded = await restarted.get_proof_checkpoint(
        "paper",
        "paper-1",
        "automatic",
    )
    assert loaded is not None

    with patch(
        "backend.autonomous.core.proof_verification_stage."
        "ProofCandidateListValidator.validate",
        new=AsyncMock(
            side_effect=AssertionError(
                "Same-round Autonomous restart must reuse approved review state."
            )
        ),
    ):
        events, validate = await _run_gate(
            candidates[:3],
            checkpoint_candidate_list_state=loaded["candidate_list_review"],
            proof_run_context={
                "proof_run_id": "durable-autonomous-run",
                "lifecycle_generation": 2,
            },
        )
    validate.assert_not_awaited()
    found = next(
        payload
        for event_type, payload in events
        if event_type == "proof_check_candidates_found"
    )
    assert found["approved_count"] == len(approved_state["approved_candidate_ids"])

    stale_scope = _candidate_list_review_scope(
        source_type="paper",
        source_id="paper-1",
        run_id="run-1",
        trigger="automatic_round_2",
        proof_round_index=2,
        proof_run_context={
            "proof_run_id": "durable-autonomous-run",
            "lifecycle_generation": 2,
        },
    )
    assert stale_scope != approved_state["review_scope"]


@pytest.mark.asyncio
async def test_real_manual_candidate_list_checkpoint_is_same_process_only():
    candidates = [_candidate(index) for index in range(4)]
    manager = ProofRunManager()
    control = ProofRunControl(
        snapshot=ProofRunSnapshot(
            proof_run_id="manual-proof-run",
            run_mode="loop_with_pruning",
            scope="manual",
            source_type="paper",
            source_id="paper-1",
            proof_store_id="manual:active",
            run_id="manual-run",
            lifecycle_generation=1,
            status="running",
        )
    )

    async def save_checkpoint(payload: dict) -> None:
        assert await manager.save_candidate_checkpoint(control, payload)

    await _run_gate(
        candidates,
        checkpoint_callback=save_checkpoint,
        proof_run_context={
            "proof_run_id": control.snapshot.proof_run_id,
            "lifecycle_generation": control.snapshot.lifecycle_generation,
        },
    )
    saved = await manager.load_candidate_checkpoint(control)
    assert saved is not None

    with patch(
        "backend.autonomous.core.proof_verification_stage."
        "ProofCandidateListValidator.validate",
        new=AsyncMock(
            side_effect=AssertionError(
                "Same-process same-round retry must reuse approved review state."
            )
        ),
    ):
        events, validate = await _run_gate(
            candidates[:3],
            checkpoint_candidate_list_state=saved["candidate_list_review"],
            proof_run_context={
                "proof_run_id": control.snapshot.proof_run_id,
                "lifecycle_generation": control.snapshot.lifecycle_generation,
            },
        )
    validate.assert_not_awaited()
    assert any(
        event_type == "proof_check_candidates_found" for event_type, _ in events
    )

    await manager.complete_round(
        control,
        round_number=1,
        candidate_checkpoint_reference="",
    )
    assert await manager.load_candidate_checkpoint(control) is None

    fresh_manager = ProofRunManager()
    fresh_control = ProofRunControl(snapshot=control.snapshot.model_copy())
    assert await fresh_manager.load_candidate_checkpoint(fresh_control) is None

    review_scope = _candidate_list_review_scope(
        source_type="paper",
        source_id="paper-1",
        run_id="run-1",
        trigger="automatic",
        proof_round_index=1,
        proof_run_context={
            "proof_run_id": control.snapshot.proof_run_id,
            "lifecycle_generation": 99,
        },
    )
    assert review_scope == saved["candidate_list_review"]["review_scope"]
    assert saved["candidate_list_review"]["list_fingerprint"] == (
        _candidate_list_fingerprint(candidates, "paper", "paper-1")
    )
