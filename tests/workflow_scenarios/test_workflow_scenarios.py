from __future__ import annotations

from tests.workflow_harness.actions import (
    clear,
    complete_brainstorm,
    complete_topic_exploration,
    disable_session_history,
    refresh_assistant_pack,
    reset_credit,
    resume,
    run_actions,
    run_manual_proof_check,
    simulate_credit_exhaustion,
    start_autonomous_papers_and_proofs,
    start_autonomous_proofs_only,
    start_manual_compiler,
    stop,
)
from tests.workflow_harness.invariants import assert_all_invariants
from tests.workflow_harness.model import WorkflowModel, WorkflowPhase


def test_proofs_only_brainstorm_completion_returns_to_topic_exploration(tmp_path):
    model = WorkflowModel(runtime_root=tmp_path)

    run_actions(
        model,
        [
            start_autonomous_proofs_only,
            complete_topic_exploration,
            complete_brainstorm,
        ],
    )

    assert model.phase is WorkflowPhase.TOPIC_EXPLORATION
    assert "auto-proof-1" in model.autonomous_proofs
    assert model.lean.invocations == 1


def test_provider_credit_pause_preserves_checkpoint_across_stop_resume(tmp_path):
    model = WorkflowModel(runtime_root=tmp_path)

    run_actions(
        model,
        [
            start_autonomous_papers_and_proofs,
            complete_topic_exploration,
            simulate_credit_exhaustion,
            complete_brainstorm,
            stop,
            resume,
            reset_credit,
        ],
    )

    assert model.phase is WorkflowPhase.TIER1_AGGREGATION
    assert model.checkpoint.get("paused") is False
    assert_all_invariants(model)


def test_disabling_session_history_clears_live_assistant_pack(tmp_path):
    model = WorkflowModel(runtime_root=tmp_path)

    run_actions(
        model,
        [
            start_autonomous_papers_and_proofs,
            refresh_assistant_pack,
            disable_session_history,
        ],
    )

    assert model.assistant.live_pack == ()


def test_manual_clear_archives_active_manual_proofs_without_autonomous_leak(tmp_path):
    model = WorkflowModel(runtime_root=tmp_path)
    model.lean.enabled = True

    run_actions(
        model,
        [
            start_manual_compiler,
            run_manual_proof_check,
            clear,
        ],
    )

    assert model.manual_proofs_active == set()
    assert model.manual_proofs_archived == {"manual-proof-1"}
    assert model.autonomous_proofs == set()

