from __future__ import annotations

import pytest

from tests.workflow_harness.actions import (
    attempt_disabled_lean_checkpoint,
    assistant_retrieve_non_blocking,
    assistant_stagnant_pack_backoff,
    clear,
    complete_brainstorm,
    complete_brainstorm_with_generated_paper,
    complete_topic_exploration,
    disable_session_history,
    emit_context_overflow_contract_events,
    emit_frontend_scoped_event,
    emit_registered_proof_verified,
    force_paper_writing,
    prepare_prompt_context,
    prepare_validator_prompt_with_live_assistant,
    prune_paper,
    reject_mandatory_source_overflow,
    refresh_assistant_pack,
    reset_credit,
    resolve_runtime_path,
    resume,
    resume_completed_brainstorm,
    run_actions,
    run_manual_proof_check,
    run_smt_hint_generation,
    simulate_credit_exhaustion,
    simulate_provider_output_truncation,
    start_autonomous_papers_and_proofs,
    start_autonomous_papers_only,
    start_autonomous_no_outputs,
    start_autonomous_proofs_only,
    start_child_task,
    start_manual_compiler,
    stop,
    stale_child_output_arrives_after_parent_action,
    try_hosted_desktop_only_route,
    verify_rag_source_exclusion,
)
from tests.workflow_harness.invariant_catalog import (
    FIRST_BUILD_FIELDS,
    INVARIANT_CATALOG,
    INVARIANTS_BY_ID,
    assert_invariant,
    invariant_ids_for_adapter,
)
from tests.workflow_harness.model import WorkflowEvent, WorkflowMode, WorkflowModel, WorkflowPhase


EXPECTED_FIRST_BUILD_FIELDS = {
    "runtime_exclusivity",
    "proof_runtime_gating",
    "allowed_outputs",
    "provider_pause_resume",
    "assistant_memory",
    "proof_scope_isolation",
}

PLAN2_INVARIANT_IDS = {
    "runtime.single_active_workflow",
    "runtime.child_tasks_count_as_active",
    "runtime.parent_action_fences_child_outputs",
    "proof_runtime.no_lean_when_disabled",
    "proof_runtime.no_smt_when_disabled",
    "proof_runtime.hosted_proof_settings_unavailable",
    "proof_runtime.truncation_is_attempt_failure",
    "outputs.at_least_one_output_enabled",
    "outputs.proofs_only_no_paper_phase",
    "outputs.papers_only_skips_proof_work",
    "provider.pause_preserves_checkpoint",
    "provider.stop_resume_preserves_pause",
    "provider.reset_wakes_without_corrupting_checkpoint",
    "assistant.no_validator_injection",
    "assistant.disable_clears_live_pack",
    "assistant.non_blocking_retrieval",
    "assistant.stagnant_pack_backoff_not_shutdown",
    "proof_scope.manual_not_in_autonomous_current",
    "proof_scope.autonomous_events_not_manual",
    "proof_scope.manual_clear_archives_active",
    "proof_scope.generated_appendices_stripped_from_prompts",
    "state.clear_removes_active_preserves_history",
    "state.completed_brainstorm_no_replay_handoff",
    "state.pruned_papers_excluded_from_context",
    "state.runtime_roots_are_active_roots",
    "events.frontend_events_include_scope_phase",
    "events.context_overflow_route_identity",
    "events.context_overflow_persists_across_reload",
    "events.context_overflow_terminal_stop_once",
    "events.proof_context_overflow_nonfatal",
    "events.proof_verified_after_registration",
    "api.hosted_desktop_only_routes_unavailable",
    "prompt.user_prompt_direct_injected",
    "prompt.validator_excludes_assistant_memory",
    "prompt.proof_source_context_required",
    "prompt.direct_sources_excluded_from_rag",
    "prompt.mandatory_source_overflow_fails_visible",
    "prompt.generated_appendices_stripped",
}


SCENARIOS_BY_INVARIANT = {
    "runtime.single_active_workflow": [start_autonomous_papers_and_proofs],
    "runtime.child_tasks_count_as_active": [start_autonomous_papers_and_proofs, start_child_task],
    "runtime.parent_action_fences_child_outputs": [
        start_autonomous_papers_and_proofs,
        complete_topic_exploration,
        start_child_task,
        force_paper_writing,
        stale_child_output_arrives_after_parent_action,
    ],
    "proof_runtime.no_lean_when_disabled": [
        start_autonomous_papers_only,
        attempt_disabled_lean_checkpoint,
    ],
    "proof_runtime.no_smt_when_disabled": [
        start_autonomous_papers_and_proofs,
        run_smt_hint_generation,
    ],
    "proof_runtime.hosted_proof_settings_unavailable": [try_hosted_desktop_only_route],
    "proof_runtime.truncation_is_attempt_failure": [simulate_provider_output_truncation],
    "outputs.at_least_one_output_enabled": [start_autonomous_no_outputs],
    "outputs.proofs_only_no_paper_phase": [
        start_autonomous_proofs_only,
        complete_topic_exploration,
        complete_brainstorm,
    ],
    "outputs.papers_only_skips_proof_work": [
        start_autonomous_papers_only,
        complete_topic_exploration,
        complete_brainstorm,
    ],
    "provider.pause_preserves_checkpoint": [
        start_autonomous_papers_and_proofs,
        complete_topic_exploration,
        simulate_credit_exhaustion,
        complete_brainstorm,
        stop,
        resume,
    ],
    "provider.stop_resume_preserves_pause": [
        start_autonomous_papers_and_proofs,
        complete_topic_exploration,
        simulate_credit_exhaustion,
        complete_brainstorm,
    ],
    "provider.reset_wakes_without_corrupting_checkpoint": [
        start_autonomous_papers_and_proofs,
        complete_topic_exploration,
        simulate_credit_exhaustion,
        complete_brainstorm,
        reset_credit,
    ],
    "assistant.no_validator_injection": [
        start_autonomous_papers_and_proofs,
        refresh_assistant_pack,
        prepare_validator_prompt_with_live_assistant,
    ],
    "assistant.disable_clears_live_pack": [
        start_autonomous_papers_and_proofs,
        refresh_assistant_pack,
        disable_session_history,
    ],
    "assistant.non_blocking_retrieval": [
        start_autonomous_papers_and_proofs,
        assistant_retrieve_non_blocking,
    ],
    "assistant.stagnant_pack_backoff_not_shutdown": [
        start_autonomous_papers_and_proofs,
        assistant_stagnant_pack_backoff,
    ],
    "proof_scope.manual_not_in_autonomous_current": [
        start_manual_compiler,
        run_manual_proof_check,
    ],
    "proof_scope.autonomous_events_not_manual": [
        start_autonomous_papers_and_proofs,
        complete_topic_exploration,
        emit_registered_proof_verified,
    ],
    "proof_scope.manual_clear_archives_active": [
        start_manual_compiler,
        run_manual_proof_check,
        clear,
    ],
    "proof_scope.generated_appendices_stripped_from_prompts": [prepare_prompt_context],
    "state.clear_removes_active_preserves_history": [
        start_manual_compiler,
        run_manual_proof_check,
        clear,
    ],
    "state.completed_brainstorm_no_replay_handoff": [
        complete_brainstorm_with_generated_paper,
        resume_completed_brainstorm,
    ],
    "state.pruned_papers_excluded_from_context": [prune_paper],
    "state.runtime_roots_are_active_roots": [resolve_runtime_path],
    "events.frontend_events_include_scope_phase": [
        start_autonomous_papers_and_proofs,
        emit_frontend_scoped_event,
    ],
    "events.context_overflow_route_identity": [emit_context_overflow_contract_events],
    "events.context_overflow_persists_across_reload": [emit_context_overflow_contract_events],
    "events.context_overflow_terminal_stop_once": [emit_context_overflow_contract_events],
    "events.proof_context_overflow_nonfatal": [emit_context_overflow_contract_events],
    "events.proof_verified_after_registration": [
        start_autonomous_papers_and_proofs,
        emit_registered_proof_verified,
    ],
    "api.hosted_desktop_only_routes_unavailable": [try_hosted_desktop_only_route],
    "prompt.user_prompt_direct_injected": [prepare_prompt_context],
    "prompt.validator_excludes_assistant_memory": [prepare_prompt_context],
    "prompt.proof_source_context_required": [prepare_prompt_context],
    "prompt.direct_sources_excluded_from_rag": [verify_rag_source_exclusion],
    "prompt.mandatory_source_overflow_fails_visible": [reject_mandatory_source_overflow],
    "prompt.generated_appendices_stripped": [prepare_prompt_context],
}


def _make_invariant_violation(model: WorkflowModel, invariant_id: str) -> None:
    model.record("inject_invariant_violation", invariant_id=invariant_id)
    if invariant_id == "runtime.single_active_workflow":
        model.active_owners = {WorkflowMode.AUTONOMOUS, WorkflowMode.LEANOJ}
    elif invariant_id == "runtime.child_tasks_count_as_active":
        model.pending_child_tasks = 1
    elif invariant_id == "runtime.parent_action_fences_child_outputs":
        model.stale_child_outputs_fenced = False
    elif invariant_id == "proof_runtime.no_lean_when_disabled":
        model.lean.blocked_invocations = 1
    elif invariant_id == "proof_runtime.no_smt_when_disabled":
        model.smt.blocked_invocations = 1
    elif invariant_id == "proof_runtime.hosted_proof_settings_unavailable":
        model.hosted_desktop_route_attempted = True
        model.hosted_desktop_route_unavailable = False
    elif invariant_id == "proof_runtime.truncation_is_attempt_failure":
        model.truncation_failures = model.provider_pause_from_truncation = 1
    elif invariant_id == "outputs.at_least_one_output_enabled":
        model.allow_mathematical_proofs = model.allow_research_papers = False
    elif invariant_id == "outputs.proofs_only_no_paper_phase":
        model.mode = WorkflowMode.AUTONOMOUS
        model.allow_mathematical_proofs = True
        model.allow_research_papers = False
        model.phase = WorkflowPhase.PAPER_WRITING
    elif invariant_id == "outputs.papers_only_skips_proof_work":
        model.mode = WorkflowMode.AUTONOMOUS
        model.allow_mathematical_proofs = False
        model.lean.invocations = 1
    elif invariant_id == "provider.pause_preserves_checkpoint":
        model.phase = WorkflowPhase.PAUSED
        model.checkpoint = {"paused": True}
    elif invariant_id == "provider.stop_resume_preserves_pause":
        model.checkpoint = {"paused": True, "stopped": True}
    elif invariant_id == "provider.reset_wakes_without_corrupting_checkpoint":
        model.provider.pause_count = 1
        model.checkpoint = {"paused": False}
    elif invariant_id in {"assistant.no_validator_injection", "prompt.validator_excludes_assistant_memory"}:
        if invariant_id.startswith("assistant."):
            model.assistant.validator_injections = 1
        else:
            model.validator_prompt_has_assistant_memory = True
    elif invariant_id == "assistant.disable_clears_live_pack":
        model.assistant.enabled = False
        model.assistant.live_pack = ("stale-proof",)
    elif invariant_id == "assistant.non_blocking_retrieval":
        model.assistant.blocked_parent_count = 1
    elif invariant_id == "assistant.stagnant_pack_backoff_not_shutdown":
        model.assistant.stagnant_backoff_count = model.assistant.shutdown_count = 1
    elif invariant_id == "proof_scope.manual_not_in_autonomous_current":
        model.autonomous_proofs = model.manual_proofs_active = {"same-proof"}
    elif invariant_id == "proof_scope.autonomous_events_not_manual":
        model.events.append(WorkflowEvent("proof_verified", {"scope": "manual", "proof_id": "auto-proof"}))
    elif invariant_id == "proof_scope.manual_clear_archives_active":
        model.manual_proofs_before_clear = {"manual-proof"}
    elif invariant_id in {
        "proof_scope.generated_appendices_stripped_from_prompts",
        "prompt.generated_appendices_stripped",
    }:
        model.generated_appendices_stripped = False
    elif invariant_id == "state.clear_removes_active_preserves_history":
        model.active_state_cleared = True
        model.history_preserved = False
    elif invariant_id == "state.completed_brainstorm_no_replay_handoff":
        model.completed_brainstorm_generated_paper = model.replayed_completed_brainstorm_handoff = True
    elif invariant_id == "state.pruned_papers_excluded_from_context":
        model.pruned_papers = model.model_context_papers = {"paper-1"}
    elif invariant_id == "state.runtime_roots_are_active_roots":
        model.runtime_root_violations = 1
    elif invariant_id == "events.frontend_events_include_scope_phase":
        model.events.append(WorkflowEvent("proof_progress", {}))
    elif invariant_id == "events.context_overflow_route_identity":
        model.events.append(WorkflowEvent("context_overflow_error", {}))
    elif invariant_id == "events.context_overflow_persists_across_reload":
        model.events.append(WorkflowEvent("context_overflow_error", {"fatal": True}))
    elif invariant_id == "events.context_overflow_terminal_stop_once":
        model.events.append(WorkflowEvent("context_overflow_error", {"fatal": True}))
    elif invariant_id == "events.proof_context_overflow_nonfatal":
        model.events.append(WorkflowEvent("proof_context_overflow", {"fatal": True}))
    elif invariant_id == "events.proof_verified_after_registration":
        model.events.append(WorkflowEvent("proof_verified", {"proof_id": "missing-proof"}))
    elif invariant_id == "api.hosted_desktop_only_routes_unavailable":
        model.hosted_route_attempts = {"pdf"}
    elif invariant_id == "prompt.user_prompt_direct_injected":
        model.prompt_user_direct_injected = False
    elif invariant_id == "prompt.proof_source_context_required":
        model.proof_source_context_present = False
    elif invariant_id == "prompt.direct_sources_excluded_from_rag":
        model.direct_sources_excluded_from_rag = False
    elif invariant_id == "prompt.mandatory_source_overflow_fails_visible":
        model.mandatory_source_overflow_visible = False
    else:
        raise AssertionError(f"No negative fixture for {invariant_id}")


def test_first_build_catalog_has_stable_ids_fields_and_model_coverage():
    invariant_ids = [spec.invariant_id for spec in INVARIANT_CATALOG]

    assert len(invariant_ids) == len(set(invariant_ids))
    assert EXPECTED_FIRST_BUILD_FIELDS.issubset(FIRST_BUILD_FIELDS)
    assert set(invariant_ids) == set(SCENARIOS_BY_INVARIANT)
    assert invariant_ids_for_adapter("model") == set(invariant_ids)
    assert set(INVARIANTS_BY_ID) == PLAN2_INVARIANT_IDS

    for spec in INVARIANT_CATALOG:
        assert spec.description
        assert spec.crossed_fields


@pytest.mark.parametrize("spec", INVARIANT_CATALOG, ids=lambda spec: spec.invariant_id)
def test_first_build_catalog_invariants_run_against_model_scenarios(tmp_path, spec):
    model = WorkflowModel(runtime_root=tmp_path / spec.invariant_id.replace(".", "_"))
    if spec.invariant_id.startswith("proof_scope.manual_"):
        model.lean.enabled = True

    run_actions(model, SCENARIOS_BY_INVARIANT[spec.invariant_id])

    assert_invariant(model, spec.invariant_id)


@pytest.mark.parametrize("spec", INVARIANT_CATALOG, ids=lambda spec: spec.invariant_id)
def test_every_catalog_checker_rejects_a_minimal_violation_with_replay(tmp_path, spec):
    model = WorkflowModel(runtime_root=tmp_path / "negative")
    _make_invariant_violation(model, spec.invariant_id)

    with pytest.raises(AssertionError, match="Replay:") as exc_info:
        assert_invariant(model, spec.invariant_id)

    assert "inject_invariant_violation" in str(exc_info.value)
