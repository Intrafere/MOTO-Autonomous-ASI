from __future__ import annotations

from dataclasses import replace

from tests.workflow_harness.coverage_metadata import InteractionCoverage
from tests.workflow_harness.source_mappings import BUILD_07_REAL_ADAPTER_COVERAGE


def _node(file_name: str, test_name: str) -> str:
    return f"tests/workflow_real_adapters/{file_name}::{test_name}"


PROOF_ROUTE_COVERAGE = (
    InteractionCoverage(
        scenario_id="real_manual_compiler_proof_only_uses_rigor_settings_and_manual_scope",
        fields=("allowed_outputs", "proof_scope_isolation", "prompt_context"),
        invariants=("proof_scope.manual_not_in_autonomous_current",),
        adapter="real_coordinator",
        result="passed",
        test_file="tests/workflow_real_adapters/test_proof_routes.py",
        evidence=("manual_scope", "rigor_settings"),
    ),
    InteractionCoverage(
        scenario_id="real_autonomous_proofs_only_handoff_returns_to_topic_exploration",
        fields=("allowed_outputs", "proof_runtime_gating", "workflow_filesystem_state"),
        invariants=("outputs.proofs_only_no_paper_phase",),
        adapter="real_coordinator",
        result="passed",
        test_file="tests/workflow_real_adapters/test_proof_routes.py",
        evidence=("proofs_only_handoff", "topic_exploration"),
    ),
    InteractionCoverage(
        scenario_id="real_autonomous_provider_credit_pause_preserves_checkpoint_and_resumes",
        fields=("provider_pause_resume", "workflow_filesystem_state", "proof_runtime_gating"),
        invariants=("provider.pause_preserves_checkpoint",),
        adapter="real_coordinator",
        result="passed",
        test_file="tests/workflow_real_adapters/test_proof_routes.py",
        evidence=("provider_pause", "checkpoint_resume"),
    ),
    InteractionCoverage(
        scenario_id="real_leanoj_master_proof_stop_resume_preserves_isolated_state",
        fields=("workflow_filesystem_state", "proof_scope_isolation", "runtime_exclusivity"),
        invariants=("proof_scope.autonomous_events_not_manual",),
        adapter="real_coordinator",
        result="passed",
        test_file="tests/workflow_real_adapters/test_proof_routes.py",
        evidence=("master_proof_persisted", "session_isolation"),
    ),
)

ROUTE_CONFLICT_COVERAGE = (
    InteractionCoverage(
        scenario_id="real_route_start_conflicts_preserve_single_workflow",
        fields=("runtime_exclusivity", "websocket_api_contracts"),
        invariants=("runtime.single_active_workflow",),
        adapter="real_route",
        result="passed",
        test_file="tests/workflow_real_adapters/test_route_conflicts.py",
        evidence=("route_conflict_response", "single_active_workflow"),
    ),
    InteractionCoverage(
        scenario_id="real_route_pending_child_activity_counts_as_active",
        fields=("runtime_exclusivity", "workflow_filesystem_state"),
        invariants=("runtime.single_active_workflow",),
        adapter="real_route",
        result="passed",
        test_file="tests/workflow_real_adapters/test_route_conflicts.py",
        evidence=("pending_child_activity", "route_conflict_response"),
    ),
)

HIGH_VALUE_GAP_COVERAGE = (
    InteractionCoverage(
        scenario_id="real_parent_action_fencing_unavailable_without_production_seam",
        fields=("runtime_exclusivity", "workflow_filesystem_state", "websocket_api_contracts"),
        invariants=("runtime.parent_action_fences_child_outputs",),
        adapter="real_coordinator",
        result="blocked",
        test_file="tests/workflow_real_adapters/test_high_value_scenarios.py",
        diagnostics={
            "reason": (
                "The invariant catalog declares model-only support and no existing isolated "
                "production seam exposes stale child output acceptance after parent takeover."
            )
        },
    ),
    InteractionCoverage(
        scenario_id="real_provider_stop_reset_checkpoint_unavailable_without_wait_seam",
        fields=("provider_pause_resume", "workflow_filesystem_state", "runtime_exclusivity"),
        invariants=("provider.stop_resume_preserves_pause",),
        adapter="real_coordinator",
        result="blocked",
        test_file="tests/workflow_real_adapters/test_high_value_scenarios.py",
        diagnostics={
            "reason": (
                "Existing coordinator coverage observes pause/reset resume, but no bounded "
                "test seam independently interleaves user Stop while the provider wait is active."
            )
        },
    ),
)

HIGH_VALUE_REAL_COVERAGE = (
    InteractionCoverage(
        scenario_id="real_disabled_proof_runtime_skips_autonomous_checkpoint",
        fields=("proof_runtime_gating", "allowed_outputs", "workflow_filesystem_state"),
        invariants=("proof_runtime.no_lean_when_disabled",),
        adapter="real_coordinator",
        result="passed",
        test_file="tests/workflow_real_adapters/test_high_value_scenarios.py",
        evidence=("disabled_runtime", "zero_proof_stage_calls"),
    ),
    InteractionCoverage(
        scenario_id="real_hosted_proof_settings_returns_desktop_unavailable",
        fields=("proof_runtime_gating", "websocket_api_contracts", "allowed_outputs"),
        invariants=("proof_runtime.hosted_proof_settings_unavailable",),
        adapter="real_route",
        result="passed",
        test_file="tests/workflow_real_adapters/test_high_value_scenarios.py",
        evidence=("hosted_mode", "http_501"),
    ),
)

BUILD_BC_REAL_COVERAGE = (
    InteractionCoverage(
        scenario_id="real_manual_aggregator_route_lifecycle_output_and_temp_root",
        fields=("workflow_filesystem_state", "runtime_exclusivity", "prompt_context"),
        invariants=("runtime.single_active_workflow",),
        adapter="real_route",
        result="passed",
        test_file="tests/workflow_real_adapters/test_build_bc_lifecycles.py",
        evidence=("route_start_stop", "saved_output", "temp_root_containment"),
    ),
    InteractionCoverage(
        scenario_id="real_manual_aggregator_clear_archives_before_clear",
        fields=("proof_scope_isolation", "workflow_filesystem_state"),
        invariants=("proof_scope.manual_clear_archives_active",),
        adapter="real_route",
        result="passed",
        test_file="tests/workflow_real_adapters/test_build_bc_lifecycles.py",
        evidence=("archive_before_clear", "upload_cleanup"),
    ),
    InteractionCoverage(
        scenario_id="real_actual_route_start_conflict_matrix_no_side_effects",
        fields=("runtime_exclusivity", "workflow_filesystem_state"),
        invariants=("runtime.single_active_workflow", "runtime.child_tasks_count_as_active"),
        adapter="real_route",
        result="passed",
        test_file="tests/workflow_real_adapters/test_build_bc_lifecycles.py",
        evidence=("sixteen_actual_route_calls", "pending_owner", "no_initialize_side_effects"),
    ),
    InteractionCoverage(
        scenario_id="real_manual_aggregator_durable_hydration_and_clear_blocker",
        fields=("workflow_filesystem_state", "proof_scope_isolation", "prompt_context"),
        invariants=("state.runtime_roots_are_active_roots",),
        adapter="real_coordinator",
        result="passed",
        test_file="tests/workflow_real_adapters/test_build_bc_lifecycles.py",
        evidence=("persisted_result_hydration", "temp_root", "blocked_clear_no_archive"),
    ),
    InteractionCoverage(
        scenario_id="real_manual_compiler_rejects_no_allowed_outputs",
        fields=("allowed_outputs", "runtime_exclusivity"),
        invariants=("outputs.at_least_one_output_enabled",),
        adapter="real_route",
        result="passed",
        test_file="tests/workflow_real_adapters/test_build_bc_lifecycles.py",
        evidence=("http_400", "coordinator_not_started"),
    ),
    InteractionCoverage(
        scenario_id="real_generated_proof_appendix_stripping",
        fields=("prompt_context", "proof_scope_isolation"),
        invariants=("proof_scope.generated_appendices_stripped_from_prompts",),
        adapter="real_coordinator",
        result="passed",
        test_file="tests/workflow_real_adapters/test_build_bc_lifecycles.py",
        evidence=("brainstorm_appendix_stripped", "paper_appendix_stripped"),
    ),
    InteractionCoverage(
        scenario_id="real_manual_compiler_papers_lifecycle_and_clear_archive",
        fields=("allowed_outputs", "workflow_filesystem_state", "proof_scope_isolation"),
        invariants=(
            "proof_scope.manual_clear_archives_active",
            "state.clear_removes_active_preserves_history",
        ),
        adapter="real_route",
        result="passed",
        test_file="tests/workflow_real_adapters/test_build_bc_lifecycles.py",
        evidence=("papers_only_start_stop", "archive_before_appendix_and_paper_clear", "temp_archive_root"),
    ),
    InteractionCoverage(
        scenario_id="real_manual_compiler_proof_only_background_ownership",
        fields=("allowed_outputs", "runtime_exclusivity", "proof_runtime_gating"),
        invariants=("runtime.child_tasks_count_as_active",),
        adapter="real_route",
        result="passed",
        test_file="tests/workflow_real_adapters/test_build_bc_lifecycles.py",
        evidence=("actual_proof_only_start_route", "live_background_task", "second_start_conflict"),
    ),
    InteractionCoverage(
        scenario_id="real_leanoj_coordinator_skip_force_clear_and_intermediate_edit",
        fields=("runtime_exclusivity", "workflow_filesystem_state", "proof_scope_isolation"),
        invariants=("state.runtime_roots_are_active_roots",),
        adapter="real_coordinator",
        result="passed",
        test_file="tests/workflow_real_adapters/test_build_bc_lifecycles.py",
        evidence=("real_skip_consume", "real_force_consume", "state_preserved_then_cleared", "master_proof_temp_root"),
    ),
    InteractionCoverage(
        scenario_id="real_shared_start_guard_representative_race",
        fields=("runtime_exclusivity", "websocket_api_contracts"),
        invariants=("runtime.single_active_workflow",),
        adapter="real_route",
        result="passed",
        test_file="tests/workflow_real_adapters/test_build_bc_lifecycles.py",
        evidence=("concurrent_start_race", "one_winner"),
    ),
    InteractionCoverage(
        scenario_id="real_leanoj_full_final_loop_not_safely_bounded",
        fields=("proof_runtime_gating", "workflow_filesystem_state", "websocket_api_contracts"),
        invariants=("events.proof_verified_after_registration",),
        adapter="real_coordinator",
        result="blocked",
        test_file="tests/workflow_real_adapters/test_build_bc_lifecycles.py",
        diagnostics={
            "reason": (
                "The production final loop combines unbounded model iteration, Lean execution, "
                "semantic review, persistence, and registration without a bounded route-level seam. "
                "Direct registration ordering is covered elsewhere; publishing a full lifecycle pass "
                "would require faking the transition owner rather than observing it."
            )
        },
    ),
)

BUILD_D_REAL_COVERAGE = (
    InteractionCoverage(
        scenario_id="real_manual_aggregator_overflow_identity_persists_across_reload",
        fields=("websocket_api_contracts", "workflow_filesystem_state", "provider_pause_resume"),
        invariants=(
            "events.context_overflow_route_identity",
            "events.context_overflow_persists_across_reload",
        ),
        adapter="real_coordinator",
        result="passed",
        test_file="tests/workflow_real_adapters/test_build_d_overflow_lifecycles.py",
        evidence=("configured_effective_route", "event_log_reload", "temp_root"),
    ),
    InteractionCoverage(
        scenario_id="real_autonomous_overflow_terminal_stop_once",
        fields=("websocket_api_contracts", "workflow_filesystem_state", "runtime_exclusivity"),
        invariants=(
            "events.context_overflow_route_identity",
            "events.context_overflow_terminal_stop_once",
        ),
        adapter="real_coordinator",
        result="passed",
        test_file="tests/workflow_real_adapters/test_build_d_overflow_lifecycles.py",
        evidence=("configured_effective_route", "single_terminal_stop"),
    ),
    InteractionCoverage(
        scenario_id="real_proof_context_overflow_is_scoped_nonfatal",
        fields=("websocket_api_contracts", "proof_runtime_gating", "provider_pause_resume"),
        invariants=(
            "events.context_overflow_route_identity",
            "events.proof_context_overflow_nonfatal",
        ),
        adapter="real_coordinator",
        result="passed",
        test_file="tests/workflow_real_adapters/test_build_d_overflow_lifecycles.py",
        evidence=("configured_effective_route", "fatal_false", "no_parent_stop"),
    ),
    InteractionCoverage(
        scenario_id="real_proof_scope_matrix_isolated_under_temp_root",
        fields=("proof_scope_isolation", "workflow_filesystem_state", "websocket_api_contracts"),
        invariants=("proof_scope.manual_not_in_autonomous_current",),
        adapter="real_route",
        result="passed",
        test_file="tests/workflow_real_adapters/test_build_d_proof_scope_filesystem.py",
        evidence=(
            "autonomous_current_and_history",
            "manual_active_and_history",
            "leanoj_current_and_history",
            "pairwise_sentinels",
        ),
    ),
    InteractionCoverage(
        scenario_id="real_manual_archive_and_leanoj_clear_preserve_scope_contract",
        fields=("proof_scope_isolation", "workflow_filesystem_state", "assistant_memory"),
        invariants=(
            "proof_scope.manual_clear_archives_active",
            "state.clear_removes_active_preserves_history",
        ),
        adapter="real_coordinator",
        result="passed",
        test_file="tests/workflow_real_adapters/test_build_d_proof_scope_filesystem.py",
        evidence=("physical_manual_archive", "active_store_empty", "leanoj_root_removed"),
    ),
    InteractionCoverage(
        scenario_id="real_pruned_paper_preserved_but_removed_from_active_context",
        fields=("workflow_filesystem_state", "prompt_context"),
        invariants=("state.pruned_papers_excluded_from_context",),
        adapter="real_route",
        result="passed",
        test_file="tests/workflow_real_adapters/test_build_d_proof_scope_filesystem.py",
        evidence=(
            "pruned_history_downloadable",
            "active_enumeration_empty",
            "brainstorm_reference_removed",
            "rag_source_removed",
            "temp_root",
        ),
    ),
)

BUILD_E_REAL_COVERAGE = (
    InteractionCoverage(
        scenario_id="real_direct_source_rag_exclusion_matrix",
        fields=("prompt_context", "workflow_filesystem_state"),
        invariants=("prompt.direct_sources_excluded_from_rag",),
        adapter="real_coordinator",
        result="passed",
        test_file="tests/workflow_real_adapters/test_build_e_rag_prompt_context.py",
        evidence=(
            "aggregator_direct_and_offloaded_sources",
            "compiler_construction_exclusions",
            "compiler_rigor_exclusions",
            "offloaded_source_retrievable",
        ),
    ),
    InteractionCoverage(
        scenario_id="real_model_visible_prompts_strip_generated_proof_appendices",
        fields=("prompt_context", "proof_scope_isolation", "proof_runtime_gating"),
        invariants=(
            "prompt.generated_appendices_stripped",
            "proof_scope.generated_appendices_stripped_from_prompts",
        ),
        adapter="real_coordinator",
        result="passed",
        test_file="tests/workflow_real_adapters/test_build_e_prompt_boundaries.py",
        evidence=(
            "paper_head_and_tail_preserved",
            "generated_appendix_absent",
            "verified_library_context_present",
            "rigor_message_captured",
        ),
    ),
    InteractionCoverage(
        scenario_id="real_rigor_mandatory_source_overflow_fails_before_model_or_rag",
        fields=("prompt_context", "proof_runtime_gating", "websocket_api_contracts"),
        invariants=("prompt.mandatory_source_overflow_fails_visible",),
        adapter="real_coordinator",
        result="passed",
        test_file="tests/workflow_real_adapters/test_build_e_prompt_boundaries.py",
        evidence=("visible_value_error", "zero_model_calls", "zero_rag_calls", "source_unchanged"),
    ),
)

_UNLINKED_REAL_ADAPTER_COVERAGE = (
    ROUTE_CONFLICT_COVERAGE
    + PROOF_ROUTE_COVERAGE
    + HIGH_VALUE_REAL_COVERAGE
    + BUILD_BC_REAL_COVERAGE
    + BUILD_D_REAL_COVERAGE
    + BUILD_E_REAL_COVERAGE
    + BUILD_07_REAL_ADAPTER_COVERAGE
    + HIGH_VALUE_GAP_COVERAGE
)


_EXPLICIT_EVIDENCE: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "real_manual_compiler_proof_only_uses_rigor_settings_and_manual_scope": (
        (_node("test_proof_routes.py", "test_manual_compiler_proof_only_uses_rigor_settings_and_manual_scope"),),
        ("proof_scope.manual_not_in_autonomous_current",),
    ),
    "real_autonomous_proofs_only_handoff_returns_to_topic_exploration": (
        (_node("test_proof_routes.py", "test_autonomous_proofs_only_brainstorm_handoff_returns_to_topic_exploration"),),
        ("outputs.proofs_only_no_paper_phase",),
    ),
    "real_autonomous_provider_credit_pause_preserves_checkpoint_and_resumes": (
        (_node("test_proof_routes.py", "test_autonomous_provider_credit_pause_preserves_checkpoint_and_resumes"),),
        ("provider.pause_preserves_checkpoint",),
    ),
    "real_leanoj_master_proof_stop_resume_preserves_isolated_state": (
        (_node("test_proof_routes.py", "test_leanoj_master_proof_stop_resume_preserves_isolated_state"),),
        ("proof_scope.autonomous_events_not_manual",),
    ),
    "real_route_start_conflicts_preserve_single_workflow": (
        (_node("test_route_conflicts.py", "test_real_route_start_conflicts_preserve_single_workflow_invariant"),),
        ("runtime.single_active_workflow",),
    ),
    "real_route_pending_child_activity_counts_as_active": (
        (_node("test_route_conflicts.py", "test_real_autonomous_route_counts_pending_child_activity_as_active"),),
        ("runtime.single_active_workflow",),
    ),
    "real_disabled_proof_runtime_skips_autonomous_checkpoint": (
        (_node("test_high_value_scenarios.py", "test_disabled_proof_runtime_skips_autonomous_checkpoint"),),
        ("proof_runtime.no_lean_when_disabled",),
    ),
    "real_hosted_proof_settings_returns_desktop_unavailable": (
        (_node("test_high_value_scenarios.py", "test_hosted_proof_settings_returns_desktop_unavailable"),),
        ("proof_runtime.hosted_proof_settings_unavailable",),
    ),
    "real_manual_aggregator_route_lifecycle_output_and_temp_root": (
        (_node("test_build_bc_lifecycles.py", "test_manual_aggregator_real_route_start_stop_output_and_temp_root"),),
        ("runtime.single_active_workflow",),
    ),
    "real_manual_aggregator_clear_archives_before_clear": (
        (_node("test_build_bc_lifecycles.py", "test_manual_aggregator_clear_archives_before_destructive_clear"),),
        ("proof_scope.manual_clear_archives_active",),
    ),
    "real_actual_route_start_conflict_matrix_no_side_effects": (
        (_node("test_build_bc_lifecycles.py", "test_actual_route_start_conflict_matrix_has_no_start_side_effects"),),
        ("runtime.single_active_workflow", "runtime.child_tasks_count_as_active"),
    ),
    "real_manual_aggregator_durable_hydration_and_clear_blocker": (
        (
            _node("test_build_bc_lifecycles.py", "test_manual_aggregator_durable_results_hydrate_from_temp_root"),
            _node("test_build_bc_lifecycles.py", "test_manual_aggregator_clear_blocker_has_no_destructive_side_effects"),
        ),
        ("state.runtime_roots_are_active_roots",),
    ),
    "real_manual_compiler_rejects_no_allowed_outputs": (
        (_node("test_build_bc_lifecycles.py", "test_manual_compiler_rejects_invalid_output_selection_before_start"),),
        ("outputs.at_least_one_output_enabled",),
    ),
    "real_generated_proof_appendix_stripping": (
        (_node("test_build_bc_lifecycles.py", "test_generated_proof_appendices_are_stripped_without_truncating_body"),),
        ("proof_scope.generated_appendices_stripped_from_prompts",),
    ),
    "real_manual_compiler_papers_lifecycle_and_clear_archive": (
        (
            _node("test_build_bc_lifecycles.py", "test_manual_compiler_real_papers_only_route_start_stop"),
            _node("test_build_bc_lifecycles.py", "test_manual_compiler_clear_archives_before_paper_clear"),
        ),
        ("proof_scope.manual_clear_archives_active", "state.clear_removes_active_preserves_history"),
    ),
    "real_manual_compiler_proof_only_background_ownership": (
        (_node("test_build_bc_lifecycles.py", "test_manual_compiler_proof_only_route_owns_background_task"),),
        ("runtime.child_tasks_count_as_active",),
    ),
    "real_leanoj_coordinator_skip_force_clear_and_intermediate_edit": (
        (
            _node("test_build_bc_lifecycles.py", "test_leanoj_real_coordinator_skip_force_consume_and_route_clear"),
            _node("test_build_bc_lifecycles.py", "test_leanoj_real_intermediate_master_proof_edit_stays_in_temp_root"),
        ),
        ("state.runtime_roots_are_active_roots",),
    ),
    "real_shared_start_guard_representative_race": (
        (_node("test_build_bc_lifecycles.py", "test_actual_compiler_and_aggregator_route_start_race_has_one_winner"),),
        ("runtime.single_active_workflow",),
    ),
    "real_manual_aggregator_overflow_identity_persists_across_reload": (
        (_node("test_build_d_overflow_lifecycles.py", "test_manual_aggregator_fatal_overflow_preserves_route_and_reloads"),),
        ("events.context_overflow_route_identity", "events.context_overflow_persists_across_reload"),
    ),
    "real_autonomous_overflow_terminal_stop_once": (
        (_node("test_build_d_overflow_lifecycles.py", "test_autonomous_fatal_overflow_terminal_stop_emits_exactly_once"),),
        ("events.context_overflow_route_identity", "events.context_overflow_terminal_stop_once"),
    ),
    "real_proof_context_overflow_is_scoped_nonfatal": (
        (_node("test_build_d_overflow_lifecycles.py", "test_nonfatal_proof_overflow_emits_once_without_parent_stop"),),
        ("events.context_overflow_route_identity", "events.proof_context_overflow_nonfatal"),
    ),
    "real_proof_scope_matrix_isolated_under_temp_root": (
        (
            _node("test_build_d_proof_scope_filesystem.py", "test_autonomous_current_and_library_routes_do_not_read_manual_scope"),
            _node("test_build_d_proof_scope_filesystem.py", "test_manual_clear_archive_moves_active_proofs_to_history_only"),
            _node("test_build_d_proof_scope_filesystem.py", "test_leanoj_current_and_history_routes_are_disjoint_then_clear_removes_root"),
        ),
        ("proof_scope.manual_not_in_autonomous_current",),
    ),
    "real_manual_archive_and_leanoj_clear_preserve_scope_contract": (
        (
            _node("test_build_d_proof_scope_filesystem.py", "test_manual_clear_archive_moves_active_proofs_to_history_only"),
            _node("test_build_d_proof_scope_filesystem.py", "test_leanoj_current_and_history_routes_are_disjoint_then_clear_removes_root"),
        ),
        ("proof_scope.manual_clear_archives_active", "state.clear_removes_active_preserves_history"),
    ),
    "real_pruned_paper_preserved_but_removed_from_active_context": (
        (_node("test_build_d_proof_scope_filesystem.py", "test_pruned_paper_moves_to_temp_history_and_route_excludes_active_file"),),
        ("state.pruned_papers_excluded_from_context",),
    ),
    "real_direct_source_rag_exclusion_matrix": (
        tuple(
            _node("test_build_e_rag_prompt_context.py", name)
            for name in (
                "test_build_e_aggregator_mixed_direct_and_offloaded_sources_are_excluded",
                "test_build_e_aggregator_source_becomes_retrievable_when_offloaded",
                "test_build_e_compiler_construction_excludes_direct_sources_and_keeps_references",
                "test_build_e_compiler_rigor_excludes_direct_outline_and_paper",
            )
        ),
        ("prompt.direct_sources_excluded_from_rag",),
    ),
    "real_model_visible_prompts_strip_generated_proof_appendices": (
        (
            _node("test_build_e_prompt_boundaries.py", "test_real_proof_formalization_message_keeps_source_boundaries_and_verified_library"),
            _node("test_build_e_prompt_boundaries.py", "test_real_compiler_rigor_message_strips_theorem_appendix_and_keeps_proof_summary"),
        ),
        ("prompt.generated_appendices_stripped", "proof_scope.generated_appendices_stripped_from_prompts"),
    ),
    "real_rigor_mandatory_source_overflow_fails_before_model_or_rag": (
        (_node("test_build_e_prompt_boundaries.py", "test_real_compiler_rigor_mandatory_full_source_overflow_is_visible_and_pre_llm"),),
        ("prompt.mandatory_source_overflow_fails_visible",),
    ),
}

_PRODUCTION_COMPONENTS_BY_TEST_FILE = {
    "tests/workflow_real_adapters/test_proof_routes.py": (
        "backend/api/routes/compiler.py",
        "backend/autonomous/core/autonomous_coordinator.py",
        "backend/leanoj/core/leanoj_coordinator.py",
    ),
    "tests/workflow_real_adapters/test_route_conflicts.py": (
        "backend/shared/workflow_start_guard.py",
        "backend/api/routes/autonomous.py",
    ),
    "tests/workflow_real_adapters/test_high_value_scenarios.py": (
        "backend/autonomous/core/autonomous_coordinator.py",
        "backend/api/routes/proofs.py",
    ),
    "tests/workflow_real_adapters/test_build_bc_lifecycles.py": (
        "backend/api/routes/aggregator.py",
        "backend/api/routes/compiler.py",
        "backend/api/routes/leanoj.py",
    ),
    "tests/workflow_real_adapters/test_build_d_overflow_lifecycles.py": (
        "backend/aggregator/core/coordinator.py",
        "backend/autonomous/core/autonomous_coordinator.py",
    ),
    "tests/workflow_real_adapters/test_build_d_proof_scope_filesystem.py": (
        "backend/api/routes/proofs.py",
        "backend/autonomous/memory/proof_database.py",
    ),
    "tests/workflow_real_adapters/test_build_e_rag_prompt_context.py": (
        "backend/aggregator/core/context_allocator.py",
        "backend/compiler/core/compiler_rag_manager.py",
    ),
    "tests/workflow_real_adapters/test_build_e_prompt_boundaries.py": (
        "backend/autonomous/agents/proof_formalization_agent.py",
        "backend/compiler/agents/high_param_submitter.py",
    ),
}


def _link_passed_records() -> tuple[InteractionCoverage, ...]:
    linked: list[InteractionCoverage] = []
    for record in _UNLINKED_REAL_ADAPTER_COVERAGE:
        if record.result != "passed":
            linked.append(
                replace(
                    record,
                    blocked_dependencies=record.blocked_dependencies
                    or ("unavailable bounded production transition seam",),
                )
            )
            continue
        if record.test_selectors:
            selectors = record.test_selectors
            asserted_invariants = record.asserted_invariants
            real_components = record.real_components
            faked_dependencies = record.faked_dependencies
            observed_transition = record.observed_transition
        else:
            try:
                selectors, asserted_invariants = _EXPLICIT_EVIDENCE[record.scenario_id]
                real_components = _PRODUCTION_COMPONENTS_BY_TEST_FILE[record.test_file]
            except KeyError as exc:
                raise AssertionError(
                    f"Passed coverage record {record.scenario_id} lacks an explicit evidence declaration"
                ) from exc
            faked_dependencies = ("external providers and proof toolchains",)
            observed_transition = "; ".join(record.evidence)
        linked.append(
            replace(
                record,
                runner=record.runner or "pytest",
                test_selectors=selectors,
                asserted_invariants=asserted_invariants,
                real_components=real_components,
                faked_dependencies=faked_dependencies,
                observed_transition=observed_transition,
            )
        )
    return tuple(linked)


REAL_ADAPTER_COVERAGE = _link_passed_records()
