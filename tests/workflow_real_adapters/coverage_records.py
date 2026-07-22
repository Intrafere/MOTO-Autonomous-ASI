from __future__ import annotations

from dataclasses import replace

from tests.workflow_harness.coverage_metadata import InteractionCoverage


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
    + HIGH_VALUE_GAP_COVERAGE
)


def _ranked_selectors_from_test_file(record: InteractionCoverage) -> tuple[str, ...]:
    """Resolve exact pytest nodes from the uniquely matching scenario words."""
    import ast
    from pathlib import Path

    test_path = Path(record.test_file)
    source = test_path.read_text(encoding="utf-8")
    functions = [
        node.name
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    ]
    scenario_words = set(record.scenario_id.removeprefix("real_").split("_"))
    ranked = sorted(
        functions,
        key=lambda name: len(scenario_words.intersection(name.removeprefix("test_").split("_"))),
        reverse=True,
    )
    if not ranked:
        raise AssertionError(f"No pytest selector found for {record.scenario_id}")
    return tuple(f"{record.test_file}::{name}" for name in ranked)


def _link_passed_records() -> tuple[InteractionCoverage, ...]:
    used_selectors: set[str] = set()
    linked: list[InteractionCoverage] = []
    for record in _UNLINKED_REAL_ADAPTER_COVERAGE:
        if record.result != "passed":
            linked.append(record)
            continue
        selector = next(
            (
                candidate
                for candidate in _ranked_selectors_from_test_file(record)
                if candidate not in used_selectors
            ),
            None,
        )
        if selector is None:
            raise AssertionError(f"No unique pytest selector found for {record.scenario_id}")
        used_selectors.add(selector)
        linked.append(
            replace(
                record,
                runner="pytest",
                test_selectors=(selector,),
                asserted_invariants=record.invariants,
            )
        )
    return tuple(linked)


REAL_ADAPTER_COVERAGE = _link_passed_records()
