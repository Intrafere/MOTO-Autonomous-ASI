"""Test-only Build 07 mappings from product laws to observable source seams."""
from __future__ import annotations

from dataclasses import dataclass

from .coverage_metadata import AdapterType, CoverageRunner, InteractionCoverage, ResultStatus
from .invariant_catalog import INVARIANTS_BY_ID


@dataclass(frozen=True)
class WorkflowSourceMapping:
    invariant_id: str
    scenario_id: str
    adapter: AdapterType
    result: ResultStatus
    production_sources: tuple[str, ...]
    test_selector: str | None
    runner: CoverageRunner = "pytest"
    evidence: tuple[str, ...] = ()
    faked_dependencies: tuple[str, ...] = ()
    observed_transition: str | None = None
    blocked_dependencies: tuple[str, ...] = ()
    blocked_reason: str | None = None

    def coverage_record(self) -> InteractionCoverage:
        invariant = INVARIANTS_BY_ID[self.invariant_id]
        test_file = (
            self.test_selector.split("::", 1)[0]
            if self.test_selector
            else "tests/workflow_real_adapters/test_build_07_source_mappings.py"
        )
        return InteractionCoverage(
            scenario_id=self.scenario_id,
            fields=(invariant.field, *invariant.crossed_fields),
            invariants=(self.invariant_id,),
            adapter=self.adapter,
            result=self.result,
            test_file=test_file,
            diagnostics={"reason": self.blocked_reason} if self.blocked_reason else None,
            evidence=self.evidence,
            runner=self.runner if self.test_selector else None,
            test_selectors=(self.test_selector,) if self.test_selector else (),
            asserted_invariants=(self.invariant_id,) if self.test_selector else (),
            real_components=self.production_sources,
            faked_dependencies=self.faked_dependencies,
            blocked_dependencies=self.blocked_dependencies,
            observed_transition=self.observed_transition,
        )


BUILD_07_FIELDS = {
    "proof_runtime.candidate_list_validator_gate": (
        "proof_runtime_gating",
        "provider_pause_resume",
        "workflow_filesystem_state",
        "websocket_api_contracts",
    ),
    "proof_runtime.candidate_list_autonomous_restart_owned": (
        "proof_runtime_gating",
        "provider_pause_resume",
        "workflow_filesystem_state",
    ),
    "proof_runtime.candidate_list_manual_process_local": (
        "proof_runtime_gating",
        "runtime_exclusivity",
        "workflow_filesystem_state",
    ),
    "proof_pruning.artifacts_and_future_memory_preserved": (
        "proof_pruning",
        "workflow_filesystem_state",
        "assistant_memory",
    ),
    "proof_pruning.owning_run_context_excludes_pruned": (
        "proof_pruning",
        "prompt_context",
        "assistant_memory",
    ),
    "proof_pruning.validator_gates_automatic_mutation": (
        "proof_pruning",
        "provider_pause_resume",
        "workflow_filesystem_state",
    ),
    "proof_pruning.semantic_distinct_review_preserves_unique_routes": (
        "proof_pruning",
        "prompt_context",
        "proof_scope_isolation",
        "workflow_filesystem_state",
    ),
    "proof_pruning.no_prune_is_valid": ("proof_pruning", "provider_pause_resume"),
    "proof_pruning.review_non_blocking": (
        "proof_pruning",
        "proof_runtime_gating",
        "provider_pause_resume",
    ),
    "proof_pruning.commit_lifecycle_fenced": (
        "proof_pruning",
        "runtime_exclusivity",
        "workflow_filesystem_state",
    ),
    "proof_pruning.context_overflow_truthful": (
        "proof_pruning",
        "websocket_api_contracts",
        "proof_runtime_gating",
    ),
    "proof_loop.continuous_explicit_ownership": (
        "continuous_proof_loop",
        "runtime_exclusivity",
        "workflow_filesystem_state",
    ),
    "proof_loop.automatic_round_policy_preserved": (
        "continuous_proof_loop",
        "proof_runtime_gating",
        "workflow_filesystem_state",
    ),
    "proof_pruning.occurrence_scope_isolated": (
        "proof_pruning",
        "proof_scope_isolation",
        "assistant_memory",
    ),
}


BUILD_07_SOURCE_MAPPINGS: tuple[WorkflowSourceMapping, ...] = (
    WorkflowSourceMapping(
        invariant_id="proof_runtime.candidate_list_validator_gate",
        scenario_id="real_candidate_list_validator_gate_and_scope",
        adapter="focused_production",
        result="passed",
        production_sources=(
            "backend/autonomous/agents/proof_candidate_list_validator.py",
            "backend/autonomous/core/proof_verification_stage.py",
        ),
        test_selector=(
            "tests/workflow_real_adapters/test_build_07_candidate_list_ownership.py::"
            "test_real_candidate_list_gate_precedes_phase_a_and_forwards_approved_subset"
        ),
        evidence=("whole_list_review", "any_novel_accepts", "approved_only"),
        faked_dependencies=("candidate Validator response", "Lean proof-stage boundary"),
        observed_transition="whole-list review accepts a list with one of four novel candidates and forwards only that candidate before any Phase A proof cost",
    ),
    WorkflowSourceMapping(
        invariant_id="proof_runtime.candidate_list_autonomous_restart_owned",
        scenario_id="real_candidate_list_autonomous_checkpoint_restarts_durably",
        adapter="focused_production",
        result="passed",
        production_sources=(
            "backend/autonomous/core/proof_verification_stage.py",
            "backend/autonomous/memory/research_metadata.py",
        ),
        test_selector=(
            "tests/workflow_real_adapters/test_build_07_candidate_list_ownership.py::"
            "test_real_autonomous_candidate_list_checkpoint_survives_lifecycle_restart"
        ),
        evidence=("durable_checkpoint", "same_scope_reuse", "round_scope_fence"),
        faked_dependencies=("candidate Validator response", "Lean proof-stage boundary"),
        observed_transition="approved list persists through a fresh ResearchMetadata lifecycle and is reused only under its original review scope",
    ),
    WorkflowSourceMapping(
        invariant_id="proof_runtime.candidate_list_manual_process_local",
        scenario_id="real_candidate_list_manual_checkpoint_is_process_local",
        adapter="focused_production",
        result="passed",
        production_sources=(
            "backend/autonomous/core/proof_verification_stage.py",
            "backend/autonomous/core/proof_run_manager.py",
        ),
        test_selector=(
            "tests/workflow_real_adapters/test_build_07_candidate_list_ownership.py::"
            "test_real_manual_candidate_list_checkpoint_is_same_process_only"
        ),
        evidence=("same_process_reuse", "round_cleanup", "fresh_manager_empty"),
        faked_dependencies=("candidate Validator response", "Lean proof-stage boundary"),
        observed_transition="manual approval state is reused within one run control, cleared at round completion, and absent from a fresh manager",
    ),
    WorkflowSourceMapping(
        invariant_id="proof_pruning.artifacts_and_future_memory_preserved",
        scenario_id="real_pruned_proof_artifacts_and_future_memory_preserved",
        adapter="focused_production",
        result="passed",
        production_sources=(
            "backend/autonomous/memory/proof_database.py",
            "backend/api/routes/proofs.py",
            "backend/shared/proof_search/indexer.py",
        ),
        test_selector=(
            "tests/workflow_real_adapters/test_build_07_proof_pruning_preservation.py::"
            "test_pruned_occurrence_remains_exportable_graph_visible_searchable_and_eligible"
        ),
        evidence=("certificate_exports", "dependency_graph", "future_run_search", "syntheticlib_novel_unchanged"),
        faked_dependencies=("external SyntheticLib donation service",),
        observed_transition="owner-run pruning preserves canonical exports and graph identity while excluding only owner-run model search and retaining future-run eligibility",
    ),
    WorkflowSourceMapping(
        invariant_id="proof_pruning.owning_run_context_excludes_pruned",
        scenario_id="real_owning_run_context_and_cached_assistant_exclude_pruned",
        adapter="real_coordinator",
        result="blocked",
        production_sources=(
            "backend/autonomous/memory/proof_database.py",
            "backend/shared/proof_search/assistant_cache.py",
        ),
        test_selector=None,
        blocked_dependencies=("direct proof injection", "all Assistant cache reuse paths"),
        blocked_reason=(
            "The linked focused test proves one cached-pack filter, not every owning-run "
            "direct-context and Assistant reuse surface claimed by the invariant."
        ),
    ),
    WorkflowSourceMapping(
        invariant_id="proof_pruning.validator_gates_automatic_mutation",
        scenario_id="real_automatic_prune_requires_validator_acceptance",
        adapter="real_coordinator",
        result="blocked",
        production_sources=(
            "backend/autonomous/agents/proof_pruning_agent.py",
            "backend/autonomous/core/proof_pruning_coordinator.py",
        ),
        test_selector=None,
        blocked_dependencies=("automatic mutation commit after independent validation",),
        blocked_reason=(
            "The focused contract test rejects Validator retargeting, but does not observe "
            "the complete proposal-to-validation-to-mutation gate."
        ),
    ),
    WorkflowSourceMapping(
        invariant_id="proof_pruning.no_prune_is_valid",
        scenario_id="real_no_prune_is_valid_non_mutating_result",
        adapter="focused_production",
        result="passed",
        production_sources=("backend/autonomous/core/proof_pruning_coordinator.py",),
        test_selector=(
            "tests/unit/test_proof_pruning_coordinator.py::"
            "ProofPruningCoordinatorTests::test_third_eligible_registration_schedules_without_blocking"
        ),
        evidence=("no_prune_outcome", "no_mutation"),
        faked_dependencies=("proof pruning review service",),
        observed_transition="scheduled review returns no_prune without changing proof revision or records",
    ),
    WorkflowSourceMapping(
        invariant_id="proof_pruning.review_non_blocking",
        scenario_id="real_pruning_review_pending_does_not_block_registration",
        adapter="focused_production",
        result="passed",
        production_sources=("backend/autonomous/core/proof_pruning_coordinator.py",),
        test_selector=(
            "tests/unit/test_proof_pruning_coordinator.py::"
            "ProofPruningCoordinatorTests::test_notify_registration_is_immediate_and_owned"
        ),
        evidence=("immediate_notification", "held_review"),
        faked_dependencies=("held proof pruning review service",),
        observed_transition="proof registration notification returns while owned review remains pending",
    ),
    WorkflowSourceMapping(
        invariant_id="proof_pruning.commit_lifecycle_fenced",
        scenario_id="real_prune_commit_rechecks_lifecycle_and_revision",
        adapter="real_coordinator",
        result="blocked",
        production_sources=(
            "backend/autonomous/core/proof_pruning_coordinator.py",
            "backend/autonomous/memory/proof_database.py",
        ),
        test_selector=None,
        blocked_dependencies=("lifecycle generation interleaving",),
        blocked_reason=(
            "The focused snapshot test proves target/revision rechecks, but not the "
            "lifecycle-generation fence included in the invariant."
        ),
    ),
    WorkflowSourceMapping(
        invariant_id="proof_pruning.context_overflow_truthful",
        scenario_id="real_pruning_overflow_commit_interleaving_unobservable",
        adapter="real_coordinator",
        result="blocked",
        production_sources=(
            "backend/autonomous/core/proof_verification_stage.py",
            "backend/autonomous/core/proof_pruning_coordinator.py",
        ),
        test_selector=None,
        blocked_dependencies=("overflow during prune commit transition owner",),
        blocked_reason=(
            "Existing isolated tests observe nonfatal proof overflow and pruning triggers separately; "
            "no bounded seam observes an overflow arriving during a prune commit without synthesizing "
            "the transition owner."
        ),
    ),
    WorkflowSourceMapping(
        invariant_id="proof_loop.continuous_explicit_ownership",
        scenario_id="real_continuous_loop_stop_terminal_zero_policy",
        adapter="real_route",
        result="passed",
        production_sources=(
            "backend/autonomous/core/proof_round_driver.py",
            "backend/autonomous/core/proof_run_manager.py",
        ),
        test_selector=(
            "tests/workflow_real_adapters/test_build_07_proof_loop_lifecycle.py::"
            "test_real_continuous_route_completes_after_ten_empty_rounds_and_cleans_up"
        ),
        evidence=("ten_round_terminal", "single_sleep_owner", "exact_once_terminal", "cleanup"),
        faked_dependencies=("manual source resolution", "provider and Lean proof-stage boundary"),
        observed_transition="real check route queues one continuous run that publishes ten rounds, terminates normally once, and releases ownership",
    ),
    WorkflowSourceMapping(
        invariant_id="proof_loop.automatic_round_policy_preserved",
        scenario_id="real_automatic_round_caller_count_policy_preserved",
        adapter="real_coordinator",
        result="passed",
        production_sources=(
            "backend/autonomous/core/autonomous_coordinator.py",
            "backend/autonomous/core/proof_round_driver.py",
        ),
        test_selector=(
            "tests/workflow_real_adapters/test_build_07_proof_loop_lifecycle.py::"
            "test_real_autonomous_automatic_round_policy_matrix"
        ),
        evidence=("proofs_only_max_4", "combined_brainstorm_max_2", "completed_paper_max_1", "first_zero_exit"),
        faked_dependencies=("provider and Lean proof-stage boundary",),
        observed_transition="real Autonomous caller selects 4/2/1 round maxima by output/source policy and exits on the first valid zero-candidate result",
    ),
    WorkflowSourceMapping(
        invariant_id="proof_pruning.occurrence_scope_isolated",
        scenario_id="real_pruning_is_occurrence_and_owning_run_scoped",
        adapter="real_coordinator",
        result="blocked",
        production_sources=("backend/autonomous/memory/proof_database.py",),
        test_selector=None,
        blocked_dependencies=("cross-run occurrence retrieval", "Assistant occurrence filtering"),
        blocked_reason=(
            "The focused snapshot test proves same-run review exclusion only, not the "
            "full occurrence and owning-run isolation contract."
        ),
    ),
    WorkflowSourceMapping(
        invariant_id="proof_pruning.semantic_distinct_review_preserves_unique_routes",
        scenario_id="real_semantic_prune_persists_and_protects_retained_support",
        adapter="real_coordinator",
        result="blocked",
        production_sources=(
            "backend/autonomous/agents/proof_pruning_agent.py",
            "backend/autonomous/memory/proof_database.py",
        ),
        test_selector=None,
        blocked_dependencies=("semantic proposer-validator-commit lifecycle", "later support-prune attempt"),
        blocked_reason=(
            "Focused unit tests cover durable retained-support protection and "
            "section arbitration, but no bounded real-coordinator adapter yet "
            "observes canonically distinct semantic routes through proposal, "
            "validation, commit, and a later support-prune attempt."
        ),
    ),
    WorkflowSourceMapping(
        invariant_id="live_activity.provider_retry_each_wait_visible",
        scenario_id="real_provider_retry_wait_broadcasts_additive_activity",
        adapter="focused_production",
        result="passed",
        production_sources=("backend/shared/api_client_manager.py",),
        test_selector=(
            "tests/integration/test_oauth_provider_cooldown.py::"
            "APIClientManagerOAuthCooldownTests::"
            "test_retryable_provider_error_reports_wait_and_resume_activity"
        ),
        evidence=("wait_callback", "provider_retry_cooldown", "attempt_and_duration"),
        faked_dependencies=("provider transport", "stop-aware sleep", "WebSocket broadcaster"),
        observed_transition="a retryable provider failure schedules one wait and emits additive scoped retry activity with attempt and duration",
    ),
    WorkflowSourceMapping(
        invariant_id="live_activity.provider_retry_identity_stable",
        scenario_id="real_provider_retry_attempt_identity_and_correlation",
        adapter="focused_production",
        result="passed",
        production_sources=("backend/shared/api_client_manager.py",),
        test_selector=(
            "tests/integration/test_oauth_provider_cooldown.py::"
            "APIClientManagerOAuthCooldownTests::"
            "test_retry_attempts_share_correlation_but_have_unique_event_ids"
        ),
        evidence=("unique_event_id", "stable_correlation_id", "ordered_retry_attempt"),
        faked_dependencies=("WebSocket broadcaster",),
        observed_transition="two retry attempts retain one correlation id while publishing distinct event ids and ordered attempt numbers",
    ),
    WorkflowSourceMapping(
        invariant_id="live_activity.provider_retry_secret_safe",
        scenario_id="real_provider_retry_activity_redacts_error_detail",
        adapter="focused_production",
        result="passed",
        production_sources=("backend/shared/api_client_manager.py",),
        test_selector=(
            "tests/integration/test_oauth_provider_cooldown.py::"
            "APIClientManagerOAuthCooldownTests::"
            "test_retry_activity_redacts_error_summary_at_broadcast_boundary"
        ),
        evidence=("bounded_error_summary", "bearer_secret_redacted", "user_visible_error_class"),
        faked_dependencies=("WebSocket broadcaster",),
        observed_transition="secret-bearing provider detail is redacted before retry activity reaches the broadcast boundary",
    ),
    WorkflowSourceMapping(
        invariant_id="live_activity.provider_retry_persists_reload",
        scenario_id="real_provider_retry_activity_persists_reload",
        adapter="focused_production",
        result="passed",
        production_sources=(
            "frontend/src/utils/liveActivityPersistence.js",
            "frontend/src/App.jsx",
        ),
        test_selector=(
            "frontend/src/utils/activityPersistence.test.js::"
            "activity persistence sanitization::"
            "persists two correlated provider retry attempts across reload distinctly and secret-safe"
        ),
        runner="vitest",
        evidence=(
            "two_distinct_attempt_identities",
            "shared_correlation_id",
            "reload_restoration",
            "persisted_secret_redaction",
        ),
        faked_dependencies=("browser localStorage supplied by Vitest jsdom",),
        observed_transition=(
            "two correlated provider retry attempts are compacted into secret-safe storage "
            "and reload as two distinct ordered activity records"
        ),
    ),
    WorkflowSourceMapping(
        invariant_id="live_activity.all_retry_wait_sources_accounted",
        scenario_id="real_all_provider_retry_wait_sources_accounted",
        adapter="real_coordinator",
        result="passed",
        production_sources=(
            "backend/shared/api_client_manager.py",
            "backend/shared/openrouter_client.py",
            "backend/shared/lm_studio_client.py",
            "backend/shared/openai_codex_client.py",
            "backend/shared/xai_grok_client.py",
            "backend/shared/subscription_providers/api_key_client.py",
            "backend/aggregator/core/coordinator.py",
            "backend/compiler/core/compiler_coordinator.py",
            "backend/autonomous/core/autonomous_coordinator.py",
            "backend/leanoj/core/leanoj_coordinator.py",
            "backend/api/routes/proofs.py",
        ),
        test_selector=(
            "tests/workflow_real_adapters/test_live_activity_retry_inventory.py::"
            "test_all_retry_wait_sources_accounted_are_assured"
        ),
        evidence=(
            "production_wide_ast_wait_inventory",
            "all_retry_sites_assured",
            "non_retry_waits_explicitly_excluded",
        ),
        faked_dependencies=("provider transports", "wall clock", "external tools"),
        observed_transition=(
            "the production-wide AST inventory classifies every production sleep and every actual "
            "retry source routes through the production assured emit-and-wait seam"
        ),
    ),
    WorkflowSourceMapping(
        invariant_id="live_activity.provider_error_route_identity",
        scenario_id="real_openrouter_no_choices_retry_names_full_route",
        adapter="focused_production",
        result="passed",
        production_sources=(
            "backend/shared/api_client_manager.py",
            "backend/shared/openrouter_client.py",
            "backend/shared/provider_route_identity.py",
            "backend/shared/provider_notification_store.py",
        ),
        test_selector=(
            "tests/integration/test_provider_error_route_identity.py::"
            "ProviderErrorRouteIdentityTests::"
            "test_openrouter_no_choices_retries_carry_mode_role_model_and_upstream_host"
        ),
        evidence=(
            "workflow_mode_and_role_label",
            "model_and_provider",
            "auto_host_routing_with_upstream_host",
            "provider_request_id",
            "route_identity_message_prefix",
            "identity_absent_from_exception_and_backoff_key",
        ),
        faked_dependencies=(
            "synthetic OpenRouter HTTP 200 error-body response shapes",
            "stop-aware sleep",
            "WebSocket broadcaster",
            "temporary data root",
        ),
        observed_transition=(
            "two adapter retries and the 60-second workflow backoff for an OpenRouter no-choices "
            "body each name the Autonomous mode, Submitter 2, its model, OpenRouter, and the "
            "auto-routed upstream host in fields and the rendered message"
        ),
    ),
    WorkflowSourceMapping(
        invariant_id="live_activity.provider_error_route_identity",
        scenario_id="real_provider_error_line_renders_route_identity_once",
        adapter="focused_production",
        result="passed",
        production_sources=("frontend/src/utils/activityStyles.js",),
        test_selector=(
            "frontend/src/utils/activityStyles.test.js::"
            "provider route identity::"
            "prefixes provider error lines with backend route identity once"
        ),
        runner="vitest",
        evidence=("backend_route_identity_prefix", "idempotent_prefix"),
        faked_dependencies=("provider-error activity payloads supplied by Vitest",),
        observed_transition=(
            "a provider-error activity line is prefixed with the backend-composed route identity "
            "exactly once"
        ),
    ),
    WorkflowSourceMapping(
        invariant_id="live_activity.provider_error_route_identity",
        scenario_id="real_live_openrouter_upstream_host_reported",
        adapter="focused_production",
        result="blocked",
        production_sources=("backend/shared/openrouter_client.py",),
        test_selector=None,
        blocked_dependencies=("live OpenRouter upstream host reporting",),
        blocked_reason=(
            "Upstream host parsing is proven only against synthetic OpenRouter response "
            "shapes; whether live OpenRouter error bodies keep reporting the upstream "
            "host cannot be observed without a credentialed network call."
        ),
    ),
    WorkflowSourceMapping(
        invariant_id="live_activity.provider_error_sources_route_attributed",
        scenario_id="real_provider_error_activity_sources_route_attributed",
        adapter="real_coordinator",
        result="passed",
        production_sources=(
            "backend/shared/api_client_manager.py",
            "backend/shared/provider_route_identity.py",
            "backend/aggregator/core/coordinator.py",
            "backend/compiler/core/compiler_coordinator.py",
            "backend/autonomous/core/autonomous_coordinator.py",
            "backend/autonomous/core/proof_run_manager.py",
            "backend/leanoj/core/leanoj_coordinator.py",
            "backend/api/routes/proofs.py",
        ),
        test_selector=(
            "tests/workflow_real_adapters/test_provider_error_activity_inventory.py::"
            "test_every_provider_error_activity_source_is_route_attributed_or_exempt"
        ),
        evidence=(
            "production_wide_ast_activity_inventory",
            "attributed_sites_syntactically_verified",
            "exempt_sites_explicitly_reasoned",
        ),
        faked_dependencies=("provider transports", "WebSocket broadcaster"),
        observed_transition=(
            "the production-wide AST inventory classifies every provider-error activity source, "
            "retry-seam call, and known dynamic forwarder as syntactically route-attributed or "
            "explicitly exempt"
        ),
    ),
    WorkflowSourceMapping(
        invariant_id="model_output.contract_failure_not_semantic",
        scenario_id="focused_proof_identification_contract_error_names_item_and_field",
        adapter="focused_production",
        result="passed",
        production_sources=("backend/autonomous/agents/proof_identification_agent.py", "backend/shared/model_output_contract.py"),
        test_selector="tests/regressions/test_proof_identification_contract_errors.py::test_one_bad_entry_names_item_and_field_without_claiming_none_were_valid",
        runner="pytest",
        evidence=("typed_contract_error", "bad_item_and_field_named", "no_false_none_valid_claim"),
        faked_dependencies=("model replies supplied as fixed strings",),
        observed_transition="one invalid theorem entry among valid ones raises a typed contract error naming that entry and field",
    ),
    WorkflowSourceMapping(
        invariant_id="model_output.contract_failure_not_semantic",
        scenario_id="focused_proof_check_complete_reports_contract_stage",
        adapter="focused_production",
        result="passed",
        production_sources=("backend/autonomous/core/proof_verification_stage.py", "backend/shared/model_output_contract.py"),
        test_selector="tests/regressions/test_proof_identification_contract_errors.py::test_stage_reports_identification_contract_failure_accurately",
        runner="pytest",
        evidence=("error_kind_model_output_contract", "error_stage_identification", "no_lean_work_note"),
        faked_dependencies=("external providers and proof toolchains", "WebSocket broadcaster"),
        observed_transition="an identification contract failure ends the proof check with a typed, route-named proof_check_complete error",
    ),
    WorkflowSourceMapping(
        invariant_id="model_output.contract_failure_not_semantic",
        scenario_id="focused_contract_errors_never_provider_classified",
        adapter="focused_production",
        result="passed",
        production_sources=("backend/shared/model_error_utils.py", "backend/shared/model_output_contract.py"),
        test_selector="tests/unit/test_model_output_contract.py::test_error_helpers_check_the_contract_type_first",
        runner="pytest",
        evidence=("not_transient", "not_context_length", "not_non_retryable"),
        faked_dependencies=("marker-laden model tokens supplied by the test",),
        observed_transition="a contract error carrying provider-like words is never classified as a provider, transient, or context failure",
    ),
    WorkflowSourceMapping(
        invariant_id="model_output.contract_failure_not_semantic",
        scenario_id="focused_aggregator_validator_contract_failure_requeued",
        adapter="focused_production",
        result="passed",
        production_sources=("backend/aggregator/agents/validator.py", "backend/aggregator/core/coordinator.py"),
        test_selector="tests/regressions/test_batch_validator_contract.py::test_coordinator_requeues_single_contract_failure_without_submitter_feedback",
        runner="pytest",
        evidence=("requeued_one_at_a_time", "no_submitter_feedback", "operational_activity"),
        faked_dependencies=("validator model replies", "WebSocket broadcaster"),
        observed_transition="an unusable validator reply requeues the submission with operational activity and no submitter rejection",
    ),
    WorkflowSourceMapping(
        invariant_id="model_output.contract_failure_not_semantic",
        scenario_id="focused_aggregator_batch_contract_failure_restored",
        adapter="focused_production",
        result="passed",
        production_sources=("backend/aggregator/agents/validator.py", "backend/aggregator/core/coordinator.py"),
        test_selector="tests/regressions/test_batch_validator_contract.py::test_coordinator_restores_unusable_batch_and_validates_it_one_at_a_time",
        runner="pytest",
        evidence=("batch_restored", "validated_one_at_a_time", "no_retry_wait"),
        faked_dependencies=("validator model replies", "WebSocket broadcaster"),
        observed_transition="an unusable batch reply restores every dequeued submission and validates them one at a time",
    ),
    WorkflowSourceMapping(
        invariant_id="model_output.contract_failure_not_semantic",
        scenario_id="focused_compiler_validator_contract_failure_revalidates",
        adapter="focused_production",
        result="passed",
        production_sources=("backend/compiler/core/compiler_coordinator.py", "backend/compiler/validation/compiler_validator.py"),
        test_selector="tests/unit/test_compiler_model_output_contract.py::test_validator_contract_failure_revalidates_same_submission",
        runner="pytest",
        evidence=("same_submission_revalidated", "no_rejection_log", "operational_compiler_warning"),
        faked_dependencies=("validator model replies", "WebSocket broadcaster"),
        observed_transition="a compiler validator contract failure re-validates the same submission without recording a rejection",
    ),
    WorkflowSourceMapping(
        invariant_id="model_output.contract_failure_not_semantic",
        scenario_id="focused_compiler_critique_contract_failure_not_provider_wait",
        adapter="focused_production",
        result="passed",
        production_sources=("backend/compiler/core/compiler_coordinator.py", "backend/compiler/agents/high_param_submitter.py"),
        test_selector="tests/unit/test_compiler_model_output_contract.py::test_critique_loop_contract_failures_are_not_provider_waits",
        runner="pytest",
        evidence=("no_emit_and_wait", "not_counted_as_attempt", "no_rejection_feedback"),
        faked_dependencies=("critique model replies", "WebSocket broadcaster"),
        observed_transition="critique and critique-validator contract failures surface as operational warnings, not provider waits",
    ),
    WorkflowSourceMapping(
        invariant_id="model_output.contract_failure_not_semantic",
        scenario_id="focused_proof_framing_gate_unresolved_defaults_on",
        adapter="focused_production",
        result="passed",
        production_sources=("backend/autonomous/core/autonomous_coordinator.py", "backend/autonomous/agents/output_contract.py"),
        test_selector="tests/unit/test_autonomous_output_contracts.py::ProofFramingGateTests::test_unresolved_gate_defaults_framing_on_and_is_visible",
        runner="pytest",
        evidence=("framing_on", "unresolved_visible", "unresolved_marker"),
        faked_dependencies=("gate model replies", "WebSocket broadcaster"),
        observed_transition="an unresolved proof framing gate turns framing on and publishes an unresolved decision",
    ),
    WorkflowSourceMapping(
        invariant_id="model_output.contract_failure_not_semantic",
        scenario_id="focused_topic_validator_contract_failure_not_rejection",
        adapter="focused_production",
        result="passed",
        production_sources=("backend/autonomous/agents/topic_validator.py", "backend/autonomous/agents/output_contract.py"),
        test_selector="tests/unit/test_autonomous_output_contracts.py::ValidatorContractTests::test_topic_validator_contract_failure_is_not_a_rejection",
        runner="pytest",
        evidence=("typed_contract_error", "no_reject_result"),
        faked_dependencies=("validator model replies",),
        observed_transition="a topic validator reply missing its decision raises a contract error instead of a reject",
    ),
    WorkflowSourceMapping(
        invariant_id="model_output.contract_failure_not_semantic",
        scenario_id="focused_completion_self_validation_fallback_labeled",
        adapter="focused_production",
        result="passed",
        production_sources=("backend/autonomous/agents/completion_reviewer.py", "backend/autonomous/agents/output_contract.py"),
        test_selector="tests/unit/test_autonomous_output_contracts.py::CompletionReviewerTests::test_unusable_self_validation_continues_without_feedback",
        runner="pytest",
        evidence=("continue_brainstorm_fallback", "no_durable_feedback", "operational_activity"),
        faked_dependencies=("self-validation model replies", "WebSocket broadcaster"),
        observed_transition="an unusable self-validation keeps the continue fallback without writing completion feedback",
    ),
    WorkflowSourceMapping(
        invariant_id="model_output.normalizations_visible_and_guarded",
        scenario_id="focused_parser_drops_stray_key_characters_and_reports",
        adapter="focused_production",
        result="passed",
        production_sources=("backend/shared/json_parser.py",),
        test_selector="tests/unit/test_json_parser_repairs.py::test_exact_gemma_stray_line_is_dropped_and_reported",
        runner="pytest",
        evidence=("stray_run_removed", "repair_recorded"),
        faked_dependencies=("model reply supplied as a fixed string",),
        observed_transition="the observed stray character before a key is removed and recorded as a repair",
    ),
    WorkflowSourceMapping(
        invariant_id="model_output.normalizations_visible_and_guarded",
        scenario_id="focused_parser_never_repairs_truncated_output",
        adapter="focused_production",
        result="passed",
        production_sources=("backend/shared/json_parser.py",),
        test_selector="tests/unit/test_json_parser_repairs.py::test_truncated_output_still_raises_and_is_never_repaired",
        runner="pytest",
        evidence=("truncation_raises", "no_repairs"),
        faked_dependencies=("model reply supplied as a fixed string",),
        observed_transition="truncated output still raises and records no repair",
    ),
    WorkflowSourceMapping(
        invariant_id="model_output.normalizations_visible_and_guarded",
        scenario_id="focused_parser_never_repairs_value_positions",
        adapter="focused_production",
        result="passed",
        production_sources=("backend/shared/json_parser.py",),
        test_selector="tests/unit/test_json_parser_repairs.py::test_value_and_array_positions_are_never_repaired",
        runner="pytest",
        evidence=("value_position_refused", "array_position_refused"),
        faked_dependencies=("model reply supplied as a fixed string",),
        observed_transition="stray characters in value or array-element position are never removed",
    ),
    WorkflowSourceMapping(
        invariant_id="model_output.normalizations_visible_and_guarded",
        scenario_id="focused_key_recovery_relabels_observed_misspellings",
        adapter="focused_production",
        result="passed",
        production_sources=("backend/shared/json_key_recovery.py",),
        test_selector="tests/unit/test_json_key_recovery.py::test_observed_gemma_misspellings_recover",
        runner="pytest",
        evidence=("misspelled_free_text_key_relabeled", "order_preserved"),
        faked_dependencies=("parsed objects supplied by the test",),
        observed_transition="each observed misspelled free-text key is relabeled to its schema key in place",
    ),
    WorkflowSourceMapping(
        invariant_id="model_output.normalizations_visible_and_guarded",
        scenario_id="focused_key_recovery_protected_keys_never_repairable",
        adapter="focused_production",
        result="passed",
        production_sources=("backend/shared/json_key_recovery.py",),
        test_selector="tests/unit/test_json_key_recovery.py::test_protected_keys_can_never_be_declared_repairable",
        runner="pytest",
        evidence=("protected_key_declaration_rejected",),
        faked_dependencies=("schemas supplied by the test",),
        observed_transition="a schema cannot declare an id, discriminator, decision, or anchor key repairable",
    ),
    WorkflowSourceMapping(
        invariant_id="model_output.normalizations_visible_and_guarded",
        scenario_id="focused_key_recovery_never_swaps_anchor_values",
        adapter="focused_production",
        result="passed",
        production_sources=("backend/shared/json_key_recovery.py",),
        test_selector="tests/unit/test_json_key_recovery.py::test_swapped_anchor_values_are_never_relabeled",
        runner="pytest",
        evidence=("anchor_values_untouched",),
        faked_dependencies=("parsed objects supplied by the test",),
        observed_transition="exact-edit anchor fields are never relabeled",
    ),
    WorkflowSourceMapping(
        invariant_id="model_output.normalizations_visible_and_guarded",
        scenario_id="focused_normalization_activity_route_identified",
        adapter="focused_production",
        result="passed",
        production_sources=("backend/shared/model_output_contract.py", "backend/shared/api_client_manager.py"),
        test_selector="tests/unit/test_model_output_contract.py::test_report_helper_broadcasts_route_identified_activity",
        runner="pytest",
        evidence=("model_output_normalized_broadcast", "route_identity_prefix", "unique_event_id"),
        faked_dependencies=("WebSocket broadcaster",),
        observed_transition="an applied repair is broadcast as route-identified model_output_normalized activity",
    ),
    WorkflowSourceMapping(
        invariant_id="model_output.normalizations_visible_and_guarded",
        scenario_id="focused_identification_normalization_published_without_retry",
        adapter="focused_production",
        result="passed",
        production_sources=("backend/autonomous/agents/proof_identification_agent.py", "backend/shared/model_output_contract.py"),
        test_selector="tests/regressions/test_proof_identification_contract_errors.py::test_gemma_payload_needs_no_retry_and_publishes_one_normalization",
        runner="pytest",
        evidence=("one_model_call", "one_normalization_event"),
        faked_dependencies=("identification model reply", "WebSocket broadcaster"),
        observed_transition="the observed misspelled rationale is accepted without retry and published once",
    ),
    WorkflowSourceMapping(
        invariant_id="model_output.normalizations_visible_and_guarded",
        scenario_id="focused_aggregator_batch_reasoning_relabel_reported",
        adapter="focused_production",
        result="passed",
        production_sources=("backend/aggregator/agents/validator.py", "backend/shared/json_key_recovery.py"),
        test_selector="tests/regressions/test_batch_validator_contract.py::test_batch_mislabeled_reasoning_key_is_relabeled_and_reported",
        runner="pytest",
        evidence=("reasoning_relabeled", "normalization_reported"),
        faked_dependencies=("validator model replies", "WebSocket broadcaster"),
        observed_transition="a misspelled batch reasoning key is relabeled and reported for the validator role",
    ),
    WorkflowSourceMapping(
        invariant_id="model_output.normalizations_visible_and_guarded",
        scenario_id="focused_rigor_discovery_protected_decision_never_relabeled",
        adapter="focused_production",
        result="passed",
        production_sources=("backend/compiler/agents/high_param_submitter.py", "backend/shared/json_key_recovery.py"),
        test_selector="tests/unit/test_compiler_model_output_contract.py::test_discovery_never_relabels_protected_decision_key",
        runner="pytest",
        evidence=("decision_key_untouched",),
        faked_dependencies=("rigor discovery model replies",),
        observed_transition="a misspelled needs_theorem_work decision key is never relabeled",
    ),
    WorkflowSourceMapping(
        invariant_id="model_output.normalizations_visible_and_guarded",
        scenario_id="focused_normalization_activity_routed_in_frontend",
        adapter="focused_production",
        result="passed",
        production_sources=("frontend/src/App.jsx", "frontend/src/utils/activityStyles.js"),
        test_selector="frontend/src/components/modelOutputNormalizedActivity.test.jsx::model_output_normalized Live Activity routing::routes by workflow mode exactly like provider retry rows",
        runner="vitest",
        evidence=("workflow_mode_routing",),
        faked_dependencies=("WebSocket events supplied by Vitest",),
        observed_transition="model_output_normalized rows reach the owning mode's Live Activity feed",
    ),
    WorkflowSourceMapping(
        invariant_id="model_output.normalizations_visible_and_guarded",
        scenario_id="focused_normalization_activity_rendered_once",
        adapter="focused_production",
        result="passed",
        production_sources=("frontend/src/utils/activityStyles.js",),
        test_selector="frontend/src/utils/activityStyles.test.js::model output normalization activity::shows the backend sentence with the route identity exactly once",
        runner="vitest",
        evidence=("route_identity_once",),
        faked_dependencies=("activity payloads supplied by Vitest",),
        observed_transition="the normalization sentence is rendered with its route identity exactly once",
    ),
)


BUILD_07_REAL_ADAPTER_COVERAGE = tuple(
    mapping.coverage_record() for mapping in BUILD_07_SOURCE_MAPPINGS
)
