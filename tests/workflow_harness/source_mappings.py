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
            "backend/shared/sakana_fugu_client.py",
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
)


BUILD_07_REAL_ADAPTER_COVERAGE = tuple(
    mapping.coverage_record() for mapping in BUILD_07_SOURCE_MAPPINGS
)
