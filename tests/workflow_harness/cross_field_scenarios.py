from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from . import actions
from .coverage_metadata import AdapterType, InteractionCoverage, ResultStatus
from .invariant_catalog import FIRST_BUILD_FIELDS, INVARIANTS_BY_ID
from .model import WorkflowModel


ScenarioAction = Callable[[WorkflowModel], None]


@dataclass(frozen=True)
class CrossFieldScenario:
    scenario_id: str
    fields: tuple[str, ...]
    invariants: tuple[str, ...]
    adapter: AdapterType
    actions: tuple[ScenarioAction, ...]
    must_exercise: tuple[str, ...] = ()
    expected_result: ResultStatus = "passed"
    notes: str = ""

    def coverage_record(self, test_file: str) -> InteractionCoverage:
        return InteractionCoverage(
            scenario_id=self.scenario_id,
            fields=self.fields,
            invariants=self.invariants,
            adapter=self.adapter,
            result=self.expected_result,
            test_file=test_file,
            replay=tuple(action.__name__ for action in self.actions),
            evidence=self.must_exercise or self.invariants,
        )


def validate_cross_field_scenario(scenario: CrossFieldScenario) -> None:
    if not scenario.scenario_id:
        raise AssertionError("Cross-field scenario is missing scenario_id.")
    if len(scenario.fields) < 2:
        raise AssertionError(f"{scenario.scenario_id} must cross at least two fields.")
    if len(set(scenario.fields)) != len(scenario.fields):
        raise AssertionError(f"{scenario.scenario_id} declares duplicate fields.")
    if not scenario.invariants:
        raise AssertionError(f"{scenario.scenario_id} does not declare invariants.")
    if scenario.expected_result == "passed" and not scenario.actions:
        raise AssertionError(f"{scenario.scenario_id} has no executable actions.")

    for field in scenario.fields:
        if field not in FIRST_BUILD_FIELDS:
            raise AssertionError(f"{scenario.scenario_id} references unknown field {field!r}.")
    catalog_fields_for_invariants: set[str] = set()
    for invariant_id in scenario.invariants:
        if invariant_id not in INVARIANTS_BY_ID:
            raise AssertionError(
                f"{scenario.scenario_id} references unknown invariant {invariant_id!r}."
            )
        invariant = INVARIANTS_BY_ID[invariant_id]
        catalog_fields_for_invariants.add(invariant.field)
        catalog_fields_for_invariants.update(invariant.crossed_fields)
        if scenario.adapter not in invariant.adapters and scenario.expected_result != "blocked":
            raise AssertionError(
                f"{scenario.scenario_id} uses adapter {scenario.adapter!r} for "
                f"invariant {invariant_id!r}, but the catalog does not list that adapter."
            )
    unmatched_fields = set(scenario.fields) - catalog_fields_for_invariants
    if unmatched_fields:
        raise AssertionError(
            f"{scenario.scenario_id} declares fields not connected to its invariants: "
            f"{sorted(unmatched_fields)!r}."
        )


CROSS_FIELD_SCENARIOS: tuple[CrossFieldScenario, ...] = (
    CrossFieldScenario(
        scenario_id="model_allowed_outputs_provider_pause_stop_resume",
        fields=("allowed_outputs", "provider_pause_resume", "workflow_filesystem_state"),
        invariants=(
            "outputs.proofs_only_no_paper_phase",
            "provider.pause_preserves_checkpoint",
            "provider.stop_resume_preserves_pause",
            "provider.reset_wakes_without_corrupting_checkpoint",
        ),
        adapter="model",
        actions=(
            actions.start_autonomous_proofs_only,
            actions.complete_topic_exploration,
            actions.simulate_credit_exhaustion,
            actions.complete_brainstorm,
            actions.stop,
            actions.resume,
            actions.reset_credit,
        ),
        must_exercise=("provider_pause", "stop_resume", "reset_credit"),
    ),
    CrossFieldScenario(
        scenario_id="model_assistant_prompt_validator_exclusion",
        fields=("assistant_memory", "prompt_context"),
        invariants=(
            "assistant.no_validator_injection",
            "assistant.disable_clears_live_pack",
            "prompt.validator_excludes_assistant_memory",
        ),
        adapter="model",
        actions=(
            actions.start_autonomous_papers_and_proofs,
            actions.refresh_assistant_pack,
            actions.prepare_prompt_context,
            actions.disable_session_history,
        ),
        must_exercise=("assistant_pack", "assistant_disable", "prompt_context"),
    ),
    CrossFieldScenario(
        scenario_id="model_manual_proof_clear_scope_assistant",
        fields=("proof_scope_isolation", "workflow_filesystem_state", "assistant_memory"),
        invariants=(
            "proof_scope.manual_not_in_autonomous_current",
            "proof_scope.manual_clear_archives_active",
            "state.clear_removes_active_preserves_history",
            "assistant.disable_clears_live_pack",
        ),
        adapter="model",
        actions=(
            actions.start_manual_compiler,
            actions.enable_manual_lean,
            actions.run_manual_proof_check,
            actions.refresh_assistant_pack,
            actions.clear,
        ),
        must_exercise=("manual_proof", "manual_archive", "assistant_pack"),
    ),
    CrossFieldScenario(
        scenario_id="model_websocket_proof_scope_events",
        fields=("websocket_api_contracts", "proof_scope_isolation"),
        invariants=(
            "events.frontend_events_include_scope_phase",
            "events.proof_verified_after_registration",
            "proof_scope.autonomous_events_not_manual",
        ),
        adapter="model",
        actions=(
            actions.start_autonomous_papers_and_proofs,
            actions.emit_frontend_scoped_event,
            actions.emit_registered_proof_verified,
        ),
        must_exercise=("scoped_event", "registered_proof_event"),
    ),
    CrossFieldScenario(
        scenario_id="model_proof_runtime_gating_allowed_outputs",
        fields=("proof_runtime_gating", "allowed_outputs"),
        invariants=(
            "outputs.at_least_one_output_enabled",
            "outputs.papers_only_skips_proof_work",
            "proof_runtime.no_lean_when_disabled",
            "proof_runtime.no_smt_when_disabled",
        ),
        adapter="model",
        actions=(
            actions.start_autonomous_no_outputs,
            actions.start_autonomous_papers_only,
            actions.complete_topic_exploration,
            actions.complete_brainstorm,
        ),
        must_exercise=("no_outputs_rejected", "papers_only"),
    ),
    CrossFieldScenario(
        scenario_id="model_proof_pruning_continuous_contracts",
        fields=(
            "proof_pruning",
            "continuous_proof_loop",
            "proof_scope_isolation",
            "workflow_filesystem_state",
            "assistant_memory",
            "prompt_context",
            "provider_pause_resume",
            "proof_runtime_gating",
            "websocket_api_contracts",
            "runtime_exclusivity",
        ),
        invariants=(
            "proof_pruning.artifacts_and_future_memory_preserved",
            "proof_pruning.owning_run_context_excludes_pruned",
            "proof_pruning.validator_gates_automatic_mutation",
            "proof_pruning.semantic_distinct_review_preserves_unique_routes",
            "proof_pruning.no_prune_is_valid",
            "proof_pruning.review_non_blocking",
            "proof_pruning.commit_lifecycle_fenced",
            "proof_pruning.context_overflow_truthful",
            "proof_loop.continuous_explicit_ownership",
            "proof_loop.automatic_round_policy_preserved",
            "proof_pruning.occurrence_scope_isolated",
        ),
        adapter="model",
        actions=(actions.exercise_proof_pruning_and_continuous_contracts,),
        must_exercise=(
            "pruned_artifact_preserved",
            "owning_context_excluded",
            "validator_gate",
            "no_prune",
            "non_blocking_pruning",
            "stale_commit_fenced",
            "overflow_truthful",
            "continuous_ownership",
            "automatic_round_policy",
            "occurrence_scope",
        ),
    ),
    CrossFieldScenario(
        scenario_id="model_overflow_scope_terminal_matrix",
        fields=(
            "websocket_api_contracts",
            "proof_runtime_gating",
            "workflow_filesystem_state",
            "runtime_exclusivity",
            "continuous_proof_loop",
        ),
        invariants=(
            "events.proof_context_overflow_nonfatal",
            "events.autonomous_mandatory_overflow_checkpoint_preserved",
            "events.continuous_manual_overflow_terminal_after_drain",
        ),
        adapter="model",
        actions=(actions.exercise_overflow_lifecycle_contracts,),
        must_exercise=(
            "candidate_local_overflow",
            "autonomous_mandatory_overflow",
            "continuous_manual_overflow",
        ),
    ),
    CrossFieldScenario(
        scenario_id="model_current_product_law_families",
        fields=(
            "solution_path",
            "proof_competition",
            "provider_routing",
            "feedback_projection",
            "bulk_mutation",
            "runtime_root_ownership",
            "proof_run_identity",
            "proof_framing",
            "sleep_ownership",
            "answer_wrappers",
            "group_novelty",
            "leanoj_parent_control",
            "runtime_exclusivity",
            "workflow_filesystem_state",
        ),
        invariants=(
            "solution_path.activation_review_fenced",
            "proof_competition.secondary_after_primary_exhaustion",
            "provider.exact_role_and_cooldown_scope",
            "feedback_projection.newest_five_whole",
            "bulk_mutation.atomic_revision_fenced",
            "runtime_root.single_owner_bound_paths",
            "proof_run.identity_generation_fenced",
            "proof_framing.allowed_outputs_gate",
            "sleep.unique_owner_released",
            "supercharge_creativity.routing_separation",
            "proof_novelty.group_atomic_same_run",
            "leanoj.parent_control_generation_fences_children",
        ),
        adapter="model",
        actions=(actions.exercise_current_product_laws,),
        must_exercise=("current_product_laws",),
    ),
    CrossFieldScenario(
        scenario_id="model_live_activity_provider_retry_assurance",
        fields=(
            "websocket_api_contracts",
            "provider_pause_resume",
            "provider_routing",
            "workflow_filesystem_state",
        ),
        invariants=(
            "live_activity.provider_retry_each_wait_visible",
            "live_activity.provider_retry_identity_stable",
            "live_activity.provider_retry_secret_safe",
            "live_activity.provider_retry_persists_reload",
            "live_activity.provider_error_route_identity",
            "live_activity.provider_error_sources_route_attributed",
        ),
        adapter="model",
        actions=(actions.exercise_live_activity_assurance,),
        must_exercise=(
            "provider_retry_each_wait_visible",
            "provider_retry_attempts_distinct",
            "provider_retry_correlation_stable",
            "provider_retry_secret_redacted",
            "provider_retry_reload_visible",
            "provider_error_route_identity",
        ),
    ),
)


def first_build_cross_field_coverage(test_file: str) -> tuple[InteractionCoverage, ...]:
    return tuple(scenario.coverage_record(test_file) for scenario in CROSS_FIELD_SCENARIOS)
