"""Test-only Build 07 mappings from product laws to observable source seams."""
from __future__ import annotations

from dataclasses import dataclass

from .coverage_metadata import AdapterType, InteractionCoverage, ResultStatus


@dataclass(frozen=True)
class WorkflowSourceMapping:
    invariant_id: str
    scenario_id: str
    adapter: AdapterType
    result: ResultStatus
    production_sources: tuple[str, ...]
    test_selector: str | None
    evidence: tuple[str, ...] = ()
    blocked_reason: str | None = None

    def coverage_record(self) -> InteractionCoverage:
        test_file = (
            self.test_selector.split("::", 1)[0]
            if self.test_selector
            else "tests/workflow_real_adapters/test_build_07_proof_pruning_mappings.py"
        )
        return InteractionCoverage(
            scenario_id=self.scenario_id,
            fields=BUILD_07_FIELDS[self.invariant_id],
            invariants=(self.invariant_id,),
            adapter=self.adapter,
            result=self.result,
            test_file=test_file,
            diagnostics={"reason": self.blocked_reason} if self.blocked_reason else None,
            evidence=self.evidence,
            runner="pytest" if self.test_selector else None,
            test_selectors=(self.test_selector,) if self.test_selector else (),
            asserted_invariants=(self.invariant_id,) if self.test_selector else (),
        )


BUILD_07_FIELDS = {
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
        invariant_id="proof_pruning.artifacts_and_future_memory_preserved",
        scenario_id="real_pruned_proof_artifacts_and_future_memory_preserved",
        adapter="real_coordinator",
        result="passed",
        production_sources=(
            "backend/autonomous/memory/proof_database.py",
            "backend/shared/proof_search/search_service.py",
        ),
        test_selector=(
            "tests/regressions/test_proof_context_regressions.py::"
            "ProofContextRegressionTests::"
            "test_owning_run_prune_filters_model_context_but_not_canonical_records"
        ),
        evidence=("human_visible", "future_run_retrievable"),
    ),
    WorkflowSourceMapping(
        invariant_id="proof_pruning.owning_run_context_excludes_pruned",
        scenario_id="real_owning_run_context_and_cached_assistant_exclude_pruned",
        adapter="real_coordinator",
        result="passed",
        production_sources=(
            "backend/autonomous/memory/proof_database.py",
            "backend/shared/proof_search/assistant_cache.py",
        ),
        test_selector=(
            "tests/unit/test_assistant_proof_search.py::"
            "AssistantProofPackPayloadTests::test_cached_pack_filters_active_run"
        ),
        evidence=("cached_pack_refiltered", "owning_run_excluded"),
    ),
    WorkflowSourceMapping(
        invariant_id="proof_pruning.validator_gates_automatic_mutation",
        scenario_id="real_automatic_prune_requires_validator_acceptance",
        adapter="real_coordinator",
        result="passed",
        production_sources=(
            "backend/autonomous/agents/proof_pruning_agent.py",
            "backend/autonomous/core/proof_pruning_coordinator.py",
        ),
        test_selector=(
            "tests/unit/test_proof_pruning_agents.py::"
            "ProofPruningContractTests::test_validator_cannot_replace_target"
        ),
        evidence=("independent_validator", "retarget_rejected"),
    ),
    WorkflowSourceMapping(
        invariant_id="proof_pruning.no_prune_is_valid",
        scenario_id="real_no_prune_is_valid_non_mutating_result",
        adapter="real_coordinator",
        result="passed",
        production_sources=("backend/autonomous/core/proof_pruning_coordinator.py",),
        test_selector=(
            "tests/unit/test_proof_pruning_coordinator.py::"
            "ProofPruningCoordinatorTests::test_third_eligible_registration_schedules_without_blocking"
        ),
        evidence=("no_prune_outcome", "no_mutation"),
    ),
    WorkflowSourceMapping(
        invariant_id="proof_pruning.review_non_blocking",
        scenario_id="real_pruning_review_pending_does_not_block_registration",
        adapter="real_coordinator",
        result="passed",
        production_sources=("backend/autonomous/core/proof_pruning_coordinator.py",),
        test_selector=(
            "tests/unit/test_proof_pruning_coordinator.py::"
            "ProofPruningCoordinatorTests::test_notify_registration_is_immediate_and_owned"
        ),
        evidence=("immediate_notification", "held_review"),
    ),
    WorkflowSourceMapping(
        invariant_id="proof_pruning.commit_lifecycle_fenced",
        scenario_id="real_prune_commit_rechecks_lifecycle_and_revision",
        adapter="real_coordinator",
        result="passed",
        production_sources=(
            "backend/autonomous/core/proof_pruning_coordinator.py",
            "backend/autonomous/memory/proof_database.py",
        ),
        test_selector=(
            "tests/unit/test_proof_pruning_snapshot.py::"
            "ProofPruningSnapshotTests::test_commit_allows_unrelated_addition_but_rechecks_target"
        ),
        evidence=("target_rechecked", "stale_target_rejected"),
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
        blocked_reason=(
            "Existing isolated tests observe nonfatal proof overflow and pruning triggers separately; "
            "no bounded seam observes an overflow arriving during a prune commit without synthesizing "
            "the transition owner."
        ),
    ),
    WorkflowSourceMapping(
        invariant_id="proof_loop.continuous_explicit_ownership",
        scenario_id="real_continuous_loop_stop_terminal_zero_policy",
        adapter="real_coordinator",
        result="blocked",
        production_sources=(
            "backend/autonomous/core/proof_round_driver.py",
            "backend/autonomous/core/proof_run_manager.py",
        ),
        test_selector=None,
        blocked_reason=(
            "Focused manager and driver tests observe no-candidate continuation, detailed round "
            "activity, Stop cleanup, and non-resumable state separately; no bounded real adapter "
            "currently observes the complete continuous route lifecycle as one interaction."
        ),
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
            "tests/unit/test_autonomous_proof_rounds.py::"
            "AutonomousProofRoundTests::test_proofs_only_brainstorm_rounds_continue_until_no_candidates"
        ),
        evidence=("first_zero_exits", "proofs_only_up_to_four"),
    ),
    WorkflowSourceMapping(
        invariant_id="proof_pruning.occurrence_scope_isolated",
        scenario_id="real_pruning_is_occurrence_and_owning_run_scoped",
        adapter="real_coordinator",
        result="passed",
        production_sources=("backend/autonomous/memory/proof_database.py",),
        test_selector=(
            "tests/unit/test_proof_pruning_snapshot.py::"
            "ProofPruningSnapshotTests::test_same_run_pruned_occurrence_is_not_reviewed"
        ),
        evidence=("same_run_filtered", "occurrence_status"),
    ),
)


BUILD_07_REAL_ADAPTER_COVERAGE = tuple(
    mapping.coverage_record() for mapping in BUILD_07_SOURCE_MAPPINGS
)
