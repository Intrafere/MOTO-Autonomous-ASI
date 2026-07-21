from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from . import actions
from .exercise_evidence import KNOWN_EXERCISE_TOKENS, observed_exercise_tokens
from .invariant_catalog import (
    FIRST_BUILD_FIELDS,
    INVARIANTS_BY_ID,
    INVARIANT_CATALOG,
)
from .model import WorkflowMode, WorkflowModel, WorkflowPhase


Eligibility = Callable[[WorkflowModel], bool]
FailurePredicate = Callable[[WorkflowModel], bool]
ModelFactory = Callable[[], WorkflowModel]


@dataclass(frozen=True)
class ActionSpec:
    action_id: str
    execute: actions.WorkflowAction
    fields: tuple[str, ...]
    eligible_when: Eligibility
    weight: int = 1
    advances: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()


@dataclass(frozen=True)
class GenerationTarget:
    target_id: str
    fields: tuple[str, ...]
    invariants: tuple[str, ...]
    must_exercise: tuple[str, ...]
    max_steps: int = 18


@dataclass(frozen=True)
class GeneratedRun:
    target_id: str
    seed: int
    fields: tuple[str, ...]
    observed_fields: frozenset[str]
    invariants: tuple[str, ...]
    exercised_invariants: frozenset[str]
    action_ids: tuple[str, ...]
    replay: tuple[str, ...]
    observed_evidence: frozenset[str]
    final_mode: WorkflowMode
    final_phase: WorkflowPhase

    @property
    def action_count(self) -> int:
        return len(self.action_ids)


class GeneratedRunFailure(AssertionError):
    pass


@dataclass(frozen=True)
class InvariantFailure(Exception):
    invariant_id: str
    detail: str


class ReplayRejected(ValueError):
    """The requested replay is not a canonical executable action sequence."""


REPLAY_NOT_REPRODUCED = "replay.not_reproduced"


@dataclass(frozen=True)
class ReplayResult:
    action_ids: tuple[str, ...]
    replay: tuple[str, ...]
    failure_category: str | None = None
    failure_detail: str = ""

    @property
    def failed(self) -> bool:
        return (
            self.failure_category is not None
            and self.failure_category != REPLAY_NOT_REPRODUCED
        )


@dataclass(frozen=True)
class ReplayReduction:
    original_action_ids: tuple[str, ...]
    shortest_failing_prefix: tuple[str, ...]
    action_ids: tuple[str, ...]
    failure_category: str | None

    @property
    def minimized(self) -> bool:
        return self.action_ids != self.original_action_ids


INVARIANT_ACTIVATION_EVIDENCE: dict[str, frozenset[str]] = {
    "runtime.single_active_workflow": frozenset({"start_blocked"}),
    "outputs.at_least_one_output_enabled": frozenset({"no_outputs_rejected"}),
    "outputs.proofs_only_no_paper_phase": frozenset({"provider_pause"}),
    "outputs.papers_only_skips_proof_work": frozenset({"papers_only"}),
    "provider.pause_preserves_checkpoint": frozenset({"provider_pause"}),
    "provider.stop_resume_preserves_pause": frozenset({"stop_resume"}),
    "provider.reset_wakes_without_corrupting_checkpoint": frozenset({"reset_credit"}),
    "assistant.no_validator_injection": frozenset({"prompt_context"}),
    "assistant.disable_clears_live_pack": frozenset({"assistant_disable"}),
    "proof_scope.manual_not_in_autonomous_current": frozenset({"manual_proof"}),
    "proof_scope.manual_clear_archives_active": frozenset({"manual_archive"}),
    "state.clear_removes_active_preserves_history": frozenset({"manual_archive"}),
    "events.frontend_events_include_scope_phase": frozenset({"scoped_event"}),
    "events.proof_verified_after_registration": frozenset({"registered_proof_event"}),
    "proof_scope.autonomous_events_not_manual": frozenset({"registered_proof_event"}),
    "proof_runtime.no_lean_when_disabled": frozenset({"papers_only"}),
    "proof_runtime.no_smt_when_disabled": frozenset({"papers_only"}),
    "prompt.validator_excludes_assistant_memory": frozenset({"prompt_context"}),
    "events.context_overflow_route_identity": frozenset({"context_overflow"}),
    "events.context_overflow_persists_across_reload": frozenset({"context_overflow"}),
    "events.context_overflow_terminal_stop_once": frozenset({"context_overflow"}),
    "events.proof_context_overflow_nonfatal": frozenset({"context_overflow"}),
    "prompt.user_prompt_direct_injected": frozenset({"prompt_context"}),
    "prompt.proof_source_context_required": frozenset({"prompt_context"}),
    "prompt.direct_sources_excluded_from_rag": frozenset({"rag_source_exclusion"}),
    "prompt.mandatory_source_overflow_fails_visible": frozenset({"mandatory_source_overflow"}),
    "prompt.generated_appendices_stripped": frozenset({"prompt_context"}),
}


def _idle(model: WorkflowModel) -> bool:
    return model.mode is WorkflowMode.NONE and not model.checkpoint


def _active(model: WorkflowModel) -> bool:
    return model.mode is not WorkflowMode.NONE


def _autonomous_phase(phase: WorkflowPhase) -> Eligibility:
    return lambda model: model.mode is WorkflowMode.AUTONOMOUS and model.phase is phase


def _manual_writing(model: WorkflowModel) -> bool:
    return (
        model.mode is WorkflowMode.MANUAL_COMPILER
        and model.phase is WorkflowPhase.PAPER_WRITING
        and not model.lean.enabled
    )


def _manual_proof_ready(model: WorkflowModel) -> bool:
    return model.mode is WorkflowMode.MANUAL_COMPILER and model.lean.enabled


def _manual_aggregating(model: WorkflowModel) -> bool:
    return (
        model.mode is WorkflowMode.MANUAL_AGGREGATOR
        and model.phase is WorkflowPhase.MANUAL_AGGREGATION
    )


def _leanoj_phase(phase: WorkflowPhase) -> Eligibility:
    return lambda model: model.mode is WorkflowMode.LEANOJ and model.phase is phase


def _leanoj_active(model: WorkflowModel) -> bool:
    return model.mode is WorkflowMode.LEANOJ


def _can_stop(model: WorkflowModel) -> bool:
    return _active(model)


def _can_resume(model: WorkflowModel) -> bool:
    return model.mode is WorkflowMode.NONE and bool(model.checkpoint)


def _can_reset_credit(model: WorkflowModel) -> bool:
    return (
        model.mode is not WorkflowMode.NONE
        and model.phase is WorkflowPhase.PAUSED
        and model.provider.credit_exhausted
    )


def _can_disable_assistant(model: WorkflowModel) -> bool:
    return model.assistant.enabled and bool(model.assistant.live_pack)


def _can_refresh_assistant(model: WorkflowModel) -> bool:
    return _active(model) and model.assistant.enabled and not model.assistant.live_pack


def _can_prepare_prompt_context(model: WorkflowModel) -> bool:
    return _active(model) and model.assistant.enabled


def _can_clear_manual(model: WorkflowModel) -> bool:
    return (
        model.mode is WorkflowMode.MANUAL_COMPILER
        and bool(model.manual_proofs_active)
    )


def _can_clear_stopped_checkpoint(model: WorkflowModel) -> bool:
    return (
        model.mode is WorkflowMode.NONE
        and bool(model.checkpoint)
        and not model.provider.credit_exhausted
    )


ACTION_SPECS: tuple[ActionSpec, ...] = (
    ActionSpec(
        "start_autonomous_proofs_only",
        actions.start_autonomous_proofs_only,
        ("allowed_outputs", "provider_pause_resume", "workflow_filesystem_state"),
        _idle,
        advances=("provider_pause", "stop_resume", "reset_credit"),
    ),
    ActionSpec(
        "start_autonomous_papers_and_proofs",
        actions.start_autonomous_papers_and_proofs,
        ("allowed_outputs", "assistant_memory", "prompt_context", "websocket_api_contracts"),
        _idle,
        advances=(
            "assistant_pack",
            "assistant_disable",
            "prompt_context",
            "registered_proof_event",
            "scoped_event",
        ),
    ),
    ActionSpec(
        "start_autonomous_papers_only",
        actions.start_autonomous_papers_only,
        ("allowed_outputs", "proof_runtime_gating"),
        _idle,
        advances=("papers_only",),
    ),
    ActionSpec(
        "start_autonomous_no_outputs",
        actions.start_autonomous_no_outputs,
        ("allowed_outputs", "proof_runtime_gating"),
        _idle,
        produces=("no_outputs_rejected",),
    ),
    ActionSpec(
        "start_manual_compiler",
        actions.start_manual_compiler,
        ("proof_scope_isolation", "workflow_filesystem_state"),
        _idle,
        advances=("manual_proof", "manual_archive"),
    ),
    ActionSpec(
        "start_manual_aggregator",
        actions.start_manual_aggregator,
        ("runtime_exclusivity", "workflow_filesystem_state"),
        _idle,
        advances=("manual_aggregator_accept", "manual_aggregator_clear", "stop_resume"),
    ),
    ActionSpec(
        "accept_manual_aggregator_submission",
        actions.accept_manual_aggregator_submission,
        ("workflow_filesystem_state", "prompt_context"),
        _manual_aggregating,
        produces=("manual_aggregator_accept",),
    ),
    ActionSpec(
        "start_leanoj",
        actions.start_leanoj,
        ("runtime_exclusivity", "workflow_filesystem_state"),
        _idle,
        advances=("leanoj_draft", "leanoj_skip", "leanoj_force", "stop_resume", "leanoj_clear"),
    ),
    ActionSpec(
        "edit_leanoj_master_proof",
        actions.edit_leanoj_master_proof,
        ("workflow_filesystem_state", "prompt_context"),
        _leanoj_active,
        produces=("leanoj_draft",),
    ),
    ActionSpec(
        "skip_leanoj_brainstorm",
        actions.skip_leanoj_brainstorm,
        ("runtime_exclusivity", "workflow_filesystem_state"),
        _leanoj_phase(WorkflowPhase.LEANOJ_BRAINSTORM),
        produces=("leanoj_skip",),
        advances=("leanoj_force",),
    ),
    ActionSpec(
        "force_leanoj_brainstorm",
        actions.force_leanoj_brainstorm,
        ("runtime_exclusivity", "workflow_filesystem_state"),
        _leanoj_phase(WorkflowPhase.LEANOJ_FINAL),
        produces=("leanoj_force",),
    ),
    ActionSpec(
        "attempt_conflicting_start",
        actions.start_manual_compiler,
        ("runtime_exclusivity", "websocket_api_contracts"),
        _active,
        produces=("start_blocked",),
    ),
    ActionSpec(
        "complete_topic_exploration",
        actions.complete_topic_exploration,
        ("allowed_outputs", "provider_pause_resume", "workflow_filesystem_state"),
        _autonomous_phase(WorkflowPhase.TOPIC_EXPLORATION),
        advances=("provider_pause", "reset_credit", "papers_only"),
    ),
    ActionSpec(
        "simulate_credit_exhaustion",
        actions.simulate_credit_exhaustion,
        ("provider_pause_resume", "proof_runtime_gating"),
        lambda model: (
            model.mode is WorkflowMode.AUTONOMOUS
            and model.phase is WorkflowPhase.TIER1_AGGREGATION
            and model.allow_mathematical_proofs
            and not model.provider.credit_exhausted
        ),
        weight=20,
        advances=("provider_pause", "reset_credit"),
    ),
    ActionSpec(
        "complete_brainstorm",
        actions.complete_brainstorm,
        ("allowed_outputs", "provider_pause_resume", "proof_runtime_gating"),
        _autonomous_phase(WorkflowPhase.TIER1_AGGREGATION),
        advances=("provider_pause", "reset_credit", "papers_only"),
    ),
    ActionSpec(
        "force_paper_writing",
        actions.force_paper_writing,
        ("allowed_outputs", "workflow_filesystem_state"),
        _autonomous_phase(WorkflowPhase.TIER1_AGGREGATION),
    ),
    ActionSpec(
        "complete_paper",
        actions.complete_paper,
        ("allowed_outputs", "proof_runtime_gating", "workflow_filesystem_state"),
        _autonomous_phase(WorkflowPhase.PAPER_TITLE),
    ),
    ActionSpec(
        "enter_autonomous_paper_checkpoint",
        actions.enter_autonomous_paper_checkpoint,
        ("allowed_outputs", "proof_runtime_gating", "workflow_filesystem_state"),
        _autonomous_phase(WorkflowPhase.PAPER_TITLE),
        advances=("paper_checkpoint",),
    ),
    ActionSpec(
        "complete_autonomous_paper_checkpoint",
        actions.complete_autonomous_paper_checkpoint,
        ("proof_runtime_gating", "workflow_filesystem_state", "websocket_api_contracts"),
        _autonomous_phase(WorkflowPhase.PAPER_PROOF),
        produces=("paper_checkpoint",),
    ),
    ActionSpec(
        "attempt_conflict_during_autonomous_paper_checkpoint",
        actions.attempt_conflict_during_autonomous_paper_checkpoint,
        ("runtime_exclusivity", "websocket_api_contracts"),
        _autonomous_phase(WorkflowPhase.PAPER_PROOF),
        produces=("start_blocked",),
    ),
    ActionSpec(
        "enable_manual_lean",
        actions.enable_manual_lean,
        ("proof_runtime_gating", "proof_scope_isolation"),
        _manual_writing,
        advances=("manual_proof", "manual_archive"),
    ),
    ActionSpec(
        "run_manual_proof_check",
        actions.run_manual_proof_check,
        ("proof_runtime_gating", "proof_scope_isolation"),
        _manual_proof_ready,
        advances=("manual_archive",),
        produces=("manual_proof",),
    ),
    ActionSpec(
        "refresh_assistant_pack",
        actions.refresh_assistant_pack,
        ("assistant_memory", "prompt_context", "proof_scope_isolation"),
        _can_refresh_assistant,
        advances=("assistant_disable", "manual_archive"),
        produces=("assistant_pack",),
    ),
    ActionSpec(
        "prepare_prompt_context",
        actions.prepare_prompt_context,
        ("prompt_context", "assistant_memory", "proof_scope_isolation"),
        _can_prepare_prompt_context,
        produces=("prompt_context",),
    ),
    ActionSpec(
        "disable_session_history",
        actions.disable_session_history,
        ("assistant_memory", "prompt_context"),
        _can_disable_assistant,
        produces=("assistant_disable",),
    ),
    ActionSpec(
        "emit_frontend_scoped_event",
        actions.emit_frontend_scoped_event,
        ("websocket_api_contracts", "proof_scope_isolation"),
        _active,
        produces=("scoped_event",),
    ),
    ActionSpec(
        "emit_registered_proof_verified",
        actions.emit_registered_proof_verified,
        ("websocket_api_contracts", "proof_scope_isolation"),
        _active,
        produces=("registered_proof_event", "scoped_event"),
    ),
    ActionSpec(
        "emit_context_overflow_contract_events",
        actions.emit_context_overflow_contract_events,
        (
            "websocket_api_contracts",
            "workflow_filesystem_state",
            "provider_pause_resume",
            "runtime_exclusivity",
            "proof_runtime_gating",
        ),
        _active,
        produces=("context_overflow",),
    ),
    ActionSpec(
        "verify_rag_source_exclusion",
        actions.verify_rag_source_exclusion,
        ("prompt_context", "workflow_filesystem_state"),
        _active,
        produces=("rag_source_exclusion",),
    ),
    ActionSpec(
        "reject_mandatory_source_overflow",
        actions.reject_mandatory_source_overflow,
        ("prompt_context", "proof_runtime_gating", "websocket_api_contracts"),
        _active,
        produces=("mandatory_source_overflow",),
    ),
    ActionSpec(
        "stop",
        actions.stop,
        ("provider_pause_resume", "workflow_filesystem_state", "runtime_exclusivity"),
        _can_stop,
        advances=("stop_resume",),
    ),
    ActionSpec(
        "resume",
        actions.resume,
        ("provider_pause_resume", "workflow_filesystem_state", "runtime_exclusivity"),
        _can_resume,
        produces=("stop_resume",),
    ),
    ActionSpec(
        "reset_credit",
        actions.reset_credit,
        ("provider_pause_resume", "workflow_filesystem_state"),
        _can_reset_credit,
        produces=("reset_credit",),
    ),
    ActionSpec(
        "clear_manual_state",
        actions.clear,
        ("proof_scope_isolation", "workflow_filesystem_state", "assistant_memory"),
        _can_clear_manual,
        produces=("manual_archive",),
    ),
    ActionSpec(
        "clear_stopped_checkpoint",
        actions.clear,
        ("workflow_filesystem_state", "runtime_exclusivity"),
        _can_clear_stopped_checkpoint,
    ),
    ActionSpec(
        "clear_manual_aggregator_state",
        actions.clear,
        ("workflow_filesystem_state", "runtime_exclusivity", "prompt_context"),
        lambda model: _manual_aggregating(model) and bool(model.manual_aggregator_submissions),
        produces=("manual_aggregator_clear",),
    ),
    ActionSpec(
        "clear_leanoj_state",
        actions.clear,
        ("workflow_filesystem_state", "runtime_exclusivity", "prompt_context"),
        _leanoj_active,
        produces=("leanoj_clear",),
    ),
)


DEFAULT_TARGETS: tuple[GenerationTarget, ...] = (
    GenerationTarget(
        target_id="generated_allowed_outputs_provider_pause_resume",
        fields=("allowed_outputs", "provider_pause_resume", "workflow_filesystem_state"),
        invariants=(
            "outputs.proofs_only_no_paper_phase",
            "provider.pause_preserves_checkpoint",
            "provider.stop_resume_preserves_pause",
            "provider.reset_wakes_without_corrupting_checkpoint",
        ),
        must_exercise=("provider_pause", "stop_resume", "reset_credit"),
        max_steps=16,
    ),
    GenerationTarget(
        target_id="generated_assistant_prompt_context",
        fields=("assistant_memory", "prompt_context"),
        invariants=(
            "assistant.disable_clears_live_pack",
            "assistant.no_validator_injection",
            "prompt.validator_excludes_assistant_memory",
        ),
        must_exercise=("assistant_pack", "prompt_context", "assistant_disable"),
        max_steps=12,
    ),
    GenerationTarget(
        target_id="generated_manual_scope_clear_archive",
        fields=("proof_scope_isolation", "workflow_filesystem_state", "assistant_memory"),
        invariants=(
            "proof_scope.manual_not_in_autonomous_current",
            "proof_scope.manual_clear_archives_active",
            "state.clear_removes_active_preserves_history",
        ),
        must_exercise=("manual_proof", "assistant_pack", "manual_archive"),
        max_steps=14,
    ),
    GenerationTarget(
        target_id="generated_scoped_registered_proof_events",
        fields=("websocket_api_contracts", "proof_scope_isolation"),
        invariants=(
            "events.frontend_events_include_scope_phase",
            "events.proof_verified_after_registration",
            "proof_scope.autonomous_events_not_manual",
        ),
        must_exercise=("scoped_event", "registered_proof_event"),
        max_steps=10,
    ),
    GenerationTarget(
        target_id="generated_output_and_proof_runtime_gating",
        fields=("proof_runtime_gating", "allowed_outputs"),
        invariants=(
            "outputs.at_least_one_output_enabled",
            "outputs.papers_only_skips_proof_work",
            "proof_runtime.no_lean_when_disabled",
            "proof_runtime.no_smt_when_disabled",
        ),
        must_exercise=("no_outputs_rejected", "papers_only"),
        max_steps=10,
    ),
    GenerationTarget(
        target_id="generated_runtime_start_exclusivity",
        fields=("runtime_exclusivity", "websocket_api_contracts"),
        invariants=("runtime.single_active_workflow",),
        must_exercise=("start_blocked",),
        max_steps=8,
    ),
    GenerationTarget(
        target_id="generated_context_overflow_contract",
        fields=(
            "websocket_api_contracts",
            "workflow_filesystem_state",
            "provider_pause_resume",
            "runtime_exclusivity",
            "proof_runtime_gating",
        ),
        invariants=(
            "events.context_overflow_route_identity",
            "events.context_overflow_persists_across_reload",
            "events.context_overflow_terminal_stop_once",
            "events.proof_context_overflow_nonfatal",
        ),
        must_exercise=("context_overflow",),
        max_steps=6,
    ),
    GenerationTarget(
        target_id="generated_rag_prompt_boundaries",
        fields=(
            "prompt_context",
            "workflow_filesystem_state",
            "proof_runtime_gating",
            "websocket_api_contracts",
        ),
        invariants=(
            "prompt.user_prompt_direct_injected",
            "prompt.proof_source_context_required",
            "prompt.direct_sources_excluded_from_rag",
            "prompt.mandatory_source_overflow_fails_visible",
            "prompt.generated_appendices_stripped",
        ),
        must_exercise=(
            "prompt_context",
            "rag_source_exclusion",
            "mandatory_source_overflow",
        ),
        max_steps=8,
    ),
    GenerationTarget(
        target_id="generated_manual_aggregator_lifecycle",
        fields=("runtime_exclusivity", "workflow_filesystem_state"),
        invariants=("runtime.single_active_workflow",),
        must_exercise=(
            "manual_aggregator_accept",
            "manual_aggregator_clear",
            "start_blocked",
        ),
        max_steps=6,
    ),
    GenerationTarget(
        target_id="generated_leanoj_durable_lifecycle",
        fields=("runtime_exclusivity", "workflow_filesystem_state"),
        invariants=(
            "runtime.single_active_workflow",
            "provider.stop_resume_preserves_pause",
        ),
        must_exercise=(
            "leanoj_draft",
            "leanoj_skip",
            "leanoj_force",
            "stop_resume",
            "leanoj_clear",
            "start_blocked",
        ),
        max_steps=10,
    ),
    GenerationTarget(
        target_id="generated_autonomous_paper_checkpoint",
        fields=(
            "runtime_exclusivity",
            "workflow_filesystem_state",
            "websocket_api_contracts",
        ),
        invariants=("runtime.single_active_workflow",),
        must_exercise=("paper_checkpoint", "start_blocked"),
        max_steps=7,
    ),
)


def validate_action_specs(action_specs: tuple[ActionSpec, ...] = ACTION_SPECS) -> None:
    action_ids = [spec.action_id for spec in action_specs]
    if len(action_ids) != len(set(action_ids)):
        raise AssertionError("Action registry contains duplicate action IDs.")
    for spec in action_specs:
        if not spec.action_id or not callable(spec.execute) or not callable(spec.eligible_when):
            raise AssertionError(f"Invalid action descriptor: {spec!r}.")
        if spec.weight <= 0:
            raise AssertionError(f"{spec.action_id} must have a positive weight.")
        unknown_fields = set(spec.fields) - FIRST_BUILD_FIELDS
        if unknown_fields:
            raise AssertionError(
                f"{spec.action_id} references unknown fields {sorted(unknown_fields)!r}."
            )
        unknown_evidence = set(spec.advances).union(spec.produces) - KNOWN_EXERCISE_TOKENS
        if unknown_evidence:
            raise AssertionError(
                f"{spec.action_id} references unknown evidence {sorted(unknown_evidence)!r}."
            )


def action_specs_by_id(
    action_specs: tuple[ActionSpec, ...] = ACTION_SPECS,
) -> dict[str, ActionSpec]:
    """Return the validated canonical action registry keyed by stable action ID."""
    validate_action_specs(action_specs)
    return {spec.action_id: spec for spec in action_specs}


def _failure_from_model(
    model: WorkflowModel,
    *,
    failure_predicate: FailurePredicate | None,
    failure_category: str | None,
) -> tuple[str, str] | None:
    try:
        _assert_all_invariants_with_id(model)
    except InvariantFailure as exc:
        return exc.invariant_id, exc.detail
    if failure_predicate is not None and failure_predicate(model):
        return failure_category or "replay.failure_predicate", "Failure predicate matched."
    return None


def replay_action_ids(
    model: WorkflowModel,
    action_ids: tuple[str, ...] | list[str],
    *,
    action_specs: tuple[ActionSpec, ...] = ACTION_SPECS,
    failure_predicate: FailurePredicate | None = None,
    failure_category: str | None = None,
) -> ReplayResult:
    """Replay stable IDs, rejecting unknown or state-ineligible actions."""
    registry = action_specs_by_id(action_specs)
    canonical_ids = tuple(action_ids)
    for index, action_id in enumerate(canonical_ids):
        spec = registry.get(action_id)
        if spec is None:
            raise ReplayRejected(f"Unknown action ID {action_id!r} at index {index}.")
        if not spec.eligible_when(model):
            raise ReplayRejected(
                f"Action {action_id!r} is ineligible at index {index} "
                f"for mode={model.mode.value}, phase={model.phase.value}."
            )
        replay_before = len(model.replay)
        spec.execute(model)
        if len(model.replay) <= replay_before:
            raise ReplayRejected(
                f"Action {action_id!r} at index {index} did not record its execution."
            )
        failure = _failure_from_model(
            model,
            failure_predicate=failure_predicate,
            failure_category=failure_category,
        )
        if failure is not None:
            category, detail = failure
            if failure_category is not None and category != failure_category:
                return ReplayResult(
                    action_ids=canonical_ids[: index + 1],
                    replay=tuple(model.replay),
                    failure_category=REPLAY_NOT_REPRODUCED,
                    failure_detail=(
                        f"Requested failure category {failure_category!r} was not "
                        f"reproduced; observed {category!r}: {detail}"
                    ),
                )
            return ReplayResult(
                action_ids=canonical_ids[: index + 1],
                replay=tuple(model.replay),
                failure_category=category,
                failure_detail=detail,
            )
    if failure_category is not None:
        return ReplayResult(
            action_ids=canonical_ids,
            replay=tuple(model.replay),
            failure_category=REPLAY_NOT_REPRODUCED,
            failure_detail=(
                f"Requested failure category {failure_category!r} was not reproduced."
            ),
        )
    return ReplayResult(action_ids=canonical_ids, replay=tuple(model.replay))


def shortest_failing_prefix(
    model_factory: ModelFactory,
    action_ids: tuple[str, ...] | list[str],
    *,
    action_specs: tuple[ActionSpec, ...] = ACTION_SPECS,
    failure_predicate: FailurePredicate | None = None,
    failure_category: str | None = None,
) -> tuple[str, ...]:
    """Return the first prefix reproducing the requested failure category."""
    canonical_ids = tuple(action_ids)
    for size in range(1, len(canonical_ids) + 1):
        result = replay_action_ids(
            model_factory(),
            canonical_ids[:size],
            action_specs=action_specs,
            failure_predicate=failure_predicate,
            failure_category=failure_category,
        )
        if result.failed and result.failure_category != REPLAY_NOT_REPRODUCED and (
            failure_category is None or result.failure_category == failure_category
        ):
            return canonical_ids[:size]
    return canonical_ids


def reduce_failing_action_ids(
    model_factory: ModelFactory,
    action_ids: tuple[str, ...] | list[str],
    *,
    action_specs: tuple[ActionSpec, ...] = ACTION_SPECS,
    failure_predicate: FailurePredicate | None = None,
    failure_category: str | None = None,
) -> ReplayReduction:
    """Minimize a reproducing replay via shortest-prefix then chunk deletion."""
    original = tuple(action_ids)
    prefix = shortest_failing_prefix(
        model_factory,
        original,
        action_specs=action_specs,
        failure_predicate=failure_predicate,
        failure_category=failure_category,
    )
    baseline = replay_action_ids(
        model_factory(),
        prefix,
        action_specs=action_specs,
        failure_predicate=failure_predicate,
        failure_category=failure_category,
    )
    if not baseline.failed or baseline.failure_category == REPLAY_NOT_REPRODUCED:
        return ReplayReduction(original, prefix, original, None)

    category = failure_category or baseline.failure_category

    def reproduces(candidate: tuple[str, ...]) -> bool:
        if not candidate:
            return False
        try:
            result = replay_action_ids(
                model_factory(),
                candidate,
                action_specs=action_specs,
                failure_predicate=failure_predicate,
                failure_category=failure_category,
            )
        except ReplayRejected:
            return False
        return result.failed and result.failure_category == category

    reduced = prefix
    granularity = 2
    while len(reduced) >= 2:
        chunk_size = max(1, (len(reduced) + granularity - 1) // granularity)
        removed = False
        for start in range(0, len(reduced), chunk_size):
            candidate = reduced[:start] + reduced[start + chunk_size :]
            if reproduces(candidate):
                reduced = candidate
                granularity = max(2, granularity - 1)
                removed = True
                break
        if removed:
            continue
        if granularity >= len(reduced):
            break
        granularity = min(len(reduced), granularity * 2)

    return ReplayReduction(original, prefix, reduced, category)


def validate_generation_target(target: GenerationTarget) -> None:
    if not target.target_id:
        raise AssertionError("Generation target is missing target_id.")
    if len(target.fields) < 2 or len(set(target.fields)) != len(target.fields):
        raise AssertionError(f"{target.target_id} must declare at least two unique fields.")
    unknown_fields = set(target.fields) - FIRST_BUILD_FIELDS
    if unknown_fields:
        raise AssertionError(
            f"{target.target_id} references unknown fields {sorted(unknown_fields)!r}."
        )
    if not target.invariants:
        raise AssertionError(f"{target.target_id} does not declare invariants.")
    if len(set(target.invariants)) != len(target.invariants):
        raise AssertionError(f"{target.target_id} declares duplicate invariants.")
    if len(set(target.must_exercise)) != len(target.must_exercise):
        raise AssertionError(f"{target.target_id} declares duplicate evidence tokens.")

    connected_fields: set[str] = set()
    for invariant_id in target.invariants:
        invariant = INVARIANTS_BY_ID.get(invariant_id)
        if invariant is None:
            raise AssertionError(
                f"{target.target_id} references unknown invariant {invariant_id!r}."
            )
        if "model" not in invariant.adapters:
            raise AssertionError(
                f"{target.target_id} invariant {invariant_id!r} does not support model adapter."
            )
        connected_fields.add(invariant.field)
        connected_fields.update(invariant.crossed_fields)
        if invariant_id not in INVARIANT_ACTIVATION_EVIDENCE:
            raise AssertionError(
                f"{target.target_id} invariant {invariant_id!r} has no activation evidence."
            )
    unmatched_fields = set(target.fields) - connected_fields
    if unmatched_fields:
        raise AssertionError(
            f"{target.target_id} fields are not connected to its invariants: "
            f"{sorted(unmatched_fields)!r}."
        )
    unknown_evidence = set(target.must_exercise) - KNOWN_EXERCISE_TOKENS
    if unknown_evidence:
        raise AssertionError(
            f"{target.target_id} references unknown evidence {sorted(unknown_evidence)!r}."
        )
    if target.max_steps <= 0:
        raise AssertionError(f"{target.target_id} max_steps must be positive.")


def eligible_action_specs(
    model: WorkflowModel,
    action_specs: tuple[ActionSpec, ...] = ACTION_SPECS,
) -> tuple[ActionSpec, ...]:
    return tuple(spec for spec in action_specs if spec.eligible_when(model))


def _weight_for(
    spec: ActionSpec,
    target: GenerationTarget,
    missing_evidence: frozenset[str],
) -> int:
    field_overlap = len(set(spec.fields).intersection(target.fields))
    evidence_overlap = len(
        set(spec.advances).union(spec.produces).intersection(missing_evidence)
    )
    return spec.weight + (8 * field_overlap) + (128 * evidence_overlap)


def _signature_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return "<runtime-root>"
    if isinstance(value, dict):
        return tuple(
            (str(key), _signature_value(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_signature_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_signature_value(item) for item in value), key=repr))
    if hasattr(value, "__dataclass_fields__"):
        ignored = {"runtime_root", "replay"}
        return tuple(
            (name, _signature_value(getattr(value, name)))
            for name in value.__dataclass_fields__
            if name not in ignored
        )
    return value


def _planning_signature(
    model: WorkflowModel,
    observed: frozenset[str],
    observed_fields: frozenset[str],
) -> tuple[object, frozenset[str], frozenset[str]]:
    return (_signature_value(model), observed, observed_fields)


def _seeded_action_order(
    specs: tuple[ActionSpec, ...],
    *,
    target: GenerationTarget,
    seed: int,
    signature: object,
    missing_evidence: frozenset[str],
) -> tuple[ActionSpec, ...]:
    canonical_state = json.dumps(
        _signature_value(signature),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )

    def tie_break(spec: ActionSpec) -> str:
        material = (
            f"{seed}\0{target.target_id}\0{canonical_state}\0"
            f"{','.join(sorted(missing_evidence))}\0{spec.action_id}"
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    return tuple(
        sorted(
            specs,
            key=lambda spec: (
                -_weight_for(spec, target, missing_evidence),
                tie_break(spec),
                spec.action_id,
            ),
        )
    )


def _plan_reachable_path(
    model: WorkflowModel,
    target: GenerationTarget,
    *,
    seed: int,
    action_specs: tuple[ActionSpec, ...],
    observed: frozenset[str],
    observed_fields: frozenset[str],
    remaining_steps: int,
) -> tuple[ActionSpec, ...]:
    required_evidence = frozenset(target.must_exercise)
    required_fields = frozenset(target.fields)
    frontier: deque[
        tuple[WorkflowModel, frozenset[str], frozenset[str], tuple[ActionSpec, ...]]
    ] = deque([(deepcopy(model), observed, observed_fields, ())])
    visited = {_planning_signature(frontier[0][0], observed, observed_fields)}
    reachable_evidence = set(observed)
    reachable_fields = set(observed_fields)

    while frontier:
        state, state_observed, state_fields, path = frontier.popleft()
        if (
            required_evidence.issubset(state_observed)
            and required_fields.issubset(state_fields)
        ):
            return path
        if len(path) >= remaining_steps:
            continue

        missing = required_evidence - state_observed
        eligible = eligible_action_specs(state, action_specs)
        progress_actions = tuple(
            spec
            for spec in eligible
            if set(spec.advances).union(spec.produces).intersection(missing)
        )
        if progress_actions:
            eligible = progress_actions
        ordered = _seeded_action_order(
            eligible,
            target=target,
            seed=seed,
            signature=_planning_signature(state, state_observed, state_fields),
            missing_evidence=missing,
        )
        for spec in ordered:
            candidate = deepcopy(state)
            replay_before = len(candidate.replay)
            spec.execute(candidate)
            if len(candidate.replay) <= replay_before:
                continue
            try:
                _assert_all_invariants_with_id(candidate)
            except InvariantFailure:
                continue
            next_observed = frozenset(
                set(state_observed).union(observed_exercise_tokens(candidate))
            )
            next_fields = frozenset(set(state_fields).union(spec.fields))
            reachable_evidence.update(next_observed)
            reachable_fields.update(next_fields)
            signature = _planning_signature(candidate, next_observed, next_fields)
            if signature in visited:
                continue
            visited.add(signature)
            frontier.append((candidate, next_observed, next_fields, path + (spec,)))

    missing_evidence = sorted(required_evidence - reachable_evidence)
    missing_fields = sorted(required_fields - reachable_fields)
    raise GeneratedRunFailure(
        format_failure(
            seed=seed,
            fields=target.fields,
            action_count=0,
            invariant="generator.target_unreachable",
            replay=model.replay,
            detail=(
                f"No path within {remaining_steps} steps; "
                f"missing evidence {missing_evidence!r}; "
                f"missing fields {missing_fields!r}; "
                f"visited {len(visited)} state/evidence signatures."
            ),
        )
    )


def _assert_all_invariants_with_id(model: WorkflowModel) -> None:
    for invariant in INVARIANT_CATALOG:
        try:
            invariant.checker(model)
        except AssertionError as exc:
            raise InvariantFailure(invariant.invariant_id, str(exc)) from exc


def exercised_target_invariants(
    target: GenerationTarget,
    observed_evidence: set[str] | frozenset[str],
) -> frozenset[str]:
    return frozenset(
        invariant_id
        for invariant_id in target.invariants
        if INVARIANT_ACTIVATION_EVIDENCE[invariant_id].issubset(observed_evidence)
    )


def format_failure(
    *,
    seed: int,
    fields: tuple[str, ...],
    action_count: int,
    invariant: str,
    replay: tuple[str, ...] | list[str],
    detail: str,
) -> str:
    numbered_replay = "\n".join(
        f"{index}. {action}" for index, action in enumerate(replay, start=1)
    )
    return (
        f"Seed: {seed}\n"
        f"Fields: {' x '.join(fields) if fields else '(unknown)'}\n"
        f"Action count: {action_count}\n"
        f"Invariant: {invariant}\n"
        f"Detail: {detail}\n"
        f"Replay:\n{numbered_replay or '(empty)'}"
    )


def run_generated_scenario(
    model: WorkflowModel,
    target: GenerationTarget,
    *,
    seed: int,
    action_specs: tuple[ActionSpec, ...] = ACTION_SPECS,
) -> GeneratedRun:
    validate_action_specs(action_specs)
    validate_generation_target(target)
    selected_action_ids: list[str] = []
    observed: set[str] = set()
    observed_fields: set[str] = set()

    try:
        plan = _plan_reachable_path(
            model,
            target,
            seed=seed,
            action_specs=action_specs,
            observed=frozenset(),
            observed_fields=frozenset(),
            remaining_steps=target.max_steps,
        )
    except GeneratedRunFailure as exc:
        raise

    for chosen in plan:
        replay_before = len(model.replay)
        chosen.execute(model)
        selected_action_ids.append(chosen.action_id)
        if len(model.replay) <= replay_before:
            raise GeneratedRunFailure(
                format_failure(
                    seed=seed,
                    fields=target.fields,
                    action_count=len(selected_action_ids),
                    invariant="generator.action_records_replay",
                    replay=model.replay,
                    detail=f"Action {chosen.action_id!r} did not record its execution.",
                )
            )

        try:
            _assert_all_invariants_with_id(model)
        except InvariantFailure as exc:
            raise GeneratedRunFailure(
                format_failure(
                    seed=seed,
                    fields=target.fields,
                    action_count=len(selected_action_ids),
                    invariant=exc.invariant_id,
                    replay=model.replay,
                    detail=exc.detail,
                )
            ) from exc
        observed_fields.update(chosen.fields)
        observed.update(observed_exercise_tokens(model))

    missing = frozenset(target.must_exercise) - observed
    missing_fields = frozenset(target.fields) - observed_fields
    exercised_invariants = exercised_target_invariants(target, observed)
    missing_invariants = frozenset(target.invariants) - exercised_invariants
    if missing or missing_fields or missing_invariants:
        raise GeneratedRunFailure(
            format_failure(
                seed=seed,
                fields=target.fields,
                action_count=len(selected_action_ids),
                invariant="coverage.required_evidence",
                replay=model.replay,
                detail=(
                    f"Missing evidence {sorted(missing)!r}; "
                    f"missing fields {sorted(missing_fields)!r}; "
                    f"unexercised invariants {sorted(missing_invariants)!r}; "
                    f"observed evidence {sorted(observed)!r}; "
                    f"observed fields {sorted(observed_fields)!r}."
                ),
            )
        )

    return GeneratedRun(
        target_id=target.target_id,
        seed=seed,
        fields=target.fields,
        observed_fields=frozenset(observed_fields),
        invariants=target.invariants,
        exercised_invariants=exercised_invariants,
        action_ids=tuple(selected_action_ids),
        replay=tuple(model.replay),
        observed_evidence=frozenset(observed),
        final_mode=model.mode,
        final_phase=model.phase,
    )
