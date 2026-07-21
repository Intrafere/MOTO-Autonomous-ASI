from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from .model import FakeAssistantMemory, FakeLean, FakeProvider, FakeSmt, WorkflowEvent, WorkflowMode, WorkflowPhase


AdapterType = Literal["model", "real_route", "real_coordinator", "browser_smoke"]


class InvariantState(Protocol):
    runtime_root: Path
    mode: WorkflowMode
    phase: WorkflowPhase
    allow_mathematical_proofs: bool
    allow_research_papers: bool
    provider: FakeProvider
    lean: FakeLean
    smt: FakeSmt
    assistant: FakeAssistantMemory
    autonomous_proofs: set[str]
    manual_proofs_active: set[str]
    manual_proofs_archived: set[str]
    manual_proofs_before_clear: set[str]
    active_owners: set[WorkflowMode]
    background_owners: set[WorkflowMode]
    pending_child_tasks: int
    stale_child_outputs_fenced: bool
    stale_child_outputs_accepted: int
    truncation_failures: int
    provider_pause_from_truncation: int
    hosted_desktop_route_attempted: bool
    hosted_desktop_route_unavailable: bool
    hosted_route_attempts: set[str]
    hosted_route_unavailable: set[str]
    active_state_cleared: bool
    history_preserved: bool
    completed_brainstorm_generated_paper: bool
    replayed_completed_brainstorm_handoff: bool
    pruned_papers: set[str]
    model_context_papers: set[str]
    runtime_root_violations: int
    resolved_runtime_paths: list[Path]
    prompt_user_direct_injected: bool
    validator_prompt_has_assistant_memory: bool
    proof_source_context_required: bool
    proof_source_context_present: bool
    generated_appendices_stripped: bool
    direct_sources_excluded_from_rag: bool
    mandatory_source_overflow_visible: bool
    exercise_observations: set[str]
    persisted_events: list[WorkflowEvent]
    terminal_stop_events: int
    checkpoint: dict[str, object]
    events: list[WorkflowEvent]
    replay: list[str]


InvariantChecker = Callable[[InvariantState], None]


PAPER_PHASES = {
    WorkflowPhase.PAPER_TITLE,
    WorkflowPhase.PAPER_WRITING,
    WorkflowPhase.PAPER_PROOF,
}


@dataclass(frozen=True)
class InvariantSpec:
    invariant_id: str
    field: str
    crossed_fields: tuple[str, ...]
    adapters: tuple[AdapterType, ...]
    checker: InvariantChecker
    description: str


def _fail(model: InvariantState, message: str) -> None:
    replay = "\n".join(f"{index + 1}. {action}" for index, action in enumerate(model.replay))
    raise AssertionError(f"{message}\n\nReplay:\n{replay}")


def _assert_single_active_workflow(model: InvariantState) -> None:
    owners = model.active_owners | model.background_owners
    if model.mode is not WorkflowMode.NONE:
        owners.add(model.mode)
    if len(owners) > 1:
        _fail(model, f"Competing workflow owners are active: {sorted(owner.value for owner in owners)!r}")
    if model.mode is WorkflowMode.NONE and model.active_owners:
        _fail(model, "Public mode is idle while a top-level owner remains active.")
    if model.mode is not WorkflowMode.NONE and model.active_owners and model.active_owners != {model.mode}:
        _fail(model, "Public mode and active workflow ownership disagree.")


def _assert_child_tasks_count_as_active(model: InvariantState) -> None:
    if model.pending_child_tasks and not (model.active_owners | model.background_owners):
        _fail(model, "Pending child workflow activity exists while no owner workflow is active.")


def _assert_parent_action_fences_child_outputs(model: InvariantState) -> None:
    if not model.stale_child_outputs_fenced or model.stale_child_outputs_accepted:
        _fail(model, "A parent workflow action did not fence stale child outputs.")


def _assert_lean_disabled_means_no_invocations(model: InvariantState) -> None:
    if model.lean.blocked_invocations:
        _fail(model, "Lean was invoked while lean4_enabled was false.")


def _assert_smt_disabled_means_no_invocations(model: InvariantState) -> None:
    if model.smt.blocked_invocations:
        _fail(model, "SMT was invoked while smt_enabled was false.")


def _assert_hosted_proof_settings_unavailable(model: InvariantState) -> None:
    if model.hosted_desktop_route_attempted and not model.hosted_desktop_route_unavailable:
        _fail(model, "Hosted mode attempted to run a desktop-only proof settings route.")


def _assert_truncation_is_attempt_failure(model: WorkflowModel) -> None:
    if model.truncation_failures and model.provider_pause_from_truncation:
        _fail(model, "Provider max-output truncation was treated as a provider pause.")


def _assert_at_least_one_output_enabled(model: WorkflowModel) -> None:
    rejected = any(
        event.event_type == "start_rejected" and event.payload.get("reason") == "no_allowed_outputs"
        for event in model.events
    )
    if not model.allow_mathematical_proofs and not model.allow_research_papers and not rejected:
        _fail(model, "A workflow state disabled both proofs and papers without rejecting start.")


def _assert_proofs_only_never_enters_paper_phase(model: WorkflowModel) -> None:
    if (
        model.mode is WorkflowMode.AUTONOMOUS
        and model.allow_mathematical_proofs
        and not model.allow_research_papers
        and model.phase in PAPER_PHASES
    ):
        _fail(model, f"Proofs-only autonomous run entered paper phase {model.phase.value!r}.")


def _assert_papers_only_skips_proof_work(model: WorkflowModel) -> None:
    if model.mode is WorkflowMode.AUTONOMOUS and not model.allow_mathematical_proofs:
        if model.lean.invocations or model.smt.invocations:
            _fail(model, "Papers-only autonomous run performed proof work.")


def _assert_provider_pause_preserves_checkpoint(model: InvariantState) -> None:
    if model.phase is WorkflowPhase.PAUSED:
        required = {
            "paused",
            "phase",
            "resume_phase",
            "source_type",
            "source_id",
            "trigger",
            "candidate_cursor",
            "proof_round",
        }
        missing = required - set(model.checkpoint)
        if missing:
            _fail(model, f"Provider pause checkpoint is missing {sorted(missing)!r}.")
        if model.checkpoint.get("paused") is not True:
            _fail(model, "Provider pause did not mark the checkpoint as paused.")


def _assert_stop_resume_preserves_pause(model: WorkflowModel) -> None:
    if model.checkpoint.get("paused") and model.checkpoint.get("stopped"):
        if not model.checkpoint.get("resume_phase"):
            _fail(model, "Stop during provider pause did not preserve the resume phase.")


def _assert_reset_wakes_without_corrupting_checkpoint(model: WorkflowModel) -> None:
    if model.provider.pause_count and model.checkpoint.get("paused") is False:
        if not model.checkpoint.get("phase"):
            _fail(model, "Provider reset cleared the checkpoint phase.")


def _assert_assistant_not_injected_into_validators(model: WorkflowModel) -> None:
    if model.assistant.validator_injections:
        _fail(model, "Assistant memory was injected into a validator role.")


def _assert_disabled_assistant_has_no_live_pack(model: WorkflowModel) -> None:
    if not model.assistant.enabled and model.assistant.live_pack:
        _fail(model, "Session History Memory is disabled but an Assistant pack remains live.")


def _assert_assistant_retrieval_non_blocking(model: WorkflowModel) -> None:
    if model.assistant.blocked_parent_count:
        _fail(model, "Assistant retrieval blocked parent workflow progress.")


def _assert_stagnant_pack_backoff_not_shutdown(model: WorkflowModel) -> None:
    if model.assistant.stagnant_backoff_count and model.assistant.shutdown_count:
        _fail(model, "Stagnant Assistant pack backoff shut down proof-memory retrieval.")


def _assert_manual_and_autonomous_proofs_are_isolated(model: WorkflowModel) -> None:
    overlap = model.autonomous_proofs.intersection(model.manual_proofs_active)
    if overlap:
        _fail(model, f"Manual active proofs overlap autonomous proofs: {sorted(overlap)!r}")


def _assert_autonomous_events_not_manual(model: WorkflowModel) -> None:
    leaked = [
        event
        for event in model.events
        if event.payload.get("scope") == "manual" and str(event.payload.get("proof_id", "")).startswith("auto")
    ]
    if leaked:
        _fail(model, "Autonomous proof events leaked into manual proof scope.")


def _assert_clear_archives_manual_proofs(model: InvariantState) -> None:
    if model.phase is WorkflowPhase.IDLE and model.mode is WorkflowMode.NONE and model.manual_proofs_active:
        _fail(model, "Clear/idle state still has active manual proofs.")
    missing = model.manual_proofs_before_clear - model.manual_proofs_archived
    if missing:
        _fail(model, f"Manual clear failed to archive active proofs: {sorted(missing)!r}")


def _assert_generated_appendices_stripped_from_prompts(model: WorkflowModel) -> None:
    if not model.generated_appendices_stripped:
        _fail(model, "Generated proof appendices were not stripped from model-visible prompts.")


def _assert_clear_removes_active_preserves_history(model: WorkflowModel) -> None:
    if model.active_state_cleared and not model.history_preserved:
        _fail(model, "Clear/reset removed preserved history.")


def _assert_completed_brainstorm_no_replay_handoff(model: WorkflowModel) -> None:
    if model.completed_brainstorm_generated_paper and model.replayed_completed_brainstorm_handoff:
        _fail(model, "Completed brainstorm with generated paper replayed proof/paper handoff.")


def _assert_pruned_papers_excluded_from_context(model: WorkflowModel) -> None:
    overlap = model.pruned_papers.intersection(model.model_context_papers)
    if overlap:
        _fail(model, f"Pruned papers remain in model context: {sorted(overlap)!r}")


def _assert_runtime_roots_are_active_roots(model: InvariantState) -> None:
    if model.runtime_root_violations:
        _fail(model, "Runtime state resolved outside the active runtime root.")
    for path in model.resolved_runtime_paths:
        try:
            path.resolve().relative_to(model.runtime_root.resolve())
        except ValueError:
            _fail(model, f"Runtime path resolved outside active root: {path!s}")


def _assert_frontend_events_include_scope_phase(model: WorkflowModel) -> None:
    checked_prefixes = ("proof_", "provider_", "assistant_", "workflow_")
    for event in model.events:
        if event.event_type.startswith(checked_prefixes):
            if "scope" not in event.payload or "phase" not in event.payload:
                _fail(model, f"Frontend event {event.event_type!r} lacks scope or phase.")


def _assert_context_overflow_route_identity(model: WorkflowModel) -> None:
    required = {"workflow_mode", "role_id", "configured_model", "configured_provider"}
    for event in model.events:
        if event.event_type not in {"context_overflow_error", "proof_context_overflow"}:
            continue
        missing = required - set(event.payload)
        if missing:
            _fail(model, f"{event.event_type!r} lacks route identity fields {sorted(missing)!r}.")
        has_effective_model = bool(event.payload.get("effective_model"))
        has_effective_provider = bool(event.payload.get("effective_provider"))
        if has_effective_model != has_effective_provider:
            _fail(model, f"{event.event_type!r} has incomplete effective route identity.")


def _assert_context_overflow_persists_across_reload(model: WorkflowModel) -> None:
    live = [event for event in model.events if event.event_type == "context_overflow_error"]
    persisted = [
        event for event in model.persisted_events if event.event_type == "context_overflow_error"
    ]
    if live and not persisted:
        _fail(model, "Fatal context overflow activity was not persisted for reload.")
    if live and persisted and live[-1].payload != persisted[-1].payload:
        _fail(model, "Reloaded context overflow activity lost live event metadata.")


def _assert_context_overflow_terminal_stop_once(model: WorkflowModel) -> None:
    fatal_overflows = [
        event
        for event in model.events
        if event.event_type == "context_overflow_error" and event.payload.get("fatal") is not False
    ]
    if fatal_overflows and model.terminal_stop_events != 1:
        _fail(model, "Fatal context overflow did not produce exactly one terminal stop activity.")


def _assert_proof_context_overflow_nonfatal(model: WorkflowModel) -> None:
    proof_overflows = [
        event for event in model.events if event.event_type == "proof_context_overflow"
    ]
    if any(event.payload.get("fatal") is not False for event in proof_overflows):
        _fail(model, "Per-proof context overflow was treated as fatal.")


def _assert_proof_verified_after_registration(model: WorkflowModel) -> None:
    for event in model.events:
        if event.event_type == "proof_verified":
            proof_id = str(event.payload.get("proof_id") or "")
            if not proof_id:
                _fail(model, "proof_verified event did not include proof_id.")
            if (
                proof_id not in model.autonomous_proofs
                and proof_id not in model.manual_proofs_active
                and proof_id not in model.manual_proofs_archived
            ):
                _fail(model, f"proof_verified emitted before proof registration: {proof_id!r}.")


def _assert_hosted_desktop_only_routes_unavailable(model: InvariantState) -> None:
    required = {"proof_settings", "update", "pdf", "desktop_oauth"}
    attempted = required & model.hosted_route_attempts
    unavailable = required & model.hosted_route_unavailable
    if attempted != unavailable:
        _fail(model, f"Hosted desktop-only routes were not uniformly unavailable: {sorted(attempted - unavailable)!r}")


def _assert_user_prompt_direct_injected(model: WorkflowModel) -> None:
    if not model.prompt_user_direct_injected:
        _fail(model, "User prompt was not direct injected.")


def _assert_validator_excludes_assistant_memory(model: WorkflowModel) -> None:
    if model.validator_prompt_has_assistant_memory:
        _fail(model, "Validator prompt included Assistant memory.")


def _assert_proof_source_context_required(model: WorkflowModel) -> None:
    if model.proof_source_context_required and not model.proof_source_context_present:
        _fail(model, "Proof source context was required but absent.")


def _assert_direct_sources_excluded_from_rag(model: WorkflowModel) -> None:
    if not model.direct_sources_excluded_from_rag:
        _fail(model, "A directly injected source was duplicated into supplemental RAG evidence.")


def _assert_mandatory_source_overflow_visible(model: WorkflowModel) -> None:
    if not model.mandatory_source_overflow_visible:
        _fail(model, "Mandatory source overflow was truncated or hidden by optional RAG context.")


INVARIANT_CATALOG: tuple[InvariantSpec, ...] = (
    InvariantSpec(
        invariant_id="runtime.single_active_workflow",
        field="runtime_exclusivity",
        crossed_fields=("provider_pause_resume", "workflow_filesystem_state", "websocket_api_contracts"),
        adapters=("model", "real_route", "browser_smoke"),
        checker=_assert_single_active_workflow,
        description="Only one top-level workflow mode may be active at a time.",
    ),
    InvariantSpec(
        invariant_id="runtime.child_tasks_count_as_active",
        field="runtime_exclusivity",
        crossed_fields=("provider_pause_resume", "workflow_filesystem_state", "websocket_api_contracts"),
        adapters=("model", "real_route"),
        checker=_assert_child_tasks_count_as_active,
        description="Pending or background child activity counts as workflow ownership.",
    ),
    InvariantSpec(
        invariant_id="runtime.parent_action_fences_child_outputs",
        field="runtime_exclusivity",
        crossed_fields=("workflow_filesystem_state", "websocket_api_contracts"),
        adapters=("model",),
        checker=_assert_parent_action_fences_child_outputs,
        description="Parent actions fence stale lower-tier child outputs.",
    ),
    InvariantSpec(
        invariant_id="proof_runtime.no_lean_when_disabled",
        field="proof_runtime_gating",
        crossed_fields=("allowed_outputs", "provider_pause_resume", "prompt_context"),
        adapters=("model", "real_coordinator"),
        checker=_assert_lean_disabled_means_no_invocations,
        description="Lean proof work must not run when Lean is disabled.",
    ),
    InvariantSpec(
        invariant_id="proof_runtime.no_smt_when_disabled",
        field="proof_runtime_gating",
        crossed_fields=("allowed_outputs", "prompt_context"),
        adapters=("model",),
        checker=_assert_smt_disabled_means_no_invocations,
        description="SMT hint generation must not run when SMT is disabled.",
    ),
    InvariantSpec(
        invariant_id="proof_runtime.hosted_proof_settings_unavailable",
        field="proof_runtime_gating",
        crossed_fields=("websocket_api_contracts", "allowed_outputs"),
        adapters=("model", "real_route"),
        checker=_assert_hosted_proof_settings_unavailable,
        description="Hosted/generic proof settings routes remain unavailable.",
    ),
    InvariantSpec(
        invariant_id="proof_runtime.truncation_is_attempt_failure",
        field="proof_runtime_gating",
        crossed_fields=("provider_pause_resume", "prompt_context"),
        adapters=("model",),
        checker=_assert_truncation_is_attempt_failure,
        description="Provider max-output truncation is a proof attempt failure, not a provider pause.",
    ),
    InvariantSpec(
        invariant_id="outputs.at_least_one_output_enabled",
        field="allowed_outputs",
        crossed_fields=("proof_runtime_gating",),
        adapters=("model", "real_route", "browser_smoke"),
        checker=_assert_at_least_one_output_enabled,
        description="Workflow starts reject requests with both proof and paper outputs disabled.",
    ),
    InvariantSpec(
        invariant_id="outputs.proofs_only_no_paper_phase",
        field="allowed_outputs",
        crossed_fields=("proof_runtime_gating", "provider_pause_resume", "workflow_filesystem_state"),
        adapters=("model", "real_coordinator"),
        checker=_assert_proofs_only_never_enters_paper_phase,
        description="Proofs-only autonomous runs must not enter title, paper writing, or paper proof phases.",
    ),
    InvariantSpec(
        invariant_id="outputs.papers_only_skips_proof_work",
        field="allowed_outputs",
        crossed_fields=("proof_runtime_gating", "provider_pause_resume"),
        adapters=("model",),
        checker=_assert_papers_only_skips_proof_work,
        description="Papers-only autonomous runs skip proof model work and proof tool execution.",
    ),
    InvariantSpec(
        invariant_id="provider.pause_preserves_checkpoint",
        field="provider_pause_resume",
        crossed_fields=("workflow_filesystem_state", "proof_runtime_gating", "runtime_exclusivity"),
        adapters=("model", "real_coordinator"),
        checker=_assert_provider_pause_preserves_checkpoint,
        description="Provider-credit pauses preserve a resumable checkpoint.",
    ),
    InvariantSpec(
        invariant_id="provider.stop_resume_preserves_pause",
        field="provider_pause_resume",
        crossed_fields=("workflow_filesystem_state", "runtime_exclusivity"),
        adapters=("model",),
        checker=_assert_stop_resume_preserves_pause,
        description="Stop/resume during provider pause preserves pause checkpoint state.",
    ),
    InvariantSpec(
        invariant_id="provider.reset_wakes_without_corrupting_checkpoint",
        field="provider_pause_resume",
        crossed_fields=("workflow_filesystem_state", "proof_runtime_gating"),
        adapters=("model",),
        checker=_assert_reset_wakes_without_corrupting_checkpoint,
        description="Provider reset wakes paused work without corrupting checkpoint recovery.",
    ),
    InvariantSpec(
        invariant_id="assistant.no_validator_injection",
        field="assistant_memory",
        crossed_fields=("prompt_context", "proof_scope_isolation", "provider_pause_resume"),
        adapters=("model",),
        checker=_assert_assistant_not_injected_into_validators,
        description="Validators must not receive Assistant proof-memory packs.",
    ),
    InvariantSpec(
        invariant_id="assistant.disable_clears_live_pack",
        field="assistant_memory",
        crossed_fields=("prompt_context", "proof_scope_isolation"),
        adapters=("model",),
        checker=_assert_disabled_assistant_has_no_live_pack,
        description="Disabling Session History Memory clears the live Assistant pack.",
    ),
    InvariantSpec(
        invariant_id="assistant.non_blocking_retrieval",
        field="assistant_memory",
        crossed_fields=("prompt_context", "provider_pause_resume"),
        adapters=("model",),
        checker=_assert_assistant_retrieval_non_blocking,
        description="Assistant retrieval is optional and never blocks parent workflow progress.",
    ),
    InvariantSpec(
        invariant_id="assistant.stagnant_pack_backoff_not_shutdown",
        field="assistant_memory",
        crossed_fields=("provider_pause_resume",),
        adapters=("model",),
        checker=_assert_stagnant_pack_backoff_not_shutdown,
        description="Stagnant same-pack retrieval backs off without shutting down the run.",
    ),
    InvariantSpec(
        invariant_id="proof_scope.manual_not_in_autonomous_current",
        field="proof_scope_isolation",
        crossed_fields=("assistant_memory", "workflow_filesystem_state", "websocket_api_contracts"),
        adapters=("model", "real_route", "real_coordinator"),
        checker=_assert_manual_and_autonomous_proofs_are_isolated,
        description="Manual active proofs must not overlap autonomous current-session proofs.",
    ),
    InvariantSpec(
        invariant_id="proof_scope.autonomous_events_not_manual",
        field="proof_scope_isolation",
        crossed_fields=("websocket_api_contracts", "workflow_filesystem_state"),
        adapters=("model", "real_coordinator"),
        checker=_assert_autonomous_events_not_manual,
        description="Autonomous proof events do not populate manual proof scopes.",
    ),
    InvariantSpec(
        invariant_id="proof_scope.manual_clear_archives_active",
        field="proof_scope_isolation",
        crossed_fields=("workflow_filesystem_state", "assistant_memory"),
        adapters=("model", "real_route", "real_coordinator"),
        checker=_assert_clear_archives_manual_proofs,
        description="Manual clear leaves no active manual proofs in an idle cleared state.",
    ),
    InvariantSpec(
        invariant_id="proof_scope.generated_appendices_stripped_from_prompts",
        field="proof_scope_isolation",
        crossed_fields=("prompt_context", "workflow_filesystem_state"),
        adapters=("model", "real_coordinator"),
        checker=_assert_generated_appendices_stripped_from_prompts,
        description="Generated proof appendices are stripped from future model-visible source prompts.",
    ),
    InvariantSpec(
        invariant_id="state.clear_removes_active_preserves_history",
        field="workflow_filesystem_state",
        crossed_fields=("proof_scope_isolation", "assistant_memory"),
        adapters=("model", "real_route", "real_coordinator"),
        checker=_assert_clear_removes_active_preserves_history,
        description="Clear removes active runtime state while preserving intended history archives.",
    ),
    InvariantSpec(
        invariant_id="state.completed_brainstorm_no_replay_handoff",
        field="workflow_filesystem_state",
        crossed_fields=("allowed_outputs", "provider_pause_resume"),
        adapters=("model",),
        checker=_assert_completed_brainstorm_no_replay_handoff,
        description="Completed brainstorms with generated papers do not replay proof/paper handoff on resume.",
    ),
    InvariantSpec(
        invariant_id="state.pruned_papers_excluded_from_context",
        field="workflow_filesystem_state",
        crossed_fields=("prompt_context",),
        adapters=("model", "real_route"),
        checker=_assert_pruned_papers_excluded_from_context,
        description="Pruned papers remain history but are excluded from model context.",
    ),
    InvariantSpec(
        invariant_id="state.runtime_roots_are_active_roots",
        field="workflow_filesystem_state",
        crossed_fields=("runtime_exclusivity",),
        adapters=("model", "real_coordinator"),
        checker=_assert_runtime_roots_are_active_roots,
        description="Runtime files resolve under active runtime roots.",
    ),
    InvariantSpec(
        invariant_id="events.frontend_events_include_scope_phase",
        field="websocket_api_contracts",
        crossed_fields=("proof_scope_isolation", "provider_pause_resume"),
        adapters=("model",),
        checker=_assert_frontend_events_include_scope_phase,
        description="Frontend-consumed proof/provider/Assistant/workflow events include scope and phase.",
    ),
    InvariantSpec(
        invariant_id="events.context_overflow_route_identity",
        field="websocket_api_contracts",
        crossed_fields=("provider_pause_resume", "workflow_filesystem_state"),
        adapters=("model", "real_coordinator", "browser_smoke"),
        checker=_assert_context_overflow_route_identity,
        description="Context-overflow activity identifies configured and effective model routes.",
    ),
    InvariantSpec(
        invariant_id="events.context_overflow_persists_across_reload",
        field="websocket_api_contracts",
        crossed_fields=("workflow_filesystem_state",),
        adapters=("model", "real_coordinator", "browser_smoke"),
        checker=_assert_context_overflow_persists_across_reload,
        description="Fatal context-overflow metadata survives the workflow's persistence and reload path.",
    ),
    InvariantSpec(
        invariant_id="events.context_overflow_terminal_stop_once",
        field="websocket_api_contracts",
        crossed_fields=("runtime_exclusivity", "workflow_filesystem_state"),
        adapters=("model", "real_coordinator", "browser_smoke"),
        checker=_assert_context_overflow_terminal_stop_once,
        description="A fatal context overflow produces one terminal workflow-stop activity.",
    ),
    InvariantSpec(
        invariant_id="events.proof_context_overflow_nonfatal",
        field="websocket_api_contracts",
        crossed_fields=("proof_runtime_gating", "provider_pause_resume"),
        adapters=("model", "real_coordinator", "browser_smoke"),
        checker=_assert_proof_context_overflow_nonfatal,
        description="Per-proof context overflow is scoped and nonfatal to its parent workflow.",
    ),
    InvariantSpec(
        invariant_id="events.proof_verified_after_registration",
        field="websocket_api_contracts",
        crossed_fields=("proof_scope_isolation",),
        adapters=("model",),
        checker=_assert_proof_verified_after_registration,
        description="proof_verified emits only after proof registration or reuse and includes proof_id.",
    ),
    InvariantSpec(
        invariant_id="api.hosted_desktop_only_routes_unavailable",
        field="websocket_api_contracts",
        crossed_fields=("proof_runtime_gating", "runtime_exclusivity"),
        adapters=("model", "browser_smoke"),
        checker=_assert_hosted_desktop_only_routes_unavailable,
        description="Hosted/generic routes report desktop-only behavior unavailable.",
    ),
    InvariantSpec(
        invariant_id="prompt.user_prompt_direct_injected",
        field="prompt_context",
        crossed_fields=("assistant_memory", "proof_runtime_gating"),
        adapters=("model",),
        checker=_assert_user_prompt_direct_injected,
        description="User prompt remains direct injected where required.",
    ),
    InvariantSpec(
        invariant_id="prompt.validator_excludes_assistant_memory",
        field="prompt_context",
        crossed_fields=("assistant_memory",),
        adapters=("model",),
        checker=_assert_validator_excludes_assistant_memory,
        description="Validator prompts exclude Assistant proof-memory packs.",
    ),
    InvariantSpec(
        invariant_id="prompt.proof_source_context_required",
        field="prompt_context",
        crossed_fields=("proof_runtime_gating",),
        adapters=("model",),
        checker=_assert_proof_source_context_required,
        description="Proof formalization receives mandatory source context or fails visibly.",
    ),
    InvariantSpec(
        invariant_id="prompt.direct_sources_excluded_from_rag",
        field="prompt_context",
        crossed_fields=("workflow_filesystem_state",),
        adapters=("model", "real_coordinator"),
        checker=_assert_direct_sources_excluded_from_rag,
        description="Directly injected sources are excluded from supplemental RAG evidence.",
    ),
    InvariantSpec(
        invariant_id="prompt.mandatory_source_overflow_fails_visible",
        field="prompt_context",
        crossed_fields=("proof_runtime_gating", "websocket_api_contracts"),
        adapters=("model", "real_coordinator"),
        checker=_assert_mandatory_source_overflow_visible,
        description="Mandatory source overflow fails visibly before model calls or RAG substitution.",
    ),
    InvariantSpec(
        invariant_id="prompt.generated_appendices_stripped",
        field="prompt_context",
        crossed_fields=("proof_scope_isolation",),
        adapters=("model", "real_coordinator"),
        checker=_assert_generated_appendices_stripped_from_prompts,
        description="Model-visible paper and brainstorm source reads strip generated proof appendices.",
    ),
)


INVARIANTS_BY_ID = {spec.invariant_id: spec for spec in INVARIANT_CATALOG}
FIRST_BUILD_INVARIANT_IDS = frozenset(INVARIANTS_BY_ID)
FIRST_BUILD_FIELDS = frozenset(spec.field for spec in INVARIANT_CATALOG)


def get_invariant(invariant_id: str) -> InvariantSpec:
    return INVARIANTS_BY_ID[invariant_id]


def assert_invariant(model: InvariantState, invariant_id: str) -> None:
    get_invariant(invariant_id).checker(model)


def assert_all_catalog_invariants(model: InvariantState) -> None:
    for spec in INVARIANT_CATALOG:
        spec.checker(model)


def invariant_ids_for_adapter(adapter: AdapterType) -> set[str]:
    return {spec.invariant_id for spec in INVARIANT_CATALOG if adapter in spec.adapters}
