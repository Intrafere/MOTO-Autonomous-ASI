from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tests.workflow_harness.model import (
    FakeAssistantMemory,
    FakeLean,
    FakeProvider,
    FakeSmt,
    WorkflowEvent,
    WorkflowMode,
    WorkflowPhase,
)


@dataclass
class RealWorkflowObservation:
    """Small invariant-compatible view over real route/coordinator behavior."""

    runtime_root: Path
    mode: WorkflowMode = WorkflowMode.NONE
    phase: WorkflowPhase = WorkflowPhase.IDLE
    previous_phase: WorkflowPhase = WorkflowPhase.IDLE
    allow_mathematical_proofs: bool = True
    allow_research_papers: bool = True
    provider: FakeProvider = field(default_factory=FakeProvider)
    lean: FakeLean = field(default_factory=FakeLean)
    smt: FakeSmt = field(default_factory=FakeSmt)
    assistant: FakeAssistantMemory = field(default_factory=FakeAssistantMemory)
    autonomous_proofs: set[str] = field(default_factory=set)
    manual_proofs_active: set[str] = field(default_factory=set)
    manual_proofs_archived: set[str] = field(default_factory=set)
    manual_proofs_before_clear: set[str] = field(default_factory=set)
    active_owners: set[WorkflowMode] = field(default_factory=set)
    background_owners: set[WorkflowMode] = field(default_factory=set)
    pending_child_tasks: int = 0
    stale_child_outputs_fenced: bool | None = None
    stale_child_outputs_accepted: int = 0
    truncation_failures: int = 0
    provider_pause_from_truncation: int = 0
    hosted_desktop_route_attempted: bool = False
    hosted_desktop_route_unavailable: bool | None = None
    hosted_route_attempts: set[str] = field(default_factory=set)
    hosted_route_unavailable: set[str] = field(default_factory=set)
    active_state_cleared: bool = False
    history_preserved: bool | None = None
    completed_brainstorm_generated_paper: bool = False
    replayed_completed_brainstorm_handoff: bool = False
    pruned_papers: set[str] = field(default_factory=set)
    model_context_papers: set[str] = field(default_factory=set)
    runtime_root_violations: int = 0
    resolved_runtime_paths: list[Path] = field(default_factory=list)
    prompt_user_direct_injected: bool | None = None
    validator_prompt_has_assistant_memory: bool | None = None
    proof_source_context_required: bool | None = None
    proof_source_context_present: bool | None = None
    generated_appendices_stripped: bool | None = None
    direct_sources_excluded_from_rag: bool | None = None
    mandatory_source_overflow_visible: bool | None = None
    observed_invariants: set[str] = field(default_factory=set)
    exercise_observations: set[str] = field(default_factory=set)
    persisted_events: list[WorkflowEvent] = field(default_factory=list)
    terminal_stop_events: int = 0
    checkpoint: dict[str, object] = field(default_factory=dict)
    events: list[WorkflowEvent] = field(default_factory=list)
    replay: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.mode is not WorkflowMode.NONE and not self.active_owners:
            self.active_owners.add(self.mode)

    def record(self, action: str, **payload: object) -> None:
        if payload:
            details = ", ".join(f"{key}={value!r}" for key, value in sorted(payload.items()))
            self.replay.append(f"{action}({details})")
        else:
            self.replay.append(f"{action}()")

    def emit(self, event_type: str, **payload: object) -> None:
        self.events.append(WorkflowEvent(event_type=event_type, payload=payload))

    def ingest_event(self, event_type: str, payload: dict[str, object]) -> None:
        self.emit(event_type, **payload)
        if event_type == "proof_verified":
            proof_id = str(payload.get("proof_id") or "")
            if proof_id:
                if payload.get("scope") == "manual":
                    self.manual_proofs_active.add(proof_id)
                else:
                    self.autonomous_proofs.add(proof_id)
        elif event_type == "autonomous_proof_provider_paused":
            self.previous_phase = self.phase
            self.phase = WorkflowPhase.PAUSED
            self.provider.exhaust_credit()
            self.checkpoint.update(
                {
                    "paused": True,
                    "phase": self.phase.value,
                    "resume_phase": self.previous_phase.value,
                    "source_type": payload.get("source_type"),
                    "source_id": payload.get("source_id"),
                    "trigger": payload.get("trigger"),
                    "candidate_cursor": payload.get("candidate_cursor", 0),
                    "proof_round": payload.get("proof_round", 1),
                }
            )
        elif event_type == "autonomous_proof_provider_resumed":
            resume_phase = self.checkpoint.get("resume_phase")
            if isinstance(resume_phase, str):
                self.phase = WorkflowPhase(resume_phase)
            self.provider.reset_credit()
            self.checkpoint["paused"] = False
            self.checkpoint["phase"] = self.phase.value
        elif event_type == "research_papers_disabled_brainstorm_complete":
            self.phase = WorkflowPhase.TOPIC_EXPLORATION
            self.checkpoint["phase"] = self.phase.value
