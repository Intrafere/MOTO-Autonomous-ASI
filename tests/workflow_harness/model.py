from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class WorkflowMode(str, Enum):
    NONE = "none"
    AUTONOMOUS = "autonomous"
    MANUAL_COMPILER = "manual_compiler"
    MANUAL_AGGREGATOR = "manual_aggregator"
    LEANOJ = "leanoj"


class WorkflowPhase(str, Enum):
    IDLE = "idle"
    TOPIC_EXPLORATION = "topic_exploration"
    TIER1_AGGREGATION = "tier1_aggregation"
    BRAINSTORM_PROOF = "brainstorm_proof_verification"
    PAPER_TITLE = "paper_title_selection"
    PAPER_WRITING = "paper_writing"
    PAPER_PROOF = "paper_proof_verification"
    TIER3 = "tier3_final_answer"
    MANUAL_PROOF = "manual_proof_check"
    MANUAL_AGGREGATION = "manual_aggregation"
    LEANOJ_BRAINSTORM = "leanoj_brainstorm"
    LEANOJ_FINAL = "leanoj_final"
    PAUSED = "provider_credit_pause"


@dataclass(frozen=True)
class WorkflowEvent:
    event_type: str
    payload: dict[str, object] = field(default_factory=dict)


@dataclass
class FakeProvider:
    credit_exhausted: bool = False
    pause_count: int = 0

    def exhaust_credit(self) -> None:
        self.credit_exhausted = True
        self.pause_count += 1

    def reset_credit(self) -> None:
        self.credit_exhausted = False


@dataclass
class FakeLean:
    enabled: bool = False
    requests: int = 0
    invocations: int = 0
    blocked_invocations: int = 0

    def run(self) -> bool:
        self.requests += 1
        if not self.enabled:
            self.blocked_invocations += 1
            return False
        self.invocations += 1
        return True


@dataclass
class FakeSmt:
    enabled: bool = False
    requests: int = 0
    invocations: int = 0
    blocked_invocations: int = 0

    def run(self) -> bool:
        self.requests += 1
        if not self.enabled:
            self.blocked_invocations += 1
            return False
        self.invocations += 1
        return True


@dataclass
class FakeAssistantMemory:
    enabled: bool = True
    live_pack: tuple[str, ...] = ()
    validator_injections: int = 0
    retrieval_count: int = 0
    blocked_parent_count: int = 0
    stagnant_backoff_count: int = 0
    shutdown_count: int = 0

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        if not enabled:
            self.live_pack = ()

    def refresh_pack(self, *proof_ids: str) -> None:
        if self.enabled:
            self.live_pack = tuple(proof_ids[:7])

    def inject_into_validator(self) -> None:
        self.validator_injections += 1

    def retrieve_non_blocking(self, *proof_ids: str) -> None:
        self.retrieval_count += 1
        self.refresh_pack(*proof_ids)

    def mark_stagnant_backoff(self) -> None:
        self.stagnant_backoff_count += 1


@dataclass
class WorkflowModel:
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
    manual_aggregator_submissions: list[str] = field(default_factory=list)
    manual_aggregator_history: list[str] = field(default_factory=list)
    leanoj_master_proof: str = ""
    leanoj_draft_written: bool = False
    leanoj_draft_preserved_on_resume: bool = False
    leanoj_cleared: bool = False
    leanoj_skip_count: int = 0
    leanoj_force_count: int = 0
    autonomous_paper_checkpoint_count: int = 0
    active_owners: set[WorkflowMode] = field(default_factory=set)
    background_owners: set[WorkflowMode] = field(default_factory=set)
    pending_child_tasks: int = 0
    stale_child_outputs_fenced: bool = True
    parent_generation: int = 0
    child_task_generations: list[int] = field(default_factory=list)
    stale_child_outputs_accepted: int = 0
    truncation_failures: int = 0
    provider_pause_from_truncation: int = 0
    hosted_desktop_route_attempted: bool = False
    hosted_desktop_route_unavailable: bool = True
    hosted_route_attempts: set[str] = field(default_factory=set)
    hosted_route_unavailable: set[str] = field(default_factory=set)
    active_state_cleared: bool = False
    history_preserved: bool = True
    completed_brainstorm_generated_paper: bool = False
    replayed_completed_brainstorm_handoff: bool = False
    pruned_papers: set[str] = field(default_factory=set)
    model_context_papers: set[str] = field(default_factory=set)
    runtime_root_violations: int = 0
    resolved_runtime_paths: list[Path] = field(default_factory=list)
    prompt_user_direct_injected: bool = True
    validator_prompt_has_assistant_memory: bool = False
    proof_source_context_required: bool = True
    proof_source_context_present: bool = True
    generated_appendices_stripped: bool = True
    direct_sources_excluded_from_rag: bool = True
    mandatory_source_overflow_visible: bool = True
    exercise_observations: set[str] = field(default_factory=set)
    persisted_events: list[WorkflowEvent] = field(default_factory=list)
    terminal_stop_events: int = 0
    checkpoint: dict[str, object] = field(default_factory=dict)
    events: list[WorkflowEvent] = field(default_factory=list)
    replay: list[str] = field(default_factory=list)

    def record(self, action: str, **payload: object) -> None:
        if payload:
            details = ", ".join(f"{key}={value!r}" for key, value in sorted(payload.items()))
            self.replay.append(f"{action}({details})")
        else:
            self.replay.append(f"{action}()")

    def emit(self, event_type: str, **payload: object) -> None:
        self.events.append(WorkflowEvent(event_type=event_type, payload=payload))

    def start_autonomous(
        self,
        *,
        proofs: bool = True,
        papers: bool = True,
        lean_enabled: bool = False,
    ) -> None:
        self.record("start_autonomous", proofs=proofs, papers=papers, lean_enabled=lean_enabled)
        if self.mode is not WorkflowMode.NONE:
            self.emit("start_blocked", active_mode=self.mode.value)
            return
        if not proofs and not papers:
            self.emit("start_rejected", reason="no_allowed_outputs")
            return

        self.mode = WorkflowMode.AUTONOMOUS
        self.active_owners.add(self.mode)
        self.phase = WorkflowPhase.TOPIC_EXPLORATION
        self.allow_mathematical_proofs = proofs
        self.allow_research_papers = papers
        self.lean.enabled = lean_enabled
        self.lean.invocations = 0
        self.lean.blocked_invocations = 0
        self.lean.requests = 0
        self.smt.invocations = 0
        self.smt.blocked_invocations = 0
        self.smt.requests = 0
        self.checkpoint = {"phase": self.phase.value, "mode": self.mode.value}
        self.emit("started", mode=self.mode.value, phase=self.phase.value)

    def start_autonomous_with_no_outputs(self) -> None:
        self.record("start_autonomous_with_no_outputs")
        self.start_autonomous(proofs=False, papers=False, lean_enabled=False)

    def start_manual_compiler(self) -> None:
        self.record("start_manual_compiler")
        if self.mode is not WorkflowMode.NONE:
            self.emit("start_blocked", active_mode=self.mode.value)
            return
        self.mode = WorkflowMode.MANUAL_COMPILER
        self.active_owners.add(self.mode)
        self.phase = WorkflowPhase.PAPER_WRITING
        self.lean.invocations = 0
        self.lean.blocked_invocations = 0
        self.lean.requests = 0
        self.smt.invocations = 0
        self.smt.blocked_invocations = 0
        self.smt.requests = 0
        self.emit("started", mode=self.mode.value, phase=self.phase.value)

    def start_manual_aggregator(self) -> None:
        self.record("start_manual_aggregator")
        if self.mode is not WorkflowMode.NONE:
            self.emit("start_blocked", active_mode=self.mode.value)
            return
        self.mode = WorkflowMode.MANUAL_AGGREGATOR
        self.active_owners.add(self.mode)
        self.phase = WorkflowPhase.MANUAL_AGGREGATION
        self.checkpoint = {"mode": self.mode.value, "phase": self.phase.value}
        self.emit("started", mode=self.mode.value, phase=self.phase.value)

    def accept_manual_aggregator_submission(self) -> None:
        self.record("accept_manual_aggregator_submission")
        if self.mode is WorkflowMode.MANUAL_AGGREGATOR:
            submission = f"submission-{len(self.manual_aggregator_submissions) + 1}"
            self.manual_aggregator_submissions.append(submission)
            self.checkpoint["submission_count"] = len(self.manual_aggregator_submissions)

    def start_leanoj(self) -> None:
        self.record("start_leanoj")
        if self.mode is not WorkflowMode.NONE:
            self.emit("start_blocked", active_mode=self.mode.value)
            return
        self.mode = WorkflowMode.LEANOJ
        self.active_owners.add(self.mode)
        self.phase = WorkflowPhase.LEANOJ_BRAINSTORM
        self.checkpoint = {"mode": self.mode.value, "phase": self.phase.value}
        self.emit("started", mode=self.mode.value, phase=self.phase.value)

    def edit_leanoj_master_proof(self) -> None:
        self.record("edit_leanoj_master_proof")
        if self.mode is WorkflowMode.LEANOJ:
            self.leanoj_master_proof = "theorem durable_draft : True := by\n  trivial\n"
            self.leanoj_draft_written = True
            self.checkpoint["master_proof"] = self.leanoj_master_proof

    def skip_leanoj_brainstorm(self) -> None:
        self.record("skip_leanoj_brainstorm")
        if self.mode is WorkflowMode.LEANOJ:
            self.leanoj_skip_count += 1
            self.phase = WorkflowPhase.LEANOJ_FINAL
            self.checkpoint["phase"] = self.phase.value

    def force_leanoj_brainstorm(self) -> None:
        self.record("force_leanoj_brainstorm")
        if self.mode is WorkflowMode.LEANOJ:
            self.leanoj_force_count += 1
            self.phase = WorkflowPhase.LEANOJ_BRAINSTORM
            self.checkpoint["phase"] = self.phase.value

    def enter_autonomous_paper_checkpoint(self) -> None:
        self.record("enter_autonomous_paper_checkpoint")
        if (
            self.mode is WorkflowMode.AUTONOMOUS
            and self.phase is WorkflowPhase.PAPER_TITLE
            and self.allow_research_papers
        ):
            self.phase = WorkflowPhase.PAPER_PROOF
            self.checkpoint.update(
                {
                    "phase": self.phase.value,
                    "source_type": "paper",
                    "source_id": "paper-1",
                    "trigger": "post_paper",
                }
            )

    def complete_autonomous_paper_checkpoint(self) -> None:
        self.record("complete_autonomous_paper_checkpoint")
        if self.mode is WorkflowMode.AUTONOMOUS and self.phase is WorkflowPhase.PAPER_PROOF:
            if self.allow_mathematical_proofs and self.lean.enabled:
                self.lean.run()
                self.autonomous_proofs.add("paper-proof-1")
                self.emit(
                    "proof_verified",
                    proof_id="paper-proof-1",
                    scope="autonomous",
                    phase=self.phase.value,
                )
            self.autonomous_paper_checkpoint_count += 1
            self.phase = WorkflowPhase.TOPIC_EXPLORATION
            self.checkpoint["phase"] = self.phase.value

    def complete_topic_exploration(self) -> None:
        self.record("complete_topic_exploration")
        if self.mode is WorkflowMode.AUTONOMOUS and self.phase is WorkflowPhase.TOPIC_EXPLORATION:
            self.phase = WorkflowPhase.TIER1_AGGREGATION
            self.checkpoint["phase"] = self.phase.value
            self.emit("phase_changed", phase=self.phase.value)

    def start_child_task(self) -> None:
        self.record("start_child_task")
        self.pending_child_tasks += 1
        if self.mode is not WorkflowMode.NONE:
            self.background_owners.add(self.mode)
        self.child_task_generations.append(self.parent_generation)

    def stale_child_output_arrives_after_parent_action(self) -> None:
        self.record("stale_child_output_arrives_after_parent_action")
        child_generation = self.child_task_generations[0] if self.child_task_generations else -1
        fenced = child_generation != self.parent_generation
        self.stale_child_outputs_fenced = fenced
        if not fenced:
            self.stale_child_outputs_accepted += 1

    def complete_brainstorm(self) -> None:
        self.record("complete_brainstorm")
        if self.mode is not WorkflowMode.AUTONOMOUS:
            return
        if self.provider.credit_exhausted and self.allow_mathematical_proofs:
            self.previous_phase = self.phase
            self.phase = WorkflowPhase.PAUSED
            self.checkpoint = {
                "phase": self.phase.value,
                "resume_phase": self.previous_phase.value,
                "mode": self.mode.value,
                "paused": True,
                "source_type": "brainstorm",
                "source_id": "brainstorm-1",
                "trigger": "post_brainstorm",
                "candidate_cursor": 0,
                "proof_round": 1,
            }
            self.emit(
                "provider_paused",
                reason="openrouter_credit_exhaustion",
                scope="autonomous",
                phase=self.phase.value,
            )
            return
        if self.allow_mathematical_proofs and self.lean.enabled:
            self.lean.run()
            self.autonomous_proofs.add("auto-proof-1")
            self.phase = WorkflowPhase.BRAINSTORM_PROOF
            self.checkpoint["proof_round_complete"] = True
            self.emit("proof_verified", proof_id="auto-proof-1", scope="autonomous", phase=self.phase.value)
        if self.allow_research_papers:
            self.phase = WorkflowPhase.PAPER_TITLE
        else:
            self.phase = WorkflowPhase.TOPIC_EXPLORATION
        self.checkpoint["phase"] = self.phase.value
        self.emit("phase_changed", phase=self.phase.value)

    def run_smt_hint_generation(self) -> None:
        self.record("run_smt_hint_generation")
        self.exercise_observations.add("smt_gate_exercised")
        if self.smt.enabled:
            self.smt.run()

    def simulate_provider_output_truncation(self) -> None:
        self.record("simulate_provider_output_truncation")
        self.truncation_failures += 1
        self.emit(
            "proof_attempt_failed",
            reason="provider_output_truncation",
            scope="autonomous",
            phase=self.phase.value,
        )

    def try_hosted_desktop_only_route(self) -> None:
        self.record("try_hosted_desktop_only_route")
        self.hosted_desktop_route_attempted = True
        self.hosted_desktop_route_unavailable = True
        route_categories = {"proof_settings", "update", "pdf", "desktop_oauth"}
        self.hosted_route_attempts.update(route_categories)
        self.hosted_route_unavailable.update(route_categories)
        self.emit("route_unavailable", deployment="hosted", reason="desktop_only")

    def attempt_disabled_lean_checkpoint(self) -> None:
        self.record("attempt_disabled_lean_checkpoint")
        self.exercise_observations.add("lean_gate_exercised")
        if self.lean.enabled:
            self.lean.run()

    def complete_brainstorm_with_generated_paper(self) -> None:
        self.record("complete_brainstorm_with_generated_paper")
        self.completed_brainstorm_generated_paper = True
        self.checkpoint["completed_brainstorm_generated_paper"] = True

    def resume_completed_brainstorm(self) -> None:
        self.record("resume_completed_brainstorm")
        if self.completed_brainstorm_generated_paper:
            self.replayed_completed_brainstorm_handoff = False

    def prune_paper(self, paper_id: str = "paper-1") -> None:
        self.record("prune_paper", paper_id=paper_id)
        self.pruned_papers.add(paper_id)
        self.model_context_papers.discard(paper_id)

    def add_paper_to_model_context(self, paper_id: str = "paper-1") -> None:
        self.record("add_paper_to_model_context", paper_id=paper_id)
        self.model_context_papers.add(paper_id)

    def prepare_prompt_context(self) -> None:
        self.record("prepare_prompt_context")
        self.prompt_user_direct_injected = True
        self.validator_prompt_has_assistant_memory = False
        self.proof_source_context_required = True
        self.proof_source_context_present = True
        self.generated_appendices_stripped = True
        self.exercise_observations.update(
            {
                "user_prompt_direct_injected",
                "validator_assistant_memory_excluded",
                "proof_source_context_present",
                "generated_appendices_stripped",
            }
        )

    def prepare_validator_prompt_with_live_assistant(self) -> None:
        self.record("prepare_validator_prompt_with_live_assistant")
        self.exercise_observations.add("validator_live_assistant_exclusion_exercised")
        self.validator_prompt_has_assistant_memory = False

    def verify_rag_source_exclusion(self) -> None:
        self.record("verify_rag_source_exclusion")
        self.direct_sources_excluded_from_rag = True
        self.exercise_observations.add("direct_sources_excluded_from_rag")

    def reject_mandatory_source_overflow(self) -> None:
        self.record("reject_mandatory_source_overflow")
        self.mandatory_source_overflow_visible = True
        self.exercise_observations.add("mandatory_source_overflow_rejected_visible")

    def emit_frontend_scoped_event(self) -> None:
        self.record("emit_frontend_scoped_event")
        self.emit("proof_progress", scope="autonomous", phase=self.phase.value)

    def emit_context_overflow_contract_events(self) -> None:
        self.record("emit_context_overflow_contract_events")
        fatal_payload = {
            "workflow_mode": "autonomous",
            "role_id": "compiler_writer",
            "configured_model": "configured-model",
            "configured_provider": "openrouter",
            "effective_model": "fallback-model",
            "effective_provider": "lm_studio",
            "fatal": True,
        }
        self.emit("context_overflow_error", **fatal_payload)
        self.persisted_events.append(
            WorkflowEvent(event_type="context_overflow_error", payload=dict(fatal_payload))
        )
        self.emit("auto_research_stopped", reason="context_overflow", **fatal_payload)
        self.terminal_stop_events += 1
        self.emit(
            "proof_context_overflow",
            workflow_mode="autonomous",
            scope="autonomous",
            phase="brainstorm_proof_verification",
            role_id="proof_formalization",
            configured_model="configured-model",
            configured_provider="openrouter",
            effective_model="fallback-model",
            effective_provider="lm_studio",
            fatal=False,
        )

    def emit_registered_proof_verified(self, proof_id: str = "registered-proof-1") -> None:
        self.record("emit_registered_proof_verified", proof_id=proof_id)
        self.autonomous_proofs.add(proof_id)
        self.emit("proof_verified", proof_id=proof_id, scope="autonomous", phase=self.phase.value)

    def force_paper_writing(self) -> None:
        self.record("force_paper_writing")
        if self.mode is WorkflowMode.AUTONOMOUS and self.phase is WorkflowPhase.TIER1_AGGREGATION:
            self.parent_generation += 1
            self.stale_child_outputs_fenced = True
            self.complete_brainstorm()

    def complete_paper(self) -> None:
        self.record("complete_paper")
        if self.mode is WorkflowMode.AUTONOMOUS and self.allow_research_papers:
            if self.allow_mathematical_proofs and self.lean.enabled:
                self.lean.run()
                self.autonomous_proofs.add("paper-proof-1")
                self.phase = WorkflowPhase.PAPER_PROOF
                self.emit("proof_verified", proof_id="paper-proof-1", scope="autonomous", phase=self.phase.value)
            self.phase = WorkflowPhase.TOPIC_EXPLORATION
            self.checkpoint["phase"] = self.phase.value
            self.emit("phase_changed", phase=self.phase.value)

    def run_manual_proof_check(self) -> None:
        self.record("run_manual_proof_check")
        if self.mode is not WorkflowMode.MANUAL_COMPILER:
            return
        self.phase = WorkflowPhase.MANUAL_PROOF
        if self.lean.enabled:
            self.lean.run()
            self.manual_proofs_active.add("manual-proof-1")
            self.emit("proof_verified", proof_id="manual-proof-1", scope="manual", phase=self.phase.value)

    def simulate_credit_exhaustion(self) -> None:
        self.record("simulate_credit_exhaustion")
        self.provider.exhaust_credit()

    def reset_credit(self) -> None:
        self.record("reset_credit")
        self.provider.reset_credit()
        if self.phase is WorkflowPhase.PAUSED:
            phase = self.checkpoint.get("resume_phase", WorkflowPhase.TOPIC_EXPLORATION.value)
            self.phase = WorkflowPhase(phase)
            self.checkpoint["paused"] = False
            self.checkpoint["phase"] = self.phase.value
            self.emit("provider_resumed", scope=self.mode.value, phase=self.phase.value)

    def toggle_session_history_memory(self, enabled: bool) -> None:
        self.record("toggle_session_history_memory", enabled=enabled)
        self.assistant.set_enabled(enabled)

    def refresh_assistant_pack(self, *proof_ids: str) -> None:
        self.record("refresh_assistant_pack", proof_ids=proof_ids)
        self.assistant.refresh_pack(*proof_ids)

    def validator_receive_assistant_pack(self) -> None:
        self.record("validator_receive_assistant_pack")
        self.assistant.inject_into_validator()

    def assistant_retrieve_non_blocking(self) -> None:
        self.record("assistant_retrieve_non_blocking")
        self.assistant.retrieve_non_blocking("prior-proof-1")

    def assistant_stagnant_pack_backoff(self) -> None:
        self.record("assistant_stagnant_pack_backoff")
        self.assistant.mark_stagnant_backoff()

    def stop(self) -> None:
        self.record("stop")
        if self.mode is WorkflowMode.NONE:
            return
        if self.phase is WorkflowPhase.PAUSED:
            self.checkpoint.update({"mode": self.mode.value, "stopped": True})
        else:
            self.checkpoint.update({"mode": self.mode.value, "phase": self.phase.value, "stopped": True})
        stopped_mode = self.mode
        self.active_owners.discard(stopped_mode)
        self.background_owners.discard(stopped_mode)
        self.mode = WorkflowMode.NONE
        self.emit("stopped")

    def resume(self) -> None:
        self.record("resume")
        if self.mode is not WorkflowMode.NONE or not self.checkpoint:
            return
        mode = self.checkpoint.get("mode")
        phase = self.checkpoint.get("phase")
        if mode:
            self.mode = WorkflowMode(mode)
            self.active_owners.add(self.mode)
        if phase:
            self.phase = WorkflowPhase(phase)
        if self.mode is WorkflowMode.LEANOJ and self.leanoj_master_proof:
            self.leanoj_draft_preserved_on_resume = True
        self.checkpoint["stopped"] = False
        self.emit("resumed", mode=self.mode.value, phase=self.phase.value)

    def clear(self) -> None:
        self.record("clear")
        self.manual_proofs_before_clear = set(self.manual_proofs_active)
        self.manual_proofs_archived.update(self.manual_proofs_active)
        self.manual_proofs_active.clear()
        self.manual_aggregator_history.extend(self.manual_aggregator_submissions)
        self.manual_aggregator_submissions.clear()
        if self.mode is WorkflowMode.LEANOJ:
            self.leanoj_cleared = True
        self.leanoj_master_proof = ""
        self.assistant.live_pack = ()
        self.lean.invocations = 0
        self.lean.blocked_invocations = 0
        self.smt.invocations = 0
        self.smt.blocked_invocations = 0
        self.active_state_cleared = True
        self.history_preserved = True
        self.pending_child_tasks = 0
        self.active_owners.clear()
        self.background_owners.clear()
        self.mode = WorkflowMode.NONE
        self.phase = WorkflowPhase.IDLE
        self.checkpoint = {}
        self.emit("cleared")

    def resolve_runtime_path(self, relative_path: str = "state/checkpoint.json") -> None:
        self.record("resolve_runtime_path", relative_path=relative_path)
        resolved = (self.runtime_root / relative_path).resolve()
        self.resolved_runtime_paths.append(resolved)
        try:
            resolved.relative_to(self.runtime_root.resolve())
        except ValueError:
            self.runtime_root_violations += 1

