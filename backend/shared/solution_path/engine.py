"""Run-scoped, crash-safe solution path proposal and review engine."""

from __future__ import annotations

import asyncio
from datetime import timedelta
import json
import os
import random
from pathlib import Path
from typing import Awaitable, Callable, Protocol

from backend.shared.api_client_manager import RetryableProviderError
from backend.shared.model_error_utils import is_non_retryable_model_error
from backend.shared.provider_pause import (
    is_provider_credit_pause_error,
    mark_provider_paused,
    wait_for_provider_resume,
)

from .models import (
    AuditRecord,
    DurableSolutionPathState,
    PlanProposal,
    ProposalStatus,
    RepairReason,
    ReviewDecision,
    SolutionPlan,
    SolutionRoute,
    RouteStep,
    utc_now,
)

MIN_ACCEPTANCES = 5
MAX_AUDIT_RECORDS = 500
MAX_PROPOSALS = 100
MAX_REVIEW_FAILURES = 12
MAX_AUTOMATIC_REVIEW_FAILURES = 3
RETRY_BASE_SECONDS = 1.0
RETRY_MAX_SECONDS = 300.0


class ProposalReviewer(Protocol):
    async def __call__(
        self, proposal: PlanProposal, current_plan: SolutionPlan | None
    ) -> ReviewDecision: ...


Reviewer = Callable[
    [PlanProposal, SolutionPlan | None], Awaitable[ReviewDecision]
]


class SolutionPathEngine:
    """Owns exactly one plan and a serial durable proposal queue for one run."""

    def __init__(
        self,
        root: Path | str,
        run_id: str,
        reviewer: Reviewer,
        *,
        workflow_mode: str = "",
        user_prompt: str = "",
        retry_base_seconds: float = RETRY_BASE_SECONDS,
        retry_max_seconds: float = RETRY_MAX_SECONDS,
    ):
        if not run_id or Path(run_id).name != run_id:
            raise ValueError("run_id must be a non-empty single path component")
        self.run_id = run_id
        self._reviewer = reviewer
        self._directory = Path(root) / run_id
        self._state_path = self._directory / "solution_path_state.json"
        self._generation_path = (
            Path(root) / ".solution_path_generations" / f"{run_id}.json"
        )
        self._lock = asyncio.Lock()
        self._worker: asyncio.Task[None] | None = None
        self._stopped = False
        self._reviewer_generation = 1
        self._retry_base_seconds = max(0.01, retry_base_seconds)
        self._retry_max_seconds = max(self._retry_base_seconds, retry_max_seconds)
        self._state = self._load()
        self._last_persisted_state = self._state.model_copy(deep=True)
        if workflow_mode:
            if self._state.workflow_mode and self._state.workflow_mode != workflow_mode:
                raise ValueError("persisted solution path workflow_mode mismatch")
            self._state.workflow_mode = workflow_mode
        if user_prompt:
            from .models import prompt_fingerprint
            canonical_prompt = user_prompt.strip()
            fingerprint = prompt_fingerprint(canonical_prompt)
            if self._state.prompt_hash and self._state.prompt_hash != fingerprint:
                raise ValueError("persisted solution path user_prompt mismatch")
            self._state.user_prompt = canonical_prompt
            self._state.prompt_hash = fingerprint

    @property
    def active(self) -> bool:
        return self._state.acceptance_count >= MIN_ACCEPTANCES

    @property
    def state(self) -> DurableSolutionPathState:
        return self._state.model_copy(deep=True)

    async def set_reviewer(self, reviewer: Reviewer) -> None:
        """Refresh runtime model configuration while retaining durable run state."""
        async with self._lock:
            self._reviewer = reviewer
            self._reviewer_generation += 1
            self._audit(
                "reviewer_replaced",
                detail=f"reviewer_generation={self._reviewer_generation}",
            )
            await self._persist()

    async def set_acceptance_count(self, count: int) -> None:
        if count < 0:
            raise ValueError("acceptance count cannot be negative")
        activation_payload = None
        async with self._lock:
            previous = self._state.acceptance_count
            effective_count = max(previous, count)
            if effective_count == previous:
                return
            self._state.acceptance_count = effective_count
            self._audit("acceptance_count_updated", detail=str(effective_count))
            if previous < MIN_ACCEPTANCES <= effective_count:
                activation_payload = self._event_payload(
                    message="Progressive solution-path tracking is now active."
                )
            await self._persist()
        if activation_payload is not None:
            await self._broadcast(
                "solution_path_activated",
                activation_payload,
            )
        self._ensure_worker()

    async def propose(
        self,
        route: SolutionRoute,
        *,
        lifecycle_generation: int | None = None,
        rationale: str = "",
        main_route: str = "",
        proposer_role: str = "validator",
        source_task_id: str | None = None,
        source_phase: str | None = None,
        source_decision: str | None = None,
    ) -> PlanProposal:
        async with self._lock:
            if self._stopped:
                raise ValueError("solution-path lifecycle is stopped")
            if (
                lifecycle_generation is not None
                and lifecycle_generation != self._state.lifecycle_generation
            ):
                raise ValueError("stale solution-path lifecycle generation")
            base_revision = self._state.plan.revision if self._state.plan else 0
            proposal = PlanProposal(
                run_id=self.run_id,
                base_revision=base_revision,
                main_route=main_route,
                route=route,
                rationale=rationale,
                proposer_role=proposer_role,
                source_task_id=source_task_id,
                source_phase=source_phase,
                source_decision=source_decision,
            )
            self._state.proposals.append(proposal)
            self._trim_proposals()
            self._audit(
                "proposal_queued",
                proposal.proposal_id,
                base_revision,
                actor=proposer_role,
                source_task_id=source_task_id,
            )
            await self._persist()
            event_payload = self._event_payload(
                proposal=proposal,
                message="A validator-proposed solution-path update was queued for Main Submitter 1 review.",
            )
        await self._broadcast("solution_path_proposal_queued", event_payload)
        self._ensure_worker()
        return proposal.model_copy(deep=True)

    async def stop(self) -> None:
        self._stopped = True
        worker = self._worker
        if worker and not worker.done():
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
        async with self._lock:
            for proposal in self._state.proposals:
                if proposal.status == ProposalStatus.REVIEWING:
                    proposal.status = ProposalStatus.QUEUED
                    proposal.updated_at = utc_now()
            self._audit("engine_stopped")
            await self._persist()

    async def start(self) -> None:
        async with self._lock:
            self._stopped = False
            self._state.lifecycle_generation += 1
            for proposal in self._state.proposals:
                if proposal.status == ProposalStatus.USER_REPAIR_REQUIRED:
                    proposal.repair_generation = self._state.lifecycle_generation
            self._audit("engine_started")
            await self._persist()
        self._ensure_worker()

    async def clear(self) -> None:
        await self.stop()
        async with self._lock:
            generation = self._state.lifecycle_generation + 1
            self._state = DurableSolutionPathState(
                run_id=self.run_id,
                workflow_mode=self._state.workflow_mode,
                user_prompt=self._state.user_prompt,
                prompt_hash=self._state.prompt_hash,
                lifecycle_generation=generation,
            )
            self._audit("engine_cleared")
            await self._persist()
            await asyncio.to_thread(self._write_generation_tombstone, generation)
            await asyncio.to_thread(self._delete_state_files)

    async def resume_proposal(
        self,
        proposal_id: str,
        *,
        lifecycle_generation: int,
    ) -> PlanProposal:
        """Explicitly resume one repair-blocked proposal in the same generation."""
        async with self._lock:
            if lifecycle_generation != self._state.lifecycle_generation:
                raise ValueError("stale solution-path lifecycle generation")
            proposal = next(
                (
                    item
                    for item in self._state.proposals
                    if item.proposal_id == proposal_id
                ),
                None,
            )
            if proposal is None:
                raise ValueError("solution-path proposal not found")
            if proposal.status != ProposalStatus.USER_REPAIR_REQUIRED:
                raise ValueError("solution-path proposal is not repair-blocked")
            if (
                proposal.repair_generation is not None
                and proposal.repair_generation != lifecycle_generation
            ):
                raise ValueError("stale solution-path proposal repair generation")
            proposal.status = ProposalStatus.QUEUED
            proposal.failure_count = 0
            proposal.last_error_type = None
            proposal.repair_reason = None
            proposal.repair_detail = ""
            proposal.repair_generation = None
            proposal.next_retry_at = None
            proposal.feedback = "explicitly resumed after user repair"
            proposal.updated_at = utc_now()
            self._audit(
                "proposal_resumed",
                proposal.proposal_id,
                detail=proposal.feedback,
            )
            await self._persist()
            payload = self._event_payload(
                proposal=proposal,
                message="The repaired solution-path proposal was explicitly resumed.",
            )
            result = proposal.model_copy(deep=True)
        await self._broadcast("solution_path_proposal_resumed", payload)
        self._ensure_worker()
        return result

    async def wait_idle(self) -> None:
        while True:
            worker = self._worker
            if worker is None:
                return
            await asyncio.shield(worker)
            if worker is self._worker:
                return

    def _ensure_worker(self) -> None:
        if self._stopped or not self.active:
            return
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._review_loop())

    async def _review_loop(self) -> None:
        while not self._stopped and self.active:
            async with self._lock:
                proposal = next(
                    (
                        item
                        for item in self._state.proposals
                        if item.status in {
                            ProposalStatus.QUEUED,
                            ProposalStatus.FOLLOWUP,
                            ProposalStatus.REVIEWING,
                        }
                    ),
                    None,
                )
                if proposal is None:
                    return
                if proposal.next_retry_at is not None:
                    delay = (proposal.next_retry_at - utc_now()).total_seconds()
                    if delay > 0:
                        retry_generation = self._state.lifecycle_generation
                    else:
                        delay = 0
                else:
                    delay = 0
                if delay > 0:
                    proposal_snapshot = None
                else:
                    retry_generation = self._state.lifecycle_generation
                    reviewer_generation = self._reviewer_generation
                    reviewer = self._reviewer
                    proposal.status = ProposalStatus.REVIEWING
                    proposal.review_count += 1
                    proposal.updated_at = utc_now()
                    plan_snapshot = (
                        self._state.plan.model_copy(deep=True)
                        if self._state.plan
                        else None
                    )
                    proposal_snapshot = proposal.model_copy(deep=True)
                    self._audit(
                        "review_started",
                        proposal.proposal_id,
                        detail=f"reviewer_generation={reviewer_generation}",
                    )
                    await self._persist()
            if delay > 0:
                await asyncio.sleep(min(delay, 1.0))
                continue

            await self._broadcast(
                "solution_path_proposal_reviewing",
                self._event_payload(
                    proposal=proposal_snapshot,
                    message="Main Submitter 1 is reviewing a proposed solution-path update.",
                ),
            )
            try:
                decision = await reviewer(proposal_snapshot, plan_snapshot)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if is_provider_credit_pause_error(exc):
                    mark_provider_paused()
                    async with self._lock:
                        proposal.status = ProposalStatus.QUEUED
                        proposal.last_error_type = "provider_credit_pause"
                        proposal.feedback = "review paused until provider credits are reset"
                        proposal.updated_at = utc_now()
                        self._audit(
                            "review_provider_paused",
                            proposal.proposal_id,
                            detail=proposal.feedback,
                        )
                        await self._persist()
                    await wait_for_provider_resume(lambda: self._stopped)
                    continue
                if self._is_context_overflow(exc):
                    async with self._lock:
                        if retry_generation != self._state.lifecycle_generation:
                            continue
                        self._mark_repair_required(
                            proposal,
                            RepairReason.CONTEXT_OVERFLOW,
                            exc,
                        )
                        self._audit(
                            "review_user_repair_required",
                            proposal.proposal_id,
                            detail=proposal.feedback,
                        )
                        await self._persist()
                        repair_payload = self._event_payload(
                            proposal=proposal,
                            message=(
                                "Solution-path review needs a larger context window or "
                                "a shorter user prompt before it can continue."
                            ),
                            detail=proposal.feedback,
                        )
                    await self._broadcast(
                        "solution_path_proposal_user_repair_required", repair_payload
                    )
                    continue
                repair_reason = self._hard_failure_reason(exc)
                if repair_reason is not None:
                    async with self._lock:
                        if retry_generation != self._state.lifecycle_generation:
                            continue
                        self._mark_repair_required(proposal, repair_reason, exc)
                        self._audit(
                            "review_user_repair_required",
                            proposal.proposal_id,
                            detail=proposal.feedback,
                        )
                        await self._persist()
                        repair_payload = self._event_payload(
                            proposal=proposal,
                            message="Solution-path review needs user configuration repair before it can continue.",
                            detail=proposal.repair_detail,
                        )
                    await self._broadcast(
                        "solution_path_proposal_user_repair_required",
                        repair_payload,
                    )
                    continue
                async with self._lock:
                    if retry_generation != self._state.lifecycle_generation:
                        continue
                    proposal.status = ProposalStatus.QUEUED
                    proposal.failure_count += 1
                    proposal.last_error_type = type(exc).__name__
                    proposal.feedback = self._safe_error(exc)
                    proposal.updated_at = utc_now()
                    if proposal.failure_count >= MAX_AUTOMATIC_REVIEW_FAILURES:
                        self._mark_repair_required(
                            proposal,
                            RepairReason.UNKNOWN_REVIEWER_FAILURE,
                            exc,
                        )
                        self._audit(
                            "review_user_repair_required",
                            proposal.proposal_id,
                            detail=proposal.feedback,
                        )
                        await self._persist()
                        repair_payload = self._event_payload(
                            proposal=proposal,
                            message="Solution-path review repeatedly failed and now requires explicit user repair.",
                            detail=proposal.repair_detail,
                        )
                        retry_payload = None
                    else:
                        retry_delay = self._retry_delay(proposal.failure_count)
                        proposal.next_retry_at = utc_now() + timedelta(
                            seconds=retry_delay
                        )
                        event = (
                            "review_retry_scheduled"
                            if isinstance(exc, RetryableProviderError)
                            else "review_failed_retry_scheduled"
                        )
                        self._audit(event, proposal.proposal_id, detail=proposal.feedback)
                        await self._persist()
                        retry_payload = self._event_payload(
                            proposal=proposal,
                            message="Solution-path review failed and the update remains queued for retry.",
                            detail=proposal.feedback,
                        )
                if retry_payload is None:
                    await self._broadcast(
                        "solution_path_proposal_user_repair_required",
                        repair_payload,
                    )
                else:
                    await self._broadcast("solution_path_proposal_retry_queued", retry_payload)
                continue

            async with self._lock:
                if (
                    retry_generation != self._state.lifecycle_generation
                    or self._stopped
                ):
                    continue
                current_revision = self._state.plan.revision if self._state.plan else 0
                if proposal.base_revision != current_revision:
                    proposal.base_revision = current_revision
                    proposal.status = ProposalStatus.QUEUED
                    proposal.feedback = "stale base revision; queued for re-review"
                    proposal.updated_at = utc_now()
                    self._audit("proposal_stale_requeued", proposal.proposal_id, current_revision)
                    await self._persist()
                    retry_payload = self._event_payload(
                        proposal=proposal,
                        message="A stale solution-path update was queued for review again.",
                        detail=proposal.feedback,
                    )
                    await self._broadcast("solution_path_proposal_retry_queued", retry_payload)
                    continue
                try:
                    self._apply_decision(proposal, decision)
                except (TypeError, ValueError) as exc:
                    proposal.failure_count += 1
                    proposal.last_error_type = type(exc).__name__
                    proposal.feedback = self._safe_error(exc)
                    proposal.updated_at = utc_now()
                    if proposal.failure_count >= MAX_AUTOMATIC_REVIEW_FAILURES:
                        self._mark_repair_required(
                            proposal,
                            RepairReason.INVALID_REVIEW,
                            exc,
                        )
                        self._audit(
                            "review_user_repair_required",
                            proposal.proposal_id,
                            detail=proposal.feedback,
                        )
                    else:
                        proposal.status = ProposalStatus.QUEUED
                        proposal.next_retry_at = utc_now() + timedelta(
                            seconds=self._retry_delay(proposal.failure_count)
                        )
                        self._audit(
                            "review_edit_validation_retry_scheduled",
                            proposal.proposal_id,
                            detail=proposal.feedback,
                        )
                await self._persist()
                event = (
                    "solution_path_updated"
                    if proposal.status == ProposalStatus.APPROVED
                    else "solution_path_proposal_rejected"
                    if proposal.status == ProposalStatus.REJECTED
                    else "solution_path_proposal_retry_queued"
                )
                event_payload = self._event_payload(
                    proposal=proposal,
                    decision=decision.decision,
                    detail=decision.reasoning,
                    message=(
                        "Main Submitter 1 approved an update to the current solution path."
                        if proposal.status == ProposalStatus.APPROVED
                        else "Main Submitter 1 rejected a proposed solution-path update."
                        if proposal.status == ProposalStatus.REJECTED
                        else "Main Submitter 1 requested another solution-path revision; it remains queued."
                    ),
                )
            await self._broadcast(event, event_payload)

    def _apply_decision(self, proposal: PlanProposal, decision: ReviewDecision) -> None:
        working = proposal.model_copy(deep=True)
        working.feedback = decision.reasoning
        working.last_error_type = None
        working.repair_reason = None
        working.repair_detail = ""
        working.repair_generation = None
        working.next_retry_at = None
        working.updated_at = utc_now()
        if decision.decision == "reject":
            working.status = ProposalStatus.REJECTED
            self._commit_proposal(proposal, working)
            self._audit("proposal_rejected", proposal.proposal_id, detail=decision.reasoning)
            return
        if decision.decision == "followup":
            if decision.followup_route is not None:
                working.route = decision.followup_route.model_copy(deep=True)
            if decision.main_route is not None:
                working.main_route = decision.main_route
            self._apply_route_edits(working.route, decision.edits)
            working = PlanProposal.model_validate(working.model_dump())
            working.status = ProposalStatus.FOLLOWUP
            self._commit_proposal(proposal, working)
            self._audit("proposal_followup", proposal.proposal_id, detail=decision.reasoning)
            return
        if decision.followup_route is not None:
            working.route = decision.followup_route.model_copy(deep=True)
        if decision.main_route is not None:
            working.main_route = decision.main_route
        self._apply_route_edits(working.route, decision.edits)
        working = PlanProposal.model_validate(working.model_dump())
        if decision.more_edits:
            working.status = ProposalStatus.FOLLOWUP
            self._commit_proposal(proposal, working)
            self._audit("proposal_more_edits_requested", proposal.proposal_id)
            return
        revision = (self._state.plan.revision + 1) if self._state.plan else 1
        created_at = self._state.plan.created_at if self._state.plan else utc_now()
        self._state.plan = SolutionPlan(
            run_id=self.run_id,
            revision=revision,
            main_route=working.main_route,
            route=working.route,
            created_at=created_at,
        )
        working.status = ProposalStatus.APPROVED
        self._commit_proposal(proposal, working)
        self._audit("proposal_approved", proposal.proposal_id, revision, decision.reasoning)

    @staticmethod
    def _commit_proposal(target: PlanProposal, source: PlanProposal) -> None:
        for field_name in type(target).model_fields:
            setattr(target, field_name, getattr(source, field_name))

    def _apply_route_edits(self, route: SolutionRoute, edits: list) -> None:
        for edit in edits:
            index = next(
                (i for i, step in enumerate(route.steps) if step.step_id == edit.step_id),
                None,
            )
            if edit.operation == "check":
                if index is None:
                    raise ValueError(f"checked route step not found: {edit.step_id}")
                if (
                    edit.expected_title is not None
                    and route.steps[index].title != edit.expected_title
                ):
                    raise ValueError(f"checked route step title changed: {edit.step_id}")
            elif edit.operation == "delete":
                if index is None:
                    raise ValueError(f"route step to delete not found: {edit.step_id}")
                route.steps.pop(index)
            elif edit.operation == "update":
                if index is None or edit.step is None:
                    raise ValueError(f"route step to update not found: {edit.step_id}")
                replacement = edit.step.model_copy(deep=True)
                replacement.step_id = edit.step_id
                route.steps[index] = replacement
            elif edit.operation == "add" and edit.step is not None:
                new_step: RouteStep = edit.step.model_copy(deep=True)
                if any(item.step_id == new_step.step_id for item in route.steps):
                    raise ValueError(f"duplicate route step id: {new_step.step_id}")
                if edit.after_step_id is None:
                    route.steps.append(new_step)
                else:
                    after = next(
                        (
                            i
                            for i, item in enumerate(route.steps)
                            if item.step_id == edit.after_step_id
                        ),
                        None,
                    )
                    if after is None:
                        raise ValueError(
                            f"route insertion anchor not found: {edit.after_step_id}"
                        )
                    route.steps.insert(after + 1, new_step)

    def _event_payload(
        self,
        *,
        message: str,
        proposal: PlanProposal | None = None,
        decision: str = "",
        detail: str = "",
    ) -> dict:
        revision = self._state.plan.revision if self._state.plan else 0
        queued_statuses = {ProposalStatus.QUEUED, ProposalStatus.FOLLOWUP}
        return {
            "workflow_mode": self._state.workflow_mode or "unknown",
            "run_id": self.run_id,
            "acceptance_count": self._state.acceptance_count,
            "prompt_hash": self._state.prompt_hash or None,
            "lifecycle_generation": self._state.lifecycle_generation,
            "revision": revision,
            "proposal_id": proposal.proposal_id if proposal else None,
            "proposal_status": proposal.status.value if proposal else None,
            "base_revision": proposal.base_revision if proposal else revision,
            "decision": decision or None,
            "source_task_id": proposal.source_task_id if proposal else None,
            "source_phase": proposal.source_phase if proposal else None,
            "source_decision": proposal.source_decision if proposal else None,
            "queued_proposals": sum(
                item.status in queued_statuses for item in self._state.proposals
            ),
            "reviewing_proposals": sum(
                item.status == ProposalStatus.REVIEWING
                for item in self._state.proposals
            ),
            "repair_required_proposals": sum(
                item.status == ProposalStatus.USER_REPAIR_REQUIRED
                for item in self._state.proposals
            ),
            "repair_reason": (
                proposal.repair_reason.value
                if proposal and proposal.repair_reason
                else None
            ),
            "message": message,
            "detail": detail,
        }

    def _audit(
        self,
        event: str,
        proposal_id: str | None = None,
        plan_revision: int | None = None,
        detail: str = "",
        actor: str = "solution_path_engine",
        source_task_id: str | None = None,
    ) -> None:
        self._state.audit.append(
            AuditRecord(
                sequence=self._state.next_audit_sequence,
                event=event,
                proposal_id=proposal_id,
                plan_revision=plan_revision,
                detail=detail,
                actor=actor,
                source_task_id=source_task_id,
            )
        )
        self._state.next_audit_sequence += 1
        if len(self._state.audit) > MAX_AUDIT_RECORDS:
            del self._state.audit[:-MAX_AUDIT_RECORDS]

    def _load(self) -> DurableSolutionPathState:
        if not self._state_path.exists():
            generation = 1
            try:
                payload = json.loads(
                    self._generation_path.read_text(encoding="utf-8")
                )
                generation = max(generation, int(payload.get("lifecycle_generation", 1)))
            except (FileNotFoundError, OSError, ValueError, TypeError):
                pass
            return DurableSolutionPathState(
                run_id=self.run_id,
                lifecycle_generation=generation,
            )
        state = DurableSolutionPathState.model_validate_json(
            self._state_path.read_text(encoding="utf-8")
        )
        if state.run_id != self.run_id:
            raise ValueError("persisted solution path run_id mismatch")
        for proposal in state.proposals:
            if proposal.status == ProposalStatus.REVIEWING:
                proposal.status = ProposalStatus.QUEUED
        if state.audit:
            state.next_audit_sequence = max(
                state.next_audit_sequence,
                max(record.sequence for record in state.audit) + 1,
            )
        return state

    async def _persist(self) -> None:
        payload = self._state.model_dump(mode="json")
        try:
            await asyncio.to_thread(self._write_payload, payload)
        except BaseException:
            # Every caller mutates under _lock. Restoring the last successfully
            # published snapshot makes those mutations transactional on I/O
            # failure without exposing a half-persisted in-memory state.
            self._state = self._last_persisted_state.model_copy(deep=True)
            raise
        self._last_persisted_state = self._state.model_copy(deep=True)

    def _write_payload(self, payload: dict) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        temporary = self._state_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self._state_path)

    def _write_generation_tombstone(self, generation: int) -> None:
        self._generation_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._generation_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                {
                    "run_id": self.run_id,
                    "lifecycle_generation": generation,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self._generation_path)

    def _delete_state_files(self) -> None:
        try:
            self._state_path.unlink()
        except FileNotFoundError:
            pass
        try:
            self._directory.rmdir()
        except (FileNotFoundError, OSError):
            pass

    def _trim_proposals(self) -> None:
        if len(self._state.proposals) <= MAX_PROPOSALS:
            return
        active = [
            proposal
            for proposal in self._state.proposals
            if proposal.status
            in {ProposalStatus.QUEUED, ProposalStatus.REVIEWING, ProposalStatus.FOLLOWUP}
        ]
        terminal = [
            proposal
            for proposal in self._state.proposals
            if proposal.status
            in {
                ProposalStatus.APPROVED,
                ProposalStatus.REJECTED,
                ProposalStatus.USER_REPAIR_REQUIRED,
            }
        ]
        keep_terminal = max(0, MAX_PROPOSALS - len(active))
        retained_terminal = terminal[-keep_terminal:] if keep_terminal else []
        self._state.proposals = retained_terminal + active

    def _retry_delay(self, failure_count: int) -> float:
        exponent = min(max(0, failure_count - 1), MAX_REVIEW_FAILURES)
        base = min(self._retry_max_seconds, self._retry_base_seconds * (2**exponent))
        return min(self._retry_max_seconds, base * random.uniform(0.8, 1.2))

    def _mark_repair_required(
        self,
        proposal: PlanProposal,
        reason: RepairReason,
        exc: Exception,
    ) -> None:
        detail = self._safe_error(exc)
        proposal.status = ProposalStatus.USER_REPAIR_REQUIRED
        proposal.last_error_type = type(exc).__name__
        proposal.repair_reason = reason
        proposal.repair_detail = detail
        proposal.repair_generation = self._state.lifecycle_generation
        proposal.feedback = detail
        proposal.next_retry_at = None
        proposal.updated_at = utc_now()

    @staticmethod
    def _hard_failure_reason(exc: Exception) -> RepairReason | None:
        if not is_non_retryable_model_error(exc):
            return None
        text = str(exc).lower()
        if "api key" in text or "missing key" in text:
            return RepairReason.MISSING_API_KEY
        if "privacy" in text:
            return RepairReason.PRIVACY_POLICY
        return RepairReason.PROVIDER_CONFIGURATION

    @staticmethod
    def _safe_error(exc: Exception) -> str:
        text = " ".join(str(exc).split())
        return f"reviewer error: {type(exc).__name__}: {text[:500]}"

    @staticmethod
    def _is_context_overflow(exc: Exception) -> bool:
        try:
            from backend.shared.provider_errors import ProviderContextLengthError

            if isinstance(exc, ProviderContextLengthError):
                return True
        except ImportError:
            pass
        text = str(exc).lower()
        return (
            "solution-path reviewer mandatory context exceeds" in text
            or "context_length_exceeded" in text
            or "context window" in text and "exceed" in text
        )

    async def _broadcast(self, event: str, payload: dict) -> None:
        try:
            from backend.api.routes.websocket import broadcast_event
            await broadcast_event(event, payload)
        except Exception:
            # Plan review remains secondary and must never block the parent workflow.
            pass
