"""Run-scoped, non-blocking orchestration for proof live-context pruning."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Awaitable, Callable, Optional

from backend.autonomous.agents.proof_pruning_agent import (
    ProofPruningContextError,
    ProofPruningContractError,
    ProofPruningReviewService,
    ProofPruningStaleSnapshotError,
)
from backend.autonomous.memory.proof_database import is_prompt_injection_novel_tier
from backend.shared.api_client_manager import RetryableProviderError, api_client_manager
from backend.shared.model_error_utils import (
    is_non_retryable_model_error,
    is_transient_model_call_error,
)
from backend.shared.models import (
    ProofPruneContextPressure,
    ProofPruneReviewResult,
    ProofPruningState,
    ProofRecord,
    ProofRuntimeConfigSnapshot,
)
from backend.shared.provider_pause import (
    get_provider_reset_generation,
    is_provider_credit_pause_error,
    wait_for_provider_reset,
)

logger = logging.getLogger(__name__)

BroadcastFn = Optional[Callable[[str, dict[str, Any]], Awaitable[None]]]
PersistFn = Optional[Callable[[dict[str, Any]], Awaitable[Optional[bool]]]]
LoadFn = Optional[Callable[[], Awaitable[Optional[dict[str, Any]]]]]
InvalidateFn = Optional[Callable[[str], Awaitable[None] | None]]
ShouldStopFn = Optional[Callable[[], bool]]

PRUNING_STATE_SCHEMA_VERSION = 1
PRUNING_POLICY_VERSION = "proof-pruning-orchestration-v1"
CADENCE_THRESHOLD = 3


class ProofPruningCoordinator:
    """Own exactly one asynchronous pruning review for one proof run."""

    def __init__(
        self,
        *,
        proof_database: Any,
        runtime_snapshot: ProofRuntimeConfigSnapshot,
        proof_run_id: str,
        run_mode: str,
        run_id: str,
        lifecycle_generation: int,
        scope: str,
        source_type: str,
        source_id: str,
        canonical_user_prompt: str,
        proof_store_id: str,
        session_id: str = "",
        round_index: int = 1,
        broadcast_fn: BroadcastFn = None,
        persist_fn: PersistFn = None,
        load_fn: LoadFn = None,
        invalidate_fn: InvalidateFn = None,
        should_stop: ShouldStopFn = None,
        review_service: Optional[ProofPruningReviewService] = None,
    ) -> None:
        self.proof_database = proof_database
        self.runtime_snapshot = runtime_snapshot
        self.proof_run_id = str(proof_run_id)
        self.run_mode = str(run_mode)
        self.run_id = str(run_id)
        self.lifecycle_generation = int(lifecycle_generation)
        self.scope = str(scope)
        self.source_type = str(source_type)
        self.source_id = str(source_id)
        self.canonical_user_prompt = str(canonical_user_prompt)
        self.proof_store_id = str(proof_store_id)
        self.session_id = str(session_id or "")
        self.round_index = max(1, int(round_index or 1))
        self.broadcast_fn = broadcast_fn
        self.persist_fn = persist_fn
        self.load_fn = load_fn
        self.invalidate_fn = invalidate_fn
        self.should_stop = should_stop
        self.review_service = review_service or ProofPruningReviewService(
            runtime_snapshot=runtime_snapshot,
            scope=scope,
            proof_run_id=proof_run_id,
        )

        self.state = ProofPruningState(
            schema_version=PRUNING_STATE_SCHEMA_VERSION,
            policy_version=PRUNING_POLICY_VERSION,
            proof_run_id=self.proof_run_id,
            run_id=self.run_id,
            lifecycle_generation=self.lifecycle_generation,
            scope=self.scope,
            source_type=self.source_type,
            source_id=self.source_id,
            proof_store_id=self.proof_store_id,
            round_index=self.round_index,
        )
        self._state_lock = asyncio.Lock()
        self._commit_lock = asyncio.Lock()
        self._active_task: Optional[asyncio.Task] = None
        self._notification_tasks: set[asyncio.Task] = set()
        self._accepting_triggers = True
        self._draining = False

    @property
    def active_task(self) -> Optional[asyncio.Task]:
        return self._active_task

    @staticmethod
    def route_config_fingerprint(runtime_snapshot: ProofRuntimeConfigSnapshot) -> str:
        payload = {
            "paper": runtime_snapshot.paper.model_dump(mode="json"),
            "validator": runtime_snapshot.validator.model_dump(mode="json"),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _owns_lifecycle(self) -> bool:
        return (
            self._accepting_triggers
            and not self._draining
            and not (self.should_stop and self.should_stop())
            and self.state.proof_run_id == self.proof_run_id
            and self.state.run_id == self.run_id
            and self.state.lifecycle_generation == self.lifecycle_generation
        )

    async def _persist(self) -> None:
        self.state.updated_at = datetime.now()
        if self.persist_fn:
            persisted = await self.persist_fn(self.state.model_dump(mode="json"))
            if persisted is False:
                self._accepting_triggers = False
                raise ProofPruningStaleSnapshotError(
                    "Proof-pruning persistence ownership changed."
                )

    def _event_payload(self, **extra: Any) -> dict[str, Any]:
        payload = {
            "proof_run_id": self.proof_run_id,
            "run_mode": self.run_mode,
            "run_id": self.run_id,
            "lifecycle_generation": self.lifecycle_generation,
            "proposal_id": self.state.active_proposal_id,
            "snapshot_revision": self.state.snapshot_revision,
            "trigger_reasons": list(self.state.queued_trigger_reasons),
            "scope": self.scope,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "round_index": self.round_index,
        }
        payload.update(extra)
        return payload

    async def _broadcast(self, event: str, **extra: Any) -> None:
        if self.broadcast_fn:
            await self.broadcast_fn(event, self._event_payload(**extra))

    async def restore(self) -> None:
        """Restore compatible counters/pressure, never an unverifiable snapshot."""
        if not self.load_fn:
            return
        raw = await self.load_fn()
        if not isinstance(raw, dict):
            return
        try:
            restored = ProofPruningState.model_validate(raw)
        except Exception:
            logger.warning("Ignoring incompatible proof-pruning state for %s", self.proof_run_id)
            return
        if (
            restored.schema_version != PRUNING_STATE_SCHEMA_VERSION
            or restored.policy_version != PRUNING_POLICY_VERSION
            or restored.proof_run_id != self.proof_run_id
            or restored.run_id != self.run_id
            or restored.proof_store_id != self.proof_store_id
            or restored.scope != self.scope
            or restored.source_type != self.source_type
            or restored.source_id != self.source_id
            or restored.lifecycle_generation > self.lifecycle_generation
        ):
            return
        # A clean Stop/Start advances the parent lifecycle. The previous owner
        # has been drained, so compatible queued state may be rebound while all
        # transient proposal/snapshot authority is discarded below.
        restored.lifecycle_generation = self.lifecycle_generation
        restored.round_index = max(restored.round_index, self.round_index)
        restored.active_proposal_id = ""
        restored.active_proposal_generation = 0
        restored.snapshot_id = ""
        restored.snapshot_revision = None
        restored.target_proof_id = ""
        if restored.status in {"proposing", "validating"}:
            for reason in restored.active_trigger_reasons:
                if reason not in restored.queued_trigger_reasons:
                    restored.queued_trigger_reasons.append(reason)
            restored.status = "queued"
            restored.follow_up_required = True
        restored.active_trigger_reasons = []
        self.state = restored
        await self._persist()
        if restored.status in {"queued", "provider_paused"} and restored.queued_trigger_reasons:
            self._ensure_task()

    async def notify_proof_registered(
        self,
        stored_record: ProofRecord,
        stage_metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Schedule registration accounting without delaying serialized Phase B."""
        self._schedule_notification(
            self.on_proof_registered(stored_record, stage_metadata)
        )

    async def notify_context_pressure(
        self,
        pressure: ProofPruneContextPressure,
        *,
        urgent: bool = False,
        proof_set_revision: Optional[int] = None,
    ) -> None:
        """Schedule pressure accounting without delaying candidate handling."""
        self._schedule_notification(
            self.on_context_pressure(
                pressure,
                urgent=urgent,
                proof_set_revision=proof_set_revision,
            )
        )

    def _schedule_notification(self, awaitable: Awaitable[None]) -> None:
        if not self._owns_lifecycle():
            if hasattr(awaitable, "close"):
                awaitable.close()
            return
        task = asyncio.create_task(awaitable)
        self._notification_tasks.add(task)
        task.add_done_callback(self._observe_notification)

    def _observe_notification(self, task: asyncio.Task) -> None:
        self._notification_tasks.discard(task)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception:
            # Registration is already durable. Pruning notification failure is
            # local observability/maintenance failure, never proof failure.
            logger.exception(
                "Proof-pruning notification failed for %s",
                self.proof_run_id,
            )

    async def on_proof_registered(
        self,
        stored_record: ProofRecord,
        stage_metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Count one eligible committed occurrence and return without model work."""
        if not self._owns_lifecycle():
            return
        proof_id = str(stored_record.proof_id or "")
        if (
            not proof_id
            or proof_id in self.state.counted_proof_ids
            or stored_record.live_context_status != "active"
            or not stored_record.novel
            or not is_prompt_injection_novel_tier(stored_record.novelty_tier)
        ):
            return
        async with self._state_lock:
            if proof_id in self.state.counted_proof_ids or not self._owns_lifecycle():
                return
            self.state.counted_proof_ids.append(proof_id)
            self.state.accepted_prompt_novel_total += 1
            revision = (stage_metadata or {}).get("proof_set_revision")
            if revision is not None:
                self.state.proof_set_revision = max(
                    self.state.proof_set_revision, int(revision)
                )
            if (
                self.state.accepted_prompt_novel_total
                - self.state.last_scheduled_acceptance_baseline
                >= CADENCE_THRESHOLD
            ):
                await self._queue_trigger_locked(
                    "three_novel_proofs",
                    requested_revision=self.state.proof_set_revision,
                    consume_cadence=True,
                )
            else:
                await self._persist()

    async def on_context_pressure(
        self,
        pressure: ProofPruneContextPressure,
        *,
        urgent: bool = False,
        proof_set_revision: Optional[int] = None,
    ) -> None:
        """Coalesce proof-memory pressure for the current revision/route."""
        if not self._owns_lifecycle() or pressure.active_proof_context_tokens <= 0:
            return
        revision = int(
            self.state.proof_set_revision
            if proof_set_revision is None
            else proof_set_revision
        )
        fingerprint = pressure.pressure_fingerprint()
        async with self._state_lock:
            if (
                not urgent
                and fingerprint == self.state.last_reviewed_pressure_fingerprint
                and revision == self.state.last_reviewed_pressure_revision
            ):
                return
            self.state.context_pressure = pressure
            reason = (
                "proof_context_overflow_urgent"
                if urgent
                else "proof_stage_context_maximum"
            )
            await self._queue_trigger_locked(reason, requested_revision=revision)

    async def _queue_trigger_locked(
        self,
        reason: str,
        *,
        requested_revision: int,
        consume_cadence: bool = False,
    ) -> None:
        if reason not in self.state.queued_trigger_reasons:
            self.state.queued_trigger_reasons.append(reason)
        self.state.requested_snapshot_revision = max(
            self.state.requested_snapshot_revision,
            int(requested_revision),
        )
        if consume_cadence:
            # Move by one threshold. A jump retains pressure for a later follow-up.
            self.state.last_scheduled_acceptance_baseline += CADENCE_THRESHOLD
        if self._active_task and not self._active_task.done():
            self.state.follow_up_required = True
        else:
            self.state.status = "queued"
        await self._persist()
        await self._broadcast("proof_prune_review_queued")
        self._ensure_task()

    def _ensure_task(self) -> None:
        if not self._owns_lifecycle():
            return
        if self._active_task and not self._active_task.done():
            return
        task = asyncio.create_task(self._review_driver())
        self._active_task = task
        task.add_done_callback(self._observe_task)

    def _observe_task(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        if exc:
            logger.error(
                "Proof pruning task failed for %s: %s",
                self.proof_run_id,
                exc,
                exc_info=exc,
            )

    async def _review_driver(self) -> None:
        while self._owns_lifecycle():
            async with self._state_lock:
                reasons = list(self.state.queued_trigger_reasons)
                if not reasons:
                    self.state.status = "idle"
                    await self._persist()
                    return
                self.state.queued_trigger_reasons = []
                self.state.active_trigger_reasons = reasons
                self.state.round_index = max(self.state.round_index, self.round_index)
                self.state.follow_up_required = False
                self.state.active_proposal_id = f"prune-{uuid.uuid4().hex}"
                self.state.active_proposal_generation += 1
                proposal_generation = self.state.active_proposal_generation
                self.state.status = "proposing"
                await self._persist()

            await self._broadcast(
                "proof_prune_review_started",
                trigger_reasons=reasons,
            )
            try:
                snapshot = await self.proof_database.capture_pruning_snapshot(
                    proof_store_id=self.proof_store_id,
                    owning_run_id=self.run_id,
                    proof_run_id=self.proof_run_id,
                    proof_run_lifecycle_generation=self.lifecycle_generation,
                    owning_lifecycle_generation=self.lifecycle_generation,
                    scope=self.scope,
                    source_type=self.source_type,
                    source_id=self.source_id,
                    session_id=self.session_id,
                    canonical_user_prompt=self.canonical_user_prompt,
                    trigger_reasons=reasons,
                    accepted_prompt_novel_total=self.state.accepted_prompt_novel_total,
                    context_pressure=self.state.context_pressure,
                )
                async with self._state_lock:
                    self.state.snapshot_id = snapshot.snapshot_id
                    self.state.snapshot_revision = snapshot.proof_set_revision
                    self.state.proof_set_revision = max(
                        self.state.proof_set_revision, snapshot.proof_set_revision
                    )
                    await self._persist()

                async def review_event(event: str, payload: dict[str, Any]) -> None:
                    if event == "proposed":
                        async with self._state_lock:
                            self.state.status = "validating"
                            self.state.sanitized_proposal = payload
                            self.state.target_proof_id = str(payload.get("proof_id") or "")
                            await self._persist()
                        await self._broadcast("proof_prune_proposed", proposal=payload)
                        await self._broadcast("proof_prune_validation_started")

                review = await self.review_service.review(
                    snapshot,
                    event_callback=review_event,
                )
                await self._handle_review_result(
                    review,
                    snapshot=snapshot,
                    proposal_generation=proposal_generation,
                    reasons=reasons,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if await self._handle_failure(exc, reasons):
                    return

            async with self._state_lock:
                self.state.active_trigger_reasons = []
                threshold_pending = (
                    self.state.accepted_prompt_novel_total
                    - self.state.last_scheduled_acceptance_baseline
                    >= CADENCE_THRESHOLD
                )
                if threshold_pending:
                    if "three_novel_proofs" not in self.state.queued_trigger_reasons:
                        self.state.queued_trigger_reasons.append("three_novel_proofs")
                    self.state.last_scheduled_acceptance_baseline += CADENCE_THRESHOLD
                if not self.state.queued_trigger_reasons:
                    self.state.status = "idle"
                    self.state.active_proposal_id = ""
                    self.state.snapshot_id = ""
                    self.state.snapshot_revision = None
                    await self._persist()
                    return
                self.state.status = "queued"
                await self._persist()

    async def _handle_review_result(
        self,
        review: ProofPruneReviewResult,
        *,
        snapshot: Any,
        proposal_generation: int,
        reasons: list[str],
    ) -> None:
        self.state.sanitized_proposal = review.proposal.model_dump(mode="json")
        self.state.sanitized_validation = (
            review.validation.model_dump(mode="json") if review.validation else {}
        )
        if review.outcome == "no_prune":
            self.state.status = "no_prune"
            self._mark_pressure_reviewed(snapshot)
            await self._persist()
            await self._broadcast(
                "proof_prune_no_change",
                reason=review.proposal.reasoning[:1000],
            )
            return
        if review.outcome == "rejected":
            self.state.status = "rejected"
            self._mark_pressure_reviewed(snapshot)
            await self._persist()
            await self._broadcast(
                "proof_prune_rejected",
                proof_id=review.proposal.proof_id,
                reason=(
                    review.validation.reasoning[:1000]
                    if review.validation
                    else review.proposal.reasoning[:1000]
                ),
            )
            return
        if not review.commit_intent:
            raise RuntimeError("Pruning review returned no terminal commit intent.")

        async with self._commit_lock:
            if (
                not self._owns_lifecycle()
                or proposal_generation != self.state.active_proposal_generation
            ):
                raise ProofPruningStaleSnapshotError(
                    "Pruning lifecycle changed before commit."
                )
            try:
                updated, revision = await self.proof_database.commit_pruning_intent(
                    review.commit_intent,
                    snapshot=snapshot,
                    expected_proof_store_id=self.proof_store_id,
                    expected_proof_run_id=self.proof_run_id,
                    expected_lifecycle_generation=self.lifecycle_generation,
                )
            except (RuntimeError, KeyError) as exc:
                self.state.status = "stale"
                self.state.last_error_summary = str(exc)[:1000]
                await self._persist()
                await self._broadcast(
                    "proof_prune_stale",
                    proof_id=review.commit_intent.proof_id,
                    reason=str(exc)[:1000],
                )
                return
            self.state.status = "applied"
            self.state.proof_set_revision = revision
            self.state.last_applied_proof_id = updated.proof_id
            self._mark_pressure_reviewed(snapshot)
            await self._persist()
            if self.invalidate_fn:
                try:
                    invalidation = self.invalidate_fn(updated.proof_id)
                    if asyncio.iscoroutine(invalidation):
                        await invalidation
                except Exception:
                    logger.exception(
                        "Derived proof-context cache refresh failed after prune %s",
                        updated.proof_id,
                    )
            await self._broadcast(
                "proof_prune_applied",
                proof_id=updated.proof_id,
                proof_set_revision=revision,
                reason=updated.live_context_prune_reason[:1000],
            )

    def _mark_pressure_reviewed(self, snapshot: Any) -> None:
        pressure = snapshot.context_pressure
        if pressure:
            self.state.last_reviewed_pressure_fingerprint = (
                pressure.pressure_fingerprint()
            )
            self.state.last_reviewed_pressure_revision = snapshot.proof_set_revision

    async def _handle_failure(self, exc: Exception, reasons: list[str]) -> bool:
        summary = str(exc or exc.__class__.__name__)[:1000]
        async with self._state_lock:
            for reason in reasons:
                if reason not in self.state.queued_trigger_reasons:
                    self.state.queued_trigger_reasons.append(reason)
            self.state.retry_count += 1
            self.state.last_error_summary = summary
            self.state.provider_error_classification = exc.__class__.__name__
            if is_provider_credit_pause_error(exc):
                failure_kind = "credit"
                self.state.status = "provider_paused"
            elif isinstance(exc, RetryableProviderError) or is_transient_model_call_error(exc):
                failure_kind = "retryable"
                self.state.status = "queued"
            elif isinstance(exc, ProofPruningStaleSnapshotError):
                failure_kind = "stale"
                self.state.status = "stale"
            elif isinstance(exc, ProofPruningContextError) or is_non_retryable_model_error(exc):
                failure_kind = "repair"
                self.state.status = "repair_required"
            else:
                failure_kind = "error"
                self.state.status = "error"
            await self._persist()

        # Provider waits are deliberately outside the state lock. Later proof
        # registrations may continue updating cadence while pruning is paused.
        if failure_kind == "credit":
            reset_generation = get_provider_reset_generation()
            await self._broadcast(
                "proof_prune_provider_paused",
                message="Proof pruning paused for provider credits; proof solving continues.",
            )
            await wait_for_provider_reset(
                reset_generation,
                should_stop=lambda: not self._owns_lifecycle(),
            )
            if self._owns_lifecycle():
                async with self._state_lock:
                    self.state.status = "queued"
                    await self._persist()
                return False
            return True

        if failure_kind == "retryable":
            try:
                if isinstance(exc, RetryableProviderError):
                    await api_client_manager.wait_for_retryable_provider_error(
                        exc,
                        role_id=exc.role_id or "proof_prune",
                        should_stop=lambda: not self._owns_lifecycle(),
                    )
                else:
                    await asyncio.sleep(min(30, 2 ** min(self.state.retry_count, 4)))
            except asyncio.CancelledError:
                return True
            return not self._owns_lifecycle()

        if failure_kind == "stale":
            await self._broadcast("proof_prune_stale", reason=summary)
            return False
        if failure_kind == "repair":
            await self._broadcast("proof_prune_repair_required", message=summary)
            return True

        await self._broadcast(
            "proof_prune_error",
            message=(
                "Proof pruning failed without changing proof context; "
                "proof solving continues."
            ),
            error_type=exc.__class__.__name__,
        )
        # Contract errors retain pressure but require an explicit later owner.
        return isinstance(exc, ProofPruningContractError)

    async def drain(self, *, preserve_pending: bool = True) -> None:
        """Fence new work, cancel and observe the owned task."""
        self._accepting_triggers = False
        self._draining = True
        notification_tasks = list(self._notification_tasks)
        for notification_task in notification_tasks:
            notification_task.cancel()
        if notification_tasks:
            await asyncio.gather(*notification_tasks, return_exceptions=True)
        self._notification_tasks.clear()
        task = self._active_task
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        async with self._state_lock:
            if preserve_pending and self.state.status in {
                "proposing",
                "validating",
                "provider_paused",
                "queued",
            }:
                for reason in self.state.active_trigger_reasons:
                    if reason not in self.state.queued_trigger_reasons:
                        self.state.queued_trigger_reasons.append(reason)
                self.state.status = "queued"
                self.state.follow_up_required = True
            elif not preserve_pending:
                self.state.queued_trigger_reasons = []
                self.state.follow_up_required = False
                self.state.status = "idle"
            self.state.active_trigger_reasons = []
            self.state.active_proposal_id = ""
            self.state.snapshot_id = ""
            self.state.snapshot_revision = None
            await self._persist()

    async def clear(self, new_lifecycle_generation: int) -> None:
        """Advance the fence before cancellation and discard pending lifecycle state."""
        self.lifecycle_generation = int(new_lifecycle_generation)
        self.state.lifecycle_generation = int(new_lifecycle_generation)
        await self.drain(preserve_pending=False)
