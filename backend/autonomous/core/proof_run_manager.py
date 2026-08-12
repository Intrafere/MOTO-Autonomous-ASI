"""Owned lifecycle registry for user-triggered proof checks."""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable, Optional

from backend.autonomous.core.proof_verification_stage import ProofVerificationStage
from backend.shared.models import (
    ProofRunCollectionItem,
    ProofRunCollectionResponse,
    ProofRunQueueResponse,
    ProofRunSnapshot,
    ProofRunSourceLookupResponse,
)
from backend.shared.path_safety import validate_single_path_component
from backend.shared.provider_pause import (
    get_provider_reset_generation,
    wait_for_provider_reset,
)
from backend.shared.sleep_inhibitor import sleep_inhibitor

logger = logging.getLogger(__name__)
ProofRunWorker = Callable[["ProofRunControl"], Awaitable[None]]
ProofRunTerminalCallback = Callable[[ProofRunSnapshot], Awaitable[None]]
ProofRunEventCallback = Callable[[str, dict], Awaitable[None]]
TERMINAL_STATUSES = {"completed", "stopped", "error"}
DEDICATED_TERMINAL_REASONS = {
    "context_overflow",
    "proof_context_overflow",
    "proof_output_truncation_recovery_exhausted",
}


class ProofRunSourceInvalidError(RuntimeError):
    """The bound source disappeared and must never be recreated by the worker."""


@dataclass
class ProofRunControl:
    snapshot: ProofRunSnapshot
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    wake_event: asyncio.Event = field(default_factory=asyncio.Event)
    task: Optional[asyncio.Task] = None
    reservation_token: str = ""
    sleep_owner: object = None
    update_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    cleanup_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    terminal_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    cleaned_up: bool = False
    pruning_coordinator: object = None
    terminal_callback: Optional[ProofRunTerminalCallback] = None
    event_callback: Optional[ProofRunEventCallback] = None
    candidate_checkpoint: Optional[dict] = None


class ProofRunManager:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._runs: dict[str, ProofRunControl] = {}

    @staticmethod
    async def purge_legacy_state(data_root: Path) -> None:
        """Remove only obsolete manual proof-run lifecycle projections."""
        state_dir = Path(data_root) / "proof_runs"

        def _purge() -> None:
            if not state_dir.is_dir():
                return
            for pattern in ("proof-run-*.json", "proof-run-*.tmp"):
                for path in state_dir.glob(pattern):
                    if path.is_file():
                        path.unlink()
            try:
                state_dir.rmdir()
            except OSError:
                # Unknown files are deliberately preserved.
                pass

        await asyncio.to_thread(_purge)

    @staticmethod
    def _collection_item(snapshot: ProofRunSnapshot) -> ProofRunCollectionItem:
        return ProofRunCollectionItem(
            **snapshot.model_dump(
                include={
                    "proof_run_id",
                    "run_mode",
                    "scope",
                    "source_type",
                    "source_id",
                    "source_title",
                    "run_id",
                    "lifecycle_generation",
                    "status",
                    "current_round",
                    "last_completed_round",
                    "proof_set_revision",
                    "updated_at",
                    "terminal_reason",
                    "pruning_status",
                }
            )
        )

    async def _collection_snapshots(self) -> list[ProofRunSnapshot]:
        """Return current-process run metadata only."""
        async with self._lock:
            snapshots = [
                control.snapshot.model_copy(deep=True)
                for control in self._runs.values()
            ]
        return sorted(snapshots, key=lambda snapshot: snapshot.updated_at, reverse=True)

    async def list_runs(
        self,
        *,
        limit: int = 20,
        scope: Optional[str] = None,
    ) -> ProofRunCollectionResponse:
        snapshots = await self._collection_snapshots()
        if scope is not None:
            snapshots = [snapshot for snapshot in snapshots if snapshot.scope == scope]
        return ProofRunCollectionResponse(
            runs=[self._collection_item(snapshot) for snapshot in snapshots[:limit]],
            count=min(len(snapshots), limit),
            limit=limit,
            truncated=len(snapshots) > limit,
        )

    async def find_by_source(
        self,
        *,
        scope: str,
        source_type: str,
        source_id: str,
        limit: int = 20,
    ) -> ProofRunSourceLookupResponse:
        snapshots = [
            snapshot
            for snapshot in await self._collection_snapshots()
            if snapshot.scope == scope
            and snapshot.source_type == source_type
            and snapshot.source_id == source_id
        ]
        active = [
            snapshot
            for snapshot in snapshots
            if snapshot.status not in TERMINAL_STATUSES
        ]
        preferred_snapshot = active[0] if active else (snapshots[0] if snapshots else None)
        bounded = snapshots[:limit]
        if (
            preferred_snapshot is not None
            and preferred_snapshot not in bounded
            and limit > 0
        ):
            bounded = [*bounded[: limit - 1], preferred_snapshot]
        return ProofRunSourceLookupResponse(
            runs=[self._collection_item(snapshot) for snapshot in bounded],
            count=len(bounded),
            limit=limit,
            truncated=len(snapshots) > limit,
            scope=scope,
            source_type=source_type,
            source_id=source_id,
            ambiguous=len(active) > 1,
            preferred_proof_run_id=(
                preferred_snapshot.proof_run_id if preferred_snapshot is not None else None
            ),
        )

    async def queue(
        self,
        *,
        scope: str,
        source_type: str,
        source_id: str,
        proof_store_id: str,
        run_id: str,
        worker: ProofRunWorker,
        run_mode: str = "one_round",
        source_title: str = "",
        source_content_fingerprint: str = "",
        source_revision: int = 0,
        proof_set_revision: int = 0,
        route_runtime_fingerprint: str = "",
        candidate_checkpoint_reference: str = "",
        terminal_callback: Optional[ProofRunTerminalCallback] = None,
        event_callback: Optional[ProofRunEventCallback] = None,
    ) -> ProofRunQueueResponse:
        if run_mode not in {"one_round", "loop_with_pruning"}:
            raise ValueError(f"Unsupported proof run mode: {run_mode}")
        proof_run_id = f"proof-run-{uuid.uuid4().hex}"
        reservation_token = f"{proof_run_id}:{uuid.uuid4().hex}"
        sleep_owner = ("manual_proof_check", proof_run_id)
        await ProofVerificationStage.reserve_source(
            source_type,
            source_id,
            owner_token=reservation_token,
        )
        try:
            sleep_inhibitor.acquire(sleep_owner)
            now = datetime.now()
            snapshot = ProofRunSnapshot(
                proof_run_id=proof_run_id,
                run_mode=run_mode,
                scope=scope,
                source_type=source_type,
                source_id=source_id,
                source_title=source_title,
                proof_store_id=proof_store_id,
                run_id=run_id,
                lifecycle_generation=1,
                status="queued",
                source_content_fingerprint=source_content_fingerprint,
                source_revision=source_revision,
                proof_set_revision=proof_set_revision,
                route_runtime_fingerprint=route_runtime_fingerprint,
                candidate_checkpoint_reference=candidate_checkpoint_reference,
                round_limit=None if run_mode == "loop_with_pruning" else 1,
                unbounded=run_mode == "loop_with_pruning",
                updated_at=now,
                pruning_status=(
                    "idle" if run_mode == "loop_with_pruning" else "disabled"
                ),
            )
            control = ProofRunControl(
                snapshot=snapshot,
                reservation_token=reservation_token,
                sleep_owner=sleep_owner,
                terminal_callback=terminal_callback,
                event_callback=event_callback,
            )
            async with self._lock:
                self._runs[proof_run_id] = control
            await self._emit(control, "proof_run_queued")
            async with self._lock:
                control.task = asyncio.create_task(self._drive(control, worker))
            return ProofRunQueueResponse(**snapshot.model_dump(), queued=True)
        except Exception:
            await ProofVerificationStage.release_source(
                source_type,
                source_id,
                owner_token=reservation_token,
            )
            sleep_inhibitor.release(sleep_owner)
            async with self._lock:
                self._runs.pop(proof_run_id, None)
            raise

    async def _drive(self, control: ProofRunControl, worker: ProofRunWorker) -> None:
        generation = control.snapshot.lifecycle_generation
        try:
            await self.update(
                control,
                expected_generation=generation,
                status="running",
                started_at=control.snapshot.started_at or datetime.now(),
                idle_reason="",
                idle_policy=None,
            )
            await worker(control)
            if control.snapshot.status in TERMINAL_STATUSES:
                return
            if control.stop_event.is_set():
                await self._finish(
                    control,
                    generation,
                    status="stopped",
                    terminal_reason="user_stopped",
                )
            else:
                await self._finish(
                    control,
                    generation,
                    status="completed",
                )
        except asyncio.CancelledError:
            control.stop_event.set()
            await self._finish(
                control,
                generation,
                status="stopped",
                terminal_reason="cancelled",
            )
            raise
        except ProofRunSourceInvalidError as exc:
            await self._finish(
                control,
                generation,
                status="error",
                terminal_reason="source_invalid",
                last_error_summary=str(exc)[:1800],
            )
        except Exception as exc:
            logger.exception("Proof run %s failed", control.snapshot.proof_run_id)
            await self._finish(
                control,
                generation,
                status="error",
                terminal_reason="proof_check_error",
                last_error_summary=str(exc)[:1800],
            )
        finally:
            await self.cleanup(control)

    @staticmethod
    def _event_payload(snapshot: ProofRunSnapshot) -> dict:
        return {
            "proof_run_id": snapshot.proof_run_id,
            "run_mode": snapshot.run_mode,
            "scope": snapshot.scope,
            "source_type": snapshot.source_type,
            "source_id": snapshot.source_id,
            "run_id": snapshot.run_id,
            "round_index": snapshot.current_round,
            "round_limit": snapshot.round_limit,
            "unbounded": snapshot.unbounded,
            "lifecycle_generation": snapshot.lifecycle_generation,
            "status": snapshot.status,
            "idle_reason": snapshot.idle_reason,
            "terminal_reason": snapshot.terminal_reason,
        }

    async def _emit(
        self,
        control: ProofRunControl,
        event_type: str,
        extra: Optional[dict] = None,
    ) -> None:
        if control.event_callback is None:
            return
        payload = self._event_payload(control.snapshot)
        if extra:
            payload.update(extra)
        await control.event_callback(event_type, payload)

    async def update(
        self,
        control: ProofRunControl,
        *,
        expected_generation: Optional[int] = None,
        **updates,
    ) -> ProofRunSnapshot:
        async with control.update_lock:
            if (
                expected_generation is not None
                and control.snapshot.lifecycle_generation != expected_generation
            ):
                return control.snapshot.model_copy(deep=True)
            requested_status = updates.get("status")
            if (
                control.snapshot.stop_requested
                and requested_status is not None
                and requested_status not in TERMINAL_STATUSES
                and requested_status != "stopping"
            ):
                return control.snapshot.model_copy(deep=True)
            updates["updated_at"] = datetime.now()
            control.snapshot = control.snapshot.model_copy(update=updates)
            return control.snapshot

    async def _finish(
        self,
        control: ProofRunControl,
        generation: int,
        *,
        status: str,
        terminal_reason: str = "",
        **updates,
    ) -> ProofRunSnapshot:
        effective_terminal_reason = terminal_reason or control.snapshot.terminal_reason
        snapshot = await self.update(
            control,
            expected_generation=generation,
            status=status,
            terminal_reason=effective_terminal_reason,
            idle_reason="",
            idle_policy=None,
            **updates,
        )
        async with control.terminal_lock:
            snapshot = control.snapshot
            if (
                snapshot.lifecycle_generation == generation
                and snapshot.status in TERMINAL_STATUSES
                and not snapshot.terminal_event_emitted
            ):
                if control.terminal_callback is not None:
                    await control.terminal_callback(snapshot.model_copy(deep=True))
                if snapshot.terminal_reason not in DEDICATED_TERMINAL_REASONS:
                    await self._emit(control, "proof_run_terminal")
                snapshot = await self.update(
                    control,
                    expected_generation=generation,
                    terminal_event_emitted=True,
                )
        return snapshot

    async def save_pruning_state(
        self,
        control: ProofRunControl,
        state: dict,
    ) -> bool:
        """Project pruning lifecycle status into the scope-neutral run record."""
        if (
            str(state.get("proof_run_id") or "")
            != control.snapshot.proof_run_id
            or int(state.get("lifecycle_generation") or 0)
            != control.snapshot.lifecycle_generation
        ):
            return False
        status = str(state.get("status") or "idle")
        await self.update(
            control,
            pruning_status=status,
            pruning_state=dict(state),
        )
        return True

    async def load_pruning_state(
        self,
        control: ProofRunControl,
    ) -> Optional[dict]:
        state = getattr(control.snapshot, "pruning_state", None)
        return dict(state) if isinstance(state, dict) else None

    async def save_candidate_checkpoint(
        self,
        control: ProofRunControl,
        payload: dict,
    ) -> bool:
        """Keep the process-local candidate checkpoint for same-round retries."""
        if not isinstance(payload, dict):
            return False
        if (
            str(payload.get("source_type") or "") != control.snapshot.source_type
            or str(payload.get("source_id") or "") != control.snapshot.source_id
        ):
            return False
        async with control.update_lock:
            control.candidate_checkpoint = dict(payload)
        return True

    async def clear_candidate_checkpoint(
        self,
        control: ProofRunControl,
    ) -> None:
        """Discard process-local candidate state after a round reaches a clean boundary."""
        async with control.update_lock:
            control.candidate_checkpoint = None

    async def load_candidate_checkpoint(
        self,
        control: ProofRunControl,
    ) -> Optional[dict]:
        async with control.update_lock:
            checkpoint = control.candidate_checkpoint
            return dict(checkpoint) if isinstance(checkpoint, dict) else None

    async def complete_round(
        self,
        control: ProofRunControl,
        *,
        round_number: int,
        valid_candidate_count: Optional[int] = None,
        summary: str = "",
        reference: str = "",
        candidate_checkpoint_reference: str = "",
        proof_set_revision: Optional[int] = None,
    ) -> ProofRunSnapshot:
        """Persist only bounded round metadata; candidate payloads stay external."""
        updates = {
            "current_round": round_number,
            "last_completed_round": round_number,
            "last_round_summary": summary[:4000],
            "last_round_reference": reference[:512],
            "candidate_checkpoint_reference": candidate_checkpoint_reference[:512],
        }
        if proof_set_revision is not None:
            updates["proof_set_revision"] = proof_set_revision
        async with control.update_lock:
            if not candidate_checkpoint_reference:
                control.candidate_checkpoint = None
            updates["updated_at"] = datetime.now()
            control.snapshot = control.snapshot.model_copy(update=updates)
            snapshot = control.snapshot.model_copy(deep=True)
        await self._emit(
            control,
            "proof_run_round_complete",
            {
                "candidate_count": valid_candidate_count,
                "round_summary": summary[:4000],
                "next_round_automatic": (
                    control.snapshot.run_mode == "loop_with_pruning"
                    and not control.stop_event.is_set()
                    and not control.snapshot.terminal_reason
                ),
            },
        )
        return snapshot

    async def begin_round(
        self,
        control: ProofRunControl,
        *,
        round_number: int,
    ) -> ProofRunSnapshot:
        snapshot = await self.update(
            control,
            expected_generation=control.snapshot.lifecycle_generation,
            status="running",
            current_round=round_number,
            idle_reason="",
            idle_policy=None,
        )
        await self._emit(control, "proof_run_round_started")
        return snapshot

    async def provider_paused(
        self,
        control: ProofRunControl,
        *,
        provider_state: Optional[dict] = None,
    ) -> ProofRunSnapshot:
        control.wake_event.clear()
        durable_provider_state = dict(provider_state or {})
        durable_provider_state.setdefault(
            "observed_reset_generation",
            get_provider_reset_generation(),
        )
        snapshot = await self.update(
            control,
            expected_generation=control.snapshot.lifecycle_generation,
            status="provider_paused",
            idle_reason="provider_credit_pause",
            idle_policy="provider_reset",
            provider_state=durable_provider_state,
        )
        await self._emit(control, "proof_run_provider_paused")
        return snapshot

    async def wait_for_provider_or_control_wake(
        self,
        control: ProofRunControl,
    ) -> None:
        observed = int(
            (control.snapshot.provider_state or {}).get(
                "observed_reset_generation",
                get_provider_reset_generation(),
            )
        )
        provider_wait = asyncio.create_task(
            wait_for_provider_reset(
                observed,
                should_stop=lambda: control.stop_event.is_set(),
            )
        )
        control_wait = asyncio.create_task(control.wake_event.wait())
        try:
            done, pending = await asyncio.wait(
                {provider_wait, control_wait},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            for task in done:
                await task
        finally:
            for task in (provider_wait, control_wait):
                if not task.done():
                    task.cancel()

    async def repair_required(
        self,
        control: ProofRunControl,
        *,
        reason: str,
    ) -> ProofRunSnapshot:
        return await self._finish(
            control,
            control.snapshot.lifecycle_generation,
            status="error",
            terminal_reason="repair_required",
            last_error_summary=reason[:1800],
        )

    async def error(
        self,
        control: ProofRunControl,
        *,
        terminal_reason: str,
        reason: str = "",
    ) -> ProofRunSnapshot:
        return await self._finish(
            control,
            control.snapshot.lifecycle_generation,
            status="error",
            terminal_reason=terminal_reason,
            last_error_summary=reason[:1800],
        )

    async def resumed(
        self,
        control: ProofRunControl,
        *,
        from_provider_pause: bool = False,
    ) -> ProofRunSnapshot:
        snapshot = await self.update(
            control,
            expected_generation=control.snapshot.lifecycle_generation,
            status="running",
            idle_reason="",
            idle_policy=None,
            provider_state=None,
        )
        await self._emit(
            control,
            "proof_run_provider_resumed"
            if from_provider_pause
            else "proof_run_resumed",
        )
        return snapshot

    async def cleanup(self, control: ProofRunControl) -> None:
        async with control.cleanup_lock:
            if control.cleaned_up:
                return
            control.cleaned_up = True
            if control.pruning_coordinator is not None:
                try:
                    await control.pruning_coordinator.drain(preserve_pending=False)
                except Exception:
                    logger.exception(
                        "Failed to drain pruning coordinator for %s",
                        control.snapshot.proof_run_id,
                    )
            try:
                await ProofVerificationStage.release_source(
                    control.snapshot.source_type,
                    control.snapshot.source_id,
                    owner_token=control.reservation_token,
                )
            finally:
                sleep_inhibitor.release(control.sleep_owner)
                await self.update(control, cleanup_completed=True)

    async def get(self, proof_run_id: str) -> Optional[ProofRunSnapshot]:
        safe_run_id = validate_single_path_component(proof_run_id, "proof run ID")
        async with self._lock:
            control = self._runs.get(safe_run_id)
            if control:
                return control.snapshot.model_copy(deep=True)
        return None

    async def stop(
        self,
        proof_run_id: str,
        expected_lifecycle_generation: int,
    ) -> Optional[ProofRunSnapshot]:
        safe_run_id = validate_single_path_component(proof_run_id, "proof run ID")
        async with self._lock:
            control = self._runs.get(safe_run_id)
        if control is None:
            snapshot = await self.get(safe_run_id)
            if snapshot and snapshot.status in {"completed", "stopped", "error"}:
                if snapshot.lifecycle_generation != expected_lifecycle_generation:
                    raise RuntimeError("Proof run lifecycle changed; refresh and retry.")
                return snapshot
            return None
        if control.snapshot.lifecycle_generation != expected_lifecycle_generation:
            raise RuntimeError("Proof run lifecycle changed; refresh and retry.")
        if control.snapshot.status in {"completed", "stopped", "error"}:
            return control.snapshot.model_copy(deep=True)
        control.stop_event.set()
        control.wake_event.set()
        return await self.update(
            control,
            expected_generation=expected_lifecycle_generation,
            status="stopping",
            stop_requested=True,
            wake_generation=control.snapshot.wake_generation + 1,
        )

    async def shutdown_all(self) -> None:
        """Stop and fully drain every current-process manual proof run."""
        async with self._lock:
            controls = list(self._runs.values())
        for control in controls:
            control.stop_event.set()
            control.wake_event.set()
            if control.task is not None and not control.task.done():
                control.task.cancel()
        tasks = [
            control.task
            for control in controls
            if control.task is not None
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for control in controls:
            await self.cleanup(control)


proof_run_manager = ProofRunManager()
