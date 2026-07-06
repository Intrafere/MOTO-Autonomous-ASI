"""Coordinator for the additive LeanOJ proof-solver mode."""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import re
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import aiofiles

from backend.leanoj.core.leanoj_context import (
    ARTIFACT_ACCEPTED_IDEAS,
    ARTIFACT_FAILED_SUBPROOFS,
    ARTIFACT_FINAL_ATTEMPTS,
    ARTIFACT_FINAL_CYCLE_PACKETS,
    ARTIFACT_PARTIAL_PROOFS,
    ARTIFACT_VERIFIED_SUBPROOFS,
    _remove_attempt_count_language,
    leanoj_context_manager,
)
from backend.leanoj.prompts import (
    CREATIVITY_EMPHASIS_BOOST_PROMPT,
    build_brainstorm_batch_validation_prompt,
    build_brainstorm_prompt,
    build_brainstorm_prune_review_prompt,
    build_brainstorm_prune_validation_prompt,
    build_brainstorm_validation_prompt,
    build_final_solution_review_prompt,
    build_final_solver_prompt,
    build_master_proof_edit_validation_prompt,
    build_path_decision_prompt,
    build_path_validation_prompt,
    build_sufficiency_prompt,
    build_topic_batch_validation_prompt,
    build_topic_candidate_prompt,
    build_topic_selection_prompt,
    build_topic_validation_prompt,
)
from backend.autonomous.memory.autonomous_api_logger import autonomous_api_logger
from backend.autonomous.memory.proof_database import proof_database
from backend.shared.api_client_manager import RetryableProviderError, api_client_manager
from backend.shared.brainstorm_proof_gate import is_lean_proof_submission, verify_brainstorm_proof_candidate
from backend.shared.config import rag_config, system_config
from backend.shared.context_overflow import (
    CONTEXT_OVERFLOW_RESOLUTION,
    CONTEXT_OVERFLOW_STOP_MESSAGE,
    CONTEXT_OVERFLOW_STOP_REASON,
)
from backend.shared.json_parser import parse_json
from backend.shared.response_extraction import extract_message_text
from backend.shared.lean4_client import Lean4Result, get_lean4_client
from backend.shared.lean_proof_integrity import strip_lean_comments_and_strings, validate_lean_proof_integrity
from backend.shared.model_error_utils import is_non_retryable_model_error
from backend.shared.models import (
    LeanOJAttemptRecord,
    LeanOJRoleConfig,
    LeanOJStartRequest,
    LeanOJState,
    LeanOJSubproofRecord,
    ModelConfig,
    ProofAttemptFeedback,
    ProofRecord,
    WorkflowTask,
)
from backend.shared.provider_pause import (
    is_provider_credit_pause_error,
    mark_provider_paused,
    wait_for_provider_resume,
)
from backend.shared.proof_search.assistant_coordinator import assistant_proof_search_coordinator
from backend.shared.proof_search.assistant_models import AssistantTargetSnapshot
from backend.shared.token_tracker import token_tracker
from backend.shared.utils import count_tokens

logger = logging.getLogger(__name__)

BroadcastFn = Optional[Callable[[str, dict[str, Any]], Awaitable[None]]]
_LEAN_PLACEHOLDER_RE = re.compile(r"(?<![A-Za-z0-9_'])(sorry|admit)(?![A-Za-z0-9_'])")
_LEAN_TOP_LEVEL_DECL_KINDS = (
    "abbrev",
    "axiom",
    "class",
    "constant",
    "def",
    "example",
    "inductive",
    "instance",
    "lemma",
    "opaque",
    "structure",
    "theorem",
)
_LEAN_TOP_LEVEL_DECL_KIND_PATTERN = "|".join(_LEAN_TOP_LEVEL_DECL_KINDS)
_LEAN_TOP_LEVEL_DECL_RE = re.compile(
    r"(?m)^\s*(?:open\s+Classical\s+in\s+)?(?:@\[[^\]]+\]\s*)*(?:(?:private|protected|noncomputable|unsafe)\s+)*"
    rf"(?:{_LEAN_TOP_LEVEL_DECL_KIND_PATTERN})\b"
)
_LEAN_DECL_KEY_RE = re.compile(
    r"^\s*(?:open\s+Classical\s+in\s+)?(?:@\[[^\]]+\]\s*)*(?:(?:private|protected|noncomputable|unsafe)\s+)*"
    rf"(?P<kind>{_LEAN_TOP_LEVEL_DECL_KIND_PATTERN})\s+(?P<name>[A-Za-z_][A-Za-z0-9_'.]*|«[^»]+»)?",
    re.DOTALL,
)
_TERMINAL_PHASES = {"verified"}
_ACTIVE_PHASES = {
    "initial_topic_candidates",
    "initial_brainstorm",
    "path_decision",
    "recursive_brainstorm",
    "final_proof_loop",
}
_PHASE_PROGRESS_RANK = {
    "idle": 0,
    "initial_topic_candidates": 1,
    "initial_brainstorm": 2,
    "recursive_brainstorm": 3,
    "path_decision": 4,
    "final_proof_loop": 5,
    "error": 7,
    "stopped": 7,
}
_LEANOJ_PATH_OPTIONS = ("solve_final_now", "need_more_brainstorming")
_LEANOJ_PATH_OPTIONS_SET = set(_LEANOJ_PATH_OPTIONS)
_LEANOJ_PROOF_EDIT_ACTIONS = {"edit_proof"}
_LEANOJ_PROOF_EDIT_OPERATIONS = {"full_content", "replace", "insert_after", "delete"}
_MASTER_PROOF_EDIT_LOG_COMPACT_RECORD_LIMIT = 500
_MASTER_PROOF_EDIT_LOG_RECENT_RECORDS_TO_KEEP = 150
_MASTER_PROOF_NO_PROGRESS_LIMIT = 8
_MASTER_PROOF_STALE_EDIT_FAILURE_HANDOFF_COUNT = 3
_MASTER_PROOF_SHORTENING_CHAR_THRESHOLD = 80
_LEANOJ_CONTEXT_ROLES = {"active_plan", "verified_hint", "refuted_construction", "scratch"}
_LEANOJ_FINAL_ACTIVE_CONTEXT_ROLES = {"active_plan"}
_LEANOJ_REFUTED_CONTEXT_TERMS = (
    "counterexample",
    "refuted",
    "do not use",
    "fails at",
    "fails for",
    "falsified",
    "false construction",
    "invalid construction",
    "construction is invalid",
    "candidate is invalid",
)
_LEANOJ_ACTIVE_PLAN_CONTEXT_TERMS = (
    "active proof plan",
    "current proof route",
    "chosen proof route",
    "current chosen proof route",
    "master proof route",
    "next obligation",
)
_LEANOJ_PROOF_SEARCH_MAX_TOKENS = 3500


class LeanOJConfigurationError(RuntimeError):
    """Non-retryable LeanOJ configuration problem."""


_BrainstormSubmission = tuple[int, str, dict[str, Any]]
_TopicCandidate = tuple[int, str, dict[str, Any]]


class _LeanOJBrainstormSubmissionQueue:
    """LeanOJ-local pending queue with aggregator-style fairness accounting."""

    def __init__(self, submitter_count: int) -> None:
        self.queue: asyncio.Queue[_BrainstormSubmission] = asyncio.Queue()
        self.submitter_count = submitter_count
        self.pending_by_submitter: dict[int, int] = {}
        self.global_paused = False
        self.paused_submitters: set[int] = set()

    def qsize(self) -> int:
        return self.queue.qsize()

    def count_for_submitter(self, submitter_index: int) -> int:
        return self.pending_by_submitter.get(submitter_index, 0)

    def should_pause_submitter(self, submitter_index: int) -> bool:
        if self.qsize() >= system_config.queue_overflow_threshold:
            return True
        if self.submitter_count <= 1:
            return False
        return (
            self.count_for_submitter(submitter_index)
            > system_config.per_submitter_queue_threshold
        )

    async def put(self, item: _BrainstormSubmission) -> None:
        await self.queue.put(item)
        submitter_index = item[0]
        self.pending_by_submitter[submitter_index] = (
            self.pending_by_submitter.get(submitter_index, 0) + 1
        )

    async def dequeue_batch(
        self,
        *,
        max_count: int = 3,
        timeout: float = 1.0,
        collect_window: float = 0.25,
    ) -> list[_BrainstormSubmission]:
        try:
            first = await asyncio.wait_for(self.queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return []

        batch = [first]
        self._decrement_submitter(first[0])
        deadline = time.monotonic() + collect_window
        while len(batch) < max_count:
            item = None
            with contextlib.suppress(asyncio.QueueEmpty):
                item = self.queue.get_nowait()
            if item is not None:
                batch.append(item)
                self._decrement_submitter(item[0])
                continue

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                batch.append(item)
                self._decrement_submitter(item[0])
            except asyncio.TimeoutError:
                break
        return batch

    def refresh_pause_transitions(self) -> dict[str, Any]:
        queue_size = self.qsize()
        next_global_paused = queue_size >= system_config.queue_overflow_threshold
        next_paused_submitters = self._current_paused_submitters()

        transitions = {
            "queue_size": queue_size,
            "global_paused": next_global_paused,
            "global_changed": next_global_paused != self.global_paused,
            "submitters_paused": next_paused_submitters - self.paused_submitters,
            "submitters_resumed": self.paused_submitters - next_paused_submitters,
        }
        self.global_paused = next_global_paused
        self.paused_submitters = next_paused_submitters
        return transitions

    def _current_paused_submitters(self) -> set[int]:
        if self.submitter_count <= 1:
            return set()
        return {
            submitter_index
            for submitter_index, pending_count in self.pending_by_submitter.items()
            if pending_count > system_config.per_submitter_queue_threshold
        }

    def _decrement_submitter(self, submitter_index: int) -> None:
        pending_count = self.pending_by_submitter.get(submitter_index, 0)
        if pending_count <= 1:
            self.pending_by_submitter.pop(submitter_index, None)
            return
        self.pending_by_submitter[submitter_index] = pending_count - 1


class LeanOJCoordinator:
    """Run the proof-only LeanOJ workflow as an isolated third mode."""

    def __init__(self) -> None:
        self._running = False
        self._state = LeanOJState()
        self._request: Optional[LeanOJStartRequest] = None
        self._stop_event = asyncio.Event()
        self._main_task: Optional[asyncio.Task] = None
        self._broadcast_callback: BroadcastFn = None
        self._task_sequences: dict[str, int] = {}

        self._validated_topics: list[str] = []
        self._accepted_ideas: list[str] = []
        self._accepted_idea_records: list[dict[str, Any]] = []
        self._failed_feedback: list[dict[str, Any]] = []
        self._last_brainstorm_validation_decisions: list[dict[str, Any]] = []
        self._final_attempts: list[dict[str, Any]] = []
        self._final_context_events: list[dict[str, Any]] = []
        self._partial_proofs: list[dict[str, Any]] = []
        self._final_cycle_packets: list[dict[str, Any]] = []
        self._current_final_cycle_packet: Optional[dict[str, Any]] = None
        self._current_working_proof_attempt: Optional[dict[str, Any]] = None

        self.workflow_tasks: list[WorkflowTask] = []
        self.completed_task_ids: set[str] = set()
        self.current_task_id: Optional[str] = None
        self._restored_from_disk = False
        self._master_proof_no_progress_count = 0
        self._last_master_proof_edit_signature = ""
        self._pending_final_solver_assistant_target_hash = ""
        self._fatal_stop_reason: Optional[str] = None
        self._fatal_stop_message: str = ""

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_active(self) -> bool:
        return self._running or (self._main_task is not None and not self._main_task.done())

    def set_broadcast_callback(self, callback: Callable[[str, dict[str, Any]], Awaitable[None]]) -> None:
        self._broadcast_callback = callback

    async def _broadcast(self, event: str, data: Optional[dict[str, Any]] = None) -> None:
        if self._broadcast_callback:
            await self._broadcast_callback(event, data or {})

    @staticmethod
    def _is_context_overflow_exception(exc: BaseException) -> bool:
        message = str(exc or "").lower()
        return (
            "context overflow" in message
            or "prompt context overflow" in message
            or "exceeds the configured input budget" in message
            or "mandatory direct-inject" in message
        )

    async def _handle_context_overflow_stop(self, exc: BaseException, *, role_id: str = "") -> None:
        self._fatal_stop_reason = CONTEXT_OVERFLOW_STOP_REASON
        self._fatal_stop_message = CONTEXT_OVERFLOW_STOP_MESSAGE
        self._state.phase = "stopped"
        self._state.last_error = str(exc)
        await self._persist_and_broadcast(
            "context_overflow_error",
            {
                "workflow_mode": "leanoj",
                "role_id": role_id,
                "phase": self._state.phase,
                "reason": CONTEXT_OVERFLOW_STOP_REASON,
                "message": CONTEXT_OVERFLOW_STOP_MESSAGE,
                "error_detail": str(exc),
                "resolution": CONTEXT_OVERFLOW_RESOLUTION,
            },
        )

    def get_state(self) -> LeanOJState:
        return self._state

    def get_status(self) -> dict[str, Any]:
        self._ensure_accepted_idea_records()
        payload = self._state.model_dump(mode="json")
        payload.update(
            {
                "validated_topics": list(self._validated_topics),
                "accepted_ideas": list(self._accepted_ideas),
                "accepted_idea_records": list(self._accepted_idea_records),
                "failed_feedback": list(self._failed_feedback[-20:]),
                "final_attempts": list(self._final_attempts[-20:]),
                "final_context_events": list(self._final_context_events[-20:]),
                "partial_proofs": list(self._partial_proofs[-20:]),
                "final_cycle_packets": list(self._final_cycle_packets[-5:]),
                "current_final_cycle_packet": self._current_final_cycle_packet,
                "current_working_proof_attempt": self._current_working_proof_attempt,
                "workflow_tasks": [task.model_dump(mode="json") for task in self.workflow_tasks],
                "resume_available": self._request is not None
                and self._state.phase not in _TERMINAL_PHASES
                and not self._state.final_solution,
            }
        )
        return payload

    async def restore_latest_session(self, *, auto_resume: bool = False) -> bool:
        """Restore the latest saved LeanOJ session and optionally resume it."""
        if self.is_active:
            return False

        state_file = self._find_best_resumable_state_file() or self._find_latest_state_file()
        if state_file is None:
            return False

        try:
            async with aiofiles.open(state_file, "r", encoding="utf-8") as f:
                payload = json.loads(await f.read())
            self._restore_from_payload(payload)
        except Exception as exc:
            logger.warning("Failed to restore LeanOJ session from %s: %s", state_file, exc)
            return False

        logger.info(
            "Restored LeanOJ session %s (phase=%s, accepted=%s, final_attempts=%s)",
            self._state.session_id,
            self._state.phase,
            len(self._accepted_ideas),
            self._state.final_attempt_count,
        )

        if (
            auto_resume
            and self._request is not None
            and self._state.phase not in _TERMINAL_PHASES
            and not self._state.final_solution
        ):
            logger.info("Auto-resuming interrupted Proof Solver session %s", self._state.session_id)
            self.start_in_background()

        return True

    async def initialize(self, request: LeanOJStartRequest) -> None:
        if self.is_active:
            raise RuntimeError("Proof Solver is already running")
        if not request.user_prompt.strip():
            raise ValueError("Proof Solver user prompt is required")
        if not request.lean_template.strip():
            raise ValueError("Proof Solver template is required")
        if not request.brainstorm_submitters:
            raise ValueError("At least one Proof Solver brainstorm submitter is required")
        missing_roles = self._missing_model_roles(request)
        if missing_roles:
            raise ValueError(
                "Proof Solver role model configuration is incomplete. Missing model for: "
                + ", ".join(missing_roles)
            )

        self._request = request
        self._stop_event = asyncio.Event()
        self._task_sequences = {}
        self._validated_topics = []
        self._accepted_ideas = []
        self._accepted_idea_records = []
        self._failed_feedback = []
        self._final_attempts = []
        self._final_context_events = []
        self._partial_proofs = []
        self._final_cycle_packets = []
        self._current_final_cycle_packet = None
        self._current_working_proof_attempt = None
        self.workflow_tasks = []
        self.completed_task_ids = set()
        self.current_task_id = None
        self._restored_from_disk = False
        self._master_proof_no_progress_count = 0
        self._last_master_proof_edit_signature = ""

        self._state = LeanOJState(
            is_running=False,
            phase="idle",
            session_id=f"leanoj_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
        )

        self._configure_roles(request)
        await self._persist_state()

    async def resume_or_initialize(self, request: LeanOJStartRequest) -> bool:
        """Resume matching saved progress when possible, otherwise create a new run."""
        if self.is_active:
            raise RuntimeError("Proof Solver is already running")
        if not request.user_prompt.strip():
            raise ValueError("Proof Solver user prompt is required")
        if not request.lean_template.strip():
            raise ValueError("Proof Solver template is required")
        if not request.brainstorm_submitters:
            raise ValueError("At least one Proof Solver brainstorm submitter is required")
        missing_roles = self._missing_model_roles(request)
        if missing_roles:
            raise ValueError(
                "Proof Solver role model configuration is incomplete. Missing model for: "
                + ", ".join(missing_roles)
            )

        matching_state_file = self._find_best_matching_state_file(request)
        if matching_state_file is None:
            await self.initialize(request)
            return False

        try:
            async with aiofiles.open(matching_state_file, "r", encoding="utf-8") as f:
                payload = json.loads(await f.read())
            self._restore_from_payload(payload)
        except Exception as exc:
            logger.warning("Failed to restore matching LeanOJ session from %s: %s", matching_state_file, exc)
            await self.initialize(request)
            return False

        # Keep accumulated proof context, but let the restarted run use the
        # latest model/fallback settings from the UI.
        self._request = request
        self._stop_event = asyncio.Event()
        self._configure_roles(request)
        self._restored_from_disk = True
        await self._persist_state()
        logger.info(
            "Prepared LeanOJ session %s for resume from %s",
            self._state.session_id,
            matching_state_file,
        )
        return True

    def start_in_background(self) -> bool:
        if self._main_task and not self._main_task.done():
            return False
        self._main_task = asyncio.create_task(self.start())
        self._main_task.add_done_callback(self._on_task_done)
        return True

    def _on_task_done(self, task: asyncio.Task) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            logger.info("LeanOJ coordinator task cancelled")
        except Exception:
            logger.exception("LeanOJ coordinator task failed")
        finally:
            if self._main_task is task:
                self._main_task = None

    def _enable_api_logging(self) -> None:
        async def log_callback(
            task_id,
            role_id,
            model,
            provider,
            prompt,
            response,
            tokens_used,
            duration_ms,
            success,
            error,
            phase,
        ):
            try:
                await autonomous_api_logger.log_api_call(
                    task_id=task_id,
                    role_id=role_id,
                    model=model,
                    provider=provider,
                    prompt=prompt,
                    response_content=response,
                    tokens_used=tokens_used,
                    duration_ms=duration_ms,
                    success=success,
                    error=error,
                    phase=phase or self._state.phase or "leanoj",
                    workflow="leanoj",
                )
            except Exception as exc:
                logger.error("Failed to log LeanOJ API call: %s", exc)

        api_client_manager.set_autonomous_logger_callback(log_callback)
        logger.info("LeanOJ API logging enabled")

    async def start(self) -> None:
        if self._request is None:
            raise RuntimeError("LeanOJ coordinator must be initialized before start")
        if self._running:
            return

        self._running = True
        self._state.is_running = True
        self._fatal_stop_reason = None
        self._fatal_stop_message = ""
        if self._state.phase == "idle":
            self._state.phase = "initial_topic_candidates"
        elif self._state.phase in {"stopped", "error"}:
            self._state.phase = self._infer_resume_phase()
        self._remember_active_phase()
        self._state.updated_at = datetime.now()
        self._state.last_error = ""
        token_tracker.reset()
        token_tracker.start_timer()
        self._enable_api_logging()
        await self._persist_and_broadcast("leanoj_started")
        if self._state.provider_paused:
            pause_payload = {
                "reason": self._state.provider_pause_reason,
                "role_id": self._state.provider_pause_role_id,
                "message": self._state.provider_pause_message,
                "phase": self._state.phase,
            }
            mark_provider_paused()
            await self._persist_and_broadcast("leanoj_provider_paused", pause_payload)
            await wait_for_provider_resume(self._should_stop)
            if self._should_stop():
                raise asyncio.CancelledError()
            self._state.provider_paused = False
            self._state.provider_pause_reason = ""
            self._state.provider_pause_role_id = ""
            self._state.provider_pause_message = ""
            await self._persist_and_broadcast("leanoj_provider_resumed", pause_payload)

        try:
            await self._run_workflow(self._request)
        except asyncio.CancelledError:
            raise
        except LeanOJConfigurationError as exc:
            if self._is_context_overflow_exception(exc):
                logger.error("LeanOJ stopped for context overflow: %s", exc)
                await self._handle_context_overflow_stop(exc)
            else:
                logger.exception("LeanOJ workflow failed")
                self._state.phase = "error"
                self._state.last_error = str(exc)
                await self._persist_and_broadcast("leanoj_error", {"message": str(exc)})
        except Exception as exc:
            if self._is_context_overflow_exception(exc):
                logger.error("LeanOJ stopped for context overflow: %s", exc)
                await self._handle_context_overflow_stop(exc)
            else:
                logger.exception("LeanOJ workflow failed")
                self._state.phase = "error"
                self._state.last_error = str(exc)
                await self._persist_and_broadcast("leanoj_error", {"message": str(exc)})
        finally:
            self._running = False
            self._state.is_running = False
            if self._state.phase not in {"verified", "error"}:
                self._remember_active_phase()
            self._state.updated_at = datetime.now()
            token_tracker.stop_timer()
            api_client_manager.set_autonomous_logger_callback(None)
            stopped_payload = None
            if self._fatal_stop_reason:
                stopped_payload = {
                    **self.get_status(),
                    "reason": self._fatal_stop_reason,
                    "message": self._fatal_stop_message or CONTEXT_OVERFLOW_STOP_MESSAGE,
                }
            await self._persist_and_broadcast("leanoj_stopped", stopped_payload)

    async def stop(self) -> None:
        if not self.is_active and not self._state.session_id:
            return
        self._stop_event.set()
        task = self._main_task
        if task and not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=5)
            except asyncio.TimeoutError:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        if not self._running:
            self._state.is_running = False
            if self._state.phase not in {"verified", "error"}:
                self._remember_active_phase()
            await self._persist_and_broadcast("leanoj_stopped")

    async def clear(self) -> None:
        """Clear Proof Solver progress. This is the explicit reset path."""
        if self.is_active:
            await self.stop()
        base = self._sessions_base_dir()
        if base.exists():
            shutil.rmtree(base)
        partial_base = self._partial_proofs_base_dir()
        if partial_base.exists():
            shutil.rmtree(partial_base)
        await leanoj_context_manager.clear_all()

        self._running = False
        self._state = LeanOJState()
        self._request = None
        self._stop_event = asyncio.Event()
        self._main_task = None
        self._task_sequences = {}
        self._validated_topics = []
        self._accepted_ideas = []
        self._accepted_idea_records = []
        self._failed_feedback = []
        self._final_attempts = []
        self._final_context_events = []
        self._partial_proofs = []
        self._final_cycle_packets = []
        self._current_final_cycle_packet = None
        self._current_working_proof_attempt = None
        self.workflow_tasks = []
        self.completed_task_ids = set()
        self.current_task_id = None
        self._restored_from_disk = False
        self._master_proof_no_progress_count = 0
        self._last_master_proof_edit_signature = ""
        self._fatal_stop_reason = None
        self._fatal_stop_message = ""
        await self._broadcast("leanoj_cleared", self.get_status())

    async def skip_brainstorm(self) -> None:
        self._state.skip_brainstorm_requested = True
        await self._persist_and_broadcast("leanoj_skip_brainstorm_requested")

    async def force_brainstorm(self) -> None:
        self._state.force_brainstorm_requested = True
        await self._persist_and_broadcast("leanoj_force_brainstorm_requested")

    async def _consume_skip_brainstorm(self) -> bool:
        if not self._state.skip_brainstorm_requested:
            return False
        self._state.skip_brainstorm_requested = False
        self._state.current_path_decision = "solve_final_now"
        self._state.user_forced_final_cycle = True
        self._state.phase = "final_proof_loop"
        await self._persist_and_broadcast("leanoj_brainstorm_skipped")
        return True

    async def _consume_force_brainstorm(self) -> bool:
        if not self._state.force_brainstorm_requested:
            return False
        self._state.force_brainstorm_requested = False
        self._state.skip_brainstorm_requested = False
        self._state.user_forced_final_cycle = False
        self._state.current_path_decision = "need_more_brainstorming"
        # A user-forced recursive brainstorm is a fresh acceptance window.
        # Otherwise the recursive sufficiency modulo can reuse the prior cycle's
        # start count and fire before five new accepted brainstorms arrive.
        self._state.active_brainstorm_phase = ""
        self._state.active_brainstorm_start_count = self._state.brainstorm_acceptance_events
        await self._set_current_working_proof_attempt(
            trigger="user_force_brainstorm",
            requested_path="need_more_brainstorming",
            stuck_reason="User requested recursive brainstorming while preserving the current master proof draft.",
        )
        self._state.phase = "recursive_brainstorm"
        await self._persist_and_broadcast("leanoj_brainstorm_forced")
        return True

    async def _run_workflow(self, request: LeanOJStartRequest) -> None:
        if self._state.phase in {"idle", "initial_topic_candidates"}:
            selected_topic = await self._initial_topic_phase(request)
            if self._should_stop():
                return
            self._state.selected_topic = selected_topic

        force_brainstorm_consumed = await self._consume_force_brainstorm()
        if not force_brainstorm_consumed and (
            self._state.phase == "initial_brainstorm" or (
                self._state.phase == "initial_topic_candidates" and self._state.selected_topic
            )
        ):
            await self._initial_brainstorm_phase(request)

        if self._state.phase == "recursive_brainstorm":
            await self._recursive_brainstorm_phase(request)

        if self._state.phase == "proof_storm":
            # Legacy sessions may resume from the removed proof-only brainstorm path.
            # Continue with recursive brainstorming because verified proofs can now
            # be generated directly inside any brainstorm phase.
            await self._recursive_brainstorm_phase(request)

        if self._state.phase == "final_proof_loop":
            await self._final_proof_loop(request)

        while not self._should_stop() and self._state.phase != "verified":
            if await self._consume_force_brainstorm():
                continue

            if self._state.phase == "final_proof_loop" or self._state.user_forced_final_cycle:
                await self._final_proof_loop(request)
                continue

            if self._state.phase == "recursive_brainstorm":
                await self._recursive_brainstorm_phase(request)
                continue

            if self._state.phase == "proof_storm":
                await self._recursive_brainstorm_phase(request)
                continue

            decision = await self._path_decision_phase(request)
            if self._should_stop():
                return

            if await self._consume_force_brainstorm():
                continue

            if decision == "solve_final_now":
                await self._final_proof_loop(request)
            elif decision == "need_more_brainstorming":
                await self._recursive_brainstorm_phase(request)
            else:
                logger.warning("Unknown Proof Solver path decision %s; falling back to recursive brainstorming", decision)
                await self._recursive_brainstorm_phase(request)

    async def _initial_topic_phase(self, request: LeanOJStartRequest) -> str:
        self._state.phase = "initial_topic_candidates"
        await self._persist_and_broadcast("leanoj_phase_changed")

        if not await self._collect_initial_topics(request, target_topics=5):
            return "Direct Proof Solver proof solving from the user's template"

        if self._should_stop():
            return ""
        if not self._validated_topics:
            return "Direct Proof Solver proof solving from the user's template"

        selected_raw = await self._call_json(
            request.topic_generator,
            "leanoj_topic",
            "leanoj_topic_selector",
            build_topic_selection_prompt(request.user_prompt, request.lean_template, self._validated_topics),
        )
        selected = str(selected_raw.get("topic") or "").strip() or self._validated_topics[0]
        if not await self._validate_topic(request, selected):
            selected = self._validated_topics[0]

        await self._persist_and_broadcast("leanoj_initial_topic_selected", {"topic": selected})
        return selected

    async def _validate_topic(
        self,
        request: LeanOJStartRequest,
        topic: str,
        accepted_topics: Optional[list[str]] = None,
    ) -> bool:
        raw = await self._call_json(
            request.topic_validator,
            "leanoj_topic_val",
            "leanoj_topic_validator",
            build_topic_validation_prompt(
                request.user_prompt,
                request.lean_template,
                topic,
                accepted_topics if accepted_topics is not None else self._validated_topics,
            ),
        )
        accepted = str(raw.get("decision") or "").strip().lower() == "accept"
        self._last_brainstorm_validation_decisions = [
            {
                "accepted": accepted,
                "reasoning": str(raw.get("reasoning") or "").strip(),
                "summary": str(raw.get("summary") or "").strip(),
            }
        ]
        return accepted

    async def _validate_topic_batch(
        self,
        request: LeanOJStartRequest,
        topics: list[str],
        accepted_topics: Optional[list[str]] = None,
    ) -> list[bool]:
        if not topics:
            return []
        if len(topics) == 1:
            return [await self._validate_topic(request, topics[0], accepted_topics)]

        raw = await self._call_json(
            request.topic_validator,
            "leanoj_topic_val",
            "leanoj_topic_validator",
            build_topic_batch_validation_prompt(
                request.user_prompt,
                request.lean_template,
                topics,
                accepted_topics if accepted_topics is not None else self._validated_topics,
            ),
        )
        decisions = raw.get("decisions")
        if not isinstance(decisions, list) or len(decisions) != len(topics):
            logger.warning(
                "LeanOJ topic batch validator returned %s decisions for %s topics",
                len(decisions) if isinstance(decisions, list) else "non-list",
                len(topics),
            )
            return [False for _ in topics]

        accepted: list[bool] = []
        for expected_number, decision_payload in enumerate(decisions, start=1):
            if not isinstance(decision_payload, dict):
                accepted.append(False)
                continue
            if decision_payload.get("topic_number") != expected_number:
                logger.warning(
                    "LeanOJ topic batch validator returned out-of-order decision: expected %s, got %s",
                    expected_number,
                    decision_payload.get("topic_number"),
                )
                return [False for _ in topics]
            accepted.append(str(decision_payload.get("decision") or "").strip().lower() == "accept")
        return accepted

    async def _collect_initial_topics(self, request: LeanOJStartRequest, *, target_topics: int) -> bool:
        if self._state.skip_brainstorm_requested:
            await self._persist_and_broadcast("leanoj_brainstorm_skip_deferred")
            return False

        topic_queue: asyncio.Queue[_TopicCandidate] = asyncio.Queue(
            maxsize=max(3, len(request.brainstorm_submitters) * 2)
        )
        submitter_tasks = [
            asyncio.create_task(
                self._topic_submitter_loop(
                    request,
                    index,
                    submitter,
                    topic_queue,
                    target_topics=target_topics,
                )
            )
            for index, submitter in enumerate(request.brainstorm_submitters, start=1)
        ]
        logger.info(
            "LeanOJ initial topic submitters started (submitters=%s, target_topics=%s)",
            len(submitter_tasks),
            target_topics,
        )
        await self._broadcast(
            "leanoj_topic_submitters_started",
            {
                "submitter_count": len(submitter_tasks),
                "target_topics": target_topics,
            },
        )

        try:
            while len(self._validated_topics) < target_topics and not self._should_stop():
                if self._state.skip_brainstorm_requested:
                    await self._persist_and_broadcast("leanoj_brainstorm_skip_deferred")
                    return False

                remaining_topics = target_topics - len(self._validated_topics)
                batch = await self._dequeue_topic_batch(topic_queue, max_count=min(3, remaining_topics))
                if not batch:
                    if all(task.done() for task in submitter_tasks):
                        errors = [
                            task.exception()
                            for task in submitter_tasks
                            if task.done() and not task.cancelled() and task.exception() is not None
                        ]
                        if errors:
                            raise RuntimeError(f"All LeanOJ topic submitters stopped: {errors[0]}")
                        return bool(self._validated_topics)
                    continue

                topics = [topic for _, topic, _ in batch]
                logger.info(
                    "LeanOJ topic batch validation started (batch_size=%s, submitters=%s)",
                    len(batch),
                    [submitter_index for submitter_index, _, _ in batch],
                )
                await self._broadcast(
                    "leanoj_topic_batch_validation_started",
                    {
                        "batch_size": len(batch),
                        "submitters": [submitter_index for submitter_index, _, _ in batch],
                        "accepted_topics": len(self._validated_topics),
                        "target_topics": target_topics,
                    },
                )
                decisions = await self._validate_topic_batch(
                    request,
                    topics,
                    accepted_topics=list(self._validated_topics),
                )
                for (submitter_index, topic, metadata), accepted in zip(batch, decisions):
                    submitter_config = request.brainstorm_submitters[submitter_index - 1]
                    creativity_emphasized = bool((metadata or {}).get("creativity_emphasized"))
                    if accepted:
                        self._validated_topics.append(topic)
                        await self._persist_and_broadcast(
                            "leanoj_topic_validated",
                            {
                                "topic": topic,
                                "submitter": submitter_index,
                                "submitter_id": submitter_index,
                                "submitter_model": submitter_config.model_id,
                                "submitter_provider": submitter_config.provider,
                                "creativity_emphasized": creativity_emphasized,
                                "accepted_topics": len(self._validated_topics),
                                "target_topics": target_topics,
                            },
                        )
                    else:
                        await self._broadcast(
                            "leanoj_topic_rejected",
                            {
                                "topic": topic,
                                "submitter": submitter_index,
                                "submitter_id": submitter_index,
                                "submitter_model": submitter_config.model_id,
                                "submitter_provider": submitter_config.provider,
                                "creativity_emphasized": creativity_emphasized,
                                "accepted_topics": len(self._validated_topics),
                                "target_topics": target_topics,
                            },
                        )
            return bool(self._validated_topics)
        finally:
            for task in submitter_tasks:
                task.cancel()
            await asyncio.gather(*submitter_tasks, return_exceptions=True)

    async def _topic_submitter_loop(
        self,
        request: LeanOJStartRequest,
        submitter_index: int,
        submitter: LeanOJRoleConfig,
        topic_queue: asyncio.Queue[_TopicCandidate],
        *,
        target_topics: int,
    ) -> None:
        task_prefix = f"leanoj_topic_sub{submitter_index}"
        role_id = f"leanoj_topic_submitter_{submitter_index}"
        attempt = 0
        queued_count = 0
        while not self._should_stop():
            try:
                attempt += 1
                creativity_emphasized = (
                    request.creativity_emphasis_boost_enabled
                    and (queued_count + 1) % 5 == 0
                )
                topic_index = min(target_topics, len(self._validated_topics) + topic_queue.qsize() + 1)
                prompt = build_topic_candidate_prompt(
                    request.user_prompt,
                    request.lean_template,
                    self._validated_topics,
                    creativity_emphasized=creativity_emphasized,
                )
                if creativity_emphasized and not self._prompt_fits_role_budget(prompt, submitter):
                    logger.warning(
                        "LeanOJ topic submitter %s skipped creativity emphasis because prompt exceeded context budget.",
                        submitter_index,
                    )
                    creativity_emphasized = False
                    prompt = build_topic_candidate_prompt(
                        request.user_prompt,
                        request.lean_template,
                        self._validated_topics,
                        creativity_emphasized=False,
                    )
                await self._broadcast(
                    "leanoj_topic_generation_started",
                    {
                        "attempt": attempt,
                        "topic_index": topic_index,
                        "target_topics": target_topics,
                        "accepted_topics": len(self._validated_topics),
                        "submitter": submitter_index,
                        "submitter_id": submitter_index,
                        "submitter_model": submitter.model_id,
                        "submitter_provider": submitter.provider,
                        "creativity_emphasized": creativity_emphasized,
                    },
                )
                raw = await self._call_json(
                    submitter,
                    task_prefix,
                    role_id,
                    prompt,
                    temperature=api_client_manager.parallel_brainstorm_submitter_temperature(submitter_index),
                )

                topic = str(raw.get("topic") or "").strip()
                if not topic:
                    await self._broadcast(
                        "leanoj_topic_empty",
                        {
                            "attempt": attempt,
                            "submitter": submitter_index,
                            "submitter_id": submitter_index,
                        },
                    )
                    continue

                metadata = {"creativity_emphasized": creativity_emphasized}
                queued_count += 1
                await topic_queue.put((submitter_index, topic, metadata))
                await self._broadcast(
                    "leanoj_topic_candidate_queued",
                    {
                        "submitter": submitter_index,
                        "submitter_id": submitter_index,
                        "submitter_model": submitter.model_id,
                        "submitter_provider": submitter.provider,
                        "creativity_emphasized": creativity_emphasized,
                        "queue_size": topic_queue.qsize(),
                        "topic_preview": self._summarize_error(topic, limit=220),
                    },
                )
            except asyncio.CancelledError:
                raise
            except LeanOJConfigurationError as exc:
                if self._is_context_overflow_exception(exc):
                    await self._handle_context_overflow_stop(exc, role_id=role_id)
                    self._stop_event.set()
                raise
            except Exception as exc:
                logger.warning("LeanOJ topic submitter %s failed: %s", submitter_index, exc)
                await self._broadcast(
                    "leanoj_topic_submitter_failed",
                    {
                        "submitter": submitter_index,
                        "submitter_id": submitter_index,
                        "message": str(exc),
                    },
                )
                await asyncio.sleep(2)

    async def _initial_brainstorm_phase(self, request: LeanOJStartRequest) -> None:
        self._state.phase = "initial_brainstorm"
        self._begin_brainstorm_acceptance_phase("initial_brainstorm")
        await self._persist_and_broadcast("leanoj_phase_changed")
        await self._brainstorm_until_path_check(
            request,
            phase_key="initial_brainstorm",
            max_accepts=request.max_initial_brainstorm_accepts,
            sufficiency_interval=10,
            force_after_max=True,
        )

    async def _recursive_brainstorm_phase(self, request: LeanOJStartRequest) -> None:
        if await self._consume_force_brainstorm():
            return
        if await self._consume_skip_brainstorm():
            return

        resuming_recursive_phase = self._state.phase == "recursive_brainstorm"
        if not resuming_recursive_phase:
            self._state.recursive_cycle_count += 1
            self._state.active_brainstorm_phase = ""

        self._state.phase = "recursive_brainstorm"
        self._begin_brainstorm_acceptance_phase("recursive_brainstorm")
        await self._persist_and_broadcast("leanoj_phase_changed")
        accepted_at_phase_entry = self._state.brainstorm_acceptance_events
        logger.info(
            "LeanOJ recursive brainstorm cycle %s %s (accepted_events=%s)",
            self._state.recursive_cycle_count,
            "resumed" if resuming_recursive_phase else "started",
            accepted_at_phase_entry,
        )
        await self._persist_and_broadcast(
            "leanoj_recursive_brainstorm_started",
            {
                "cycle": self._state.recursive_cycle_count,
                "resumed": resuming_recursive_phase,
                "accepted_events": accepted_at_phase_entry,
            },
        )

        try:
            if await self._consume_skip_brainstorm():
                return
            await self._brainstorm_until_path_check(
                request,
                phase_key="recursive_brainstorm",
                max_accepts=request.max_recursive_brainstorm_accepts,
                sufficiency_interval=5,
                force_after_max=True,
            )
            if not self._should_stop():
                accepted_delta = self._state.brainstorm_acceptance_events - accepted_at_phase_entry
                logger.info(
                    "LeanOJ recursive brainstorm cycle %s completed (accepted_delta=%s, total_acceptances=%s)",
                    self._state.recursive_cycle_count,
                    accepted_delta,
                    self._state.accepted_brainstorm_count,
                )
                await self._persist_and_broadcast(
                    "leanoj_recursive_brainstorm_completed",
                    {
                        "cycle": self._state.recursive_cycle_count,
                        "accepted_delta": accepted_delta,
                        "total_acceptances": self._state.accepted_brainstorm_count,
                        "total_brainstorm_acceptance_events": self._state.brainstorm_acceptance_events,
                    },
                )
        finally:
            if not self._should_stop():
                self._clear_current_final_cycle_packet()

    async def _brainstorm_until_path_check(
        self,
        request: LeanOJStartRequest,
        *,
        phase_key: str = "initial_brainstorm",
        max_accepts: int,
        sufficiency_interval: int,
        force_after_max: bool,
    ) -> None:
        accepted_at_start = self._get_brainstorm_acceptance_start(phase_key)
        run_exit_review = False
        submission_queue = _LeanOJBrainstormSubmissionQueue(
            submitter_count=len(request.brainstorm_submitters)
        )
        submitter_tasks = [
            asyncio.create_task(
                self._brainstorm_submitter_loop(request, index, submitter, submission_queue)
            )
            for index, submitter in enumerate(request.brainstorm_submitters, start=1)
        ]
        logger.info(
            "LeanOJ brainstorm submitters started (phase=%s, submitters=%s, max_accepts=%s, accepted_at_start=%s)",
            phase_key,
            len(submitter_tasks),
            max_accepts,
            accepted_at_start,
        )
        await self._broadcast(
            "leanoj_brainstorm_submitters_started",
            {
                "phase": phase_key,
                "submitter_count": len(submitter_tasks),
                "max_accepts": max_accepts,
                "accepted_at_start": accepted_at_start,
            },
        )

        try:
            while not self._should_stop():
                if await self._consume_force_brainstorm():
                    run_exit_review = False
                    return
                if await self._consume_skip_brainstorm():
                    run_exit_review = False
                    return

                accepted_delta = self._state.brainstorm_acceptance_events - accepted_at_start
                if accepted_delta >= max_accepts and force_after_max:
                    run_exit_review = True
                    logger.info(
                        "LeanOJ brainstorm phase limit reached (phase=%s, accepted_delta=%s, max_accepts=%s)",
                        phase_key,
                        accepted_delta,
                        max_accepts,
                    )
                    await self._broadcast(
                        "leanoj_brainstorm_phase_limit_reached",
                        {
                            "phase": phase_key,
                            "accepted_delta": accepted_delta,
                            "max_accepts": max_accepts,
                            "total_acceptances": self._state.accepted_brainstorm_count,
                        },
                    )
                    self._finish_brainstorm_acceptance_phase_for_path_decision()
                    return
                if (
                    accepted_delta > 0
                    and accepted_delta % sufficiency_interval == 0
                    and self._state.brainstorm_acceptance_events
                    != self._state.active_brainstorm_last_sufficiency_check_count
                ):
                    self._state.active_brainstorm_last_sufficiency_check_count = (
                        self._state.brainstorm_acceptance_events
                    )
                    logger.info(
                        "LeanOJ brainstorm sufficiency check started (phase=%s, accepted_delta=%s)",
                        phase_key,
                        accepted_delta,
                    )
                    await self._broadcast(
                        "leanoj_sufficiency_check_started",
                        {
                            "phase": phase_key,
                            "accepted_delta": accepted_delta,
                            "total_acceptances": self._state.accepted_brainstorm_count,
                        },
                    )
                    enough = await self._sufficiency_check(request)
                    await self._persist_and_broadcast("leanoj_sufficiency_checked", {"enough": enough})
                    if enough:
                        run_exit_review = True
                        self._finish_brainstorm_acceptance_phase_for_path_decision()
                        return

                batch = await self._dequeue_brainstorm_batch(submission_queue)
                await self._sync_brainstorm_queue_pause_state(submission_queue, phase_key)
                if not batch:
                    if all(task.done() for task in submitter_tasks):
                        errors = [
                            task.exception()
                            for task in submitter_tasks
                            if task.done() and not task.cancelled() and task.exception() is not None
                        ]
                        if errors:
                            raise RuntimeError(f"All LeanOJ brainstorm submitters stopped: {errors[0]}")
                        run_exit_review = True
                        self._finish_brainstorm_acceptance_phase_for_path_decision()
                        return
                    continue

                submissions = [submission for _, submission, _ in batch]
                logger.info(
                    "LeanOJ brainstorm batch validation started (phase=%s, batch_size=%s, submitters=%s)",
                    phase_key,
                    len(batch),
                    [submitter_index for submitter_index, _, _ in batch],
                )
                await self._broadcast(
                    "leanoj_brainstorm_batch_validation_started",
                    {
                        "phase": phase_key,
                        "batch_size": len(batch),
                        "submitters": [submitter_index for submitter_index, _, _ in batch],
                    },
                )
                decisions = await self._validate_brainstorm_batch(request, submissions)
                validation_decisions = list(self._last_brainstorm_validation_decisions)
                for batch_index, ((submitter_index, submission, metadata), accepted) in enumerate(
                    zip(batch, decisions)
                ):
                    submitter_config = request.brainstorm_submitters[submitter_index - 1]
                    creativity_emphasized = bool((metadata or {}).get("creativity_emphasized"))
                    proof_payload = (metadata or {}).get("brainstorm_lean_proof")
                    lean_verified_proof = (
                        isinstance(proof_payload, dict)
                        and bool(str(proof_payload.get("theorem_statement") or "").strip())
                        and bool(str(proof_payload.get("lean_code") or "").strip())
                    )
                    accepted = accepted or lean_verified_proof
                    if accepted:
                        await self._record_accepted_brainstorm_proof(request, submitter_index, metadata)
                        validation_feedback = (
                            validation_decisions[batch_index]
                            if batch_index < len(validation_decisions)
                            else {}
                        )
                        self._record_accepted_brainstorm_idea(
                            submission,
                            submitter_index,
                            phase_key,
                            validation_feedback,
                            metadata,
                        )
                        self._state.accepted_brainstorm_count = len(self._accepted_ideas)
                        submission_preview = self._summarize_error(submission, limit=220)
                        logger.info(
                            "LeanOJ brainstorm ACCEPTED: Submitter %s [%s] (phase=%s, total_acceptances=%s, event=%s) - %s",
                            submitter_index,
                            submitter_config.model_id,
                            phase_key,
                            self._state.accepted_brainstorm_count,
                            self._state.brainstorm_acceptance_events,
                            submission_preview,
                        )
                        await self._persist_and_broadcast(
                            "leanoj_brainstorm_accepted",
                            {
                                "submitter": submitter_index,
                                "submitter_id": submitter_index,
                                "submitter_model": submitter_config.model_id,
                                "submitter_provider": submitter_config.provider,
                                "creativity_emphasized": creativity_emphasized,
                                "submission": submission,
                                "submission_preview": submission_preview,
                                "phase": phase_key,
                                "total_acceptances": self._state.accepted_brainstorm_count,
                                "total_brainstorm_acceptance_events": self._state.brainstorm_acceptance_events,
                            },
                        )
                        accepted_delta = self._state.brainstorm_acceptance_events - accepted_at_start
                        if (
                            accepted_delta > 0
                            and accepted_delta % 7 == 0
                            and self._state.brainstorm_acceptance_events
                            != self._state.active_brainstorm_last_prune_review_count
                        ):
                            self._state.active_brainstorm_last_prune_review_count = (
                                self._state.brainstorm_acceptance_events
                            )
                            await self._perform_brainstorm_prune_review(
                                request,
                                phase_key,
                                reason=f"scheduled review after {accepted_delta} accepted brainstorm events",
                            )
                        if (
                            force_after_max
                            and self._state.brainstorm_acceptance_events - accepted_at_start >= max_accepts
                        ):
                            run_exit_review = True
                            self._finish_brainstorm_acceptance_phase_for_path_decision()
                            return
                    else:
                        validation_feedback = (
                            validation_decisions[batch_index]
                            if batch_index < len(validation_decisions)
                            else {}
                        )
                        self._state.rejected_brainstorm_count += 1
                        self._record_brainstorm_rejection_feedback(
                            submitter_index,
                            submission,
                            validation_feedback,
                        )
                        submission_preview = self._summarize_error(submission, limit=220)
                        rejection_reason = self._summarize_error(
                            validation_feedback.get("summary")
                            or validation_feedback.get("reasoning")
                            or "Rejected by brainstorm validator.",
                            limit=220,
                        )
                        logger.info(
                            "LeanOJ brainstorm REJECTED: Submitter %s [%s] (phase=%s, total_rejections=%s) - %s",
                            submitter_index,
                            submitter_config.model_id,
                            phase_key,
                            self._state.rejected_brainstorm_count,
                            rejection_reason,
                        )
                        await self._persist_and_broadcast(
                            "leanoj_brainstorm_rejected",
                            {
                                "submitter": submitter_index,
                                "submitter_id": submitter_index,
                                "submitter_model": submitter_config.model_id,
                                "submitter_provider": submitter_config.provider,
                                "creativity_emphasized": creativity_emphasized,
                                "submission": submission,
                                "submission_preview": submission_preview,
                                "validator_reasoning": validation_feedback.get("reasoning", ""),
                                "validator_summary": validation_feedback.get("summary", ""),
                                "rejection_reason": rejection_reason,
                                "phase": phase_key,
                                "total_acceptances": self._state.accepted_brainstorm_count,
                                "total_rejections": self._state.rejected_brainstorm_count,
                            },
                        )
        finally:
            for task in submitter_tasks:
                task.cancel()
            await asyncio.gather(*submitter_tasks, return_exceptions=True)
            accepted_delta = self._state.brainstorm_acceptance_events - accepted_at_start
            if (
                run_exit_review
                and not self._should_stop()
                and accepted_delta > 0
                and self._state.brainstorm_acceptance_events
                != self._state.active_brainstorm_last_prune_review_count
            ):
                self._state.active_brainstorm_last_prune_review_count = (
                    self._state.brainstorm_acceptance_events
                )
                await self._perform_brainstorm_prune_review(
                    request,
                    phase_key,
                    reason=f"phase-exit review after {accepted_delta} accepted brainstorm events",
                )

    async def _wait_for_brainstorm_queue_turn(
        self,
        submission_queue: _LeanOJBrainstormSubmissionQueue,
        submitter_index: int,
    ) -> None:
        while not self._should_stop() and submission_queue.should_pause_submitter(submitter_index):
            await self._sync_brainstorm_queue_pause_state(
                submission_queue,
                self._state.active_brainstorm_phase or self._state.phase,
            )
            await asyncio.sleep(2)
        await self._sync_brainstorm_queue_pause_state(
            submission_queue,
            self._state.active_brainstorm_phase or self._state.phase,
        )

    async def _sync_brainstorm_queue_pause_state(
        self,
        submission_queue: _LeanOJBrainstormSubmissionQueue,
        phase_key: str,
    ) -> None:
        transitions = submission_queue.refresh_pause_transitions()
        queue_size = transitions["queue_size"]
        if transitions["global_changed"]:
            if transitions["global_paused"]:
                logger.info(
                    "LeanOJ brainstorm queue size (%s) >= threshold (%s). Pausing all submitters.",
                    queue_size,
                    system_config.queue_overflow_threshold,
                )
                await self._broadcast(
                    "leanoj_brainstorm_submitters_paused",
                    {
                        "phase": phase_key,
                        "queue_size": queue_size,
                        "threshold": system_config.queue_overflow_threshold,
                    },
                )
            else:
                logger.info(
                    "LeanOJ brainstorm queue size (%s) < threshold (%s). Resuming all submitters.",
                    queue_size,
                    system_config.queue_overflow_threshold,
                )
                await self._broadcast(
                    "leanoj_brainstorm_submitters_resumed",
                    {
                        "phase": phase_key,
                        "queue_size": queue_size,
                        "threshold": system_config.queue_overflow_threshold,
                    },
                )

        for paused_submitter in sorted(transitions["submitters_paused"]):
            pending_count = submission_queue.count_for_submitter(paused_submitter)
            logger.info(
                "LeanOJ brainstorm submitter %s paused for fairness (pending=%s, threshold=%s).",
                paused_submitter,
                pending_count,
                system_config.per_submitter_queue_threshold,
            )
            await self._broadcast(
                "leanoj_brainstorm_submitter_paused",
                {
                    "phase": phase_key,
                    "queue_size": queue_size,
                    "submitter": paused_submitter,
                    "submitter_id": paused_submitter,
                    "submitter_pending": pending_count,
                    "threshold": system_config.per_submitter_queue_threshold,
                },
            )

        for resumed_submitter in sorted(transitions["submitters_resumed"]):
            pending_count = submission_queue.count_for_submitter(resumed_submitter)
            logger.info(
                "LeanOJ brainstorm submitter %s resumed for fairness (pending=%s, threshold=%s).",
                resumed_submitter,
                pending_count,
                system_config.per_submitter_queue_threshold,
            )
            await self._broadcast(
                "leanoj_brainstorm_submitter_resumed",
                {
                    "phase": phase_key,
                    "queue_size": queue_size,
                    "submitter": resumed_submitter,
                    "submitter_id": resumed_submitter,
                    "submitter_pending": pending_count,
                    "threshold": system_config.per_submitter_queue_threshold,
                },
            )

    async def _brainstorm_submitter_loop(
        self,
        request: LeanOJStartRequest,
        submitter_index: int,
        submitter: LeanOJRoleConfig,
        submission_queue: _LeanOJBrainstormSubmissionQueue,
    ) -> None:
        task_prefix = f"leanoj_brainstorm_sub{submitter_index}"
        role_id = f"leanoj_brainstorm_submitter_{submitter_index}"
        queued_count = 0
        while not self._should_stop():
            try:
                await self._wait_for_brainstorm_queue_turn(submission_queue, submitter_index)
                if self._should_stop():
                    break
                creativity_emphasized = (
                    request.creativity_emphasis_boost_enabled
                    and (queued_count + 1) % 5 == 0
                )
                active_topic = self._active_brainstorm_topic()
                prompt_failed_feedback = self._general_brainstorm_feedback_records()
                task_request = (
                    "Generate one concrete proof-solving brainstorm idea for the active LeanOJ topic: "
                    f"{active_topic}"
                )
                allocation_task_request = (
                    f"{task_request}\n\n{CREATIVITY_EMPHASIS_BOOST_PROMPT}"
                    if creativity_emphasized
                    else task_request
                )
                context_blocks = await self._build_context_blocks(
                    request,
                    submitter,
                    mode="brainstorm",
                    task_request=allocation_task_request,
                    include_current_final_cycle_packet=True,
                    capped_rejection_feedback=self._format_capped_rejection_feedback(
                        "RECENT FAILED / REJECTION FEEDBACK SUMMARIES",
                        prompt_failed_feedback,
                        limit=10,
                    ),
                )
                prompt = build_brainstorm_prompt(
                    request.user_prompt,
                    request.lean_template,
                    active_topic,
                    self._accepted_ideas,
                    [item.model_dump(mode="json") for item in self._state.verified_subproofs],
                    prompt_failed_feedback,
                    context_blocks=context_blocks,
                    creativity_emphasized=creativity_emphasized,
                )
                if creativity_emphasized and not self._prompt_fits_role_budget(prompt, submitter):
                    logger.warning(
                        "LeanOJ brainstorm submitter %s skipped creativity emphasis because prompt exceeded context budget.",
                        submitter_index,
                    )
                    creativity_emphasized = False
                    context_blocks = await self._build_context_blocks(
                        request,
                        submitter,
                        mode="brainstorm",
                        task_request=task_request,
                        include_current_final_cycle_packet=True,
                        capped_rejection_feedback=self._format_capped_rejection_feedback(
                            "RECENT FAILED / REJECTION FEEDBACK SUMMARIES",
                            prompt_failed_feedback,
                            limit=10,
                        ),
                    )
                    prompt = build_brainstorm_prompt(
                        request.user_prompt,
                        request.lean_template,
                        active_topic,
                        self._accepted_ideas,
                        [item.model_dump(mode="json") for item in self._state.verified_subproofs],
                        prompt_failed_feedback,
                        context_blocks=context_blocks,
                        creativity_emphasized=False,
                    )
                raw = await self._call_json(
                    submitter,
                    task_prefix,
                    role_id,
                    prompt,
                    temperature=api_client_manager.parallel_brainstorm_submitter_temperature(submitter_index),
                )
                metadata: dict[str, Any] = {"creativity_emphasized": creativity_emphasized}
                if is_lean_proof_submission(raw):
                    source_context = "\n\n".join(
                        part
                        for part in [
                            request.lean_template,
                            active_topic,
                            "\n\n".join(self._accepted_ideas),
                            "\n\n".join(str(value) for value in context_blocks.values() if value),
                        ]
                        if part
                    )
                    gate_result = await verify_brainstorm_proof_candidate(
                        parsed=raw,
                        user_prompt=request.user_prompt,
                        source_context=source_context,
                        model_id=submitter.model_id,
                        role_id=role_id,
                        task_id_prefix=f"{task_prefix}_lean",
                        max_tokens=submitter.max_output_tokens,
                        validator_model=request.brainstorm_validator.model_id,
                        validator_context=request.brainstorm_validator.context_window,
                        validator_max_tokens=request.brainstorm_validator.max_output_tokens,
                        validator_role_id="leanoj_brainstorm_validator",
                        allowed_baseline=request.lean_template,
                        max_attempts=5,
                    )
                    if not gate_result.accepted:
                        feedback = {
                            "request": str(raw.get("theorem_statement") or raw.get("submission") or active_topic),
                            "error_summary": self._summarize_error(gate_result.failure_feedback, limit=1200),
                            "lean_code": gate_result.lean_code,
                        }
                        self._failed_feedback.append(feedback)
                        await self._persist_and_broadcast(
                            "leanoj_brainstorm_proof_failed",
                            {
                                "submitter": submitter_index,
                                "submitter_id": submitter_index,
                                "submitter_model": submitter.model_id,
                                "submitter_provider": submitter.provider,
                                "creativity_emphasized": creativity_emphasized,
                                "feedback": feedback,
                            },
                        )
                        continue
                    raw = {
                        **raw,
                        "submission": gate_result.submission_content,
                        "reasoning": gate_result.reasoning or raw.get("reasoning", ""),
                    }
                    metadata["brainstorm_lean_proof"] = {
                        "theorem_statement": gate_result.theorem_statement,
                        "theorem_name": gate_result.theorem_name,
                        "formal_sketch": gate_result.formal_sketch,
                        "expected_novelty_tier": gate_result.expected_novelty_tier,
                        "prompt_relevance_rationale": gate_result.prompt_relevance_rationale,
                        "novelty_rationale": gate_result.novelty_rationale,
                        "why_not_standard_known_result": gate_result.why_not_standard_known_result,
                        "lean_code": gate_result.lean_code,
                        "lean_feedback": gate_result.lean_feedback,
                        "reasoning": gate_result.reasoning,
                        "attempts": [
                            attempt.model_dump(mode="json")
                            for attempt in (gate_result.attempts or [])
                        ],
                        "attempt_count": len(gate_result.attempts or []),
                    }
                submission = str(raw.get("submission") or "").strip()
                if submission:
                    await self._wait_for_brainstorm_queue_turn(submission_queue, submitter_index)
                    if self._should_stop():
                        break
                    queued_count += 1
                    await submission_queue.put((submitter_index, submission, metadata))
                    await self._sync_brainstorm_queue_pause_state(
                        submission_queue,
                        self._state.active_brainstorm_phase or self._state.phase,
                    )
                    logger.info(
                        "LeanOJ brainstorm submission queued (phase=%s, submitter=%s, queue_size=%s)",
                        self._state.active_brainstorm_phase or self._state.phase,
                        submitter_index,
                        submission_queue.qsize(),
                    )
                    await self._broadcast(
                        "leanoj_brainstorm_submission_queued",
                        {
                            "phase": self._state.active_brainstorm_phase or self._state.phase,
                            "submitter": submitter_index,
                            "submitter_id": submitter_index,
                            "submitter_model": submitter.model_id,
                            "submitter_provider": submitter.provider,
                            "creativity_emphasized": creativity_emphasized,
                            "queue_size": submission_queue.qsize(),
                            "submission_preview": self._summarize_error(submission, limit=220),
                        },
                    )
            except asyncio.CancelledError:
                raise
            except LeanOJConfigurationError as exc:
                if self._is_context_overflow_exception(exc):
                    await self._handle_context_overflow_stop(exc, role_id=role_id)
                    self._stop_event.set()
                raise
            except Exception as exc:
                logger.warning("LeanOJ brainstorm submitter %s failed: %s", submitter_index, exc)
                await self._broadcast(
                    "leanoj_brainstorm_submitter_failed",
                    {"submitter": submitter_index, "message": str(exc)},
                )
                await asyncio.sleep(2)

    async def _dequeue_brainstorm_batch(
        self,
        submission_queue: _LeanOJBrainstormSubmissionQueue,
        *,
        max_count: int = 3,
    ) -> list[_BrainstormSubmission]:
        return await submission_queue.dequeue_batch(max_count=max_count)

    async def _dequeue_topic_batch(
        self,
        topic_queue: asyncio.Queue[_TopicCandidate],
        *,
        max_count: int = 3,
    ) -> list[_TopicCandidate]:
        try:
            first = await asyncio.wait_for(topic_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return []

        batch = [first]
        deadline = time.monotonic() + 0.25
        while len(batch) < max_count:
            item = None
            with contextlib.suppress(asyncio.QueueEmpty):
                item = topic_queue.get_nowait()
            if item is not None:
                batch.append(item)
                continue

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(topic_queue.get(), timeout=remaining))
            except asyncio.TimeoutError:
                break
        return batch

    def _topic_validation_context(self) -> list[str]:
        topics: list[str] = []
        seen: set[str] = set()
        for topic in self._validated_topics:
            normalized = topic.strip()
            if not normalized or normalized in seen:
                continue
            topics.append(normalized)
            seen.add(normalized)
        return topics

    async def _record_accepted_brainstorm_proof(
        self,
        request: LeanOJStartRequest,
        submitter_index: int,
        metadata: dict[str, Any],
    ) -> None:
        proof_payload = (metadata or {}).get("brainstorm_lean_proof")
        if not isinstance(proof_payload, dict):
            return

        theorem_statement = str(proof_payload.get("theorem_statement") or "").strip()
        lean_code = str(proof_payload.get("lean_code") or "").strip()
        if not theorem_statement or not lean_code:
            return

        subproof_id = f"brainstorm_proof_{self._state.brainstorm_acceptance_events + 1}_{uuid.uuid4().hex[:6]}"
        lean_feedback = str(proof_payload.get("lean_feedback") or "").strip()
        proof_attempts = [
            item if isinstance(item, ProofAttemptFeedback) else ProofAttemptFeedback.model_validate(item)
            for item in (proof_payload.get("attempts") or [])
        ]
        proof_record: Optional[ProofRecord] = None
        try:
            proof_record = await self._register_verified_leanoj_proof(
                request,
                proof_kind="subproof",
                theorem_statement=theorem_statement,
                theorem_name=str(proof_payload.get("theorem_name") or subproof_id),
                lean_code=lean_code,
                attempt_count=int(proof_payload.get("attempt_count") or 1),
                formal_sketch=str(proof_payload.get("formal_sketch") or "LeanOJ brainstorm proof candidate"),
                theorem_id=subproof_id,
                source_title=f"LeanOJ brainstorm proof from submitter {submitter_index}",
                verification_notes=(
                    lean_feedback
                    or "Proof Solver verified this brainstorm subproof with Lean 4 and template/device checks."
                ),
                attempts=proof_attempts,
            )
        except Exception as exc:
            logger.warning("LeanOJ accepted brainstorm proof registration failed: %s", exc, exc_info=True)
            await self._broadcast(
                "leanoj_brainstorm_proof_registration_failed",
                {
                    "subproof_id": subproof_id,
                    "submitter": submitter_index,
                    "error": str(exc),
                },
            )

        record = LeanOJSubproofRecord(
            subproof_id=subproof_id,
            request=theorem_statement,
            role="Verified during brainstorm before validator acceptance.",
            theorem_or_lemma=str(proof_payload.get("theorem_name") or theorem_statement),
            verified=True,
            lean_code=lean_code,
            lean_feedback=lean_feedback,
            attempts_used=int(proof_payload.get("attempt_count") or 1),
            proof_id=proof_record.proof_id if proof_record else "",
            novel=proof_record.novel if proof_record else False,
            novelty_tier=proof_record.novelty_tier if proof_record else "not_novel",
            novelty_reasoning=proof_record.novelty_reasoning if proof_record else "",
        )
        self._state.verified_subproofs.append(record)
        await self._persist_and_broadcast(
            "leanoj_brainstorm_proof_verified",
            {
                "subproof": record.model_dump(mode="json"),
                "submitter": submitter_index,
                "submitter_id": submitter_index,
            },
        )

    async def _validate_brainstorm(self, request: LeanOJStartRequest, submission: str) -> bool:
        raw = await self._call_json(
            request.brainstorm_validator,
            "leanoj_brainstorm_val",
            "leanoj_brainstorm_validator",
            build_brainstorm_validation_prompt(
                request.user_prompt,
                request.lean_template,
                submission,
                self._accepted_ideas,
                context_blocks=await self._build_context_blocks(
                    request,
                    request.brainstorm_validator,
                    mode="brainstorm",
                    task_request="Validate whether a LeanOJ brainstorm submission is useful and non-redundant.",
                    include_current_final_cycle_packet=True,
                ),
            ),
        )
        accepted = str(raw.get("decision") or "").strip().lower() == "accept"
        self._last_brainstorm_validation_decisions = [
            {
                "accepted": accepted,
                "context_role": self._normalize_brainstorm_context_role(raw, submission),
                "reasoning": str(raw.get("reasoning") or "").strip(),
                "summary": str(raw.get("summary") or "").strip(),
            }
        ]
        return accepted

    async def _validate_brainstorm_batch(self, request: LeanOJStartRequest, submissions: list[str]) -> list[bool]:
        if not submissions:
            self._last_brainstorm_validation_decisions = []
            return []
        if len(submissions) == 1:
            return [await self._validate_brainstorm(request, submissions[0])]

        raw = await self._call_json(
            request.brainstorm_validator,
            "leanoj_brainstorm_val",
            "leanoj_brainstorm_validator",
            build_brainstorm_batch_validation_prompt(
                request.user_prompt,
                request.lean_template,
                submissions,
                self._accepted_ideas,
                context_blocks=await self._build_context_blocks(
                    request,
                    request.brainstorm_validator,
                    mode="brainstorm",
                    task_request="Batch-validate LeanOJ brainstorm submissions for usefulness and redundancy.",
                    include_current_final_cycle_packet=True,
                ),
            ),
        )
        decisions = raw.get("decisions")
        if not isinstance(decisions, list) or len(decisions) != len(submissions):
            logger.warning(
                "LeanOJ brainstorm batch validator returned %s decisions for %s submissions",
                len(decisions) if isinstance(decisions, list) else "non-list",
                len(submissions),
            )
            self._last_brainstorm_validation_decisions = [
                {
                    "accepted": False,
                    "reasoning": "Brainstorm validator returned malformed decision payload.",
                    "summary": "Validator did not return one ordered decision per submission.",
                }
                for _ in submissions
            ]
            return [False for _ in submissions]

        accepted: list[bool] = []
        validation_decisions: list[dict[str, Any]] = []
        for expected_number, decision_payload in enumerate(decisions, start=1):
            if not isinstance(decision_payload, dict):
                accepted.append(False)
                validation_decisions.append(
                    {
                        "accepted": False,
                        "reasoning": "Decision payload was not an object.",
                        "summary": "Validator returned a malformed decision entry.",
                    }
                )
                continue
            if decision_payload.get("submission_number") != expected_number:
                logger.warning(
                    "LeanOJ brainstorm batch validator returned out-of-order decision: expected %s, got %s",
                    expected_number,
                    decision_payload.get("submission_number"),
                )
                self._last_brainstorm_validation_decisions = [
                    {
                        "accepted": False,
                        "reasoning": "Brainstorm validator returned out-of-order decisions.",
                        "summary": "Validator decisions could not be matched to submissions.",
                    }
                    for _ in submissions
                ]
                return [False for _ in submissions]
            is_accepted = str(decision_payload.get("decision") or "").strip().lower() == "accept"
            accepted.append(is_accepted)
            validation_decisions.append(
                {
                    "accepted": is_accepted,
                    "context_role": self._normalize_brainstorm_context_role(decision_payload, submissions[expected_number - 1]),
                    "reasoning": str(decision_payload.get("reasoning") or "").strip(),
                    "summary": str(decision_payload.get("summary") or "").strip(),
                }
            )
        self._last_brainstorm_validation_decisions = validation_decisions
        return accepted

    def _record_brainstorm_rejection_feedback(
        self,
        submitter_index: int,
        submission: str,
        validation_feedback: dict[str, Any],
    ) -> None:
        summary = str(validation_feedback.get("summary") or "").strip()
        reasoning = str(validation_feedback.get("reasoning") or "").strip()
        feedback_parts = [
            "VALIDATOR REJECTED BRAINSTORM SUBMISSION",
            f"Summary: {summary}" if summary else "",
            f"Reasoning: {reasoning}" if reasoning else "",
            f"Rejected submission: {self._summarize_error(submission, limit=500)}",
        ]
        error_summary = "\n".join(part for part in feedback_parts if part)
        self._failed_feedback.append(
            {
                "request": f"brainstorm submitter {submitter_index} rejected submission",
                "error_summary": self._summarize_error(error_summary, limit=1200),
                "submission": self._summarize_error(submission, limit=500),
                "submitter_index": submitter_index,
                "source": "brainstorm_validator",
            }
        )

    def _record_accepted_brainstorm_idea(
        self,
        submission: str,
        submitter_index: int,
        phase_key: str,
        validation_feedback: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        validation_feedback = validation_feedback or {}
        metadata = metadata or {}
        context_role = self._normalize_brainstorm_context_role(validation_feedback, submission)
        self._accepted_ideas.append(submission)
        self._state.brainstorm_acceptance_events += 1
        self._accepted_idea_records.append(
            {
                "content": submission,
                "context_role": context_role,
                "submitter_index": submitter_index,
                "phase": phase_key,
                "validator_summary": str(validation_feedback.get("summary") or "").strip(),
                "validator_reasoning": str(validation_feedback.get("reasoning") or "").strip(),
                "creativity_emphasized": bool(metadata.get("creativity_emphasized")),
                "created_at": datetime.now().isoformat(),
                "acceptance_event": self._state.brainstorm_acceptance_events,
            }
        )

    def _ensure_accepted_idea_records(self) -> None:
        existing = [
            dict(record)
            for record in self._accepted_idea_records
            if isinstance(record, dict) and str(record.get("content") or "").strip()
        ]
        used_existing_indices: set[int] = set()
        existing_by_content: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        for record_index, record in enumerate(existing):
            existing_by_content.setdefault(str(record.get("content") or ""), []).append((record_index, record))

        def take_existing_record(content: str) -> dict[str, Any] | None:
            candidates = existing_by_content.get(content, [])
            while candidates:
                record_index, record = candidates.pop(0)
                if record_index not in used_existing_indices:
                    used_existing_indices.add(record_index)
                    return dict(record)
            return None

        records: list[dict[str, Any]] = []
        for index, idea in enumerate(self._accepted_ideas):
            content = str(idea)
            aligned_record = (
                existing[index]
                if index < len(existing)
                and str(existing[index].get("content") or "") == content
                and index not in used_existing_indices
                else None
            )
            if aligned_record is not None:
                used_existing_indices.add(index)
                record = dict(aligned_record)
            else:
                record = take_existing_record(content) or {}
            if not record:
                record = {
                    "content": content,
                    "submitter_index": 1,
                    "phase": "legacy",
                    "created_at": "",
                    "acceptance_event": index + 1,
                    "legacy": True,
                }
            record["content"] = content
            record["context_role"] = self._normalize_brainstorm_context_role(record, idea)
            records.append(record)
        self._accepted_idea_records = records

    @staticmethod
    def _normalize_brainstorm_context_role(record: dict[str, Any] | None, text: str = "") -> str:
        role = str((record or {}).get("context_role") or "").strip().lower()
        if role in _LEANOJ_CONTEXT_ROLES:
            return role

        combined = " ".join(
            part.lower()
            for part in [
                text,
                str((record or {}).get("content") or ""),
                str((record or {}).get("summary") or ""),
                str((record or {}).get("reasoning") or ""),
                str((record or {}).get("validator_summary") or ""),
                str((record or {}).get("validator_reasoning") or ""),
            ]
            if part
        )
        if any(term in combined for term in _LEANOJ_REFUTED_CONTEXT_TERMS):
            return "refuted_construction"
        if "[lean 4 verified brainstorm proof]" in combined:
            return "verified_hint"
        if any(term in combined for term in _LEANOJ_ACTIVE_PLAN_CONTEXT_TERMS):
            return "active_plan"
        return "scratch"

    def _final_solver_active_plan_items(self) -> list[str]:
        self._ensure_accepted_idea_records()
        return [
            str(record.get("content") or "").strip()
            for record in self._accepted_idea_records
            if str(record.get("context_role") or "") in _LEANOJ_FINAL_ACTIVE_CONTEXT_ROLES
            and str(record.get("content") or "").strip()
        ]

    def _final_solver_refuted_construction_records(self) -> list[dict[str, Any]]:
        self._ensure_accepted_idea_records()
        accepted_refutations = [
            record
            for record in self._accepted_idea_records
            if str(record.get("context_role") or "") == "refuted_construction"
        ]
        verified_refutations = [
            {
                "content": record.get("request", record.get("theorem_or_lemma", "")),
                "reasoning": record.get("theorem_or_lemma", record.get("role", "")),
                "source": "verified_subproof",
            }
            for record in self._verified_subproof_dicts()
            if self._record_mentions_refuted_construction(record)
        ]
        failure_refutations = [
            {
                "content": record.get("error_summary", record.get("summary", "")),
                "reasoning": record.get("reasoning", record.get("lean_feedback", "")),
                "source": record.get("request", "failure feedback"),
            }
            for record in [*self._failed_feedback, *self._failed_context_dicts(), *self._final_attempts]
            if self._record_mentions_refuted_construction(record)
        ]
        return self._dedupe_dict_records([*accepted_refutations, *verified_refutations, *failure_refutations])

    def _final_solver_verified_subproof_dicts(self) -> list[dict[str, Any]]:
        return [
            record
            for record in self._verified_subproof_dicts()
            if not self._record_mentions_refuted_construction(record)
        ]

    @staticmethod
    def _record_mentions_refuted_construction(record: dict[str, Any]) -> bool:
        combined = " ".join(
            str(record.get(key) or "").lower()
            for key in (
                "request",
                "theorem_or_lemma",
                "role",
                "error_summary",
                "summary",
                "reasoning",
                "lean_feedback",
                "submission",
            )
        )
        return any(term in combined for term in _LEANOJ_REFUTED_CONTEXT_TERMS)

    def _active_brainstorm_topic(self, phase_key: str = "") -> str:
        phase = phase_key or self._state.phase
        if phase == "recursive_brainstorm":
            if self._current_working_proof_attempt:
                summary = _remove_attempt_count_language(
                    self._current_working_proof_attempt.get("summary") or ""
                ).strip()
                base = "Repair and complete the current Proof Solver master proof attempt."
                return f"{base} {summary}".strip() if summary else base
            return "Continue the recursive Proof Solver brainstorm from the current proof state and accepted proof memory."
        if phase == "initial_brainstorm":
            return self._state.selected_topic or "Solve the user's Proof Solver template."
        return self._state.selected_topic or "Solve the user's Proof Solver template."

    def _select_brainstorm_prune_reviewer(
        self,
        request: LeanOJStartRequest,
        phase_key: str,
    ) -> tuple[LeanOJRoleConfig, int]:
        self._ensure_accepted_idea_records()
        phase_records = [
            record
            for record in self._accepted_idea_records
            if str(record.get("phase") or "") == phase_key
        ]
        submitter_index = 1
        if phase_records:
            try:
                submitter_index = int(phase_records[-1].get("submitter_index") or 1)
            except (TypeError, ValueError):
                submitter_index = 1
        submitter_index = max(1, min(submitter_index, len(request.brainstorm_submitters)))
        return request.brainstorm_submitters[submitter_index - 1], submitter_index

    async def _perform_brainstorm_prune_review(
        self,
        request: LeanOJStartRequest,
        phase_key: str,
        *,
        reason: str,
    ) -> None:
        if not self._accepted_ideas:
            return
        self._state.brainstorm_prune_reviews_performed += 1
        reviewer, reviewer_index = self._select_brainstorm_prune_reviewer(request, phase_key)
        active_topic = self._active_brainstorm_topic(phase_key)
        try:
            context_blocks = await self._build_context_blocks(
                request,
                reviewer,
                mode="brainstorm",
                task_request=f"Review LeanOJ brainstorm memory for one conservative prune operation: {reason}.",
                include_current_final_cycle_packet=True,
            )
            raw = await self._call_json(
                reviewer,
                "leanoj_brainstorm_prune",
                f"leanoj_brainstorm_prune_reviewer_{reviewer_index}",
                build_brainstorm_prune_review_prompt(
                    request.user_prompt,
                    request.lean_template,
                    active_topic,
                    self._accepted_ideas,
                    context_blocks=context_blocks,
                ),
            )
            operation = self._normalize_brainstorm_prune_operation(raw)
            if operation["action"] == "none":
                await self._persist_and_broadcast(
                    "leanoj_brainstorm_prune_review_complete",
                    {"action": "none", "reason": reason, "reviewer": reviewer_index},
                )
                return

            validator_context = await self._build_context_blocks(
                request,
                request.brainstorm_validator,
                mode="brainstorm",
                task_request="Validate one proposed LeanOJ brainstorm prune operation.",
                include_current_final_cycle_packet=True,
            )
            validation = await self._call_json(
                request.brainstorm_validator,
                "leanoj_brainstorm_prune_val",
                "leanoj_brainstorm_validator",
                build_brainstorm_prune_validation_prompt(
                    request.user_prompt,
                    request.lean_template,
                    active_topic,
                    self._accepted_ideas,
                    operation,
                    context_blocks=validator_context,
                ),
            )
            if str(validation.get("decision") or "").strip().lower() != "accept":
                await self._persist_and_broadcast(
                    "leanoj_brainstorm_prune_rejected",
                    {
                        "operation": operation,
                        "reasoning": validation.get("reasoning", ""),
                        "reviewer": reviewer_index,
                    },
                )
                return
            applied = self._apply_brainstorm_prune_operation(operation, reviewer_index, phase_key)
            await self._persist_and_broadcast(
                "leanoj_brainstorm_prune_applied" if applied else "leanoj_brainstorm_prune_apply_failed",
                {
                    "operation": operation,
                    "reasoning": validation.get("reasoning", ""),
                    "reviewer": reviewer_index,
                },
            )
        except asyncio.CancelledError:
            raise
        except LeanOJConfigurationError:
            raise
        except Exception as exc:
            logger.warning("LeanOJ brainstorm prune review failed: %s", exc, exc_info=True)
            await self._persist_and_broadcast(
                "leanoj_brainstorm_prune_error",
                {"message": str(exc), "reason": reason},
            )

    def _normalize_brainstorm_prune_operation(self, raw: dict[str, Any]) -> dict[str, Any]:
        action = str(raw.get("action") or "none").strip().lower()
        if action not in {"none", "delete", "edit", "add"}:
            action = "none"
        idea_index: Optional[int] = None
        try:
            if raw.get("idea_index") is not None:
                idea_index = int(raw.get("idea_index"))
        except (TypeError, ValueError):
            idea_index = None
        new_content = str(raw.get("new_content") or "").strip()
        reasoning = str(raw.get("reasoning") or "").strip()
        if action in {"delete", "edit"} and (idea_index is None or idea_index < 1 or idea_index > len(self._accepted_ideas)):
            action = "none"
            reasoning = f"Invalid idea_index for prune operation. {reasoning}".strip()
        if action in {"edit", "add"} and not new_content:
            action = "none"
            reasoning = f"Missing new_content for prune operation. {reasoning}".strip()
        return {
            "action": action,
            "idea_index": idea_index,
            "new_content": new_content,
            "reasoning": reasoning,
        }

    def _apply_brainstorm_prune_operation(self, operation: dict[str, Any], reviewer_index: int, phase_key: str) -> bool:
        self._ensure_accepted_idea_records()
        action = operation["action"]
        idea_index = operation.get("idea_index")
        if action == "delete":
            if not isinstance(idea_index, int) or idea_index < 1 or idea_index > len(self._accepted_ideas):
                return False
            del self._accepted_ideas[idea_index - 1]
            del self._accepted_idea_records[idea_index - 1]
        elif action == "edit":
            if not isinstance(idea_index, int) or idea_index < 1 or idea_index > len(self._accepted_ideas):
                return False
            self._accepted_ideas[idea_index - 1] = operation["new_content"]
            self._accepted_idea_records[idea_index - 1]["content"] = operation["new_content"]
            self._accepted_idea_records[idea_index - 1]["context_role"] = self._normalize_brainstorm_context_role(
                {"reasoning": operation.get("reasoning", "")},
                operation["new_content"],
            )
            self._accepted_idea_records[idea_index - 1]["edited_at"] = datetime.now().isoformat()
            self._accepted_idea_records[idea_index - 1]["edit_reasoning"] = operation.get("reasoning", "")
        elif action == "add":
            self._accepted_ideas.append(operation["new_content"])
            self._accepted_idea_records.append(
                {
                    "content": operation["new_content"],
                    "context_role": self._normalize_brainstorm_context_role(
                        {"reasoning": operation.get("reasoning", "")},
                        operation["new_content"],
                    ),
                    "submitter_index": reviewer_index,
                    "phase": phase_key,
                    "created_at": datetime.now().isoformat(),
                    "acceptance_event": self._state.brainstorm_acceptance_events,
                    "prune_add": True,
                    "reasoning": operation.get("reasoning", ""),
                }
            )
        else:
            return False
        self._state.accepted_brainstorm_count = len(self._accepted_ideas)
        self._state.brainstorm_prune_operations_applied += 1
        return True

    async def _sufficiency_check(self, request: LeanOJStartRequest) -> bool:
        raw = await self._call_json(
            request.brainstorm_validator,
            "leanoj_sufficiency",
            "leanoj_brainstorm_validator",
            build_sufficiency_prompt(
                request.user_prompt,
                request.lean_template,
                self._accepted_ideas,
                [item.model_dump(mode="json") for item in self._state.verified_subproofs],
                context_blocks=await self._build_context_blocks(
                    request,
                    request.brainstorm_validator,
                    mode="brainstorm",
                    task_request="Decide whether the accumulated Proof Solver context is sufficient for the final loop.",
                    include_current_final_cycle_packet=True,
                ),
            ),
        )
        return bool(raw.get("enough"))

    async def _path_decision_phase(self, request: LeanOJStartRequest) -> str:
        self._state.phase = "path_decision"
        await self._persist_and_broadcast("leanoj_phase_changed")
        decision_actor, decision_role_id = self._path_decision_actor(request)
        prompt_failed_feedback = self._general_brainstorm_feedback_records()
        raw = await self._call_json(
            decision_actor,
            "leanoj_path",
            decision_role_id,
            build_path_decision_prompt(
                request.user_prompt,
                request.lean_template,
                self._accepted_ideas,
                [item.model_dump(mode="json") for item in self._state.verified_subproofs],
                prompt_failed_feedback,
                context_blocks=await self._build_context_blocks(
                    request,
                    decision_actor,
                    mode="brainstorm",
                    task_request="Choose the next LeanOJ path after reviewing accumulated proof memory.",
                    include_current_final_cycle_packet=True,
                    capped_rejection_feedback=self._format_capped_rejection_feedback(
                        "RECENT FAILED / REJECTION FEEDBACK SUMMARIES",
                        prompt_failed_feedback,
                        limit=10,
                    ),
                ),
            ),
        )
        decision = str(raw.get("path") or "").strip()
        if decision not in _LEANOJ_PATH_OPTIONS_SET:
            decision = "need_more_brainstorming"
        path_valid, corrected_path = await self._validate_path_decision(request, decision, str(raw.get("reasoning") or ""))
        if not path_valid:
            decision = corrected_path or "need_more_brainstorming"
        self._state.current_path_decision = decision
        await self._persist_and_broadcast("leanoj_path_decided", {"decision": decision, "reasoning": raw.get("reasoning", "")})
        return decision

    @staticmethod
    def _path_decision_actor(
        request: LeanOJStartRequest,
        valid_paths: tuple[str, ...] = _LEANOJ_PATH_OPTIONS,
    ) -> tuple[LeanOJRoleConfig, str]:
        if "solve_final_now" in valid_paths:
            return request.final_solver, "leanoj_final_solver"
        return request.topic_generator, "leanoj_topic_generator"

    async def _validate_path_decision(self, request: LeanOJStartRequest, decision: str, reasoning: str) -> tuple[bool, str]:
        raw = await self._call_json(
            request.topic_validator,
            "leanoj_path_val",
            "leanoj_path_validator",
            build_path_validation_prompt(
                request.user_prompt,
                request.lean_template,
                decision,
                reasoning,
                self._accepted_ideas,
                [item.model_dump(mode="json") for item in self._state.verified_subproofs],
                context_blocks=await self._build_context_blocks(
                    request,
                    request.topic_validator,
                    mode="brainstorm",
                    task_request="Validate the proposed LeanOJ path decision.",
                    include_current_final_cycle_packet=True,
                ),
            ),
        )
        accepted = str(raw.get("decision") or "").strip().lower() == "accept"
        corrected_path = str(raw.get("corrected_path") or "").strip()
        if corrected_path not in _LEANOJ_PATH_OPTIONS_SET:
            corrected_path = ""
        await self._persist_and_broadcast(
            "leanoj_path_validated",
            {
                "decision": decision,
                "validated": accepted,
                "corrected_path": corrected_path,
                "reasoning": raw.get("reasoning", ""),
            },
        )
        return accepted, corrected_path

    async def _register_verified_leanoj_proof(
        self,
        request: LeanOJStartRequest,
        *,
        proof_kind: str,
        theorem_statement: str,
        theorem_name: str,
        lean_code: str,
        attempt_count: int,
        formal_sketch: str = "",
        theorem_id: str = "",
        source_title: str = "",
        verification_notes: str = "",
        attempts: Optional[list[ProofAttemptFeedback]] = None,
    ) -> Optional[ProofRecord]:
        """Register a Proof Solver verified proof in the shared proof database."""
        if not request.topic_validator.model_id:
            raise LeanOJConfigurationError("Proof Solver proof novelty validator model is unavailable")

        source_type = "leanoj_final" if proof_kind == "final" else "leanoj_subproof"
        task_id = self._next_task_id(f"leanoj_{proof_kind}_novelty")
        self.current_task_id = task_id
        self._refresh_workflow_tasks(f"leanoj_{proof_kind}_novelty", "Proof Novelty Validator")
        api_client_manager.set_autonomous_phase(self._state.phase or "leanoj")
        try:
            # Lazy import avoids pulling autonomous coordinator into LeanOJ module load.
            from backend.autonomous.core.proof_registration import register_verified_lean_proof

            registration = await register_verified_lean_proof(
                proof_database=proof_database,
                user_prompt=request.user_prompt,
                theorem_statement=theorem_statement,
                lean_code=lean_code,
                validator_model=request.topic_validator.model_id,
                validator_context=request.topic_validator.context_window,
                validator_max_tokens=request.topic_validator.max_output_tokens,
                task_id=task_id,
                role_id="leanoj_proof_novelty",
                source_type=source_type,
                source_id=self._state.session_id,
                source_title=source_title or self._state.selected_topic or request.user_prompt,
                theorem_id=theorem_id,
                theorem_name=theorem_name,
                formal_sketch=formal_sketch,
                solver="Proof Solver",
                verification_notes=(
                    verification_notes
                    or "Proof Solver verified this proof with Lean 4 and template/device checks."
                ),
                attempt_count=attempt_count,
                attempts=attempts,
                broadcast_fn=self._broadcast,
                base_event={
                    "source_type": source_type,
                    "source_id": self._state.session_id,
                    "source_title": source_title or self._state.selected_topic or request.user_prompt,
                    "trigger": "leanoj_verified",
                },
            )
            self.completed_task_ids.add(task_id)
            return registration.record
        except Exception as exc:
            logger.warning("Proof Solver proof registration failed for %s: %s", proof_kind, exc)
            raise
        finally:
            self.current_task_id = None
            self._refresh_workflow_tasks(f"leanoj_{proof_kind}_novelty", "Proof Novelty Validator")

    async def _check_proof_and_capture_partial(
        self,
        request: LeanOJStartRequest,
        lean_code: str,
        *,
        target: str,
        attempt_number: int,
        proof_request: str,
        reasoning: str,
        theorem_or_lemma: str = "",
    ) -> Lean4Result:
        placeholder_tokens = self._placeholder_tokens(lean_code)
        if not placeholder_tokens:
            return await get_lean4_client().check_proof(lean_code, timeout=system_config.lean4_proof_timeout)

        lean_result = await get_lean4_client().check_proof(
            lean_code,
            timeout=system_config.lean4_proof_timeout,
            allow_placeholders=True,
        )
        if not lean_result.success:
            return lean_result

        device_error = self._validate_no_new_declaration_devices(
            request.lean_template,
            lean_code,
            target=f"partial {target}",
        )
        if device_error:
            return Lean4Result(
                success=False,
                error_output=device_error,
                goal_states=lean_result.goal_states,
                raw_stderr=lean_result.raw_stderr,
            )
        if target == "final":
            template_error = self._validate_final_solution_integrity(
                request.lean_template,
                lean_code,
            )
            if template_error:
                return Lean4Result(
                    success=False,
                    error_output=template_error,
                    goal_states=lean_result.goal_states,
                    raw_stderr=lean_result.raw_stderr,
                )

        partial_record = {
            "session_id": self._state.session_id,
            "attempt": attempt_number,
            "target": target,
            "request": proof_request,
            "theorem_or_lemma": theorem_or_lemma,
            "placeholder_tokens": sorted(set(placeholder_tokens)),
            "lean_code": lean_code,
            "reasoning": reasoning,
            "high_value_scaffold": False,
            "master_seed_eligible": False,
            "created_at": datetime.now().isoformat(),
            "summary": (
                "Lean accepted this incomplete scaffold with placeholders. "
                "It is stored for future reference, but it is not a verified proof and is not eligible "
                "to seed the master proof unless a validator explicitly marks it high-value."
            ),
        }
        await self._record_partial_proof(partial_record)
        return Lean4Result(
            success=False,
            error_output=(
                "PARTIAL PROOF SAVED: Lean accepted this scaffold with placeholder token(s) "
                f"{', '.join(partial_record['placeholder_tokens'])}. It has been stored in the "
                "LeanOJ partial-proof database for future reference, but final verification must "
                "continue until every `sorry`/`admit` is replaced by a complete proof."
            ),
            goal_states=lean_result.goal_states,
            raw_stderr=lean_result.raw_stderr,
        )

    async def _record_partial_proof(self, partial_record: dict[str, Any]) -> None:
        self._partial_proofs.append(partial_record)
        await self._append_partial_proof_database(partial_record)
        await self._persist_and_broadcast(
            "leanoj_partial_proof_saved",
            {"partial_proof": partial_record},
        )

    async def _append_partial_proof_database(self, partial_record: dict[str, Any]) -> None:
        path = self._partial_proof_database_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(path, "a", encoding="utf-8") as f:
            await f.write(json.dumps(partial_record, ensure_ascii=False) + "\n")
        await leanoj_context_manager.append_record(
            self._state.session_id,
            ARTIFACT_PARTIAL_PROOFS,
            partial_record,
        )

    @staticmethod
    def _placeholder_tokens(lean_code: str) -> list[str]:
        stripped = strip_lean_comments_and_strings(lean_code or "")
        return [match.group(1) for match in _LEAN_PLACEHOLDER_RE.finditer(stripped)]

    @staticmethod
    def _partial_proofs_base_dir() -> Path:
        return Path(system_config.data_dir) / "leanoj_partial_proofs"

    def _partial_proof_database_path(self, session_id: str = "") -> Path:
        return self._partial_proofs_base_dir() / f"{session_id or self._state.session_id or 'latest'}.jsonl"

    def _load_partial_proof_database(self, session_id: str) -> list[dict[str, Any]]:
        path = self._partial_proof_database_path(session_id)
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    records.append(item)
        except Exception as exc:
            logger.warning("Failed to load LeanOJ partial proof database from %s: %s", path, exc)
        return records

    @staticmethod
    def _dedupe_partial_proofs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str, str, str]] = set()
        for record in records:
            key = (
                str(record.get("session_id") or ""),
                str(record.get("target") or ""),
                str(record.get("attempt") or ""),
                str(record.get("request") or ""),
                str(record.get("lean_code") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(record)
        return deduped

    @staticmethod
    def _dedupe_strings(records: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for record in records:
            value = str(record).strip()
            if not value or value in seen:
                continue
            seen.add(value)
            deduped.append(value)
        return deduped

    @staticmethod
    def _dedupe_dict_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for record in records:
            key = LeanOJCoordinator._dict_record_key(record)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(record)
        return deduped

    @staticmethod
    def _dict_record_key(record: dict[str, Any]) -> str:
        try:
            return json.dumps(record, sort_keys=True, default=str)
        except TypeError:
            return str(record)

    def _verified_subproof_dicts(self) -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in self._state.verified_subproofs]

    def _failed_context_dicts(self) -> list[dict[str, Any]]:
        return self._dedupe_dict_records(
            [
                *[item.model_dump(mode="json") for item in self._state.failed_subproofs],
                *self._failed_feedback,
            ]
        )

    @staticmethod
    def _is_subproof_or_final_failure_feedback(record: dict[str, Any]) -> bool:
        request = str(record.get("request") or "").lower()
        return bool(record.get("lean_code")) or "subproof" in request or "final proof solver" in request

    def _general_brainstorm_feedback_records(self) -> list[dict[str, Any]]:
        if self._state.phase == "recursive_brainstorm":
            return []
        return [
            record
            for record in self._failed_feedback
            if isinstance(record, dict) and not self._is_subproof_or_final_failure_feedback(record)
        ]

    async def _build_context_blocks(
        self,
        request: LeanOJStartRequest,
        role_config: LeanOJRoleConfig,
        *,
        mode: str,
        task_request: str,
        include_current_final_cycle_packet: bool = False,
        capped_rejection_feedback: str = "",
        context_scope: str = "",
    ) -> dict[str, str]:
        resolved_scope = context_scope or self._infer_context_scope(mode)
        current_packet = self._current_final_cycle_packet if include_current_final_cycle_packet else None
        working_proof_attempt = None
        if resolved_scope == "recursive_brainstorm":
            working_proof_attempt = await self._working_proof_attempt_context_packet()
            capped_rejection_feedback = ""
        include_failed_subproofs = resolved_scope == "subproof"
        accepted_context = (
            self._final_solver_active_plan_items()
            if resolved_scope == "final_solver"
            else self._accepted_ideas
        )
        refuted_constructions = (
            self._final_solver_refuted_construction_records()
            if resolved_scope == "final_solver"
            else []
        )
        allocation = await leanoj_context_manager.allocate_context(
            session_id=self._state.session_id,
            mode=resolved_scope,
            user_prompt=request.user_prompt,
            lean_template=request.lean_template,
            task_request=task_request,
            context_window=role_config.context_window,
            max_output_tokens=role_config.max_output_tokens,
            accepted_ideas=accepted_context,
            verified_subproofs=(
                self._final_solver_verified_subproof_dicts()
                if resolved_scope == "final_solver"
                else self._verified_subproof_dicts()
            ),
            partial_proofs=self._partial_proofs,
            failed_subproofs=self._failed_context_dicts() if include_failed_subproofs else [],
            final_attempts=self._final_attempts[-5:] if resolved_scope == "final_solver" else [],
            final_cycle_packets=[],
            refuted_constructions=refuted_constructions,
            current_final_cycle_packet=current_packet,
            current_working_proof_attempt=working_proof_attempt,
            capped_rejection_feedback=capped_rejection_feedback,
        )
        context_blocks = allocation.as_prompt_blocks()
        if resolved_scope == "final_solver":
            proof_search_context = await self._build_final_solver_proof_search_context(
                request=request,
                task_request=task_request,
            )
            if proof_search_context:
                context_blocks["proof_search_context"] = proof_search_context
        return context_blocks

    async def _build_final_solver_proof_search_context(
        self,
        *,
        request: LeanOJStartRequest,
        task_request: str,
    ) -> str:
        """Return compact optional proof-search context for the LeanOJ final solver."""
        self._pending_final_solver_assistant_target_hash = ""
        master_proof = await self._read_master_proof()
        snapshot = AssistantTargetSnapshot(
            workflow_mode="leanoj",
            target_kind="final_solver",
            user_prompt=request.user_prompt,
            target_statement=request.lean_template,
            lean_template=request.lean_template,
            formal_sketch=master_proof[-4000:],
            rejection_feedback="\n".join(
                str(item.get("feedback") or item.get("reasoning") or item)
                for item in self._final_solver_failure_window()[-5:]
            ),
            accepted_solver_summary="\n".join(self._final_solver_active_plan_items()[-8:]),
            source_title="LeanOJ final proof solver",
            source_type="leanoj",
            source_id=self._state.session_id,
            imports=["Mathlib"],
        )
        target_hash = assistant_proof_search_coordinator.submit_target(snapshot)
        pack = assistant_proof_search_coordinator.get_latest_pack(target_hash)
        if not pack:
            return ""
        formatted = pack.to_prompt_context(max_code_chars_per_result=0)
        if count_tokens(formatted) > _LEANOJ_PROOF_SEARCH_MAX_TOKENS:
            return "[Assistant proof-support results omitted because they exceeded the final-solver optional context budget.]"
        if pack.results:
            self._pending_final_solver_assistant_target_hash = target_hash
        return formatted

    @staticmethod
    def _format_final_solver_proof_search_results(records: list[dict[str, Any]]) -> str:
        lines = [
            "[Optional retrieved proof context. Use only as verified-helper/proof-pattern guidance for the current LeanOJ template. Do not paste unrelated proofs or build a known-knowledge library.]",
        ]
        for index, record in enumerate(records[:7], start=1):
            source = " ".join(
                str(part).strip()
                for part in [
                    record.get("corpus"),
                    record.get("corpus_scope") or record.get("release_id"),
                ]
                if str(part or "").strip()
            )
            lines.extend(
                [
                    "",
                    f"Result {index}",
                    f"Source: {source or '[unknown]'}",
                    f"Source kind: {record.get('source_kind') or '[unknown]'}",
                    f"Proof ID: {record.get('proof_id') or '[none]'}",
                    f"Fingerprint: {record.get('fingerprint') or '[none]'}",
                    f"Theorem: {record.get('theorem_name') or record.get('display_title') or '[unnamed]'}",
                    f"Statement: {record.get('theorem_statement') or '[missing]'}",
                    f"Description: {record.get('proof_description') or record.get('formal_sketch') or '[none]'}",
                    f"Imports: {', '.join(record.get('imports') or []) or '[none]'}",
                    f"Dependencies: {', '.join(record.get('dependency_names') or []) or '[none]'}",
                    f"Theorem statement hash: {record.get('theorem_statement_hash') or '[none]'}",
                    f"Lean code hash: {record.get('lean_code_hash') or '[none]'}",
                    f"Canonical URI: {record.get('canonical_uri') or '[none]'}",
                ]
            )
        return "\n".join(lines)

    def _infer_context_scope(self, mode: str) -> str:
        if mode == "final_solver":
            return "final_solver"
        if mode == "subproof":
            return "subproof"
        if self._state.phase == "recursive_brainstorm" or self._current_working_proof_attempt:
            return "recursive_brainstorm"
        return "brainstorm"

    async def _set_current_working_proof_attempt(
        self,
        *,
        trigger: str,
        requested_path: str,
        stuck_reason: str,
    ) -> None:
        master_proof = await self._read_master_proof()
        if master_proof:
            self._set_master_proof_metadata(master_proof)
        prompt_safe_stuck_reason = _remove_attempt_count_language(
            stuck_reason or self._state.master_proof_last_stuck_reason or "Final proof needs more context."
        )
        summary = self._summarize_error(
            f"{trigger}: {prompt_safe_stuck_reason}",
            limit=500,
        )
        self._current_working_proof_attempt = {
            "session_id": self._state.session_id,
            "trigger": trigger,
            "requested_path": requested_path,
            "stuck_reason": self._summarize_error(prompt_safe_stuck_reason, limit=1200),
            "summary": summary,
            "master_proof_version": self._state.master_proof_version,
            "master_proof_hash": self._state.master_proof_hash,
            "master_proof_line_count": self._state.master_proof_line_count,
            "master_proof_char_count": self._state.master_proof_char_count,
            "master_proof_last_edit_summary": self._state.master_proof_last_edit_summary,
            "created_at": datetime.now().isoformat(),
        }

    async def _working_proof_attempt_context_packet(self) -> Optional[dict[str, Any]]:
        if not self._current_working_proof_attempt:
            return None
        master_proof = await self._read_master_proof()
        if master_proof:
            self._set_master_proof_metadata(master_proof)
        old_attempt_before_redo = await self._read_master_proof_old_attempt_before_redo()
        packet = dict(self._current_working_proof_attempt)
        packet.update(
            {
                "master_proof": master_proof,
                "master_proof_version": self._state.master_proof_version,
                "master_proof_hash": self._state.master_proof_hash,
                "master_proof_line_count": self._state.master_proof_line_count,
                "master_proof_char_count": self._state.master_proof_char_count,
                "master_proof_last_edit_summary": self._state.master_proof_last_edit_summary,
                "old_attempt_before_redo": old_attempt_before_redo,
                "old_attempt_before_redo_version": self._state.master_proof_old_attempt_before_redo_version,
                "old_attempt_before_redo_hash": self._state.master_proof_old_attempt_before_redo_hash,
                "old_attempt_before_redo_line_count": self._state.master_proof_old_attempt_before_redo_line_count,
                "old_attempt_before_redo_char_count": self._state.master_proof_old_attempt_before_redo_char_count,
                "old_attempt_before_redo_summary": self._state.master_proof_old_attempt_before_redo_summary,
                "old_attempt_before_redo_validator_justification": (
                    self._state.master_proof_old_attempt_before_redo_validator_justification
                ),
                "old_attempt_before_redo_apparent_issue": (
                    self._state.master_proof_old_attempt_before_redo_apparent_issue
                ),
                "recent_final_attempts": leanoj_context_manager._format_attempts(self._final_attempts[-10:]),
                "verified_subproofs": self._verified_subproof_dicts(),
                "partial_final_proofs": [
                    proof for proof in self._partial_proofs[-10:] if str(proof.get("target") or "") == "final"
                ],
            }
        )
        return packet

    def _clear_current_final_cycle_packet(self) -> None:
        """Clear one-shot direct final-cycle context after its next phase has completed."""
        self._current_final_cycle_packet = None

    @staticmethod
    def _format_capped_rejection_feedback(
        title: str,
        records: list[dict[str, Any]],
        *,
        limit: int,
    ) -> str:
        visible = [record for record in records[-limit:] if isinstance(record, dict)]
        if not visible:
            return ""
        lines = [title]
        for index, record in enumerate(visible, start=1):
            lines.append(
                f"{index}. {_remove_attempt_count_language(record.get('request', 'proof feedback'))} :: "
                f"{_remove_attempt_count_language(record.get('error_summary', record.get('error_output', '')))}"
            )
            lean_feedback = _remove_attempt_count_language(record.get("lean_feedback") or "")
            if lean_feedback:
                lines.append(f"   Lean feedback: {lean_feedback}")
        return "\n".join(lines)

    @staticmethod
    def _is_final_prompt_feedback_safe(record: dict[str, Any]) -> bool:
        text = "\n".join(
            str(record.get(key) or "")
            for key in ("request", "error_summary", "error_output", "lean_feedback", "reasoning")
        ).lower()
        if not text.strip():
            return False
        blocked_terms = (
            "brainstorm",
            "need_more_brainstorming",
            "stuck_needs_brainstorm",
            "final proof solver proof cycle",
            "failed-attempt count",
            "failed attempts",
        )
        if not any(term in text for term in blocked_terms):
            return True
        concrete_terms = (
            "old_string",
            "unexpected token",
            "missing cases",
            "unsolved goals",
            "error:",
            "rejected",
            "invalid",
            "json",
            "max_tokens",
            "lean",
            "verification",
            "watchdog",
        )
        return any(term in text for term in concrete_terms)

    def _record_final_context_event(
        self,
        event_type: str,
        *,
        request: str,
        error_summary: str = "",
        lean_feedback: str = "",
        reasoning: str = "",
    ) -> None:
        record = {
            "event_type": event_type,
            "request": self._summarize_error(request, limit=300),
            "error_summary": self._summarize_error(error_summary, limit=1200),
            "lean_feedback": self._summarize_error(lean_feedback, limit=1200),
            "reasoning": self._summarize_error(reasoning, limit=800),
            "created_at": datetime.now().isoformat(),
        }
        self._final_context_events.append(record)
        self._final_context_events = self._final_context_events[-50:]

    def _final_solver_failure_window(self) -> list[dict[str, Any]]:
        recent_events = [
            event
            for event in self._final_context_events[-5:]
            if isinstance(event, dict)
        ]
        return [
            event
            for event in recent_events
            if event.get("event_type") == "failure" and self._is_final_prompt_feedback_safe(event)
        ]

    def _master_proof_path(self, session_id: str = "") -> Path:
        resolved_session_id = session_id or self._state.session_id or "latest"
        return self._sessions_base_dir() / resolved_session_id / "master_proof.lean"

    def _master_proof_old_attempt_before_redo_path(self, session_id: str = "") -> Path:
        resolved_session_id = session_id or self._state.session_id or "latest"
        return self._sessions_base_dir() / resolved_session_id / "master_proof_old_attempt_before_redo.lean"

    def _master_proof_edit_log_path(self, session_id: str = "") -> Path:
        resolved_session_id = session_id or self._state.session_id or "latest"
        return self._sessions_base_dir() / resolved_session_id / "master_proof_edits.jsonl"

    def _master_proof_snapshot_log_path(self, session_id: str = "") -> Path:
        resolved_session_id = session_id or self._state.session_id or "latest"
        return self._sessions_base_dir() / resolved_session_id / "master_proof_snapshots.jsonl"

    @staticmethod
    def _hash_master_proof(content: str) -> str:
        return hashlib.sha256((content or "").encode("utf-8")).hexdigest() if content else ""

    def _set_master_proof_metadata(
        self,
        content: str,
        *,
        summary: str = "",
        increment_version: bool = False,
    ) -> None:
        if increment_version:
            self._state.master_proof_version += 1
        self._state.master_proof_initialized = bool((content or "").strip())
        self._state.master_proof_hash = self._hash_master_proof(content)
        self._state.master_proof_char_count = len(content or "")
        self._state.master_proof_line_count = len((content or "").splitlines()) if content else 0
        if summary:
            self._state.master_proof_last_edit_summary = self._summarize_error(summary, limit=500)

    async def _read_master_proof(self) -> str:
        path = self._master_proof_path()
        if not path.exists():
            return ""
        try:
            async with aiofiles.open(path, "r", encoding="utf-8") as f:
                return await f.read()
        except Exception as exc:
            logger.warning("Failed to read Proof Solver master proof from %s: %s", path, exc)
            return ""

    async def _write_master_proof(self, content: str, *, summary: str = "") -> None:
        path = self._master_proof_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(content or "")
        self._set_master_proof_metadata(content or "", summary=summary, increment_version=True)

    async def _read_master_proof_old_attempt_before_redo(self) -> str:
        path = self._master_proof_old_attempt_before_redo_path()
        if not path.exists():
            return ""
        try:
            async with aiofiles.open(path, "r", encoding="utf-8") as f:
                return await f.read()
        except Exception as exc:
            logger.warning("Failed to read Proof Solver old attempt before redo from %s: %s", path, exc)
            return ""

    async def _write_master_proof_old_attempt_before_redo(self, content: str) -> None:
        path = self._master_proof_old_attempt_before_redo_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(content or "")

    async def _append_master_proof_edit(self, record: dict[str, Any]) -> None:
        path = self._master_proof_edit_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "session_id": self._state.session_id,
            "master_proof_version": self._state.master_proof_version,
            "created_at": datetime.now().isoformat(),
            **record,
        }
        async with aiofiles.open(path, "a", encoding="utf-8") as f:
            await f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        await self._compact_master_proof_edit_log_if_needed()

    async def get_master_proof_draft(self) -> dict[str, Any]:
        content = await self._read_master_proof()
        if content:
            self._set_master_proof_metadata(content)
        return {
            "session_id": self._state.session_id,
            "exists": bool(content.strip()),
            "content": content,
            "metadata": self._master_proof_metadata_payload(),
        }

    async def get_master_proof_edit_summaries(self, *, limit: int = 50) -> dict[str, Any]:
        safe_limit = max(1, min(500, int(limit or 50)))
        records = self._read_master_proof_edit_records()
        visible = records[-safe_limit:]
        return {
            "session_id": self._state.session_id,
            "total_edits": len(records),
            "limit": safe_limit,
            "edits": [self._summarize_master_proof_edit_record(record) for record in visible],
            "metadata": self._master_proof_metadata_payload(),
        }

    def _master_proof_metadata_payload(self) -> dict[str, Any]:
        return {
            "initialized": self._state.master_proof_initialized,
            "version": self._state.master_proof_version,
            "sha256": self._state.master_proof_hash,
            "line_count": self._state.master_proof_line_count,
            "char_count": self._state.master_proof_char_count,
            "last_edit_summary": self._state.master_proof_last_edit_summary,
            "last_stuck_reason": self._state.master_proof_last_stuck_reason,
        }

    def _read_master_proof_edit_records(self, session_id: str = "") -> list[dict[str, Any]]:
        path = self._master_proof_edit_log_path(session_id)
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    records.append(item)
        except Exception as exc:
            logger.warning("Failed to read Proof Solver master proof edit log %s: %s", path, exc)
        return records

    def _summarize_master_proof_edit_record(self, record: dict[str, Any]) -> dict[str, Any]:
        summary_keys = [
            "session_id",
            "master_proof_version",
            "created_at",
            "action",
            "operation",
            "accepted",
            "needs_more_time",
            "requested_path",
            "master_proof_hash",
            "master_proof_line_count",
            "master_proof_char_count",
        ]
        summary = {key: record.get(key) for key in summary_keys if key in record}
        for key in ("reasoning", "stuck_reason", "error_summary", "validator_feedback", "validator_reasoning"):
            if record.get(key):
                summary[key] = self._summarize_error(str(record.get(key)), limit=500)
        if isinstance(record.get("shortening_metrics"), dict):
            summary["shortening_metrics"] = record.get("shortening_metrics")
        if record.get("old_string"):
            summary["old_string_preview"] = self._summarize_error(str(record.get("old_string")), limit=240)
        if record.get("new_string"):
            summary["new_string_preview"] = self._summarize_error(str(record.get("new_string")), limit=240)
            summary["new_string_char_count"] = len(str(record.get("new_string") or ""))
        return summary

    async def _compact_master_proof_edit_log_if_needed(self) -> None:
        path = self._master_proof_edit_log_path()
        records = self._read_master_proof_edit_records()
        if len(records) <= _MASTER_PROOF_EDIT_LOG_COMPACT_RECORD_LIMIT:
            return

        keep_count = min(_MASTER_PROOF_EDIT_LOG_RECENT_RECORDS_TO_KEEP, len(records))
        retained = records[-keep_count:]
        compacted_count = len(records) - len(retained)
        current_proof = await self._read_master_proof()
        snapshot = {
            "session_id": self._state.session_id,
            "created_at": datetime.now().isoformat(),
            "snapshot_kind": "master_proof_edit_log_compaction",
            "compacted_edit_count": compacted_count,
            "retained_edit_count": len(retained),
            "master_proof_version": self._state.master_proof_version,
            "master_proof_hash": self._hash_master_proof(current_proof),
            "master_proof_line_count": len(current_proof.splitlines()) if current_proof else 0,
            "master_proof_char_count": len(current_proof or ""),
            "first_compacted_edit": self._summarize_master_proof_edit_record(records[0]),
            "last_compacted_edit": self._summarize_master_proof_edit_record(records[compacted_count - 1]),
        }
        snapshot_path = self._master_proof_snapshot_log_path()
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(snapshot_path, "a", encoding="utf-8") as f:
            await f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
        await self._write_jsonl_records(path, retained)

    @staticmethod
    async def _write_jsonl_records(path: Path, records: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            for record in records:
                await f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _select_master_proof_seed(self, request: LeanOJStartRequest) -> str:
        for proof in reversed(self._partial_proofs):
            if str(proof.get("target") or "") != "final":
                continue
            lean_code = str(proof.get("lean_code") or "").strip()
            if (
                lean_code
                and self._is_high_value_master_seed_partial(request, proof, lean_code)
                and not self._validate_final_solution_integrity(request.lean_template, lean_code)
            ):
                return lean_code

        for attempt in reversed(self._final_attempts):
            lean_code = str(attempt.get("lean_code") or "").strip()
            if not lean_code:
                continue
            if not self._is_explicit_master_seed_candidate(request, attempt, lean_code):
                continue
            if lean_code and not self._validate_final_solution_integrity(request.lean_template, lean_code):
                return lean_code

        return request.lean_template.strip()

    def _is_high_value_master_seed_partial(
        self,
        request: LeanOJStartRequest,
        proof: dict[str, Any],
        lean_code: str,
    ) -> bool:
        """Only explicitly elevated partials may seed the durable master proof."""
        return self._is_explicit_master_seed_candidate(
            request,
            proof,
            lean_code,
            require_placeholders=True,
        )

    def _is_explicit_master_seed_candidate(
        self,
        request: LeanOJStartRequest,
        record: dict[str, Any],
        lean_code: str,
        *,
        require_placeholders: bool = False,
    ) -> bool:
        """Require an explicit validator/metadata signal before seeding from prior attempts."""
        if not (record.get("high_value_scaffold") is True or record.get("master_seed_eligible") is True):
            return False
        if not lean_code.strip():
            return False
        if require_placeholders and not self._placeholder_tokens(lean_code):
            return False
        normalized_code = self._normalize_lean_for_template_check(lean_code)
        normalized_template = self._normalize_lean_for_template_check(request.lean_template)
        if normalized_code == normalized_template:
            return False
        text = " ".join(
            str(record.get(key) or "").lower()
            for key in ("request", "reasoning", "error_summary")
        )
        blocked_terms = (
            "template unchanged",
            "minimal scaffold",
            "best achievable",
            "infeasible",
            "cannot be completed",
            "sorry placeholders",
        )
        return not any(term in text for term in blocked_terms)

    async def _ensure_master_proof_initialized(self, request: LeanOJStartRequest) -> str:
        current = await self._read_master_proof()
        if current.strip():
            self._set_master_proof_metadata(current)
            return current

        seed = self._select_master_proof_seed(request)
        await self._write_master_proof(seed, summary="Initialized Proof Solver master proof draft")
        await self._append_master_proof_edit(
            {
                "action": "initialize_master_proof",
                "operation": "full_content",
                "reasoning": "Seeded the durable master proof draft from existing Proof Solver context.",
                "new_string": seed,
            }
        )
        await self._persist_and_broadcast("leanoj_master_proof_initialized")
        return seed

    @staticmethod
    def _normalize_final_solver_edit(raw: dict[str, Any]) -> dict[str, Any]:
        if raw.get("lean_code") and not raw.get("action"):
            return {
                "action": "edit_proof",
                "operation": "full_content",
                "old_string": "",
                "new_string": str(raw.get("lean_code") or ""),
                "needs_more_time": False,
                "reasoning": str(raw.get("reasoning") or "Legacy whole-file final proof response."),
            }

        action = str(raw.get("action") or "").strip()
        if not action and raw.get("operation"):
            action = "edit_proof"
        needs_more_time = bool(raw.get("needs_more_time"))
        return {
            "action": action,
            "operation": str(raw.get("operation") or "").strip(),
            "old_string": str(raw.get("old_string") or ""),
            "new_string": str(raw.get("new_string") or ""),
            "needs_more_time": needs_more_time,
            "reasoning": str(raw.get("reasoning") or raw.get("summary") or "").strip(),
            "stuck_reason": str(raw.get("stuck_reason") or raw.get("reasoning") or "").strip(),
            "requested_path": str(raw.get("requested_path") or raw.get("path") or "").strip(),
        }

    def _apply_master_proof_edit(self, current_proof: str, edit: dict[str, Any]) -> tuple[Optional[str], str]:
        action = str(edit.get("action") or "").strip()
        if action not in _LEANOJ_PROOF_EDIT_ACTIONS:
            return None, (
                f"Invalid final solver action `{action}`. Final proof mode accepts only `edit_proof`; "
                "phase transitions are selected by the separate path-decision mode."
            )

        operation = str(edit.get("operation") or "").strip()
        old_string = str(edit.get("old_string") or "")
        new_string = str(edit.get("new_string") or "")
        if operation not in _LEANOJ_PROOF_EDIT_OPERATIONS:
            return None, (
                f"Invalid master proof edit operation `{operation}`. "
                "Use full_content, replace, insert_after, or delete."
            )

        if operation == "full_content":
            if not new_string.strip():
                return None, "full_content requires non-empty new_string Lean code."
            return new_string.strip(), ""

        if not current_proof.strip():
            return None, f"Master proof is empty; operation `{operation}` must be full_content."
        if not old_string:
            return None, f"Operation `{operation}` requires a non-empty old_string copied from the current master proof."
        match_count = current_proof.count(old_string)
        if match_count == 0:
            return None, "old_string was not found verbatim in the current master proof."
        if match_count > 1:
            return None, f"old_string appears {match_count} times in the current master proof; include more context."

        if operation == "replace":
            return current_proof.replace(old_string, new_string, 1), ""
        if operation == "insert_after":
            if not new_string.strip():
                return None, "insert_after requires non-empty new_string Lean code."
            insert_pos = current_proof.find(old_string) + len(old_string)
            return (
                current_proof[:insert_pos].rstrip()
                + "\n\n"
                + new_string.strip()
                + "\n\n"
                + current_proof[insert_pos:].lstrip()
            ), ""
        if operation == "delete":
            updated = current_proof.replace(old_string, "", 1)
            while "\n\n\n" in updated:
                updated = updated.replace("\n\n\n", "\n\n")
            return updated, ""

        return None, f"Unsupported master proof edit operation `{operation}`."

    @classmethod
    def _master_proof_shortening_metrics(cls, before_proof: str, after_proof: str) -> dict[str, Any]:
        before = before_proof or ""
        after = after_proof or ""
        before_chars = len(before)
        after_chars = len(after)
        before_lines = len(before.splitlines()) if before else 0
        after_lines = len(after.splitlines()) if after else 0
        before_placeholders = len(cls._placeholder_tokens(before))
        after_placeholders = len(cls._placeholder_tokens(after))
        return {
            "before_char_count": before_chars,
            "after_char_count": after_chars,
            "char_delta_removed": max(0, before_chars - after_chars),
            "before_line_count": before_lines,
            "after_line_count": after_lines,
            "line_delta_removed": max(0, before_lines - after_lines),
            "before_placeholder_count": before_placeholders,
            "after_placeholder_count": after_placeholders,
            "placeholder_delta_added": max(0, after_placeholders - before_placeholders),
            "after_to_before_char_ratio": round(after_chars / before_chars, 4) if before_chars else 1.0,
        }

    @staticmethod
    def _should_validate_master_proof_shortening_edit(edit: dict[str, Any], metrics: dict[str, Any]) -> bool:
        char_delta = int(metrics.get("char_delta_removed") or 0)
        line_delta = int(metrics.get("line_delta_removed") or 0)
        placeholder_delta = int(metrics.get("placeholder_delta_added") or 0)
        if char_delta <= 0:
            return False
        operation = str(edit.get("operation") or "").strip()
        return (
            line_delta > 0
            or char_delta >= _MASTER_PROOF_SHORTENING_CHAR_THRESHOLD
            or placeholder_delta > 0
            or operation == "delete"
        )

    async def _validate_master_proof_shortening_edit(
        self,
        request: LeanOJStartRequest,
        edit: dict[str, Any],
        before_proof: str,
        after_proof: str,
        metrics: dict[str, Any],
    ) -> tuple[bool, str, str, str, str]:
        await self._broadcast(
            "leanoj_master_proof_edit_validation_started",
            {
                "master_proof_version": self._state.master_proof_version,
                "operation": str(edit.get("operation") or ""),
                "char_delta_removed": metrics.get("char_delta_removed", 0),
                "line_delta_removed": metrics.get("line_delta_removed", 0),
            },
        )
        raw = await self._call_json(
            request.brainstorm_validator,
            "leanoj_master_proof_edit_val",
            "leanoj_master_proof_edit_validator",
            build_master_proof_edit_validation_prompt(
                request.user_prompt,
                request.lean_template,
                before_proof,
                after_proof,
                edit,
                metrics,
            ),
        )
        decision = str(raw.get("decision") or "").strip().lower()
        reasoning = str(raw.get("reasoning") or raw.get("summary") or "").strip()
        feedback = str(raw.get("feedback_to_submitter") or raw.get("summary") or reasoning).strip()
        approval_justification = str(
            raw.get("shortening_approval_justification")
            or raw.get("approval_justification")
            or reasoning
        ).strip()
        apparent_issue = str(
            raw.get("apparent_issue_with_old_attempt")
            or raw.get("old_attempt_apparent_issue")
            or raw.get("old_attempt_issue")
            or ""
        ).strip()
        if decision == "accept":
            accepted_reasoning = reasoning or "Master proof edit validator accepted the shortening as progressive."
            return (
                True,
                feedback,
                accepted_reasoning,
                approval_justification or accepted_reasoning,
                apparent_issue
                or "Validator judged the removed material redundant, superseded, or less progressive than the shorter edit.",
            )
        return (
            False,
            feedback or "Restore the deleted proof progress or replace it with an equivalent stronger proof before shortening.",
            reasoning or "Master proof edit validator rejected the shortening as non-progressive.",
            "",
            "",
        )

    def _build_master_proof_direct_context(
        self,
        master_proof: str,
        request: LeanOJStartRequest,
        context_blocks: dict[str, str] | None,
    ) -> tuple[str, dict[str, Any]]:
        proof = master_proof or request.lean_template
        proof_tokens = count_tokens(proof)
        available_input = rag_config.get_available_input_tokens(
            request.final_solver.context_window,
            request.final_solver.max_output_tokens,
        )
        nonproof_parts = [
            request.user_prompt,
            request.lean_template,
            "\n\n".join(str(value) for value in (context_blocks or {}).values() if value),
        ]
        nonproof_tokens = sum(count_tokens(part) for part in nonproof_parts)
        nonproof_tokens += rag_config.get_prompt_assembly_overhead_estimate() + 2500
        proof_token_budget = available_input - nonproof_tokens

        if proof_tokens > proof_token_budget:
            raise LeanOJConfigurationError(
                "PROOF SOLVER MANDATORY DIRECT CONTEXT OVERFLOW: The full master proof is mandatory direct-inject "
                "context and cannot be truncated, summarized, windowed, or RAG-substituted. "
                f"Full master proof tokens: {proof_tokens}. Available mandatory direct-inject proof budget after "
                f"user prompt, Lean template, proof memory, schema, and output reserve: {proof_token_budget}. "
                f"Configured final-solver context window: {request.final_solver.context_window}. "
                f"Configured final-solver max output tokens: {request.final_solver.max_output_tokens}. "
                "Increase the final solver context window or reduce other mandatory prompt context before resuming."
            )

        return proof, {
            "direct_context_mode": "full_mandatory",
            "master_proof_tokens": proof_tokens,
            "mandatory_direct_proof_token_budget": proof_token_budget,
        }

    @classmethod
    def _normalize_master_proof_for_progress(cls, content: str) -> str:
        return cls._normalize_lean_for_template_check(strip_lean_comments_and_strings(content or ""))

    def _record_master_proof_progress(
        self,
        edit: dict[str, Any],
        before_proof: str,
        after_proof: str,
    ) -> str:
        before_hash = self._hash_master_proof(before_proof)
        after_hash = self._hash_master_proof(after_proof)
        before_semantic = self._normalize_master_proof_for_progress(before_proof)
        after_semantic = self._normalize_master_proof_for_progress(after_proof)
        signature = self._master_proof_edit_signature(edit)
        no_hash_change = before_hash == after_hash
        no_semantic_change = before_semantic == after_semantic
        repeated_region = bool(signature and signature == self._last_master_proof_edit_signature)

        if no_hash_change or no_semantic_change or repeated_region:
            self._master_proof_no_progress_count += 1
        else:
            self._master_proof_no_progress_count = 0

        self._last_master_proof_edit_signature = signature

        if self._master_proof_no_progress_count < _MASTER_PROOF_NO_PROGRESS_LIMIT:
            return ""

        reason_parts = [
            f"LeanOJ final solver made {_MASTER_PROOF_NO_PROGRESS_LIMIT} consecutive edit-only steps",
        ]
        if no_semantic_change:
            reason_parts.append("without changing non-comment Lean code")
        elif no_hash_change:
            reason_parts.append("without changing the master proof hash")
        if repeated_region:
            reason_parts.append("while repeatedly editing/inserting at the same proof region")
        reason_parts.append("so the run is returning to recursive brainstorming for fresh context instead of looping indefinitely.")
        return "; ".join(reason_parts)

    def _reset_master_proof_progress_watchdog(self) -> None:
        self._master_proof_no_progress_count = 0
        self._last_master_proof_edit_signature = ""

    @classmethod
    def _master_proof_edit_signature(cls, edit: dict[str, Any]) -> str:
        operation = str(edit.get("operation") or "")
        old_string = str(edit.get("old_string") or "")
        if operation == "full_content":
            new_string = str(edit.get("new_string") or "")
            normalized_new = cls._normalize_master_proof_for_progress(new_string)
            return f"full_content:{hashlib.sha256(normalized_new[:1000].encode('utf-8')).hexdigest()}"
        if not old_string:
            return operation
        normalized_old = cls._normalize_master_proof_for_progress(old_string)
        return f"{operation}:{hashlib.sha256(normalized_old.encode('utf-8')).hexdigest()}"

    @staticmethod
    def _final_cycle_should_handoff_to_recursive(cycle_attempts: list[dict[str, Any]]) -> bool:
        if any(
            str(attempt.get("request") or "") == "final Proof Solver master proof progress watchdog"
            for attempt in cycle_attempts
        ):
            return True
        stale_edit_failures = sum(
            1
            for attempt in cycle_attempts
            if "old_string was not found verbatim" in str(attempt.get("error_summary") or "")
        )
        return stale_edit_failures >= _MASTER_PROOF_STALE_EDIT_FAILURE_HANDOFF_COUNT

    @staticmethod
    def _format_lean_success_feedback(lean_result: Lean4Result) -> str:
        diagnostics = str(getattr(lean_result, "diagnostic_output", "") or "").strip()
        if not diagnostics:
            diagnostics = str(getattr(lean_result, "raw_stderr", "") or "").strip()
        goal_states = str(getattr(lean_result, "goal_states", "") or "").strip()
        parts = []
        if diagnostics:
            parts.append(diagnostics)
        if goal_states:
            parts.append(f"Goal state output:\n{goal_states}")
        return "\n\n".join(parts).strip() or "Lean 4 accepted with no diagnostics."

    async def _review_final_solution_completion(
        self,
        request: LeanOJStartRequest,
        *,
        lean_code: str,
        final_solver_reasoning: str,
        lean_result: Lean4Result,
    ) -> tuple[bool, str, str]:
        lean_feedback = self._format_lean_success_feedback(lean_result)
        raw = await self._call_json(
            request.final_solver,
            "leanoj_final_review",
            "leanoj_final_solver",
            build_final_solution_review_prompt(
                request.user_prompt,
                request.lean_template,
                lean_code,
                final_solver_reasoning,
                lean_feedback,
            ),
        )
        raw_solved = raw.get("solved")
        solved = raw_solved if isinstance(raw_solved, bool) else str(raw_solved).strip().lower() == "true"
        reasoning = str(raw.get("reasoning") or raw.get("summary") or "").strip()
        continuation_feedback = str(raw.get("continuation_feedback") or "").strip()
        if solved:
            return True, reasoning or "Final solver review accepted the Lean-verified solution.", lean_feedback
        return (
            False,
            continuation_feedback or reasoning or "Final solver review rejected this Lean-accepted code as not complete.",
            lean_feedback,
        )

    async def _check_master_proof_edit_before_persist(
        self,
        request: LeanOJStartRequest,
        *,
        lean_code: str,
        needs_more_time: bool,
        attempt_number: int,
        reasoning: str,
        final_solver_metadata: dict[str, Any],
    ) -> tuple[Lean4Result, str]:
        if needs_more_time:
            lean_result = await get_lean4_client().check_proof(
                lean_code,
                timeout=system_config.lean4_proof_timeout,
                allow_placeholders=True,
            )
            lean_pass_feedback = self._format_lean_success_feedback(lean_result) if lean_result.success else ""
            if lean_result.success:
                template_error = self._validate_final_solution_integrity(
                    request.lean_template,
                    lean_code,
                )
                if template_error:
                    lean_result.success = False
                    lean_result.error_output = template_error
            return lean_result, lean_pass_feedback

        lean_result = await self._check_proof_and_capture_partial(
            request,
            lean_code,
            target="final",
            attempt_number=attempt_number,
            proof_request="final Proof Solver solution",
            reasoning=reasoning,
        )
        lean_pass_feedback = self._format_lean_success_feedback(lean_result) if lean_result.success else ""
        if lean_result.success:
            template_error = self._validate_final_solution_integrity(
                request.lean_template,
                lean_code,
            )
            if template_error:
                lean_result.success = False
                lean_result.error_output = template_error
            else:
                adequacy_error = self._validate_final_answer_adequacy(
                    request.lean_template,
                    lean_code,
                )
                if adequacy_error:
                    await self._record_partial_proof(
                        {
                            "session_id": self._state.session_id,
                            "attempt": attempt_number,
                            "target": "final",
                            "request": "final Proof Solver answer adequacy continuation",
                            "theorem_or_lemma": "Lean-accepted final scaffold not yet final-ready",
                            "placeholder_tokens": [],
                            "lean_code": lean_code,
                            "reasoning": reasoning,
                            "high_value_scaffold": False,
                            "master_seed_eligible": False,
                            "created_at": datetime.now().isoformat(),
                            "summary": (
                                "Lean accepted this code, but MOTO classified it as not final-ready "
                                f"for the LeanOJ answer obligation: {adequacy_error}"
                            ),
                        }
                    )
                    lean_result.success = False
                    lean_result.error_output = adequacy_error
            if lean_result.success:
                review_solved, review_feedback, lean_pass_feedback = await self._review_final_solution_completion(
                    request,
                    lean_code=lean_code,
                    final_solver_reasoning=reasoning,
                    lean_result=lean_result,
                )
                if not review_solved:
                    await self._record_partial_proof(
                        {
                            "session_id": self._state.session_id,
                            "attempt": attempt_number,
                            "target": "final",
                            "request": "final Proof Solver semantic-review continuation",
                            "theorem_or_lemma": "Lean-accepted final code requiring semantic continuation",
                            "placeholder_tokens": [],
                            "lean_code": lean_code,
                            "reasoning": reasoning,
                            "high_value_scaffold": False,
                            "master_seed_eligible": False,
                            "created_at": datetime.now().isoformat(),
                            "summary": (
                                "Lean accepted this code, but final semantic review requested continuation: "
                                f"{review_feedback}"
                            ),
                        }
                    )
                    lean_result.success = False
                    lean_result.error_output = (
                        "PROOF SOLVER FINAL SOLUTION REVIEW REJECTED: Lean 4 accepted the code, but the "
                        "Final Proof Solver judged that it does not yet solve the actual Proof Solver problem. "
                        f"Continuation feedback: {review_feedback}"
                    )
                    self._failed_feedback.append(
                        {
                            "request": "final Proof Solver solution semantic review",
                            "error_summary": self._summarize_error(lean_result.error_output, limit=1200),
                            "lean_feedback": self._summarize_error(lean_pass_feedback, limit=1200),
                            "lean_code": lean_code,
                        }
                    )
                    await self._persist_and_broadcast(
                        "leanoj_final_solution_review_rejected",
                        {
                            "attempt": attempt_number,
                            "continuation_feedback": self._summarize_error(review_feedback, limit=1200),
                            "lean_feedback": self._summarize_error(lean_pass_feedback, limit=1200),
                            **final_solver_metadata,
                        },
                    )
        return lean_result, lean_pass_feedback

    async def _final_proof_loop(self, request: LeanOJStartRequest) -> None:
        if await self._consume_force_brainstorm():
            return

        self._state.phase = "final_proof_loop"
        await self._persist_and_broadcast("leanoj_phase_changed")

        await self._ensure_master_proof_initialized(request)
        final_solver_metadata = {
            "solver_model": request.final_solver.model_id,
            "solver_provider": request.final_solver.provider,
        }
        failed_attempts_this_cycle = 0
        cycle_start_attempt = self._state.final_attempt_count + 1
        max_failed_attempts = max(1, request.final_attempts_per_cycle)
        while not self._should_stop() and failed_attempts_this_cycle < max_failed_attempts:
            if await self._consume_force_brainstorm():
                return

            current_master_proof = await self._read_master_proof()
            self._set_master_proof_metadata(current_master_proof)
            final_prompt_feedback = self._final_solver_failure_window()
            await self._broadcast(
                "leanoj_master_proof_edit_started",
                {
                    "next_verification_attempt": self._state.final_attempt_count + 1,
                    "master_proof_version": self._state.master_proof_version,
                },
            )
            try:
                context_blocks = await self._build_context_blocks(
                    request,
                    request.final_solver,
                    mode="final_solver",
                    task_request="Edit the durable Proof Solver master proof and decide whether it is ready for Lean verification.",
                    capped_rejection_feedback=self._format_capped_rejection_feedback(
                        "RECENT PROOF FEEDBACK SUMMARIES",
                        final_prompt_feedback,
                        limit=10,
                    ),
                )
                master_proof_direct_context, direct_context_metadata = self._build_master_proof_direct_context(
                    current_master_proof,
                    request,
                    context_blocks,
                )
                raw = await self._call_json(
                    request.final_solver,
                    "leanoj_final",
                    "leanoj_final_solver",
                    build_final_solver_prompt(
                        request.user_prompt,
                        request.lean_template,
                        master_proof_direct_context,
                        {
                            "version": self._state.master_proof_version,
                            "line_count": self._state.master_proof_line_count,
                            "char_count": self._state.master_proof_char_count,
                            "sha256": self._state.master_proof_hash,
                            "last_edit_summary": self._state.master_proof_last_edit_summary,
                            "last_shortening_approval_justification": (
                                self._state.master_proof_last_shortening_approval_justification
                            ),
                            "last_shortening_apparent_issue": self._state.master_proof_last_shortening_apparent_issue,
                            **direct_context_metadata,
                        },
                        self._final_solver_active_plan_items(),
                        self._final_solver_verified_subproof_dicts(),
                        self._partial_proofs,
                        final_prompt_feedback,
                        self._final_attempts[-5:],
                        context_blocks=context_blocks,
                    ),
                )
            except asyncio.CancelledError:
                raise
            except LeanOJConfigurationError:
                raise
            except Exception as exc:
                attempt_number = self._state.final_attempt_count + 1
                self._state.final_attempt_count = attempt_number
                failed_attempts_this_cycle += 1
                error_text = f"Final solver failed before Lean verification: {type(exc).__name__}: {exc}"
                attempt = LeanOJAttemptRecord(
                    attempt=attempt_number,
                    target="final",
                    request="final Proof Solver master proof edit",
                    success=False,
                    error_output=error_text,
                    reasoning="Model/API output could not be parsed or generated; retrying in the final loop.",
                )
                self._final_attempts.append(
                    {
                        "request": "final Proof Solver master proof edit",
                        "error_summary": self._summarize_error(error_text, limit=1200),
                        "lean_code": current_master_proof,
                    }
                )
                self._record_final_context_event(
                    "failure",
                    request="final Proof Solver master proof edit",
                    error_summary=error_text,
                )
                await self._persist_and_broadcast(
                    "leanoj_final_attempt_failed",
                    {"attempt": attempt.model_dump(mode="json"), **final_solver_metadata},
                )
                continue

            edit = self._normalize_final_solver_edit(raw)
            reasoning = str(edit.get("reasoning") or "")
            updated_master_proof, edit_error = self._apply_master_proof_edit(current_master_proof, edit)
            if edit_error or updated_master_proof is None:
                attempt_number = self._state.final_attempt_count + 1
                self._state.final_attempt_count = attempt_number
                failed_attempts_this_cycle += 1
                error_text = f"MASTER PROOF EDIT REJECTED: {edit_error}"
                attempt = LeanOJAttemptRecord(
                    attempt=attempt_number,
                    target="final",
                    request="final Proof Solver master proof edit",
                    lean_code=current_master_proof,
                    success=False,
                    error_output=error_text,
                    reasoning=reasoning,
                )
                self._final_attempts.append(
                    {
                        "request": "final Proof Solver master proof edit",
                        "error_summary": self._summarize_error(error_text, limit=1200),
                        "lean_code": current_master_proof,
                    }
                )
                self._record_final_context_event(
                    "failure",
                    request="final Proof Solver master proof edit",
                    error_summary=error_text,
                    reasoning=reasoning,
                )
                await self._append_master_proof_edit(
                    {
                        **edit,
                        "accepted": False,
                        "error_summary": self._summarize_error(error_text, limit=1200),
                    }
                )
                await self._persist_and_broadcast(
                    "leanoj_final_attempt_failed",
                    {"attempt": attempt.model_dump(mode="json"), **final_solver_metadata},
                )
                continue

            shortening_metrics = self._master_proof_shortening_metrics(current_master_proof, updated_master_proof)
            shortening_approval_justification = ""
            old_attempt_apparent_issue = ""
            if self._should_validate_master_proof_shortening_edit(edit, shortening_metrics):
                (
                    edit_valid,
                    validator_feedback,
                    validator_reasoning,
                    shortening_approval_justification,
                    old_attempt_apparent_issue,
                ) = await self._validate_master_proof_shortening_edit(
                    request,
                    edit,
                    current_master_proof,
                    updated_master_proof,
                    shortening_metrics,
                )
                if not edit_valid:
                    attempt_number = self._state.final_attempt_count + 1
                    self._state.final_attempt_count = attempt_number
                    failed_attempts_this_cycle += 1
                    error_text = (
                        "MASTER PROOF EDIT VALIDATOR REJECTED SHORTENING: "
                        f"{validator_feedback}"
                    )
                    error_summary = self._summarize_error(error_text, limit=1200)
                    self._failed_feedback.append(
                        {
                            "request": "final Proof Solver master proof edit validator",
                            "error_summary": error_summary,
                            "reasoning": self._summarize_error(validator_reasoning, limit=1200),
                        }
                    )
                    attempt = LeanOJAttemptRecord(
                        attempt=attempt_number,
                        target="final",
                        request="final Proof Solver master proof edit validator",
                        lean_code=current_master_proof,
                        success=False,
                        error_output=error_text,
                        reasoning=reasoning,
                    )
                    self._final_attempts.append(
                        {
                            "request": "final Proof Solver master proof edit validator",
                            "error_summary": error_summary,
                            "lean_code": current_master_proof,
                            "validator_feedback": self._summarize_error(validator_feedback, limit=1200),
                            "validator_reasoning": self._summarize_error(validator_reasoning, limit=1200),
                        }
                    )
                    self._record_final_context_event(
                        "failure",
                        request="final Proof Solver master proof edit validator",
                        error_summary=error_summary,
                        reasoning=validator_reasoning,
                    )
                    await self._append_master_proof_edit(
                        {
                            **edit,
                            "accepted": False,
                            "error_summary": error_summary,
                            "validator_feedback": self._summarize_error(validator_feedback, limit=1200),
                            "validator_reasoning": self._summarize_error(validator_reasoning, limit=1200),
                            "shortening_metrics": shortening_metrics,
                        }
                    )
                    await self._persist_and_broadcast(
                        "leanoj_master_proof_edit_rejected",
                        {
                            "attempt": attempt_number,
                            "error_summary": error_summary,
                            "validator_feedback": self._summarize_error(validator_feedback, limit=1200),
                            "validator_reasoning": self._summarize_error(validator_reasoning, limit=1200),
                            "shortening_metrics": shortening_metrics,
                            **final_solver_metadata,
                        },
                    )
                    await self._persist_and_broadcast(
                        "leanoj_final_attempt_failed",
                        {"attempt": attempt.model_dump(mode="json"), **final_solver_metadata},
                    )
                    continue

            needs_more_time = bool(edit.get("needs_more_time"))
            lean_code = updated_master_proof.strip()
            attempt_number = self._state.final_attempt_count + 1
            if not needs_more_time:
                await self._broadcast(
                    "leanoj_final_attempt_started",
                    {"attempt": attempt_number, **final_solver_metadata},
                )
            lean_result, lean_pass_feedback = await self._check_master_proof_edit_before_persist(
                request,
                lean_code=lean_code,
                needs_more_time=needs_more_time,
                attempt_number=attempt_number,
                reasoning=reasoning,
                final_solver_metadata=final_solver_metadata,
            )
            if not lean_result.success:
                self._state.final_attempt_count = attempt_number
                failed_attempts_this_cycle += 1
                failure_request = (
                    "final Proof Solver master proof edit Lean gate"
                    if needs_more_time
                    else "final Proof Solver solution from master proof"
                )
                error_summary = self._summarize_error(lean_result.error_output, limit=1200)
                attempt = LeanOJAttemptRecord(
                    attempt=attempt_number,
                    target="final",
                    request=failure_request,
                    lean_code=lean_code,
                    success=False,
                    error_output=lean_result.error_output,
                    reasoning=reasoning,
                )
                failure = {
                    "request": failure_request,
                    "error_summary": error_summary,
                    "lean_code": lean_code,
                }
                if lean_pass_feedback:
                    failure["lean_feedback"] = self._summarize_error(lean_pass_feedback, limit=1200)
                lean_diagnostics = {
                    key: self._summarize_error(str(value), limit=1200)
                    for key, value in {
                        "diagnostic_output": getattr(lean_result, "diagnostic_output", ""),
                        "goal_states": getattr(lean_result, "goal_states", ""),
                        "raw_stderr": getattr(lean_result, "raw_stderr", ""),
                    }.items()
                    if str(value or "").strip()
                }
                failure.update(lean_diagnostics)
                self._final_attempts.append(failure)
                self._record_final_context_event(
                    "failure",
                    request=failure_request,
                    error_summary=error_summary,
                    lean_feedback=str(failure.get("lean_feedback") or ""),
                    reasoning=reasoning,
                )
                await self._append_master_proof_edit(
                    {
                        **edit,
                        "accepted": False,
                        "error_summary": error_summary,
                        "lean_code": lean_code,
                        **lean_diagnostics,
                        **({"lean_feedback": failure["lean_feedback"]} if "lean_feedback" in failure else {}),
                    }
                )
                await self._persist_and_broadcast(
                    "leanoj_master_proof_edit_rejected",
                    {
                        "attempt": attempt_number,
                        "error_summary": error_summary,
                        **lean_diagnostics,
                        **({"lean_feedback": failure["lean_feedback"]} if "lean_feedback" in failure else {}),
                        **final_solver_metadata,
                    },
                )
                await self._persist_and_broadcast(
                    "leanoj_final_attempt_failed",
                    {"attempt": attempt.model_dump(mode="json"), **final_solver_metadata},
                )
                continue

            if shortening_approval_justification or old_attempt_apparent_issue:
                self._state.master_proof_last_shortening_approval_justification = self._summarize_error(
                    shortening_approval_justification,
                    limit=1200,
                )
                self._state.master_proof_last_shortening_apparent_issue = self._summarize_error(
                    old_attempt_apparent_issue,
                    limit=1200,
                )
                old_char_count = len(current_master_proof or "")
                stored_old_char_count = self._state.master_proof_old_attempt_before_redo_char_count
                if old_char_count > stored_old_char_count:
                    await self._write_master_proof_old_attempt_before_redo(current_master_proof)
                    self._state.master_proof_old_attempt_before_redo_version = self._state.master_proof_version
                    self._state.master_proof_old_attempt_before_redo_hash = self._hash_master_proof(current_master_proof)
                    self._state.master_proof_old_attempt_before_redo_line_count = (
                        len(current_master_proof.splitlines()) if current_master_proof else 0
                    )
                    self._state.master_proof_old_attempt_before_redo_char_count = old_char_count
                    self._state.master_proof_old_attempt_before_redo_summary = (
                        f"Submitter chose to redo/shorten this v{self._state.master_proof_version} attempt "
                        f"({old_char_count} chars, "
                        f"{self._state.master_proof_old_attempt_before_redo_line_count} lines)."
                    )
                    self._state.master_proof_old_attempt_before_redo_validator_justification = (
                        self._summarize_error(shortening_approval_justification, limit=1200)
                    )
                    self._state.master_proof_old_attempt_before_redo_apparent_issue = self._summarize_error(
                        old_attempt_apparent_issue,
                        limit=1200,
                    )

            edit_summary = reasoning or f"Applied {edit.get('operation')} edit to Proof Solver master proof."
            if shortening_approval_justification or old_attempt_apparent_issue:
                edit_summary = " ".join(
                    part
                    for part in (
                        edit_summary,
                        (
                            f"Validator allowed shortening because: {shortening_approval_justification}"
                            if shortening_approval_justification
                            else ""
                        ),
                        (
                            f"Apparent issue with old longer attempt: {old_attempt_apparent_issue}"
                            if old_attempt_apparent_issue
                            else ""
                        ),
                    )
                    if part
                )
            shortening_audit = {}
            if shortening_approval_justification:
                shortening_audit["shortening_approval_justification"] = self._summarize_error(
                    shortening_approval_justification,
                    limit=1200,
                )
            if old_attempt_apparent_issue:
                shortening_audit["old_attempt_apparent_issue"] = self._summarize_error(
                    old_attempt_apparent_issue,
                    limit=1200,
                )
            await self._write_master_proof(updated_master_proof, summary=edit_summary)
            await self._append_master_proof_edit(
                {
                    **edit,
                    "accepted": True,
                    "master_proof_hash": self._state.master_proof_hash,
                    "master_proof_line_count": self._state.master_proof_line_count,
                    "master_proof_char_count": self._state.master_proof_char_count,
                    **shortening_audit,
                }
            )
            await self._persist_and_broadcast(
                "leanoj_master_proof_edit_applied",
                {
                    "master_proof_version": self._state.master_proof_version,
                    "needs_more_time": needs_more_time,
                    "reasoning": self._summarize_error(edit_summary, limit=500),
                },
            )
            self._record_final_context_event(
                "acceptance",
                request="final Proof Solver master proof edit accepted",
                reasoning=edit_summary,
            )

            if needs_more_time:
                watchdog_reason = self._record_master_proof_progress(edit, current_master_proof, updated_master_proof)
                if watchdog_reason:
                    self._state.master_proof_last_stuck_reason = self._summarize_error(watchdog_reason, limit=500)
                    self._failed_feedback.append(
                        {
                            "request": "final Proof Solver master proof progress watchdog",
                            "error_summary": self._summarize_error(watchdog_reason, limit=1200),
                        }
                    )
                    await self._append_master_proof_edit(
                        {
                            "action": "progress_watchdog",
                            "reasoning": watchdog_reason,
                            "master_proof_hash": self._state.master_proof_hash,
                            "master_proof_line_count": self._state.master_proof_line_count,
                            "master_proof_char_count": self._state.master_proof_char_count,
                        }
                    )
                    self._reset_master_proof_progress_watchdog()
                    attempt_number = self._state.final_attempt_count + 1
                    self._state.final_attempt_count = attempt_number
                    failed_attempts_this_cycle += 1
                    self._final_attempts.append(
                        {
                            "request": "final Proof Solver master proof progress watchdog",
                            "error_summary": self._summarize_error(watchdog_reason, limit=1200),
                            "lean_code": updated_master_proof,
                        }
                    )
                    self._record_final_context_event(
                        "failure",
                        request="final Proof Solver master proof progress watchdog",
                        error_summary=watchdog_reason,
                        reasoning=reasoning,
                    )
                    await self._persist_and_broadcast(
                        "leanoj_final_attempt_failed",
                        {
                            "attempt": LeanOJAttemptRecord(
                                attempt=attempt_number,
                                target="final",
                                request="final Proof Solver master proof progress watchdog",
                                lean_code=updated_master_proof,
                                success=False,
                                error_output=watchdog_reason,
                                reasoning=reasoning,
                            ).model_dump(mode="json"),
                            **final_solver_metadata,
                        },
                    )
                    await self._persist_and_broadcast(
                        "leanoj_master_proof_progress_watchdog",
                        {
                            "reasoning": watchdog_reason,
                            "continuing_final_cycle": (
                                failed_attempts_this_cycle < max_failed_attempts
                                and self._state.user_forced_final_cycle
                            ),
                        },
                    )
                    if failed_attempts_this_cycle < max_failed_attempts and self._state.user_forced_final_cycle:
                        logger.info(
                            "LeanOJ final cycle continuing after progress watchdog",
                        )
                        self._state.phase = "final_proof_loop"
                        self._state.current_path_decision = "solve_final_now"
                        continue
                    break
                continue
            self._reset_master_proof_progress_watchdog()

            self._state.final_attempt_count = attempt_number
            attempt = LeanOJAttemptRecord(
                attempt=attempt_number,
                target="final",
                request="final Proof Solver solution from master proof",
                lean_code=lean_code,
                success=lean_result.success,
                error_output=lean_result.error_output,
                reasoning=reasoning,
            )

            try:
                proof_record = await self._register_verified_leanoj_proof(
                    request,
                    proof_kind="final",
                    theorem_statement=request.user_prompt,
                    theorem_name="Final Proof Solver Submission",
                    lean_code=lean_code,
                    attempt_count=attempt_number,
                    formal_sketch="Final Proof Solver solution for the user's template.",
                    theorem_id=f"{self._state.session_id}_final",
                    source_title=self._state.selected_topic or request.user_prompt,
                )
            except Exception as exc:
                if self._is_non_retryable_model_error(exc):
                    raise LeanOJConfigurationError(str(exc)) from exc
                lean_result.success = False
                lean_result.error_output = f"PROOF SOLVER PROOF REGISTRATION FAILED: {exc}"
                attempt.success = False
                attempt.error_output = lean_result.error_output

            if lean_result.success:
                self._state.phase = "verified"
                self._state.user_forced_final_cycle = False
                self._state.final_solution = lean_code
                self._state.final_proof_id = proof_record.proof_id if proof_record else ""
                self._state.final_novel = proof_record.novel if proof_record else False
                self._state.final_novelty_tier = proof_record.novelty_tier if proof_record else "not_novel"
                self._state.final_novelty_reasoning = proof_record.novelty_reasoning if proof_record else ""
                self._current_final_cycle_packet = None
                self._current_working_proof_attempt = None
                await self._persist_and_broadcast(
                    "leanoj_final_verified",
                    {"attempt": attempt.model_dump(mode="json"), **final_solver_metadata},
                )
                return

            failure = {
                "request": "final Proof Solver solution from master proof",
                "error_summary": self._summarize_error(lean_result.error_output, limit=1200),
                "lean_code": lean_code,
            }
            if lean_pass_feedback:
                failure["lean_feedback"] = self._summarize_error(lean_pass_feedback, limit=1200)
            self._final_attempts.append(failure)
            self._record_final_context_event(
                "failure",
                request=str(failure.get("request") or "final Proof Solver solution from master proof"),
                error_summary=str(failure.get("error_summary") or ""),
                lean_feedback=str(failure.get("lean_feedback") or ""),
                reasoning=reasoning,
            )
            failed_attempts_this_cycle += 1
            await self._persist_and_broadcast(
                "leanoj_final_attempt_failed",
                {"attempt": attempt.model_dump(mode="json"), **final_solver_metadata},
            )

        if self._should_stop() or self._state.phase == "verified":
            return

        cycle_end_attempt = self._state.final_attempt_count
        last_error = ""
        if self._final_attempts:
            last_error = str(self._final_attempts[-1].get("error_summary") or "")
        cycle_summary = (
            "The final master proof loop did not verify yet. "
            f"Latest blocker: {last_error or 'No final attempt error was recorded.'} "
            "Use the concrete Lean/edit feedback to choose the next proof action."
        )
        cycle_attempts = list(self._final_attempts[-failed_attempts_this_cycle:])
        cycle_partials = [
            proof
            for proof in self._partial_proofs
            if str(proof.get("target") or "") == "final"
            and cycle_start_attempt <= int(proof.get("attempt") or 0) <= cycle_end_attempt
        ]
        cycle_packet = {
            "session_id": self._state.session_id,
            "cycle_start_attempt": cycle_start_attempt,
            "cycle_end_attempt": cycle_end_attempt,
            "failed_attempt_count": failed_attempts_this_cycle,
            "attempts": cycle_attempts,
            "partial_proofs": cycle_partials,
            "created_at": datetime.now().isoformat(),
            "summary": self._summarize_error(cycle_summary, limit=1200),
        }
        self._final_cycle_packets.append(cycle_packet)
        self._current_final_cycle_packet = cycle_packet
        self._failed_feedback.append(
            {
                "request": "final Proof Solver proof cycle",
                "error_summary": self._summarize_error(cycle_summary, limit=1200),
            }
        )
        handoff_to_recursive = self._final_cycle_should_handoff_to_recursive(cycle_attempts)
        self._state.user_forced_final_cycle = False
        self._state.phase = "recursive_brainstorm" if handoff_to_recursive else "path_decision"
        self._state.current_path_decision = "need_more_brainstorming"
        await self._set_current_working_proof_attempt(
            trigger="final_attempt_cycle_exhausted",
            requested_path="need_more_brainstorming",
            stuck_reason=cycle_summary,
        )
        await self._persist_and_broadcast(
            "leanoj_final_attempt_cycle_exhausted",
            {
                "attempts_in_cycle": failed_attempts_this_cycle,
                "cycle_start_attempt": cycle_start_attempt,
                "cycle_end_attempt": cycle_end_attempt,
                "message": self._summarize_error(cycle_summary, limit=500),
            },
        )

    @staticmethod
    def _normalize_lean_for_template_check(code: str) -> str:
        return " ".join((code or "").split())

    @classmethod
    def _validate_final_solution_integrity(cls, lean_template: str, lean_code: str) -> str:
        device_error = cls._validate_no_new_declaration_devices(lean_template, lean_code, target="solution")
        if device_error:
            return device_error
        template_error = cls._validate_final_solution_matches_template(lean_template, lean_code)
        if template_error:
            return template_error
        return ""

    @classmethod
    def _validate_final_answer_adequacy(cls, lean_template: str, lean_code: str) -> str:
        """Reject final-only answer definitions that restate the extremal target."""
        if cls._placeholder_tokens(lean_code):
            return ""
        template_answer = cls._find_declaration_block(lean_template, "def answer")
        if not template_answer or not cls._declaration_has_placeholder(template_answer):
            return ""
        candidate_answer = cls._find_declaration_block(lean_code, "def answer")
        if not candidate_answer:
            return ""

        body = cls._normalize_lean_for_semantic_scan(cls._lean_declaration_body(candidate_answer))
        if not body:
            return ""

        extremal_markers = (
            "ssup",
            "csup",
            "nat.ssup",
            "sup ",
            "isgreatest",
            "bddabove",
            "upperbounds",
        )
        self_reference_markers = (
            "s n",
            "set ",
            "finset",
            "card",
            "exists",
            "u in",
            "v in",
            "divides",
            " ∣ ",
            "∣",
        )
        if any(marker in body for marker in extremal_markers) and any(
            marker in body for marker in self_reference_markers
        ):
            return (
                "PROOF SOLVER ANSWER ADEQUACY REJECTED: Lean accepted the final code, but `answer` is defined "
                "using an extremal/supremum construction over the same feasible-cardinality problem instead "
                "of determining the requested largest size in terms of n. This may remain in the durable "
                "master proof as intermediate context, but it is not final-ready. Continue the Proof Solver loop, "
                "derive an explicit formula for `answer n`, and then prove `IsGreatest (S n)` for that formula."
            )
        return ""

    @classmethod
    def _validate_no_new_declaration_devices(cls, lean_template: str, lean_code: str, *, target: str) -> str:
        integrity = validate_lean_proof_integrity(
            lean_code=lean_code,
            allowed_baseline=lean_template,
        )
        if integrity.valid:
            return ""
        return (
            "PROOF SOLVER FORBIDDEN PROOF DEVICE: Lean accepted the submitted code, but the "
            f"{target} introduced new axiom/constant/opaque declarations not present in the original template: "
            f"{', '.join(integrity.introduced_devices[:8])}. Do not solve Proof Solver problems by adding fake assumptions; "
            "preserve the template and fill the proof using constructive Lean/Mathlib proof terms or tactics."
        )

    @classmethod
    def _validate_final_solution_matches_template(cls, lean_template: str, lean_code: str) -> str:
        """Return an error message when a compiling final answer does not solve the template."""
        template = lean_template or ""
        candidate = lean_code or ""
        hole_aware_error = cls._validate_final_solution_matches_template_declarations(template, candidate)
        if hole_aware_error is not None:
            return hole_aware_error

        template_parts = [
            cls._normalize_lean_for_template_check(part)
            for part in _LEAN_PLACEHOLDER_RE.split(template)
            if part not in {"sorry", "admit"}
        ]
        significant_parts = [part for part in template_parts if len(part) >= 12]
        normalized_candidate = cls._normalize_lean_for_template_check(candidate)

        if not significant_parts:
            normalized_template = cls._normalize_lean_for_template_check(template)
            if normalized_template and normalized_template not in normalized_candidate:
                return (
                    "PROOF SOLVER TEMPLATE MISMATCH: Lean accepted the submitted code, but the code does not preserve "
                    "the user's original Proof Solver template/declaration. Return the complete original template with "
                    "only the proof holes filled unless a template change is explicitly required."
                )
            return ""

        search_from = 0
        for part in significant_parts:
            found_at = normalized_candidate.find(part, search_from)
            if found_at < 0:
                return (
                    "PROOF SOLVER TEMPLATE MISMATCH: Lean accepted the submitted code, but the code does not contain "
                    "the original Proof Solver template structure around the proof hole. Do not replace the task with "
                    "an unrelated theorem; preserve the user's declarations and fill the required proof."
                )
            search_from = found_at + len(part)
        return ""

    @classmethod
    def _validate_final_solution_matches_template_declarations(cls, lean_template: str, lean_code: str) -> Optional[str]:
        """Validate LeanOJ submissions by preserving declarations while allowing hole bodies to change."""
        template_decls = cls._lean_declaration_blocks(lean_template)
        candidate_decls = cls._lean_declaration_blocks(lean_code)
        if not template_decls or not candidate_decls:
            return None

        candidate_by_key: dict[str, str] = {}
        for declaration in candidate_decls:
            key = cls._lean_declaration_key(declaration)
            if key and key not in candidate_by_key:
                candidate_by_key[key] = declaration

        candidate_imports = set(cls._lean_imports(lean_code))
        for import_line in cls._lean_imports(lean_template):
            if import_line not in candidate_imports:
                return (
                    "PROOF SOLVER TEMPLATE MISMATCH: Lean accepted the submitted code, but the code removed an "
                    f"original Proof Solver import required by the template: {import_line}. Preserve original imports; "
                    "additional imports are allowed when Lean needs them."
                )

        for template_decl in template_decls:
            key = cls._lean_declaration_key(template_decl)
            if not key:
                continue
            candidate_decl = candidate_by_key.get(key)
            if not candidate_decl:
                return (
                    "PROOF SOLVER TEMPLATE MISMATCH: Lean accepted the submitted code, but it does not preserve "
                    f"the original Proof Solver declaration `{key}`. Do not replace the task with unrelated declarations."
                )

            if cls._declaration_has_placeholder(template_decl):
                template_header = cls._normalize_lean_declaration_header(template_decl)
                candidate_header = cls._normalize_lean_declaration_header(candidate_decl)
                if template_header != candidate_header:
                    return (
                        "PROOF SOLVER TEMPLATE MISMATCH: Lean accepted the submitted code, but it changed the "
                        f"signature/target of original declaration `{key}`. Fill only the `sorry`/`admit` body."
                    )
                continue

            normalized_template_decl = cls._normalize_lean_for_template_check(template_decl)
            normalized_candidate_decl = cls._normalize_lean_for_template_check(candidate_decl)
            if normalized_template_decl != normalized_candidate_decl:
                return (
                    "PROOF SOLVER TEMPLATE MISMATCH: Lean accepted the submitted code, but it changed a fixed "
                    f"non-hole declaration `{key}` from the original template. Preserve fixed definitions exactly."
                )

        return ""

    @staticmethod
    def _lean_imports(code: str) -> list[str]:
        return [
            line.strip()
            for line in (code or "").splitlines()
            if line.strip().startswith("import ")
        ]

    @staticmethod
    def _lean_declaration_blocks(code: str) -> list[str]:
        matches = list(_LEAN_TOP_LEVEL_DECL_RE.finditer(code or ""))
        blocks: list[str] = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(code or "")
            block = (code or "")[match.start() : end].strip()
            if block:
                blocks.append(block)
        return blocks

    @staticmethod
    def _lean_declaration_key(declaration: str) -> str:
        match = _LEAN_DECL_KEY_RE.search(declaration or "")
        if not match:
            return ""
        kind = match.group("kind") or ""
        name = match.group("name") or ""
        return f"{kind} {name}".strip()

    @classmethod
    def _find_declaration_block(cls, code: str, declaration_key: str) -> str:
        for declaration in cls._lean_declaration_blocks(code):
            if cls._lean_declaration_key(declaration) == declaration_key:
                return declaration
        return ""

    @staticmethod
    def _lean_declaration_body(declaration: str) -> str:
        if ":=" not in (declaration or ""):
            return ""
        return (declaration or "").split(":=", 1)[1].strip()

    @classmethod
    def _normalize_lean_for_semantic_scan(cls, code: str) -> str:
        return cls._normalize_lean_for_template_check(strip_lean_comments_and_strings(code or "")).lower()

    @staticmethod
    def _declaration_has_placeholder(declaration: str) -> bool:
        return bool(_LEAN_PLACEHOLDER_RE.search(strip_lean_comments_and_strings(declaration or "")))

    @classmethod
    def _normalize_lean_declaration_header(cls, declaration: str) -> str:
        header = (declaration or "").split(":=", 1)[0]
        normalized = cls._normalize_lean_for_template_check(header)
        while True:
            previous = normalized
            normalized = re.sub(r"^open\s+Classical\s+in\s+", "", normalized)
            normalized = re.sub(r"^(?:@\[[^\]]+\]\s*)+", "", normalized)
            normalized = re.sub(r"^(?:(?:private|protected|noncomputable|unsafe)\s+)+", "", normalized)
            normalized = normalized.strip()
            if normalized == previous:
                break
        return normalized.strip()

    @staticmethod
    def _json_retry_schema_hint(role_id: str) -> str:
        if role_id.startswith("leanoj_brainstorm_submitter_"):
            return (
                "ROLE-SPECIFIC COMPACT RETRY CONTRACT:\n"
                "- For this retry, use the normal idea schema only; do not choose `lean_proof`.\n"
                "- Return exactly: "
                "{\"submission_type\":\"idea\",\"submission\":\"...\",\"reasoning\":\"...\"}\n"
                "- Keep `submission` under 600 characters and `reasoning` under 200 characters.\n"
                "- Do not quote the Lean template, prior proof, accepted ideas, or failure log."
            )
        if role_id == "leanoj_brainstorm_validator":
            return (
                "ROLE-SPECIFIC COMPACT RETRY CONTRACT:\n"
                "- If the original prompt requested batch validation, return exactly one compact "
                "`decisions` entry per submission in the original order.\n"
                "- If the original prompt requested single validation, return one compact "
                "`decision` object.\n"
                "- Keep every `reasoning` and `summary` string under 160 characters.\n"
                "- Do not quote submissions, Lean code, accepted ideas, or proof context."
            )
        if role_id == "leanoj_topic_validator":
            return (
                "ROLE-SPECIFIC COMPACT RETRY CONTRACT:\n"
                "- Keep every topic, reasoning, and summary field under 200 characters.\n"
                "- Do not quote the Lean template or prior topics."
            )
        return (
            "ROLE-SPECIFIC COMPACT RETRY CONTRACT:\n"
            "- Keep every string field short; do not quote large context blocks or code."
        )

    @classmethod
    def _summarize_model_call_result(
        cls,
        role_id: str,
        task_id: str,
        parsed: dict[str, Any],
        *,
        limit: int = 700,
    ) -> str:
        """Return a compact outcome summary for live logs and INFO output."""
        if not isinstance(parsed, dict):
            return cls._summarize_error(str(parsed), limit=limit)

        def clean(value: Any, text_limit: int = 320) -> str:
            return cls._summarize_error(str(value or ""), limit=text_limit)

        def first_text(*keys: str, text_limit: int = 320) -> str:
            for key in keys:
                value = parsed.get(key)
                if value:
                    return clean(value, text_limit)
            return ""

        decisions = parsed.get("decisions")
        if isinstance(decisions, list):
            accepted = 0
            rejected = 0
            samples: list[str] = []
            for index, decision in enumerate(decisions, start=1):
                if not isinstance(decision, dict):
                    continue
                verdict = clean(decision.get("decision") or decision.get("verdict"), 40).lower()
                if verdict == "accept":
                    accepted += 1
                elif verdict == "reject":
                    rejected += 1
                reason = clean(
                    decision.get("summary") or decision.get("reasoning") or decision.get("feedback"),
                    160,
                )
                if reason and len(samples) < 2:
                    samples.append(f"{index}: {verdict or 'decision'} - {reason}")
            prefix = f"batch result: {accepted} accepted, {rejected} rejected"
            return cls._summarize_error(
                f"{prefix}; {'; '.join(samples)}" if samples else prefix,
                limit=limit,
            )

        if "enough" in parsed:
            status = "ready for path decision" if bool(parsed.get("enough")) else "continue brainstorming"
            reason = first_text("reasoning", "summary", "feedback", text_limit=260)
            return cls._summarize_error(
                f"sufficiency result: {status}{f' - {reason}' if reason else ''}",
                limit=limit,
            )

        if parsed.get("path"):
            reason = first_text("reasoning", "summary", text_limit=300)
            return cls._summarize_error(
                f"path result: {clean(parsed.get('path'), 80)}{f' - {reason}' if reason else ''}",
                limit=limit,
            )

        if parsed.get("decision"):
            reason = first_text("summary", "reasoning", "feedback_to_submitter", text_limit=300)
            return cls._summarize_error(
                f"decision: {clean(parsed.get('decision'), 80)}{f' - {reason}' if reason else ''}",
                limit=limit,
            )

        if parsed.get("action") or parsed.get("operation"):
            action = clean(parsed.get("action") or "action", 80)
            operation = clean(parsed.get("operation"), 80)
            reason = first_text("reasoning", "summary", "stuck_reason", text_limit=300)
            label = f"{action}{f'/{operation}' if operation else ''}"
            return cls._summarize_error(
                f"action result: {label}{f' - {reason}' if reason else ''}",
                limit=limit,
            )

        if parsed.get("topic"):
            reason = first_text("reasoning", "summary", text_limit=220)
            return cls._summarize_error(
                f"topic: {clean(parsed.get('topic'), 360)}{f' - {reason}' if reason else ''}",
                limit=limit,
            )

        if parsed.get("submission"):
            submission_type = clean(parsed.get("submission_type") or "idea", 60)
            reason = first_text("reasoning", "formal_sketch", text_limit=220)
            return cls._summarize_error(
                f"{submission_type}: {clean(parsed.get('submission'), 420)}{f' - {reason}' if reason else ''}",
                limit=limit,
            )

        if parsed.get("theorem_statement"):
            theorem = clean(parsed.get("theorem_name") or parsed.get("theorem_statement"), 360)
            sketch = first_text("formal_sketch", "reasoning", text_limit=220)
            return cls._summarize_error(
                f"lean proof: {theorem}{f' - {sketch}' if sketch else ''}",
                limit=limit,
            )

        if "solved" in parsed:
            reason = first_text("reasoning", "summary", "continuation_feedback", text_limit=320)
            return cls._summarize_error(
                f"solver review: {'solved' if bool(parsed.get('solved')) else 'not solved'}{f' - {reason}' if reason else ''}",
                limit=limit,
            )

        summary = first_text(
            "summary",
            "reasoning",
            "feedback",
            "message",
            "answer",
            text_limit=500,
        )
        if summary:
            return summary

        keys = ", ".join(sorted(str(key) for key in parsed.keys())[:8])
        return f"{role_id or task_id} returned JSON fields: {keys or 'none'}"

    async def _pause_for_provider_credits(
        self,
        *,
        role_id: str,
        call_payload: dict[str, Any],
        message: str,
        duration_ms: int,
    ) -> None:
        mark_provider_paused()
        pause_payload = {
            **call_payload,
            "duration_ms": duration_ms,
            "retryable": True,
            "reason": "openrouter_credit_exhaustion",
            "message": message,
        }
        self._state.provider_paused = True
        self._state.provider_pause_reason = "openrouter_credit_exhaustion"
        self._state.provider_pause_role_id = role_id
        self._state.provider_pause_message = message
        await self._persist_and_broadcast("leanoj_provider_paused", pause_payload)

        await wait_for_provider_resume(self._should_stop)
        if self._should_stop():
            raise asyncio.CancelledError()

        self._state.provider_paused = False
        self._state.provider_pause_reason = ""
        self._state.provider_pause_role_id = ""
        self._state.provider_pause_message = ""
        await self._persist_and_broadcast("leanoj_provider_resumed", pause_payload)

    async def _call_json(
        self,
        config: LeanOJRoleConfig,
        task_prefix: str,
        role_id: str,
        prompt: str,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        if not config.model_id:
            raise LeanOJConfigurationError(f"Proof Solver role {role_id} has no configured model")
        current_prompt = prompt
        attempt_index = 0
        while not self._should_stop():
            attempt_index += 1
            if self._should_stop():
                raise asyncio.CancelledError()
            task_id = self._next_task_id(task_prefix)
            self.current_task_id = task_id
            self._refresh_workflow_tasks(task_prefix, role_id)
            api_client_manager.set_autonomous_phase(self._state.phase or "leanoj")
            started = time.monotonic()
            call_payload = {
                "role_id": role_id,
                "task_id": task_id,
                "phase": self._state.phase or "leanoj",
                "attempt": attempt_index,
                "provider": config.provider,
                "model": config.model_id,
                "context_window": config.context_window,
                "max_output_tokens": config.max_output_tokens,
                "temperature": temperature,
            }
            logger.debug(
                "Proof Solver model call started (role=%s, task=%s, phase=%s, provider=%s, model=%s, attempt=%s)",
                role_id,
                task_id,
                call_payload["phase"],
                config.provider,
                config.model_id,
                attempt_index,
            )
            try:
                prompt_tokens = count_tokens(current_prompt)
                max_input_tokens = rag_config.get_available_input_tokens(
                    config.context_window,
                    config.max_output_tokens,
                )
                await api_client_manager.prewarm_assistant_memory_context(
                    task_id=task_id,
                    role_id=role_id,
                    prompt=current_prompt,
                )
                call_payload["prompt_tokens"] = prompt_tokens
                call_payload["max_input_tokens"] = max_input_tokens
                if prompt_tokens > max_input_tokens:
                    self._pending_final_solver_assistant_target_hash = ""
                    raise LeanOJConfigurationError(
                        "PROOF SOLVER PROMPT CONTEXT OVERFLOW: assembled prompt exceeds the configured "
                        f"input budget for role {role_id}. Prompt tokens: {prompt_tokens}. "
                        f"Available input tokens: {max_input_tokens}. Context window: {config.context_window}. "
                        f"Max output tokens: {config.max_output_tokens}."
                    )
                response = await api_client_manager.generate_completion(
                    task_id=task_id,
                    role_id=role_id,
                    model=config.model_id,
                    messages=[{"role": "user", "content": current_prompt}],
                    max_tokens=config.max_output_tokens,
                    temperature=temperature,
                )
                if self._pending_final_solver_assistant_target_hash:
                    assistant_proof_search_coordinator.mark_pack_consumed_by_solver(
                        self._pending_final_solver_assistant_target_hash,
                        role_id=role_id,
                        task_id=task_id,
                    )
                    self._pending_final_solver_assistant_target_hash = ""
                self.completed_task_ids.add(task_id)

                choices = response.get("choices") or []
                content = ""
                if choices:
                    message = choices[0].get("message") or {}
                    content = extract_message_text(message)
                parsed = parse_json(content)
                if isinstance(parsed, list):
                    parsed = parsed[0] if parsed else {}
                if isinstance(parsed, dict):
                    duration_ms = round((time.monotonic() - started) * 1000)
                    result_summary = self._summarize_model_call_result(role_id, task_id, parsed)
                    logger.info(
                        "Proof Solver model call result (role=%s, task=%s, phase=%s, duration_ms=%s, response_chars=%s): %s",
                        role_id,
                        task_id,
                        call_payload["phase"],
                        duration_ms,
                        len(content),
                        result_summary,
                    )
                    await self._broadcast(
                        "leanoj_model_call_completed",
                        {
                            **call_payload,
                            "duration_ms": duration_ms,
                            "response_chars": len(content),
                            "result_summary": result_summary,
                        },
                    )
                    return parsed
                raise ValueError("Proof Solver role returned JSON that was not an object.")
            except asyncio.CancelledError:
                raise
            except LeanOJConfigurationError:
                raise
            except RetryableProviderError as exc:
                duration_ms = round((time.monotonic() - started) * 1000)
                message = self._summarize_error(str(exc), limit=700)
                logger.warning(
                    "Proof Solver model call paused for retryable provider failure (role=%s, task=%s, phase=%s, duration_ms=%s): %s",
                    role_id,
                    task_id,
                    call_payload["phase"],
                    duration_ms,
                    message,
                )
                await self._broadcast(
                    "leanoj_model_call_failed",
                    {
                        **call_payload,
                        "duration_ms": duration_ms,
                        "retryable": True,
                        "reason": exc.reason or "transient_provider_error",
                        "message": message,
                    },
                )
                await api_client_manager.wait_for_retryable_provider_error(
                    exc,
                    role_id=exc.role_id or role_id,
                    should_stop=lambda: not self._running or self._stop_event.is_set(),
                )
                current_prompt = prompt
                continue
            except Exception as exc:
                duration_ms = round((time.monotonic() - started) * 1000)
                if is_provider_credit_pause_error(exc):
                    message = self._summarize_error(str(exc), limit=700)
                    logger.warning(
                        "Proof Solver model call paused for provider credits (role=%s, task=%s, phase=%s, duration_ms=%s): %s",
                        role_id,
                        task_id,
                        call_payload["phase"],
                        duration_ms,
                        message,
                    )
                    await self._broadcast(
                        "leanoj_model_call_failed",
                        {
                            **call_payload,
                            "duration_ms": duration_ms,
                            "retryable": True,
                            "reason": "openrouter_credit_exhaustion",
                            "message": message,
                        },
                    )
                    await self._pause_for_provider_credits(
                        role_id=role_id,
                        call_payload=call_payload,
                        message=message,
                        duration_ms=duration_ms,
                    )
                    current_prompt = prompt
                    continue
                if self._is_non_retryable_model_error(exc):
                    logger.error(
                        "Proof Solver model call failed with non-retryable error (role=%s, task=%s, phase=%s, duration_ms=%s): %s",
                        role_id,
                        task_id,
                        call_payload["phase"],
                        duration_ms,
                        exc,
                    )
                    await self._broadcast(
                        "leanoj_model_call_failed",
                        {
                            **call_payload,
                            "duration_ms": duration_ms,
                            "retryable": False,
                            "message": self._summarize_error(str(exc), limit=700),
                        },
                    )
                    raise LeanOJConfigurationError(str(exc)) from exc
                logger.warning(
                    "Proof Solver role %s task %s failed to produce valid JSON on retryable attempt %s: %s",
                    role_id,
                    task_id,
                    attempt_index,
                    exc,
                )
                error_summary = self._summarize_error(
                    f"Proof Solver role {role_id} returned unusable JSON on retryable attempt {attempt_index}: "
                    f"{type(exc).__name__}: {exc}",
                    limit=1200,
                )
                await self._broadcast(
                    "leanoj_model_call_failed",
                    {
                        **call_payload,
                        "duration_ms": duration_ms,
                        "retryable": True,
                        "message": error_summary,
                    },
                )
                if attempt_index == 1 or attempt_index % 3 == 0:
                    self._failed_feedback.append(
                        {
                            "request": f"{role_id} JSON generation",
                            "error_summary": error_summary,
                            "role_id": role_id,
                            "attempt": attempt_index,
                        }
                    )
                await self._persist_and_broadcast(
                    "leanoj_role_json_retrying",
                    {
                        "role_id": role_id,
                        "task_id": task_id,
                        "attempt": attempt_index,
                        "message": error_summary,
                    },
                )
                current_prompt = (
                    f"{prompt}\n\n"
                    "IMPORTANT - YOUR PREVIOUS RESPONSE WAS REJECTED BY THE JSON PARSER:\n"
                    "REJECTION REASON: INVALID_OR_TRUNCATED_JSON\n"
                    f"ISSUE: {type(exc).__name__}: {self._summarize_error(str(exc), limit=700)}\n"
                    "FIX REQUIRED:\n"
                    "- Return raw JSON only, with no markdown fences, commentary, or analysis.\n"
                    "- Start with `{` and end with `}`.\n"
                    "- Keep every string field concise enough to finish before max_tokens.\n"
                    "- Preserve the requested schema exactly.\n"
                    "- Escape Lean/LaTeX backslashes so the result is valid JSON.\n\n"
                    f"{self._json_retry_schema_hint(role_id)}"
                )
                if attempt_index % 3 == 0:
                    await asyncio.sleep(min(5.0, 0.5 * (attempt_index // 3)))
            finally:
                self.current_task_id = None
                self._refresh_workflow_tasks(task_prefix, role_id)

        raise asyncio.CancelledError()

    @staticmethod
    def _prompt_fits_role_budget(prompt: str, config: LeanOJRoleConfig) -> bool:
        max_input_tokens = rag_config.get_available_input_tokens(
            config.context_window,
            config.max_output_tokens,
        )
        return count_tokens(prompt) <= max_input_tokens

    @staticmethod
    def _missing_model_roles(request: LeanOJStartRequest) -> list[str]:
        role_configs: list[tuple[str, LeanOJRoleConfig]] = [
            ("topic_generator", request.topic_generator),
            ("topic_validator", request.topic_validator),
            ("brainstorm_validator", request.brainstorm_validator),
            ("final_solver", request.final_solver),
        ]
        if (request.assistant.model_id or "").strip():
            role_configs.append(("assistant", request.assistant))
        role_configs.extend(
            (f"brainstorm_submitter_{index}", submitter)
            for index, submitter in enumerate(request.brainstorm_submitters, start=1)
        )
        return [role_name for role_name, config in role_configs if not (config.model_id or "").strip()]

    def _next_task_id(self, prefix: str) -> str:
        current = self._task_sequences.get(prefix, 0)
        self._task_sequences[prefix] = current + 1
        return f"{prefix}_{current:03d}"

    def _refresh_workflow_tasks(self, active_prefix: str = "leanoj_topic", active_role: str = "LeanOJ") -> None:
        submitter_count = max(1, len(self._request.brainstorm_submitters) if self._request else 1)
        brainstorm_submitter_patterns = [
            (f"leanoj_brainstorm_sub{index}", f"Brainstorm Submitter {index}", "Cumulative Brainstorm")
            for index in range(1, submitter_count + 1)
        ]
        pattern = [
            ("leanoj_topic", "Topic Generator", "Topic Selection"),
            ("leanoj_topic_val", "Topic Validator", "Topic Validation"),
            *brainstorm_submitter_patterns,
            ("leanoj_brainstorm_val", "Brainstorm Validator", "Brainstorm Validation"),
            ("leanoj_brainstorm_prune", "Brainstorm Prune Reviewer", "Brainstorm Pruning"),
            ("leanoj_brainstorm_prune_val", "Brainstorm Prune Validator", "Brainstorm Pruning"),
            ("leanoj_path", "Final Proof Solver", "Path Decision"),
            ("leanoj_path_val", "Path Validator", "Path Validation"),
            ("leanoj_final", "Final Solver", "Final Lean Loop"),
            ("leanoj_master_proof_edit_val", "Master Proof Edit Validator", "Final Lean Loop"),
            ("leanoj_final_review", "Final Solver Review", "Final Lean Loop"),
        ]
        tasks: list[WorkflowTask] = []
        start_seq = sum(self._task_sequences.values())
        for offset in range(20):
            prefix, role, mode = pattern[offset % len(pattern)]
            seq = self._task_sequences.get(prefix, 0) + offset
            task_id = f"{prefix}_{seq:03d}"
            tasks.append(
                WorkflowTask(
                    task_id=task_id,
                    sequence_number=start_seq + offset + 1,
                    role=active_role if prefix == active_prefix else role,
                    mode=mode,
                    provider="lm_studio",
                    active=prefix == active_prefix,
                    completed=task_id in self.completed_task_ids,
                )
            )
        self.workflow_tasks = tasks

    def _configure_roles(self, request: LeanOJStartRequest) -> None:
        assistant_config = request.assistant if (request.assistant.model_id or "").strip() else request.topic_validator
        self._configure_role("leanoj_topic_generator", request.topic_generator)
        self._configure_role("leanoj_topic_selector", request.topic_generator)
        self._configure_role("leanoj_topic_validator", request.topic_validator)
        self._configure_role("leanoj_path_validator", request.topic_validator)
        self._configure_role("leanoj_proof_novelty", request.topic_validator)
        self._configure_role("leanoj_brainstorm_validator", request.brainstorm_validator)
        self._configure_role("leanoj_master_proof_edit_validator", request.brainstorm_validator)
        self._configure_role("leanoj_final_solver", request.final_solver)
        self._configure_role("leanoj_assistant", assistant_config)
        for index, submitter in enumerate(request.brainstorm_submitters, start=1):
            self._configure_role(f"leanoj_topic_submitter_{index}", submitter)
            self._configure_role(f"leanoj_brainstorm_submitter_{index}", submitter)
            self._configure_role(f"leanoj_brainstorm_prune_reviewer_{index}", submitter)

    @staticmethod
    def _configure_role(role_id: str, config: LeanOJRoleConfig) -> None:
        api_client_manager.configure_role(
            role_id,
            ModelConfig(
                provider=config.provider,
                model_id=config.model_id,
                openrouter_model_id=config.model_id if config.provider == "openrouter" else None,
                openrouter_provider=config.openrouter_provider,
                openrouter_reasoning_effort=config.openrouter_reasoning_effort,
                lm_studio_fallback_id=config.lm_studio_fallback_id,
                context_window=config.context_window,
                max_output_tokens=config.max_output_tokens,
                supercharge_enabled=config.supercharge_enabled,
            ),
        )

    async def _persist_and_broadcast(self, event: str, data: Optional[dict[str, Any]] = None) -> None:
        self._state.updated_at = datetime.now()
        self._remember_active_phase()
        await self._persist_state()
        await self._broadcast(event, data or self.get_status())
        await self._broadcast("leanoj_status_updated", self.get_status())

    async def _persist_state(self) -> None:
        session_dir = self._session_dir()
        session_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_accepted_idea_records()
        payload = self.get_status()
        if self._request is not None:
            payload["request"] = self._request.model_dump(mode="json")
        payload["task_sequences"] = dict(self._task_sequences)
        payload["completed_task_ids"] = sorted(self.completed_task_ids)
        await leanoj_context_manager.write_session_artifacts(
            session_id=self._state.session_id,
            accepted_ideas=self._accepted_ideas,
            accepted_idea_records=self._accepted_idea_records,
            verified_subproofs=self._verified_subproof_dicts(),
            partial_proofs=self._partial_proofs,
            failed_subproofs=self._failed_context_dicts(),
            final_attempts=self._final_attempts,
            final_cycle_packets=self._final_cycle_packets,
        )
        async with aiofiles.open(session_dir / "state.json", "w", encoding="utf-8") as f:
            await f.write(json.dumps(payload, indent=2))

    def _session_dir(self) -> Path:
        session_id = self._state.session_id or "latest"
        return self._sessions_base_dir() / session_id

    @staticmethod
    def _sessions_base_dir() -> Path:
        return Path(system_config.data_dir) / "leanoj_sessions"

    def _find_latest_state_file(self) -> Optional[Path]:
        base = self._sessions_base_dir()
        if not base.exists():
            return None
        state_files = [path for path in base.glob("*/state.json") if path.is_file()]
        if not state_files:
            return None
        return max(state_files, key=lambda path: path.stat().st_mtime)

    def _find_best_resumable_state_file(self) -> Optional[Path]:
        """Prefer the most valuable interrupted session after process restart."""
        return self._find_best_state_file()

    def _find_best_matching_state_file(self, request: LeanOJStartRequest) -> Optional[Path]:
        """Prefer the most-progressed saved session for this exact Proof Solver problem."""
        return self._find_best_state_file(request)

    def _find_best_state_file(self, request: Optional[LeanOJStartRequest] = None) -> Optional[Path]:
        base = self._sessions_base_dir()
        if not base.exists():
            return None

        candidates: list[tuple[tuple[int, int, int, int, int, int, int, float], Path]] = []
        for path in base.glob("*/state.json"):
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if request is not None and not self._payload_matches_request(payload, request):
                continue
            if payload.get("final_solution"):
                continue
            phase = str(payload.get("phase") or "")
            if phase in _TERMINAL_PHASES:
                continue
            candidates.append((self._payload_progress_score(payload, path), path))

        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]

    @staticmethod
    def _payload_matches_request(payload: dict[str, Any], request: LeanOJStartRequest) -> bool:
        request_payload = payload.get("request")
        if not isinstance(request_payload, dict):
            return False
        return (
            str(request_payload.get("user_prompt") or "").strip() == request.user_prompt.strip()
            and str(request_payload.get("lean_template") or "").strip() == request.lean_template.strip()
        )

    @staticmethod
    def _payload_progress_score(payload: dict[str, Any], path: Path) -> tuple[int, int, int, int, int, int, int, float]:
        verified_subproofs = payload.get("verified_subproofs") or []
        validated_topics = payload.get("validated_topics") or []
        failed_subproofs = payload.get("failed_subproofs") or []
        accepted_count = int(payload.get("accepted_brainstorm_count") or 0)
        topic_count = len(validated_topics) if isinstance(validated_topics, list) else 0
        final_attempt_count = int(payload.get("final_attempt_count") or 0)
        master_proof_version = int(payload.get("master_proof_version") or 0)
        return (
            LeanOJCoordinator._payload_phase_rank(payload),
            master_proof_version,
            final_attempt_count,
            len(verified_subproofs) if isinstance(verified_subproofs, list) else 0,
            accepted_count,
            topic_count,
            len(failed_subproofs) if isinstance(failed_subproofs, list) else 0,
            path.stat().st_mtime,
        )

    @staticmethod
    def _payload_phase_rank(payload: dict[str, Any]) -> int:
        phase = str(payload.get("phase") or "")
        if phase in {"stopped", "error"}:
            last_active_phase = str(payload.get("last_active_phase") or "")
            inferred_rank = LeanOJCoordinator._infer_payload_phase_rank(payload)
            last_active_rank = _PHASE_PROGRESS_RANK.get(last_active_phase, 0)
            return max(inferred_rank, last_active_rank)
        return _PHASE_PROGRESS_RANK.get(phase, 0)

    @staticmethod
    def _infer_payload_phase_rank(payload: dict[str, Any]) -> int:
        if payload.get("master_proof_initialized") or int(payload.get("master_proof_version") or 0) > 0:
            return _PHASE_PROGRESS_RANK["final_proof_loop"]
        if int(payload.get("final_attempt_count") or 0) > 0 or payload.get("final_attempts"):
            return _PHASE_PROGRESS_RANK["final_proof_loop"]
        if payload.get("current_path_decision") == "solve_final_now":
            return _PHASE_PROGRESS_RANK["final_proof_loop"]
        if payload.get("verified_subproofs") or payload.get("failed_subproofs"):
            return _PHASE_PROGRESS_RANK["path_decision"]
        if int(payload.get("accepted_brainstorm_count") or 0) > 0 or payload.get("accepted_ideas"):
            return _PHASE_PROGRESS_RANK["initial_brainstorm"]
        if payload.get("selected_topic"):
            return _PHASE_PROGRESS_RANK["initial_brainstorm"]
        if payload.get("validated_topics"):
            return _PHASE_PROGRESS_RANK["initial_topic_candidates"]
        return _PHASE_PROGRESS_RANK["idle"]

    def _restore_from_payload(self, payload: dict[str, Any]) -> None:
        request_payload = payload.get("request")
        restored_request = LeanOJStartRequest.model_validate(request_payload) if request_payload else None

        self._state = LeanOJState.model_validate(payload)
        self._state.is_running = False
        master_proof_path = self._master_proof_path(self._state.session_id)
        if master_proof_path.exists():
            try:
                self._set_master_proof_metadata(master_proof_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Failed to restore Proof Solver master proof metadata from %s: %s", master_proof_path, exc)
        artifacts = leanoj_context_manager.load_session_artifacts(self._state.session_id)
        self._validated_topics = [str(item) for item in payload.get("validated_topics") or []]
        restored_accepted_ideas = [
            *[str(item) for item in payload.get("accepted_ideas") or []],
            *[str(item) for item in artifacts.get(ARTIFACT_ACCEPTED_IDEAS, [])],
        ]
        self._accepted_idea_records = [
            dict(item) for item in (payload.get("accepted_idea_records") or []) if isinstance(item, dict)
        ]
        artifact_idea_records = [
            dict(item)
            for item in artifacts.get("accepted_idea_records", [])
            if isinstance(item, dict)
        ]
        if artifact_idea_records:
            record_keys = {
                self._dict_record_key(record)
                for record in self._accepted_idea_records
            }
            for record in artifact_idea_records:
                content = str(record.get("content") or "")
                record_key = self._dict_record_key(record)
                if content.strip() and record_key not in record_keys:
                    self._accepted_idea_records.append(record)
                    record_keys.add(record_key)
        if self._accepted_idea_records:
            self._accepted_ideas = [
                str(record.get("content") or "")
                for record in self._accepted_idea_records
                if str(record.get("content") or "").strip()
            ]
            recorded_contents = set(self._accepted_ideas)
            self._accepted_ideas.extend(
                idea
                for idea in restored_accepted_ideas
                if str(idea).strip() and idea not in recorded_contents
            )
        else:
            self._accepted_ideas = self._dedupe_strings(restored_accepted_ideas)
        self._ensure_accepted_idea_records()
        if self._state.brainstorm_acceptance_events < len(self._accepted_ideas):
            self._state.brainstorm_acceptance_events = max(
                int(payload.get("brainstorm_acceptance_events") or 0),
                len(self._accepted_ideas),
            )
        self._failed_feedback = [
            dict(item) for item in (payload.get("failed_feedback") or []) if isinstance(item, dict)
        ]
        self._failed_feedback = self._dedupe_dict_records(
            [
                *self._failed_feedback,
                *[
                    dict(item)
                    for item in artifacts.get(ARTIFACT_FAILED_SUBPROOFS, [])
                    if isinstance(item, dict)
                ],
            ]
        )
        self._final_attempts = [
            dict(item) for item in (payload.get("final_attempts") or []) if isinstance(item, dict)
        ]
        self._final_attempts = self._dedupe_dict_records(
            [
                *[
                    dict(item)
                    for item in artifacts.get(ARTIFACT_FINAL_ATTEMPTS, [])
                    if isinstance(item, dict)
                ],
                *self._final_attempts,
            ]
        )
        self._final_context_events = [
            dict(item)
            for item in payload.get("final_context_events") or []
            if isinstance(item, dict)
        ][-50:]
        partial_proofs = [
            dict(item) for item in (payload.get("partial_proofs") or []) if isinstance(item, dict)
        ]
        persisted_partial_proofs = self._load_partial_proof_database(self._state.session_id)
        self._partial_proofs = self._dedupe_partial_proofs(
            [
                *partial_proofs,
                *persisted_partial_proofs,
                *[
                    dict(item)
                    for item in artifacts.get(ARTIFACT_PARTIAL_PROOFS, [])
                    if isinstance(item, dict)
                ],
            ]
        )
        verified_records = self._dedupe_dict_records(
            [
                *[item.model_dump(mode="json") for item in self._state.verified_subproofs],
                *[
                    dict(item)
                    for item in artifacts.get(ARTIFACT_VERIFIED_SUBPROOFS, [])
                    if isinstance(item, dict)
                ],
            ]
        )
        self._state.verified_subproofs = [
            LeanOJSubproofRecord.model_validate(item)
            for item in verified_records
        ]
        self._final_cycle_packets = self._dedupe_dict_records(
            [
                *[
                    dict(item)
                    for item in artifacts.get(ARTIFACT_FINAL_CYCLE_PACKETS, [])
                    if isinstance(item, dict)
                ],
                *[
                    dict(item)
                    for item in payload.get("final_cycle_packets") or []
                    if isinstance(item, dict)
                ],
            ]
        )
        current_packet = payload.get("current_final_cycle_packet")
        self._current_final_cycle_packet = dict(current_packet) if isinstance(current_packet, dict) else None
        working_packet = payload.get("current_working_proof_attempt")
        self._current_working_proof_attempt = dict(working_packet) if isinstance(working_packet, dict) else None
        self._task_sequences = {
            str(key): int(value)
            for key, value in (payload.get("task_sequences") or {}).items()
            if isinstance(value, int) or str(value).isdigit()
        }
        self.completed_task_ids = {str(item) for item in payload.get("completed_task_ids") or []}
        self.workflow_tasks = []
        self.current_task_id = None
        self._stop_event = asyncio.Event()
        self._request = restored_request
        self._running = False
        self._restored_from_disk = True
        self._reset_master_proof_progress_watchdog()

        if self._request is not None:
            self._configure_roles(self._request)

    def _should_stop(self) -> bool:
        return self._stop_event.is_set()

    def _begin_brainstorm_acceptance_phase(self, phase_key: str) -> None:
        if self._state.active_brainstorm_phase != phase_key:
            self._state.active_brainstorm_phase = phase_key
            self._state.active_brainstorm_start_count = self._state.brainstorm_acceptance_events
            self._state.active_brainstorm_last_sufficiency_check_count = 0
            self._state.active_brainstorm_last_prune_review_count = 0

    def _get_brainstorm_acceptance_start(self, phase_key: str) -> int:
        if self._state.active_brainstorm_phase != phase_key:
            self._begin_brainstorm_acceptance_phase(phase_key)
        if self._state.active_brainstorm_start_count > self._state.brainstorm_acceptance_events:
            self._state.active_brainstorm_start_count = self._state.brainstorm_acceptance_events
        return self._state.active_brainstorm_start_count

    def _finish_brainstorm_acceptance_phase_for_path_decision(self) -> None:
        self._state.phase = "path_decision"
        self._state.active_brainstorm_phase = ""
        self._state.active_brainstorm_start_count = self._state.brainstorm_acceptance_events
        self._state.active_brainstorm_last_sufficiency_check_count = 0
        self._state.active_brainstorm_last_prune_review_count = 0

    @staticmethod
    def _is_non_retryable_model_error(exc: Exception) -> bool:
        return is_non_retryable_model_error(exc)

    def _remember_active_phase(self) -> None:
        if self._state.phase in _ACTIVE_PHASES:
            self._state.last_active_phase = self._state.phase

    def _infer_resume_phase(self) -> str:
        if (
            self._current_working_proof_attempt
            or self._state.master_proof_initialized
            or self._state.master_proof_version > 0
        ):
            return "final_proof_loop"
        if self._state.final_attempt_count > 0 or self._final_attempts:
            return "final_proof_loop"
        if self._state.current_path_decision == "solve_final_now":
            return "final_proof_loop"
        if self._state.last_active_phase in _ACTIVE_PHASES:
            return self._state.last_active_phase
        if self._state.verified_subproofs or self._state.failed_subproofs:
            return "path_decision"
        if self._accepted_ideas or self._state.accepted_brainstorm_count > 0 or self._state.selected_topic:
            return "initial_brainstorm"
        if self._validated_topics:
            return "initial_topic_candidates"
        return "initial_topic_candidates"

    @staticmethod
    def _summarize_error(error_text: str, limit: int = 800) -> str:
        cleaned = " ".join((error_text or "").split())
        return cleaned[:limit] + ("..." if len(cleaned) > limit else "")


leanoj_coordinator = LeanOJCoordinator()
