"""
Orchestrates proof identification, Lean 4 attempts, retry handling, and novelty checks.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from backend.autonomous.agents.lemma_search_agent import MathlibLemmaSearchAgent
from backend.autonomous.agents.proof_formalization_agent import ProofFormalizationAgent
from backend.autonomous.agents.proof_identification_agent import ProofIdentificationAgent
from backend.autonomous.agents.proof_candidate_list_validator import (
    ProofCandidateListContextError,
    ProofCandidateListValidator,
)
from backend.autonomous.agents.proof_pruning_agent import proof_run_role_suffix
from backend.autonomous.memory.brainstorm_memory import brainstorm_memory
from backend.autonomous.memory.paper_library import paper_library
from backend.autonomous.memory.proof_database import is_prompt_injection_novel_tier
from backend.autonomous.core.proof_registration import register_verified_lean_proof
from backend.shared.config import system_config
from backend.shared.context_overflow import (
    CONTEXT_OVERFLOW_RESOLUTION,
    CONTEXT_OVERFLOW_STOP_REASON,
    context_overflow_model_payload,
)
from backend.shared.lean_proof_integrity import validate_full_lean_proof_integrity
from backend.shared.model_error_utils import (
    format_transient_provider_error,
    is_non_retryable_model_error,
    is_transient_model_call_error,
)
from backend.shared.api_client_manager import RetryableProviderError, api_client_manager
from backend.shared.models import (
    ProofAttemptFeedback,
    ProofAttemptResult,
    ProofCandidate,
    ProofCandidateListRejection,
    ProofCandidateListValidation,
    ProofPruneContextPressure,
    ProofStageResult,
    SmtHint,
)
from backend.shared.openrouter_client import FreeModelExhaustedError
from backend.shared.provider_errors import ProviderContextLengthError
from backend.shared.provider_pause import is_provider_credit_pause_error
from backend.shared.smt_client import get_smt_client
from backend.shared.utils import count_tokens
from .proof_dependency_extractor import ProofDependencyExtractor

logger = logging.getLogger(__name__)

BroadcastFn = Optional[Callable[[str, dict[str, Any]], Awaitable[None]]]
ShouldStopFn = Optional[Callable[[], bool]]
ProofCheckpointCallback = Optional[Callable[[dict[str, Any]], Awaitable[None]]]
ProofAppendCallback = Optional[Callable[[Any], Awaitable[None]]]
ProofPruningRegisteredCallback = Optional[
    Callable[[Any, dict[str, Any]], Awaitable[None]]
]
ProofPruningPressureCallback = Optional[
    Callable[..., Awaitable[None]]
]
LEAN_WORKSPACE_ERROR_PREFIX = "LEAN 4 WORKSPACE ERROR"
PROOF_TRUNCATION_STOP_REASON = "proof_output_truncation_recovery_exhausted"
PROOF_TRUNCATION_POLICY_VERSION = "proof-truncation-v1"


def _candidate_fingerprint(candidate: ProofCandidate, source_type: str, source_id: str) -> str:
    normalized = "\n".join(
        [
            source_type.strip().lower(),
            source_id.strip().lower(),
            " ".join((candidate.statement or "").split()).lower(),
            " ".join((candidate.formal_sketch or "").split()).lower(),
            candidate.theorem_id.strip(),
            candidate.expected_novelty_tier.strip().lower(),
            " ".join((candidate.prompt_relevance_rationale or "").split()).lower(),
            " ".join((candidate.novelty_rationale or "").split()).lower(),
            " ".join((candidate.why_not_standard_known_result or "").split()).lower(),
        ]
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _candidate_list_fingerprint(
    candidates: list[ProofCandidate],
    source_type: str,
    source_id: str,
) -> str:
    material = "\n".join(
        _candidate_fingerprint(candidate, source_type, source_id)
        for candidate in candidates
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _candidate_list_review_scope(
    *,
    source_type: str,
    source_id: str,
    run_id: str,
    trigger: str,
    proof_round_index: int,
    proof_run_context: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Return the exact ownership fence for one candidate-list review round."""
    context = dict(proof_run_context or {})
    return {
        "source_type": source_type,
        "source_id": source_id,
        "run_id": run_id,
        "trigger": trigger,
        "proof_round_index": proof_round_index,
        "proof_run_id": str(context.get("proof_run_id") or ""),
    }


def _normalize_candidate_list_checkpoint(
    raw_state: Optional[dict[str, Any]],
    *,
    expected_scope: dict[str, Any],
    source_type: str,
    source_id: str,
) -> dict[str, Any]:
    """Validate candidate-list checkpoint authority atomically or discard it."""
    if not isinstance(raw_state, dict) or not raw_state:
        return {}
    if raw_state.get("review_scope") != expected_scope:
        return {}
    status = str(raw_state.get("status") or "")
    if status not in {"reviewing", "approved", "rejected"}:
        return {}
    try:
        generation_attempt = int(raw_state.get("generation_attempt") or 1)
    except (TypeError, ValueError):
        return {}
    if generation_attempt < 1:
        return {}
    raw_proposed = raw_state.get("proposed_candidates")
    if not isinstance(raw_proposed, list):
        return {}
    proposed: list[ProofCandidate] = []
    try:
        proposed = [
            ProofCandidate.model_validate(item)
            for item in raw_proposed
            if isinstance(item, dict)
        ]
    except Exception:
        return {}
    if len(proposed) != len(raw_proposed):
        return {}
    proposed_ids = [candidate.theorem_id for candidate in proposed]
    if len(set(proposed_ids)) != len(proposed_ids):
        return {}
    expected_fingerprint = (
        _candidate_list_fingerprint(proposed, source_type, source_id)
        if proposed
        else ""
    )
    if str(raw_state.get("list_fingerprint") or "") != expected_fingerprint:
        return {}
    raw_approved_ids = raw_state.get("approved_candidate_ids")
    if not isinstance(raw_approved_ids, list):
        return {}
    approved_ids = [str(item) for item in raw_approved_ids]
    if len(set(approved_ids)) != len(approved_ids):
        return {}
    if any(theorem_id not in proposed_ids for theorem_id in approved_ids):
        return {}
    raw_rejections = raw_state.get("semantic_rejections")
    if not isinstance(raw_rejections, list):
        return {}
    try:
        semantic_rejections = [
            ProofCandidateListRejection.model_validate(item)
            for item in raw_rejections
            if isinstance(item, dict)
        ]
    except Exception:
        return {}
    if len(semantic_rejections) != len(raw_rejections):
        return {}
    normalized = {
        **raw_state,
        "status": status,
        "review_scope": dict(expected_scope),
        "generation_attempt": generation_attempt,
        "list_fingerprint": expected_fingerprint,
        "proposed_candidates": [
            candidate.model_dump(mode="json") for candidate in proposed
        ],
        "approved_candidate_ids": approved_ids,
        "semantic_rejections": [
            item.model_dump(mode="json") for item in semantic_rejections[-5:]
        ],
    }
    if status == "approved":
        raw_validation = raw_state.get("validation")
        if not isinstance(raw_validation, dict):
            return {}
        try:
            validation = ProofCandidateListValidation.model_validate(raw_validation)
        except Exception:
            return {}
        result_ids = [item.theorem_id for item in validation.results]
        if result_ids != proposed_ids:
            return {}
        validated_approved_ids = [
            item.theorem_id
            for item in validation.results
            if item.decision == "approve_novel"
        ]
        if validated_approved_ids != approved_ids:
            return {}
        if not ProofCandidateListValidator.threshold_met(
            approved_count=len(approved_ids),
            proposed_count=len(proposed_ids),
        ):
            return {}
        normalized["validation"] = validation.model_dump(mode="json")
    return normalized


def _truncation_chain_exhausted(attempts: list[ProofAttemptFeedback]) -> bool:
    return len(attempts) >= 5 and all(
        attempt.failure_kind == "output_truncated" and not attempt.lean_code
        for attempt in attempts
    )


@dataclass
class _LeanVerificationOutcome:
    """Outcome of a single candidate's Lean 4 formalization pipeline (Phase A)."""
    candidate: ProofCandidate
    proof_label: str
    success: bool
    theorem_name: str
    lean_code: str
    attempts: list[ProofAttemptFeedback] = field(default_factory=list)
    context_overflow_payload: dict[str, Any] = field(default_factory=dict)


class ProofVerificationProviderPause(Exception):
    """Raised when proof verification must pause for provider retry/resume."""

    def __init__(self, message: str, remaining_candidates: Optional[list[ProofCandidate]] = None):
        super().__init__(message)
        self.remaining_candidates = remaining_candidates or []


class ProofVerificationStage:
    """Run the full proof-verification checkpoint pipeline."""

    _active_sources: dict[str, str] = {}
    _active_sources_lock: Optional[asyncio.Lock] = None

    def __init__(self, solution_path_manager: Any = None) -> None:
        self._novelty_task_sequence = 0
        self._integrity_task_sequence = 0
        self._dependency_extractor = ProofDependencyExtractor()
        self.solution_path_manager = solution_path_manager

    @staticmethod
    def _proof_workflow_mode(trigger: str) -> str:
        if trigger == "manual_compiler_save":
            return "compiler"
        if trigger == "manual_compiler_aggregator":
            return "aggregator"
        if trigger == "manual":
            return "manual_proof_check"
        return "autonomous"

    @classmethod
    def _get_active_sources_lock(cls) -> asyncio.Lock:
        if cls._active_sources_lock is None:
            cls._active_sources_lock = asyncio.Lock()
        return cls._active_sources_lock

    @classmethod
    def _source_key(cls, source_type: str, source_id: str) -> str:
        return f"{source_type}:{source_id}"

    @classmethod
    async def is_source_running(cls, source_type: str, source_id: str) -> bool:
        async with cls._get_active_sources_lock():
            return cls._source_key(source_type, source_id) in cls._active_sources

    @classmethod
    async def active_source_keys(cls) -> set[str]:
        """Return a snapshot of currently reserved proof source keys."""
        async with cls._get_active_sources_lock():
            return set(cls._active_sources)

    @classmethod
    async def reserve_source(
        cls,
        source_type: str,
        source_id: str,
        owner_token: str = "",
    ) -> str:
        """Reserve a source before background execution begins."""
        resolved_token = owner_token or uuid.uuid4().hex
        await cls._acquire_source(source_type, source_id, resolved_token)
        return resolved_token

    @classmethod
    async def release_source(
        cls,
        source_type: str,
        source_id: str,
        owner_token: str = "",
    ) -> bool:
        """Release a reservation only when the caller presents its owner token."""
        return await cls._release_source(source_type, source_id, owner_token)

    @classmethod
    async def _acquire_source(
        cls,
        source_type: str,
        source_id: str,
        owner_token: str = "",
    ) -> None:
        async with cls._get_active_sources_lock():
            source_key = cls._source_key(source_type, source_id)
            if source_key in cls._active_sources:
                raise RuntimeError(f"Proof verification already running for {source_type} {source_id}")
            cls._active_sources[source_key] = owner_token or uuid.uuid4().hex

    @classmethod
    async def _release_source(
        cls,
        source_type: str,
        source_id: str,
        owner_token: str = "",
    ) -> bool:
        async with cls._get_active_sources_lock():
            source_key = cls._source_key(source_type, source_id)
            current_owner = cls._active_sources.get(source_key)
            if current_owner is None:
                return False
            if not owner_token or owner_token != current_owner:
                return False
            cls._active_sources.pop(source_key, None)
            return True

    async def _broadcast(self, broadcast_fn: BroadcastFn, event: str, data: dict[str, Any]) -> None:
        if broadcast_fn:
            await broadcast_fn(event, data)

    @staticmethod
    def _role_suffix(
        source_type: str,
        override: Optional[str] = None,
        proof_run_context: Optional[dict[str, Any]] = None,
    ) -> str:
        if override:
            base = override
        else:
            base = "brainstorm" if source_type == "brainstorm" else "paper"
        proof_run_id = str((proof_run_context or {}).get("proof_run_id", "") or "")
        if not proof_run_id:
            return base
        scope = "manual" if base.startswith("manual_") else "autonomous"
        return f"{base}_{proof_run_role_suffix(scope, proof_run_id)}"

    @staticmethod
    def _summarize_error(error_text: str, limit: int = 500) -> str:
        raw = error_text or ""
        if not raw.strip():
            return ""

        # Surface placeholder-rejection banners unchanged. These come from the
        # Lean 4 client when a proof used `sorry`/`admit` or otherwise would
        # have passed Lean with only a warning. The model must see the full
        # rejection reason on retries, not a whitespace-collapsed fragment.
        if "PROOF REJECTED: PLACEHOLDER USED" in raw:
            cleaned = raw.strip()
            return cleaned[:limit] + ("..." if len(cleaned) > limit else "")

        # Surface real Lean 4 errors (and their trailing context) before
        # deprecation warnings so retry prompts and the UI see the actual
        # failure reason instead of a truncated `warning: ... deprecated` line.
        lines = raw.splitlines()
        error_pattern = re.compile(r":\s*error\s*:", re.IGNORECASE)
        error_indices = [idx for idx, line in enumerate(lines) if error_pattern.search(line)]

        if error_indices:
            ordered_lines: list[str] = []
            seen: set[int] = set()
            for idx in error_indices:
                for offset in range(idx, min(len(lines), idx + 4)):
                    if offset in seen:
                        continue
                    seen.add(offset)
                    ordered_lines.append(lines[offset])
            for idx, line in enumerate(lines):
                if idx in seen:
                    continue
                seen.add(idx)
                ordered_lines.append(line)
            raw = "\n".join(ordered_lines)

        cleaned = " ".join(raw.split())
        return cleaned[:limit] + ("..." if len(cleaned) > limit else "")

    @staticmethod
    def _proof_label_for_index(index: int) -> str:
        """Return Proof A..Z, then AA..ZZ, then AAA.. for a 1-based index."""
        safe_index = max(1, int(index or 1))
        label = ""
        while safe_index:
            safe_index, remainder = divmod(safe_index - 1, 26)
            label = chr(ord("A") + remainder) + label
        return label

    @staticmethod
    def _should_append_verified_proof(
        *,
        is_novel: bool,
        duplicate: bool,
        append_proof_callback: ProofAppendCallback,
        append_known_proofs: bool = False,
    ) -> bool:
        """Decide whether a verified proof should be written into the source appendix.

        Automatic checkpoints keep the source appendix novelty-focused. User
        triggered/manual checks append every verified proof so the operator can
        see the exact Lean result they requested, even when novelty is low.
        """
        if append_known_proofs:
            return True
        if not is_novel:
            return False
        return bool(not duplicate or append_proof_callback is not None)

    @staticmethod
    def _should_append_known_proofs_for_trigger(trigger: str) -> bool:
        """Known proofs are appended only for explicit user/manual proof checks."""
        return trigger in {"manual", "manual_compiler_aggregator"}

    def _lean_response_summary(self, feedback: ProofAttemptFeedback) -> str:
        if feedback.success:
            return "Lean 4 response: proof verified."
        error_summary = self._summarize_error(feedback.error_output, limit=960)
        if error_summary:
            if "timed out after" in error_summary.lower() and "Advanced Settings" not in error_summary:
                error_summary = f"{error_summary} You can change this timeout in Advanced Settings."
            return (
                "Lean 4 proof-attempt feedback (not a MOTO system error): "
                f"{error_summary} The model uses these diagnostics if another attempt follows. "
                "Proof not verified."
            )
        return (
            "Lean 4 proof-attempt feedback (not a MOTO system error): proof not verified. "
            "The model uses these diagnostics if another attempt follows."
        )

    @staticmethod
    def _extract_suggested_lemma_targets(error_text: str) -> list[str]:
        targets: list[str] = []
        for pattern in (
            r"unknown (?:constant|identifier)\s+'?([A-Za-z][A-Za-z0-9_'.]*)'?",
            r"failed to synthesize\s+([A-Za-z][A-Za-z0-9_'.]*)",
        ):
            for match in re.findall(pattern, error_text or "", flags=re.IGNORECASE):
                candidate = str(match or "").strip()
                if candidate and candidate not in targets:
                    targets.append(candidate)
        return targets[:6]

    @staticmethod
    def _extract_theorem_name_from_lean(lean_code: str) -> str:
        match = re.search(
            r"\b(?:theorem|lemma)\s+([A-Za-z_][A-Za-z0-9_'.]*)",
            lean_code or "",
        )
        return match.group(1) if match else ""

    @staticmethod
    def _is_smt_amenable(candidate: ProofCandidate) -> bool:
        text = f"{candidate.statement}\n{candidate.formal_sketch}".lower()
        if not text.strip():
            return False

        blocked_markers = (
            "forall",
            "for all",
            "there exists",
            "exists",
            "∃",
            "∀",
            "set",
            "finset",
            "topological",
            "continuous",
            "measure",
            "category",
            "functor",
            "matrix",
            "module",
            "vector",
            "group",
            "monoid",
            "ring_hom",
            "filter",
        )
        if any(marker in text for marker in blocked_markers):
            return False

        arithmetic_markers = (
            "nat",
            "int",
            "real",
            "integer",
            "arithmetic",
            "linear",
            "inequal",
            "=",
            "<",
            ">",
            "≤",
            "≥",
            "+",
            "-",
            "*",
        )
        return any(marker in text for marker in arithmetic_markers)

    @staticmethod
    def _build_smt_tactic_suggestions(candidate: ProofCandidate) -> list[str]:
        text = f"{candidate.statement}\n{candidate.formal_sketch}".lower()
        suggestions: list[str] = []

        if any(token in text for token in ("nat", "int")):
            suggestions.extend(["omega", "norm_num"])
        if any(token in text for token in ("real", "linear", "inequal", "≤", "≥", "<", ">")):
            suggestions.extend(["linarith", "polyrith"])
        if "=" in text or "decidable" in text:
            suggestions.extend(["nativeDecide", "decide"])

        deduped: list[str] = []
        for suggestion in suggestions:
            if suggestion not in deduped:
                deduped.append(suggestion)
        return deduped

    @staticmethod
    def _first_attempt_used_smt_hint(
        attempts: list[ProofAttemptFeedback],
        smt_hint: Optional[SmtHint],
    ) -> bool:
        if not attempts or not smt_hint or smt_hint.result != "unsat" or not smt_hint.suggested_tactics:
            return False

        first_attempt = attempts[0]
        if not first_attempt.success or first_attempt.attempt != 1:
            return False

        haystack = "\n".join(
            [
                first_attempt.lean_code or "",
                "\n".join(first_attempt.tactic_trace or []),
            ]
        ).lower()
        return any(tactic.lower() in haystack for tactic in smt_hint.suggested_tactics)

    async def _run_smt_check(
        self,
        *,
        user_prompt: str,
        source_type: str,
        source_id: str,
        base_event: dict[str, Any],
        candidate: ProofCandidate,
        proof_label: str,
        source_content: str,
        source_title: str,
        identification_agent: ProofIdentificationAgent,
        broadcast_fn: BroadcastFn,
    ) -> Optional[SmtHint]:
        if not system_config.smt_enabled or not self._is_smt_amenable(candidate):
            return None

        started_at = time.monotonic()
        try:
            smtlib = await identification_agent.translate_candidate_to_smt(
                user_research_prompt=user_prompt,
                source_type=source_type,
                theorem_candidate=candidate,
                source_content=source_content,
                source_title=source_title,
            )
            if not smtlib:
                return SmtHint(result="unknown", suggested_tactics=[], smtlib="")

            smt_result = await get_smt_client().check_smt2(
                smtlib,
                timeout=system_config.smt_timeout,
            )
            result_name = smt_result.result if smt_result.result in {"sat", "unsat", "unknown"} else "unknown"
            suggestions = self._build_smt_tactic_suggestions(candidate) if result_name == "unsat" else []
            z3_raw = "\n".join(part for part in [smt_result.stdout.strip(), smt_result.stderr.strip()] if part).strip()
            return SmtHint(
                result=result_name,
                suggested_tactics=suggestions,
                smtlib=smtlib,
                z3_output=z3_raw[:2000],
            )
        except Exception as exc:
            if (
                is_non_retryable_model_error(exc)
                or isinstance(exc, RetryableProviderError)
                or is_transient_model_call_error(exc)
            ):
                raise
            logger.debug("SMT check failed for theorem %s in %s %s: %s", candidate.theorem_id, source_type, source_id, exc)
            elapsed_ms = int((time.monotonic() - started_at) * 1000)
            await self._broadcast(
                broadcast_fn,
                "smt_check_error",
                {
                    **base_event,
                    "theorem_id": candidate.theorem_id,
                    "theorem_statement": candidate.statement,
                    "proof_label": proof_label,
                    "error_summary": self._summarize_error(str(exc), limit=960),
                    "elapsed_ms": elapsed_ms,
                },
            )
            return SmtHint(result="unknown", suggested_tactics=[], smtlib="")

    async def _resolve_candidates(
        self,
        *,
        theorem_candidates: Optional[list[ProofCandidate]],
        identification_agent: ProofIdentificationAgent,
        user_prompt: str,
        source_type: str,
        source_id: str,
        source_title: str,
        content: str,
        proof_round_index: int = 1,
        proof_max_rounds: int = 1,
        prior_round_results: str = "",
    ) -> list[ProofCandidate]:
        if theorem_candidates is not None:
            return theorem_candidates

        has_candidates, resolved_candidates = await identification_agent.identify_candidates(
            user_research_prompt=user_prompt,
            source_type=source_type,
            source_id=source_id,
            source_content=content,
            source_title=source_title,
            proof_round_index=proof_round_index,
            proof_max_rounds=proof_max_rounds,
            prior_round_results=prior_round_results,
        )
        return resolved_candidates if has_candidates else []

    async def _prepare_candidate(
        self,
        *,
        user_prompt: str,
        source_type: str,
        theorem_candidate: ProofCandidate,
        source_content: str,
        source_title: str,
        lemma_search_agent: MathlibLemmaSearchAgent,
    ) -> ProofCandidate:
        source_excerpt = theorem_candidate.source_excerpt or ProofFormalizationAgent._build_source_excerpt(
            theorem_candidate.statement,
            source_content,
        )
        candidate = theorem_candidate.model_copy(update={"source_excerpt": source_excerpt})
        relevant_lemmas = await lemma_search_agent.suggest_relevant_lemmas(
            user_research_prompt=user_prompt,
            source_type=source_type,
            theorem_candidate=candidate,
            source_content=source_content,
            source_title=source_title,
        )
        if relevant_lemmas:
            candidate = candidate.model_copy(update={"relevant_lemmas": relevant_lemmas})
        return candidate

    async def run(
        self,
        content: str,
        source_type: str,
        source_id: str,
        user_prompt: str,
        submitter_model: str,
        submitter_context: int,
        submitter_max_tokens: int,
        validator_model: str,
        validator_context: int,
        validator_max_tokens: int,
        broadcast_fn: BroadcastFn,
        novel_proofs_db,
        source_title: str = "",
        theorem_candidates: Optional[list[ProofCandidate]] = None,
        role_suffix_override: Optional[str] = None,
        trigger: str = "automatic",
        source_reserved: bool = False,
        source_reservation_token: str = "",
        release_source_on_exit: bool = True,
        should_stop: ShouldStopFn = None,
        append_to_source: bool = True,
        append_proof_callback: ProofAppendCallback = None,
        proof_candidate_indexes: Optional[dict[str, int]] = None,
        checkpoint_attempts_by_candidate: Optional[dict[str, list[ProofAttemptFeedback]]] = None,
        checkpoint_theorem_names_by_candidate: Optional[dict[str, str]] = None,
        checkpoint_truncation_streak: Optional[list[dict[str, Any]]] = None,
        checkpoint_result: Optional[ProofStageResult] = None,
        checkpoint_candidate_list_state: Optional[dict[str, Any]] = None,
        checkpoint_processed_candidate_ids: Optional[list[str]] = None,
        checkpoint_callback: ProofCheckpointCallback = None,
        proof_round_index: int = 1,
        proof_max_rounds: int = 1,
        prior_round_results: str = "",
        canonical_user_prompt: str = "",
        run_id: str = "",
        terminal_truncation_stop_enabled: bool = True,
        proof_run_context: Optional[dict[str, Any]] = None,
        proof_pruning_registered_callback: ProofPruningRegisteredCallback = None,
        proof_pruning_pressure_callback: ProofPruningPressureCallback = None,
        proof_pruning_route_fingerprint: str = "",
    ) -> ProofStageResult:
        """Run proof identification, formalization, Lean 4 checking, and novelty review."""
        result = (
            checkpoint_result.model_copy(deep=True)
            if checkpoint_result is not None
            else ProofStageResult(source_type=source_type, source_id=source_id)
        )
        resolved_candidates: list[ProofCandidate] = []
        candidate_indexes: dict[str, int] = dict(proof_candidate_indexes or {})
        processed_candidate_ids: set[str] = {
            str(candidate_id)
            for candidate_id in (checkpoint_processed_candidate_ids or [])
            if candidate_id
        }
        attempts_by_candidate: dict[str, list[ProofAttemptFeedback]] = {
            theorem_id: list(attempts or [])
            for theorem_id, attempts in (checkpoint_attempts_by_candidate or {}).items()
        }
        theorem_names_by_candidate: dict[str, str] = {
            theorem_id: str(theorem_name or "")
            for theorem_id, theorem_name in (checkpoint_theorem_names_by_candidate or {}).items()
            if theorem_name
        }
        truncation_streak: list[dict[str, Any]] = list(checkpoint_truncation_streak or [])
        checkpoint_state_lock = asyncio.Lock()
        abort_event = asyncio.Event()
        checkpoint_revision = 0
        candidate_list_review_active = False
        candidate_list_review_attempt = 0
        candidate_list_review_count = 0
        candidate_list_scope = _candidate_list_review_scope(
            source_type=source_type,
            source_id=source_id,
            run_id=run_id,
            trigger=trigger,
            proof_round_index=proof_round_index,
            proof_run_context=proof_run_context,
        )
        candidate_list_state = _normalize_candidate_list_checkpoint(
            checkpoint_candidate_list_state,
            expected_scope=candidate_list_scope,
            source_type=source_type,
            source_id=source_id,
        )

        async def save_checkpoint(status: str) -> None:
            nonlocal checkpoint_revision
            if checkpoint_callback is None:
                return
            async with checkpoint_state_lock:
                if not resolved_candidates and status not in {
                    "complete",
                    "error",
                    "no_candidates",
                    "candidate_list_rejected",
                    "stopped",
                }:
                    return
                checkpoint_revision += 1
                payload = {
                    "checkpoint_revision": checkpoint_revision,
                    "lifecycle_generation": int(
                        (proof_run_context or {}).get("lifecycle_generation") or 0
                    ),
                    "recovery_policy_version": PROOF_TRUNCATION_POLICY_VERSION,
                    "source_type": source_type,
                    "source_id": source_id,
                    "source_title": source_title,
                    "trigger": trigger,
                    "proof_round_index": proof_round_index,
                    "proof_max_rounds": proof_max_rounds,
                    "prior_round_results": prior_round_results,
                    "status": status,
                    "candidates": [
                        {
                            "index": candidate_indexes.get(candidate.theorem_id, index),
                            "candidate": candidate.model_dump(mode="json"),
                        }
                        for index, candidate in enumerate(list(resolved_candidates), start=1)
                    ],
                    "processed_candidate_ids": sorted(processed_candidate_ids),
                    "deferred_candidate_ids": list(result.deferred_candidate_ids),
                    "context_overflow_payload": dict(result.context_overflow_payload),
                    "attempts_by_candidate": {
                        theorem_id: [
                            attempt.model_dump(mode="json")
                            for attempt in list(attempts)
                        ]
                        for theorem_id, attempts in list(attempts_by_candidate.items())
                    },
                    "theorem_names_by_candidate": dict(theorem_names_by_candidate),
                    "results": [
                        proof_result.model_dump(mode="json")
                        for proof_result in list(result.results)
                    ],
                    "total_candidates": result.total_candidates,
                    "verified_count": result.verified_count,
                    "novel_count": result.novel_count,
                    "truncation_streak": list(truncation_streak),
                    "fatal_stop_reason": result.fatal_stop_reason,
                    "fatal_stop_payload": dict(result.fatal_stop_payload),
                    "candidate_list_review": dict(candidate_list_state),
                }
                # Keep persistence in the same critical section as snapshot
                # creation so a slower older write cannot land after a newer
                # concurrent candidate snapshot.
                await checkpoint_callback(payload)

        def _stop_requested() -> bool:
            if abort_event.is_set():
                return True
            if should_stop is None:
                return False
            try:
                return bool(should_stop())
            except Exception:
                return False

        async def broadcast_candidate_list_interrupted(
            *,
            error_kind: str,
            error_message: str,
        ) -> None:
            nonlocal candidate_list_review_active
            if not candidate_list_review_active:
                return
            candidate_list_review_active = False
            await self._broadcast(
                broadcast_fn,
                "proof_candidate_list_review_interrupted",
                {
                    **base_event,
                    "list_attempt": candidate_list_review_attempt,
                    "proposed_count": candidate_list_review_count,
                    "error_kind": error_kind,
                    "message": (
                        "Validator proof-list review was interrupted without an "
                        "acceptance or semantic rejection. "
                        f"{self._summarize_error(error_message, limit=600)}"
                    ),
                },
            )
        owned_reservation_token = source_reservation_token
        if not source_reserved:
            owned_reservation_token = uuid.uuid4().hex
            await self._acquire_source(
                source_type,
                source_id,
                owned_reservation_token,
            )
        try:
            base_event = {
                "source_type": source_type,
                "source_id": source_id,
                "source_title": source_title,
                "run_id": run_id,
                "trigger": trigger,
                "proof_round_index": proof_round_index,
                "proof_max_rounds": proof_max_rounds,
                **dict(proof_run_context or {}),
            }
            await self._broadcast(
                broadcast_fn,
                "proof_check_started",
                base_event,
            )

            if not system_config.lean4_enabled:
                await self._broadcast(
                    broadcast_fn,
                    "proof_check_complete",
                    {
                        **base_event,
                        "novel_count": 0,
                        "verified_count": 0,
                        "total_candidates": 0,
                        "message": "Lean 4 is disabled; proof verification was skipped.",
                    },
                )
                return result

            role_suffix = self._role_suffix(
                source_type,
                role_suffix_override,
                proof_run_context,
            )
            novelty_role_id = f"autonomous_proof_novelty_{role_suffix}"
            list_validator = ProofCandidateListValidator(
                model_id=validator_model,
                context_window=validator_context,
                max_output_tokens=validator_max_tokens,
                role_id=f"autonomous_proof_candidate_list_validator_{role_suffix}",
            )
            candidate_list_validation_prompt = canonical_user_prompt or user_prompt
            identification_agent = ProofIdentificationAgent(
                model_id=submitter_model,
                context_window=submitter_context,
                max_output_tokens=submitter_max_tokens,
                role_id=f"autonomous_proof_identification_{role_suffix}",
                solution_path_manager=self.solution_path_manager,
            )

            resume_requires_regeneration = (
                candidate_list_state.get("status") == "rejected"
                and bool(candidate_list_state.get("semantic_rejections"))
            )
            resolved_candidates = (
                []
                if resume_requires_regeneration
                else await self._resolve_candidates(
                    theorem_candidates=theorem_candidates,
                    identification_agent=identification_agent,
                    user_prompt=user_prompt,
                    source_type=source_type,
                    source_id=source_id,
                    source_title=source_title,
                    content=content,
                    proof_round_index=proof_round_index,
                    proof_max_rounds=proof_max_rounds,
                    prior_round_results=prior_round_results,
                )
            )
            generation_attempt = max(
                1,
                int(candidate_list_state.get("generation_attempt", 1) or 1),
            )
            semantic_rejections = [
                ProofCandidateListRejection.model_validate(item)
                for item in candidate_list_state.get("semantic_rejections", [])
                if isinstance(item, dict)
            ][-5:]

            async def regenerate_candidate_list(feedback_text: str) -> list[ProofCandidate]:
                nonlocal generation_attempt, candidate_list_state
                generation_attempt += 1
                await self._broadcast(
                    broadcast_fn,
                    "proof_candidate_list_regeneration_started",
                    {
                        **base_event,
                        "list_attempt": generation_attempt,
                        "prior_rejection_count": len(semantic_rejections),
                        "feedback": feedback_text,
                    },
                )
                feedback = "\n\n".join(
                    f"Attempt {item.generation_attempt}: {item.feedback}"
                    for item in semantic_rejections
                )
                has_candidates, regenerated = (
                    await identification_agent.identify_candidates(
                        user_research_prompt=user_prompt,
                        source_type=source_type,
                        source_id=source_id,
                        source_content=content,
                        source_title=source_title,
                        proof_round_index=proof_round_index,
                        proof_max_rounds=proof_max_rounds,
                        prior_round_results=prior_round_results,
                        candidate_list_rejection_feedback=feedback,
                    )
                )
                candidate_list_state["generation_attempt"] = generation_attempt
                return regenerated if has_candidates else []

            if (
                candidate_list_state.get("status") == "rejected"
                and semantic_rejections
            ):
                resolved_candidates = await regenerate_candidate_list(
                    semantic_rejections[-1].feedback
                )
                while not resolved_candidates and not _stop_requested():
                    candidate_list_state = {
                        **candidate_list_state,
                        "status": "rejected",
                        "list_fingerprint": "",
                        "proposed_candidates": [],
                        "approved_candidate_ids": [],
                    }
                    await save_checkpoint("candidate_list_rejected")
                    resolved_candidates = await regenerate_candidate_list(
                        semantic_rejections[-1].feedback
                    )
                if _stop_requested():
                    await save_checkpoint("stopped")
                    return result

            while resolved_candidates:
                if _stop_requested():
                    await save_checkpoint("stopped")
                    return result
                list_fingerprint = _candidate_list_fingerprint(
                    resolved_candidates,
                    source_type,
                    source_id,
                )
                restored_fingerprint = str(
                    candidate_list_state.get("list_fingerprint", "")
                )
                restored_status = str(candidate_list_state.get("status", ""))
                restored_proposed: list[ProofCandidate] = []
                for raw_candidate in candidate_list_state.get(
                    "proposed_candidates", []
                ):
                    if not isinstance(raw_candidate, dict):
                        restored_proposed = []
                        break
                    try:
                        restored_proposed.append(
                            ProofCandidate.model_validate(raw_candidate)
                        )
                    except Exception:
                        restored_proposed = []
                        break
                restored_proposed_fingerprint = (
                    _candidate_list_fingerprint(
                        restored_proposed,
                        source_type,
                        source_id,
                    )
                    if restored_proposed
                    else ""
                )
                restored_by_id = {
                    candidate.theorem_id: candidate
                    for candidate in restored_proposed
                }
                approved_sequence = [
                    str(item)
                    for item in candidate_list_state.get(
                        "approved_candidate_ids", []
                    )
                ]
                current_matches_approved_checkpoint = (
                    [candidate.theorem_id for candidate in resolved_candidates]
                    == [
                        theorem_id
                        for theorem_id in approved_sequence
                        if theorem_id not in processed_candidate_ids
                    ]
                    and all(
                        candidate.theorem_id in restored_by_id
                        and _candidate_fingerprint(candidate, source_type, source_id)
                        == _candidate_fingerprint(
                            restored_by_id[candidate.theorem_id],
                            source_type,
                            source_id,
                        )
                        for candidate in resolved_candidates
                    )
                )
                if (
                    restored_status == "approved"
                    and restored_fingerprint == restored_proposed_fingerprint
                    and (
                        restored_fingerprint == list_fingerprint
                        or current_matches_approved_checkpoint
                    )
                ):
                    approved_ids = {
                        str(item)
                        for item in candidate_list_state.get(
                            "approved_candidate_ids", []
                        )
                    }
                    resolved_candidates = [
                        candidate
                        for candidate in resolved_candidates
                        if candidate.theorem_id in approved_ids
                    ]
                    break
                candidate_list_state = {
                    "status": "reviewing",
                    "review_scope": candidate_list_scope,
                    "list_fingerprint": list_fingerprint,
                    "generation_attempt": generation_attempt,
                    "proposed_candidates": [
                        candidate.model_dump(mode="json")
                        for candidate in resolved_candidates
                    ],
                    "approved_candidate_ids": [],
                    "semantic_rejections": [
                        item.model_dump(mode="json") for item in semantic_rejections
                    ],
                }
                await save_checkpoint("candidate_list_reviewing")
                await self._broadcast(
                    broadcast_fn,
                    "proof_candidate_list_review_started",
                    {
                        **base_event,
                        "list_attempt": generation_attempt,
                        "proposed_count": len(resolved_candidates),
                        "threshold_percent": 75,
                        "candidate_ids": [
                            candidate.theorem_id for candidate in resolved_candidates
                        ],
                    },
                )
                candidate_list_review_active = True
                candidate_list_review_attempt = generation_attempt
                candidate_list_review_count = len(resolved_candidates)
                validation = await list_validator.validate(
                    user_prompt=candidate_list_validation_prompt,
                    source_type=source_type,
                    source_id=source_id,
                    source_title=source_title,
                    candidates=resolved_candidates,
                )
                approved_candidates = list_validator.approved_candidates(
                    resolved_candidates,
                    validation,
                )
                approved_ids = [
                    candidate.theorem_id for candidate in approved_candidates
                ]
                candidate_list_state = {
                    "status": (
                        "approved"
                        if list_validator.threshold_met(
                            approved_count=len(approved_candidates),
                            proposed_count=len(resolved_candidates),
                        )
                        else "rejected"
                    ),
                    "review_scope": candidate_list_scope,
                    "list_fingerprint": list_fingerprint,
                    "generation_attempt": generation_attempt,
                    "proposed_candidates": [
                        candidate.model_dump(mode="json")
                        for candidate in resolved_candidates
                    ],
                    "approved_candidate_ids": approved_ids,
                    "validation": validation.model_dump(mode="json"),
                    "semantic_rejections": [
                        item.model_dump(mode="json") for item in semantic_rejections
                    ],
                }
                if candidate_list_state["status"] == "approved":
                    candidate_list_review_active = False
                    resolved_candidates = approved_candidates
                    await save_checkpoint("candidate_list_approved")
                    await self._broadcast(
                        broadcast_fn,
                        "proof_candidate_list_review_accepted",
                        {
                            **base_event,
                            "list_attempt": generation_attempt,
                            "proposed_count": len(validation.results),
                            "approved_count": len(approved_candidates),
                            "threshold_percent": 75,
                            "candidate_ids": approved_ids,
                            "candidate_reasons": [
                                item.model_dump(mode="json")
                                for item in validation.results
                            ],
                            "feedback": validation.feedback,
                        },
                    )
                    break
                rejected_ids = [
                    item.theorem_id
                    for item in validation.results
                    if item.decision == "reject_not_novel"
                ]
                rejection = ProofCandidateListRejection(
                    list_fingerprint=list_fingerprint,
                    generation_attempt=generation_attempt,
                    proposed_count=len(resolved_candidates),
                    approved_count=len(approved_candidates),
                    rejected_candidate_ids=rejected_ids,
                    feedback=validation.feedback,
                )
                semantic_rejections = [*semantic_rejections, rejection][-5:]
                candidate_list_review_active = False
                candidate_list_state["semantic_rejections"] = [
                    item.model_dump(mode="json") for item in semantic_rejections
                ]
                await save_checkpoint("candidate_list_rejected")
                await self._broadcast(
                    broadcast_fn,
                    "proof_candidate_list_review_rejected",
                    {
                        **base_event,
                        "list_attempt": generation_attempt,
                        "proposed_count": len(resolved_candidates),
                        "approved_count": len(approved_candidates),
                        "threshold_percent": 75,
                        "candidate_ids": [
                            candidate.theorem_id for candidate in resolved_candidates
                        ],
                        "rejected_candidate_ids": rejected_ids,
                        "candidate_reasons": [
                            item.model_dump(mode="json")
                            for item in validation.results
                        ],
                        "feedback": validation.feedback,
                    },
                )
                resolved_candidates = await regenerate_candidate_list(validation.feedback)
                while not resolved_candidates and not _stop_requested():
                    candidate_list_state = {
                        **candidate_list_state,
                        "status": "rejected",
                        "list_fingerprint": "",
                        "proposed_candidates": [],
                        "approved_candidate_ids": [],
                    }
                    await save_checkpoint("candidate_list_rejected")
                    resolved_candidates = await regenerate_candidate_list(
                        validation.feedback
                    )
                if _stop_requested():
                    await save_checkpoint("stopped")
                    return result
            for index, candidate in enumerate(resolved_candidates, start=1):
                candidate_indexes.setdefault(candidate.theorem_id, index)

            if not resolved_candidates:
                await save_checkpoint("no_candidates")
                await self._broadcast(
                    broadcast_fn,
                    "proof_check_no_candidates",
                    base_event,
                )
                await self._broadcast(
                    broadcast_fn,
                    "proof_check_complete",
                    {
                        **base_event,
                        "novel_count": 0,
                        "verified_count": 0,
                        "total_candidates": 0,
                    },
                )
                return result

            if trigger == "retry":
                await self._broadcast(
                    broadcast_fn,
                    "proof_retry_started",
                    {
                        **base_event,
                        "count": len(resolved_candidates),
                    },
                )

            result.total_candidates = max(result.total_candidates, len(resolved_candidates))
            await save_checkpoint("running")
            await self._broadcast(
                broadcast_fn,
                "proof_check_candidates_found",
                {
                    **base_event,
                    "count": len(resolved_candidates),
                    "proposed_count": len(
                        candidate_list_state.get("proposed_candidates", [])
                    ),
                    "approved_count": len(resolved_candidates),
                    "message": (
                        f"Validator approved {len(resolved_candidates)} of "
                        f"{len(candidate_list_state.get('proposed_candidates', []))} "
                        "proposed candidates for Lean."
                    ),
                    "theorems_preview": [
                        f"Proof {self._proof_label_for_index(candidate_indexes.get(candidate.theorem_id, index))}: {candidate.statement[:180]}"
                        for index, candidate in enumerate(resolved_candidates, start=1)
                    ],
                },
            )

            max_parallel_raw = getattr(system_config, "proof_max_parallel_candidates", 6)
            max_parallel_setting = 0 if max_parallel_raw is None else int(max_parallel_raw)
            indexed_candidates = [
                (candidate_indexes.get(candidate.theorem_id, index), candidate)
                for index, candidate in enumerate(resolved_candidates, start=1)
            ]
            batch_size = (
                len(indexed_candidates)
                if max_parallel_setting <= 0
                else max(1, max_parallel_setting)
            )
            candidate_batches = [
                indexed_candidates[index : index + batch_size]
                for index in range(0, len(indexed_candidates), batch_size)
            ]

            async def run_phase_a(theorem_candidate: ProofCandidate, proof_label: str) -> _LeanVerificationOutcome:
                if _stop_requested():
                    return _LeanVerificationOutcome(
                        candidate=theorem_candidate,
                        proof_label=proof_label,
                        success=False,
                        theorem_name="",
                        lean_code="",
                        attempts=[],
                    )

                async def record_attempts(updated_candidate: ProofCandidate, attempts: list[ProofAttemptFeedback]) -> None:
                    async with checkpoint_state_lock:
                        for idx, candidate in enumerate(resolved_candidates):
                            if candidate.theorem_id == updated_candidate.theorem_id:
                                resolved_candidates[idx] = updated_candidate
                                break
                        attempts_by_candidate[updated_candidate.theorem_id] = list(attempts)
                    await save_checkpoint("running")

                return await self._run_lean_pipeline_for_candidate(
                    theorem_candidate=theorem_candidate,
                    base_event=base_event,
                    proof_label=proof_label,
                    user_prompt=user_prompt,
                    source_type=source_type,
                    source_id=source_id,
                    source_content=content,
                    source_title=source_title,
                    submitter_model=submitter_model,
                    submitter_context=submitter_context,
                    submitter_max_tokens=submitter_max_tokens,
                    role_suffix=role_suffix,
                    trigger=trigger,
                    novel_proofs_db=novel_proofs_db,
                    broadcast_fn=broadcast_fn,
                    should_stop=_stop_requested,
                    prior_attempts=attempts_by_candidate.get(theorem_candidate.theorem_id, []),
                    prior_theorem_name=theorem_names_by_candidate.get(theorem_candidate.theorem_id, ""),
                    attempt_checkpoint_callback=record_attempts,
                    proof_pruning_pressure_callback=proof_pruning_pressure_callback,
                    proof_pruning_route_fingerprint=proof_pruning_route_fingerprint,
                    run_id=run_id,
                )

            verification_tasks = []
            pending_tasks = set()
            batch_events = [asyncio.Event() for _ in candidate_batches]
            if batch_events:
                batch_events[0].set()
            batch_remaining = {
                batch_index: len(candidate_batch)
                for batch_index, candidate_batch in enumerate(candidate_batches)
            }

            async def run_gated_phase_a(
                theorem_candidate: ProofCandidate,
                proof_label: str,
                batch_index: int,
            ) -> tuple[int, _LeanVerificationOutcome]:
                await batch_events[batch_index].wait()
                if _stop_requested():
                    raise asyncio.CancelledError()
                return batch_index, await run_phase_a(theorem_candidate, proof_label)

            verification_tasks = [
                asyncio.create_task(
                    run_gated_phase_a(
                        candidate,
                        self._proof_label_for_index(index),
                        batch_index,
                    )
                )
                for batch_index, candidate_batch in enumerate(candidate_batches)
                for index, candidate in candidate_batch
            ]
            pending_tasks = set(verification_tasks)
            ordered_outcomes: dict[int, tuple[str, dict[str, Any]]] = {}
            next_ordered_index = min((index for index, _candidate in indexed_candidates), default=1)

            async def commit_ordered_outcomes() -> bool:
                nonlocal next_ordered_index, truncation_streak
                while next_ordered_index in ordered_outcomes:
                    outcome_kind, detail = ordered_outcomes.pop(next_ordered_index)
                    next_ordered_index += 1
                    if outcome_kind == "neutral":
                        truncation_streak = []
                        continue
                    if outcome_kind != "truncation_exhausted":
                        truncation_streak = []
                        continue
                    fingerprint = str(detail["candidate_fingerprint"])
                    if truncation_streak and truncation_streak[-1].get("candidate_fingerprint") == fingerprint:
                        continue
                    truncation_streak = (truncation_streak + [detail])[-2:]
                    await self._broadcast(
                        broadcast_fn,
                        "proof_truncation_recovery_exhausted",
                        {
                            **base_event,
                            **detail,
                            "consecutive_distinct_candidates": len(truncation_streak),
                            "threshold": 2,
                            "message": (
                                f"{detail['proof_label']} exhausted all output-truncation recovery attempts "
                                f"without returning usable Lean code ({len(truncation_streak)}/2 consecutive candidates)."
                            ),
                        },
                    )
                    if terminal_truncation_stop_enabled and len(truncation_streak) >= 2:
                        abort_event.set()
                        result.fatal_stop_reason = PROOF_TRUNCATION_STOP_REASON
                        result.fatal_stop_payload = {
                            **base_event,
                            "consecutive_distinct_candidates": 2,
                            "threshold": 2,
                            "candidates": list(truncation_streak),
                            "lean_was_run": False,
                            "terminal_guidance": (
                                "Increase the proof model output allowance, lower reasoning, or configure "
                                "a different proof/Rigor & Proofs model or provider, then restart the run."
                            ),
                        }
                        await save_checkpoint("fatal_truncation_exhausted")
                        return True
                return False

            def remaining_unprocessed_candidates() -> list[ProofCandidate]:
                return [
                    candidate
                    for candidate in resolved_candidates
                    if candidate.theorem_id not in processed_candidate_ids
                ]

            def mark_batch_outcome_processed(batch_index: int) -> None:
                if batch_index not in batch_remaining:
                    return
                batch_remaining[batch_index] -= 1
                if batch_remaining[batch_index] <= 0 and batch_index + 1 < len(batch_events):
                    batch_events[batch_index + 1].set()

            async def cancel_and_drain(extra_tasks=()) -> None:
                tasks_to_drain = list(pending_tasks) + list(extra_tasks or [])
                for task in tasks_to_drain:
                    if not task.done():
                        task.cancel()
                if tasks_to_drain:
                    await asyncio.gather(*tasks_to_drain, return_exceptions=True)

            partial_stop = False
            try:
                while pending_tasks:
                    if _stop_requested():
                        logger.info(
                            "Proof verification stopping early for %s %s (stop requested before next outcome).",
                            source_type,
                            source_id,
                        )
                        await cancel_and_drain()
                        await save_checkpoint("stopped")
                        partial_stop = True
                        break

                    done_tasks, pending_tasks = await asyncio.wait(
                        pending_tasks,
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    for future in done_tasks:
                        try:
                            batch_index, outcome = future.result()
                        except FreeModelExhaustedError as exc:
                            await cancel_and_drain(set(done_tasks) - {future})
                            await save_checkpoint("provider_paused")
                            raise ProofVerificationProviderPause(
                                str(exc),
                                remaining_unprocessed_candidates(),
                            ) from exc
                        except RetryableProviderError:
                            await cancel_and_drain(set(done_tasks) - {future})
                            await save_checkpoint("provider_paused")
                            raise
                        except asyncio.CancelledError:
                            continue
                        except Exception as exc:
                            if is_provider_credit_pause_error(exc):
                                await cancel_and_drain(set(done_tasks) - {future})
                                await save_checkpoint("provider_paused")
                                raise ProofVerificationProviderPause(
                                    str(exc),
                                    remaining_unprocessed_candidates(),
                                ) from exc
                            # Any other per-candidate exception aborts the whole
                            # parallel batch; the outer `except Exception` handler
                            # will broadcast `proof_check_complete` with the error.
                            logger.error(
                                "Proof verification candidate task failed for %s %s: %s",
                                source_type,
                                source_id,
                                exc,
                            )
                            await cancel_and_drain(set(done_tasks) - {future})
                            raise

                        candidate = outcome.candidate
                        proof_label = outcome.proof_label
                        attempts = outcome.attempts
                        lean_code = outcome.lean_code
                        if outcome.theorem_name:
                            theorem_names_by_candidate[candidate.theorem_id] = outcome.theorem_name
                        if attempts:
                            attempts_by_candidate[candidate.theorem_id] = list(attempts)
                        await save_checkpoint("running")

                        # Skip the expensive Phase B post-processing (novelty,
                        # dependency extraction, DB writes) if the user has asked
                        # us to stop. The outcome itself is dropped.
                        if _stop_requested():
                            logger.info(
                                "Proof verification skipping phase B for %s %s (stop requested).",
                                source_type,
                                source_id,
                            )
                            await cancel_and_drain(set(done_tasks) - {future})
                            await save_checkpoint("stopped")
                            partial_stop = True
                            break

                        if not outcome.success:
                            error_summary = self._summarize_error(attempts[-1].error_output if attempts else "")
                            suggested_targets = self._extract_suggested_lemma_targets(
                                attempts[-1].error_output if attempts else ""
                            )
                            context_overflow = bool(
                                attempts
                                and ProofFormalizationAgent.is_context_overflow_feedback(attempts[-1])
                            )
                            if context_overflow:
                                # This candidate is deferred, not failed. Do not add it
                                # to results or processed IDs: its checkpoint remains
                                # eligible after proof-model/context settings change.
                                if candidate.theorem_id not in result.deferred_candidate_ids:
                                    result.deferred_candidate_ids.append(candidate.theorem_id)
                                if (
                                    not result.context_overflow_payload
                                    and outcome.context_overflow_payload
                                ):
                                    result.context_overflow_payload = dict(
                                        outcome.context_overflow_payload
                                    )
                                ordered_outcomes[candidate_indexes.get(candidate.theorem_id, 0)] = (
                                    "neutral",
                                    {},
                                )
                                mark_batch_outcome_processed(batch_index)
                                await save_checkpoint("running")
                                if await commit_ordered_outcomes():
                                    await cancel_and_drain(set(done_tasks) - {future})
                                    partial_stop = True
                                    break
                                continue
                            if source_type == "brainstorm" and trigger != "retry" and not context_overflow:
                                await novel_proofs_db.record_failed_candidate(
                                    source_id,
                                    candidate,
                                    error_summary,
                                    suggested_lemma_targets=suggested_targets,
                                )
                            fingerprint = _candidate_fingerprint(candidate, source_type, source_id)
                            truncation_exhausted = _truncation_chain_exhausted(attempts)
                            result.results.append(
                                ProofAttemptResult(
                                    theorem_id=candidate.theorem_id,
                                    theorem_statement=candidate.statement,
                                    lean_code=lean_code,
                                    success=False,
                                    novel=False,
                                    attempts_used=len(attempts),
                                    error_summary=error_summary,
                                    candidate_fingerprint=fingerprint,
                                    truncation_recovery_exhausted=truncation_exhausted,
                                )
                            )
                            ordered_outcomes[candidate_indexes.get(candidate.theorem_id, 0)] = (
                                "truncation_exhausted" if truncation_exhausted else "other",
                                {
                                    "theorem_id": candidate.theorem_id,
                                    "theorem_statement": candidate.statement,
                                    "proof_label": proof_label,
                                    "candidate_fingerprint": fingerprint,
                                    "attempts_used": len(attempts),
                                    "strategies_tried": [
                                        {
                                            "attempt": attempt.attempt,
                                            "recovery_mode": attempt.recovery_mode,
                                            "reasoning_effort": attempt.reasoning_effort,
                                            "requested_output_tokens": attempt.requested_output_tokens,
                                            "response_mode": attempt.response_mode,
                                        }
                                        for attempt in attempts
                                    ],
                                },
                            )
                            processed_candidate_ids.add(candidate.theorem_id)
                            mark_batch_outcome_processed(batch_index)
                            await save_checkpoint("running")
                            if await commit_ordered_outcomes():
                                await cancel_and_drain(set(done_tasks) - {future})
                                partial_stop = True
                                break
                            continue

                        integrity_task_id = f"proof_integrity_{self._integrity_task_sequence:03d}"
                        self._integrity_task_sequence += 1
                        while True:
                            try:
                                integrity = await validate_full_lean_proof_integrity(
                                    user_prompt=user_prompt,
                                    theorem_statement=candidate.statement,
                                    formal_sketch=candidate.formal_sketch,
                                    lean_code=lean_code,
                                    source_excerpt=candidate.source_excerpt or content,
                                    allowed_baseline="",
                                    validator_model=validator_model,
                                    validator_context=validator_context,
                                    validator_max_tokens=validator_max_tokens,
                                    task_id=integrity_task_id,
                                    role_id=novelty_role_id,
                                    require_statement_alignment=True,
                                )
                                break
                            except RetryableProviderError as provider_error:
                                await save_checkpoint("provider_paused")
                                await api_client_manager.wait_for_retryable_provider_error(
                                    provider_error,
                                    role_id=novelty_role_id,
                                    should_stop=should_stop,
                                )
                                if should_stop and should_stop():
                                    raise asyncio.CancelledError()
                        if not integrity.valid:
                            integrity_feedback = ProofAttemptFeedback(
                                attempt=(attempts[-1].attempt + 1 if attempts else 1),
                                theorem_id=candidate.theorem_id,
                                reasoning="Post-Lean proof integrity check failed.",
                                lean_code=lean_code,
                                error_output=integrity.reason,
                                strategy="full_script",
                                success=False,
                            )
                            attempts = list(attempts) + [integrity_feedback]
                            attempts_by_candidate[candidate.theorem_id] = list(attempts)
                            error_summary = self._summarize_error(integrity.reason)
                            suggested_targets = self._extract_suggested_lemma_targets(integrity.reason)
                            if source_type == "brainstorm" and trigger != "retry":
                                await novel_proofs_db.record_failed_candidate(
                                    source_id,
                                    candidate,
                                    error_summary,
                                    suggested_lemma_targets=suggested_targets,
                                )
                            await self._broadcast(
                                broadcast_fn,
                                "proof_integrity_rejected",
                                {
                                    **base_event,
                                    "theorem_id": candidate.theorem_id,
                                    "theorem_statement": candidate.statement,
                                    "proof_label": proof_label,
                                    "category": integrity.category,
                                    "reason": integrity.reason,
                                },
                            )
                            result.results.append(
                                ProofAttemptResult(
                                    theorem_id=candidate.theorem_id,
                                    theorem_statement=candidate.statement,
                                    lean_code=lean_code,
                                    success=False,
                                    novel=False,
                                    attempts_used=len(attempts),
                                    error_summary=error_summary,
                                )
                            )
                            ordered_outcomes[candidate_indexes.get(candidate.theorem_id, 0)] = ("other", {})
                            processed_candidate_ids.add(candidate.theorem_id)
                            mark_batch_outcome_processed(batch_index)
                            await save_checkpoint("running")
                            if await commit_ordered_outcomes():
                                await cancel_and_drain(set(done_tasks) - {future})
                                partial_stop = True
                                break
                            continue

                        stored_theorem_statement = (
                            integrity.actual_theorem_statement.strip()
                            or candidate.statement
                        )
                        stored_theorem_name = (
                            integrity.actual_theorem_name.strip()
                            or outcome.theorem_name
                        )
                        stored_formal_sketch = candidate.formal_sketch
                        verification_notes = "Lean 4 accepted the submitted proof."
                        if integrity.category in {"statement_downshifted", "statement_alignment_uncertain", "statement_alignment_unavailable"}:
                            stored_formal_sketch = (
                                f"{stored_formal_sketch}\n\n"
                                f"Original intended theorem candidate: {candidate.statement}\n"
                                f"Statement-alignment classification: {integrity.category}. "
                                f"{integrity.reason or integrity.downshift_reason}"
                            ).strip()
                            verification_notes = (
                                "Lean 4 accepted the submitted proof. "
                                "MOTO preserved it under the actual Lean-verified statement "
                                "instead of discarding it for candidate mismatch."
                            )
                            await self._broadcast(
                                broadcast_fn,
                                "proof_downshifted",
                                {
                                    **base_event,
                                    "theorem_id": candidate.theorem_id,
                                    "intended_theorem_statement": candidate.statement,
                                    "theorem_statement": stored_theorem_statement,
                                    "proof_label": proof_label,
                                    "category": integrity.category,
                                    "reason": integrity.reason or integrity.downshift_reason,
                                },
                            )

                        novelty_task_id = f"proof_novelty_{self._novelty_task_sequence:03d}"
                        self._novelty_task_sequence += 1

                        solver_hints = []
                        if self._first_attempt_used_smt_hint(attempts, candidate.smt_hint):
                            solver_hints.append("smt-z3")

                        registration_kwargs = {
                            "proof_database": novel_proofs_db,
                            "user_prompt": canonical_user_prompt or user_prompt,
                            "theorem_statement": stored_theorem_statement,
                            "lean_code": lean_code,
                            "validator_model": validator_model,
                            "validator_context": validator_context,
                            "validator_max_tokens": validator_max_tokens,
                            "task_id": novelty_task_id,
                            "role_id": novelty_role_id,
                            "source_type": source_type,
                            "source_id": source_id,
                            "source_title": source_title,
                            "theorem_id": candidate.theorem_id,
                            "theorem_name": stored_theorem_name,
                            "formal_sketch": stored_formal_sketch,
                            "solver": "Lean 4",
                            "verification_notes": verification_notes,
                            "attempt_count": len(attempts),
                            "attempts": attempts,
                            "solver_hints": solver_hints,
                            "broadcast_fn": broadcast_fn,
                            "base_event": base_event,
                            "proof_label": proof_label,
                            "retry_origin_source_id": candidate.origin_source_id,
                            "run_id": run_id,
                        }
                        while True:
                            try:
                                registration = await register_verified_lean_proof(
                                    **registration_kwargs
                                )
                                break
                            except RetryableProviderError as provider_error:
                                await save_checkpoint("provider_paused")
                                if trigger == "manual":
                                    registration_kwargs["novelty_classification"] = (
                                        "not_novel",
                                        (
                                            "Conservatively classified as not novel after a "
                                            "transient failure in the post-Lean novelty check."
                                        ),
                                    )
                                    continue
                                await api_client_manager.wait_for_retryable_provider_error(
                                    provider_error,
                                    role_id=novelty_role_id,
                                    should_stop=should_stop,
                                )
                                if should_stop and should_stop():
                                    raise asyncio.CancelledError()
                        stored_record = registration.record
                        is_novel = stored_record.novel
                        is_prompt_novel = is_prompt_injection_novel_tier(stored_record.novelty_tier)
                        result.verified_count += 1

                        await self._broadcast(
                            broadcast_fn,
                            "proof_verified",
                            {
                                **base_event,
                                "proof_id": stored_record.proof_id,
                                "theorem_id": candidate.theorem_id,
                                "theorem_statement": stored_theorem_statement,
                                "intended_theorem_statement": candidate.statement,
                                "proof_label": proof_label,
                                "strategy": attempts[-1].strategy if attempts else "full_script",
                                "is_novel": is_novel,
                                "novelty_tier": stored_record.novelty_tier,
                                "novelty_reasoning": stored_record.novelty_reasoning,
                                "retry_origin_source_id": candidate.origin_source_id,
                            },
                        )

                        dep_lemma_agent = MathlibLemmaSearchAgent(
                            model_id=submitter_model,
                            context_window=submitter_context,
                            max_output_tokens=submitter_max_tokens,
                            role_id=f"autonomous_proof_lemma_search_{role_suffix}_dep",
                        )
                        dependencies = []
                        try:
                            dependencies = await self._dependency_extractor.extract_dependencies(
                                lean_code=lean_code,
                                theorem_name=stored_theorem_name,
                                proof_database=novel_proofs_db,
                                lemma_search_agent=dep_lemma_agent,
                                relevant_lemmas=candidate.relevant_lemmas,
                                current_proof_id=stored_record.proof_id,
                            )
                            updated_record = await novel_proofs_db.update_proof_dependencies(
                                stored_record.proof_id,
                                dependencies,
                                extraction_status="complete",
                            )
                            if (
                                updated_record is not None
                                and isinstance(
                                    getattr(updated_record, "proof_id", None),
                                    str,
                                )
                            ):
                                stored_record = updated_record
                            if dependencies:
                                await self._broadcast(
                                    broadcast_fn,
                                    "proof_dependency_added",
                                    {
                                        **base_event,
                                        "proof_id": stored_record.proof_id,
                                        "theorem_name": stored_record.theorem_name,
                                        "proof_label": proof_label,
                                        "dependencies": [
                                            dependency.model_dump(mode="json")
                                            for dependency in dependencies
                                        ],
                                    },
                                )
                        except Exception as exc:
                            updated_record = await novel_proofs_db.update_proof_dependencies(
                                stored_record.proof_id,
                                [],
                                extraction_status="failed",
                                extraction_detail=str(exc),
                            )
                            if (
                                updated_record is not None
                                and isinstance(
                                    getattr(updated_record, "proof_id", None),
                                    str,
                                )
                            ):
                                stored_record = updated_record
                            logger.debug(
                                "Dependency extraction failed for theorem %s: %s",
                                candidate.theorem_id,
                                exc,
                            )

                        # Pruning is notified only after registration and
                        # dependency/checkpoint state are safe. The callback
                        # itself only updates counters/schedules owned work.
                        if proof_pruning_registered_callback is not None:
                            await proof_pruning_registered_callback(
                                stored_record,
                                {
                                    "proof_set_revision": (
                                        await novel_proofs_db.get_proof_set_revision()
                                    ),
                                    "proof_round_index": proof_round_index,
                                    "trigger": trigger,
                                    "duplicate": registration.duplicate,
                                },
                            )

                        if candidate.origin_source_id:
                            await novel_proofs_db.mark_resolved_retry(
                                candidate.origin_source_id,
                                candidate.theorem_id,
                                stored_record.proof_id,
                            )

                        if self._should_append_verified_proof(
                            is_novel=is_prompt_novel,
                            duplicate=registration.duplicate,
                            append_proof_callback=append_proof_callback,
                            append_known_proofs=self._should_append_known_proofs_for_trigger(trigger),
                        ):
                            if is_prompt_novel and not registration.duplicate:
                                result.novel_count += 1
                            if append_proof_callback is not None:
                                await append_proof_callback(stored_record)
                            elif append_to_source and source_type == "brainstorm":
                                await brainstorm_memory.append_proofs_section(source_id, stored_record)
                            elif append_to_source and source_type == "paper":
                                await paper_library.append_proofs_section(source_id, stored_record)

                        result.results.append(
                            ProofAttemptResult(
                                theorem_id=candidate.theorem_id,
                                theorem_statement=stored_theorem_statement,
                                lean_code=lean_code,
                                success=True,
                                novel=is_novel,
                                attempts_used=len(attempts),
                                proof_id=stored_record.proof_id,
                                error_summary="",
                            )
                        )
                        ordered_outcomes[candidate_indexes.get(candidate.theorem_id, 0)] = ("other", {})
                        processed_candidate_ids.add(candidate.theorem_id)
                        mark_batch_outcome_processed(batch_index)
                        await save_checkpoint("running")
                        if await commit_ordered_outcomes():
                            await cancel_and_drain(set(done_tasks) - {future})
                            partial_stop = True
                            break
                    if partial_stop:
                        break
            finally:
                # Defensive cleanup: make sure we don't leak pending tasks if
                # the consumer loop exits early for any reason.
                leftover = [task for task in verification_tasks if not task.done()]
                for task in leftover:
                    task.cancel()
                if leftover:
                    await asyncio.gather(*leftover, return_exceptions=True)

            if partial_stop:
                return result

            checkpoint_status = "deferred" if result.deferred_candidate_ids else "complete"
            await save_checkpoint(checkpoint_status)
            await self._broadcast(
                broadcast_fn,
                "proof_check_complete",
                {
                    **base_event,
                    "novel_count": result.novel_count,
                    "verified_count": result.verified_count,
                    "total_candidates": result.total_candidates,
                    "deferred_candidate_ids": list(result.deferred_candidate_ids),
                },
            )
            return result
        except asyncio.CancelledError:
            await broadcast_candidate_list_interrupted(
                error_kind="cancelled",
                error_message="The proof run stopped before list review completed.",
            )
            raise
        except ProofVerificationProviderPause:
            await broadcast_candidate_list_interrupted(
                error_kind="provider_pause",
                error_message="The configured provider paused before list review completed.",
            )
            raise
        except RetryableProviderError as exc:
            await broadcast_candidate_list_interrupted(
                error_kind="retryable_provider_error",
                error_message=str(exc),
            )
            await save_checkpoint("provider_paused")
            raise
        except FreeModelExhaustedError as exc:
            await broadcast_candidate_list_interrupted(
                error_kind="provider_exhausted",
                error_message=str(exc),
            )
            await save_checkpoint("provider_paused")
            raise
        except (ProofCandidateListContextError, ProviderContextLengthError) as exc:
            await broadcast_candidate_list_interrupted(
                error_kind="context_overflow",
                error_message=str(exc),
            )
            role_suffix = self._role_suffix(
                source_type,
                role_suffix_override,
                proof_run_context,
            )
            role_id = f"autonomous_proof_candidate_list_validator_{role_suffix}"
            route = exc.route if isinstance(exc, ProviderContextLengthError) else None
            overflow_payload = {
                **base_event,
                "workflow_mode": self._proof_workflow_mode(trigger),
                "overflow_origin": (
                    "provider"
                    if isinstance(exc, ProviderContextLengthError)
                    else "local_preflight"
                ),
                "role_id": role_id,
                **context_overflow_model_payload(
                    api_client_manager.get_role_config(role_id),
                    route=route,
                ),
                "reason": CONTEXT_OVERFLOW_STOP_REASON,
                "message": (
                    "Proof candidate-list validation cannot continue because its mandatory "
                    "review context exceeds the configured Validator context budget. "
                    "Choose a larger-context Validator model or reduce the proof source context."
                ),
                "resolution": CONTEXT_OVERFLOW_RESOLUTION,
                "error_detail": self._summarize_error(str(exc), limit=1000),
            }
            result.context_overflow_payload = overflow_payload
            result.had_error = True
            result.error_message = str(exc)
            await save_checkpoint("error")
            await self._broadcast(
                broadcast_fn,
                "proof_context_overflow",
                {
                    **overflow_payload,
                    "fatal": False,
                },
            )
            return result
        except Exception as exc:
            if is_non_retryable_model_error(exc):
                await broadcast_candidate_list_interrupted(
                    error_kind="provider_repair_required",
                    error_message=str(exc),
                )
                await save_checkpoint("provider_paused")
                raise
            if is_transient_model_call_error(exc):
                await broadcast_candidate_list_interrupted(
                    error_kind="transient_provider_error",
                    error_message=str(exc),
                )
                await save_checkpoint("provider_paused")
                logger.warning(
                    "Proof verification transient provider failure for %s %s; preserving checkpoint: %s",
                    source_type,
                    source_id,
                    exc,
                )
                role_suffix = self._role_suffix(
                    source_type,
                    role_suffix_override,
                    proof_run_context,
                )
                raise RetryableProviderError(
                    provider="unknown",
                    provider_label="Inference provider",
                    role_id=f"autonomous_proof_{role_suffix}",
                    model=submitter_model,
                    reason="transient_provider_error",
                    message=format_transient_provider_error(exc),
                ) from exc
            await broadcast_candidate_list_interrupted(
                error_kind=(
                    "context_error"
                    if "context" in str(exc).lower()
                    else "contract_error"
                ),
                error_message=str(exc),
            )
            await save_checkpoint("error")
            result.had_error = True
            result.error_message = str(exc)
            logger.error(
                "Proof verification stage failed for %s %s: %s",
                source_type,
                source_id,
                exc,
            )
            await self._broadcast(
                broadcast_fn,
                "proof_check_complete",
                {
                    "source_type": source_type,
                    "source_id": source_id,
                    "source_title": source_title,
                    "trigger": trigger,
                    "proof_round_index": proof_round_index,
                    "proof_max_rounds": proof_max_rounds,
                    "novel_count": result.novel_count,
                    "verified_count": result.verified_count,
                    "total_candidates": result.total_candidates,
                    "message": (
                        "Proof verification encountered an error: "
                        f"{self._summarize_error(str(exc), limit=1800)}"
                    ),
                },
            )
            return result
        finally:
            if release_source_on_exit:
                await self._release_source(
                    source_type,
                    source_id,
                    owned_reservation_token,
                )

    async def _run_lean_pipeline_for_candidate(
        self,
        *,
        theorem_candidate: ProofCandidate,
        base_event: dict[str, Any],
        proof_label: str,
        user_prompt: str,
        source_type: str,
        source_id: str,
        source_content: str,
        source_title: str,
        submitter_model: str,
        submitter_context: int,
        submitter_max_tokens: int,
        role_suffix: str,
        trigger: str,
        novel_proofs_db,
        broadcast_fn: BroadcastFn,
        should_stop: ShouldStopFn = None,
        prior_attempts: Optional[list[ProofAttemptFeedback]] = None,
        prior_theorem_name: str = "",
        attempt_checkpoint_callback: Optional[Callable[[ProofCandidate, list[ProofAttemptFeedback]], Awaitable[None]]] = None,
        proof_pruning_pressure_callback: ProofPruningPressureCallback = None,
        proof_pruning_route_fingerprint: str = "",
        run_id: str = "",
    ) -> _LeanVerificationOutcome:
        """Phase A for one candidate: lemma prep, SMT hint, and Lean 4 attempts.

        Each invocation creates its own agent instances so that concurrent
        candidates do not race on shared ``task_sequence`` counters and so the
        ``role_id`` remains the same for all attempts belonging to one
        candidate.
        """
        if should_stop and should_stop():
            raise asyncio.CancelledError()
        identification_agent = ProofIdentificationAgent(
            model_id=submitter_model,
            context_window=submitter_context,
            max_output_tokens=submitter_max_tokens,
            role_id=f"autonomous_proof_identification_{role_suffix}",
        )
        lemma_search_agent = MathlibLemmaSearchAgent(
            model_id=submitter_model,
            context_window=submitter_context,
            max_output_tokens=submitter_max_tokens,
            role_id=f"autonomous_proof_lemma_search_{role_suffix}",
        )
        formalization_agent = ProofFormalizationAgent(
            model_id=submitter_model,
            context_window=submitter_context,
            max_output_tokens=submitter_max_tokens,
            role_id=f"autonomous_proof_formalization_{role_suffix}",
        )

        candidate = await self._prepare_candidate(
            user_prompt=user_prompt,
            source_type=source_type,
            theorem_candidate=theorem_candidate,
            source_content=source_content,
            source_title=source_title,
            lemma_search_agent=lemma_search_agent,
        )
        if should_stop and should_stop():
            raise asyncio.CancelledError()
        smt_hint = await self._run_smt_check(
            user_prompt=user_prompt,
            source_type=source_type,
            source_id=source_id,
            base_event=base_event,
            candidate=candidate,
            proof_label=proof_label,
            source_content=source_content,
            source_title=source_title,
            identification_agent=identification_agent,
            broadcast_fn=broadcast_fn,
        )
        if should_stop and should_stop():
            raise asyncio.CancelledError()
        if smt_hint:
            candidate = candidate.model_copy(update={"smt_hint": smt_hint})
        if trigger == "retry" and candidate.origin_source_id:
            await novel_proofs_db.mark_retried(
                candidate.origin_source_id,
                candidate.theorem_id,
                source_id,
            )

        active_attempts: list[ProofAttemptFeedback] = list(prior_attempts or [])
        context_overflow_payload: dict[str, Any] = {}
        prior_success = next((attempt for attempt in active_attempts if attempt.success), None)
        if prior_success:
            theorem_name = prior_theorem_name or self._extract_theorem_name_from_lean(prior_success.lean_code)
            return _LeanVerificationOutcome(
                candidate=candidate,
                proof_label=proof_label,
                success=True,
                theorem_name=theorem_name,
                lean_code=prior_success.lean_code,
                attempts=active_attempts,
            )

        async def on_attempt_started(
            attempt_number: int,
            strategy: str,
            current_candidate=candidate,
        ) -> None:
            recovery_mode = "configured"
            reasoning_effort = None
            response_mode = "json"
            truncation_count = sum(
                1 for attempt in active_attempts
                if attempt.failure_kind == "output_truncated"
            )
            if truncation_count:
                recovery_step = min(5, truncation_count + 1)
                if recovery_step == 2:
                    recovery_mode, response_mode = "compact_same_model", "compact_json"
                elif recovery_step in {3, 4}:
                    recovery_mode, reasoning_effort, response_mode = (
                        "compact_reduced_reasoning" if recovery_step == 3 else "tactic_reduced_reasoning",
                        "low",
                        "compact_json",
                    )
                else:
                    recovery_mode, reasoning_effort, response_mode = (
                        "tactic_minimal_reasoning",
                        "none",
                        "compact_json",
                    )
            await self._broadcast(
                broadcast_fn,
                "proof_attempt_started",
                {
                    **base_event,
                    "theorem_id": current_candidate.theorem_id,
                    "theorem_statement": current_candidate.statement,
                    "proof_label": proof_label,
                    "attempt": attempt_number,
                    "strategy": strategy,
                    "recovery_mode": recovery_mode,
                    "reasoning_effort": reasoning_effort,
                    "response_mode": response_mode,
                    "message": (
                        f"{proof_label}, Attempt {attempt_number} started"
                        if recovery_mode == "configured"
                        else (
                            f"{proof_label}, Attempt {attempt_number} started after truncation using "
                            f"{recovery_mode.replace('_', ' ')}"
                            + (f" with reasoning={reasoning_effort}" if reasoning_effort else "")
                            + "."
                        )
                    ),
                    "retry_origin_source_id": current_candidate.origin_source_id,
                },
            )

        async def on_attempt_feedback(feedback, current_candidate=candidate) -> None:
            active_attempts.append(feedback)
            if attempt_checkpoint_callback:
                await attempt_checkpoint_callback(current_candidate, active_attempts)
            if feedback.success:
                await self._broadcast(
                    broadcast_fn,
                    "proof_lean_accepted",
                    {
                        **base_event,
                        "theorem_id": current_candidate.theorem_id,
                        "theorem_statement": current_candidate.statement,
                        "proof_label": proof_label,
                        "attempt": feedback.attempt,
                        "strategy": feedback.strategy,
                        "lean_response": self._lean_response_summary(feedback),
                        "proof_verified": True,
                        "retry_origin_source_id": current_candidate.origin_source_id,
                    },
                )
            elif ProofFormalizationAgent.is_context_overflow_feedback(feedback):
                configured_payload = context_overflow_model_payload(
                    api_client_manager.get_role_config(formalization_agent.role_id)
                )
                route_payload = {
                    key: value
                    for key, value in {
                        "configured_model": feedback.configured_model,
                        "configured_provider": feedback.configured_provider,
                        "effective_model": feedback.effective_model,
                        "effective_provider": feedback.effective_provider,
                    }.items()
                    if value
                }
                overflow_origin = feedback.overflow_origin or "provider"
                is_local_preflight = overflow_origin == "local_preflight"
                message = (
                    "Proof formalization deferred before provider invocation: the mandatory "
                    f"prompt requires {feedback.prompt_tokens:,} input tokens but the configured "
                    f"input budget is {feedback.max_input_tokens:,}. Lean 4 was not run. Increase "
                    "the proof model context window, reduce its output reserve, or reduce source context."
                    if is_local_preflight
                    and feedback.prompt_tokens is not None
                    and feedback.max_input_tokens is not None
                    else (
                        "Proof formalization deferred before provider invocation because the mandatory "
                        "prompt exceeds the configured local input budget. Lean 4 was not run. Increase "
                        "the proof model context window, reduce its output reserve, or reduce source context."
                        if is_local_preflight
                        else (
                            "Proof formalization deferred: the selected provider rejected the proof "
                            "prompt because it exceeded the model context window. Lean 4 was not run. "
                            "Choose a larger-context proof model or reduce source context."
                        )
                    )
                )
                await self._broadcast(
                    broadcast_fn,
                    "proof_context_overflow",
                    {
                        **base_event,
                        "workflow_mode": self._proof_workflow_mode(trigger),
                        "fatal": False,
                        "overflow_origin": overflow_origin,
                        "role_id": formalization_agent.role_id,
                        **configured_payload,
                        **route_payload,
                        "reason": CONTEXT_OVERFLOW_STOP_REASON,
                        "message": message,
                        "resolution": CONTEXT_OVERFLOW_RESOLUTION,
                        "error_detail": feedback.error_output,
                        "theorem_id": current_candidate.theorem_id,
                        "theorem_statement": current_candidate.statement,
                        "proof_label": proof_label,
                        "attempt": feedback.attempt,
                        "strategy": feedback.strategy,
                        "prompt_tokens": feedback.prompt_tokens,
                        "max_input_tokens": feedback.max_input_tokens,
                        "retry_origin_source_id": current_candidate.origin_source_id,
                    },
                )
                context_overflow_payload = {
                        **base_event,
                        "workflow_mode": self._proof_workflow_mode(trigger),
                        "overflow_origin": overflow_origin,
                        "role_id": formalization_agent.role_id,
                        **configured_payload,
                        **route_payload,
                        "reason": CONTEXT_OVERFLOW_STOP_REASON,
                        "message": message,
                        "resolution": CONTEXT_OVERFLOW_RESOLUTION,
                        "error_detail": feedback.error_output,
                        "theorem_id": current_candidate.theorem_id,
                        "theorem_statement": current_candidate.statement,
                        "proof_label": proof_label,
                        "attempt": feedback.attempt,
                        "strategy": feedback.strategy,
                        "prompt_tokens": feedback.prompt_tokens,
                        "max_input_tokens": feedback.max_input_tokens,
                        "retry_origin_source_id": current_candidate.origin_source_id,
                }
                if (
                    proof_pruning_pressure_callback is not None
                    and novel_proofs_db is not None
                ):
                    active_proof_block = (
                        novel_proofs_db.get_novel_proofs_for_injection(run_id)
                        if hasattr(
                            novel_proofs_db,
                            "get_novel_proofs_for_injection",
                        )
                        else ""
                    )
                    active_proof_tokens = count_tokens(active_proof_block)
                    pressure = ProofPruneContextPressure(
                        trigger="context_pressure",
                        prompt_tokens=feedback.prompt_tokens,
                        available_input_tokens=feedback.max_input_tokens,
                        active_proof_tokens=active_proof_tokens,
                        mandatory_source_tokens=count_tokens(source_content),
                        candidate_and_feedback_tokens=max(
                            0,
                            int(feedback.prompt_tokens or 0)
                            - count_tokens(source_content)
                            - active_proof_tokens,
                        ),
                        active_proof_context_tokens=active_proof_tokens,
                        output_reserve_tokens=max(
                            0,
                            int(submitter_max_tokens or 0),
                        ),
                        configured_context_window=max(
                            0,
                            int(submitter_context or 0),
                        ),
                        route_config_fingerprint=proof_pruning_route_fingerprint,
                        proof_set_revision=(
                            await novel_proofs_db.get_proof_set_revision()
                        ),
                        detail=(
                            "Candidate-local proof formalization overflow; "
                            "mandatory source context remains authoritative."
                        ),
                    )
                    await proof_pruning_pressure_callback(
                        pressure,
                        urgent=True,
                        proof_set_revision=pressure.proof_set_revision,
                    )
            elif feedback.failure_kind == "output_truncated":
                await self._broadcast(
                    broadcast_fn,
                    "proof_attempt_failed",
                    {
                        **base_event,
                        "theorem_id": current_candidate.theorem_id,
                        "theorem_statement": current_candidate.statement,
                        "proof_label": proof_label,
                        "attempt": feedback.attempt,
                        "strategy": feedback.strategy,
                        "failure_kind": feedback.failure_kind,
                        "lean_was_run": False,
                        "recovery_mode": feedback.recovery_mode,
                        "reasoning_effort": feedback.reasoning_effort,
                        "response_mode": feedback.response_mode,
                        "requested_output_tokens": feedback.requested_output_tokens,
                        "message": (
                            "Model output truncated before usable Lean code was returned; "
                            "Lean 4 was not run."
                        ),
                        "error_summary": self._summarize_error(feedback.error_output),
                        "proof_verified": False,
                        "retry_origin_source_id": current_candidate.origin_source_id,
                    },
                )
            else:
                lean_response = self._lean_response_summary(feedback)
                await self._broadcast(
                    broadcast_fn,
                    "proof_attempt_failed",
                    {
                        **base_event,
                        "theorem_id": current_candidate.theorem_id,
                        "theorem_statement": current_candidate.statement,
                        "proof_label": proof_label,
                        "attempt": feedback.attempt,
                        "strategy": feedback.strategy,
                        "failure_kind": feedback.failure_kind,
                        "lean_was_run": feedback.lean_was_run,
                        "error_summary": self._summarize_error(feedback.error_output),
                        "lean_response": lean_response,
                        "proof_verified": False,
                        "retry_origin_source_id": current_candidate.origin_source_id,
                    },
                )

        full_attempt_count = sum(
            1
            for attempt in active_attempts
            if attempt.strategy == "full_script"
            and not ProofFormalizationAgent.is_context_overflow_feedback(attempt)
        )
        tactic_attempt_count = sum(
            1
            for attempt in active_attempts
            if attempt.strategy == "tactic_script"
            and not ProofFormalizationAgent.is_context_overflow_feedback(attempt)
        )
        full_remaining = max(0, 3 - full_attempt_count)
        tactic_remaining = max(0, 2 - tactic_attempt_count)
        success = False
        theorem_name = ""
        lean_code = active_attempts[-1].lean_code if active_attempts else ""
        attempts = active_attempts

        if full_remaining > 0 and tactic_attempt_count == 0:
            success, theorem_name, lean_code, attempts = await formalization_agent.prove_candidate(
                user_research_prompt=user_prompt,
                source_type=source_type,
                theorem_candidate=candidate,
                source_content=source_content,
                max_attempts=full_remaining,
                attempt_callback=on_attempt_feedback,
                attempt_start_callback=on_attempt_started,
                prior_attempts=active_attempts,
                smt_hint=candidate.smt_hint,
                source_title=source_title,
                should_stop=should_stop,
            )
        workspace_error = bool(
            attempts
            and (attempts[-1].error_output or "").startswith(LEAN_WORKSPACE_ERROR_PREFIX)
        )
        context_overflow = bool(
            attempts
            and ProofFormalizationAgent.is_context_overflow_feedback(attempts[-1])
        )
        if (
            not success
            and not workspace_error
            and not context_overflow
            and tactic_remaining > 0
            and not (should_stop and should_stop())
        ):
            tactic_success, tactic_theorem_name, lean_code, attempts = await formalization_agent.prove_candidate_tactic_script(
                user_research_prompt=user_prompt,
                source_type=source_type,
                theorem_candidate=candidate,
                source_content=source_content,
                max_attempts=tactic_remaining,
                attempt_callback=on_attempt_feedback,
                attempt_start_callback=on_attempt_started,
                prior_attempts=attempts,
                starting_attempt_number=(attempts[-1].attempt + 1 if attempts else 4),
                smt_hint=candidate.smt_hint,
                source_title=source_title,
                should_stop=should_stop,
            )
            if tactic_theorem_name:
                theorem_name = tactic_theorem_name
            success = tactic_success
            context_overflow = bool(
                attempts
                and ProofFormalizationAgent.is_context_overflow_feedback(attempts[-1])
            )

        if (
            not success
            and not workspace_error
            and not context_overflow
            and not _truncation_chain_exhausted(attempts)
            and not (should_stop and should_stop())
        ):
            await self._broadcast(
                broadcast_fn,
                "proof_attempts_exhausted",
                {
                    **base_event,
                    "theorem_id": candidate.theorem_id,
                    "theorem_statement": candidate.statement,
                    "proof_label": proof_label,
                    "retry_origin_source_id": candidate.origin_source_id,
                },
            )

        return _LeanVerificationOutcome(
            candidate=candidate,
            proof_label=proof_label,
            success=success,
            theorem_name=theorem_name,
            lean_code=lean_code,
            attempts=attempts,
            context_overflow_payload=(
                context_overflow_payload
                if attempts
                and ProofFormalizationAgent.is_context_overflow_feedback(attempts[-1])
                else {}
            ),
        )

    async def run_manual(
        self,
        *,
        content: str,
        source_type: str,
        source_id: str,
        user_prompt: str,
        canonical_user_prompt: str = "",
        run_id: str = "",
        submitter_model: str,
        submitter_context: int,
        submitter_max_tokens: int,
        validator_model: str,
        validator_context: int,
        validator_max_tokens: int,
        broadcast_fn: BroadcastFn,
        novel_proofs_db,
        source_title: str = "",
        source_reserved: bool = False,
        source_reservation_token: str = "",
        append_to_source: bool = True,
        append_proof_callback: ProofAppendCallback = None,
        should_stop: ShouldStopFn = None,
        release_source_on_exit: bool = True,
        proof_run_context: Optional[dict[str, Any]] = None,
        proof_pruning_registered_callback: ProofPruningRegisteredCallback = None,
        proof_pruning_pressure_callback: ProofPruningPressureCallback = None,
        proof_pruning_route_fingerprint: str = "",
        proof_round_index: int = 1,
        proof_max_rounds: int = 1,
        prior_round_results: str = "",
        theorem_candidates: Optional[list[ProofCandidate]] = None,
        proof_candidate_indexes: Optional[dict[str, int]] = None,
        checkpoint_attempts_by_candidate: Optional[dict[str, list[ProofAttemptFeedback]]] = None,
        checkpoint_theorem_names_by_candidate: Optional[dict[str, str]] = None,
        checkpoint_truncation_streak: Optional[list[dict[str, Any]]] = None,
        checkpoint_result: Optional[ProofStageResult] = None,
        checkpoint_candidate_list_state: Optional[dict[str, Any]] = None,
        checkpoint_processed_candidate_ids: Optional[list[str]] = None,
        checkpoint_callback: ProofCheckpointCallback = None,
    ) -> ProofStageResult:
        """Run a user-triggered proof check using manual proof role IDs."""
        return await self.run(
            content=content,
            source_type=source_type,
            source_id=source_id,
            user_prompt=user_prompt,
            canonical_user_prompt=canonical_user_prompt,
            run_id=run_id,
            submitter_model=submitter_model,
            submitter_context=submitter_context,
            submitter_max_tokens=submitter_max_tokens,
            validator_model=validator_model,
            validator_context=validator_context,
            validator_max_tokens=validator_max_tokens,
            broadcast_fn=broadcast_fn,
            novel_proofs_db=novel_proofs_db,
            source_title=source_title,
            role_suffix_override=f"manual_{source_type}",
            trigger="manual",
            source_reserved=source_reserved,
            source_reservation_token=source_reservation_token,
            release_source_on_exit=release_source_on_exit,
            append_to_source=append_to_source,
            append_proof_callback=append_proof_callback,
            should_stop=should_stop,
            terminal_truncation_stop_enabled=False,
            proof_run_context=proof_run_context,
            proof_pruning_registered_callback=proof_pruning_registered_callback,
            proof_pruning_pressure_callback=proof_pruning_pressure_callback,
            proof_pruning_route_fingerprint=proof_pruning_route_fingerprint,
            proof_round_index=proof_round_index,
            proof_max_rounds=proof_max_rounds,
            prior_round_results=prior_round_results,
            theorem_candidates=theorem_candidates,
            proof_candidate_indexes=proof_candidate_indexes,
            checkpoint_attempts_by_candidate=checkpoint_attempts_by_candidate,
            checkpoint_theorem_names_by_candidate=checkpoint_theorem_names_by_candidate,
            checkpoint_truncation_streak=checkpoint_truncation_streak,
            checkpoint_result=checkpoint_result,
            checkpoint_candidate_list_state=checkpoint_candidate_list_state,
            checkpoint_processed_candidate_ids=checkpoint_processed_candidate_ids,
            checkpoint_callback=checkpoint_callback,
        )
