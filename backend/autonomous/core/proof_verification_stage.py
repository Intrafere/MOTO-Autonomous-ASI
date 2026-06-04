"""
Orchestrates proof identification, Lean 4 attempts, retry handling, and novelty checks.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from backend.autonomous.agents.lemma_search_agent import MathlibLemmaSearchAgent
from backend.autonomous.agents.proof_formalization_agent import ProofFormalizationAgent
from backend.autonomous.agents.proof_identification_agent import ProofIdentificationAgent
from backend.autonomous.memory.brainstorm_memory import brainstorm_memory
from backend.autonomous.memory.paper_library import paper_library
from backend.autonomous.core.proof_registration import register_verified_lean_proof
from backend.shared.config import system_config
from backend.shared.lean_proof_integrity import validate_full_lean_proof_integrity
from backend.shared.model_error_utils import is_non_retryable_model_error
from backend.shared.models import ProofAttemptFeedback, ProofAttemptResult, ProofCandidate, ProofStageResult, SmtHint
from backend.shared.openrouter_client import FreeModelExhaustedError
from backend.shared.provider_pause import is_provider_credit_pause_error
from backend.shared.smt_client import get_smt_client
from .proof_dependency_extractor import ProofDependencyExtractor

logger = logging.getLogger(__name__)

BroadcastFn = Optional[Callable[[str, dict[str, Any]], Awaitable[None]]]
ShouldStopFn = Optional[Callable[[], bool]]
ProofCheckpointCallback = Optional[Callable[[dict[str, Any]], Awaitable[None]]]
ProofAppendCallback = Optional[Callable[[Any], Awaitable[None]]]
LEAN_WORKSPACE_ERROR_PREFIX = "LEAN 4 WORKSPACE ERROR"


@dataclass
class _LeanVerificationOutcome:
    """Outcome of a single candidate's Lean 4 formalization pipeline (Phase A)."""
    candidate: ProofCandidate
    proof_label: str
    success: bool
    theorem_name: str
    lean_code: str
    attempts: list[ProofAttemptFeedback] = field(default_factory=list)


class ProofVerificationProviderPause(Exception):
    """Raised when proof verification must pause for provider credits."""

    def __init__(self, message: str, remaining_candidates: Optional[list[ProofCandidate]] = None):
        super().__init__(message)
        self.remaining_candidates = remaining_candidates or []


class ProofVerificationStage:
    """Run the full proof-verification checkpoint pipeline."""

    _active_sources: set[str] = set()
    _active_sources_lock: Optional[asyncio.Lock] = None
    _PROOF_CONTEXT_START = "=== VERIFIED NOVEL MATHEMATICAL PROOFS (Lean 4 Verified) ==="
    _PROOF_CONTEXT_END = "=== END VERIFIED PROOFS ==="
    _DIRECT_LEAN_TARGET_RE = re.compile(
        r"(?ms)^\s*((?:theorem|lemma)\s+[A-Za-z_][A-Za-z0-9_'.]*\b.{0,8000}?)"
        r"\s*:=\s*by\b.*?(?=^\s*(?:----|Helper Proof|SOURCE CONTEXT METADATA|"
        r"VERIFIED PROOF LIBRARY|SOURCE TYPE|SOURCE CONTENT|theorem|lemma)\b|\Z)"
    )

    def __init__(self) -> None:
        self._novelty_task_sequence = 0
        self._integrity_task_sequence = 0
        self._dependency_extractor = ProofDependencyExtractor()

    @classmethod
    def _get_active_sources_lock(cls) -> asyncio.Lock:
        if cls._active_sources_lock is None:
            cls._active_sources_lock = asyncio.Lock()
        return cls._active_sources_lock

    @classmethod
    def _source_key(cls, source_type: str, source_id: str) -> str:
        return f"{source_type}:{source_id}"

    @classmethod
    def _strip_injected_proof_context(cls, prompt: str) -> str:
        clean_prompt = prompt or ""
        while cls._PROOF_CONTEXT_START in clean_prompt:
            start = clean_prompt.find(cls._PROOF_CONTEXT_START)
            end = clean_prompt.find(cls._PROOF_CONTEXT_END, start)
            if end < 0:
                break
            end += len(cls._PROOF_CONTEXT_END)
            clean_prompt = f"{clean_prompt[:start]}\n{clean_prompt[end:]}"
        return clean_prompt.strip()

    @classmethod
    def _direct_user_prompt_candidate(cls, user_prompt: str) -> Optional[ProofCandidate]:
        clean_prompt = cls._strip_injected_proof_context(user_prompt)
        match = cls._DIRECT_LEAN_TARGET_RE.search(clean_prompt)
        if not match:
            return None

        theorem_header = match.group(1).strip()
        theorem_header = re.sub(r"\s+", " ", theorem_header)
        theorem_name_match = re.match(r"(?:theorem|lemma)\s+([A-Za-z_][A-Za-z0-9_'.]*)", theorem_header)
        theorem_id = theorem_name_match.group(1) if theorem_name_match else "direct_user_prompt_target"
        return ProofCandidate(
            theorem_id=f"direct_{theorem_id}",
            statement=theorem_header,
            formal_sketch=(
                "Direct target extracted from the user's Lean theorem prompt. "
                "Try to prove this theorem exactly first. If exact closure is not possible, "
                "only prove a faithful intermediate lemma that visibly builds toward this target."
            ),
            expected_novelty_tier="",
            prompt_relevance_rationale="This is the explicit Lean theorem requested by the user.",
            novelty_rationale=(
                "Direct user-requested Lean targets are not pre-classified as novel; "
                "the post-Lean novelty classifier must decide whether the verified "
                "result is public/citable novelty or standard known mathematics."
            ),
            why_not_standard_known_result=(
                "This is the user's concrete theorem, but if it is a standard Mathlib/textbook result "
                "the final novelty classifier should mark it not_novel."
            ),
            source_excerpt=match.group(0).strip(),
        )

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
    async def reserve_source(cls, source_type: str, source_id: str) -> None:
        """Reserve a source before background execution begins."""
        await cls._acquire_source(source_type, source_id)

    @classmethod
    async def release_source(cls, source_type: str, source_id: str) -> None:
        """Release a previously reserved source."""
        await cls._release_source(source_type, source_id)

    @classmethod
    async def _acquire_source(cls, source_type: str, source_id: str) -> None:
        async with cls._get_active_sources_lock():
            source_key = cls._source_key(source_type, source_id)
            if source_key in cls._active_sources:
                raise RuntimeError(f"Proof verification already running for {source_type} {source_id}")
            cls._active_sources.add(source_key)

    @classmethod
    async def _release_source(cls, source_type: str, source_id: str) -> None:
        async with cls._get_active_sources_lock():
            cls._active_sources.discard(cls._source_key(source_type, source_id))

    async def _broadcast(self, broadcast_fn: BroadcastFn, event: str, data: dict[str, Any]) -> None:
        if broadcast_fn:
            await broadcast_fn(event, data)

    @staticmethod
    def _role_suffix(source_type: str, override: Optional[str] = None) -> str:
        if override:
            return override
        return "brainstorm" if source_type == "brainstorm" else "paper"

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
        letter = chr(ord("A") + ((safe_index - 1) % 26))
        repeat_count = ((safe_index - 1) // 26) + 1
        return letter * repeat_count

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
            return f"Lean 4 response: {error_summary} - proof not verified."
        return "Lean 4 response: proof not verified."

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
            if is_non_retryable_model_error(exc):
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

        if proof_round_index == 1:
            direct_candidate = self._direct_user_prompt_candidate(user_prompt)
            if direct_candidate is not None:
                logger.info(
                    "ProofVerificationStage extracted direct Lean target %s for %s %s; skipping initial discovery.",
                    direct_candidate.theorem_id,
                    source_type,
                    source_id,
                )
                return [direct_candidate]

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
        release_source_on_exit: bool = True,
        should_stop: ShouldStopFn = None,
        append_to_source: bool = True,
        append_proof_callback: ProofAppendCallback = None,
        proof_candidate_indexes: Optional[dict[str, int]] = None,
        checkpoint_attempts_by_candidate: Optional[dict[str, list[ProofAttemptFeedback]]] = None,
        checkpoint_theorem_names_by_candidate: Optional[dict[str, str]] = None,
        checkpoint_callback: ProofCheckpointCallback = None,
        proof_round_index: int = 1,
        proof_max_rounds: int = 1,
        prior_round_results: str = "",
    ) -> ProofStageResult:
        """Run proof identification, formalization, Lean 4 checking, and novelty review."""
        result = ProofStageResult(source_type=source_type, source_id=source_id)
        resolved_candidates: list[ProofCandidate] = []
        candidate_indexes: dict[str, int] = dict(proof_candidate_indexes or {})
        processed_candidate_ids: set[str] = set()
        attempts_by_candidate: dict[str, list[ProofAttemptFeedback]] = {
            theorem_id: list(attempts or [])
            for theorem_id, attempts in (checkpoint_attempts_by_candidate or {}).items()
        }
        theorem_names_by_candidate: dict[str, str] = {
            theorem_id: str(theorem_name or "")
            for theorem_id, theorem_name in (checkpoint_theorem_names_by_candidate or {}).items()
            if theorem_name
        }
        checkpoint_state_lock = asyncio.Lock()

        async def save_checkpoint(status: str) -> None:
            if checkpoint_callback is None:
                return
            async with checkpoint_state_lock:
                if not resolved_candidates and status not in {"complete", "error", "no_candidates"}:
                    return
                payload = {
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
                }
            await checkpoint_callback(payload)

        def _stop_requested() -> bool:
            if should_stop is None:
                return False
            try:
                return bool(should_stop())
            except Exception:
                return False
        if not source_reserved:
            await self._acquire_source(source_type, source_id)
        try:
            base_event = {
                "source_type": source_type,
                "source_id": source_id,
                "source_title": source_title,
                "trigger": trigger,
                "proof_round_index": proof_round_index,
                "proof_max_rounds": proof_max_rounds,
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

            role_suffix = self._role_suffix(source_type, role_suffix_override)
            identification_agent = ProofIdentificationAgent(
                model_id=submitter_model,
                context_window=submitter_context,
                max_output_tokens=submitter_max_tokens,
                role_id=f"autonomous_proof_identification_{role_suffix}",
            )

            resolved_candidates = await self._resolve_candidates(
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

            result.total_candidates = len(resolved_candidates)
            await save_checkpoint("running")
            await self._broadcast(
                broadcast_fn,
                "proof_check_candidates_found",
                {
                    **base_event,
                    "count": len(resolved_candidates),
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
                    should_stop=should_stop,
                    prior_attempts=attempts_by_candidate.get(theorem_candidate.theorem_id, []),
                    prior_theorem_name=theorem_names_by_candidate.get(theorem_candidate.theorem_id, ""),
                    attempt_checkpoint_callback=record_attempts,
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
                                result.had_error = True
                                result.error_message = error_summary
                            if source_type == "brainstorm" and trigger != "retry" and not context_overflow:
                                await novel_proofs_db.record_failed_candidate(
                                    source_id,
                                    candidate,
                                    error_summary,
                                    suggested_lemma_targets=suggested_targets,
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
                            processed_candidate_ids.add(candidate.theorem_id)
                            mark_batch_outcome_processed(batch_index)
                            await save_checkpoint("running")
                            continue

                        integrity_task_id = f"proof_integrity_{self._integrity_task_sequence:03d}"
                        self._integrity_task_sequence += 1
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
                            role_id="autonomous_proof_novelty",
                            require_statement_alignment=True,
                        )
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
                            processed_candidate_ids.add(candidate.theorem_id)
                            mark_batch_outcome_processed(batch_index)
                            await save_checkpoint("running")
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

                        registration = await register_verified_lean_proof(
                            proof_database=novel_proofs_db,
                            user_prompt=user_prompt,
                            theorem_statement=stored_theorem_statement,
                            lean_code=lean_code,
                            validator_model=validator_model,
                            validator_context=validator_context,
                            validator_max_tokens=validator_max_tokens,
                            task_id=novelty_task_id,
                            role_id="autonomous_proof_novelty",
                            source_type=source_type,
                            source_id=source_id,
                            source_title=source_title,
                            theorem_id=candidate.theorem_id,
                            theorem_name=stored_theorem_name,
                            formal_sketch=stored_formal_sketch,
                            solver="Lean 4",
                            verification_notes=verification_notes,
                            attempt_count=len(attempts),
                            attempts=attempts,
                            solver_hints=solver_hints,
                            broadcast_fn=broadcast_fn,
                            base_event=base_event,
                            proof_label=proof_label,
                            retry_origin_source_id=candidate.origin_source_id,
                        )
                        stored_record = registration.record
                        is_novel = stored_record.novel
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
                            if dependencies:
                                updated_record = await novel_proofs_db.update_proof_dependencies(
                                    stored_record.proof_id,
                                    dependencies,
                                )
                                if updated_record is not None:
                                    stored_record = updated_record
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
                            logger.debug(
                                "Dependency extraction failed for theorem %s: %s",
                                candidate.theorem_id,
                                exc,
                            )

                        if candidate.origin_source_id:
                            await novel_proofs_db.mark_resolved_retry(
                                candidate.origin_source_id,
                                candidate.theorem_id,
                                stored_record.proof_id,
                            )

                        if self._should_append_verified_proof(
                            is_novel=is_novel,
                            duplicate=registration.duplicate,
                            append_proof_callback=append_proof_callback,
                            append_known_proofs=self._should_append_known_proofs_for_trigger(trigger),
                        ):
                            if is_novel and not registration.duplicate:
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
                        processed_candidate_ids.add(candidate.theorem_id)
                        mark_batch_outcome_processed(batch_index)
                        await save_checkpoint("running")
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

            direct_prompt_target_failed = (
                theorem_candidates is None
                and proof_round_index == 1
                and trigger.startswith("manual")
                and not trigger.endswith("_fallback")
                and result.verified_count == 0
                and len(resolved_candidates) == 1
                and resolved_candidates[0].theorem_id.startswith("direct_")
            )
            if direct_prompt_target_failed and not _stop_requested():
                fallback_prior = (
                    "The exact Lean theorem requested by the user was tried through "
                    "all configured Lean attempts, but did not verify. Now look only "
                    "for intermediate lemmas or supporting theorems that would help "
                    "prove that exact requested theorem. Do not collect merely "
                    "brainstorm-related or background proofs."
                )
                has_fallback_candidates, fallback_candidates = await identification_agent.identify_candidates(
                    user_research_prompt=user_prompt,
                    source_type=source_type,
                    source_id=source_id,
                    source_content=content,
                    source_title=source_title,
                    proof_round_index=proof_round_index,
                    proof_max_rounds=proof_max_rounds,
                    prior_round_results=fallback_prior,
                )
                if has_fallback_candidates and fallback_candidates:
                    fallback_result = await self.run(
                        content=content,
                        source_type=source_type,
                        source_id=source_id,
                        user_prompt=user_prompt,
                        submitter_model=submitter_model,
                        submitter_context=submitter_context,
                        submitter_max_tokens=submitter_max_tokens,
                        validator_model=validator_model,
                        validator_context=validator_context,
                        validator_max_tokens=validator_max_tokens,
                        broadcast_fn=broadcast_fn,
                        novel_proofs_db=novel_proofs_db,
                        source_title=source_title,
                        theorem_candidates=fallback_candidates,
                        role_suffix_override=role_suffix_override,
                        trigger=f"{trigger}_fallback",
                        source_reserved=True,
                        release_source_on_exit=False,
                        should_stop=should_stop,
                        append_to_source=append_to_source,
                        append_proof_callback=append_proof_callback,
                        proof_round_index=proof_round_index,
                        proof_max_rounds=proof_max_rounds,
                        prior_round_results=fallback_prior,
                    )
                    fallback_result.results = result.results + fallback_result.results
                    fallback_result.total_candidates += result.total_candidates
                    fallback_result.verified_count += result.verified_count
                    fallback_result.novel_count += result.novel_count
                    await self._broadcast(
                        broadcast_fn,
                        "proof_check_complete",
                        {
                            **base_event,
                            "novel_count": fallback_result.novel_count,
                            "verified_count": fallback_result.verified_count,
                            "total_candidates": fallback_result.total_candidates,
                            "message": (
                                "Direct target attempt completed; fallback discovery "
                                "was also checked for prompt-solving support."
                            ),
                        },
                    )
                    return fallback_result

            await save_checkpoint("complete")
            await self._broadcast(
                broadcast_fn,
                "proof_check_complete",
                {
                    **base_event,
                    "novel_count": result.novel_count,
                    "verified_count": result.verified_count,
                    "total_candidates": result.total_candidates,
                },
            )
            return result
        except ProofVerificationProviderPause:
            raise
        except FreeModelExhaustedError:
            await save_checkpoint("provider_paused")
            raise
        except Exception as exc:
            if is_non_retryable_model_error(exc):
                await save_checkpoint("provider_paused")
                raise
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
                        f"{self._summarize_error(str(exc), limit=960)}"
                    ),
                },
            )
            return result
        finally:
            if release_source_on_exit:
                await self._release_source(source_type, source_id)

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
    ) -> _LeanVerificationOutcome:
        """Phase A for one candidate: lemma prep, SMT hint, and Lean 4 attempts.

        Each invocation creates its own agent instances so that concurrent
        candidates do not race on shared ``task_sequence`` counters and so the
        ``role_id`` remains the same for all attempts belonging to one
        candidate.
        """
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
        if smt_hint:
            candidate = candidate.model_copy(update={"smt_hint": smt_hint})
        if trigger == "retry" and candidate.origin_source_id:
            await novel_proofs_db.mark_retried(
                candidate.origin_source_id,
                candidate.theorem_id,
                source_id,
            )

        active_attempts: list[ProofAttemptFeedback] = list(prior_attempts or [])
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
                        "error_summary": self._summarize_error(feedback.error_output),
                        "lean_response": lean_response,
                        "proof_verified": False,
                        "retry_origin_source_id": current_candidate.origin_source_id,
                    },
                )

        full_attempt_count = sum(1 for attempt in active_attempts if attempt.strategy == "full_script")
        tactic_attempt_count = sum(1 for attempt in active_attempts if attempt.strategy == "tactic_script")
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

        if not success and not workspace_error and not context_overflow and not (should_stop and should_stop()):
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
        )

    async def run_manual(
        self,
        *,
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
        source_reserved: bool = False,
        append_to_source: bool = True,
        append_proof_callback: ProofAppendCallback = None,
        should_stop: ShouldStopFn = None,
    ) -> ProofStageResult:
        """Run a user-triggered proof check using manual proof role IDs."""
        return await self.run(
            content=content,
            source_type=source_type,
            source_id=source_id,
            user_prompt=user_prompt,
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
            release_source_on_exit=True,
            append_to_source=append_to_source,
            append_proof_callback=append_proof_callback,
            should_stop=should_stop,
        )
