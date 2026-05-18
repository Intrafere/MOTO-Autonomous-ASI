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
from backend.aggregator.prompts.validator_prompts import build_validator_prompt
from backend.shared.api_client_manager import api_client_manager
from backend.shared.brainstorm_proof_gate import BRAINSTORM_LEAN_PROOF_MARKER
from backend.shared.config import system_config
from backend.shared.json_parser import parse_json
from backend.shared.lean_proof_integrity import validate_full_lean_proof_integrity
from backend.shared.model_error_utils import is_non_retryable_model_error
from backend.shared.models import ProofAttemptFeedback, ProofAttemptResult, ProofCandidate, ProofStageResult, SmtHint
from backend.shared.openrouter_client import FreeModelExhaustedError
from backend.shared.smt_client import get_smt_client
from .proof_dependency_extractor import ProofDependencyExtractor

logger = logging.getLogger(__name__)

BroadcastFn = Optional[Callable[[str, dict[str, Any]], Awaitable[None]]]
ShouldStopFn = Optional[Callable[[], bool]]
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


class ProofVerificationStage:
    """Run the full proof-verification checkpoint pipeline."""

    _active_sources: set[str] = set()
    _active_sources_lock: Optional[asyncio.Lock] = None

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
    async def is_source_running(cls, source_type: str, source_id: str) -> bool:
        async with cls._get_active_sources_lock():
            return cls._source_key(source_type, source_id) in cls._active_sources

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
        identification_agent: ProofIdentificationAgent,
        broadcast_fn: BroadcastFn,
    ) -> Optional[SmtHint]:
        if not system_config.smt_enabled or not self._is_smt_amenable(candidate):
            return None

        started_at = time.monotonic()
        result_name = "unknown"
        try:
            smtlib = await identification_agent.translate_candidate_to_smt(
                user_research_prompt=user_prompt,
                source_type=source_type,
                theorem_candidate=candidate,
                source_content=source_content,
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
        content: str,
    ) -> list[ProofCandidate]:
        if theorem_candidates is not None:
            return theorem_candidates

        has_candidates, resolved_candidates = await identification_agent.identify_candidates(
            user_research_prompt=user_prompt,
            source_type=source_type,
            source_id=source_id,
            source_content=content,
        )
        return resolved_candidates if has_candidates else []

    async def _prepare_candidate(
        self,
        *,
        user_prompt: str,
        source_type: str,
        theorem_candidate: ProofCandidate,
        source_content: str,
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
        )
        if relevant_lemmas:
            candidate = candidate.model_copy(update={"relevant_lemmas": relevant_lemmas})
        return candidate

    @staticmethod
    def _format_verified_proof_for_brainstorm_validation(
        *,
        theorem_statement: str,
        formal_sketch: str,
        lean_code: str,
        attempt_count: int,
    ) -> str:
        sections = [
            BRAINSTORM_LEAN_PROOF_MARKER,
            "",
            "Lean 4 has accepted the following proof. Decide whether it is useful, non-redundant brainstorm progress before it is appended to the brainstorm database.",
            "",
            f"Theorem statement: {theorem_statement}",
        ]
        if formal_sketch:
            sections.extend(["", f"Formalization notes: {formal_sketch}"])
        sections.extend(
            [
                "",
                f"Lean verification: accepted after {attempt_count} attempt{'s' if attempt_count != 1 else ''}.",
                "",
                "Lean 4 code:",
                "```lean",
                lean_code,
                "```",
            ]
        )
        return "\n".join(sections).strip()

    async def _validate_brainstorm_verified_proof_addition(
        self,
        *,
        user_prompt: str,
        source_content: str,
        proof_submission: str,
        validator_model: str,
        validator_context: int,
        validator_max_tokens: int,
        task_id: str,
        role_id: str,
        broadcast_fn: BroadcastFn,
        base_event: dict[str, Any],
    ) -> bool:
        """Run the normal brainstorm usefulness gate before appending verified proofs."""
        context = source_content or ""
        while len(context) > 24000:
            context = context[: max(len(context) // 2, 24000)]
        prompt = build_validator_prompt(
            user_prompt=user_prompt,
            submission_content=proof_submission,
            context=f"CURRENT BRAINSTORM DATABASE:\n{context}",
        )
        try:
            response = await api_client_manager.generate_completion(
                task_id=task_id,
                role_id=role_id,
                model=validator_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=validator_max_tokens,
                temperature=0.0,
            )
            if not response or not response.get("choices"):
                raise ValueError("Proof brainstorm validator returned no choices.")
            message = response["choices"][0].get("message", {})
            content = message.get("content") or message.get("reasoning") or ""
            raw = parse_json(content)
            if isinstance(raw, list):
                raw = raw[0] if raw else {}
            if not isinstance(raw, dict):
                raw = {}
            accepted = str(raw.get("decision") or "").strip().lower() == "accept"
            await self._broadcast(
                broadcast_fn,
                "proof_brainstorm_validation_complete",
                {
                    **base_event,
                    "accepted": accepted,
                    "reasoning": str(raw.get("reasoning") or raw.get("summary") or ""),
                },
            )
            return accepted
        except Exception as exc:
            if is_non_retryable_model_error(exc):
                raise
            logger.warning("Verified brainstorm proof usefulness validation failed: %s", exc)
            await self._broadcast(
                broadcast_fn,
                "proof_brainstorm_validation_complete",
                {
                    **base_event,
                    "accepted": False,
                    "reasoning": f"Validator failed before producing a usable decision: {exc}",
                },
            )
            return False

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
        should_stop: ShouldStopFn = None,
        append_to_source: bool = True,
    ) -> ProofStageResult:
        """Run proof identification, formalization, Lean 4 checking, and novelty review."""
        result = ProofStageResult(source_type=source_type, source_id=source_id)
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
                content=content,
            )

            if not resolved_candidates:
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
            await self._broadcast(
                broadcast_fn,
                "proof_check_candidates_found",
                {
                    **base_event,
                    "count": len(resolved_candidates),
                    "theorems_preview": [
                        f"Proof {self._proof_label_for_index(index)}: {candidate.statement[:180]}"
                        for index, candidate in enumerate(resolved_candidates, start=1)
                    ],
                },
            )

            max_parallel = max(1, int(getattr(system_config, "proof_max_parallel_candidates", 6) or 1))
            semaphore = asyncio.Semaphore(max_parallel)

            async def run_phase_a(theorem_candidate: ProofCandidate, proof_label: str) -> _LeanVerificationOutcome:
                async with semaphore:
                    if _stop_requested():
                        return _LeanVerificationOutcome(
                            candidate=theorem_candidate,
                            proof_label=proof_label,
                            success=False,
                            theorem_name="",
                            lean_code="",
                            attempts=[],
                        )
                    return await self._run_lean_pipeline_for_candidate(
                        theorem_candidate=theorem_candidate,
                        base_event=base_event,
                        proof_label=proof_label,
                        user_prompt=user_prompt,
                        source_type=source_type,
                        source_id=source_id,
                        source_content=content,
                        submitter_model=submitter_model,
                        submitter_context=submitter_context,
                        submitter_max_tokens=submitter_max_tokens,
                        role_suffix=role_suffix,
                        trigger=trigger,
                        novel_proofs_db=novel_proofs_db,
                        broadcast_fn=broadcast_fn,
                        should_stop=should_stop,
                    )

            verification_tasks = [
                asyncio.create_task(run_phase_a(candidate, self._proof_label_for_index(index)))
                for index, candidate in enumerate(resolved_candidates, start=1)
            ]

            pending_tasks = set(verification_tasks)
            try:
                for future in asyncio.as_completed(verification_tasks):
                    if _stop_requested():
                        logger.info(
                            "Proof verification stopping early for %s %s (stop requested before next outcome).",
                            source_type,
                            source_id,
                        )
                        for task in pending_tasks:
                            if not task.done():
                                task.cancel()
                        await asyncio.gather(*pending_tasks, return_exceptions=True)
                        break
                    try:
                        outcome = await future
                    except FreeModelExhaustedError:
                        for task in pending_tasks:
                            if not task.done():
                                task.cancel()
                        await asyncio.gather(*pending_tasks, return_exceptions=True)
                        raise
                    except asyncio.CancelledError:
                        pending_tasks = {task for task in pending_tasks if not task.done()}
                        continue
                    except Exception as exc:
                        # Any other per-candidate exception aborts the whole
                        # parallel batch; the outer `except Exception` handler
                        # will broadcast `proof_check_complete` with the error.
                        logger.error(
                            "Proof verification candidate task failed for %s %s: %s",
                            source_type,
                            source_id,
                            exc,
                        )
                        for task in pending_tasks:
                            if not task.done():
                                task.cancel()
                        await asyncio.gather(*pending_tasks, return_exceptions=True)
                        raise

                    pending_tasks = {task for task in pending_tasks if not task.done()}

                    # Skip the expensive Phase B post-processing (novelty,
                    # dependency extraction, DB writes) if the user has asked
                    # us to stop. The outcome itself is dropped.
                    if _stop_requested():
                        logger.info(
                            "Proof verification skipping phase B for %s %s (stop requested).",
                            source_type,
                            source_id,
                        )
                        for task in pending_tasks:
                            if not task.done():
                                task.cancel()
                        await asyncio.gather(*pending_tasks, return_exceptions=True)
                        break

                    candidate = outcome.candidate
                    proof_label = outcome.proof_label
                    attempts = outcome.attempts
                    lean_code = outcome.lean_code

                    if not outcome.success:
                        error_summary = self._summarize_error(attempts[-1].error_output if attempts else "")
                        suggested_targets = self._extract_suggested_lemma_targets(
                            attempts[-1].error_output if attempts else ""
                        )
                        if source_type == "brainstorm" and trigger != "retry":
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
                        continue

                    novelty_task_id = f"proof_novelty_{self._novelty_task_sequence:03d}"
                    self._novelty_task_sequence += 1

                    solver_hints = []
                    if self._first_attempt_used_smt_hint(attempts, candidate.smt_hint):
                        solver_hints.append("smt-z3")

                    registration = await register_verified_lean_proof(
                        proof_database=novel_proofs_db,
                        user_prompt=user_prompt,
                        theorem_statement=candidate.statement,
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
                        theorem_name=outcome.theorem_name,
                        formal_sketch=candidate.formal_sketch,
                        solver="Lean 4",
                        verification_notes="Lean 4 accepted the submitted proof.",
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
                    novelty_tier = stored_record.novelty_tier
                    result.verified_count += 1

                    await self._broadcast(
                        broadcast_fn,
                        "proof_verified",
                        {
                            **base_event,
                            "proof_id": stored_record.proof_id,
                            "theorem_id": candidate.theorem_id,
                            "theorem_statement": candidate.statement,
                            "proof_label": proof_label,
                            "strategy": attempts[-1].strategy if attempts else "full_script",
                            "retry_origin_source_id": candidate.origin_source_id,
                        },
                    )

                    # Dependency extraction runs in Phase B so later candidates
                    # in the same paper can see earlier proofs. We instantiate
                    # a scoped lemma search agent here (the Phase A agents are
                    # already owned by their candidate tasks).
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
                            theorem_name=outcome.theorem_name,
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

                    if is_novel and not registration.duplicate:
                        result.novel_count += 1
                        # Novel proofs are appended to their source document so the
                        # paper/brainstorm they came from retains a record of them.
                        # They are also stored in ProofDatabase and direct-injected
                        # into all prompts via inject_into_prompt().
                        if append_to_source and source_type == "brainstorm":
                            validator_accepted = await self._validate_brainstorm_verified_proof_addition(
                                user_prompt=user_prompt,
                                source_content=content,
                                proof_submission=self._format_verified_proof_for_brainstorm_validation(
                                    theorem_statement=candidate.statement,
                                    formal_sketch=candidate.formal_sketch,
                                    lean_code=lean_code,
                                    attempt_count=len(attempts),
                                ),
                                validator_model=validator_model,
                                validator_context=validator_context,
                                validator_max_tokens=validator_max_tokens,
                                task_id=f"proof_brainstorm_val_{self._novelty_task_sequence:03d}",
                                role_id="autonomous_proof_novelty",
                                broadcast_fn=broadcast_fn,
                                base_event={
                                    **base_event,
                                    "theorem_id": candidate.theorem_id,
                                    "theorem_statement": candidate.statement,
                                    "proof_id": stored_record.proof_id,
                                    "proof_label": proof_label,
                                },
                            )
                            if validator_accepted:
                                await brainstorm_memory.append_proofs_section(source_id, stored_record)
                        elif append_to_source and source_type == "paper":
                            await paper_library.append_proofs_section(source_id, stored_record)
                    # Non-novel (known) proofs are stored in ProofDatabase only.
                    # They are NOT appended to brainstorm/paper files to avoid
                    # polluting compiler and RAG context with standard Lean 4 code.
                    # They remain browsable via proof_database.get_known_proofs_summary_for_browsing().

                    result.results.append(
                        ProofAttemptResult(
                            theorem_id=candidate.theorem_id,
                            theorem_statement=candidate.statement,
                            lean_code=lean_code,
                            success=True,
                            novel=is_novel,
                            attempts_used=len(attempts),
                            proof_id=stored_record.proof_id,
                            error_summary="",
                        )
                    )
            finally:
                # Defensive cleanup: make sure we don't leak pending tasks if
                # the consumer loop exits early for any reason.
                leftover = [task for task in verification_tasks if not task.done()]
                for task in leftover:
                    task.cancel()
                if leftover:
                    await asyncio.gather(*leftover, return_exceptions=True)

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
        except FreeModelExhaustedError:
            raise
        except Exception as exc:
            if is_non_retryable_model_error(exc):
                raise
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
                    "novel_count": result.novel_count,
                    "verified_count": result.verified_count,
                    "total_candidates": result.total_candidates,
                    "message": "Proof verification encountered an error",
                },
            )
            return result
        finally:
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
        submitter_model: str,
        submitter_context: int,
        submitter_max_tokens: int,
        role_suffix: str,
        trigger: str,
        novel_proofs_db,
        broadcast_fn: BroadcastFn,
        should_stop: ShouldStopFn = None,
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

        success, theorem_name, lean_code, attempts = await formalization_agent.prove_candidate(
            user_research_prompt=user_prompt,
            source_type=source_type,
            theorem_candidate=candidate,
            source_content=source_content,
            max_attempts=3,
            attempt_callback=on_attempt_feedback,
            attempt_start_callback=on_attempt_started,
            smt_hint=candidate.smt_hint,
            should_stop=should_stop,
        )
        workspace_error = bool(
            attempts
            and (attempts[-1].error_output or "").startswith(LEAN_WORKSPACE_ERROR_PREFIX)
        )
        if not success and not workspace_error and not (should_stop and should_stop()):
            tactic_success, tactic_theorem_name, lean_code, attempts = await formalization_agent.prove_candidate_tactic_script(
                user_research_prompt=user_prompt,
                source_type=source_type,
                theorem_candidate=candidate,
                source_content=source_content,
                max_attempts=2,
                attempt_callback=on_attempt_feedback,
                attempt_start_callback=on_attempt_started,
                prior_attempts=attempts,
                starting_attempt_number=(attempts[-1].attempt + 1 if attempts else 4),
                smt_hint=candidate.smt_hint,
                should_stop=should_stop,
            )
            if tactic_theorem_name:
                theorem_name = tactic_theorem_name
            success = tactic_success

        if not success and not workspace_error and not (should_stop and should_stop()):
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
            should_stop=should_stop,
        )
