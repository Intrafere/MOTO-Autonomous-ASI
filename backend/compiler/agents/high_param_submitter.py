"""
High-parameter submitter agent for the compiler's rigor loop.

The rigor loop no longer rewrites paper text. Instead it runs a two-stage
Lean-4-verified-theorem flow (see RIGOR_LEAN_BUILD_PLAN.md):

    Stage 1 (discovery): pick a theorem worth formalizing using the full
        writing context.
    Stage 2 (formalization): hand the candidate to ProofFormalizationAgent
        for up to 5 Lean 4 attempts with error-feedback chaining.
    Stage 3 (novelty): classify the verified proof and persist it via
        proof_database.add_proof.
    Stage 4 (placement): propose an inline edit that introduces the
        theorem with a "verified in Lean 4" marker and an appendix
        reference. The coordinator owns the 2-attempt validator retry loop
        and the appendix fallback.

The Wolfram sub-mode that used to live here has been removed in Phase 2.
Wolfram Alpha is now a tool available to HighContextSubmitter.submit_construction
(see Phase 3 of the build plan).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from backend.autonomous.memory.proof_database import proof_database
from backend.compiler.core.compiler_rag_manager import compiler_rag_manager
from backend.compiler.memory.outline_memory import outline_memory
from backend.compiler.memory.paper_memory import (
    paper_memory,
)
from backend.compiler.prompts.rigor_prompts import (
    build_rigor_placement_prompt,
    build_rigor_theorem_discovery_prompt,
)
from backend.shared.api_client_manager import api_client_manager
from backend.shared.config import rag_config, system_config
from backend.shared.json_parser import parse_json
from backend.shared.lm_studio_client import lm_studio_client
from backend.shared.models import (
    CompilerSubmission,
    ProofAttemptFeedback,
    ProofCandidate,
    ProofRecord,
)
from backend.shared.utils import count_tokens

logger = logging.getLogger(__name__)


def _normalize_string_field(value) -> str:
    """Normalize string field from LLM response (tolerates list-of-strings mistakes)."""
    if isinstance(value, list):
        logger.warning(f"LLM returned field as list (length {len(value)}), converting to string")
        return " ".join(str(item) for item in value if item)
    elif isinstance(value, str):
        return value
    elif value is None:
        return ""
    else:
        logger.warning(f"LLM returned field as {type(value)}, converting to string")
        return str(value)


def _strip_paper_markers_for_llm(paper_content: str) -> str:
    """Prepare paper text before handing it to the LLM.

    The submitter must see the same editable paper text that exact-match
    validation checks. Keep placeholders and theorem appendix bracket markers
    visible so old_string anchors can be copied verbatim from the real paper.
    """
    if not paper_content:
        return ""
    return paper_content.strip()


def format_theorem_appendix_entry(
    *,
    proof_id: str,
    theorem_statement: str,
    lean_code: str,
    is_novel: bool,
    theorem_name: str = "",
    placement_outcome: str = "appendix_fallback",
) -> str:
    """Format a verified-theorem entry for the Theorems Appendix.

    Used both when placement is inline (a short cross-reference stub) and
    when placement fails and the full entry is the only record (appendix
    fallback). Caller selects via `placement_outcome`.
    """
    header_name = theorem_name.strip() or proof_id
    novelty_label = "Novel" if is_novel else "Known"
    status_suffix = {
        "appendix_fallback": "inline placement rejected; preserved here because Lean 4 verified the math",
        "inline": "also placed inline in the body",
    }.get(placement_outcome, placement_outcome)

    lines = [
        f"Theorem ({proof_id}) [{novelty_label}] - {header_name}",
        f"Status: verified by Lean 4 ({status_suffix})",
        f"Statement: {theorem_statement.strip()}",
        "Lean 4 proof:",
        lean_code.strip() or "[lean code unavailable]",
        "---",
    ]
    return "\n".join(lines)


@dataclass
class RigorTheoremResult:
    """Bundle returned from submit_rigor_lean_theorem on a verified proof.

    The coordinator owns the 2-attempt validator loop and the appendix
    fallback, so the submitter returns everything the coordinator needs to
    drive retries without re-running discovery / formalization.
    """
    proof_id: str
    theorem_statement: str
    theorem_name: str
    lean_code: str
    is_novel: bool
    novelty_reasoning: str
    attempts: List[ProofAttemptFeedback]
    source_id: str
    initial_placement_submission: Optional[CompilerSubmission] = None
    # Retained for retry-prompt assembly
    formal_sketch: str = ""
    source_excerpt: str = ""
    # Metadata pass-through
    metadata: Dict[str, Any] = field(default_factory=dict)


class HighParamSubmitter:
    """High-parameter submitter for the compiler's rigor loop.

    Drives the Lean-4-verified-theorem flow end-to-end: discovery -> 5 Lean
    attempts -> novelty classification -> persist -> initial placement
    submission. Placement retries are driven by `submit_rigor_placement_retry`
    (called by the coordinator after a validator rejection).
    """

    def __init__(
        self,
        model_name: str,
        user_prompt: str,
        websocket_broadcaster: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None,
    ):
        self.model_name = model_name
        # NOTE: proof_database.inject_into_prompt prepends all novel proofs
        # so later discovery calls naturally avoid re-proposing them.
        self.user_prompt = proof_database.inject_into_prompt(user_prompt)
        self.raw_user_prompt = user_prompt
        self.websocket_broadcaster = websocket_broadcaster
        self._initialized = False
        self._standalone_session_id = f"standalone_{uuid.uuid4().hex[:12]}"

        # Task tracking for workflow panel and boost integration
        self.task_sequence: int = 0
        self.role_id = "compiler_high_param"
        self.task_tracking_callback: Optional[Callable[[str, str], None]] = None

        # Populated by initialize()
        self.context_window: int = system_config.compiler_high_param_context_window
        self.max_output_tokens: int = system_config.compiler_high_param_max_output_tokens
        self.available_input_tokens: int = rag_config.get_available_input_tokens(
            self.context_window, self.max_output_tokens
        )

    # ------------------------------------------------------------------ setup

    def set_task_tracking_callback(self, callback: Callable[[str, str], None]) -> None:
        self.task_tracking_callback = callback

    def get_current_task_id(self) -> str:
        return f"comp_hp_{self.task_sequence:03d}"

    async def initialize(self) -> None:
        if self._initialized:
            return

        self.context_window = system_config.compiler_high_param_context_window
        self.max_output_tokens = system_config.compiler_high_param_max_output_tokens
        self.available_input_tokens = rag_config.get_available_input_tokens(
            self.context_window, self.max_output_tokens
        )

        self._initialized = True
        logger.info(f"High-param submitter initialized with model: {self.model_name}")
        logger.info(
            f"Context budget: {self.available_input_tokens} tokens "
            f"(window: {self.context_window})"
        )

    # -------------------------------------------------------- broadcast helpers

    async def _broadcast(self, event: str, data: Dict[str, Any]) -> None:
        if not self.websocket_broadcaster:
            return
        try:
            await self.websocket_broadcaster(event, data)
        except Exception as exc:
            logger.debug("Rigor broadcast failed (%s): %s", event, exc)

    # -------------------------------------------------------- session helpers

    def _resolve_session_id(self) -> str:
        """Best-effort session id for proof / failure tracking.

        When the autonomous session manager is active, proof_database is
        already storing in the session directory. Otherwise each manual
        compiler instance gets its own id so failed theorem candidates do not
        bleed into later standalone compiler runs.
        """
        sm = getattr(proof_database, "_session_manager", None)
        if sm is not None and getattr(sm, "is_session_active", False):
            return str(getattr(sm, "session_id", "") or "autonomous_active")
        return self._standalone_session_id

    def _compiler_source_id(self) -> str:
        """Source id used on ProofRecord / failed candidate storage.

        Format: ``compiler_rigor:<session>``. The session suffix lets the
        failure-hint log cleanly scope retries per session (same as how
        brainstorm-driven proofs scope by brainstorm id).
        """
        return f"compiler_rigor:{self._resolve_session_id()}"

    # ---------------------------------------------------- context assembly

    async def _build_rigor_rag_context(
        self,
        *,
        query_seed: str,
        reserved_tokens: int,
    ) -> str:
        """Retrieve RAG evidence for the rigor prompts.

        Mirrors the HighContextSubmitter.submit_construction budget
        pattern: outline + paper are direct-injected by the caller, so
        we exclude them from RAG. The remaining budget goes to the
        RAG offload priority (Shared Training DB -> Local Submitter DB
        -> Rejection Log -> User Upload Files) handled inside the
        aggregator RAG manager.
        """
        max_allowed = rag_config.get_available_input_tokens(
            self.context_window, self.max_output_tokens
        )
        remaining = max(1000, max_allowed - reserved_tokens - 200)

        try:
            context_pack = await compiler_rag_manager.retrieve_for_mode(
                query=query_seed,
                mode="rigor",
                max_tokens=remaining,
                exclude_sources=["compiler_outline.txt", "compiler_paper.txt"],
            )
            return context_pack.text or ""
        except Exception as exc:
            logger.warning("Rigor RAG retrieval failed (%s); proceeding without RAG", exc)
            return ""

    # -------------------------------------------------------- public entrypoint

    async def submit_rigor_lean_theorem(self) -> Optional[RigorTheoremResult]:
        """Run discovery + 5 Lean 4 attempts + novelty + initial placement.

        Returns a RigorTheoremResult on a verified proof (coordinator then
        drives the 2-attempt placement validator loop + appendix fallback).
        Returns None on any decline path: no theorem worth trying, 5 Lean
        attempts failed, or the placement submitter refused on attempt 1.
        """
        # Guard: if Lean 4 is disabled system-wide, there is nothing this
        # submitter can do - the coordinator also guards on this but we add
        # a belt-and-suspenders check here so callers can't bypass it.
        if not system_config.lean4_enabled:
            logger.info("submit_rigor_lean_theorem: Lean 4 disabled; declining rigor cycle")
            return None

        logger.info("Rigor cycle: Stage 1 - theorem discovery")
        discovery = await self._step_discovery()
        if discovery is None:
            logger.info("Rigor cycle: discovery declined")
            return None

        theorem_statement = str(discovery.get("theorem_statement") or "").strip()
        formal_sketch = str(discovery.get("formal_sketch") or "").strip()
        source_excerpt = str(discovery.get("source_excerpt") or "").strip()
        retry_failure_id = str(discovery.get("retry_existing_failure_id") or "").strip()

        if not theorem_statement:
            logger.info("Rigor cycle: discovery returned empty theorem_statement; declining")
            return None

        logger.info(
            "Rigor cycle: Stage 2 - Lean 4 formalization (up to 5 attempts), "
            f"retry_failure_id={retry_failure_id or 'none'}"
        )

        candidate = ProofCandidate(
            theorem_id=retry_failure_id or f"compiler_rigor_{uuid.uuid4().hex[:12]}",
            statement=theorem_statement,
            formal_sketch=formal_sketch,
            source_excerpt=source_excerpt,
            origin_source_id=self._compiler_source_id() if retry_failure_id else "",
        )

        formalizer_result = await self._step_formalize(candidate, theorem_statement)
        if formalizer_result is None:
            return None

        theorem_name, lean_code, attempts = formalizer_result

        logger.info("Rigor cycle: Stage 3 - novelty classification + persistence")
        is_novel, novelty_reasoning, stored_record = await self._step_assess_novelty_and_store(
            theorem_statement=theorem_statement,
            theorem_name=theorem_name,
            lean_code=lean_code,
            formal_sketch=formal_sketch,
            attempts=attempts,
        )

        await self._broadcast(
            "proof_verified",
            {
                "source_type": "compiler_rigor",
                "source_id": self._compiler_source_id(),
                "theorem_id": candidate.theorem_id,
                "theorem_statement": theorem_statement,
                "proof_id": stored_record.proof_id,
                "is_novel": is_novel,
            },
        )

        # If we retried a previously-failed candidate and it succeeded, mark it
        # resolved so it stops appearing in future failure-hint lists.
        if retry_failure_id:
            try:
                await proof_database.mark_resolved_retry(
                    source_brainstorm_id=self._compiler_source_id(),
                    theorem_id=retry_failure_id,
                    proof_id=stored_record.proof_id,
                )
            except Exception as exc:
                logger.debug("mark_resolved_retry failed (non-fatal): %s", exc)

        logger.info("Rigor cycle: Stage 4 - initial placement proposal")
        initial_submission = await self._step_initial_placement(
            proof_id=stored_record.proof_id,
            theorem_statement=theorem_statement,
            theorem_name=theorem_name,
            lean_code=lean_code,
            is_novel=is_novel,
        )

        return RigorTheoremResult(
            proof_id=stored_record.proof_id,
            theorem_statement=theorem_statement,
            theorem_name=theorem_name,
            lean_code=lean_code,
            is_novel=is_novel,
            novelty_reasoning=novelty_reasoning,
            attempts=attempts,
            source_id=self._compiler_source_id(),
            initial_placement_submission=initial_submission,
            formal_sketch=formal_sketch,
            source_excerpt=source_excerpt,
            metadata={
                "retry_failure_id": retry_failure_id,
                "attempt_count": len(attempts),
            },
        )

    # --------------------------------------------------------- stage 1

    async def _step_discovery(self) -> Optional[dict]:
        """Ask the LLM whether a Lean 4 theorem is worth pursuing right now."""
        current_outline = await outline_memory.get_outline()
        current_paper_raw = await paper_memory.get_paper()
        current_paper = _strip_paper_markers_for_llm(current_paper_raw)

        # Existing verified proofs - compact blob of statements so the model
        # can recognize duplicates without blowing the token budget.
        existing_proofs: List[dict] = []
        try:
            for record in await proof_database.get_all_proofs():
                existing_proofs.append(
                    {
                        "proof_id": record.proof_id,
                        "novel": record.novel,
                        "theorem_statement": record.theorem_statement,
                    }
                )
        except Exception as exc:
            logger.debug("proof_database.get_all_proofs failed: %s", exc)

        try:
            failure_hints = await proof_database.get_recent_failure_hints(
                self._compiler_source_id(), limit=5
            )
        except Exception as exc:
            logger.debug("proof_database.get_recent_failure_hints failed: %s", exc)
            failure_hints = []

        # Build with empty RAG first to measure the mandatory footprint,
        # then allocate the rest to RAG.
        base_prompt = await build_rigor_theorem_discovery_prompt(
            user_prompt=self.user_prompt,
            current_outline=current_outline,
            current_paper=current_paper,
            rag_evidence="",
            existing_verified_proofs=existing_proofs,
            recent_failure_hints=failure_hints,
        )
        mandatory_tokens = count_tokens(base_prompt)
        query_seed = (self.raw_user_prompt + " " + current_paper[-1500:]).strip()
        rag_evidence = await self._build_rigor_rag_context(
            query_seed=query_seed,
            reserved_tokens=mandatory_tokens,
        )

        prompt = await build_rigor_theorem_discovery_prompt(
            user_prompt=self.user_prompt,
            current_outline=current_outline,
            current_paper=current_paper,
            rag_evidence=rag_evidence,
            existing_verified_proofs=existing_proofs,
            recent_failure_hints=failure_hints,
        )

        max_allowed = rag_config.get_available_input_tokens(
            self.context_window, self.max_output_tokens
        )
        if count_tokens(prompt) > max_allowed:
            logger.warning("Rigor discovery prompt too large; retrying without RAG evidence")
            prompt = base_prompt

        data = await self._call_llm_and_parse(
            prompt=prompt,
            task_label="rigor_discovery",
        )
        if data is None:
            return None
        if isinstance(data, list):
            data = data[0] if data else {}
        if not isinstance(data, dict):
            return None
        if not data.get("needs_theorem_work", False):
            return None
        return data

    # --------------------------------------------------------- stage 2

    async def _step_formalize(
        self,
        candidate: ProofCandidate,
        theorem_statement: str,
    ) -> Optional[tuple]:
        """Run up to 5 Lean 4 attempts with feedback chaining.

        Returns (theorem_name, lean_code, attempts) on success, None on
        all-5-fail. On failure, records the candidate in proof_database so
        future rigor cycles can see it as an open lemma target.
        """
        current_paper_raw = await paper_memory.get_paper()
        current_paper = _strip_paper_markers_for_llm(current_paper_raw)

        # Imported lazily to avoid a circular-import chain through the
        # autonomous agents package at module load time.
        from backend.autonomous.agents.proof_formalization_agent import (
            ProofFormalizationAgent,
        )

        formalizer = ProofFormalizationAgent(
            model_id=self.model_name,
            context_window=self.context_window,
            max_output_tokens=self.max_output_tokens,
            role_id="compiler_rigor_formalization",
        )

        async def _on_attempt_started(attempt_number: int, strategy: str) -> None:
            await self._broadcast(
                "proof_attempt_started",
                {
                    "source_type": "compiler_rigor",
                    "source_id": self._compiler_source_id(),
                    "theorem_id": candidate.theorem_id,
                    "theorem_statement": theorem_statement,
                    "attempt": attempt_number,
                    "strategy": strategy,
                },
            )

        async def _on_attempt_feedback(feedback: ProofAttemptFeedback) -> None:
            event = "proof_verified" if feedback.success else "proof_attempt_failed"
            await self._broadcast(
                event,
                {
                    "source_type": "compiler_rigor",
                    "source_id": self._compiler_source_id(),
                    "theorem_id": candidate.theorem_id,
                    "theorem_statement": theorem_statement,
                    "attempt": feedback.attempt,
                    "strategy": feedback.strategy,
                    "error_output": feedback.error_output[:500] if feedback.error_output else "",
                },
            )

        await self._broadcast(
            "proof_check_started",
            {
                "source_type": "compiler_rigor",
                "source_id": self._compiler_source_id(),
                "trigger": "rigor_loop",
            },
        )

        try:
            success, theorem_name, lean_code, attempts = await formalizer.prove_candidate(
                user_research_prompt=self.raw_user_prompt,
                source_type="paper",  # ProofCandidate expects "paper" | "brainstorm"
                theorem_candidate=candidate,
                source_content=current_paper,
                max_attempts=5,
                attempt_callback=_on_attempt_feedback,
                attempt_start_callback=_on_attempt_started,
            )
        except Exception as exc:
            logger.error("Rigor formalization raised (%s); declining cycle", exc, exc_info=True)
            await self._broadcast(
                "proof_check_complete",
                {
                    "source_type": "compiler_rigor",
                    "source_id": self._compiler_source_id(),
                    "verified_count": 0,
                    "message": f"formalization error: {exc}",
                },
            )
            return None

        if not success:
            # Record as an open lemma target so the next rigor cycle's
            # discovery step can optionally retry it.
            try:
                error_summary = attempts[-1].error_output if attempts else ""
                await proof_database.record_failed_candidate(
                    source_brainstorm_id=self._compiler_source_id(),
                    theorem_candidate=candidate,
                    error_summary=error_summary[:2000] if error_summary else "No Lean diagnostics captured.",
                )
            except Exception as exc:
                logger.debug("record_failed_candidate failed: %s", exc)

            await self._broadcast(
                "proof_check_complete",
                {
                    "source_type": "compiler_rigor",
                    "source_id": self._compiler_source_id(),
                    "verified_count": 0,
                    "message": "5 Lean 4 attempts failed",
                },
            )
            return None

        return theorem_name, lean_code, attempts

    # --------------------------------------------------------- stage 3

    async def _step_assess_novelty_and_store(
        self,
        *,
        theorem_statement: str,
        theorem_name: str,
        lean_code: str,
        formal_sketch: str,
        attempts: List[ProofAttemptFeedback],
    ) -> tuple:
        """Classify the verified proof and persist it via proof_database.

        Returns (is_novel, novelty_reasoning, stored_record).
        """
        # Lazy import to break an early-load circular chain through the
        # autonomous.core package __init__.
        from backend.autonomous.core.proof_novelty import assess_proof_novelty

        existing_block = proof_database.get_novel_proofs_for_injection()

        task_id = f"{self.get_current_task_id()}_novelty"
        self.task_sequence += 1

        try:
            is_novel, novelty_reasoning = await assess_proof_novelty(
                user_prompt=self.raw_user_prompt,
                theorem_statement=theorem_statement,
                lean_code=lean_code,
                validator_model=self.model_name,
                validator_context=self.context_window,
                validator_max_tokens=self.max_output_tokens,
                existing_novel_proofs=existing_block,
                task_id=task_id,
                role_id="compiler_rigor_novelty",
            )
        except Exception as exc:
            logger.warning("Novelty assessment failed (%s); defaulting to non-novel", exc)
            is_novel, novelty_reasoning = False, f"Novelty assessment error: {exc}"

        record = ProofRecord(
            proof_id="",  # proof_database assigns proof_XXX on add_proof
            theorem_id="",
            theorem_statement=theorem_statement,
            theorem_name=theorem_name,
            formal_sketch=formal_sketch,
            source_type="paper",  # compiler rigor proofs live under the "paper" channel
            source_id=self._compiler_source_id(),
            source_title="Compiler Rigor Theorem",
            solver="Lean 4",
            lean_code=lean_code,
            novel=is_novel,
            novelty_reasoning=novelty_reasoning,
            verification_notes="Produced by compiler rigor loop (HighParamSubmitter).",
            attempt_count=len(attempts),
            attempts=list(attempts),
            dependencies=[],
            solver_hints=[],
        )

        stored = await proof_database.add_proof(record)
        return is_novel, novelty_reasoning, stored

    # --------------------------------------------------------- stage 4

    async def _step_initial_placement(
        self,
        *,
        proof_id: str,
        theorem_statement: str,
        theorem_name: str,
        lean_code: str,
        is_novel: bool,
    ) -> Optional[CompilerSubmission]:
        """Produce the attempt-1 placement submission.

        Returns None when the submitter refuses a legal placement on attempt 1.
        The coordinator treats a None attempt-1 submission the same way it
        treats a double rejection: appendix fallback + acceptance counter.
        """
        return await self._build_placement_submission(
            proof_id=proof_id,
            theorem_statement=theorem_statement,
            theorem_name=theorem_name,
            lean_code=lean_code,
            is_novel=is_novel,
            placement_attempt=1,
            validator_rejection_feedback="",
        )

    async def submit_rigor_placement_retry(
        self,
        prior: RigorTheoremResult,
        validator_feedback: str,
    ) -> Optional[CompilerSubmission]:
        """Produce the attempt-2 placement submission, with validator feedback."""
        return await self._build_placement_submission(
            proof_id=prior.proof_id,
            theorem_statement=prior.theorem_statement,
            theorem_name=prior.theorem_name,
            lean_code=prior.lean_code,
            is_novel=prior.is_novel,
            placement_attempt=2,
            validator_rejection_feedback=validator_feedback or "",
        )

    async def _build_placement_submission(
        self,
        *,
        proof_id: str,
        theorem_statement: str,
        theorem_name: str,
        lean_code: str,
        is_novel: bool,
        placement_attempt: int,
        validator_rejection_feedback: str,
    ) -> Optional[CompilerSubmission]:
        current_outline = await outline_memory.get_outline()
        current_paper_raw = await paper_memory.get_paper()
        current_paper = _strip_paper_markers_for_llm(current_paper_raw)

        base_prompt = await build_rigor_placement_prompt(
            user_prompt=self.user_prompt,
            current_outline=current_outline,
            current_paper=current_paper,
            rag_evidence="",
            theorem_statement=theorem_statement,
            lean_code=lean_code,
            proof_id=proof_id,
            placement_attempt=placement_attempt,
            validator_rejection_feedback=validator_rejection_feedback,
        )
        mandatory_tokens = count_tokens(base_prompt)
        query_seed = (theorem_statement + " " + current_paper[-1500:]).strip()
        rag_evidence = await self._build_rigor_rag_context(
            query_seed=query_seed,
            reserved_tokens=mandatory_tokens,
        )

        prompt = await build_rigor_placement_prompt(
            user_prompt=self.user_prompt,
            current_outline=current_outline,
            current_paper=current_paper,
            rag_evidence=rag_evidence,
            theorem_statement=theorem_statement,
            lean_code=lean_code,
            proof_id=proof_id,
            placement_attempt=placement_attempt,
            validator_rejection_feedback=validator_rejection_feedback,
        )

        max_allowed = rag_config.get_available_input_tokens(
            self.context_window, self.max_output_tokens
        )
        if count_tokens(prompt) > max_allowed:
            logger.warning("Rigor placement prompt too large; retrying without RAG evidence")
            prompt = base_prompt

        data = await self._call_llm_and_parse(
            prompt=prompt,
            task_label=f"rigor_placement_{placement_attempt}",
        )
        if data is None:
            return None
        if isinstance(data, list):
            data = data[0] if data else {}
        if not isinstance(data, dict):
            return None
        if not data.get("proceed", True):
            logger.info(
                "Rigor placement attempt %s: submitter refused a legal placement",
                placement_attempt,
            )
            return None

        new_string = _normalize_string_field(data.get("new_string", ""))
        old_string = _normalize_string_field(data.get("old_string", ""))
        if not new_string or not old_string:
            logger.info(
                "Rigor placement attempt %s: missing old_string or new_string",
                placement_attempt,
            )
            return None

        operation = data.get("operation", "insert_after")
        if operation not in ("replace", "insert_after"):
            operation = "insert_after"

        submission = CompilerSubmission(
            submission_id=str(uuid.uuid4()),
            mode="rigor",
            content=new_string,
            operation=operation,
            old_string=old_string,
            new_string=new_string,
            reasoning=str(data.get("reasoning", "")),
            metadata={
                "rigor_mode": "lean_placement",
                "lean_proof_id": proof_id,
                "lean_code": lean_code,
                "theorem_statement": theorem_statement,
                "theorem_name": theorem_name,
                "is_novel": is_novel,
                "placement_attempt": placement_attempt,
                "validator_rejection_feedback": validator_rejection_feedback,
            },
        )
        return submission

    # -------------------------------------------------------- llm helper

    async def _call_llm_and_parse(
        self,
        *,
        prompt: str,
        task_label: str,
    ) -> Optional[Any]:
        """Send `prompt` to the high-param model and return parsed JSON.

        On a JSON parse failure, issues a single conversational retry that
        feeds the failed output back with a JSON-escape-rules reminder.
        """
        task_id = self.get_current_task_id()
        self.task_sequence += 1

        # LM Studio cache warmup (silent no-op for OpenRouter)
        try:
            await lm_studio_client.cache_model_load_config(
                self.model_name,
                {"context_length": self.context_window, "model_path": self.model_name},
            )
        except Exception:
            pass

        if self.task_tracking_callback:
            self.task_tracking_callback("started", task_id)

        try:
            response = await api_client_manager.generate_completion(
                task_id=task_id,
                role_id=self.role_id,
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=self.max_output_tokens,
            )
        except Exception as exc:
            logger.error("High-param LLM call failed (%s): %s", task_label, exc)
            if self.task_tracking_callback:
                self.task_tracking_callback("completed", task_id)
            return None

        if not response or not response.get("choices") or not response["choices"][0].get("message"):
            logger.error("High-param LLM returned empty response (%s)", task_label)
            if self.task_tracking_callback:
                self.task_tracking_callback("completed", task_id)
            return None

        message = response["choices"][0]["message"]
        llm_output = message.get("content") or message.get("reasoning") or ""
        if not llm_output.strip():
            logger.error("High-param LLM returned empty content (%s)", task_label)
            if self.task_tracking_callback:
                self.task_tracking_callback("completed", task_id)
            return None

        try:
            parsed = parse_json(llm_output)
            if self.task_tracking_callback:
                self.task_tracking_callback("completed", task_id)
            return parsed
        except Exception as parse_error:
            logger.info(
                "High-param submitter (%s): initial JSON parse failed, attempting one retry: %s",
                task_label,
                parse_error,
            )

        # Single conversational retry with a JSON-escape reminder
        retry_prompt = (
            "Your previous response could not be parsed as valid JSON.\n\n"
            f"PARSE ERROR: {parse_error}\n\n"
            "JSON ESCAPING RULES FOR LaTeX:\n"
            "1. Every backslash in content needs ONE extra escape in JSON "
            "(write \\\\mathbb{Z} not \\mathbb{Z}).\n"
            "2. Escape double quotes inside strings as \\\".\n"
            "3. Newlines: \\n (not \\\\n).\n"
            "4. Do not include any system-managed bracket markers.\n\n"
            "Please respond again with ONLY the JSON object, no markdown."
        )

        try:
            truncated_preview = llm_output[:2000] + (
                "\n[...truncated...]" if len(llm_output) > 2000 else ""
            )
            retry_response = await api_client_manager.generate_completion(
                task_id=f"{task_id}_retry",
                role_id=self.role_id,
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": truncated_preview},
                    {"role": "user", "content": retry_prompt},
                ],
                temperature=0.0,
                max_tokens=self.max_output_tokens,
            )
            if retry_response and retry_response.get("choices"):
                retry_msg = retry_response["choices"][0]["message"]
                retry_output = retry_msg.get("content") or retry_msg.get("reasoning") or ""
                parsed = parse_json(retry_output)
                logger.info("High-param submitter (%s): retry succeeded", task_label)
                if self.task_tracking_callback:
                    self.task_tracking_callback("completed", task_id)
                return parsed
        except Exception as retry_error:
            logger.warning(
                "High-param submitter (%s): retry failed: %s", task_label, retry_error
            )

        if self.task_tracking_callback:
            self.task_tracking_callback("completed", task_id)
        return None







