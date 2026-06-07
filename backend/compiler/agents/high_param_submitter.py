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
    Stage 4 (placement): either propose an inline edit that introduces the
        theorem with a "verified in Lean 4" marker and an appendix
        reference, or explicitly request appendix-only storage for extension
        theorems. The coordinator owns the 2-attempt validator retry loop
        and appendix insertion.

The Wolfram sub-mode that used to live here has been removed in Phase 2.
Wolfram Alpha is now a tool available to HighContextSubmitter.submit_construction
(see Phase 3 of the build plan).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from backend.autonomous.memory.paper_library import PaperLibrary
from backend.autonomous.memory.proof_database import proof_database as autonomous_proof_database
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
from backend.shared.json_parser import parse_json, sanitize_model_output_for_retry_context
from backend.shared.response_extraction import extract_message_text
from backend.shared.lean_proof_integrity import validate_full_lean_proof_integrity
from backend.shared.lm_studio_client import lm_studio_client
from backend.shared.models import (
    CompilerSubmission,
    ProofAttemptFeedback,
    ProofCandidate,
)
from backend.shared.utils import count_tokens

logger = logging.getLogger(__name__)

_NOVEL_PROOF_TIERS = {
    "major_mathematical_discovery",
    "mathematical_discovery",
    "novel_variant",
    "novel_formulation",
}


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


def _strip_generated_proofs_for_rigor_context(paper_content: str) -> str:
    """Remove generated theorem/proof appendix entries from proof-discovery context."""
    return PaperLibrary.strip_verified_proofs_from_content(paper_content or "")


def format_theorem_appendix_entry(
    *,
    proof_id: str,
    theorem_statement: str,
    lean_code: str,
    is_novel: bool,
    theorem_name: str = "",
    novelty_tier: str = "",
    placement_outcome: str = "appendix_fallback",
) -> str:
    """Format a verified-theorem entry for the Theorems Appendix.

    Used both when placement is inline (a short cross-reference stub) and
    when placement fails and the full entry is the only record (appendix
    fallback). Caller selects via `placement_outcome`.
    """
    header_name = theorem_name.strip() or proof_id
    tier_labels = {
        "major_mathematical_discovery": "Major Mathematical Discovery",
        "mathematical_discovery": "Mathematical Discovery",
        "novel_variant": "Novel Reformulation",
        "novel_formulation": "Novel Formalization",
    }
    novelty_label = tier_labels.get(novelty_tier, "Novel" if is_novel else "Known")
    status_suffix = {
        "appendix_fallback": "inline placement rejected; preserved here because Lean 4 verified the math",
        "appendix_requested": "stored here by rigor discovery request",
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
    novelty_tier: str
    novelty_reasoning: str
    attempts: List[ProofAttemptFeedback]
    source_id: str
    initial_placement_submission: Optional[CompilerSubmission] = None
    # Retained for retry-prompt assembly
    formal_sketch: str = ""
    source_excerpt: str = ""
    theorem_origin: str = "existing_paper_claim"
    placement_preference: str = "inline"
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
        *,
        validator_model: str = "",
        validator_context_window: Optional[int] = None,
        validator_max_tokens: Optional[int] = None,
        proof_database_store=None,
    ):
        self.model_name = model_name
        self.proof_database = proof_database_store or autonomous_proof_database
        # Rigor discovery receives compact existing-proof summaries separately.
        # Avoid injecting full Lean proof library text into the user prompt,
        # which would duplicate the paper appendix/proof-list context.
        self.user_prompt = user_prompt
        self.raw_user_prompt = user_prompt
        self.websocket_broadcaster = websocket_broadcaster
        self.validator_model = validator_model or model_name
        self.validator_context_window = (
            validator_context_window
            if validator_context_window is not None
            else system_config.compiler_validator_context_window
        )
        self.validator_max_tokens = (
            validator_max_tokens
            if validator_max_tokens is not None
            else system_config.compiler_validator_max_output_tokens
        )
        self._initialized = False
        self._standalone_session_id = f"standalone_{uuid.uuid4().hex[:12]}"
        self._source_material_context: str = ""
        self._source_material_label: str = ""
        self._rigor_proof_source_id: str = ""
        self._rigor_proof_source_title: str = ""

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

    def set_source_material_context(self, content: str, label: str = "") -> None:
        """Set direct paper-source context used by rigor theorem discovery."""
        self._source_material_context = (content or "").strip()
        self._source_material_label = (label or "").strip()

    def set_rigor_proof_source(self, source_id: str = "", source_title: str = "") -> None:
        """Set the real paper source for rigor-created proof records."""
        self._rigor_proof_source_id = (source_id or "").strip()
        self._rigor_proof_source_title = (source_title or "").strip()

    def _get_direct_source_material_context(self, max_chars: int = 50000) -> str:
        """Return bounded direct source context; full content remains available via RAG."""
        context = self._source_material_context.strip()
        if not context:
            return ""
        if len(context) <= max_chars:
            return context
        head = max_chars // 2
        tail = max_chars - head
        return (
            context[:head].rstrip()
            + "\n\n[... direct source context truncated; full source remains available through RAG ...]\n\n"
            + context[-tail:].lstrip()
        )

    def _get_paper_proof_source_content(self, current_paper: str) -> str:
        """Combine current paper with direct source material for formal proof attempts."""
        proof_paper = _strip_generated_proofs_for_rigor_context(current_paper)
        parts = [
            "CURRENT PAPER UNDER CONSTRUCTION:\n" + (proof_paper or "").strip(),
        ]
        source_context = self._get_direct_source_material_context(max_chars=30000)
        if source_context:
            label = self._source_material_label or "Source brainstorm / paper-writing database"
            parts.append(f"{label.upper()}:\n{source_context}")
        return "\n\n---\n\n".join(part for part in parts if part.strip())

    async def initialize(self) -> None:
        if self._initialized:
            return

        self.context_window = system_config.compiler_high_param_context_window
        self.max_output_tokens = system_config.compiler_high_param_max_output_tokens
        if int(self.validator_context_window or 0) <= 0 or int(self.validator_max_tokens or 0) <= 0:
            raise ValueError("High-param validator context and max output settings must be configured.")
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

        When the autonomous session manager is active, the active proof database is
        already storing in the session directory. Otherwise each manual
        compiler instance gets its own id so failed theorem candidates do not
        bleed into later standalone compiler runs.
        """
        sm = getattr(self.proof_database, "_session_manager", None)
        if sm is not None and getattr(sm, "is_session_active", False):
            return str(getattr(sm, "session_id", "") or "autonomous_active")
        return self._standalone_session_id

    def _compiler_source_id(self) -> str:
        """Source id used on ProofRecord / failed candidate storage.

        Prefer the actual paper id supplied by the compiler coordinator. The
        manual fallback stays filename-safe because failed-candidate storage
        also keys off this id.
        """
        return self._rigor_proof_source_id or f"manual_compiler_{self._resolve_session_id()}"

    def _compiler_source_title(self) -> str:
        """Human-readable source title for rigor-created proof records."""
        return self._rigor_proof_source_title or "Compiler Rigor Theorem"

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
        remaining = max_allowed - reserved_tokens
        if remaining <= 0:
            logger.warning(
                "Skipping rigor RAG retrieval because mandatory direct context uses the configured input budget "
                f"(reserved={reserved_tokens}, max_input={max_allowed})."
            )
            return ""

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
        theorem_origin = str(discovery.get("theorem_origin") or "").strip()
        placement_preference = str(discovery.get("placement_preference") or "").strip()
        expected_novelty_tier = str(discovery.get("expected_novelty_tier") or "").strip().lower()
        prompt_relevance_rationale = str(discovery.get("prompt_relevance_rationale") or "").strip()
        novelty_rationale = str(discovery.get("novelty_rationale") or "").strip()
        why_not_standard_known_result = str(
            discovery.get("why_not_standard_known_result") or ""
        ).strip()

        if not theorem_statement:
            logger.info("Rigor cycle: discovery returned empty theorem_statement; declining")
            return None
        if expected_novelty_tier == "not_novel":
            logger.info("Rigor cycle: discovery marked theorem not_novel; declining before Lean cost")
            return None
        if expected_novelty_tier not in _NOVEL_PROOF_TIERS:
            logger.info(
                "Rigor cycle: discovery omitted a valid expected_novelty_tier; declining before Lean cost"
            )
            return None
        if not (
            prompt_relevance_rationale
            and novelty_rationale
            and why_not_standard_known_result
        ):
            logger.info(
                "Rigor cycle: discovery omitted required novelty/relevance rationales; declining before Lean cost"
            )
            return None

        if theorem_origin not in {
            "existing_paper_claim",
            "extension_from_partial_work",
            "extension_from_user_prompt",
        }:
            theorem_origin = "existing_paper_claim"

        if placement_preference not in {"inline", "appendix_only"}:
            placement_preference = "inline"

        if theorem_origin in {
            "extension_from_partial_work",
            "extension_from_user_prompt",
        }:
            # Extension proofs are useful evidence for the paper, but they
            # should not silently mutate the main body narrative.
            placement_preference = "appendix_only"

        logger.info(
            "Rigor cycle: Stage 2 - Lean 4 formalization (up to 5 attempts), "
            f"retry_failure_id={retry_failure_id or 'none'}"
        )

        candidate = ProofCandidate(
            theorem_id=retry_failure_id or f"compiler_rigor_{uuid.uuid4().hex[:12]}",
            statement=theorem_statement,
            formal_sketch=formal_sketch,
            expected_novelty_tier=expected_novelty_tier,
            prompt_relevance_rationale=prompt_relevance_rationale,
            novelty_rationale=novelty_rationale,
            why_not_standard_known_result=why_not_standard_known_result,
            source_excerpt=source_excerpt,
            origin_source_id=self._compiler_source_id() if retry_failure_id else "",
        )

        formalizer_result = await self._step_formalize(candidate, theorem_statement)
        if formalizer_result is None:
            return None

        theorem_name, lean_code, attempts, integrity = formalizer_result

        stored_theorem_statement = (
            integrity.actual_theorem_statement.strip()
            or theorem_statement
        )
        stored_theorem_name = (
            integrity.actual_theorem_name.strip()
            or theorem_name
        )
        stored_formal_sketch = formal_sketch
        verification_notes = "Produced by compiler rigor loop (HighParamSubmitter)."
        if integrity.category in {"statement_downshifted", "statement_alignment_uncertain", "statement_alignment_unavailable"}:
            stored_formal_sketch = (
                f"{stored_formal_sketch}\n\n"
                f"Original intended theorem candidate: {theorem_statement}\n"
                f"Statement-alignment classification: {integrity.category}. "
                f"{integrity.reason or integrity.downshift_reason}"
            ).strip()
            verification_notes = (
                "Produced by compiler rigor loop (HighParamSubmitter). "
                "Lean accepted the proof; MOTO preserved it under the actual "
                "Lean-verified statement instead of discarding it for candidate mismatch."
            )
            await self._broadcast(
                "proof_downshifted",
                {
                    "source_type": "compiler_rigor",
                    "source_id": self._compiler_source_id(),
                    "theorem_id": candidate.theorem_id,
                    "intended_theorem_statement": theorem_statement,
                    "theorem_statement": stored_theorem_statement,
                    "category": integrity.category,
                    "reason": integrity.reason or integrity.downshift_reason,
                },
            )

        logger.info("Rigor cycle: Stage 3 - novelty classification + persistence")
        novelty_result = await self._step_assess_novelty_and_store(
            theorem_statement=stored_theorem_statement,
            theorem_name=stored_theorem_name,
            lean_code=lean_code,
            formal_sketch=stored_formal_sketch,
            attempts=attempts,
            verification_notes=verification_notes,
        )
        if novelty_result is None:
            return None
        is_novel, novelty_reasoning, stored_record, duplicate = novelty_result

        await self._broadcast(
            "proof_verified",
            {
                "source_type": "compiler_rigor",
                "source_id": self._compiler_source_id(),
                "theorem_id": candidate.theorem_id,
                "theorem_statement": stored_theorem_statement,
                "intended_theorem_statement": theorem_statement,
                "proof_id": stored_record.proof_id,
                "is_novel": is_novel,
                "novelty_tier": stored_record.novelty_tier,
                "novelty_reasoning": novelty_reasoning,
            },
        )
        await self._broadcast(
            "proof_check_complete",
            {
                "source_type": "compiler_rigor",
                "source_id": self._compiler_source_id(),
                "source_title": self._compiler_source_title(),
                "trigger": "rigor_loop",
                "verified_count": 1,
                "novel_count": 1 if is_novel and not duplicate else 0,
                "total_candidates": 1,
                "proof_id": stored_record.proof_id,
                "theorem_id": candidate.theorem_id,
                "theorem_statement": stored_theorem_statement,
                "is_novel": is_novel,
                "novelty_tier": stored_record.novelty_tier,
                "novelty_reasoning": novelty_reasoning,
                "duplicate": duplicate,
                "message": "Compiler rigor proof verified, ranked, and indexed.",
            },
        )

        # If we retried a previously-failed candidate and it succeeded, mark it
        # resolved so it stops appearing in future failure-hint lists.
        if retry_failure_id:
            try:
                await self.proof_database.mark_resolved_retry(
                    source_brainstorm_id=self._compiler_source_id(),
                    theorem_id=retry_failure_id,
                    proof_id=stored_record.proof_id,
                )
            except Exception as exc:
                logger.debug("mark_resolved_retry failed (non-fatal): %s", exc)

        initial_submission = None
        if placement_preference == "appendix_only":
            logger.info(
                "Rigor cycle: discovery requested appendix-only placement "
                "(origin=%s)",
                theorem_origin,
            )
        else:
            logger.info("Rigor cycle: Stage 4 - initial placement proposal")
            initial_submission = await self._step_initial_placement(
                proof_id=stored_record.proof_id,
                theorem_statement=stored_theorem_statement,
                theorem_name=stored_theorem_name,
                lean_code=lean_code,
                is_novel=is_novel,
            )

        return RigorTheoremResult(
            proof_id=stored_record.proof_id,
            theorem_statement=stored_theorem_statement,
            theorem_name=stored_theorem_name,
            lean_code=lean_code,
            is_novel=is_novel,
            novelty_tier=stored_record.novelty_tier,
            novelty_reasoning=novelty_reasoning,
            attempts=attempts,
            source_id=self._compiler_source_id(),
            initial_placement_submission=initial_submission,
            formal_sketch=stored_formal_sketch,
            source_excerpt=source_excerpt,
            theorem_origin=theorem_origin,
            placement_preference=placement_preference,
            metadata={
                "retry_failure_id": retry_failure_id,
                "attempt_count": len(attempts),
                "theorem_origin": theorem_origin,
                "placement_preference": placement_preference,
                "intended_theorem_statement": theorem_statement,
                "statement_alignment_category": integrity.category,
                "duplicate": duplicate,
            },
        )

    # --------------------------------------------------------- stage 1

    async def _step_discovery(self) -> Optional[dict]:
        """Ask the LLM whether a Lean 4 theorem is worth pursuing right now."""
        current_outline = await outline_memory.get_outline()
        current_paper_raw = await paper_memory.get_paper()
        current_paper = _strip_generated_proofs_for_rigor_context(
            _strip_paper_markers_for_llm(current_paper_raw)
        )

        # Existing verified proofs - compact blob of statements so the model
        # can recognize duplicates without blowing the token budget.
        existing_proofs: List[dict] = []
        try:
            for record in await self.proof_database.get_all_proofs():
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
            failure_hints = await self.proof_database.get_recent_failure_hints(
                self._compiler_source_id(), limit=5
            )
        except Exception as exc:
            logger.debug("proof_database.get_recent_failure_hints failed: %s", exc)
            failure_hints = []

        source_material_context = self._get_direct_source_material_context()
        max_allowed = rag_config.get_available_input_tokens(
            self.context_window, self.max_output_tokens
        )

        # Build with empty RAG first to measure the mandatory footprint,
        # then allocate the rest to RAG. If the direct source context itself
        # is too large, shrink it before falling back to RAG.
        while True:
            base_prompt = await build_rigor_theorem_discovery_prompt(
                user_prompt=self.user_prompt,
                current_outline=current_outline,
                current_paper=current_paper,
                rag_evidence="",
                existing_verified_proofs=existing_proofs,
                recent_failure_hints=failure_hints,
                source_material_context=source_material_context,
                source_material_label=self._source_material_label,
            )
            if count_tokens(base_prompt) <= max_allowed or len(source_material_context) <= 4000:
                break
            source_material_context = source_material_context[: max(len(source_material_context) // 2, 4000)]

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
            source_material_context=source_material_context,
            source_material_label=self._source_material_label,
        )

        if count_tokens(prompt) > max_allowed:
            logger.warning("Rigor discovery prompt too large; retrying without RAG evidence")
            prompt = base_prompt
        prompt_tokens = count_tokens(prompt)
        if prompt_tokens > max_allowed:
            raise ValueError(
                "Rigor discovery prompt exceeds available input budget "
                f"({prompt_tokens} tokens > {max_allowed} tokens) even without RAG evidence."
            )

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

        Returns (theorem_name, lean_code, attempts, integrity) on success, None on
        all-5-fail. On failure, records the candidate in proof_database so
        future rigor cycles can see it as an open lemma target.
        """
        current_paper_raw = await paper_memory.get_paper()
        current_paper = _strip_paper_markers_for_llm(current_paper_raw)
        proof_source_content = self._get_paper_proof_source_content(current_paper)

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
        proof_label = "A"

        def _lean_response_summary(feedback: ProofAttemptFeedback) -> str:
            if feedback.success:
                return "Lean 4 response: proof verified."
            error = " ".join((feedback.error_output or "").split())
            if len(error) > 960:
                error = f"{error[:960]}..."
            if error:
                return f"Lean 4 response: {error} - proof not verified."
            return "Lean 4 response: proof not verified."

        async def _on_attempt_started(attempt_number: int, strategy: str) -> None:
            await self._broadcast(
                "proof_attempt_started",
                {
                    "source_type": "compiler_rigor",
                    "source_id": self._compiler_source_id(),
                    "theorem_id": candidate.theorem_id,
                    "theorem_statement": theorem_statement,
                    "proof_label": proof_label,
                    "attempt": attempt_number,
                    "strategy": strategy,
                },
            )

        async def _on_attempt_feedback(feedback: ProofAttemptFeedback) -> None:
            event = "proof_lean_accepted" if feedback.success else "proof_attempt_failed"
            await self._broadcast(
                event,
                {
                    "source_type": "compiler_rigor",
                    "source_id": self._compiler_source_id(),
                    "theorem_id": candidate.theorem_id,
                    "theorem_statement": theorem_statement,
                    "proof_label": proof_label,
                    "attempt": feedback.attempt,
                    "strategy": feedback.strategy,
                    "error_output": feedback.error_output[:500] if feedback.error_output else "",
                    "lean_response": _lean_response_summary(feedback),
                    "proof_verified": feedback.success,
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
                source_content=proof_source_content,
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
                await self.proof_database.record_failed_candidate(
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

        integrity = await validate_full_lean_proof_integrity(
            user_prompt=self.raw_user_prompt,
            theorem_statement=theorem_statement,
            formal_sketch=candidate.formal_sketch,
            lean_code=lean_code,
            source_excerpt=candidate.source_excerpt or proof_source_content,
            allowed_baseline="",
            validator_model=self.validator_model,
            validator_context=self.validator_context_window,
            validator_max_tokens=self.validator_max_tokens,
            task_id=f"{self.get_current_task_id()}_integrity",
            role_id="compiler_rigor_novelty",
            require_statement_alignment=True,
        )
        if not integrity.valid:
            try:
                await self.proof_database.record_failed_candidate(
                    source_brainstorm_id=self._compiler_source_id(),
                    theorem_candidate=candidate,
                    error_summary=integrity.reason[:2000],
                )
            except Exception as exc:
                logger.debug("record_failed_candidate failed after integrity rejection: %s", exc)
            await self._broadcast(
                "proof_integrity_rejected",
                {
                    "source_type": "compiler_rigor",
                    "source_id": self._compiler_source_id(),
                    "theorem_id": candidate.theorem_id,
                    "theorem_statement": theorem_statement,
                    "category": integrity.category,
                    "reason": integrity.reason,
                },
            )
            await self._broadcast(
                "proof_check_complete",
                {
                    "source_type": "compiler_rigor",
                    "source_id": self._compiler_source_id(),
                    "verified_count": 0,
                    "message": "Lean proof failed post-verification integrity checks",
                },
            )
            return None

        return theorem_name, lean_code, attempts, integrity

    # --------------------------------------------------------- stage 3

    async def _step_assess_novelty_and_store(
        self,
        *,
        theorem_statement: str,
        theorem_name: str,
        lean_code: str,
        formal_sketch: str,
        attempts: List[ProofAttemptFeedback],
        verification_notes: str,
    ) -> Optional[tuple]:
        """Classify the verified proof and persist it via proof_database.

        Returns (is_novel, novelty_reasoning, stored_record, duplicate).
        """
        task_id = f"{self.get_current_task_id()}_novelty"
        self.task_sequence += 1

        try:
            # Lazy import avoids an early-load cycle through autonomous.core.
            from backend.autonomous.core.proof_registration import register_verified_lean_proof

            registration = await register_verified_lean_proof(
                proof_database=self.proof_database,
                user_prompt=self.raw_user_prompt,
                theorem_statement=theorem_statement,
                lean_code=lean_code,
                validator_model=self.validator_model,
                validator_context=self.validator_context_window,
                validator_max_tokens=self.validator_max_tokens,
                task_id=task_id,
                role_id="compiler_rigor_novelty",
                source_type="paper",
                source_id=self._compiler_source_id(),
                source_title=self._compiler_source_title(),
                theorem_name=theorem_name,
                formal_sketch=formal_sketch,
                solver="Lean 4",
                verification_notes=verification_notes,
                attempt_count=len(attempts),
                attempts=list(attempts),
                broadcast_fn=self.websocket_broadcaster,
                base_event={
                    "source_type": "compiler_rigor",
                    "source_id": self._compiler_source_id(),
                    "source_title": self._compiler_source_title(),
                    "trigger": "rigor_loop",
                },
            )
            stored = registration.record
            return stored.novel, stored.novelty_reasoning, stored, registration.duplicate
        except Exception as exc:
            logger.warning("Novelty assessment failed; rigor proof will not be stored: %s", exc)
            await self._broadcast(
                "proof_check_complete",
                {
                    "source_type": "compiler_rigor",
                    "source_id": self._compiler_source_id(),
                    "verified_count": 0,
                    "message": f"novelty validation failed: {exc}",
                },
            )
            return None

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
        except Exception as exc:
            logger.debug("LM Studio cache warmup skipped for high-param submitter: %s", exc)

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
        llm_output = extract_message_text(message)
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
            truncated_preview = sanitize_model_output_for_retry_context(llm_output, max_chars=2000)
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
                retry_output = extract_message_text(retry_msg)
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







