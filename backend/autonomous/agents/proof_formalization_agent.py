"""
Lean 4 formalization agent with iterative retry loop.
"""
from __future__ import annotations

import json
import hashlib
import logging
from typing import Any, Awaitable, Callable, List, Optional, Tuple

from backend.shared.api_client_manager import RetryableProviderError, api_client_manager
from backend.shared.json_parser import parse_json
from backend.shared.response_extraction import extract_message_text
from backend.shared.lean4_client import get_lean4_client
from backend.shared.model_error_utils import (
    format_transient_provider_error,
    is_non_retryable_model_error,
    is_provider_context_length_error,
    is_retryable_model_output_error,
    is_transient_model_call_error,
)
from backend.shared.models import ProofAttemptFeedback, ProofCandidate, SmtHint
from backend.shared.openrouter_client import FreeModelExhaustedError
from backend.shared.provider_errors import ProviderContextLengthError
from backend.shared.proof_search.tool_adapter import execute_search_lean_proofs
from backend.shared.proof_search.assistant_coordinator import assistant_proof_search_coordinator
from backend.shared.utils import count_tokens
from backend.shared.config import rag_config, system_config
from backend.autonomous.prompts.proof_prompts import (
    build_proof_formalization_prompt,
    build_proof_tactic_script_prompt,
)

logger = logging.getLogger(__name__)


AttemptCallback = Callable[[ProofAttemptFeedback], Awaitable[None]]
AttemptStartCallback = Callable[[int, str], Awaitable[None]]
ShouldStopFn = Optional[Callable[[], bool]]

_JSON_PARSE_ERROR_MARKERS = (
    "empty or whitespace-only response",
    "empty response from formalization model",
    "empty response from tactic formalization model",
    "expecting property name",
    "expecting value",
    "extra data",
    "invalid control character",
    "json response truncated",
    "no content in formalization model response",
    "no content in tactic formalization model response",
    "no json found",
    "openrouter connection failed",
    "codex connection failed",
    "openrouter response missing 'choices'",
    "openrouter returned non-json body",
    "response too short",
    "unterminated string",
    "upstream provider timeout",
)
_MALFORMED_MODEL_OUTPUT_REASON = "Model returned malformed output (not valid JSON); retrying with clean context."
_INCOMPLETE_MODEL_OUTPUT_REASON = (
    "Model/provider output reached its maximum output length and was truncated before returning usable Lean proof code."
)
_LEAN_WORKSPACE_ERROR_PREFIX = "LEAN 4 WORKSPACE ERROR"
_MANDATORY_FULL_SOURCE_CONTEXT_OVERFLOW_PREFIX = "MANDATORY FULL SOURCE CONTEXT OVERFLOW"
_PROOF_SEARCH_CONTEXT_OMITTED = (
    "[Proof-search context omitted because it was unavailable or did not fit the configured context budget.]"
)
_TRUNCATED_FINISH_REASONS = {
    "incomplete",
    "length",
    "max_tokens",
    "max_output_tokens",
    "token_limit",
    "output_limit",
}
_TRUNCATED_RESPONSE_STATUSES = {
    "incomplete",
    "length",
    "max_tokens",
    "max_output_tokens",
}


class ProofFormalizationContextOverflowError(ValueError):
    """Compiler-facing overflow that retains the failed proof route."""

    def __init__(self, feedback: ProofAttemptFeedback) -> None:
        super().__init__(feedback.error_output)
        self.feedback = feedback


def _truncated_model_output_feedback(
    theorem_candidate: ProofCandidate,
    *,
    attempt_number: int,
    strategy: str,
) -> ProofAttemptFeedback:
    return ProofAttemptFeedback(
        attempt=attempt_number,
        theorem_id=theorem_candidate.theorem_id,
        reasoning=_INCOMPLETE_MODEL_OUTPUT_REASON,
        lean_code="",
        error_output=(
            "MODEL OUTPUT TRUNCATED: the selected model/provider reached its maximum output length "
            "before returning usable Lean proof code. This attempt is counted as failed; "
            "research will continue with the next proof attempt or candidate."
        ),
        goal_states="",
        strategy=strategy,
        success=False,
    )


def _latest_assistant_pack_for_lean_attempts() -> tuple[str, str, list[dict[str, Any]]]:
    """Reuse the latest Assistant proof context without refreshing during Lean attempts."""
    assistant_pack = assistant_proof_search_coordinator.get_latest_reusable_pack()
    if not assistant_pack or not assistant_pack.results:
        return "", "", []
    return (
        assistant_pack.target_hash,
        assistant_pack.to_prompt_context(),
        [support.model_dump(mode="json") for support in assistant_pack.results],
    )


def _response_indicates_output_truncation(response: dict[str, Any]) -> bool:
    if not isinstance(response, dict):
        return False
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0] if isinstance(choices[0], dict) else {}
        finish_reason = str(first_choice.get("finish_reason") or "").strip().lower()
        if finish_reason in _TRUNCATED_FINISH_REASONS:
            return True
    status = str(response.get("status") or "").strip().lower()
    if status in _TRUNCATED_RESPONSE_STATUSES:
        return True
    incomplete_details = response.get("incomplete_details")
    if isinstance(incomplete_details, dict):
        reason = str(incomplete_details.get("reason") or "").strip().lower()
        if reason in _TRUNCATED_FINISH_REASONS:
            return True
    return False


def _is_stop_requested(should_stop: ShouldStopFn) -> bool:
    if should_stop is None:
        return False
    try:
        return bool(should_stop())
    except Exception:
        return False


def _format_attempt_feedback_for_assistant(attempts: list[ProofAttemptFeedback], limit: int = 5) -> str:
    if not attempts:
        return ""
    lines: list[str] = []
    for attempt in attempts[-limit:]:
        parts = [
            f"Attempt {attempt.attempt}",
            f"strategy={attempt.strategy}",
            f"success={attempt.success}",
            f"reasoning={attempt.reasoning}",
            f"lean_error={attempt.error_output}",
            f"goal_states={attempt.goal_states}",
        ]
        lines.append("\n".join(part for part in parts if part and part.split("=", 1)[-1].strip()))
    text = "\n\n---\n\n".join(lines)
    return text[:5000] + ("..." if len(text) > 5000 else "")


def _is_json_parse_error(exc: Exception) -> bool:
    if isinstance(exc, json.JSONDecodeError):
        return True
    if not isinstance(exc, ValueError):
        return False
    message = str(exc).lower()
    return any(marker in message for marker in _JSON_PARSE_ERROR_MARKERS)


def _is_malformed_model_output_feedback(feedback: ProofAttemptFeedback) -> bool:
    return (
        not feedback.success
        and not feedback.lean_code
        and not feedback.error_output
        and feedback.reasoning == _MALFORMED_MODEL_OUTPUT_REASON
    )


def _is_lean_workspace_error_feedback(feedback: ProofAttemptFeedback) -> bool:
    error_output = feedback.error_output or ""
    return (
        not feedback.success
        and error_output.startswith(_LEAN_WORKSPACE_ERROR_PREFIX)
    )


def _is_context_overflow_feedback(feedback: ProofAttemptFeedback) -> bool:
    error_output = feedback.error_output or ""
    return (
        not feedback.success
        and error_output.startswith(_MANDATORY_FULL_SOURCE_CONTEXT_OVERFLOW_PREFIX)
    )


def _is_provider_context_overflow(exc: Exception) -> bool:
    return isinstance(exc, ProviderContextLengthError) or is_provider_context_length_error(exc)


def _provider_context_overflow_feedback(
    theorem_candidate: ProofCandidate,
    *,
    attempt_number: int,
    strategy: str,
    context_window: int,
    max_output_tokens: int,
    exc: Exception,
) -> ProofAttemptFeedback:
    typed_error = isinstance(exc, ProviderContextLengthError)
    route = getattr(exc, "route", None) if typed_error else None
    safe_detail = str(getattr(exc, "safe_message", "") or "").strip() if typed_error else str(exc)
    return ProofAttemptFeedback(
        attempt=attempt_number,
        theorem_id=theorem_candidate.theorem_id,
        reasoning="Provider rejected the proof prompt because it exceeded the model context window.",
        lean_code="",
        error_output=(
            f"{_MANDATORY_FULL_SOURCE_CONTEXT_OVERFLOW_PREFIX}: Provider rejected the proof "
            "formalization prompt before generation because the input exceeded the selected "
            f"model context window. Configured total context={context_window}, "
            f"max output reserve={max_output_tokens}. Lean 4 was not run."
            + (f" Provider detail: {safe_detail}" if safe_detail else "")
        ),
        goal_states="",
        strategy=strategy,
        success=False,
        configured_model=getattr(route, "configured_model", None),
        configured_provider=getattr(route, "configured_provider", None),
        effective_model=getattr(route, "model", None),
        effective_provider=getattr(route, "provider", None),
        overflow_origin="provider",
    )


class ProofFormalizationAgent:
    """Turn theorem candidates into Lean 4 code and retry with feedback."""

    def __init__(
        self,
        model_id: str,
        context_window: int,
        max_output_tokens: int,
        role_id: str,
    ) -> None:
        self.model_id = model_id
        self.context_window = context_window
        self.max_output_tokens = max_output_tokens
        self.role_id = role_id
        self.task_sequence = 0

    def get_current_task_id(self) -> str:
        return f"proof_form_{self.task_sequence:03d}"

    @staticmethod
    def _build_source_excerpt(theorem_statement: str, source_content: str) -> str:
        statement = (theorem_statement or "").strip()
        content = source_content or ""
        if not content:
            return ""

        search_token = statement[:80]
        if search_token:
            match_index = content.find(search_token)
            if match_index >= 0:
                start = max(0, match_index - 2500)
                end = min(len(content), match_index + max(len(statement), 1) + 2500)
                return content[start:end]

        return content[:6000]

    @staticmethod
    def _normalize_tactic_trace(raw_tactics) -> tuple[List[str], List[str]]:
        tactic_commands: List[str] = []
        tactic_trace: List[str] = []
        for item in raw_tactics or []:
            tactic = ""
            reasoning = ""
            if isinstance(item, dict):
                tactic = str(item.get("tactic") or item.get("command") or "").strip()
                reasoning = str(item.get("reasoning") or item.get("note") or "").strip()
            else:
                tactic = str(item or "").strip()

            if not tactic:
                continue

            tactic_commands.append(tactic)
            tactic_trace.append(f"{tactic} -- {reasoning}" if reasoning else tactic)
        return tactic_commands, tactic_trace

    @staticmethod
    def _compose_tactic_script_code(theorem_header: str, tactic_commands: List[str]) -> str:
        header = (theorem_header or "").strip()
        if not header:
            return ""
        if ":= by" not in header and not header.rstrip().endswith("by"):
            header = f"{header} := by"

        lines = header.splitlines()
        for tactic in tactic_commands:
            stripped = str(tactic or "").rstrip()
            if not stripped:
                continue
            for line in stripped.splitlines():
                lines.append(f"  {line.rstrip()}")

        code = "\n".join(lines).strip()
        if not code:
            return ""

        first_lines = code.splitlines()[:5]
        if not any(line.strip().startswith("import ") for line in first_lines):
            code = f"import Mathlib\n\n{code}"
        return code + "\n"

    def _fit_prompt_to_context(
        self,
        prompt_builder,
        *,
        min_excerpt_length: int,
        source_excerpt: str,
        **prompt_kwargs,
    ) -> tuple[str, str, int, int]:
        prompt = prompt_builder(source_excerpt=source_excerpt, **prompt_kwargs)
        prompt_tokens = count_tokens(prompt)
        try:
            max_input_tokens = rag_config.get_available_input_tokens(self.context_window, self.max_output_tokens)
        except ValueError:
            return prompt, source_excerpt, 0, prompt_tokens
        # Full source content is mandatory proof context. Only the focused
        # excerpt may be reduced to fit the prompt.
        while prompt_tokens > max_input_tokens and len(source_excerpt) > min_excerpt_length:
            source_excerpt = source_excerpt[: max(len(source_excerpt) // 2, min_excerpt_length)]
            prompt = prompt_builder(source_excerpt=source_excerpt, **prompt_kwargs)
            prompt_tokens = count_tokens(prompt)
        if (
            prompt_tokens > max_input_tokens
            and prompt_kwargs.get("retrieved_proofs_context")
            and prompt_kwargs.get("retrieved_proofs_context") != _PROOF_SEARCH_CONTEXT_OMITTED
        ):
            prompt_kwargs["retrieved_proofs_context"] = _PROOF_SEARCH_CONTEXT_OMITTED
            prompt = prompt_builder(source_excerpt=source_excerpt, **prompt_kwargs)
            prompt_tokens = count_tokens(prompt)
        return prompt, source_excerpt, max_input_tokens, prompt_tokens

    async def _record_syntheticlib4_context_exposure(
        self,
        records: list[dict[str, Any]],
        *,
        theorem_candidate: ProofCandidate,
        lean_code: str,
    ) -> None:
        """
        Persist local usage metadata when full SyntheticLib4 code was model-visible.

        This is intentionally conservative: it records `entire_code_used=false`
        because MOTO cannot prove the generated proof consumed an external proof
        as a whole dependency unless a later artifact/dependency extractor says so.
        """
        used_proofs: list[dict[str, str]] = []
        for record in records:
            if record.get("corpus") != "syntheticlib4":
                continue
            if not str(record.get("lean_code") or "").strip():
                continue
            fingerprint = str(record.get("fingerprint") or record.get("proof_id") or "").strip()
            statement_hash = str(record.get("theorem_statement_hash") or "").strip()
            code_hash = str(record.get("lean_code_hash") or "").strip()
            if not fingerprint:
                continue
            used_proofs.append(
                {
                    "fingerprint": fingerprint,
                    "theorem_statement_hash": statement_hash,
                    "lean_code_hash": code_hash,
                }
            )
        if not used_proofs:
            return

        artifact_hash = hashlib.sha256(
            "\n\n".join(
                [
                    theorem_candidate.theorem_id,
                    theorem_candidate.statement,
                    lean_code or "",
                ]
            ).encode("utf-8")
        ).hexdigest()
        result = await execute_search_lean_proofs(
            {
                "action": "attest_usage",
                "usage_attestation": {
                    "retrieval_batch_id": "local_proof_search_prefetch",
                    "used_proofs": used_proofs,
                    "entire_code_used": False,
                    "usage_type": "model_visible_context",
                    "moto_artifact_hash": artifact_hash,
                },
            }
        )
        if not result.get("success"):
            logger.debug("SyntheticLib4 local context-exposure attestation failed: %s", result.get("error"))

    async def _run_full_script_attempt(
        self,
        *,
        user_research_prompt: str,
        source_type: str,
        theorem_candidate: ProofCandidate,
        prior_attempts: List[ProofAttemptFeedback],
        source_excerpt: str,
        source_content: str,
        attempt_number: int,
        smt_hint: Optional[SmtHint] = None,
        source_title: str = "",
        retrieved_proofs_context: str = "",
        assistant_memory_target_hash: str = "",
    ) -> tuple[str, str, ProofAttemptFeedback]:
        prompt, source_excerpt, max_input_tokens, prompt_tokens = self._fit_prompt_to_context(
            build_proof_formalization_prompt,
            min_excerpt_length=1500,
            user_prompt=user_research_prompt,
            source_type=source_type,
            theorem_statement=theorem_candidate.statement,
            formal_sketch=theorem_candidate.formal_sketch,
            full_source_content=source_content,
            source_excerpt=source_excerpt,
            prior_attempts=prior_attempts,
            relevant_lemmas=theorem_candidate.relevant_lemmas,
            smt_hint=smt_hint,
            source_title=source_title,
            expected_novelty_tier=theorem_candidate.expected_novelty_tier,
            prompt_relevance_rationale=theorem_candidate.prompt_relevance_rationale,
            novelty_rationale=theorem_candidate.novelty_rationale,
            why_not_standard_known_result=theorem_candidate.why_not_standard_known_result,
            retrieved_proofs_context=retrieved_proofs_context,
        )

        if prompt_tokens > max_input_tokens:
            feedback = ProofAttemptFeedback(
                attempt=attempt_number,
                theorem_id=theorem_candidate.theorem_id,
                reasoning="Mandatory full-source proof context is too large for the configured context window.",
                error_output=(
                    f"{_MANDATORY_FULL_SOURCE_CONTEXT_OVERFLOW_PREFIX}: Prompt too large after shrinking only the focused excerpt "
                    f"({prompt_tokens} > {max_input_tokens}). Configured total context={self.context_window}, "
                    f"max output reserve={self.max_output_tokens}, safety buffer={rag_config.context_buffer_tokens}. "
                    "Full source content is mandatory "
                    "and was not truncated or dropped."
                ),
                strategy="full_script",
                success=False,
                overflow_origin="local_preflight",
                prompt_tokens=prompt_tokens,
                max_input_tokens=max_input_tokens,
            )
            return "", source_excerpt, feedback

        task_id = self.get_current_task_id()
        self.task_sequence += 1

        try:
            response = await api_client_manager.generate_completion(
                task_id=task_id,
                role_id=self.role_id,
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_output_tokens,
                temperature=0.0,
            )
            if assistant_memory_target_hash:
                assistant_proof_search_coordinator.mark_pack_consumed_by_solver(
                    assistant_memory_target_hash,
                    role_id=self.role_id,
                    task_id=task_id,
                )
            if _response_indicates_output_truncation(response):
                logger.warning(
                    "ProofFormalizationAgent full-script attempt %s for %s returned a length-truncated provider response.",
                    attempt_number,
                    theorem_candidate.theorem_id,
                )
                return (
                    "",
                    source_excerpt,
                    _truncated_model_output_feedback(
                        theorem_candidate,
                        attempt_number=attempt_number,
                        strategy="full_script",
                    ),
                )
            if not response or not response.get("choices"):
                raise ValueError("Empty response from formalization model.")

            message = response["choices"][0].get("message", {})
            content = extract_message_text(message)
            if not content:
                raise ValueError("No content in formalization model response.")

            data = parse_json(content)
            if isinstance(data, list):
                data = data[0] if data else {}
            if not isinstance(data, dict):
                data = {}

            theorem_name = str(data.get("theorem_name", "")).strip()
            lean_code = str(data.get("lean_code", "")).strip()
            reasoning = str(data.get("reasoning", "")).strip()
            if not lean_code:
                raise ValueError("Formalization model did not return Lean 4 code.")

            lean_result = await get_lean4_client().check_proof(
                lean_code,
                timeout=system_config.lean4_proof_timeout,
            )
            feedback = ProofAttemptFeedback(
                attempt=attempt_number,
                theorem_id=theorem_candidate.theorem_id,
                reasoning=reasoning,
                lean_code=lean_code,
                error_output=lean_result.error_output,
                goal_states=lean_result.goal_states,
                strategy="full_script",
                success=lean_result.success,
            )
            return theorem_name, source_excerpt, feedback
        except FreeModelExhaustedError:
            raise
        except RetryableProviderError:
            raise
        except Exception as exc:
            if _is_provider_context_overflow(exc):
                feedback = _provider_context_overflow_feedback(
                    theorem_candidate,
                    attempt_number=attempt_number,
                    strategy="full_script",
                    context_window=self.context_window,
                    max_output_tokens=self.max_output_tokens,
                    exc=exc,
                )
                logger.warning(
                    "ProofFormalizationAgent full-script attempt %s context overflow for %s: %s",
                    attempt_number,
                    theorem_candidate.theorem_id,
                    exc,
                )
                return "", source_excerpt, feedback
            if is_non_retryable_model_error(exc):
                raise
            if is_retryable_model_output_error(exc):
                logger.warning(
                    "ProofFormalizationAgent full-script attempt %s for %s hit provider output truncation: %s",
                    attempt_number,
                    theorem_candidate.theorem_id,
                    exc,
                )
                return (
                    "",
                    source_excerpt,
                    _truncated_model_output_feedback(
                        theorem_candidate,
                        attempt_number=attempt_number,
                        strategy="full_script",
                    ),
                )
            if is_transient_model_call_error(exc):
                raise RetryableProviderError(
                    provider="unknown",
                    provider_label="Inference provider",
                    role_id=self.role_id,
                    model=self.model_id,
                    reason="transient_provider_error",
                    message=format_transient_provider_error(exc),
                ) from exc
            is_parse_error = _is_json_parse_error(exc)
            feedback = ProofAttemptFeedback(
                attempt=attempt_number,
                theorem_id=theorem_candidate.theorem_id,
                reasoning=(
                    _MALFORMED_MODEL_OUTPUT_REASON
                    if is_parse_error
                    else "Formalization attempt failed before Lean 4 verification."
                ),
                lean_code="",
                error_output="" if is_parse_error else str(exc),
                goal_states="",
                strategy="full_script",
                success=False,
            )
            logger.warning(
                "ProofFormalizationAgent full-script attempt %s failed for %s: %s",
                attempt_number,
                theorem_candidate.theorem_id,
                exc,
            )
            return "", source_excerpt, feedback

    async def prove_candidate(
        self,
        user_research_prompt: str,
        source_type: str,
        theorem_candidate: ProofCandidate,
        source_content: str,
        *,
        max_attempts: int = 5,
        attempt_callback: Optional[AttemptCallback] = None,
        attempt_start_callback: Optional[AttemptStartCallback] = None,
        prior_attempts: Optional[List[ProofAttemptFeedback]] = None,
        starting_attempt_number: Optional[int] = None,
        smt_hint: Optional[SmtHint] = None,
        source_title: str = "",
        should_stop: ShouldStopFn = None,
    ) -> Tuple[bool, str, str, List[ProofAttemptFeedback]]:
        """Attempt to formalize and verify one theorem candidate with full scripts."""
        attempts: List[ProofAttemptFeedback] = list(prior_attempts or [])
        source_excerpt = theorem_candidate.source_excerpt or self._build_source_excerpt(
            theorem_candidate.statement,
            source_content,
        )
        assistant_target_hash, retrieved_proofs_context, retrieved_proof_records = (
            _latest_assistant_pack_for_lean_attempts()
        )
        theorem_name = ""

        next_attempt_number = (
            starting_attempt_number
            if starting_attempt_number is not None
            else (attempts[-1].attempt + 1 if attempts else 1)
        )

        attempt_offset = 0
        malformed_output_retries = 0
        max_malformed_output_retries = max(1, max_attempts)

        while attempt_offset < max_attempts:
            if _is_stop_requested(should_stop):
                logger.info(
                    "ProofFormalizationAgent.prove_candidate: stop requested, aborting before attempt %s for %s.",
                    next_attempt_number + attempt_offset,
                    theorem_candidate.theorem_id,
                )
                break
            attempt_number = next_attempt_number + attempt_offset
            if attempt_start_callback and malformed_output_retries == 0:
                await attempt_start_callback(attempt_number, "full_script")

            current_theorem_name, source_excerpt, feedback = await self._run_full_script_attempt(
                user_research_prompt=user_research_prompt,
                source_type=source_type,
                theorem_candidate=theorem_candidate,
                prior_attempts=attempts,
                source_excerpt=source_excerpt,
                source_content=source_content,
                attempt_number=attempt_number,
                smt_hint=smt_hint,
                source_title=source_title,
                retrieved_proofs_context=retrieved_proofs_context,
                assistant_memory_target_hash=assistant_target_hash,
            )

            terminal_malformed_output = False
            if _is_malformed_model_output_feedback(feedback):
                malformed_output_retries += 1
                logger.warning(
                    "ProofFormalizationAgent full-script attempt %s for %s produced malformed model output; retrying without consuming Lean attempt budget (%s/%s).",
                    attempt_number,
                    theorem_candidate.theorem_id,
                    malformed_output_retries,
                    max_malformed_output_retries,
                )
                if malformed_output_retries < max_malformed_output_retries:
                    continue
                terminal_malformed_output = True
            else:
                malformed_output_retries = 0
            if current_theorem_name:
                theorem_name = current_theorem_name

            attempts.append(feedback)
            if attempt_callback:
                await attempt_callback(feedback)

            if feedback.success:
                await self._record_syntheticlib4_context_exposure(
                    retrieved_proof_records,
                    theorem_candidate=theorem_candidate,
                    lean_code=feedback.lean_code,
                )
                return True, theorem_name, feedback.lean_code, attempts
            if _is_lean_workspace_error_feedback(feedback):
                break
            if _is_context_overflow_feedback(feedback):
                break
            if terminal_malformed_output:
                break
            attempt_offset += 1

        final_code = attempts[-1].lean_code if attempts else ""
        return False, theorem_name, final_code, attempts

    async def prove_candidate_tactic_script(
        self,
        user_research_prompt: str,
        source_type: str,
        theorem_candidate: ProofCandidate,
        source_content: str,
        *,
        max_attempts: int = 2,
        attempt_callback: Optional[AttemptCallback] = None,
        attempt_start_callback: Optional[AttemptStartCallback] = None,
        prior_attempts: Optional[List[ProofAttemptFeedback]] = None,
        starting_attempt_number: Optional[int] = None,
        smt_hint: Optional[SmtHint] = None,
        source_title: str = "",
        should_stop: ShouldStopFn = None,
    ) -> Tuple[bool, str, str, List[ProofAttemptFeedback]]:
        """Attempt to formalize and verify one theorem candidate with tactic scripts."""
        attempts: List[ProofAttemptFeedback] = list(prior_attempts or [])
        source_excerpt = theorem_candidate.source_excerpt or self._build_source_excerpt(
            theorem_candidate.statement,
            source_content,
        )
        assistant_target_hash, retrieved_proofs_context, retrieved_proof_records = (
            _latest_assistant_pack_for_lean_attempts()
        )
        theorem_name = ""

        next_attempt_number = (
            starting_attempt_number
            if starting_attempt_number is not None
            else (attempts[-1].attempt + 1 if attempts else 1)
        )

        attempt_offset = 0
        malformed_output_retries = 0
        max_malformed_output_retries = max(1, max_attempts)

        while attempt_offset < max_attempts:
            if _is_stop_requested(should_stop):
                logger.info(
                    "ProofFormalizationAgent.prove_candidate_tactic_script: stop requested, aborting before attempt %s for %s.",
                    next_attempt_number + attempt_offset,
                    theorem_candidate.theorem_id,
                )
                break
            attempt_number = next_attempt_number + attempt_offset
            if attempt_start_callback and malformed_output_retries == 0:
                await attempt_start_callback(attempt_number, "tactic_script")
            prompt, source_excerpt, max_input_tokens, prompt_tokens = self._fit_prompt_to_context(
                build_proof_tactic_script_prompt,
                min_excerpt_length=1500,
                user_prompt=user_research_prompt,
                source_type=source_type,
                theorem_statement=theorem_candidate.statement,
                formal_sketch=theorem_candidate.formal_sketch,
                full_source_content=source_content,
                source_excerpt=source_excerpt,
                prior_attempts=attempts,
                relevant_lemmas=theorem_candidate.relevant_lemmas,
                smt_hint=smt_hint,
                source_title=source_title,
                expected_novelty_tier=theorem_candidate.expected_novelty_tier,
                prompt_relevance_rationale=theorem_candidate.prompt_relevance_rationale,
                novelty_rationale=theorem_candidate.novelty_rationale,
                why_not_standard_known_result=theorem_candidate.why_not_standard_known_result,
                retrieved_proofs_context=retrieved_proofs_context,
            )

            if prompt_tokens > max_input_tokens:
                feedback = ProofAttemptFeedback(
                    attempt=attempt_number,
                    theorem_id=theorem_candidate.theorem_id,
                    reasoning="Mandatory full-source proof context is too large for the configured context window.",
                    error_output=(
                        f"{_MANDATORY_FULL_SOURCE_CONTEXT_OVERFLOW_PREFIX}: Prompt too large after shrinking only the focused excerpt "
                        f"({prompt_tokens} > {max_input_tokens}). Configured total context={self.context_window}, "
                        f"max output reserve={self.max_output_tokens}, safety buffer={rag_config.context_buffer_tokens}. "
                        "Full source content is mandatory "
                        "and was not truncated or dropped."
                    ),
                    strategy="tactic_script",
                    success=False,
                    overflow_origin="local_preflight",
                    prompt_tokens=prompt_tokens,
                    max_input_tokens=max_input_tokens,
                )
                attempts.append(feedback)
                if attempt_callback:
                    await attempt_callback(feedback)
                break

            task_id = self.get_current_task_id()
            self.task_sequence += 1

            try:
                response = await api_client_manager.generate_completion(
                    task_id=task_id,
                    role_id=self.role_id,
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_output_tokens,
                    temperature=0.0,
                )
                if assistant_target_hash:
                    assistant_proof_search_coordinator.mark_pack_consumed_by_solver(
                        assistant_target_hash,
                        role_id=self.role_id,
                        task_id=task_id,
                    )
                if _response_indicates_output_truncation(response):
                    logger.warning(
                        "ProofFormalizationAgent tactic-script attempt %s for %s returned a length-truncated provider response.",
                        attempt_number,
                        theorem_candidate.theorem_id,
                    )
                    feedback = _truncated_model_output_feedback(
                        theorem_candidate,
                        attempt_number=attempt_number,
                        strategy="tactic_script",
                    )
                    attempts.append(feedback)
                    if attempt_callback:
                        await attempt_callback(feedback)
                    attempt_offset += 1
                    continue
                if not response or not response.get("choices"):
                    raise ValueError("Empty response from tactic formalization model.")

                message = response["choices"][0].get("message", {})
                content = extract_message_text(message)
                if not content:
                    raise ValueError("No content in tactic formalization model response.")

                data = parse_json(content)
                if isinstance(data, list):
                    data = data[0] if data else {}
                if not isinstance(data, dict):
                    data = {}

                theorem_name = str(data.get("theorem_name", "")).strip()
                theorem_header = str(data.get("theorem_header", "")).strip()
                reasoning = str(data.get("reasoning", "")).strip()
                tactic_commands, tactic_trace = self._normalize_tactic_trace(
                    data.get("tactics") or data.get("tactic_steps") or []
                )

                if not theorem_header or not tactic_commands:
                    logger.info(
                        "Tactic script response malformed for %s attempt %s; falling back to full-script mode.",
                        theorem_candidate.theorem_id,
                        attempt_number,
                    )
                    current_theorem_name, source_excerpt, feedback = await self._run_full_script_attempt(
                        user_research_prompt=user_research_prompt,
                        source_type=source_type,
                        theorem_candidate=theorem_candidate,
                        prior_attempts=attempts,
                        source_excerpt=source_excerpt,
                        source_content=source_content,
                        attempt_number=attempt_number,
                        smt_hint=smt_hint,
                        source_title=source_title,
                        retrieved_proofs_context=retrieved_proofs_context,
                        assistant_memory_target_hash=assistant_target_hash,
                    )
                    if current_theorem_name:
                        theorem_name = current_theorem_name
                    terminal_malformed_output = False
                    if _is_malformed_model_output_feedback(feedback):
                        malformed_output_retries += 1
                        logger.warning(
                            "ProofFormalizationAgent fallback full-script attempt %s for %s produced malformed model output; retrying without consuming Lean attempt budget (%s/%s).",
                            attempt_number,
                            theorem_candidate.theorem_id,
                            malformed_output_retries,
                            max_malformed_output_retries,
                        )
                        if malformed_output_retries < max_malformed_output_retries:
                            continue
                        terminal_malformed_output = True
                    else:
                        malformed_output_retries = 0
                    attempts.append(feedback)
                    if attempt_callback:
                        await attempt_callback(feedback)
                    if feedback.success:
                        await self._record_syntheticlib4_context_exposure(
                            retrieved_proof_records,
                            theorem_candidate=theorem_candidate,
                            lean_code=feedback.lean_code,
                        )
                        return True, theorem_name, feedback.lean_code, attempts
                    if _is_lean_workspace_error_feedback(feedback):
                        break
                    if _is_context_overflow_feedback(feedback):
                        break
                    if terminal_malformed_output:
                        break
                    attempt_offset += 1
                    continue

                lean_code = self._compose_tactic_script_code(theorem_header, tactic_commands)
                lean_result = await get_lean4_client().check_tactic_script(
                    theorem_header,
                    tactic_commands,
                    timeout=system_config.lean4_proof_timeout,
                )
                feedback = ProofAttemptFeedback(
                    attempt=attempt_number,
                    theorem_id=theorem_candidate.theorem_id,
                    reasoning=reasoning,
                    lean_code=lean_code,
                    error_output=lean_result.tactic_error_slice or lean_result.error_output,
                    goal_states=lean_result.goal_states,
                    strategy="tactic_script",
                    tactic_trace=tactic_trace,
                    success=lean_result.success,
                )
                malformed_output_retries = 0
                attempts.append(feedback)
                if attempt_callback:
                    await attempt_callback(feedback)

                if lean_result.success:
                    await self._record_syntheticlib4_context_exposure(
                        retrieved_proof_records,
                        theorem_candidate=theorem_candidate,
                        lean_code=lean_code,
                    )
                    return True, theorem_name, lean_code, attempts
                if _is_lean_workspace_error_feedback(feedback):
                    break
                if _is_context_overflow_feedback(feedback):
                    break
                attempt_offset += 1
            except FreeModelExhaustedError:
                raise
            except RetryableProviderError:
                raise
            except Exception as exc:
                if _is_provider_context_overflow(exc):
                    feedback = _provider_context_overflow_feedback(
                        theorem_candidate,
                        attempt_number=attempt_number,
                        strategy="tactic_script",
                        context_window=self.context_window,
                        max_output_tokens=self.max_output_tokens,
                        exc=exc,
                    )
                    logger.warning(
                        "ProofFormalizationAgent tactic-script attempt %s context overflow for %s: %s",
                        attempt_number,
                        theorem_candidate.theorem_id,
                        exc,
                    )
                    attempts.append(feedback)
                    if attempt_callback:
                        await attempt_callback(feedback)
                    break
                if is_non_retryable_model_error(exc):
                    raise
                if is_retryable_model_output_error(exc):
                    logger.warning(
                        "ProofFormalizationAgent tactic-script attempt %s for %s hit provider output truncation: %s",
                        attempt_number,
                        theorem_candidate.theorem_id,
                        exc,
                    )
                    feedback = _truncated_model_output_feedback(
                        theorem_candidate,
                        attempt_number=attempt_number,
                        strategy="tactic_script",
                    )
                    attempts.append(feedback)
                    if attempt_callback:
                        await attempt_callback(feedback)
                    attempt_offset += 1
                    continue
                if is_transient_model_call_error(exc):
                    raise RetryableProviderError(
                        provider="unknown",
                        provider_label="Inference provider",
                        role_id=self.role_id,
                        model=self.model_id,
                        reason="transient_provider_error",
                        message=format_transient_provider_error(exc),
                    ) from exc
                is_parse_error = _is_json_parse_error(exc)
                feedback = ProofAttemptFeedback(
                    attempt=attempt_number,
                    theorem_id=theorem_candidate.theorem_id,
                    reasoning=(
                        _MALFORMED_MODEL_OUTPUT_REASON
                        if is_parse_error
                        else "Tactic-script formalization attempt failed before Lean 4 verification."
                    ),
                    lean_code="",
                    error_output="" if is_parse_error else str(exc),
                    goal_states="",
                    strategy="tactic_script",
                    success=False,
                )
                logger.warning(
                    "ProofFormalizationAgent tactic-script attempt %s failed for %s: %s",
                    attempt_number,
                    theorem_candidate.theorem_id,
                    exc,
                )
                terminal_malformed_output = False
                if _is_malformed_model_output_feedback(feedback):
                    malformed_output_retries += 1
                    logger.warning(
                        "ProofFormalizationAgent tactic-script attempt %s for %s produced malformed model output; retrying without consuming Lean attempt budget (%s/%s).",
                        attempt_number,
                        theorem_candidate.theorem_id,
                        malformed_output_retries,
                        max_malformed_output_retries,
                    )
                    if malformed_output_retries < max_malformed_output_retries:
                        continue
                    terminal_malformed_output = True
                else:
                    malformed_output_retries = 0
                attempts.append(feedback)
                if attempt_callback:
                    await attempt_callback(feedback)
                if terminal_malformed_output:
                    break
                if _is_context_overflow_feedback(feedback):
                    break
                attempt_offset += 1

        final_code = attempts[-1].lean_code if attempts else ""
        return False, theorem_name, final_code, attempts

    @staticmethod
    def is_context_overflow_feedback(feedback: ProofAttemptFeedback) -> bool:
        """True when the attempt failed because mandatory full source did not fit."""
        return _is_context_overflow_feedback(feedback)
