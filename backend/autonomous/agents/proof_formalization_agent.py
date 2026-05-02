"""
Lean 4 formalization agent with iterative retry loop.
"""
from __future__ import annotations

import json
import logging
from typing import Awaitable, Callable, List, Optional, Tuple

from backend.shared.api_client_manager import api_client_manager
from backend.shared.json_parser import parse_json
from backend.shared.lean4_client import get_lean4_client
from backend.shared.models import ProofAttemptFeedback, ProofCandidate, SmtHint
from backend.shared.openrouter_client import FreeModelExhaustedError
from backend.shared.utils import count_tokens
from backend.shared.config import system_config
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
    "openrouter response missing 'choices'",
    "openrouter returned non-json body",
    "response too short",
    "unterminated string",
    "upstream provider timeout",
)
_MALFORMED_MODEL_OUTPUT_REASON = "Model returned malformed output (not valid JSON); retrying with clean context."
_LEAN_WORKSPACE_ERROR_PREFIX = "LEAN 4 WORKSPACE ERROR"


def _is_stop_requested(should_stop: ShouldStopFn) -> bool:
    if should_stop is None:
        return False
    try:
        return bool(should_stop())
    except Exception:
        return False


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
        max_input_tokens = self.context_window - self.max_output_tokens
        prompt_tokens = count_tokens(prompt)
        while prompt_tokens > max_input_tokens and len(source_excerpt) > min_excerpt_length:
            source_excerpt = source_excerpt[: max(len(source_excerpt) // 2, min_excerpt_length)]
            prompt = prompt_builder(source_excerpt=source_excerpt, **prompt_kwargs)
            prompt_tokens = count_tokens(prompt)
        return prompt, source_excerpt, max_input_tokens, prompt_tokens

    async def _run_full_script_attempt(
        self,
        *,
        user_research_prompt: str,
        source_type: str,
        theorem_candidate: ProofCandidate,
        prior_attempts: List[ProofAttemptFeedback],
        source_excerpt: str,
        attempt_number: int,
        smt_hint: Optional[SmtHint] = None,
    ) -> tuple[str, str, ProofAttemptFeedback]:
        prompt, source_excerpt, max_input_tokens, prompt_tokens = self._fit_prompt_to_context(
            build_proof_formalization_prompt,
            min_excerpt_length=1500,
            user_prompt=user_research_prompt,
            source_type=source_type,
            theorem_statement=theorem_candidate.statement,
            formal_sketch=theorem_candidate.formal_sketch,
            source_excerpt=source_excerpt,
            prior_attempts=prior_attempts,
            relevant_lemmas=theorem_candidate.relevant_lemmas,
            smt_hint=smt_hint,
        )

        if prompt_tokens > max_input_tokens:
            feedback = ProofAttemptFeedback(
                attempt=attempt_number,
                theorem_id=theorem_candidate.theorem_id,
                reasoning="Prompt too large for configured context window.",
                error_output=f"Prompt too large ({prompt_tokens} > {max_input_tokens}).",
                strategy="full_script",
                success=False,
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
            if not response or not response.get("choices"):
                raise ValueError("Empty response from formalization model.")

            message = response["choices"][0].get("message", {})
            content = message.get("content") or message.get("reasoning") or ""
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
        except Exception as exc:
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
        should_stop: ShouldStopFn = None,
    ) -> Tuple[bool, str, str, List[ProofAttemptFeedback]]:
        """Attempt to formalize and verify one theorem candidate with full scripts."""
        attempts: List[ProofAttemptFeedback] = list(prior_attempts or [])
        source_excerpt = theorem_candidate.source_excerpt or self._build_source_excerpt(
            theorem_candidate.statement,
            source_content,
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
                attempt_number=attempt_number,
                smt_hint=smt_hint,
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
                return True, theorem_name, feedback.lean_code, attempts
            if _is_lean_workspace_error_feedback(feedback):
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
        should_stop: ShouldStopFn = None,
    ) -> Tuple[bool, str, str, List[ProofAttemptFeedback]]:
        """Attempt to formalize and verify one theorem candidate with tactic scripts."""
        attempts: List[ProofAttemptFeedback] = list(prior_attempts or [])
        source_excerpt = theorem_candidate.source_excerpt or self._build_source_excerpt(
            theorem_candidate.statement,
            source_content,
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
                source_excerpt=source_excerpt,
                prior_attempts=attempts,
                relevant_lemmas=theorem_candidate.relevant_lemmas,
                smt_hint=smt_hint,
            )

            if prompt_tokens > max_input_tokens:
                malformed_output_retries = 0
                feedback = ProofAttemptFeedback(
                    attempt=attempt_number,
                    theorem_id=theorem_candidate.theorem_id,
                    reasoning="Prompt too large for configured context window.",
                    error_output=f"Prompt too large ({prompt_tokens} > {max_input_tokens}).",
                    strategy="tactic_script",
                    success=False,
                )
                attempts.append(feedback)
                if attempt_callback:
                    await attempt_callback(feedback)
                attempt_offset += 1
                continue

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
                if not response or not response.get("choices"):
                    raise ValueError("Empty response from tactic formalization model.")

                message = response["choices"][0].get("message", {})
                content = message.get("content") or message.get("reasoning") or ""
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
                        attempt_number=attempt_number,
                        smt_hint=smt_hint,
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
                        return True, theorem_name, feedback.lean_code, attempts
                    if _is_lean_workspace_error_feedback(feedback):
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
                    return True, theorem_name, lean_code, attempts
                if _is_lean_workspace_error_feedback(feedback):
                    break
                attempt_offset += 1
            except FreeModelExhaustedError:
                raise
            except Exception as exc:
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
                attempt_offset += 1

        final_code = attempts[-1].lean_code if attempts else ""
        return False, theorem_name, final_code, attempts
