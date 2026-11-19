"""Model roles for master proof scaffolds: scaffold author, sorry solver, change Validator."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Sequence, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from backend.autonomous.agents.proof_formalization_agent import (
    _MANDATORY_FULL_SOURCE_CONTEXT_OVERFLOW_PREFIX,
    _is_json_parse_error,
    _is_provider_context_overflow,
    _provider_context_overflow_feedback,
    _response_indicates_output_truncation,
    _truncated_model_output_feedback,
    transient_provider_retry_error,
)
from backend.autonomous.prompts.master_proof_prompts import (
    MASTER_PROOF_ATTEMPTS_PER_OBLIGATION,
    build_master_proof_obligation_prompt,
    build_master_proof_scaffold_change_validation_prompt,
    build_master_proof_scaffold_prompt,
    without_verified_proof_library,
)
from backend.shared.api_client_manager import RetryableProviderError, api_client_manager
from backend.shared.config import rag_config
from backend.shared.json_parser import parse_json, sanitize_model_output_for_retry_context
from backend.shared.lean4_client import LeanWorkspaceRepairRequiredError
from backend.shared.model_error_utils import (
    is_non_retryable_model_error,
    is_retryable_model_output_error,
    is_transient_model_call_error,
)
from backend.shared.models import ProofAttemptFeedback, ProofCandidate
from backend.shared.openrouter_client import FreeModelExhaustedError
from backend.shared.prompt_feedback_budget import fit_prompt_with_feedback
from backend.shared.provider_errors import ProviderContextLengthError
from backend.shared.response_extraction import extract_message_text
from backend.shared.utils import count_tokens

logger = logging.getLogger(__name__)

_MALFORMED_OUTPUT_ERROR = (
    "MODEL OUTPUT REJECTED: the response was not the required JSON object with a "
    "non-empty lean_code string. Lean 4 was not run. Return only the required JSON."
)


class MasterProofContextOverflow(Exception):
    """Mandatory master proof context does not fit even with only the newest feedback."""

    def __init__(self, feedback: ProofAttemptFeedback) -> None:
        super().__init__(feedback.error_output)
        self.feedback = feedback


@dataclass(frozen=True)
class MasterProofSubmission:
    """Parsed model answer for a scaffold or obligation attempt."""

    lean_code: str
    reasoning: str = ""
    action: str = "submit_proof"
    root_theorem_name: str = ""


class _ScaffoldChangeDecision(BaseModel):
    model_config = ConfigDict(extra="ignore")

    decision: str = Field(min_length=1)
    reasoning: str = Field(min_length=1)


@dataclass(frozen=True)
class ScaffoldChangeReview:
    """Validator outcome; ``decided`` is False when no usable decision was produced."""

    decided: bool
    approved: bool = False
    reasoning: str = ""


class MasterProofAgent:
    """Writes master proof scaffolds and attempts individual sorry obligations."""

    def __init__(self, model_id: str, context_window: int, max_output_tokens: int, role_id: str) -> None:
        self.model_id = model_id
        self.context_window = context_window
        self.max_output_tokens = max_output_tokens
        self.role_id = role_id
        self.task_sequence = 0

    def _next_task_id(self) -> str:
        task_id = f"proof_master_{self.task_sequence:03d}"
        self.task_sequence += 1
        return task_id

    def _overflow_feedback(
        self,
        candidate: ProofCandidate,
        *,
        attempt_number: int,
        prompt_tokens: int,
        max_input_tokens: int,
    ) -> ProofAttemptFeedback:
        return ProofAttemptFeedback(
            attempt=attempt_number,
            theorem_id=candidate.theorem_id,
            reasoning="Mandatory master proof context is too large for the configured context window.",
            error_output=(
                f"{_MANDATORY_FULL_SOURCE_CONTEXT_OVERFLOW_PREFIX}: Master proof prompt too large "
                f"with only the newest feedback entry retained ({prompt_tokens} > {max_input_tokens}). "
                f"Configured total context={self.context_window}, max output reserve="
                f"{self.max_output_tokens}, safety buffer={rag_config.context_buffer_tokens}. "
                "The full source content and the full scaffold are mandatory and were not "
                "truncated or dropped."
            ),
            strategy="full_script",
            success=False,
            overflow_origin="local_preflight",
            prompt_tokens=prompt_tokens,
            max_input_tokens=max_input_tokens,
        )

    async def _attempt(
        self,
        *,
        candidate: ProofCandidate,
        attempt_number: int,
        prior_attempts: Sequence[ProofAttemptFeedback],
        build_prompt: Callable[[Tuple[ProofAttemptFeedback, ...], str], str],
        user_prompt: str,
        parse: Callable[[dict[str, Any]], MasterProofSubmission],
    ) -> Union[MasterProofSubmission, ProofAttemptFeedback]:
        """Run one model call; return a submission or the failed-attempt feedback.

        The verified proof library inside ``user_prompt`` is optional context. When
        the complete prompt cannot carry it together with the whole feedback window,
        the library block is left out before any feedback entry is dropped.
        """
        try:
            max_input_tokens = rag_config.get_available_input_tokens(
                self.context_window, self.max_output_tokens
            )
        except ValueError:
            max_input_tokens = 0
        window = tuple(prior_attempts)[-MASTER_PROOF_ATTEMPTS_PER_OBLIGATION:]

        def fit_with(prompt_text: str):
            return fit_prompt_with_feedback(
                prior_attempts,
                build_prompt=lambda entries: build_prompt(entries, prompt_text),
                available_tokens=max(0, max_input_tokens),
                max_entries=MASTER_PROOF_ATTEMPTS_PER_OBLIGATION,
            )

        active_user_prompt = user_prompt
        fit = fit_with(active_user_prompt)
        if not fit.fits or len(fit.retained_entries) < len(window):
            reduced_user_prompt = without_verified_proof_library(user_prompt)
            if reduced_user_prompt != (user_prompt or "").strip():
                active_user_prompt = reduced_user_prompt
                fit = fit_with(active_user_prompt)
                logger.info(
                    "Master proof attempt %s for %s omitted the verified proof library "
                    "to fit the configured input budget.",
                    attempt_number,
                    candidate.theorem_id,
                )
        if not fit.fits:
            raise MasterProofContextOverflow(
                self._overflow_feedback(
                    candidate,
                    attempt_number=attempt_number,
                    prompt_tokens=fit.prompt_tokens,
                    max_input_tokens=max_input_tokens,
                )
            )
        prompt = fit.prompt
        retained = fit.retained_entries
        task_id = self._next_task_id()
        response_was_truncated = False
        try:
            while True:
                try:
                    response = await api_client_manager.generate_completion(
                        messages=[{"role": "user", "content": prompt}],
                        task_id=task_id,
                        role_id=self.role_id,
                        model=self.model_id,
                        max_tokens=self.max_output_tokens,
                        temperature=0.0,
                    )
                    break
                except ProviderContextLengthError:
                    # Shed the oldest retained feedback entry; the newest always stays.
                    if len(retained) <= 1:
                        raise
                    refit = fit_prompt_with_feedback(
                        retained[1:],
                        build_prompt=lambda entries: build_prompt(entries, active_user_prompt),
                        available_tokens=max(0, max_input_tokens),
                        max_entries=MASTER_PROOF_ATTEMPTS_PER_OBLIGATION,
                    )
                    prompt = refit.prompt
                    retained = refit.retained_entries
            response_was_truncated = _response_indicates_output_truncation(response)
            if not response or not response.get("choices"):
                raise ValueError("Empty response from formalization model.")
            content = extract_message_text(response["choices"][0].get("message", {}))
            if not content:
                raise ValueError("No content in formalization model response.")
            data = parse_json(content)
            if isinstance(data, list):
                data = data[0] if data else {}
            if not isinstance(data, dict):
                data = {}
            submission = parse(data)
            if not submission.lean_code.strip():
                raise ValueError("no json found: lean_code missing")
            return submission
        except (FreeModelExhaustedError, RetryableProviderError, LeanWorkspaceRepairRequiredError):
            raise
        except Exception as exc:
            if _is_provider_context_overflow(exc):
                raise MasterProofContextOverflow(
                    _provider_context_overflow_feedback(
                        candidate,
                        attempt_number=attempt_number,
                        strategy="full_script",
                        context_window=self.context_window,
                        max_output_tokens=self.max_output_tokens,
                        exc=exc,
                    )
                ) from exc
            if response_was_truncated or is_retryable_model_output_error(exc):
                logger.warning(
                    "Master proof attempt %s for %s returned a length-truncated response.",
                    attempt_number,
                    candidate.theorem_id,
                )
                return _truncated_model_output_feedback(
                    candidate,
                    attempt_number=attempt_number,
                    strategy="full_script",
                    requested_output_tokens=self.max_output_tokens,
                )
            if is_non_retryable_model_error(exc):
                raise
            if is_transient_model_call_error(exc):
                raise transient_provider_retry_error(
                    exc, role_id=self.role_id, model=self.model_id
                ) from exc
            if not (_is_json_parse_error(exc) or isinstance(exc, (ValueError, TypeError))):
                raise
            logger.warning(
                "Master proof attempt %s for %s returned malformed output: %s",
                attempt_number,
                candidate.theorem_id,
                type(exc).__name__,
            )
            return ProofAttemptFeedback(
                attempt=attempt_number,
                theorem_id=candidate.theorem_id,
                reasoning="Model returned malformed output.",
                error_output=_MALFORMED_OUTPUT_ERROR,
                strategy="full_script",
                success=False,
                failure_kind="malformed_output",
                requested_output_tokens=self.max_output_tokens,
            )

    async def propose_scaffold(
        self,
        *,
        user_prompt: str,
        source_type: str,
        source_title: str,
        candidate: ProofCandidate,
        source_content: str,
        prior_attempts: Sequence[ProofAttemptFeedback],
        attempt_number: int,
    ) -> Union[MasterProofSubmission, ProofAttemptFeedback]:
        def build_prompt(entries: Tuple[ProofAttemptFeedback, ...], prompt_text: str) -> str:
            return build_master_proof_scaffold_prompt(
                user_prompt=prompt_text,
                source_type=source_type,
                source_title=source_title,
                theorem_statement=candidate.statement,
                formal_sketch=candidate.formal_sketch,
                full_source_content=source_content,
                relevant_lemmas=candidate.relevant_lemmas,
                prior_attempts=entries,
            )

        def parse(data: dict[str, Any]) -> MasterProofSubmission:
            lean_code = data.get("lean_code")
            return MasterProofSubmission(
                lean_code=lean_code.strip() if isinstance(lean_code, str) else "",
                reasoning=str(data.get("reasoning") or "").strip(),
                root_theorem_name=str(data.get("root_theorem_name") or "").strip(),
            )

        return await self._attempt(
            candidate=candidate,
            attempt_number=attempt_number,
            prior_attempts=prior_attempts,
            build_prompt=build_prompt,
            user_prompt=user_prompt,
            parse=parse,
        )

    async def attempt_obligation(
        self,
        *,
        user_prompt: str,
        source_type: str,
        source_title: str,
        candidate: ProofCandidate,
        source_content: str,
        root_theorem_name: str,
        scaffold_code: str,
        obligation_name: str,
        obligation_signature: str,
        prior_attempts: Sequence[ProofAttemptFeedback],
        attempt_number: int,
    ) -> Union[MasterProofSubmission, ProofAttemptFeedback]:
        def build_prompt(entries: Tuple[ProofAttemptFeedback, ...], prompt_text: str) -> str:
            return build_master_proof_obligation_prompt(
                user_prompt=prompt_text,
                source_type=source_type,
                source_title=source_title,
                theorem_statement=candidate.statement,
                root_theorem_name=root_theorem_name,
                scaffold_code=scaffold_code,
                obligation_name=obligation_name,
                obligation_signature=obligation_signature,
                full_source_content=source_content,
                relevant_lemmas=candidate.relevant_lemmas,
                prior_attempts=entries,
            )

        def parse(data: dict[str, Any]) -> MasterProofSubmission:
            lean_code = data.get("lean_code")
            action = str(data.get("action") or "").strip().lower()
            if action not in {"submit_proof", "change_scaffold"}:
                raise ValueError("no json found: action must be submit_proof or change_scaffold")
            return MasterProofSubmission(
                lean_code=lean_code.strip() if isinstance(lean_code, str) else "",
                reasoning=str(data.get("reasoning") or "").strip(),
                action=action,
            )

        return await self._attempt(
            candidate=candidate,
            attempt_number=attempt_number,
            prior_attempts=prior_attempts,
            build_prompt=build_prompt,
            user_prompt=user_prompt,
            parse=parse,
        )


class MasterProofScaffoldChangeValidator:
    """Independent Validator deciding whether one solver may replace the scaffold."""

    def __init__(self, model_id: str, context_window: int, max_output_tokens: int, role_id: str) -> None:
        self.model_id = model_id
        self.context_window = context_window
        self.max_output_tokens = max_output_tokens
        self.role_id = role_id
        self.task_sequence = 0

    @staticmethod
    def _parse(content: str) -> ScaffoldChangeReview:
        data = parse_json(content)
        if isinstance(data, list):
            data = data[0] if data else {}
        decision = _ScaffoldChangeDecision.model_validate(data)
        verdict = decision.decision.strip().lower()
        if verdict not in {"approve", "reject"}:
            raise ValueError('decision must be "approve" or "reject"')
        return ScaffoldChangeReview(
            decided=True,
            approved=verdict == "approve",
            reasoning=decision.reasoning.strip()[:3000],
        )

    async def _complete(self, task_id: str, messages: list[dict[str, str]]) -> str:
        response = await api_client_manager.generate_completion(
            task_id=task_id,
            role_id=self.role_id,
            model=self.model_id,
            messages=messages,
            max_tokens=self.max_output_tokens,
            temperature=0.0,
        )
        if not response or not response.get("choices"):
            return ""
        return extract_message_text(response["choices"][0].get("message", {})) or ""

    async def review(
        self,
        *,
        user_prompt: str,
        theorem_statement: str,
        root_theorem_name: str,
        current_scaffold: str,
        proposed_scaffold: str,
        proposing_obligation: str,
        proposer_reasoning: str,
    ) -> ScaffoldChangeReview:
        """Return the Validator decision; provider failures propagate to the caller."""
        prompt = build_master_proof_scaffold_change_validation_prompt(
            user_prompt=user_prompt,
            theorem_statement=theorem_statement,
            root_theorem_name=root_theorem_name,
            current_scaffold=current_scaffold,
            proposed_scaffold=proposed_scaffold,
            proposing_obligation=proposing_obligation,
            proposer_reasoning=proposer_reasoning,
        )
        max_input_tokens = rag_config.get_available_input_tokens(
            self.context_window, self.max_output_tokens
        )
        if count_tokens(prompt) > max_input_tokens:
            return ScaffoldChangeReview(
                decided=False,
                reasoning=(
                    "The current and proposed scaffolds together exceed the Validator's "
                    "configured input budget, so the change could not be reviewed."
                ),
            )
        task_id = f"proof_master_val_{self.task_sequence:03d}"
        self.task_sequence += 1
        try:
            content = await self._complete(task_id, [{"role": "user", "content": prompt}])
        except ProviderContextLengthError:
            return ScaffoldChangeReview(
                decided=False,
                reasoning=(
                    "The Validator's provider rejected the scaffold-change review as too "
                    "large, so the change could not be reviewed."
                ),
            )
        try:
            return self._parse(content)
        except (ValueError, TypeError, ValidationError) as first_error:
            repair = (
                "Your prior response violated the required JSON contract "
                f"({type(first_error).__name__}). Return only one corrected JSON object with "
                'decision "approve" or "reject" and a non-empty reasoning string.'
            )
            messages = [
                {"role": "user", "content": prompt},
                {
                    "role": "assistant",
                    "content": sanitize_model_output_for_retry_context(content, max_chars=2000),
                },
                {"role": "user", "content": repair},
            ]
            if sum(count_tokens(item["content"]) for item in messages) > max_input_tokens:
                messages = [{"role": "user", "content": prompt}, {"role": "user", "content": repair}]
            if sum(count_tokens(item["content"]) for item in messages) > max_input_tokens:
                return ScaffoldChangeReview(
                    decided=False,
                    reasoning="The Validator returned no usable decision for the scaffold change.",
                )
            try:
                retry_content = await self._complete(f"{task_id}_repair", messages)
                return self._parse(retry_content)
            except (ValueError, TypeError, ValidationError):
                return ScaffoldChangeReview(
                    decided=False,
                    reasoning="The Validator returned no usable decision for the scaffold change.",
                )
