"""Bounded JSON-contract handling for Autonomous strategy and Tier 3 roles.

Each role parses its reply strictly through the shared parser. A reply that does
not match the role's contract gets one bounded retry whose ERROR line names the
failed field. If the retry also fails, the typed ``ModelOutputContractError``
reaches the caller, which reports it as operational activity: never a semantic
rejection, a provider retry wait, or a silently defaulted decision.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar
from uuid import uuid4

from backend.shared.api_client_manager import api_client_manager
from backend.shared.config import rag_config
from backend.shared.json_parser import parse_json, sanitize_model_output_for_retry_context
from backend.shared.model_output_contract import (
    ModelOutputContractError,
    OutputRepair,
    report_model_output_normalizations,
    safe_model_token,
    wait_after_repeated_contract_failure,
)
from backend.shared.response_extraction import extract_message_text
from backend.shared.utils import count_tokens

logger = logging.getLogger(__name__)

MODEL_OUTPUT_CONTRACT_FAILED_EVENT = "model_output_contract_failed"

T = TypeVar("T")


def response_text(response: Any) -> str:
    """Return the assistant text of a completion response, or an empty string."""
    if not isinstance(response, dict):
        return ""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return ""
    return extract_message_text(message) or ""


# ---------------------------------------------------------------------------
# Field checks. Every failure is a typed contract error naming the field; the
# summaries are system-authored and model tokens only enter through the
# structured (sanitized) field lists.
# ---------------------------------------------------------------------------


def json_object(data: Any, *, role: str) -> Dict[str, Any]:
    """Return the reply object, accepting the parser's supported array wrapper."""
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    if isinstance(data, dict):
        return data
    raise ModelOutputContractError("The response was not a JSON object", role=role, stage="shape")


def _missing(key: str, *, role: str) -> ModelOutputContractError:
    return ModelOutputContractError(
        "The response is missing a required field",
        role=role,
        stage="missing_key",
        missing_keys=(key,),
    )


def require_choice(
    data: Dict[str, Any],
    key: str,
    choices: Sequence[str],
    *,
    role: str,
    case_insensitive: bool = False,
) -> str:
    """Return ``data[key]`` when it is exactly one of ``choices``."""
    if key not in data or data[key] is None:
        raise _missing(key, role=role)
    value = data[key]
    if case_insensitive and isinstance(value, str):
        value = value.strip().lower()
    if not isinstance(value, str) or value not in choices:
        raise ModelOutputContractError(
            f"A field must be exactly one of: {', '.join(choices)}",
            role=role,
            stage="invalid_value",
            invalid_fields=(key,),
        )
    return value


def require_text(data: Dict[str, Any], key: str, *, role: str) -> str:
    """Return a required non-empty string field, stripped."""
    if key not in data or data[key] is None:
        raise _missing(key, role=role)
    value = data[key]
    if not isinstance(value, str) or not value.strip():
        raise ModelOutputContractError(
            "A field must be a non-empty string",
            role=role,
            stage="invalid_value",
            invalid_fields=(key,),
        )
    return value.strip()


def optional_text(data: Dict[str, Any], key: str, *, role: str) -> str:
    """Return an optional string field ("" when omitted or null)."""
    value = data.get(key)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ModelOutputContractError(
            "A field must be a string",
            role=role,
            stage="invalid_value",
            invalid_fields=(key,),
        )
    return value.strip()


def require_bool(data: Dict[str, Any], key: str, *, role: str) -> bool:
    """Return a required JSON boolean field (strings such as "false" are invalid)."""
    if key not in data or data[key] is None:
        raise _missing(key, role=role)
    value = data[key]
    if type(value) is not bool:
        raise ModelOutputContractError(
            "A field must be a JSON boolean (true or false)",
            role=role,
            stage="invalid_value",
            invalid_fields=(key,),
        )
    return value


def require_string_list(data: Dict[str, Any], key: str, *, role: str) -> List[str]:
    """Return a required list of non-empty strings, stripped."""
    if key not in data or data[key] is None:
        raise _missing(key, role=role)
    value = data[key]
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ModelOutputContractError(
            "A field must be a list of non-empty strings",
            role=role,
            stage="invalid_value",
            invalid_fields=(key,),
        )
    return [item.strip() for item in value]


# ---------------------------------------------------------------------------
# Parsing and the one bounded contract retry.
# ---------------------------------------------------------------------------


def parse_contract_reply(
    content: str,
    parser: Callable[[Any], T],
    *,
    role: str,
) -> Tuple[T, List[OutputRepair]]:
    """Strictly parse ``content`` and apply ``parser``; contract failures are typed."""
    repairs: List[OutputRepair] = []
    try:
        data = parse_json(content or "", repairs=repairs)
    except ModelOutputContractError:
        raise
    except ValueError as exc:
        raise ModelOutputContractError.syntax(exc, role=role) from exc
    try:
        value = parser(data)
    except ModelOutputContractError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        # Never echo the exception text: it may quote model output.
        raise ModelOutputContractError(
            f"The response did not match the required JSON format ({type(exc).__name__})",
            role=role,
            stage="shape",
        ) from exc
    return value, repairs


def build_contract_retry_messages(
    *,
    prompt: str,
    failed_output: str,
    error: ModelOutputContractError,
    response_noun: str,
    response_shape: str,
    max_input_tokens: int,
) -> List[Dict[str, str]]:
    """Return the bounded conversational retry messages for one contract failure."""
    retry_prompt = (
        f"Your previous {response_noun} response could not be used.\n\n"
        f"ERROR: {error}\n\n"
        f"Return the same {response_noun} in valid JSON only. "
        "Use this exact top-level shape:\n"
        f"{response_shape}\n\n"
        "Respond with ONLY the JSON object, no markdown and no explanation."
    )
    failed_preview = sanitize_model_output_for_retry_context(failed_output or "", max_chars=2000)
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": failed_preview},
        {"role": "user", "content": retry_prompt},
    ]

    def total(items: List[Dict[str, str]]) -> int:
        return sum(count_tokens(str(item.get("content") or "")) for item in items)

    if not failed_preview.strip() or total(messages) > max_input_tokens:
        messages[1]["content"] = (
            "[failed output omitted because retry context would exceed the model input budget]"
            if failed_preview.strip()
            else "[the previous response contained no usable text]"
        )
    if total(messages) > max_input_tokens:
        combined = f"{prompt}\n\n---\n{retry_prompt}"
        if count_tokens(combined) <= max_input_tokens:
            messages = [{"role": "user", "content": combined}]
        else:
            logger.warning(
                "Contract retry instruction for %s does not fit; retrying the original prompt.",
                response_noun,
            )
            messages = [{"role": "user", "content": prompt}]
    return messages


async def parse_with_contract_retry(
    content: str,
    parser: Callable[[Any], T],
    *,
    prompt: str,
    task_id: str,
    role_id: str,
    model: str,
    context_window: int,
    max_output_tokens: int,
    response_noun: str,
    response_shape: str,
) -> T:
    """Parse a role reply; on a contract failure make exactly one bounded retry.

    Provider failures from the retry call propagate unchanged through the shared
    provider recovery path. A second contract failure raises the typed error
    (chained to the first). Applied parser normalizations are published as
    ``model_output_normalized`` activity.
    """
    report_task_id = task_id
    try:
        value, repairs = parse_contract_reply(content, parser, role=role_id)
    except ModelOutputContractError as first_error:
        logger.info(
            "%s output did not match its JSON contract; attempting one bounded retry: %s",
            safe_model_token(role_id, 120),
            first_error,
        )
        max_input_tokens = rag_config.get_available_input_tokens(context_window, max_output_tokens)
        report_task_id = f"{task_id}_retry"
        response = await api_client_manager.generate_completion(
            task_id=report_task_id,
            role_id=role_id,
            model=model,
            messages=build_contract_retry_messages(
                prompt=prompt,
                failed_output=content,
                error=first_error,
                response_noun=response_noun,
                response_shape=response_shape,
                max_input_tokens=max_input_tokens,
            ),
            max_tokens=max_output_tokens,
            temperature=0.0,
        )
        try:
            value, repairs = parse_contract_reply(response_text(response), parser, role=role_id)
        except ModelOutputContractError as retry_error:
            raise retry_error from first_error
    await report_model_output_normalizations(role_id, repairs, task_id=report_task_id, model=model)
    return value


async def report_model_output_contract_failure(
    role_id: str,
    error: ModelOutputContractError,
    *,
    task_id: str = "",
    model: str = "",
    outcome: str = "",
) -> Optional[Dict[str, Any]]:
    """Publish one operational ``model_output_contract_failed`` activity item.

    ``outcome`` is system-authored text saying what the workflow does next. The
    item is operational (not a semantic rejection) and carries the role's route
    identity. Publication failure is logged and never fails the caller.
    """
    from backend.shared.activity import normalize_activity_payload
    from backend.shared.provider_route_identity import merge_route_identity, with_route_identity

    role = str(role_id or "").strip()
    logger.warning(
        "Model output contract failure for role %s task %s (stage=%s fields=%s)",
        safe_model_token(role, 160),
        safe_model_token(task_id, 120),
        error.stage,
        error.field_summary,
    )
    try:
        fields = api_client_manager.route_identity_for_role(role, model=model, task_id=task_id)
        model_label = str(fields.get("model") or fields.get("configured_model") or "").strip()
        detail = error.field_summary or error.stage
        message = (
            f"{model_label or 'The model'} returned output that did not match the required "
            f"JSON format after one retry ({detail})."
        )
        if outcome:
            message = f"{message} {outcome.strip()}"
        payload: Dict[str, Any] = {
            # Each occurrence is its own activity item, so repeated failures of
            # one role are never collapsed into the first.
            "event_id": f"{MODEL_OUTPUT_CONTRACT_FAILED_EVENT}:{uuid4().hex}",
            "role_id": role,
            "task_id": str(task_id or ""),
            "workflow_mode": str(fields.get("workflow_mode") or ""),
            "contract_stage": error.stage,
            "item_path": error.item_path,
            "missing_keys": list(error.missing_keys),
            "unknown_keys": list(error.unknown_keys),
            "invalid_fields": list(error.invalid_fields),
            "error_summary": str(error),
            "message": with_route_identity(message, fields),
            "activity_classification": "operational",
            "activity_category": "model_output",
            "activity_severity": "warning",
            "activity_outcome": "contract_failed",
            **api_client_manager._role_activity_source_fields(role),
        }
        payload = merge_route_identity(payload, fields)
        published = normalize_activity_payload(MODEL_OUTPUT_CONTRACT_FAILED_EVENT, payload)
        await api_client_manager._broadcast(MODEL_OUTPUT_CONTRACT_FAILED_EVENT, published)
        return published
    except Exception as exc:  # visibility must never fail the role's work
        logger.warning(
            "Could not publish model output contract activity for role %s: %s",
            safe_model_token(role, 160),
            type(exc).__name__,
        )
        return None


def never_stop() -> bool:
    """Stop check for agents that are not bound to a workflow's stop state."""
    return False


class ContractFailurePacer:
    """Paces one loop's re-asks after consecutive model-output contract failures.

    The first failure is re-asked at once; each further consecutive failure waits
    a little longer through the shared stop-aware emit-and-wait seam. Re-asks are
    never capped. Any call that returns a usable reply resets the count.
    """

    def __init__(self, should_stop: Optional[Callable[[], bool]] = None) -> None:
        self._should_stop = should_stop or never_stop
        self.consecutive_failures = 0

    def reset(self) -> None:
        self.consecutive_failures = 0

    async def after_failure(self, role_id: str) -> None:
        self.consecutive_failures += 1
        await wait_after_repeated_contract_failure(
            role_id,
            self.consecutive_failures,
            should_stop=self._should_stop,
        )
