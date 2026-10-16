"""Helpers for extracting assistant text from OpenAI-compatible responses.

Some local/OpenAI-compatible servers put the only usable final answer in
provider-specific fields such as ``reasoning`` or ``thinking`` when
``message.content`` is empty.  Keep that compatibility in one place so callers
do not each invent their own response-field policy.

A reply that stopped at the output-token limit before writing any answer is
different: its reasoning fields hold unfinished thinking, never an answer, so
the fallback does not apply to it.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional


logger = logging.getLogger(__name__)

FALLBACK_TEXT_FIELDS = ("reasoning", "reasoning_content", "thinking")

# Finish reasons meaning the provider stopped the reply at the output-token limit.
TRUNCATED_FINISH_REASONS = frozenset({
    "incomplete",
    "length",
    "max_tokens",
    "max_output_tokens",
    "token_limit",
    "output_limit",
})


def _coerce_text(value: Any) -> str:
    """Return text from common OpenAI-compatible content shapes."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text") or item.get("content") or item.get("output_text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(value)


def choice_hit_output_limit(choice: Any) -> bool:
    """Return True when a completion choice stopped at the output-token limit."""
    if not isinstance(choice, Mapping):
        return False
    finish_reason = str(choice.get("finish_reason") or "").strip().lower()
    return finish_reason in TRUNCATED_FINISH_REASONS


def drop_unfinished_thinking(response: Any, *, context: str = "") -> None:
    """Remove reasoning fields from replies cut off before any answer was written.

    Callers that read ``choices[0].message`` directly cannot see the finish
    reason, so the shared completion path strips the unfinished thinking here.
    Those callers then see an empty answer instead of the model's thoughts.
    """
    if not isinstance(response, Mapping):
        return
    choices = response.get("choices")
    if not isinstance(choices, list):
        return
    for choice in choices:
        if not choice_hit_output_limit(choice):
            continue
        message = choice.get("message")
        if not isinstance(message, dict) or _coerce_text(message.get("content")):
            continue
        removed = [field for field in FALLBACK_TEXT_FIELDS if message.pop(field, None) is not None]
        if removed:
            label = f" for {context}" if context else ""
            logger.info(
                "Reply%s hit the output limit before any answer; discarded unfinished %s.",
                label,
                "/".join(removed),
            )


def extract_message_text(
    message: Optional[Mapping[str, Any]],
    *,
    allow_fallback_fields: bool = True,
    context: str = "",
) -> str:
    """Extract assistant text from a single chat-completion message.

    ``content`` is always preferred.  If it is empty, fallback fields are used
    deliberately to support compatibility providers/models that expose final
    answer text through reasoning/thinking fields.
    """
    if not message:
        return ""

    content = _coerce_text(message.get("content"))
    if content:
        return content

    if not allow_fallback_fields:
        return ""

    for field in FALLBACK_TEXT_FIELDS:
        fallback_text = _coerce_text(message.get(field))
        if fallback_text:
            label = f" for {context}" if context else ""
            logger.info(
                "Assistant message.content was empty%s; using message.%s fallback (%s chars).",
                label,
                field,
                len(fallback_text),
            )
            return fallback_text

    return ""


def extract_response_text(
    response: Optional[Mapping[str, Any]],
    *,
    allow_fallback_fields: bool = True,
    context: str = "",
) -> str:
    """Extract assistant text from an OpenAI-compatible completion response."""
    if not response:
        return ""
    choices = response.get("choices")
    if not choices:
        return ""
    first_choice = choices[0] if isinstance(choices, list) and choices else {}
    if not isinstance(first_choice, Mapping):
        return ""
    message = first_choice.get("message")
    if not isinstance(message, Mapping):
        return ""
    return extract_message_text(
        message,
        allow_fallback_fields=allow_fallback_fields and not choice_hit_output_limit(first_choice),
        context=context,
    )
