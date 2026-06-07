"""Helpers for extracting assistant text from OpenAI-compatible responses.

Some local/OpenAI-compatible servers put the only usable final answer in
provider-specific fields such as ``reasoning`` or ``thinking`` when
``message.content`` is empty.  Keep that compatibility in one place so callers
do not each invent their own response-field policy.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional


logger = logging.getLogger(__name__)

FALLBACK_TEXT_FIELDS = ("reasoning", "reasoning_content", "thinking")


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
        allow_fallback_fields=allow_fallback_fields,
        context=context,
    )
