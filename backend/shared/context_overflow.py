"""Shared user-facing context overflow messages and event metadata."""

import re
from typing import Any, Optional

from backend.shared.models import ModelConfig
from backend.shared.provider_errors import ProviderRouteIdentity

CONTEXT_OVERFLOW_STOP_REASON = "context_overflow"
CONTEXT_OVERFLOW_STOP_MESSAGE = (
    "Research stopped. Some required source content must be injected directly to preserve "
    "answer quality, and it reached the maximum context size for the selected model. "
    "Start a new session with a condensed prompt, or choose a model with a higher "
    "context limit."
)
CONTEXT_OVERFLOW_RESOLUTION = (
    "Start a new session with a condensed prompt, or choose a model with a higher context limit."
)


def context_overflow_model_payload(
    config: Optional[ModelConfig] = None,
    *,
    route: Optional[ProviderRouteIdentity] = None,
) -> dict[str, Any]:
    """Return safe configured/effective model metadata for an overflow event."""
    payload: dict[str, Any] = {}
    configured_model = route.configured_model if route else ""
    configured_provider = route.configured_provider if route else ""
    if not configured_model and config is not None:
        configured_model = config.model_id
    if not configured_provider and config is not None:
        configured_provider = config.provider
    if configured_model:
        payload["configured_model"] = configured_model
    if configured_provider:
        payload["configured_provider"] = configured_provider
    configured_host = route.host_provider if route else ""
    if not configured_host and config is not None:
        configured_host = config.openrouter_provider or ""
    if configured_host:
        payload["configured_host_provider"] = configured_host
    if route is not None:
        if route.model:
            payload["effective_model"] = route.model
        if route.provider:
            payload["effective_provider"] = route.provider
        effective_host = route.upstream_host_provider or route.host_provider
        if effective_host:
            payload["effective_host_provider"] = effective_host
        if route.route_kind:
            payload["route_kind"] = route.route_kind
    return payload


CONTEXT_OVERFLOW_TOKEN_FIELDS = (
    "required_tokens",
    "available_tokens",
    "context_window",
    "output_reserve",
)
# Older error types name the same counts differently.
_TOKEN_FIELD_ALIASES = {
    "required_tokens": ("required_tokens", "prompt_tokens"),
    "available_tokens": ("available_tokens", "max_input_tokens", "available_input_tokens"),
    "context_window": ("context_window", "context_limit"),
    "output_reserve": ("output_reserve",),
}
_NUMBER = r"(\d[\d,]*)"
# Overflow messages written across MOTO, matched only to fill counts the error
# object did not carry. Each pattern yields (required, available) or
# (required, context window).
_REQUIRED_VS_AVAILABLE_PATTERNS = (
    re.compile(rf"{_NUMBER}\s*(?:tokens?)?\s*>\s*{_NUMBER}", re.IGNORECASE),
    re.compile(
        rf"{_NUMBER}\s*tokens?\)?\s*exceeds?[^\d]{{0,60}}?{_NUMBER}",
        re.IGNORECASE,
    ),
    re.compile(
        rf"mandatory tokens:\s*{_NUMBER}\.\s*available input tokens:\s*{_NUMBER}",
        re.IGNORECASE,
    ),
    re.compile(
        rf"requires\s+{_NUMBER}\s+input tokens but the configured input budget is\s+{_NUMBER}",
        re.IGNORECASE,
    ),
)
_CONTEXT_WINDOW_PATTERN = re.compile(
    rf"(?:configured total context|context window)\s*[=(:]?\s*\(?{_NUMBER}",
    re.IGNORECASE,
)


def _positive_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str):
        digits = value.replace(",", "").strip()
        if digits.isdigit():
            number = int(digits)
            return number if number > 0 else None
    return None


def _error_chain(error: Optional[BaseException]) -> list[Any]:
    chain: list[Any] = []
    seen: set[int] = set()
    pending: list[Any] = [error]
    while pending and len(chain) < 8:
        item = pending.pop(0)
        if item is None or id(item) in seen:
            continue
        seen.add(id(item))
        chain.append(item)
        for attr in ("cause", "__cause__", "__context__", "feedback"):
            nested = getattr(item, attr, None)
            if nested is not None:
                pending.append(nested)
    return chain


def context_overflow_token_payload(
    error: Optional[BaseException] = None,
    *,
    detail: str = "",
    config: Optional[ModelConfig] = None,
) -> dict[str, int]:
    """Return the attempted and allowed token counts behind an overflow.

    Structured attributes on the error (or anything it wraps) win; the overflow
    message text fills counts only when no error carried them, and the role's
    configured window and output reserve fill whatever is still missing.
    """
    payload: dict[str, int] = {}
    chain = _error_chain(error)
    for field, aliases in _TOKEN_FIELD_ALIASES.items():
        for item in chain:
            value = next(
                (
                    number
                    for alias in aliases
                    if (number := _positive_int(getattr(item, alias, None))) is not None
                ),
                None,
            )
            if value is not None:
                payload[field] = value
                break

    texts = [detail] if detail else []
    texts.extend(str(item) for item in chain if isinstance(item, BaseException))
    for text in texts:
        if "required_tokens" in payload and (
            "available_tokens" in payload or "context_window" in payload
        ):
            break
        for pattern in _REQUIRED_VS_AVAILABLE_PATTERNS:
            match = pattern.search(text)
            if not match:
                continue
            required = _positive_int(match.group(1))
            allowed = _positive_int(match.group(2))
            if required is None or allowed is None or required <= allowed:
                continue
            payload.setdefault("required_tokens", required)
            if "context window" in match.group(0).lower():
                payload.setdefault("context_window", allowed)
            else:
                payload.setdefault("available_tokens", allowed)
            break
        if "context_window" not in payload:
            window_match = _CONTEXT_WINDOW_PATTERN.search(text)
            if window_match:
                window = _positive_int(window_match.group(1))
                if window is not None:
                    payload["context_window"] = window
    if config is not None:
        configured_window = _positive_int(getattr(config, "context_window", None))
        if configured_window is not None:
            payload.setdefault("context_window", configured_window)
        configured_output = _positive_int(getattr(config, "max_output_tokens", None))
        if configured_output is not None and "available_tokens" not in payload:
            payload.setdefault("output_reserve", configured_output)
    return payload
