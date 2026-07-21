"""Helpers for distinguishing model availability failures from ordinary output errors."""
from __future__ import annotations

import httpx

from backend.shared.openrouter_client import (
    CreditExhaustionError,
    FreeModelExhaustedError,
    OpenRouterInvalidResponseError,
    OpenRouterPrivacyPolicyError,
)
from backend.shared.provider_errors import ProviderContextLengthError, ProviderRouteError


_NON_RETRYABLE_MODEL_ERROR_MARKERS = (
    "account free credits exhausted",
    "all free model options exhausted",
    "and no fallback configured",
    "and no lm studio fallback",
    "boost requested but no openrouter api key",
    "free credits exhausted",
    "no api key is set",
    "no fallback configured",
    "no lm studio fallback",
    "no openrouter api key is available",
    "openrouter credits exhausted",
    "openrouter privacy settings are blocking",
)

_RETRYABLE_OUTPUT_FAILURE_MARKERS = (
    ("response.incomplete", "max_output_tokens"),
    ("response.incomplete", "max output tokens"),
)

_TRANSIENT_MODEL_CALL_MARKERS = (
    "an error occurred while processing your request",
    "bad gateway",
    "codex connection failed",
    "connecterror",
    "connection timeout",
    "disconnect/reset before headers",
    "gateway timeout",
    "http 500",
    "http 502",
    "http 503",
    "http 504",
    "incomplete chunked read",
    "openai codex connection failed",
    "openai codex transient",
    "openrouter connection failed",
    "peer closed connection",
    "readerror",
    "remoteprotocolerror",
    "response missing 'choices'",
    "sakana fugu connection failed",
    "sakana fugu transient",
    "server_error",
    "service unavailable",
    "temporarily unavailable",
    "upstream connect error",
    "upstream provider timeout",
    "xai grok connection failed",
    "xai grok transient",
    "you can retry your request",
)

_TRANSIENT_MODEL_PROVIDER_MARKERS = (
    "codex",
    "grok",
    "openrouter",
    "sakana",
    "xai",
)

_TRANSIENT_PROVIDER_ERROR_PREFIX = "TRANSIENT PROVIDER ERROR"
_TRANSIENT_INVALID_RESPONSE_BODY_MARKERS = (
    "bad gateway",
    "cloudflare",
    "gateway error",
    "gateway timeout",
    "service unavailable",
    "temporarily unavailable",
    "upstream connect error",
)

_PROVIDER_CONTEXT_LENGTH_MARKERS = (
    "context_length_exceeded",
    "context length exceeded",
    "exceeds the context window",
    "exceeded the context window",
    "too large for the context window",
    "maximum context size",
    "input exceeds",
    "request too large",
    "prompt is too long",
)


def format_transient_provider_error(exc: Exception) -> str:
    """Return a checkpoint-preserving transient provider error message."""
    message = str(exc or "").strip()
    if _TRANSIENT_PROVIDER_ERROR_PREFIX in message:
        return message
    return (
        "TRANSIENT PROVIDER ERROR: provider connection failed before usable proof output. "
        "Preserve the proof checkpoint and retry later."
        + (f" Original error: {message}" if message else "")
    )


def is_retryable_model_output_error(exc: Exception) -> bool:
    """Return true when the provider returned a usable request with unusable output."""
    message = str(exc or "").lower()
    return any(all(marker in message for marker in markers) for markers in _RETRYABLE_OUTPUT_FAILURE_MARKERS)


def is_provider_context_length_error(exc: Exception) -> bool:
    """Return true when a provider rejects a request before generation because input is too large."""
    if isinstance(exc, ProviderContextLengthError):
        return True
    if isinstance(exc, ProviderRouteError) and exc.cause is not None:
        return is_provider_context_length_error(exc.cause)
    message = str(exc or "").lower()
    return any(marker in message for marker in _PROVIDER_CONTEXT_LENGTH_MARKERS)


def is_transient_model_call_error(exc: Exception) -> bool:
    """Return true for provider/network failures that should not be treated as config errors."""
    if isinstance(exc, ProviderRouteError) and exc.cause is not None:
        if isinstance(
            exc.cause,
            (
                httpx.ConnectError,
                httpx.ReadError,
                httpx.RemoteProtocolError,
                httpx.TimeoutException,
            ),
        ):
            return True
        return is_transient_model_call_error(exc.cause)
    if isinstance(exc, OpenRouterInvalidResponseError):
        content_type = str(getattr(exc, "content_type", "") or "").lower()
        body_preview = str(getattr(exc, "body_preview", "") or "").lower()
        status_code = int(getattr(exc, "status_code", 0) or 0)
        if status_code >= 500:
            return True
        return any(marker in body_preview for marker in _TRANSIENT_INVALID_RESPONSE_BODY_MARKERS)
    message = str(exc or "").lower()
    if not any(marker in message for marker in _TRANSIENT_MODEL_PROVIDER_MARKERS):
        return False
    return any(marker in message for marker in _TRANSIENT_MODEL_CALL_MARKERS)


def is_non_retryable_model_error(exc: Exception) -> bool:
    """Return true when a model/API failure should halt workflow progress."""
    if isinstance(exc, ProviderContextLengthError):
        return True
    if isinstance(exc, ProviderRouteError) and exc.cause is not None:
        return is_non_retryable_model_error(exc.cause)
    if isinstance(
        exc,
        (
            CreditExhaustionError,
            FreeModelExhaustedError,
            OpenRouterPrivacyPolicyError,
        ),
    ):
        return True
    message = str(exc).lower()
    if is_retryable_model_output_error(exc):
        return False
    if is_transient_model_call_error(exc):
        return False
    return any(marker in message for marker in _NON_RETRYABLE_MODEL_ERROR_MARKERS)
