"""Helpers for distinguishing model availability failures from ordinary output errors."""
from __future__ import annotations

from backend.shared.openrouter_client import (
    CreditExhaustionError,
    FreeModelExhaustedError,
    OpenRouterPrivacyPolicyError,
)


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


def is_retryable_model_output_error(exc: Exception) -> bool:
    """Return true when the provider returned a usable request with unusable output."""
    message = str(exc or "").lower()
    return any(all(marker in message for marker in markers) for markers in _RETRYABLE_OUTPUT_FAILURE_MARKERS)


def is_non_retryable_model_error(exc: Exception) -> bool:
    """Return true when a model/API failure should halt workflow progress."""
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
    return any(marker in message for marker in _NON_RETRYABLE_MODEL_ERROR_MARKERS)
