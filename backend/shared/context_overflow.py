"""Shared user-facing context overflow messages and event metadata."""

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
    if route is not None:
        if route.model:
            payload["effective_model"] = route.model
        if route.provider:
            payload["effective_provider"] = route.provider
        if route.host_provider:
            payload["effective_host_provider"] = route.host_provider
        if route.route_kind:
            payload["route_kind"] = route.route_kind
    return payload
