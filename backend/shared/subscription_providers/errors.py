"""Typed errors raised by API-key subscription provider clients.

Every error carries ``provider`` and ``provider_label`` so routing, cooldown, and
notification code stay provider-neutral. Messages start with the provider label
so the shared text classifiers in ``model_error_utils`` recognize the provider.
"""
from __future__ import annotations

from typing import Optional


class SubscriptionProviderError(RuntimeError):
    """Base error for an API-key subscription provider request."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        provider_label: str,
        status_code: Optional[int] = None,
        error_code: str = "",
    ) -> None:
        self.provider = provider
        self.provider_label = provider_label
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(message)


class SubscriptionAuthError(SubscriptionProviderError):
    """The API key is missing or rejected."""


class SubscriptionRequestError(SubscriptionProviderError):
    """The provider rejected or failed a non-auth request."""


class SubscriptionUsageLimitError(SubscriptionRequestError):
    """The subscription reported a timed usage window with a credible reset time."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        provider_label: str,
        resets_at: Optional[int] = None,
        resets_in_seconds: Optional[int] = None,
        plan_type: str = "",
        status_code: Optional[int] = None,
        error_code: str = "",
    ) -> None:
        self.resets_at = resets_at
        self.resets_in_seconds = resets_in_seconds
        self.plan_type = plan_type
        detail = message or f"The {provider_label} usage limit has been reached."
        if resets_in_seconds is not None:
            detail = f"{detail} Resets in {resets_in_seconds} seconds."
        super().__init__(
            detail,
            provider=provider,
            provider_label=provider_label,
            status_code=status_code,
            error_code=error_code,
        )


class SubscriptionEntitlementError(SubscriptionRequestError):
    """The subscription is not entitled to the request; operator repair is required."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        provider_label: str,
        status_code: Optional[int] = None,
        error_code: str = "",
        terminal_guidance: str = "",
    ) -> None:
        self.terminal_guidance = terminal_guidance
        super().__init__(
            message or f"The {provider_label} subscription is not entitled to this request.",
            provider=provider,
            provider_label=provider_label,
            status_code=status_code,
            error_code=error_code,
        )


class SubscriptionContextLimitError(SubscriptionRequestError):
    """The provider rejected the request because its input is too large.

    The message carries the shared ``context_length_exceeded`` marker so the
    context-overflow lifecycle recognizes it like any other provider rejection.
    """
