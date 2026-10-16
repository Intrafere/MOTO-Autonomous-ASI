"""Registered API-key subscription providers and their recovery adapters.

Each provider profile gets one ``SubscriptionApiKeyClient`` singleton and one
``ProviderAdapter`` registered with the shared provider-recovery registry, which
the provider conformance suite exercises for every registered provider.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from backend.shared.provider_errors import ProviderRouteIdentity
from backend.shared.provider_recovery.models import FailureKind, ProviderFailure
from backend.shared.provider_recovery.registry import ProviderRegistry
from backend.shared.subscription_providers.api_key_client import SubscriptionApiKeyClient
from backend.shared.subscription_providers.errors import (
    SubscriptionAuthError,
    SubscriptionContextLimitError,
    SubscriptionEntitlementError,
    SubscriptionProviderError,
    SubscriptionUsageLimitError,
)
from backend.shared.subscription_providers.kimi_code import KIMI_CODE_PROFILE
from backend.shared.subscription_providers.metadata import API_KEY_SUBSCRIPTION_PROVIDER_IDS
from backend.shared.subscription_providers.profile import ApiKeyProviderProfile
from backend.shared.subscription_providers.sakana_fugu import SAKANA_FUGU_PROFILE

_PROFILES: Tuple[ApiKeyProviderProfile, ...] = (SAKANA_FUGU_PROFILE, KIMI_CODE_PROFILE)

if tuple(sorted(profile.provider_id for profile in _PROFILES)) != tuple(sorted(API_KEY_SUBSCRIPTION_PROVIDER_IDS)):
    raise RuntimeError("Every API-key subscription provider in metadata needs exactly one registered profile.")

_CLIENTS: Dict[str, SubscriptionApiKeyClient] = {
    profile.provider_id: SubscriptionApiKeyClient(profile) for profile in _PROFILES
}


class SubscriptionProviderAdapter:
    """``ProviderAdapter`` over one API-key subscription client."""

    def __init__(self, client: SubscriptionApiKeyClient) -> None:
        self.client = client
        self.provider_id = client.provider_id
        self.provider_label = client.provider_label

    async def complete(self, **kwargs: Any) -> Dict[str, Any]:
        return await self.client.generate_completion(**kwargs)

    def classify_error(self, error: BaseException, **context: Any) -> ProviderFailure:
        route = ProviderRouteIdentity(
            provider=self.provider_id,
            model=str(context.get("model") or ""),
            role_id=str(context.get("role_id") or ""),
            task_id=str(context.get("task_id") or ""),
            route_kind="primary",
        )
        detail = str(error)
        status_code = getattr(error, "status_code", None)
        provider_code = str(getattr(error, "error_code", "") or "")
        if isinstance(error, SubscriptionUsageLimitError):
            return ProviderFailure(
                kind=FailureKind.TIMED_USAGE_LIMIT,
                route=route,
                safe_detail=detail,
                retry_after_seconds=error.resets_in_seconds,
                reset_at=error.resets_at,
                status_code=status_code,
                provider_code=provider_code,
                cause=error,
            )
        if isinstance(error, SubscriptionContextLimitError):
            kind = FailureKind.CONTEXT_LIMIT
        elif isinstance(error, SubscriptionEntitlementError):
            kind = FailureKind.ENTITLEMENT
        elif isinstance(error, SubscriptionAuthError):
            kind = FailureKind.AUTHENTICATION
        elif isinstance(error, SubscriptionProviderError) and "connection failed after retries" in detail:
            kind = FailureKind.TRANSIENT_TRANSPORT
        else:
            kind = FailureKind.UNKNOWN_HARD
        return ProviderFailure(
            kind=kind,
            route=route,
            safe_detail=detail,
            status_code=status_code,
            provider_code=provider_code,
            fallback_eligible=kind != FailureKind.CONTEXT_LIMIT,
            cause=error,
        )


provider_adapter_registry = ProviderRegistry()
for _client in _CLIENTS.values():
    provider_adapter_registry.register(SubscriptionProviderAdapter(_client))


def get_api_key_subscription_client(provider_id: str) -> Optional[SubscriptionApiKeyClient]:
    return _CLIENTS.get(str(provider_id or "").strip())


def api_key_subscription_clients() -> Tuple[SubscriptionApiKeyClient, ...]:
    return tuple(_CLIENTS.values())


async def close_api_key_subscription_clients() -> None:
    for client in _CLIENTS.values():
        await client.close()
