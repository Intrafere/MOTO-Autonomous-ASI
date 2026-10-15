"""Single source of truth for MOTO's desktop cloud subscription providers.

Every list of provider ids, labels, capability flags, and provider error markers
is derived from ``CLOUD_PROVIDERS``. This module has no MOTO imports so models,
route identity, and error helpers can depend on it without import cycles.

Adding a provider: append one ``CloudProviderMetadata`` entry here, then (for an
API-key subscription) add its profile module and register it in ``registry``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

AuthKind = Literal["oauth_pkce", "api_key"]


@dataclass(frozen=True)
class CloudProviderMetadata:
    provider_id: str
    label: str
    short_label: str
    auth_kind: AuthKind
    # Field name in /api/features and the frontend capability map.
    capability_flag: str
    # REST path segment under /api/cloud-access/.
    api_slug: str
    # Literal reasoning effort "max" is accepted for this provider's roles.
    allows_max_effort: bool = False
    desktop_only: bool = True
    # Extra lowercase markers that identify this provider in error text, beyond
    # the label itself (for example the short label or the vendor name).
    error_markers: Tuple[str, ...] = ()

    @property
    def error_marker_set(self) -> Tuple[str, ...]:
        markers = {self.label.lower(), *(marker.lower() for marker in self.error_markers)}
        return tuple(sorted(markers))

    @property
    def transient_markers(self) -> Tuple[str, ...]:
        label = self.label.lower()
        return (f"{label} connection failed", f"{label} transient")


CLOUD_PROVIDERS: Tuple[CloudProviderMetadata, ...] = (
    CloudProviderMetadata(
        provider_id="openai_codex_oauth",
        label="OpenAI Codex",
        short_label="Codex",
        auth_kind="oauth_pkce",
        capability_flag="openai_codex_oauth_available",
        api_slug="openai-codex",
        allows_max_effort=True,
        error_markers=("codex",),
    ),
    CloudProviderMetadata(
        provider_id="xai_grok_oauth",
        label="xAI Grok",
        short_label="Grok",
        auth_kind="oauth_pkce",
        capability_flag="xai_grok_oauth_available",
        api_slug="xai-grok",
        error_markers=("grok", "xai"),
    ),
    CloudProviderMetadata(
        provider_id="sakana_fugu",
        label="Sakana Fugu",
        short_label="Fugu",
        auth_kind="api_key",
        capability_flag="sakana_fugu_available",
        api_slug="sakana-fugu",
        error_markers=("sakana",),
    ),
    CloudProviderMetadata(
        provider_id="kimi_code",
        label="Kimi Code",
        short_label="Kimi",
        auth_kind="api_key",
        capability_flag="kimi_code_available",
        api_slug="kimi-code",
        allows_max_effort=True,
    ),
)

CLOUD_PROVIDER_IDS: Tuple[str, ...] = tuple(provider.provider_id for provider in CLOUD_PROVIDERS)
API_KEY_SUBSCRIPTION_PROVIDER_IDS: Tuple[str, ...] = tuple(
    provider.provider_id for provider in CLOUD_PROVIDERS if provider.auth_kind == "api_key"
)
MODEL_PROVIDER_IDS: Tuple[str, ...] = ("lm_studio", "openrouter", *CLOUD_PROVIDER_IDS)
MAX_EFFORT_PROVIDER_IDS = frozenset(
    provider.provider_id for provider in CLOUD_PROVIDERS if provider.allows_max_effort
)

_BY_ID = {provider.provider_id: provider for provider in CLOUD_PROVIDERS}


def get_cloud_provider(provider_id: str) -> Optional[CloudProviderMetadata]:
    return _BY_ID.get(str(provider_id or "").strip())


def is_cloud_provider(provider_id: str) -> bool:
    return get_cloud_provider(provider_id) is not None


def is_api_key_subscription_provider(provider_id: str) -> bool:
    provider = get_cloud_provider(provider_id)
    return provider is not None and provider.auth_kind == "api_key"


def cloud_provider_label(provider_id: str) -> str:
    provider = get_cloud_provider(provider_id)
    return provider.label if provider else str(provider_id or "")


def cloud_provider_error_markers() -> Tuple[str, ...]:
    markers = {marker for provider in CLOUD_PROVIDERS for marker in provider.error_marker_set}
    return tuple(sorted(markers))


def cloud_provider_transient_markers() -> Tuple[str, ...]:
    markers = {marker for provider in CLOUD_PROVIDERS for marker in provider.transient_markers}
    return tuple(sorted(markers))
