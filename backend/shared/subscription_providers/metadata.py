"""Single source of truth for MOTO's desktop role providers beyond LM Studio and OpenRouter.

Every list of provider ids, labels, capability flags, and provider error markers
is derived from ``CLOUD_PROVIDERS``. This module has no MOTO imports so models,
route identity, and error helpers can depend on it without import cycles.

Most entries are cloud subscriptions. A ``local_server`` entry (vLLM) is a
user-run OpenAI-compatible server that shares the same registry plumbing.

Adding a provider: append one ``CloudProviderMetadata`` entry here, then (for an
API-key subscription or a local server) add its profile module and register it
in ``registry``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

AuthKind = Literal["oauth_pkce", "api_key", "local_server"]


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
    CloudProviderMetadata(
        provider_id="claude_api",
        label="Anthropic Claude",
        short_label="Claude",
        auth_kind="api_key",
        capability_flag="claude_api_available",
        api_slug="claude-api",
        allows_max_effort=True,
        error_markers=("anthropic",),
    ),
    CloudProviderMetadata(
        provider_id="amazon_bedrock",
        label="Amazon Bedrock",
        short_label="Bedrock",
        auth_kind="api_key",
        capability_flag="amazon_bedrock_available",
        api_slug="amazon-bedrock",
        allows_max_effort=True,
        error_markers=("bedrock",),
    ),
    CloudProviderMetadata(
        provider_id="vllm",
        label="vLLM",
        short_label="vLLM",
        auth_kind="local_server",
        capability_flag="vllm_available",
        api_slug="vllm",
    ),
)

CLOUD_PROVIDER_IDS: Tuple[str, ...] = tuple(provider.provider_id for provider in CLOUD_PROVIDERS)
API_KEY_SUBSCRIPTION_PROVIDER_IDS: Tuple[str, ...] = tuple(
    provider.provider_id for provider in CLOUD_PROVIDERS if provider.auth_kind == "api_key"
)
LOCAL_SERVER_PROVIDER_IDS: Tuple[str, ...] = tuple(
    provider.provider_id for provider in CLOUD_PROVIDERS if provider.auth_kind == "local_server"
)
# Providers served by one registered client (see ``registry``).
REGISTRY_CLIENT_PROVIDER_IDS: Tuple[str, ...] = (*API_KEY_SUBSCRIPTION_PROVIDER_IDS, *LOCAL_SERVER_PROVIDER_IDS)
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


def is_local_server_provider(provider_id: str) -> bool:
    provider = get_cloud_provider(provider_id)
    return provider is not None and provider.auth_kind == "local_server"


LOCAL_FALLBACK_SEPARATOR = "::"


def local_fallback_id(provider_id: str, model_id: str) -> str:
    """Return the fallback value that names ``model_id`` on a local provider."""
    if provider_id == "lm_studio":
        return model_id
    return f"{provider_id}{LOCAL_FALLBACK_SEPARATOR}{model_id}"


def split_local_fallback(fallback_id: object) -> Tuple[str, str]:
    """Return ``(provider_id, model_id)`` for a role's local fallback value.

    A plain value is an LM Studio model. ``<local server id>::<model>`` (for
    example ``vllm::Qwen/Qwen2.5``) names a model on that local server.
    """
    text = str(fallback_id or "").strip()
    for provider_id in LOCAL_SERVER_PROVIDER_IDS:
        prefix = f"{provider_id}{LOCAL_FALLBACK_SEPARATOR}"
        if text.startswith(prefix) and len(text) > len(prefix):
            return provider_id, text[len(prefix):]
    return "lm_studio", text


def cloud_provider_label(provider_id: str) -> str:
    provider = get_cloud_provider(provider_id)
    return provider.label if provider else str(provider_id or "")


def cloud_provider_error_markers() -> Tuple[str, ...]:
    markers = {marker for provider in CLOUD_PROVIDERS for marker in provider.error_marker_set}
    return tuple(sorted(markers))


def cloud_provider_transient_markers() -> Tuple[str, ...]:
    markers = {marker for provider in CLOUD_PROVIDERS for marker in provider.transient_markers}
    return tuple(sorted(markers))
