"""Claude API profile (Anthropic's first-party, pay-as-you-go API).

Access method: an API key from the Claude Console, billed per token. This is not
a Claude.ai or Claude Code subscription login.

Wire format: the Messages API through the official Anthropic SDK (see
``claude_messages``). Models, context windows, output caps, and supported effort
levels come from the Claude Models API, which also authenticates a key without
spending tokens.
"""
from __future__ import annotations

import os

from backend.shared.subscription_providers.claude_messages import (
    CLAUDE_MODEL_TRAITS,
    ClaudeFamilyProfile,
)
from backend.shared.subscription_providers.metadata import get_cloud_provider
from backend.shared.subscription_providers.profile import (
    ApiKeyProviderSettings,
    EndpointRegion,
    KnownModel,
)

# Shown only when the Models API returns nothing; verified-organization-only
# models are left to the live list.
_FALLBACK_MODEL_IDS = (
    "claude-fable-5-2",
    "claude-opus-5-5",
    "claude-sonnet-5-5",
    "claude-haiku-5-5",
)

CLAUDE_API_SETTINGS = ApiKeyProviderSettings(
    metadata=get_cloud_provider("claude_api"),
    secret_name="claude_api_key",
    env_var_names=("MOTO_CLAUDE_API_KEY", "ANTHROPIC_API_KEY"),
    regions=(EndpointRegion("default", "Anthropic", "https://api.anthropic.com"),),
    base_url_env_var="MOTO_CLAUDE_API_BASE_URL",
    wire_api="anthropic_messages",
    model_list_api="anthropic_models",
    known_models=tuple(
        KnownModel(
            model_id,
            CLAUDE_MODEL_TRAITS[model_id].name,
            CLAUDE_MODEL_TRAITS[model_id].context_length,
            CLAUDE_MODEL_TRAITS[model_id].max_output_tokens,
            ("auto", *reversed(CLAUDE_MODEL_TRAITS[model_id].efforts)),
        )
        for model_id in _FALLBACK_MODEL_IDS
    ),
    default_model=os.getenv("MOTO_CLAUDE_API_DEFAULT_MODEL", "claude-opus-5-5"),
    console_url="https://platform.claude.com/settings/keys",
    # 529 is the Claude API's "overloaded" response.
    transient_status_codes=frozenset({408, 409, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529}),
    entitlement_with_fallback="repair",
    generic_error_fallback="permanent",
)


class ClaudeApiProfile(ClaudeFamilyProfile):
    auth_guidance = (
        "The Claude API did not accept this API key. Create or copy a key in the Claude Console "
        "(API keys), save it in OpenRouter/OAuth, and retry."
    )
    billing_guidance = (
        "The Claude API account has no usable credit or has reached its spend limit. Add credits or "
        "raise the limit in the Claude Console (Billing), then retry the workflow."
    )
    permission_guidance = (
        "This Claude API key is not allowed to make this request. Check the key's workspace and the "
        "organization's model access in the Claude Console, then retry."
    )
    model_guidance = (
        "The Claude API did not find this model for your organization. Select a model from the "
        "Anthropic Claude model list in Settings."
    )


CLAUDE_API_PROFILE = ClaudeApiProfile(CLAUDE_API_SETTINGS)
