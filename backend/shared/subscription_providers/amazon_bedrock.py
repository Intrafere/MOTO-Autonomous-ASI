"""Amazon Bedrock profile (the ``bedrock-mantle`` endpoint with a Bedrock API key).

Access method: an Amazon Bedrock API key (a bearer token) plus the AWS region it
is used in. Short-term keys last at most 12 hours and work only in the region
that created them; long-term keys last until their expiry date. MOTO does not
sign requests with AWS credentials.

Wire formats: Claude models use the Messages API at ``/anthropic/v1/messages``
through the official Anthropic SDK (see ``claude_messages``). Every other model
uses Bedrock's OpenAI-compatible streamed Chat Completions under ``/v1``; Bedrock
serves some of those under ``/openai/v1`` instead and names the mismatch in its
rejection, so that route is tried once. The model list (which also authenticates
the key) comes from the OpenAI-compatible ``/v1/models``.

Bedrock accepts the Messages API version the Anthropic SDK sends only for Claude
4.7 and later, so earlier Claude models are left out of the model list and are
refused with repair advice instead of being sent.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx

from backend.shared.subscription_providers.claude_messages import (
    CLAUDE_MODEL_TRAITS,
    ClaudeFamilyProfile,
    claude_family_id,
    claude_model_record,
    is_claude_model,
)
from backend.shared.subscription_providers.errors import SubscriptionProviderError
from backend.shared.subscription_providers.metadata import get_cloud_provider
from backend.shared.subscription_providers.profile import (
    ApiKeyProviderSettings,
    EndpointRegion,
    KnownModel,
    WireApi,
)

_MODEL_PREFIX = "anthropic."
_WRONG_ROUTE_MARKER = "supported on this route"
# Regions where Bedrock serves Claude on the bedrock-mantle endpoint.
_REGIONS = (
    ("us-east-1", "US East (N. Virginia)"),
    ("us-east-2", "US East (Ohio)"),
    ("us-west-1", "US West (N. California)"),
    ("us-west-2", "US West (Oregon)"),
    ("ca-central-1", "Canada (Central)"),
    ("ca-west-1", "Canada West (Calgary)"),
    ("sa-east-1", "South America (Sao Paulo)"),
    ("eu-central-1", "Europe (Frankfurt)"),
    ("eu-central-2", "Europe (Zurich)"),
    ("eu-north-1", "Europe (Stockholm)"),
    ("eu-south-1", "Europe (Milan)"),
    ("eu-south-2", "Europe (Spain)"),
    ("eu-west-1", "Europe (Ireland)"),
    ("eu-west-2", "Europe (London)"),
    ("eu-west-3", "Europe (Paris)"),
    ("il-central-1", "Israel (Tel Aviv)"),
    ("me-central-1", "Middle East (UAE)"),
    ("af-south-1", "Africa (Cape Town)"),
    ("ap-northeast-1", "Asia Pacific (Tokyo)"),
    ("ap-northeast-2", "Asia Pacific (Seoul)"),
    ("ap-northeast-3", "Asia Pacific (Osaka)"),
    ("ap-south-1", "Asia Pacific (Mumbai)"),
    ("ap-south-2", "Asia Pacific (Hyderabad)"),
    ("ap-southeast-1", "Asia Pacific (Singapore)"),
    ("ap-southeast-2", "Asia Pacific (Sydney)"),
    ("ap-southeast-3", "Asia Pacific (Jakarta)"),
    ("ap-southeast-4", "Asia Pacific (Melbourne)"),
)
# Claude models Bedrock documents on this endpoint, newest first.
_CLAUDE_MODEL_IDS = (
    "claude-opus-5-5",
    "claude-sonnet-5-5",
    "claude-haiku-5-5",
    "claude-fable-5-1",
    "claude-fable-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-5",
)
# Claude models older than 4.7, which Bedrock does not serve on the SDK's API version.
_PRE_4_7_FAMILIES = frozenset({
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-sonnet-4-5",
    "claude-opus-4-5",
    "claude-opus-4-1",
    "claude-sonnet-4",
    "claude-opus-4",
})
_PRE_4_7_PREFIXES = ("claude-3", "claude-2", "claude-instant")

AMAZON_BEDROCK_SETTINGS = ApiKeyProviderSettings(
    metadata=get_cloud_provider("amazon_bedrock"),
    secret_name="amazon_bedrock_api_key",
    env_var_names=("MOTO_AMAZON_BEDROCK_API_KEY", "AWS_BEARER_TOKEN_BEDROCK"),
    regions=tuple(
        EndpointRegion(region_id, f"{label} - {region_id}", f"https://bedrock-mantle.{region_id}.api.aws/v1")
        for region_id, label in _REGIONS
    ),
    region_selection="user",
    region_env_var_names=("MOTO_AMAZON_BEDROCK_REGION", "AWS_REGION", "AWS_DEFAULT_REGION"),
    base_url_env_var="MOTO_AMAZON_BEDROCK_BASE_URL",
    wire_api="anthropic_messages",
    known_models=tuple(
        KnownModel(
            f"{_MODEL_PREFIX}{model_id}",
            CLAUDE_MODEL_TRAITS[model_id].name,
            CLAUDE_MODEL_TRAITS[model_id].context_length,
            CLAUDE_MODEL_TRAITS[model_id].max_output_tokens,
            ("auto", *reversed(CLAUDE_MODEL_TRAITS[model_id].efforts)),
        )
        for model_id in _CLAUDE_MODEL_IDS
    ),
    default_model=os.getenv("MOTO_AMAZON_BEDROCK_DEFAULT_MODEL", f"{_MODEL_PREFIX}claude-opus-5-5"),
    console_url="https://console.aws.amazon.com/bedrock/home#/api-keys",
    transient_status_codes=frozenset({408, 409, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529}),
    entitlement_with_fallback="repair",
    generic_error_fallback="permanent",
)


class AmazonBedrockProfile(ClaudeFamilyProfile):
    auth_guidance = (
        "Amazon Bedrock did not accept this API key in the selected region. Short-term Bedrock API keys "
        "expire within 12 hours and work only in the region that created them. Create a new key in the "
        "Amazon Bedrock console (API keys), save it with its region in OpenRouter/OAuth, and retry."
    )
    billing_guidance = (
        "AWS reported a billing problem for this account. Fix it in the AWS Billing console, then retry "
        "the workflow."
    )
    permission_guidance = (
        "The AWS identity behind this Bedrock API key is not allowed to use this model. Enable access to "
        "the model in the Amazon Bedrock console and make sure the key's IAM policy allows Bedrock "
        "inference with an API key, then retry."
    )
    model_guidance = (
        "Amazon Bedrock does not serve this model for your account in the selected region. Select another "
        "model in Settings, or save the key again with a region that offers the model."
    )

    # --- wire selection ---------------------------------------------------------
    def wire_api_for_model(self, model: str) -> WireApi:
        return "anthropic_messages" if is_claude_model(model) else "chat_completions_stream"

    @staticmethod
    def _is_pre_4_7(model: str) -> bool:
        family = claude_family_id(model)
        return family in _PRE_4_7_FAMILIES or family.startswith(_PRE_4_7_PREFIXES)

    def unsupported_model_guidance(self, model: str) -> str:
        if not self._is_pre_4_7(model):
            return ""
        return (
            "MOTO reaches Claude on Amazon Bedrock through a Messages API version that Bedrock offers "
            "only for Claude 4.7 and later. Select a newer Claude model for this role, or use this model "
            "through the Anthropic Claude provider or OpenRouter."
        )

    def messages_base_url(self, base_url: str) -> str:
        root = base_url.rstrip("/")
        for suffix in ("/openai/v1", "/v1"):
            if root.endswith(suffix):
                root = root[: -len(suffix)]
                break
        return f"{root}/anthropic"

    def alternate_chat_base_url(self, base_url: str, response: httpx.Response) -> Optional[str]:
        if _WRONG_ROUTE_MARKER not in self._error_text(response).lower():
            return None
        root = base_url.rstrip("/")
        if root.endswith("/openai/v1"):
            return f"{root[: -len('/openai/v1')]}/v1"
        if root.endswith("/v1"):
            return f"{root[: -len('/v1')]}/openai/v1"
        return None

    def finalize_model_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        finalized = [
            record for record in super().finalize_model_records(records)
            if not self._is_pre_4_7(str(record.get("id") or ""))
        ]
        if any(is_claude_model(str(record.get("id") or "")) for record in finalized):
            return finalized
        # Bedrock's OpenAI-compatible list may leave Claude out; it is still served
        # on the Messages API route.
        claude = [
            claude_model_record(self.provider_id, f"{_MODEL_PREFIX}{model_id}", CLAUDE_MODEL_TRAITS[model_id])
            for model_id in _CLAUDE_MODEL_IDS
        ]
        return [*claude, *finalized]

    # --- request shaping (OpenAI-compatible models) -------------------------------
    def apply_stream_options(self, payload: Dict[str, Any]) -> None:
        # Token accounting needs the usage chunk, which is opt-in on this wire.
        payload["stream_options"] = {"include_usage": True}

    # --- error classification (OpenAI-compatible routes) ---------------------------
    def _error_fields(self, response: httpx.Response) -> tuple[str, str]:
        payload = self.response_payload(response)
        error_type = ""
        for item in self.iter_error_objects(payload):
            if not error_type and isinstance(item.get("type"), str):
                error_type = item["type"].strip()
            message = item.get("message")
            if isinstance(message, str) and message.strip():
                return error_type, message.strip()
        return error_type, response.text or ""

    def _error_text(self, response: httpx.Response) -> str:
        return self._error_fields(response)[1]

    def classify_error(self, response: httpx.Response) -> Optional[SubscriptionProviderError]:
        status = response.status_code
        if status == 404 and response.request.url.path.rstrip("/").endswith("/models"):
            # A missing model-list route says nothing about any one model.
            return None
        error_type, message = self._error_fields(response)
        return self.classify_status(status, error_type, message)


AMAZON_BEDROCK_PROFILE = AmazonBedrockProfile(AMAZON_BEDROCK_SETTINGS)
