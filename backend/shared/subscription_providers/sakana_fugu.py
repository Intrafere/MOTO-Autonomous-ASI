"""Sakana Fugu subscription API profile.

Fugu is OpenAI-compatible. Plain prompts use ``/responses``; tool turns and
endpoints that reject ``/responses`` (400/404/422) use ``/chat/completions``.
Usage limits and entitlement failures are reported through error codes.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import httpx

from backend.shared.log_redaction import redact_log_text
from backend.shared.openrouter_client import sanitize_provider_error_text
from backend.shared.subscription_providers.errors import (
    SubscriptionEntitlementError,
    SubscriptionProviderError,
    SubscriptionUsageLimitError,
)
from backend.shared.subscription_providers.metadata import get_cloud_provider
from backend.shared.subscription_providers.profile import (
    ApiKeyProviderProfile,
    ApiKeyProviderSettings,
    EndpointRegion,
    KnownModel,
)

_EFFORTS = ("auto", "xhigh", "high")

SAKANA_FUGU_SETTINGS = ApiKeyProviderSettings(
    metadata=get_cloud_provider("sakana_fugu"),
    secret_name="sakana_fugu_api_key",
    env_var_names=("MOTO_SAKANA_FUGU_API_KEY", "SAKANA_API_KEY", "FUGU_API_KEY"),
    regions=(EndpointRegion("default", "Sakana", "https://api.sakana.ai/v1"),),
    base_url_env_var="MOTO_SAKANA_FUGU_BASE_URL",
    wire_api="responses_then_chat",
    known_models=(
        KnownModel("fugu", "Fugu", 1_000_000, 100_000, _EFFORTS),
        KnownModel("fugu-ultra", "Fugu Ultra", 1_000_000, 100_000, _EFFORTS),
        KnownModel("fugu-ultra-20260615", "Fugu Ultra 20260615", 1_000_000, 100_000, _EFFORTS),
    ),
    default_model=os.getenv("MOTO_SAKANA_FUGU_DEFAULT_MODEL", "fugu"),
    console_url="https://console.sakana.ai/",
    entitlement_with_fallback="repair",
    generic_error_fallback="permanent",
)


class SakanaFuguProfile(ApiKeyProviderProfile):
    USAGE_LIMIT_CODES = frozenset({"usage_limit_reached"})
    HARD_ENTITLEMENT_CODES = frozenset({
        "entitlement_required",
        "insufficient_entitlement",
        "insufficient_quota",
        "model_not_entitled",
        "not_entitled",
        "plan_not_eligible",
        "subscription_required",
    })

    def normalize_reasoning_effort(self, reasoning_effort: Optional[str], *, model: str) -> Optional[str]:
        effort = (reasoning_effort or "").strip().lower()
        if not effort or effort == "none":
            return None
        if effort in {"auto", "xhigh", "max", "maximum", "highest"}:
            return "xhigh"
        if effort == "high":
            return "high"
        return None

    def classify_error(self, response: httpx.Response) -> Optional[SubscriptionProviderError]:
        payload = self.response_payload(response)
        matched: Optional[Dict[str, Any]] = None
        matched_code = ""
        entitlement = False
        for item in self.iter_error_objects(payload):
            codes = {
                str(item.get(field) or "").strip().lower()
                for field in ("type", "code", "error_code")
                if str(item.get(field) or "").strip()
            }
            usage_code = next((code for code in codes if code in self.USAGE_LIMIT_CODES), "")
            if usage_code:
                matched = item
                matched_code = usage_code
                break
            entitlement_code = next((code for code in codes if code in self.HARD_ENTITLEMENT_CODES), "")
            if entitlement_code and matched is None:
                matched = item
                matched_code = entitlement_code
                entitlement = True

        label = self.label
        if matched_code in self.USAGE_LIMIT_CODES:
            resets_at = self.coerce_positive_int(matched.get("resets_at") if matched else None)
            resets_in_seconds = self.coerce_positive_int(matched.get("resets_in_seconds") if matched else None)
            header_resets_at, header_resets_in = self.header_reset_values(response)
            resets_at = resets_at or header_resets_at
            resets_in_seconds = resets_in_seconds or header_resets_in
            now = int(time.time())
            if resets_at is None and resets_in_seconds is not None:
                resets_at = now + resets_in_seconds
            elif resets_in_seconds is None and resets_at is not None:
                resets_in_seconds = max(1, resets_at - now)
            message = redact_log_text(
                sanitize_provider_error_text(
                    str((matched or {}).get("message") or f"The {label} usage limit has been reached.")
                )
            )
            if resets_at is None:
                return SubscriptionEntitlementError(
                    message,
                    provider=self.provider_id,
                    provider_label=label,
                    error_code=matched_code,
                    status_code=response.status_code,
                )
            return SubscriptionUsageLimitError(
                message,
                provider=self.provider_id,
                provider_label=label,
                resets_at=resets_at,
                resets_in_seconds=resets_in_seconds,
                plan_type=str((matched or {}).get("plan_type") or "").strip(),
                status_code=response.status_code,
                error_code=matched_code,
            )

        if entitlement and matched is not None:
            message = redact_log_text(
                sanitize_provider_error_text(
                    str(matched.get("message") or f"The {label} subscription is not entitled to this request.")
                )
            )
            return SubscriptionEntitlementError(
                message,
                provider=self.provider_id,
                provider_label=label,
                error_code=matched_code,
                status_code=response.status_code,
            )
        return None


SAKANA_FUGU_PROFILE = SakanaFuguProfile(SAKANA_FUGU_SETTINGS)
