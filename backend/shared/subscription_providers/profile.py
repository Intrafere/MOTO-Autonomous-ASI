"""Provider profiles for API-key subscription clients.

A profile is everything provider-specific about an OpenAI-compatible
subscription API: endpoints, key storage names, known model metadata, request
shaping, and error classification. ``SubscriptionApiKeyClient`` supplies the
shared transport, retry, key-lifecycle, and response-normalization logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
import json
import time
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple

import httpx

from backend.shared.subscription_providers.errors import SubscriptionProviderError
from backend.shared.subscription_providers.metadata import CloudProviderMetadata

WireApi = Literal["responses_then_chat", "chat_completions_stream"]


@dataclass(frozen=True)
class EndpointRegion:
    region_id: str
    label: str
    base_url: str


@dataclass(frozen=True)
class KnownModel:
    model_id: str
    name: str
    # None means the provider does not publish the value; MOTO never invents one.
    context_length: Optional[int] = None
    max_output_tokens: Optional[int] = None
    reasoning_efforts: Tuple[str, ...] = ()

    def to_record(self, provider_id: str) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "id": self.model_id,
            "name": self.name,
            "pricing": {"prompt": "subscription", "completion": "subscription"},
            "provider_metadata": {
                "source": provider_id,
                "supports_reasoning_effort": bool(self.reasoning_efforts),
            },
        }
        if self.reasoning_efforts:
            record["provider_metadata"]["reasoning_efforts"] = list(self.reasoning_efforts)
        if self.context_length:
            record["context_length"] = self.context_length
        if self.max_output_tokens:
            record["max_output_tokens"] = self.max_output_tokens
        return record


@dataclass(frozen=True)
class ApiKeyProviderSettings:
    metadata: CloudProviderMetadata
    secret_name: str
    # Environment overrides, highest precedence first (MOTO_* names first).
    env_var_names: Tuple[str, ...]
    regions: Tuple[EndpointRegion, ...]
    wire_api: WireApi
    known_models: Tuple[KnownModel, ...]
    default_model: str
    console_url: str
    base_url_env_var: str = ""
    # GET paths tried in order to validate a key without spending model quota.
    validation_paths: Tuple[str, ...] = ("/models",)
    supports_usage: bool = False
    invalid_key_message: str = ""
    transient_status_codes: frozenset[int] = field(
        default_factory=lambda: frozenset({408, 409, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524})
    )
    max_retries: int = 4
    retry_delay_seconds: float = 2.0
    retry_max_delay_seconds: float = 30.0
    retry_jitter_ratio: float = 0.2
    # Routing policy switches that preserve each provider's established behavior.
    # "repair": an entitlement failure requires operator repair even with a fallback.
    entitlement_with_fallback: Literal["repair", "fallback"] = "repair"
    # "permanent": any non-truncation failure with a fallback moves the role to
    # LM Studio until reset; "non_transient_only" keeps transient failures call-only.
    generic_error_fallback: Literal["permanent", "non_transient_only"] = "permanent"

    @property
    def provider_id(self) -> str:
        return self.metadata.provider_id

    @property
    def label(self) -> str:
        return self.metadata.label


class ApiKeyProviderProfile:
    """Provider-specific behavior for ``SubscriptionApiKeyClient``.

    Subclasses override the hooks they need; the defaults describe a plain
    OpenAI-compatible API.
    """

    def __init__(self, settings: ApiKeyProviderSettings) -> None:
        self.settings = settings

    # --- identity -------------------------------------------------------------
    @property
    def provider_id(self) -> str:
        return self.settings.provider_id

    @property
    def label(self) -> str:
        return self.settings.label

    # --- request shaping --------------------------------------------------------
    def normalize_reasoning_effort(self, reasoning_effort: Optional[str], *, model: str) -> Optional[str]:
        """Return the provider effort value to send, or None to omit it."""
        return None

    def apply_reasoning(self, payload: Dict[str, Any], effort: Optional[str], *, model: str, wire: str) -> None:
        if effort:
            payload["reasoning"] = {"effort": effort}

    def apply_temperature(self, payload: Dict[str, Any], temperature: float, *, model: str) -> None:
        payload["temperature"] = temperature

    def prepare_chat_messages(
        self,
        messages: List[Dict[str, Any]],
        payload: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Return the outbound Chat Completions messages for one request."""
        return messages

    def observe_chat_message(self, message: Dict[str, Any]) -> None:
        """Inspect a normalized assistant message returned by the provider."""

    # --- error classification -----------------------------------------------------
    def classify_error(self, response: httpx.Response) -> Optional[SubscriptionProviderError]:
        """Return a typed, non-retried error for a definitive provider rejection."""
        return None

    def is_transient_response(self, response: httpx.Response) -> bool:
        """Return whether an unclassified error response should be retried."""
        return response.status_code in self.settings.transient_status_codes

    async def resolve_usage_limit(
        self,
        error: SubscriptionProviderError,
        client: "Any",
    ) -> SubscriptionProviderError:
        """Give a profile a chance to attach credible reset metadata to an error."""
        return error

    # --- usage ------------------------------------------------------------------
    def parse_usage(self, payload: Any) -> Dict[str, Any]:
        raise NotImplementedError

    # --- shared parsing helpers ---------------------------------------------------
    @staticmethod
    def coerce_positive_int(value: Any) -> Optional[int]:
        try:
            parsed = int(float(value))
        except (TypeError, ValueError, OverflowError):
            return None
        return parsed if parsed > 0 else None

    @staticmethod
    def response_payload(response: httpx.Response) -> Any:
        try:
            return response.json()
        except (json.JSONDecodeError, ValueError, TypeError, UnicodeError):
            return None

    @staticmethod
    def iter_error_objects(value: Any, *, max_depth: int = 8, max_nodes: int = 256) -> Iterator[Dict[str, Any]]:
        """Yield nested OpenAI-compatible error objects without trusting their shape."""
        seen: set[int] = set()
        visited = 0

        def visit(item: Any, depth: int) -> Iterator[Dict[str, Any]]:
            nonlocal visited
            if depth > max_depth or not isinstance(item, (dict, list)):
                return
            identity = id(item)
            if identity in seen or visited >= max_nodes:
                return
            seen.add(identity)
            visited += 1
            if isinstance(item, dict):
                yield item
                for nested in item.values():
                    yield from visit(nested, depth + 1)
            else:
                for nested in item:
                    yield from visit(nested, depth + 1)

        yield from visit(value, 0)

    @classmethod
    def header_reset_values(cls, response: httpx.Response) -> Tuple[Optional[int], Optional[int]]:
        """Return (resets_at, resets_in_seconds) from Retry-After and reset headers."""
        now = int(time.time())
        resets_at: Optional[int] = None
        resets_in_seconds: Optional[int] = None

        retry_after = response.headers.get("Retry-After")
        if retry_after:
            seconds = cls.coerce_positive_int(retry_after)
            if seconds is None:
                try:
                    seconds = max(1, int(parsedate_to_datetime(retry_after).timestamp()) - now)
                except (TypeError, ValueError, OverflowError):
                    seconds = None
            if seconds is not None:
                resets_in_seconds = seconds
                resets_at = now + seconds

        for name in (
            "x-ratelimit-reset",
            "x-ratelimit-reset-requests",
            "x-usage-limit-reset",
            "x-usage-reset",
        ):
            parsed = cls.coerce_positive_int(response.headers.get(name))
            if parsed is None:
                continue
            header_resets_at = parsed if parsed > now else now + parsed
            header_seconds = max(1, header_resets_at - now)
            if resets_at is None or header_resets_at < resets_at:
                resets_at = header_resets_at
                resets_in_seconds = header_seconds
        return resets_at, resets_in_seconds
