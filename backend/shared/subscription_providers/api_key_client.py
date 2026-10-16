"""Shared client for OpenAI-compatible subscription APIs that use an API key.

One instance serves one provider profile (for example Sakana Fugu or Kimi Code).
The client owns the key lifecycle (environment override, then the desktop
keyring), endpoint-region detection, bounded transient retries through the
shared stop-aware wait seam, and normalization of every response to the Chat
Completions shape the rest of MOTO consumes.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
import random
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import httpx

from backend.shared.log_redaction import redact_log_text
from backend.shared.openrouter_client import sanitize_provider_error_text
from backend.shared.provider_recovery.runtime import assured_retry_wait
from backend.shared.secret_store import (
    SecretStoreError,
    clear_subscription_api_key,
    load_subscription_api_key,
    store_subscription_api_key,
)
from backend.shared.subscription_providers.errors import (
    SubscriptionAuthError,
    SubscriptionProviderError,
    SubscriptionRequestError,
)
from backend.shared.subscription_providers.profile import ApiKeyProviderProfile, EndpointRegion

logger = logging.getLogger(__name__)

RetryCallback = Callable[[Dict[str, Any]], Awaitable[None]]
MOTO_USER_AGENT_URL = "https://intrafere.com/moto-autonomous-home-ai/"
_TOOL_TURN_ROLES = {"tool"}
_MAX_REMEMBERED_SECRETS = 8


def moto_user_agent() -> str:
    """Return MOTO's honest client identifier for provider requests."""
    from backend.shared.build_info import get_build_info

    return f"MOTO-Autonomous-ASI/{get_build_info().version} (+{MOTO_USER_AGENT_URL})"


@dataclass(frozen=True)
class CandidateValidation:
    region_id: str
    models: List[Dict[str, Any]]


@dataclass
class _ChatStreamResult:
    terminal: bool
    data: Dict[str, Any]


class SubscriptionApiKeyClient:
    """Client for one OpenAI-compatible API-key subscription provider."""

    def __init__(self, profile: ApiKeyProviderProfile) -> None:
        self.profile = profile
        self.settings = profile.settings
        self._api_key: Optional[str] = self._read_env_key()
        self._region_id: Optional[str] = None
        self._persisted_region_checked = False
        self._region_detection_attempted = False
        # Keys this client has sent; scrubbed from every provider-supplied error text.
        self._secrets_seen: Dict[str, None] = {}
        self.client = httpx.AsyncClient(
            timeout=None,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50, keepalive_expiry=30.0),
        )

    # --- identity ---------------------------------------------------------------
    @property
    def provider_id(self) -> str:
        return self.settings.provider_id

    @property
    def provider_label(self) -> str:
        return self.settings.label

    # --- key lifecycle ------------------------------------------------------------
    def _read_env_key(self) -> Optional[str]:
        for name in self.settings.env_var_names:
            value = (os.getenv(name) or "").strip()
            if value:
                return value
        return None

    def _load_api_key(self) -> Optional[str]:
        if self._api_key:
            return self._api_key
        key = load_subscription_api_key(self.settings.secret_name)
        if key:
            self._api_key = key
        return self._api_key

    def set_api_key(self, api_key: str, *, region_id: str = "", persist: bool = True) -> None:
        key = (api_key or "").strip()
        if not key:
            raise ValueError(f"{self.provider_label} API key is required.")
        if persist:
            store_subscription_api_key(self.settings.secret_name, key)
            if load_subscription_api_key(self.settings.secret_name) != key:
                raise SecretStoreError(f"Stored {self.provider_label} credential could not be verified")
        self._api_key = key
        if self._is_multi_region():
            self._region_id = region_id or None
            self._persisted_region_checked = True
            self._region_detection_attempted = bool(region_id)
            if persist:
                self._persist_region(region_id)

    async def clear_api_key(self) -> None:
        clear_subscription_api_key(self.settings.secret_name)
        self._api_key = None
        self._region_id = None
        self._persisted_region_checked = True
        self._region_detection_attempted = False
        if self._is_multi_region():
            self._persist_region("")

    async def status(self) -> Dict[str, Any]:
        configured = bool(self._load_api_key())
        status: Dict[str, Any] = {
            "configured": configured,
            "provider": self.provider_id,
            "updated_at": int(time.time()) if configured else None,
        }
        if self._is_multi_region():
            region = self._known_region()
            status["region"] = region.region_id if region else None
            status["region_label"] = region.label if region else None
        if self.settings.supports_usage:
            status["supports_usage"] = True
        return status

    # --- endpoint regions -----------------------------------------------------------
    def _is_multi_region(self) -> bool:
        return len(self.settings.regions) > 1 and not self._base_url_override()

    def _base_url_override(self) -> str:
        name = self.settings.base_url_env_var
        return (os.getenv(name) or "").strip().rstrip("/") if name else ""

    def _region_by_id(self, region_id: str) -> Optional[EndpointRegion]:
        return next((region for region in self.settings.regions if region.region_id == region_id), None)

    def _persist_region(self, region_id: str) -> None:
        from backend.shared.runtime_settings import RuntimeSettingsError, set_subscription_endpoint_region

        try:
            set_subscription_endpoint_region(self.provider_id, region_id)
        except RuntimeSettingsError:
            # The key is committed; the region is re-detected on next use.
            logger.warning("Could not persist the %s endpoint region; it will be re-detected.", self.provider_label)

    def _known_region(self) -> Optional[EndpointRegion]:
        if not self._is_multi_region():
            return self.settings.regions[0]
        if self._region_id:
            return self._region_by_id(self._region_id)
        if self._persisted_region_checked:
            return None
        from backend.shared.runtime_settings import get_subscription_endpoint_region

        self._persisted_region_checked = True
        persisted = self._region_by_id(get_subscription_endpoint_region(self.provider_id))
        if persisted is not None:
            self._region_id = persisted.region_id
        return persisted

    def _candidate_regions(self) -> Tuple[EndpointRegion, ...]:
        override = self._base_url_override()
        if override:
            return (EndpointRegion("override", "Custom endpoint", override),)
        return self.settings.regions

    @property
    def base_url(self) -> str:
        override = self._base_url_override()
        if override:
            return override
        region = self._known_region() or self.settings.regions[0]
        return region.base_url.rstrip("/")

    async def _ensure_region(self) -> None:
        """Detect the endpoint region once when a key exists but no region is recorded."""
        if not self._is_multi_region() or self._known_region() is not None or self._region_detection_attempted:
            return
        key = self._load_api_key()
        if not key:
            return
        self._region_detection_attempted = True
        try:
            validation = await self.validate_candidate(key)
        except SubscriptionProviderError:
            return
        if validation.region_id != "override":
            self._region_id = validation.region_id
            self._persist_region(validation.region_id)

    # --- headers and retries ----------------------------------------------------------
    def _remember_secret(self, key: str) -> None:
        self._secrets_seen.pop(key, None)
        self._secrets_seen[key] = None
        while len(self._secrets_seen) > _MAX_REMEMBERED_SECRETS:
            self._secrets_seen.pop(next(iter(self._secrets_seen)))

    def _scrub(self, text: str) -> str:
        for secret in self._secrets_seen:
            text = text.replace(secret, "[redacted]")
        return text

    def _detail(self, text: str) -> str:
        return self._scrub(sanitize_provider_error_text(text))

    def _headers_for_key(self, key: Optional[str]) -> Dict[str, str]:
        if not key:
            raise SubscriptionAuthError(
                f"{self.provider_label} API key is not configured.",
                provider=self.provider_id,
                provider_label=self.provider_label,
            )
        self._remember_secret(key)
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": moto_user_agent(),
        }

    def _headers(self) -> Dict[str, str]:
        return self._headers_for_key(self._load_api_key())

    def _retry_delay(self, attempt: int) -> float:
        settings = self.settings
        base = min(settings.retry_delay_seconds * (2 ** attempt), settings.retry_max_delay_seconds)
        jitter = random.uniform(0.0, base * settings.retry_jitter_ratio)
        return min(base + jitter, settings.retry_max_delay_seconds)

    def _retry_after_delay(self, response: httpx.Response, attempt: int) -> float:
        _resets_at, resets_in_seconds = self.profile.header_reset_values(response)
        if resets_in_seconds is not None:
            base = min(float(resets_in_seconds), self.settings.retry_max_delay_seconds)
            jitter = random.uniform(0.0, base * self.settings.retry_jitter_ratio)
            return min(base + jitter, self.settings.retry_max_delay_seconds)
        return self._retry_delay(attempt)

    async def _wait_before_retry(
        self,
        *,
        attempt: int,
        delay: float,
        reason: str,
        error_summary: str,
        retry_model: str,
        retry_callback: Optional[RetryCallback],
    ) -> None:
        info = {
            "provider": self.provider_id,
            "provider_label": self.provider_label,
            "model": retry_model,
            "retry_attempt": attempt + 1,
            "max_attempts": self.settings.max_retries,
            "retry_after_seconds": delay,
            "reason": reason,
            "error_summary": error_summary,
        }
        if retry_callback is not None:
            await retry_callback(info)
        else:
            await assured_retry_wait(info, role_id=f"{self.provider_id}_direct")

    def _request_failed(self, detail: str, *, status_code: Optional[int] = None) -> SubscriptionRequestError:
        return SubscriptionRequestError(
            f"{self.provider_label} connection failed after retries: {detail}",
            provider=self.provider_id,
            provider_label=self.provider_label,
            status_code=status_code,
        )

    async def _classified(self, response: httpx.Response) -> Optional[SubscriptionProviderError]:
        classified = self.profile.classify_error(response)
        if classified is None:
            return None
        resolved = await self.profile.resolve_usage_limit(classified, self)
        resolved.args = (self._scrub(str(resolved)),)
        return resolved

    def _http_error(self, response: httpx.Response, action: str) -> SubscriptionProviderError:
        message = self._detail(response.text)
        error_cls = SubscriptionAuthError if response.status_code in {401, 403} else SubscriptionRequestError
        return error_cls(
            f"{self.provider_label} {action} failed: {message}",
            provider=self.provider_id,
            provider_label=self.provider_label,
            status_code=response.status_code,
        )

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        *,
        retry_callback: Optional[RetryCallback] = None,
        retry_model: str = "",
        classify: bool = True,
        **kwargs: Any,
    ) -> httpx.Response:
        max_retries = self.settings.max_retries
        for attempt in range(max_retries):
            try:
                response = await self.client.request(method, url, **kwargs)
            except httpx.TransportError as exc:
                detail = self._detail(str(exc) or repr(exc))
                if attempt < max_retries - 1:
                    await self._wait_before_retry(
                        attempt=attempt,
                        delay=self._retry_delay(attempt),
                        reason="connection_error",
                        error_summary=f"[{type(exc).__name__}] {detail}",
                        retry_model=retry_model,
                        retry_callback=retry_callback,
                    )
                    continue
                raise self._request_failed(detail) from exc
            if response.status_code >= 400:
                classified = await self._classified(response) if classify else None
                if classified is not None:
                    raise classified
                if self.profile.is_transient_response(response):
                    detail = self._detail(response.text)
                    if attempt < max_retries - 1:
                        await self._wait_before_retry(
                            attempt=attempt,
                            delay=self._retry_after_delay(response, attempt),
                            reason="transient_http_error",
                            error_summary=detail,
                            retry_model=retry_model,
                            retry_callback=retry_callback,
                        )
                        continue
                    raise self._request_failed(f"HTTP {response.status_code}: {detail}", status_code=response.status_code)
            return response
        raise self._request_failed("no attempts remained")

    # --- models, validation and usage ---------------------------------------------------
    def known_model_records(self) -> List[Dict[str, Any]]:
        return [model.to_record(self.provider_id) for model in self.settings.known_models]

    def _merge_model_records(self, records: List[Any]) -> List[Dict[str, Any]]:
        known_by_id = {model["id"]: model for model in self.known_model_records()}
        models: List[Dict[str, Any]] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            model_id = str(record.get("id") or "").strip()
            if not model_id:
                continue
            base = known_by_id.get(model_id, {})
            merged = {
                **base,
                "id": model_id,
                "name": base.get("name") or model_id,
                "provider_metadata": {"source": self.provider_id, **base.get("provider_metadata", {})},
            }
            for field_name in ("context_length", "max_output_tokens"):
                if field_name not in merged:
                    value = self.profile.coerce_positive_int(record.get(field_name))
                    if value:
                        merged[field_name] = value
            models.append(merged)
        return models

    async def _validate_without_model_list(self, base_url: str, headers: Dict[str, str]) -> None:
        """Validate a key through the profile's non-quota paths when /models is absent."""
        for path in self.settings.validation_paths:
            if path == "/models":
                continue
            response = await self._request_with_retry("GET", f"{base_url}{path}", headers=headers)
            if response.status_code < 400:
                return
            if response.status_code == 404:
                continue
            raise self._http_error(response, "key validation")
        raise SubscriptionRequestError(
            f"{self.provider_label} key validation failed: no validation endpoint responded.",
            provider=self.provider_id,
            provider_label=self.provider_label,
        )

    async def _list_models_at(self, base_url: str, api_key: Optional[str]) -> List[Dict[str, Any]]:
        try:
            headers = self._headers_for_key(api_key)
            response = await self._request_with_retry("GET", f"{base_url}/models", headers=headers)
            if response.status_code == 404 and len(self.settings.validation_paths) > 1:
                await self._validate_without_model_list(base_url, headers)
                return self.known_model_records()
            if response.status_code >= 400:
                raise self._http_error(response, "model list")
            payload = response.json()
            records = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(records, list):
                return self.known_model_records()
            return self._merge_model_records(records) or self.known_model_records()
        except SubscriptionProviderError:
            raise
        except Exception as exc:
            raise SubscriptionRequestError(
                f"{self.provider_label} model list failed before a valid model list was returned.",
                provider=self.provider_id,
                provider_label=self.provider_label,
            ) from exc

    async def list_models(self, api_key: Optional[str] = None) -> List[Dict[str, Any]]:
        if api_key is not None:
            return await self._list_models_at(self.base_url, api_key.strip())
        await self._ensure_region()
        return await self._list_models_at(self.base_url, self._load_api_key())

    async def validate_candidate(self, api_key: str) -> CandidateValidation:
        """Authenticate a candidate key without changing any stored or in-memory state."""
        key = (api_key or "").strip()
        if not key:
            raise ValueError(f"{self.provider_label} API key is required.")
        regions = self._candidate_regions()
        last_auth_error: Optional[SubscriptionAuthError] = None
        for region in regions:
            try:
                models = await self._list_models_at(region.base_url.rstrip("/"), key)
            except SubscriptionAuthError as exc:
                last_auth_error = exc
                continue
            return CandidateValidation(region_id=region.region_id, models=models)
        if len(regions) > 1 and self.settings.invalid_key_message:
            raise SubscriptionAuthError(
                self.settings.invalid_key_message,
                provider=self.provider_id,
                provider_label=self.provider_label,
                status_code=getattr(last_auth_error, "status_code", None),
            )
        assert last_auth_error is not None
        raise last_auth_error

    async def usage(self) -> Dict[str, Any]:
        if not self.settings.supports_usage:
            raise SubscriptionRequestError(
                f"{self.provider_label} does not report subscription usage.",
                provider=self.provider_id,
                provider_label=self.provider_label,
            )
        await self._ensure_region()
        # Unclassified: usage is also how a usage-limit rejection learns its reset time.
        response = await self._request_with_retry(
            "GET",
            f"{self.base_url}/usages",
            headers=self._headers(),
            classify=False,
        )
        if response.status_code >= 400:
            raise self._http_error(response, "usage query")
        try:
            payload = response.json()
        except ValueError as exc:
            raise SubscriptionRequestError(
                f"{self.provider_label} usage query returned an invalid response.",
                provider=self.provider_id,
                provider_label=self.provider_label,
            ) from exc
        return self.profile.parse_usage(payload)

    # --- response normalization -------------------------------------------------------------
    @staticmethod
    def _chat_usage(usage: Dict[str, Any]) -> Dict[str, Any]:
        prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
        completion_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
        total_tokens = usage.get("total_tokens")
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = int(prompt_tokens) + int(completion_tokens)
        normalized = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        if usage.get("input_tokens_details"):
            normalized["prompt_tokens_details"] = usage["input_tokens_details"]
        if usage.get("output_tokens_details"):
            normalized["completion_tokens_details"] = usage["output_tokens_details"]
        if usage.get("completion_tokens_details"):
            normalized["completion_tokens_details"] = usage["completion_tokens_details"]
        return {key: value for key, value in normalized.items() if value is not None}

    def _response_id(self, data: Dict[str, Any]) -> str:
        return str(data.get("id") or f"{self.provider_id.replace('_', '-')}-{int(time.time() * 1000)}")

    @staticmethod
    def _extract_responses_text(data: Dict[str, Any]) -> str:
        if isinstance(data.get("output_text"), str):
            return data["output_text"]
        parts: List[str] = []
        for item in data.get("output") or []:
            if not isinstance(item, dict):
                continue
            for content in item.get("content") or []:
                if isinstance(content, dict) and isinstance(content.get("text"), str):
                    parts.append(content["text"])
        return "".join(parts)

    @staticmethod
    def _messages_to_responses_payload(messages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        instructions: List[str] = []
        inputs: List[Dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role") or "user")
            content = message.get("content", "")
            if role in {"system", "developer"}:
                instructions.append(content if isinstance(content, str) else str(content))
                continue
            responses_role = "assistant" if role == "assistant" else "user"
            inputs.append({"role": responses_role, "content": content})
        return "\n\n".join(part for part in instructions if part), inputs

    def _normalize_responses_to_chat(self, data: Dict[str, Any], model: str) -> Dict[str, Any]:
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        status = data.get("status")
        return {
            "id": self._response_id(data),
            "object": "chat.completion",
            "created": data.get("created_at") or data.get("created") or int(time.time()),
            "model": data.get("model") or model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": self._extract_responses_text(data)},
                    "finish_reason": status if status not in {None, "completed"} else "stop",
                }
            ],
            "usage": self._chat_usage(usage),
            "_moto_wire_api": "responses",
        }

    @staticmethod
    def _messages_need_chat_completions(
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
    ) -> bool:
        if tools:
            return True
        return any(
            str(message.get("role") or "") in _TOOL_TURN_ROLES or message.get("tool_calls")
            for message in messages
        )

    def _normalize_chat_completion(self, data: Dict[str, Any], model: str, wire_api: str) -> Dict[str, Any]:
        choices = data.get("choices") if isinstance(data.get("choices"), list) else []
        if not choices:
            raise SubscriptionRequestError(
                f"{self.provider_label} chat completion returned no choices.",
                provider=self.provider_id,
                provider_label=self.provider_label,
            )
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        first_message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(first_message, dict):
            self.profile.observe_chat_message(first_message)
        return {
            "id": self._response_id(data),
            "object": "chat.completion",
            "created": data.get("created") or int(time.time()),
            "model": data.get("model") or model,
            "choices": choices,
            "usage": self._chat_usage(usage),
            "_moto_wire_api": wire_api,
        }

    # --- request building ---------------------------------------------------------------------
    def _chat_payload(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: Optional[int],
        response_format: Optional[Dict[str, Any]],
        reasoning_effort: Optional[str],
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[Any],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"model": model}
        self.profile.apply_temperature(payload, temperature, model=model)
        if max_tokens:
            payload["max_tokens"] = int(max_tokens)
        effort = self.profile.normalize_reasoning_effort(reasoning_effort, model=model)
        self.profile.apply_reasoning(payload, effort, model=model, wire="chat")
        if response_format:
            payload["response_format"] = response_format
        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        payload["messages"] = self.profile.prepare_chat_messages(messages, payload)
        return payload

    async def _generate_via_chat_completions(
        self,
        *,
        retry_callback: Optional[RetryCallback],
        **request: Any,
    ) -> Dict[str, Any]:
        model = request["model"]
        payload = self._chat_payload(**request)
        response = await self._request_with_retry(
            "POST",
            f"{self.base_url}/chat/completions",
            retry_callback=retry_callback,
            retry_model=model,
            json=payload,
            headers=self._headers(),
        )
        if response.status_code >= 400:
            raise self._http_error(response, "completion")
        data = response.json()
        if not isinstance(data, dict):
            raise SubscriptionRequestError(
                f"{self.provider_label} chat completion returned an invalid response shape.",
                provider=self.provider_id,
                provider_label=self.provider_label,
            )
        return self._normalize_chat_completion(data, model, "chat.completions")

    async def _generate_via_responses(
        self,
        *,
        retry_callback: Optional[RetryCallback],
        **request: Any,
    ) -> Dict[str, Any]:
        model = request["model"]
        instructions, input_items = self._messages_to_responses_payload(request["messages"])
        payload: Dict[str, Any] = {"model": model, "input": input_items or ""}
        self.profile.apply_temperature(payload, request["temperature"], model=model)
        if instructions:
            payload["instructions"] = instructions
        if request["max_tokens"]:
            payload["max_output_tokens"] = int(request["max_tokens"])
        effort = self.profile.normalize_reasoning_effort(request["reasoning_effort"], model=model)
        self.profile.apply_reasoning(payload, effort, model=model, wire="responses")
        response_format = request["response_format"]
        if response_format and response_format.get("type") == "json_object":
            payload["text"] = {"format": {"type": "json_object"}}

        response = await self._request_with_retry(
            "POST",
            f"{self.base_url}/responses",
            retry_callback=retry_callback,
            retry_model=model,
            json=payload,
            headers=self._headers(),
        )
        if response.status_code >= 400:
            if response.status_code in {401, 403}:
                raise self._http_error(response, "completion")
            if response.status_code in {400, 404, 422}:
                return await self._generate_via_chat_completions(retry_callback=retry_callback, **request)
            raise self._http_error(response, "completion")
        data = response.json()
        if not isinstance(data, dict):
            raise SubscriptionRequestError(
                f"{self.provider_label} completion returned an invalid response shape.",
                provider=self.provider_id,
                provider_label=self.provider_label,
            )
        result = self._normalize_responses_to_chat(data, model)
        if not result["choices"][0]["message"]["content"]:
            raise SubscriptionRequestError(
                f"{self.provider_label} response did not include output text: {self._scrub(redact_log_text(str(data)[:500]))}",
                provider=self.provider_id,
                provider_label=self.provider_label,
            )
        return result

    # --- streamed Chat Completions --------------------------------------------------------------
    async def _consume_chat_stream(self, response: httpx.Response, model: str) -> _ChatStreamResult:
        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        tool_calls: Dict[int, Dict[str, Any]] = {}
        finish_reason: Optional[str] = None
        usage: Dict[str, Any] = {}
        response_id = ""
        response_model = ""
        created: Optional[int] = None
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data:
                continue
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except ValueError:
                continue
            if not isinstance(chunk, dict):
                continue
            if isinstance(chunk.get("error"), dict) or isinstance(chunk.get("error"), str):
                error = chunk["error"]
                message = error.get("message") if isinstance(error, dict) else error
                raise SubscriptionRequestError(
                    f"{self.provider_label} completion failed: {self._detail(str(message or ''))}",
                    provider=self.provider_id,
                    provider_label=self.provider_label,
                )
            response_id = response_id or str(chunk.get("id") or "")
            response_model = response_model or str(chunk.get("model") or "")
            created = created or chunk.get("created")
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                if not isinstance(choice, dict):
                    continue
                if isinstance(choice.get("usage"), dict):
                    usage = choice["usage"]
                delta = choice.get("delta") or {}
                if isinstance(delta.get("content"), str):
                    content_parts.append(delta["content"])
                reasoning = delta.get("reasoning_content")
                if not isinstance(reasoning, str):
                    reasoning = delta.get("reasoning")
                if isinstance(reasoning, str):
                    reasoning_parts.append(reasoning)
                for call in delta.get("tool_calls") or []:
                    if not isinstance(call, dict):
                        continue
                    slot = tool_calls.setdefault(
                        int(call.get("index", len(tool_calls)) or 0),
                        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                    )
                    slot["id"] = call.get("id") or slot["id"]
                    function = call.get("function") or {}
                    if function.get("name"):
                        slot["function"]["name"] = function["name"]
                    if isinstance(function.get("arguments"), str):
                        slot["function"]["arguments"] += function["arguments"]
                if choice.get("finish_reason"):
                    finish_reason = str(choice["finish_reason"])

        message: Dict[str, Any] = {"role": "assistant", "content": "".join(content_parts)}
        if reasoning_parts:
            message["reasoning_content"] = "".join(reasoning_parts)
        if tool_calls:
            message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
        data_out = {
            "id": response_id,
            "created": created,
            "model": response_model or model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": usage,
        }
        return _ChatStreamResult(terminal=finish_reason is not None, data=data_out)

    async def _generate_via_chat_stream(
        self,
        *,
        retry_callback: Optional[RetryCallback],
        **request: Any,
    ) -> Dict[str, Any]:
        model = request["model"]
        payload = {**self._chat_payload(**request), "stream": True}
        max_retries = self.settings.max_retries
        for attempt in range(max_retries):
            try:
                async with self.client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=self._headers(),
                ) as response:
                    if response.status_code >= 400:
                        await response.aread()
                        classified = await self._classified(response)
                        if classified is not None:
                            raise classified
                        if not self.profile.is_transient_response(response):
                            raise self._http_error(response, "completion")
                        detail = self._detail(response.text)
                        if attempt >= max_retries - 1:
                            raise self._request_failed(
                                f"HTTP {response.status_code}: {detail}",
                                status_code=response.status_code,
                            )
                        delay = self._retry_after_delay(response, attempt)
                        result = None
                    else:
                        result = await self._consume_chat_stream(response, model)
            except httpx.TransportError as exc:
                detail = self._detail(str(exc) or repr(exc))
                if attempt >= max_retries - 1:
                    raise self._request_failed(detail) from exc
                await self._wait_before_retry(
                    attempt=attempt,
                    delay=self._retry_delay(attempt),
                    reason="connection_error",
                    error_summary=f"[{type(exc).__name__}] {detail}",
                    retry_model=model,
                    retry_callback=retry_callback,
                )
                continue
            if result is None:
                await self._wait_before_retry(
                    attempt=attempt,
                    delay=delay,
                    reason="transient_http_error",
                    error_summary=detail,
                    retry_model=model,
                    retry_callback=retry_callback,
                )
                continue
            if result.terminal:
                return self._normalize_chat_completion(result.data, model, "chat.completions.stream")
            if attempt >= max_retries - 1:
                raise self._request_failed("the response stream ended before the provider completed the answer")
            await self._wait_before_retry(
                attempt=attempt,
                delay=self._retry_delay(attempt),
                reason="stream_incomplete",
                error_summary="The response stream ended before the provider completed the answer.",
                retry_model=model,
                retry_callback=retry_callback,
            )
        raise self._request_failed("no attempts remained")

    # --- public completion --------------------------------------------------------------------
    async def generate_completion(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        reasoning_effort: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        retry_callback: Optional[RetryCallback] = None,
    ) -> Dict[str, Any]:
        await self._ensure_region()
        request = {
            "model": model or self.settings.default_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
            "reasoning_effort": reasoning_effort,
            "tools": tools,
            "tool_choice": tool_choice,
        }
        if self.settings.wire_api == "chat_completions_stream":
            return await self._generate_via_chat_stream(retry_callback=retry_callback, **request)
        if self._messages_need_chat_completions(messages, tools):
            return await self._generate_via_chat_completions(retry_callback=retry_callback, **request)
        return await self._generate_via_responses(retry_callback=retry_callback, **request)

    async def close(self) -> None:
        await self.client.aclose()
