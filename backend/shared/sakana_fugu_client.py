"""
Sakana Fugu subscription API client.

Fugu is exposed as an OpenAI-compatible provider. This adapter stores the
desktop API key in the backend keyring and returns Chat-Completions-compatible
responses so the rest of MOTO can keep using the shared extraction/logging path.
"""
from __future__ import annotations

import asyncio
from email.utils import parsedate_to_datetime
import json
import os
import random
import time
from typing import Any, Dict, List, Optional

import httpx

from backend.shared.log_redaction import redact_log_text
from backend.shared.openrouter_client import sanitize_provider_error_text
from backend.shared.secret_store import (
    clear_sakana_fugu_api_key,
    load_sakana_fugu_api_key,
    store_sakana_fugu_api_key,
)


class SakanaFuguError(RuntimeError):
    """Base error for Sakana Fugu requests."""


class SakanaFuguAuthError(SakanaFuguError):
    """Raised when the Sakana Fugu API key is missing or rejected."""


class SakanaFuguRequestError(SakanaFuguError):
    """Raised when Sakana rejects a non-auth request."""


class SakanaFuguUsageLimitError(SakanaFuguRequestError):
    """Raised when Sakana reports a timed subscription usage window."""

    provider = "sakana_fugu"
    provider_label = "Sakana Fugu"

    def __init__(
        self,
        message: str,
        *,
        resets_at: Optional[int] = None,
        resets_in_seconds: Optional[int] = None,
        plan_type: str = "",
    ) -> None:
        self.resets_at = resets_at
        self.resets_in_seconds = resets_in_seconds
        self.plan_type = plan_type
        detail = message or "The Sakana Fugu usage limit has been reached."
        if resets_in_seconds is not None:
            detail = f"{detail} Resets in {resets_in_seconds} seconds."
        super().__init__(detail)


class SakanaFuguEntitlementError(SakanaFuguRequestError):
    """Raised when the configured subscription is not entitled to the request."""

    def __init__(self, message: str, *, error_code: str = "", status_code: Optional[int] = None) -> None:
        self.error_code = error_code
        self.status_code = status_code
        super().__init__(message or "The Sakana Fugu subscription is not entitled to this request.")


class SakanaFuguClient:
    """Client for Sakana Fugu's OpenAI-compatible API."""

    API_BASE_URL = os.getenv("MOTO_SAKANA_FUGU_BASE_URL", "https://api.sakana.ai/v1").rstrip("/")
    DEFAULT_MODEL = os.getenv("MOTO_SAKANA_FUGU_DEFAULT_MODEL", "fugu")
    KNOWN_MODELS = [
        {
            "id": "fugu",
            "name": "Fugu",
            "context_length": 1_000_000,
            "max_output_tokens": 100_000,
            "pricing": {"prompt": "subscription", "completion": "subscription"},
            "provider_metadata": {"source": "sakana_fugu", "supports_reasoning_effort": True},
        },
        {
            "id": "fugu-ultra",
            "name": "Fugu Ultra",
            "context_length": 1_000_000,
            "max_output_tokens": 100_000,
            "pricing": {"prompt": "subscription", "completion": "subscription"},
            "provider_metadata": {"source": "sakana_fugu", "supports_reasoning_effort": True},
        },
        {
            "id": "fugu-ultra-20260615",
            "name": "Fugu Ultra 20260615",
            "context_length": 1_000_000,
            "max_output_tokens": 100_000,
            "pricing": {"prompt": "subscription", "completion": "subscription"},
            "provider_metadata": {"source": "sakana_fugu", "supports_reasoning_effort": True},
        },
    ]
    TRANSIENT_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524}
    MAX_RETRIES = 4
    RETRY_DELAY = 2.0
    RETRY_MAX_DELAY = 30.0
    RETRY_JITTER_RATIO = 0.2
    USAGE_LIMIT_CODES = {"usage_limit_reached"}
    HARD_ENTITLEMENT_CODES = {
        "entitlement_required",
        "insufficient_entitlement",
        "insufficient_quota",
        "model_not_entitled",
        "not_entitled",
        "plan_not_eligible",
        "subscription_required",
    }

    def __init__(self) -> None:
        self._api_key: Optional[str] = None
        self.client = httpx.AsyncClient(
            timeout=None,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50, keepalive_expiry=30.0),
        )
        env_key = os.getenv("SAKANA_API_KEY") or os.getenv("FUGU_API_KEY") or os.getenv("MOTO_SAKANA_FUGU_API_KEY")
        if env_key:
            self._api_key = env_key.strip() or None

    def _load_api_key(self) -> Optional[str]:
        if self._api_key:
            return self._api_key
        key = load_sakana_fugu_api_key()
        if key:
            self._api_key = key
        return self._api_key

    def set_api_key(self, api_key: str, *, persist: bool = True) -> None:
        key = (api_key or "").strip()
        if not key:
            raise ValueError("Sakana Fugu API key is required.")
        self._api_key = key
        if persist:
            store_sakana_fugu_api_key(key)

    async def clear_api_key(self) -> None:
        self._api_key = None
        clear_sakana_fugu_api_key()

    async def status(self) -> Dict[str, Any]:
        return {
            "configured": bool(self._load_api_key()),
            "provider": "sakana_fugu",
            "updated_at": int(time.time()) if self._load_api_key() else None,
        }

    @staticmethod
    def _headers_for_key(key: Optional[str]) -> Dict[str, str]:
        if not key:
            raise SakanaFuguAuthError("Sakana Fugu API key is not configured.")
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

    def _headers(self) -> Dict[str, str]:
        return self._headers_for_key(self._load_api_key())

    @classmethod
    def _retry_delay(cls, attempt: int) -> float:
        base = min(cls.RETRY_DELAY * (2 ** attempt), cls.RETRY_MAX_DELAY)
        jitter = random.uniform(0.0, base * cls.RETRY_JITTER_RATIO)
        return min(base + jitter, cls.RETRY_MAX_DELAY)

    @staticmethod
    def _coerce_positive_int(value: Any) -> Optional[int]:
        try:
            parsed = int(float(value))
        except (TypeError, ValueError, OverflowError):
            return None
        return parsed if parsed > 0 else None

    @classmethod
    def _iter_error_objects(cls, value: Any, *, max_depth: int = 8, max_nodes: int = 256):
        """Yield nested OpenAI-compatible error objects without trusting their shape."""
        seen: set[int] = set()
        visited = 0

        def visit(item: Any, depth: int):
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

    @staticmethod
    def _response_payload(response: httpx.Response) -> Any:
        try:
            return response.json()
        except (json.JSONDecodeError, ValueError, TypeError, UnicodeError):
            return None

    @classmethod
    def _header_reset_values(cls, response: httpx.Response) -> tuple[Optional[int], Optional[int]]:
        now = int(time.time())
        resets_at: Optional[int] = None
        resets_in_seconds: Optional[int] = None

        retry_after = response.headers.get("Retry-After")
        if retry_after:
            seconds = cls._coerce_positive_int(retry_after)
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
            raw = response.headers.get(name)
            parsed = cls._coerce_positive_int(raw)
            if parsed is None:
                continue
            header_resets_at = parsed if parsed > now else now + parsed
            header_seconds = max(1, header_resets_at - now)
            if resets_at is None or header_resets_at < resets_at:
                resets_at = header_resets_at
                resets_in_seconds = header_seconds
        return resets_at, resets_in_seconds

    @classmethod
    def _classify_provider_error(cls, response: httpx.Response) -> Optional[SakanaFuguError]:
        payload = cls._response_payload(response)
        matched: Optional[Dict[str, Any]] = None
        matched_code = ""
        entitlement = False
        for item in cls._iter_error_objects(payload):
            codes = {
                str(item.get(field) or "").strip().lower()
                for field in ("type", "code", "error_code")
                if str(item.get(field) or "").strip()
            }
            usage_code = next((code for code in codes if code in cls.USAGE_LIMIT_CODES), "")
            if usage_code:
                matched = item
                matched_code = usage_code
                break
            entitlement_code = next((code for code in codes if code in cls.HARD_ENTITLEMENT_CODES), "")
            if entitlement_code and matched is None:
                matched = item
                matched_code = entitlement_code
                entitlement = True

        if matched_code in cls.USAGE_LIMIT_CODES:
            resets_at = cls._coerce_positive_int(matched.get("resets_at") if matched else None)
            resets_in_seconds = cls._coerce_positive_int(
                matched.get("resets_in_seconds") if matched else None
            )
            header_resets_at, header_resets_in = cls._header_reset_values(response)
            resets_at = resets_at or header_resets_at
            resets_in_seconds = resets_in_seconds or header_resets_in
            now = int(time.time())
            if resets_at is None and resets_in_seconds is not None:
                resets_at = now + resets_in_seconds
            elif resets_in_seconds is None and resets_at is not None:
                resets_in_seconds = max(1, resets_at - now)
            message = redact_log_text(
                sanitize_provider_error_text(
                    str((matched or {}).get("message") or "The Sakana Fugu usage limit has been reached.")
                )
            )
            return SakanaFuguUsageLimitError(
                message,
                resets_at=resets_at,
                resets_in_seconds=resets_in_seconds,
                plan_type=str((matched or {}).get("plan_type") or "").strip(),
            )

        if entitlement and matched is not None:
            message = redact_log_text(
                sanitize_provider_error_text(
                    str(matched.get("message") or "The Sakana Fugu subscription is not entitled to this request.")
                )
            )
            return SakanaFuguEntitlementError(
                message,
                error_code=matched_code,
                status_code=response.status_code,
            )
        return None

    @classmethod
    def _retry_after_delay(cls, response: httpx.Response, attempt: int) -> float:
        _resets_at, resets_in_seconds = cls._header_reset_values(response)
        if resets_in_seconds is not None:
            base = min(float(resets_in_seconds), cls.RETRY_MAX_DELAY)
            jitter = random.uniform(0.0, base * cls.RETRY_JITTER_RATIO)
            return min(base + jitter, cls.RETRY_MAX_DELAY)
        return cls._retry_delay(attempt)

    async def _request_with_retry(self, method: str, url: str, **kwargs) -> httpx.Response:
        for attempt in range(self.MAX_RETRIES):
            try:
                response = await self.client.request(method, url, **kwargs)
                classified = self._classify_provider_error(response) if response.status_code >= 400 else None
                if classified is not None:
                    raise classified
                if response.status_code >= 400 and response.status_code in self.TRANSIENT_STATUS_CODES:
                    detail = sanitize_provider_error_text(response.text)
                    if attempt < self.MAX_RETRIES - 1:
                        await asyncio.sleep(self._retry_after_delay(response, attempt))
                        continue
                    raise SakanaFuguRequestError(
                        f"Sakana Fugu connection failed after retries: HTTP {response.status_code}: {detail}"
                    )
                return response
            except httpx.TransportError as exc:
                detail = sanitize_provider_error_text(str(exc) or repr(exc))
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self._retry_delay(attempt))
                    continue
                raise SakanaFuguRequestError(f"Sakana Fugu connection failed after retries: {detail}") from exc
        raise SakanaFuguRequestError("Sakana Fugu request failed after retries.")

    async def _post_with_retry(self, url: str, **kwargs) -> httpx.Response:
        return await self._request_with_retry("POST", url, **kwargs)

    async def _get_with_retry(self, url: str, **kwargs) -> httpx.Response:
        return await self._request_with_retry("GET", url, **kwargs)

    @staticmethod
    def _normalize_reasoning_effort(reasoning_effort: Optional[str]) -> Optional[str]:
        effort = (reasoning_effort or "").strip().lower()
        if not effort or effort == "none":
            return None
        if effort in {"auto", "xhigh", "max", "maximum", "highest"}:
            return "xhigh"
        if effort == "high":
            return "high"
        return None

    @staticmethod
    def _chat_usage_from_responses_usage(usage: Dict[str, Any]) -> Dict[str, Any]:
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
        return {key: value for key, value in normalized.items() if value is not None}

    @staticmethod
    def _extract_responses_text(data: Dict[str, Any]) -> str:
        if isinstance(data.get("output_text"), str):
            return data["output_text"]
        parts: List[str] = []
        for item in data.get("output") or []:
            if not isinstance(item, dict):
                continue
            for content in item.get("content") or []:
                if not isinstance(content, dict):
                    continue
                text = content.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

    @staticmethod
    def _messages_to_responses_payload(messages: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
        instructions: List[str] = []
        inputs: List[Dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role") or "user")
            content = message.get("content", "")
            if role in {"system", "developer"}:
                if isinstance(content, str):
                    instructions.append(content)
                else:
                    instructions.append(str(content))
                continue
            responses_role = "assistant" if role == "assistant" else "user"
            if role == "tool":
                responses_role = "user"
            inputs.append({"role": responses_role, "content": content})
        return "\n\n".join(part for part in instructions if part), inputs

    @classmethod
    def _normalize_responses_to_chat(cls, data: Dict[str, Any], model: str) -> Dict[str, Any]:
        text = cls._extract_responses_text(data)
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        return {
            "id": data.get("id") or f"sakana-fugu-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": data.get("created_at") or data.get("created") or int(time.time()),
            "model": data.get("model") or model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": data.get("status") if data.get("status") not in {None, "completed"} else "stop",
                }
            ],
            "usage": cls._chat_usage_from_responses_usage(usage),
            "_moto_sakana_wire_api": "responses",
        }

    @staticmethod
    def _messages_need_chat_completions(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]]) -> bool:
        if tools:
            return True
        for message in messages:
            role = str(message.get("role") or "")
            if role == "tool" or message.get("tool_calls"):
                return True
        return False

    @classmethod
    def _normalize_chat_completion(cls, data: Dict[str, Any], model: str) -> Dict[str, Any]:
        choices = data.get("choices") if isinstance(data.get("choices"), list) else []
        if not choices:
            raise SakanaFuguRequestError("Sakana Fugu chat completion returned no choices.")
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        return {
            "id": data.get("id") or f"sakana-fugu-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": data.get("created") or int(time.time()),
            "model": data.get("model") or model,
            "choices": choices,
            "usage": cls._chat_usage_from_responses_usage(usage),
            "_moto_sakana_wire_api": "chat.completions",
        }

    async def _generate_via_chat_completions(
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
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = int(max_tokens)
        effort = self._normalize_reasoning_effort(reasoning_effort)
        if effort:
            payload["reasoning"] = {"effort": effort}
        if response_format:
            payload["response_format"] = response_format
        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice

        response = await self._post_with_retry(
            f"{self.API_BASE_URL}/chat/completions",
            json=payload,
            headers=self._headers(),
        )
        if response.status_code >= 400:
            message = sanitize_provider_error_text(response.text)
            if response.status_code in {401, 403}:
                raise SakanaFuguAuthError(f"Sakana Fugu completion failed: {message}")
            raise SakanaFuguRequestError(f"Sakana Fugu completion failed: {message}")
        data = response.json()
        if not isinstance(data, dict):
            raise SakanaFuguRequestError("Sakana Fugu chat completion returned an invalid response shape.")
        return self._normalize_chat_completion(data, model)

    async def list_models(self, api_key: Optional[str] = None) -> List[Dict[str, Any]]:
        try:
            headers = self._headers_for_key(api_key.strip() if api_key is not None else self._load_api_key())
            response = await self._get_with_retry(f"{self.API_BASE_URL}/models", headers=headers)
            if response.status_code >= 400:
                if response.status_code in {401, 403}:
                    raise SakanaFuguAuthError(f"Sakana Fugu model list failed: {sanitize_provider_error_text(response.text)}")
                raise SakanaFuguRequestError(f"Sakana Fugu model list failed: {sanitize_provider_error_text(response.text)}")
            payload = response.json()
            records = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(records, list):
                return self.KNOWN_MODELS
            known_by_id = {model["id"]: model for model in self.KNOWN_MODELS}
            models = []
            for record in records:
                model_id = str(record.get("id") or "").strip()
                if not model_id:
                    continue
                base = known_by_id.get(model_id, {})
                models.append({
                    **base,
                    "id": model_id,
                    "name": base.get("name") or model_id,
                    "provider_metadata": {"source": "sakana_fugu", **base.get("provider_metadata", {})},
                })
            return models or self.KNOWN_MODELS
        except SakanaFuguError:
            raise
        except Exception as exc:
            raise SakanaFuguRequestError("Sakana Fugu model list failed before a valid model list was returned.") from exc

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
    ) -> Dict[str, Any]:
        selected_model = model or self.DEFAULT_MODEL
        if self._messages_need_chat_completions(messages, tools):
            return await self._generate_via_chat_completions(
                model=selected_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                reasoning_effort=reasoning_effort,
                tools=tools,
                tool_choice=tool_choice,
            )

        instructions, input_items = self._messages_to_responses_payload(messages)
        payload: Dict[str, Any] = {
            "model": selected_model,
            "input": input_items or "",
            "temperature": temperature,
        }
        if instructions:
            payload["instructions"] = instructions
        if max_tokens:
            payload["max_output_tokens"] = int(max_tokens)
        effort = self._normalize_reasoning_effort(reasoning_effort)
        if effort:
            payload["reasoning"] = {"effort": effort}
        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        if response_format and response_format.get("type") == "json_object":
            payload["text"] = {"format": {"type": "json_object"}}

        response = await self._post_with_retry(f"{self.API_BASE_URL}/responses", json=payload, headers=self._headers())
        if response.status_code >= 400:
            message = sanitize_provider_error_text(response.text)
            if response.status_code in {401, 403}:
                raise SakanaFuguAuthError(f"Sakana Fugu completion failed: {message}")
            if response.status_code in {400, 404, 422}:
                return await self._generate_via_chat_completions(
                    model=selected_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    reasoning_effort=reasoning_effort,
                    tools=tools,
                    tool_choice=tool_choice,
                )
            raise SakanaFuguRequestError(f"Sakana Fugu completion failed: {message}")
        data = response.json()
        if not isinstance(data, dict):
            raise SakanaFuguRequestError("Sakana Fugu completion returned an invalid response shape.")
        result = self._normalize_responses_to_chat(data, selected_model)
        if not result["choices"][0]["message"]["content"]:
            raise SakanaFuguRequestError(
                f"Sakana Fugu response did not include output text: {redact_log_text(str(data)[:500])}"
            )
        return result

    async def close(self) -> None:
        await self.client.aclose()


sakana_fugu_client = SakanaFuguClient()
