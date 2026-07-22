"""
OpenAI Codex/ChatGPT subscription OAuth client.

This adapter intentionally targets the Codex backend used by ChatGPT account
login flows. It is distinct from the regular OpenAI API-key billing path.
"""
from __future__ import annotations

import base64
import asyncio
import hashlib
import json
import logging
import secrets
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode, urlparse, parse_qs

import httpx

from backend.shared.log_redaction import redact_log_text
from backend.shared.openrouter_client import sanitize_provider_error_text
from backend.shared.secret_store import (
    SecretStoreError,
    clear_openai_codex_oauth_tokens,
    load_openai_codex_oauth_tokens,
    store_openai_codex_oauth_tokens,
)

logger = logging.getLogger(__name__)


class OpenAICodexError(RuntimeError):
    """Base error for Codex OAuth-backed requests."""


class OpenAICodexAuthError(OpenAICodexError):
    """Raised when Codex OAuth credentials are missing or unusable."""


class OpenAICodexRequestError(OpenAICodexError):
    """Raised when Codex rejects a completion request after authentication."""


class OAuthUsageLimitError(OpenAICodexError):
    """Raised when an OAuth-backed provider reports a timed usage limit."""

    def __init__(
        self,
        *,
        provider: str,
        provider_label: str,
        message: str,
        plan_type: str = "",
        resets_at: Optional[int] = None,
        resets_in_seconds: Optional[int] = None,
    ) -> None:
        self.provider = provider
        self.provider_label = provider_label
        self.plan_type = plan_type
        self.resets_at = resets_at
        self.resets_in_seconds = resets_in_seconds
        detail = message or "The usage limit has been reached."
        if resets_in_seconds is not None:
            detail = f"{detail} Resets in {resets_in_seconds} seconds."
        super().__init__(detail)


class OpenAICodexClient:
    """Client for OpenAI Codex OAuth and the ChatGPT Codex Responses backend."""

    CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
    AUTH_URL = "https://auth.openai.com/oauth/authorize"
    TOKEN_URL = "https://auth.openai.com/oauth/token"
    REVOKE_URL = "https://auth.openai.com/oauth/revoke"
    CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
    DEFAULT_REDIRECT_URI = "http://localhost:1455/auth/callback"
    DEFAULT_ORIGINATOR = "moto-autonomous-asi"
    DEFAULT_INSTRUCTIONS = "Follow the user's instructions and produce the requested response."
    REFRESH_SKEW_SECONDS = 60
    REASONING_EFFORT_LEVELS = {"xhigh", "high", "medium", "low", "none"}
    MAX_RETRIES = 4
    RETRY_DELAY = 2.0
    RETRY_MAX_DELAY = 30.0
    TRANSIENT_COMPLETION_STATUS_CODES = {408, 409, 425, 500, 502, 503, 504, 520, 521, 522, 523, 524}
    TRANSIENT_COMPLETION_MARKERS = (
        "bad gateway",
        "connection timeout",
        "disconnect/reset before headers",
        "gateway timeout",
        "incomplete chunked read",
        "peer closed connection",
        "service unavailable",
        "server_error",
        "temporarily unavailable",
        "upstream connect error",
        "upstream provider timeout",
        "you can retry",
    )
    CODEX_SPARK_MODEL_ID = "gpt-5.3-codex-spark"
    CODEX_SPARK_HIGH_MODEL_ID = "gpt-5.3-codex-spark-high"
    PUBLIC_MODEL_CATALOG = (
        {
            "slug": CODEX_SPARK_HIGH_MODEL_ID,
            "title": "GPT-5.3 Codex Spark (high)",
            "canonical_model": CODEX_SPARK_MODEL_ID,
            "reasoning_effort": "high",
            "context_length": 128000,
            "input_context_window": 128000,
            "max_output_tokens": 32768,
        },
    )
    KNOWN_MODEL_LIMITS = {
        # OpenAI documents GPT-5.5 in Codex as a 400K-window product. Public
        # Codex runtime metadata has exposed this as 272K input + 128K output
        # with a 95% effective input budget, which is distinct from the 1M API.
        "gpt-5.5": {
            "context_length": 400000,
            "input_context_window": 272000,
            "effective_input_context_window": 258400,
            "max_output_tokens": 128000,
            "effective_context_window_percent": 95,
        },
        CODEX_SPARK_HIGH_MODEL_ID: {
            "context_length": 128000,
            "input_context_window": 128000,
            "max_output_tokens": 32768,
            "canonical_model": CODEX_SPARK_MODEL_ID,
            "reasoning_effort": "high",
        },
    }

    def __init__(self) -> None:
        self._refresh_lock = asyncio.Lock()
        self.client = httpx.AsyncClient(
            timeout=None,
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=50,
                keepalive_expiry=30.0,
            ),
        )

    @staticmethod
    def generate_pkce_pair() -> tuple[str, str]:
        """Return a PKCE verifier and S256 challenge."""
        verifier = secrets.token_urlsafe(64)
        digest = hashlib.sha256(verifier.encode("ascii")).digest()
        challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
        return verifier, challenge

    @classmethod
    def build_authorization_url(
        cls,
        *,
        code_challenge: str,
        state: str,
        redirect_uri: str = DEFAULT_REDIRECT_URI,
    ) -> str:
        """Build the OpenAI OAuth authorization URL for Codex login."""
        params = {
            "response_type": "code",
            "client_id": cls.CLIENT_ID,
            "redirect_uri": redirect_uri,
            "scope": "openid profile email offline_access api.connectors.read api.connectors.invoke",
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
            "originator": cls.DEFAULT_ORIGINATOR,
            "state": state,
        }
        return f"{cls.AUTH_URL}?{urlencode(params)}"

    @staticmethod
    def extract_code_and_state(code: str = "", redirect_url: str = "") -> tuple[str, str]:
        """Extract authorization code/state from explicit code or pasted callback URL."""
        if redirect_url:
            parsed = urlparse(redirect_url.strip())
            query = parse_qs(parsed.query)
            return (query.get("code", [""])[0], query.get("state", [""])[0])
        return code.strip(), ""

    @staticmethod
    def _jwt_payload(token: str) -> Dict[str, Any]:
        try:
            payload_b64 = token.split(".")[1]
            payload_b64 += "=" * (-len(payload_b64) % 4)
            payload = json.loads(base64.urlsafe_b64decode(payload_b64.encode("ascii")))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    @classmethod
    def _normalize_token_payload(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        access_token = str(payload.get("access_token") or payload.get("access") or "")
        refresh_token = str(payload.get("refresh_token") or payload.get("refresh") or "")
        id_token = str(payload.get("id_token") or "")
        jwt_payload = cls._jwt_payload(access_token)
        expires_at = payload.get("expires_at") or payload.get("expires")
        if not expires_at and jwt_payload.get("exp"):
            expires_at = int(jwt_payload["exp"])
        elif payload.get("expires_in"):
            expires_at = int(time.time()) + int(payload["expires_in"])

        auth_claim = jwt_payload.get("https://api.openai.com/auth")
        auth_account_id = auth_claim.get("chatgpt_account_id") if isinstance(auth_claim, dict) else None
        account_id = payload.get("account_id") or payload.get("accountId") or auth_account_id
        account_id = account_id or jwt_payload.get("chatgpt_account_id") or jwt_payload.get("account_id")
        email = payload.get("email") or jwt_payload.get("email")

        normalized = {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "id_token": id_token,
            "expires_at": int(expires_at or 0),
            "account_id": str(account_id or ""),
            "email": str(email or ""),
            "provider": "openai_codex_oauth",
            "updated_at": int(time.time()),
        }
        return {key: value for key, value in normalized.items() if value not in ("", None)}

    async def exchange_code(
        self,
        *,
        code: str,
        code_verifier: str,
        redirect_uri: str = DEFAULT_REDIRECT_URI,
    ) -> Dict[str, Any]:
        """Exchange an OAuth authorization code for persisted Codex tokens."""
        if not code or not code_verifier:
            raise OpenAICodexAuthError("Authorization code and PKCE verifier are required.")

        response = await self.client.post(
            self.TOKEN_URL,
            data={
                "client_id": self.CLIENT_ID,
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "code_verifier": code_verifier,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if response.status_code >= 400:
            raise OpenAICodexAuthError(
                f"OpenAI Codex OAuth exchange failed: {sanitize_provider_error_text(response.text)}"
            )
        tokens = self._normalize_token_payload(response.json())
        if not tokens.get("access_token") or not tokens.get("refresh_token"):
            raise OpenAICodexAuthError("OpenAI Codex OAuth exchange did not return usable tokens.")
        store_openai_codex_oauth_tokens(tokens)
        return self.safe_status(tokens)

    async def refresh_tokens(self, tokens: Dict[str, Any]) -> Dict[str, Any]:
        """Refresh stored Codex OAuth tokens and persist the replacement bundle."""
        refresh_token = str(tokens.get("refresh_token") or "")
        if not refresh_token:
            raise OpenAICodexAuthError("OpenAI Codex refresh token is missing.")

        response = await self.client.post(
            self.TOKEN_URL,
            data={
                "client_id": self.CLIENT_ID,
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if response.status_code >= 400:
            raise OpenAICodexAuthError(
                f"OpenAI Codex token refresh failed: {sanitize_provider_error_text(response.text)}"
            )
        refreshed = self._normalize_token_payload({**tokens, **response.json()})
        store_openai_codex_oauth_tokens(refreshed)
        return refreshed

    def load_tokens(self) -> Optional[Dict[str, Any]]:
        """Load persisted Codex OAuth tokens."""
        try:
            return load_openai_codex_oauth_tokens()
        except SecretStoreError:
            raise
        except Exception as exc:
            raise OpenAICodexAuthError("Failed to load OpenAI Codex OAuth tokens.") from exc

    async def get_valid_tokens(self) -> Dict[str, Any]:
        """Load and refresh tokens if they are close to expiry."""
        tokens = self.load_tokens()
        if not tokens or not tokens.get("access_token"):
            raise OpenAICodexAuthError("OpenAI Codex OAuth is not configured.")

        expires_at = int(tokens.get("expires_at") or 0)
        if expires_at and time.time() < expires_at - self.REFRESH_SKEW_SECONDS:
            return tokens

        async with self._refresh_lock:
            tokens = self.load_tokens()
            if not tokens or not tokens.get("access_token"):
                raise OpenAICodexAuthError("OpenAI Codex OAuth is not configured.")
            expires_at = int(tokens.get("expires_at") or 0)
            if expires_at and time.time() < expires_at - self.REFRESH_SKEW_SECONDS:
                return tokens
            return await self.refresh_tokens(tokens)

    @staticmethod
    def _access_token(tokens: Optional[Dict[str, Any]]) -> str:
        return str((tokens or {}).get("access_token") or "")

    @classmethod
    def _is_auth_failure_text(cls, text: str) -> bool:
        lowered = (text or "").lower()
        return (
            "token_revoked" in lowered
            or "invalidated oauth token" in lowered
            or "invalid oauth token" in lowered
            or "expired oauth token" in lowered
        )

    async def _recover_tokens_after_auth_failure(
        self,
        used_tokens: Dict[str, Any],
        *,
        context: str,
    ) -> Dict[str, Any]:
        """Reload or refresh Codex OAuth tokens after a server-side auth rejection."""
        async with self._refresh_lock:
            current_tokens = self.load_tokens()
            if not current_tokens or not current_tokens.get("access_token"):
                raise OpenAICodexAuthError(
                    f"OpenAI Codex {context} failed because OAuth is no longer configured."
                )
            if self._access_token(current_tokens) != self._access_token(used_tokens):
                logger.info(
                    "OpenAI Codex %s auth failed with an older token; retrying with the newer stored OAuth token.",
                    context,
                )
                return current_tokens
            logger.info(
                "OpenAI Codex %s auth failed; refreshing OAuth token once before retrying.",
                context,
            )
            return await self.refresh_tokens(current_tokens)

    @staticmethod
    def safe_status(tokens: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Return token status without exposing token material."""
        if not tokens:
            return {"configured": False}
        expires_at = int(tokens.get("expires_at") or 0)
        return {
            "configured": bool(tokens.get("access_token") and tokens.get("refresh_token")),
            "expires_at": expires_at,
            "expired": bool(expires_at and time.time() >= expires_at),
            "updated_at": int(tokens.get("updated_at") or 0),
            "account_id": redact_log_text(tokens.get("account_id", ""), 80),
            "email": redact_log_text(tokens.get("email", ""), 120),
        }

    async def status(self) -> Dict[str, Any]:
        """Return current Codex OAuth status."""
        return self.safe_status(self.load_tokens())

    @staticmethod
    def _positive_int(*values: Any) -> Optional[int]:
        """Return the first positive integer-like metadata value."""
        for value in values:
            if value in (None, ""):
                continue
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                return parsed
        return None

    @classmethod
    def _normalize_model_metadata(cls, model: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize Codex model-catalog fields into the frontend model shape."""
        slug = model.get("slug") or model.get("id")
        if not slug:
            return None
        slug = str(slug)
        known = cls.KNOWN_MODEL_LIMITS.get(slug, {})

        max_output_tokens = cls._positive_int(
            model.get("max_output_tokens"),
            model.get("max_completion_tokens"),
            model.get("output_tokens"),
            known.get("max_output_tokens"),
        )
        input_context_window = cls._positive_int(
            model.get("input_context_window"),
            model.get("context_window"),
            model.get("max_context_window"),
            known.get("input_context_window"),
        )
        raw_total_context = cls._positive_int(
            model.get("context_length"),
            model.get("contextTokens"),
            model.get("total_context_window"),
            model.get("max_total_context_window"),
        )
        context_length = cls._positive_int(
            raw_total_context,
            known.get("context_length"),
            input_context_window,
        )
        effective_percent = cls._positive_int(
            model.get("effective_context_window_percent"),
            known.get("effective_context_window_percent"),
        )
        computed_effective_input = (
            int(input_context_window * effective_percent / 100)
            if input_context_window and effective_percent
            else None
        )
        effective_input_context_window = cls._positive_int(
            model.get("effective_input_context_window"),
            model.get("effective_context_window"),
            computed_effective_input,
            known.get("effective_input_context_window"),
        )

        normalized: Dict[str, Any] = {
            "id": slug,
            "name": model.get("title") or model.get("name") or slug,
            "pricing": {"prompt": "subscription", "completion": "subscription"},
            "provider_metadata": {
                "source": "openai_codex_oauth",
                "raw_context_length": model.get("context_length") or model.get("contextTokens"),
                "raw_context_window": model.get("context_window"),
                "raw_max_context_window": model.get("max_context_window"),
                "raw_max_output_tokens": model.get("max_output_tokens") or model.get("max_completion_tokens"),
                "effective_context_window_percent": effective_percent,
            },
        }
        canonical_model = model.get("canonical_model") or known.get("canonical_model")
        if canonical_model:
            normalized["canonical_model"] = str(canonical_model)
            normalized["provider_metadata"]["canonical_model"] = str(canonical_model)
        model_reasoning_effort = model.get("reasoning_effort") or known.get("reasoning_effort")
        if model_reasoning_effort:
            normalized["reasoning_effort"] = str(model_reasoning_effort)
            normalized["provider_metadata"]["reasoning_effort"] = str(model_reasoning_effort)
        if context_length:
            normalized["context_length"] = context_length
        if max_output_tokens:
            normalized["max_output_tokens"] = max_output_tokens
        if input_context_window:
            normalized["input_context_window"] = input_context_window
        if effective_input_context_window:
            normalized["effective_input_context_window"] = effective_input_context_window
        return normalized

    async def clear_tokens(self) -> None:
        """Revoke best-effort and clear persisted Codex OAuth tokens."""
        tokens = self.load_tokens()
        token = str((tokens or {}).get("refresh_token") or "")
        if token:
            try:
                await self.client.post(
                    self.REVOKE_URL,
                    json={"client_id": self.CLIENT_ID, "token": token},
                    headers={"Content-Type": "application/json"},
                )
            except Exception:
                logger.debug("OpenAI Codex token revoke failed; clearing local credential anyway.")
        clear_openai_codex_oauth_tokens()

    def _headers(self, tokens: Dict[str, Any], *, accept_stream: bool = False) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {tokens['access_token']}",
            "Content-Type": "application/json",
        }
        if accept_stream:
            headers["Accept"] = "text/event-stream"
        account_id = tokens.get("account_id")
        if account_id:
            headers["ChatGPT-Account-ID"] = str(account_id)
        return headers

    async def list_models(self) -> List[Dict[str, Any]]:
        """List Codex-backed models available to the signed-in ChatGPT account."""
        tokens = await self.get_valid_tokens()
        response = await self.client.get(
            f"{self.CODEX_BASE_URL}/models?client_version=1.0.0",
            headers=self._headers(tokens),
        )
        if response.status_code in {401, 403}:
            tokens = await self._recover_tokens_after_auth_failure(tokens, context="model listing")
            response = await self.client.get(
                f"{self.CODEX_BASE_URL}/models?client_version=1.0.0",
                headers=self._headers(tokens),
            )
        if response.status_code >= 400:
            raise OpenAICodexAuthError(
                f"OpenAI Codex model listing failed: {sanitize_provider_error_text(response.text)}"
            )
        data = response.json()
        models = []
        seen_model_ids = set()
        for model in data.get("models", []):
            if not isinstance(model, dict):
                continue
            if model.get("supported_in_api") is False or model.get("visibility") not in (None, "list"):
                continue
            normalized = self._normalize_model_metadata(model)
            if normalized:
                seen_model_ids.add(normalized["id"])
                models.append(normalized)
        for model in self.PUBLIC_MODEL_CATALOG:
            normalized = self._normalize_model_metadata(model)
            if normalized and normalized["id"] not in seen_model_ids:
                models.append(normalized)
                seen_model_ids.add(normalized["id"])
        return models

    @classmethod
    def _resolve_model_request(cls, model: str, reasoning_effort: Optional[str]) -> tuple[str, Optional[str]]:
        """Map user-facing Codex aliases onto the backend model id/request knobs."""
        if model == cls.CODEX_SPARK_HIGH_MODEL_ID:
            return cls.CODEX_SPARK_MODEL_ID, "high"
        # Some Codex catalogs expose named variants as selectable entries while
        # the Responses backend accepts the base model plus a reasoning effort.
        # Keep the selected alias in MOTO metadata, but normalize the request.
        named_effort = {
            "gpt-5.6-luna": "high",
            "gpt-5.6-terra": "medium",
        }.get(model)
        if named_effort:
            return "gpt-5.6-sol", named_effort
        return model, reasoning_effort

    @staticmethod
    def _split_instructions(messages: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
        instruction_parts: List[str] = []
        input_items: List[Dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role") or "user")
            content = message.get("content") or ""
            if role in {"system", "developer"}:
                instruction_parts.append(str(content))
                continue
            if role == "tool":
                input_items.append({
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id") or message.get("id") or "tool_call",
                    "output": str(content),
                })
                continue
            if role == "assistant" and message.get("tool_calls"):
                if content:
                    input_items.append({"role": "assistant", "content": str(content)})
                for tool_call in message.get("tool_calls") or []:
                    function = tool_call.get("function") or {}
                    input_items.append({
                        "type": "function_call",
                        "call_id": tool_call.get("id") or "tool_call",
                        "name": function.get("name") or "",
                        "arguments": function.get("arguments") or "{}",
                    })
                continue
            input_items.append({"role": role if role in {"user", "assistant"} else "user", "content": str(content)})
        return "\n\n".join(part for part in instruction_parts if part), input_items

    @staticmethod
    def _convert_tools(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        if not tools:
            return None
        converted: List[Dict[str, Any]] = []
        for tool in tools:
            if tool.get("type") != "function":
                continue
            function = tool.get("function") or {}
            name = function.get("name")
            if not name:
                continue
            converted.append({
                "type": "function",
                "name": name,
                "description": function.get("description") or "",
                "parameters": function.get("parameters") or {"type": "object", "properties": {}},
                "strict": False,
            })
        return converted or None

    @staticmethod
    def _response_format(response_format: Optional[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        if not response_format:
            return None
        if response_format.get("type") == "json_object":
            return {"format": {"type": "json_object"}}
        return None

    @classmethod
    def _is_transient_completion_text(cls, text: str) -> bool:
        lowered = (text or "").lower()
        return any(marker in lowered for marker in cls.TRANSIENT_COMPLETION_MARKERS)

    @classmethod
    def _is_transient_completion_response(cls, response: httpx.Response) -> bool:
        body = ""
        try:
            body = response.text or ""
        except Exception:
            body = ""
        return (
            response.status_code in cls.TRANSIENT_COMPLETION_STATUS_CODES
            or cls._is_transient_completion_text(body)
        )

    @staticmethod
    def _coerce_positive_int(value: Any) -> Optional[int]:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    @classmethod
    def _extract_usage_limit_payload(cls, value: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(value, dict):
            return None
        error_type = str(value.get("type") or value.get("code") or "").strip().lower()
        if error_type == "usage_limit_reached":
            return value
        for key in ("error", "response"):
            nested = value.get(key)
            if isinstance(nested, dict):
                found = cls._extract_usage_limit_payload(nested)
                if found is not None:
                    return found
        response = value.get("response")
        if isinstance(response, dict):
            nested_error = response.get("error")
            if isinstance(nested_error, dict):
                return cls._extract_usage_limit_payload(nested_error)
        return None

    @classmethod
    def _usage_limit_error_from_payload(cls, value: Any) -> Optional[OAuthUsageLimitError]:
        payload = cls._extract_usage_limit_payload(value)
        if payload is None:
            return None
        resets_at = cls._coerce_positive_int(payload.get("resets_at"))
        resets_in_seconds = cls._coerce_positive_int(payload.get("resets_in_seconds"))
        if resets_at is None and resets_in_seconds is not None:
            resets_at = int(time.time()) + resets_in_seconds
        elif resets_in_seconds is None and resets_at is not None:
            resets_in_seconds = max(1, resets_at - int(time.time()))
        message = str(payload.get("message") or "The usage limit has been reached.").strip()
        return OAuthUsageLimitError(
            provider="openai_codex_oauth",
            provider_label="OpenAI Codex",
            message=message,
            plan_type=str(payload.get("plan_type") or "").strip(),
            resets_at=resets_at,
            resets_in_seconds=resets_in_seconds,
        )

    @classmethod
    def _usage_limit_error_from_text(cls, text: str) -> Optional[OAuthUsageLimitError]:
        raw = str(text or "")
        start = raw.find("{")
        end = raw.rfind("}")
        if 0 <= start < end:
            try:
                parsed = json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                parsed = None
            usage_error = cls._usage_limit_error_from_payload(parsed)
            if usage_error is not None:
                return usage_error
        lowered = raw.lower()
        if "usage_limit_reached" in lowered or "usage limit has been reached" in lowered:
            return OAuthUsageLimitError(
                provider="openai_codex_oauth",
                provider_label="OpenAI Codex",
                message="The usage limit has been reached.",
            )
        return None

    @classmethod
    def _max_attempts(cls) -> int:
        return cls.MAX_RETRIES + 1

    @classmethod
    def _retry_delay(cls, retry_index: int) -> float:
        return min(cls.RETRY_MAX_DELAY, cls.RETRY_DELAY * (2 ** max(0, retry_index)))

    @classmethod
    def _reasoning_config(cls, reasoning_effort: Optional[str]) -> Optional[Dict[str, str]]:
        if not reasoning_effort:
            return None
        effort = str(reasoning_effort).strip().lower()
        if effort in {"auto", "max", "maximum", "highest"}:
            effort = "xhigh"
        elif effort == "minimal":
            effort = "low"
        if effort == "none":
            return None
        if effort not in cls.REASONING_EFFORT_LEVELS:
            logger.warning("Unknown OpenAI Codex reasoning effort '%s'; defaulting to xhigh", reasoning_effort)
            effort = "xhigh"
        return {"effort": effort}

    @staticmethod
    def _iter_sse_data(raw_body: str) -> List[str]:
        events: List[str] = []
        current_data_lines: List[str] = []
        for line in raw_body.splitlines():
            stripped = line.rstrip("\r")
            if not stripped:
                if current_data_lines:
                    events.append("\n".join(current_data_lines))
                    current_data_lines = []
                continue
            if stripped.startswith(":"):
                continue
            if stripped.startswith("data:"):
                current_data_lines.append(stripped[5:].lstrip())
        if current_data_lines:
            events.append("\n".join(current_data_lines))
        return events

    @classmethod
    def _decode_response_body(cls, raw_body: str) -> Dict[str, Any]:
        body = raw_body.strip()
        if not body:
            raise OpenAICodexRequestError("OpenAI Codex completion failed: empty response body")

        try:
            data = json.loads(body)
            if isinstance(data, dict):
                usage_error = cls._usage_limit_error_from_payload(data)
                if usage_error is not None:
                    raise usage_error
                return data
        except json.JSONDecodeError:
            logger.debug("OpenAI Codex response body is not plain JSON; parsing stream events")

        response_data: Optional[Dict[str, Any]] = None
        output_text_parts: List[str] = []
        for event_data in cls._iter_sse_data(body):
            if event_data == "[DONE]":
                continue
            try:
                event = json.loads(event_data)
            except json.JSONDecodeError:
                logger.debug("Ignoring malformed OpenAI Codex stream event: %s", redact_log_text(event_data[:500]))
                continue
            if not isinstance(event, dict):
                continue

            event_type = str(event.get("type") or "")
            if event_type in {"response.failed", "response.incomplete"}:
                response = event.get("response")
                response_error = response.get("error") if isinstance(response, dict) else None
                error = event.get("error") or response_error
                usage_error = cls._usage_limit_error_from_payload(error or event)
                if usage_error is not None:
                    raise usage_error
                raise OpenAICodexRequestError(
                    f"OpenAI Codex completion failed: {sanitize_provider_error_text(json.dumps(error or event))}"
                )
            if event_type == "response.output_text.delta":
                output_text_parts.append(str(event.get("delta") or ""))
            elif event_type == "response.output_text.done" and not output_text_parts:
                text = event.get("text")
                if text:
                    output_text_parts.append(str(text))

            response = event.get("response")
            if isinstance(response, dict):
                response_data = response

        if response_data is not None:
            if output_text_parts and not response_data.get("output_text"):
                response_data = {**response_data, "output_text": "".join(output_text_parts)}
            return response_data

        if output_text_parts:
            return {"id": "", "output_text": "".join(output_text_parts)}

        raise OpenAICodexRequestError("OpenAI Codex streamed response contained no completion output.")

    @staticmethod
    def _extract_output(response: Dict[str, Any]) -> tuple[str, List[Dict[str, Any]]]:
        aggregate_text = response.get("output_text") or ""
        output_text = "" if aggregate_text else ""
        tool_calls: List[Dict[str, Any]] = []
        for item in response.get("output") or []:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "function_call":
                tool_calls.append({
                    "id": item.get("call_id") or item.get("id") or f"call_{len(tool_calls) + 1}",
                    "type": "function",
                    "function": {
                        "name": item.get("name") or "",
                        "arguments": item.get("arguments") or "{}",
                    },
                })
                continue
            if item.get("type") == "message":
                for content in item.get("content") or []:
                    if isinstance(content, dict) and content.get("type") in {"output_text", "text"}:
                        output_text += content.get("text") or ""
        return aggregate_text or output_text, tool_calls

    async def _post_with_retry(self, url: str, **kwargs) -> httpx.Response:
        """POST with retry on transport errors (peer close, read error, connect error)."""
        max_attempts = self._max_attempts()
        for attempt in range(max_attempts):
            try:
                response = await self.client.post(url, **kwargs)
                if response.status_code >= 400 and self._is_transient_completion_response(response):
                    error_detail = sanitize_provider_error_text(response.text)
                    delay = self._retry_delay(attempt)
                    logger.warning(
                        "OpenAI Codex transient completion response (attempt %s/%s): "
                        "status=%s error=%s%s",
                        attempt + 1,
                        max_attempts,
                        response.status_code,
                        error_detail,
                        f"; retrying in {delay:.1f}s" if attempt < max_attempts - 1 else "",
                    )
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(delay)
                        continue
                    raise OpenAICodexRequestError(
                        f"OpenAI Codex connection failed after {self.MAX_RETRIES} retries: "
                        f"HTTP {response.status_code}: {error_detail}"
                    )
                return response
            except httpx.TransportError as e:
                error_type = type(e).__name__
                error_detail = sanitize_provider_error_text(str(e) or repr(e))
                delay = self._retry_delay(attempt)
                logger.warning(
                    "OpenAI Codex connection error (attempt %s/%s): [%s] %s%s",
                    attempt + 1,
                    max_attempts,
                    error_type,
                    error_detail,
                    f"; retrying in {delay:.1f}s" if attempt < max_attempts - 1 else "",
                )
                if attempt < max_attempts - 1:
                    await asyncio.sleep(delay)
                    continue
                raise OpenAICodexRequestError(
                    f"OpenAI Codex connection failed after {self.MAX_RETRIES} retries: "
                    f"[{error_type}] {error_detail}"
                )

    async def generate_completion(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        reasoning_effort: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Generate a completion and return a Chat Completions-compatible shape."""
        requested_model = model
        model, reasoning_effort = self._resolve_model_request(model, reasoning_effort)
        tokens = await self.get_valid_tokens()
        instructions, input_items = self._split_instructions(messages)
        payload: Dict[str, Any] = {
            "model": model,
            "input": input_items,
            "store": False,
            "stream": True,
        }
        payload["instructions"] = instructions or self.DEFAULT_INSTRUCTIONS
        # ChatGPT's Codex backend rejects standard Responses knobs such as
        # max_output_tokens and temperature; keep them compatibility-only here.
        reasoning = self._reasoning_config(reasoning_effort)
        if reasoning:
            payload["reasoning"] = reasoning
        text_format = self._response_format(response_format)
        if text_format:
            payload["text"] = text_format
        converted_tools = self._convert_tools(tools)
        if converted_tools:
            payload["tools"] = converted_tools
            if tool_choice is not None:
                payload["tool_choice"] = "auto" if tool_choice == "auto" else tool_choice

        auth_retry_used = False
        stream_retries_used = 0
        while True:
            response = await self._post_with_retry(
                f"{self.CODEX_BASE_URL}/responses",
                json=payload,
                headers=self._headers(tokens, accept_stream=True),
            )
            if response.status_code >= 400:
                usage_error = self._usage_limit_error_from_text(response.text)
                if usage_error is not None:
                    raise usage_error
                message = f"OpenAI Codex completion failed: {sanitize_provider_error_text(response.text)}"
                if response.status_code in {401, 403}:
                    if not auth_retry_used:
                        tokens = await self._recover_tokens_after_auth_failure(tokens, context="completion")
                        auth_retry_used = True
                        continue
                    raise OpenAICodexAuthError(message)
                raise OpenAICodexRequestError(message)
            try:
                data = self._decode_response_body(response.text)
                break
            except OpenAICodexRequestError as exc:
                if self._is_auth_failure_text(str(exc)) and not auth_retry_used:
                    tokens = await self._recover_tokens_after_auth_failure(tokens, context="completion")
                    auth_retry_used = True
                    continue
                if self._is_auth_failure_text(str(exc)):
                    raise OpenAICodexAuthError(str(exc)) from exc
                if self._is_transient_completion_text(str(exc)):
                    if stream_retries_used < self.MAX_RETRIES:
                        delay = self._retry_delay(stream_retries_used)
                        stream_retries_used += 1
                        logger.warning(
                            "OpenAI Codex transient streamed response (retry %s/%s): %s; "
                            "retrying in %.1fs",
                            stream_retries_used,
                            self.MAX_RETRIES,
                            sanitize_provider_error_text(str(exc)),
                            delay,
                        )
                        await asyncio.sleep(delay)
                        continue
                    raise OpenAICodexRequestError(
                        f"OpenAI Codex connection failed after {self.MAX_RETRIES} retries "
                        f"while reading streamed response: {exc}"
                    ) from exc
                raise
            except OAuthUsageLimitError:
                raise
        content, tool_calls = self._extract_output(data)
        message: Dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls

        usage = data.get("usage") or {}
        prompt_tokens = usage.get("input_tokens") or usage.get("prompt_tokens")
        completion_tokens = usage.get("output_tokens") or usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens

        return {
            "id": data.get("id") or "",
            "object": "chat.completion",
            "model": requested_model,
            "choices": [{"index": 0, "message": message, "finish_reason": data.get("status") or "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }

    async def close(self) -> None:
        await self.client.aclose()


openai_codex_client = OpenAICodexClient()
