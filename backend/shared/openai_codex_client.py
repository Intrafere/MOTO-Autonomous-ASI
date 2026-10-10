"""
OpenAI Codex/ChatGPT subscription OAuth client.

This adapter intentionally targets the Codex backend used by ChatGPT account
login flows. It is distinct from the regular OpenAI API-key billing path.
"""
from __future__ import annotations

import base64
import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional
from urllib.parse import urlencode, urlparse, parse_qs

import httpx
from websockets.asyncio.client import connect as websocket_connect
from websockets.exceptions import ConnectionClosed, InvalidStatus, WebSocketException

from backend.shared.log_redaction import redact_log_text
from backend.shared.oauth_pkce import generate_pkce_pair
from backend.shared.openrouter_client import sanitize_provider_error_text
from backend.shared.secret_store import (
    SecretStoreError,
    clear_openai_codex_oauth_tokens,
    load_openai_codex_oauth_tokens,
    store_openai_codex_oauth_tokens,
)
from backend.shared.credential_locks import codex_credential_mutation
from backend.shared.provider_recovery.runtime import assured_retry_wait

logger = logging.getLogger(__name__)


class OpenAICodexError(RuntimeError):
    """Base error for Codex OAuth-backed requests."""


class OpenAICodexAuthError(OpenAICodexError):
    """Raised when Codex OAuth credentials are missing or unusable."""


class OpenAICodexRequestError(OpenAICodexError):
    """Raised when Codex rejects a completion request after authentication."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        error_code: str = "",
        failure_kind: str = "request_rejected",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = str(error_code or "").strip().lower()
        self.failure_kind = str(failure_kind or "request_rejected").strip().lower()


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


class OpenAICodexUntimedUsageLimitError(OpenAICodexRequestError):
    """Raised when Codex reports usage exhaustion without a credible reset."""

    def __init__(self, message: str = "The usage limit has been reached.") -> None:
        super().__init__(
            message,
            error_code="usage_limit_reached",
            failure_kind="untimed_usage_limit",
        )


class OpenAICodexWebsocketTransportError(OpenAICodexError):
    """Raised when the Codex websocket stream ends before a terminal completion.

    Retried exactly like an SSE transport failure: the remote outcome is unknown.
    """


class OpenAICodexWebsocketUnavailableError(OpenAICodexError):
    """Raised when the websocket handshake fails and the request must use SSE instead."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        permanent: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.permanent = permanent


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
    # The Codex CLI streams turns over the Responses websocket; the HTTP/SSE path
    # is closed by the provider at roughly 15 minutes, so long turns need this.
    CODEX_WEBSOCKET_BETA_HEADER = "responses_websockets=2026-02-06"
    WEBSOCKET_OPEN_TIMEOUT_SECONDS = 30.0
    WEBSOCKET_PING_INTERVAL_SECONDS = 20.0
    WEBSOCKET_PING_TIMEOUT_SECONDS = 60.0
    # Handshake statuses that mean the backend does not serve this transport.
    WEBSOCKET_UNSUPPORTED_STATUS_CODES = frozenset({404, 405, 426, 501})
    # Error events that require a fresh connection rather than a provider repair.
    WEBSOCKET_RECONNECT_ERROR_CODES = frozenset({"websocket_connection_limit_reached"})
    # Production default. The test overlay turns this off for the session and
    # websocket tests opt back in on their own instance with a fake connector.
    websocket_transport_enabled = True
    REASONING_EFFORT_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
    REASONING_EFFORT_LEVELS = set(REASONING_EFFORT_ORDER)
    # Exact documented IDs only; provider variants must use their own catalog row.
    KNOWN_MODEL_REASONING = {
        "gpt-5.5": {
            "supported_reasoning_levels": ["low", "medium", "high", "xhigh"],
            "default_reasoning_level": "medium",
        },
    }
    MAX_RETRIES = 4
    RETRY_DELAY = 2.0
    RETRY_MAX_DELAY = 30.0
    LIVE_ERROR_DETAIL_LIMIT = 1800
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
        self._model_reasoning_cache: Dict[str, Dict[str, Any]] = {}
        self._model_catalog_lock = asyncio.Lock()
        self._websocket_disabled_reason: Optional[str] = None
        self.client = httpx.AsyncClient(
            timeout=None,
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=50,
                keepalive_expiry=30.0,
            ),
        )

    generate_pkce_pair = staticmethod(generate_pkce_pair)

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

    @codex_credential_mutation
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

    @codex_credential_mutation
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
    def _normalize_reasoning_metadata(cls, model: Dict[str, Any]) -> Dict[str, Any]:
        """Accept catalog strings or Codex effort/description objects without inventing levels."""
        metadata: Dict[str, Any] = {}
        raw_levels = model.get("supported_reasoning_levels")
        if isinstance(raw_levels, list):
            levels = []
            for entry in raw_levels:
                value = entry.get("effort") if isinstance(entry, dict) else entry
                if isinstance(value, str) and value.strip().lower() in cls.REASONING_EFFORT_LEVELS:
                    effort = value.strip().lower()
                    if effort not in levels:
                        levels.append(effort)
            if levels or not raw_levels:
                metadata["supported_reasoning_levels"] = levels
        default = model.get("default_reasoning_level")
        if isinstance(default, str) and default.strip().lower() in cls.REASONING_EFFORT_LEVELS:
            metadata["default_reasoning_level"] = default.strip().lower()
        return metadata

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
        reasoning_metadata = cls._normalize_reasoning_metadata(model)
        if slug == cls.CODEX_SPARK_HIGH_MODEL_ID:
            reasoning_metadata = {
                "supported_reasoning_levels": ["high"],
                "default_reasoning_level": "high",
            }
        elif not reasoning_metadata and "supported_reasoning_levels" not in model:
            reasoning_metadata = cls.KNOWN_MODEL_REASONING.get(slug, {})
        normalized.update(reasoning_metadata)
        normalized["provider_metadata"].update(reasoning_metadata)
        if context_length:
            normalized["context_length"] = context_length
        if max_output_tokens:
            normalized["max_output_tokens"] = max_output_tokens
        if input_context_window:
            normalized["input_context_window"] = input_context_window
        if effective_input_context_window:
            normalized["effective_input_context_window"] = effective_input_context_window
        return normalized

    @codex_credential_mutation
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
        self._model_reasoning_cache.clear()

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
        self._model_reasoning_cache = {
            row["id"]: self._normalize_reasoning_metadata(row) for row in models
        }
        return models

    async def _model_reasoning_metadata(self, model: str) -> Dict[str, Any]:
        if model in self._model_reasoning_cache:
            return self._model_reasoning_cache[model]
        if model in self.KNOWN_MODEL_REASONING:
            return self.KNOWN_MODEL_REASONING[model]
        async with self._model_catalog_lock:
            if model not in self._model_reasoning_cache:
                await self.list_models()
                self._model_reasoning_cache.setdefault(model, {})
        return self._model_reasoning_cache[model]

    @classmethod
    def _resolve_model_request(cls, model: str, reasoning_effort: Optional[str]) -> tuple[str, Optional[str]]:
        """Map user-facing Codex aliases onto the backend model id/request knobs."""
        if model == cls.CODEX_SPARK_HIGH_MODEL_ID:
            return cls.CODEX_SPARK_MODEL_ID, "high"
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
    def _usage_limit_error_from_payload(
        cls,
        value: Any,
    ) -> Optional[OpenAICodexError]:
        payload = cls._extract_usage_limit_payload(value)
        if payload is None:
            return None
        resets_at = cls._coerce_positive_int(payload.get("resets_at"))
        resets_in_seconds = cls._coerce_positive_int(payload.get("resets_in_seconds"))
        if resets_at is None and resets_in_seconds is None:
            return OpenAICodexUntimedUsageLimitError(
                str(payload.get("message") or "The usage limit has been reached.").strip()
            )
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
    def _usage_limit_error_from_text(cls, text: str) -> Optional[OpenAICodexError]:
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
            return OpenAICodexUntimedUsageLimitError()
        return None

    @staticmethod
    def _error_code_from_value(value: Any) -> str:
        if not isinstance(value, dict):
            return ""
        for key in ("code", "type"):
            code = value.get(key)
            if code not in (None, ""):
                return str(code).strip().lower()
        for key in ("error", "response"):
            nested = value.get(key)
            code = OpenAICodexClient._error_code_from_value(nested)
            if code:
                return code
        return ""

    @classmethod
    def _error_code_from_text(cls, text: str) -> str:
        raw = str(text or "")
        start = raw.find("{")
        end = raw.rfind("}")
        if 0 <= start < end:
            try:
                return cls._error_code_from_value(json.loads(raw[start : end + 1]))
            except json.JSONDecodeError:
                return ""
        return ""

    @classmethod
    def _max_attempts(cls) -> int:
        return cls.MAX_RETRIES + 1

    @classmethod
    def _retry_delay(cls, retry_index: int) -> float:
        return min(cls.RETRY_MAX_DELAY, cls.RETRY_DELAY * (2 ** max(0, retry_index)))

    @classmethod
    def _reasoning_config(
        cls, reasoning_effort: Optional[str], metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, str]]:
        if not reasoning_effort:
            return None
        effort = str(reasoning_effort).strip().lower()
        supported = (metadata or {}).get("supported_reasoning_levels", [])
        if effort in {"auto", "maximum", "highest"}:
            if metadata is not None and metadata.get("supported_reasoning_levels") == []:
                return None
            effort = next((level for level in reversed(cls.REASONING_EFFORT_ORDER) if level in supported), "")
            if not effort:
                raise OpenAICodexRequestError(
                    "OpenAI Codex reasoning capabilities are unknown for this model. "
                    "Refresh the model catalog or select an explicit reasoning effort.",
                    error_code="reasoning_capabilities_unknown", failure_kind="configuration",
                )
        if effort not in cls.REASONING_EFFORT_LEVELS:
            raise OpenAICodexRequestError(
                "Unsupported OpenAI Codex reasoning effort. Select a catalog-supported effort.",
                error_code="unsupported_reasoning_effort", failure_kind="configuration",
            )
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
            raise OpenAICodexRequestError(
                "OpenAI Codex completion failed: empty response body",
                failure_kind="empty_response",
            )

        try:
            data = json.loads(body)
            if isinstance(data, dict):
                usage_error = cls._usage_limit_error_from_payload(data)
                if usage_error is not None:
                    raise usage_error
                cls._validate_plain_response_status(data)
                return data
        except json.JSONDecodeError:
            logger.debug("OpenAI Codex response body is not plain JSON; parsing stream events")

        state = cls._new_stream_state()
        for event_data in cls._iter_sse_data(body):
            cls._consume_stream_event_data(event_data, state)
        return cls._finalize_stream_state(state)

    @staticmethod
    def _new_stream_state() -> Dict[str, Any]:
        return {
            "response_data": None,
            "output_text_parts": [],
            "valid_events": 0,
            "malformed_events": 0,
            "terminal_seen": False,
            "done_seen": False,
        }

    @classmethod
    def _validate_plain_response_status(cls, data: Dict[str, Any]) -> None:
        status = str(data.get("status") or "").strip().lower()
        if status in {"failed", "incomplete", "cancelled", "canceled"}:
            cls._raise_terminal_response_error(data, status=status, source="JSON response")
        if status in {"queued", "in_progress"}:
            raise OpenAICodexRequestError(
                f"OpenAI Codex completion did not reach a terminal state (status={status}).",
                failure_kind="terminal_missing",
            )

    @classmethod
    def _terminal_error_detail(cls, value: Dict[str, Any], *, status: str) -> tuple[str, str]:
        error = value.get("error")
        incomplete = value.get("incomplete_details")
        detail_value: Any = error or incomplete or {"status": status}
        error_code = cls._error_code_from_value(error or incomplete or value)
        if isinstance(incomplete, dict):
            reason = str(incomplete.get("reason") or "").strip()
            if reason:
                error_code = error_code or reason.lower()
                detail_value = {"status": status, "reason": reason}
        safe_detail = sanitize_provider_error_text(
            json.dumps(detail_value, ensure_ascii=False),
            max_chars=cls.LIVE_ERROR_DETAIL_LIMIT,
        )
        return safe_detail, error_code

    @classmethod
    def _raise_terminal_response_error(
        cls,
        value: Dict[str, Any],
        *,
        status: str,
        source: str,
    ) -> None:
        usage_error = cls._usage_limit_error_from_payload(value)
        if usage_error is not None:
            raise usage_error
        detail, error_code = cls._terminal_error_detail(value, status=status)
        failure_kind = (
            "output_truncated"
            if status == "incomplete" and error_code == "max_output_tokens"
            else "response_incomplete"
            if status == "incomplete"
            else "stream_rejected"
        )
        raise OpenAICodexRequestError(
            f"OpenAI Codex {source} ended with status={status}. Detail: {detail}",
            error_code=error_code,
            failure_kind=failure_kind,
        )

    @classmethod
    def _consume_stream_event_data(cls, event_data: str, state: Dict[str, Any]) -> None:
        if event_data == "[DONE]":
            state["done_seen"] = True
            return
        event = cls._parse_stream_event(event_data, state, transport="SSE")
        if event is not None:
            cls._consume_stream_event(event, state)

    @classmethod
    def _consume_websocket_event_text(cls, text: str, state: Dict[str, Any]) -> None:
        event = cls._parse_stream_event(text, state, transport="websocket")
        if event is None:
            return
        if str(event.get("type") or "").strip() == "error":
            error_value = event.get("error")
            code = cls._error_code_from_value(error_value if isinstance(error_value, dict) else event)
            if code in cls.WEBSOCKET_RECONNECT_ERROR_CODES:
                raise OpenAICodexWebsocketTransportError(
                    f"OpenAI Codex websocket reported {code}; the turn must be retried on a new connection."
                )
        cls._consume_stream_event(event, state)

    @classmethod
    def _parse_stream_event(
        cls,
        event_data: str,
        state: Dict[str, Any],
        *,
        transport: str,
    ) -> Optional[Dict[str, Any]]:
        try:
            event = json.loads(event_data)
        except json.JSONDecodeError as exc:
            state["malformed_events"] += 1
            logger.warning(
                "Ignoring malformed OpenAI Codex %s event (length=%s, json_error=%s at %s); raw event omitted",
                transport,
                len(event_data),
                type(exc).__name__,
                exc.pos,
            )
            return None
        if not isinstance(event, dict):
            return None
        return event

    @classmethod
    def _consume_stream_event(cls, event: Dict[str, Any], state: Dict[str, Any]) -> None:
        state["valid_events"] += 1
        event_type = str(event.get("type") or "").strip()
        response = event.get("response")
        response_dict = response if isinstance(response, dict) else {}

        if event_type == "error":
            usage_error = cls._usage_limit_error_from_payload(event)
            if usage_error is not None:
                raise usage_error
            error_value = event.get("error")
            structured_error = error_value if isinstance(error_value, dict) else event
            safe_detail = sanitize_provider_error_text(
                json.dumps(structured_error, ensure_ascii=False),
                max_chars=cls.LIVE_ERROR_DETAIL_LIMIT,
            )
            raise OpenAICodexRequestError(
                f"OpenAI Codex stream error event. Detail: {safe_detail}",
                error_code=cls._error_code_from_value(structured_error),
                failure_kind="stream_rejected",
            )

        if event_type in {"response.failed", "response.incomplete"}:
            status = "failed" if event_type == "response.failed" else "incomplete"
            terminal_value = response_dict or event
            cls._raise_terminal_response_error(
                terminal_value,
                status=status,
                source=f"SSE event {event_type}",
            )

        if event_type == "response.output_text.delta":
            state["output_text_parts"].append(str(event.get("delta") or ""))
        elif event_type == "response.output_text.done" and not state["output_text_parts"]:
            text = event.get("text")
            if text:
                state["output_text_parts"].append(str(text))

        if response_dict:
            state["response_data"] = response_dict

        if event_type == "response.completed":
            if not response_dict:
                raise OpenAICodexRequestError(
                    "OpenAI Codex response.completed event did not include a response object.",
                    failure_kind="invalid_terminal_state",
                )
            status = str(response_dict.get("status") or "").strip().lower()
            if status and status != "completed":
                raise OpenAICodexRequestError(
                    "OpenAI Codex response.completed event carried a non-completed "
                    f"response status ({status or 'missing'}).",
                    failure_kind="invalid_terminal_state",
                )
            state["terminal_seen"] = True

    @classmethod
    def _finalize_stream_state(cls, state: Dict[str, Any]) -> Dict[str, Any]:
        if not state["terminal_seen"]:
            raise OpenAICodexRequestError(
                "OpenAI Codex stream ended without response.completed "
                f"(valid_events={state['valid_events']}, malformed_events={state['malformed_events']}, "
                f"output_characters={sum(len(part) for part in state['output_text_parts'])}, "
                f"done_seen={bool(state['done_seen'])}). Remote outcome is unknown.",
                failure_kind="terminal_missing",
            )
        response_data = dict(state["response_data"] or {})
        if state["output_text_parts"] and not response_data.get("output_text"):
            response_data["output_text"] = "".join(state["output_text_parts"])
        return response_data

    @classmethod
    async def _iter_stream_data(cls, response: httpx.Response) -> AsyncIterator[str]:
        data_lines: List[str] = []
        async for line in response.aiter_lines():
            stripped = line.rstrip("\r")
            if not stripped:
                if data_lines:
                    yield "\n".join(data_lines)
                    data_lines = []
                continue
            if stripped.startswith(":"):
                continue
            if stripped.startswith("data:"):
                data_lines.append(stripped[5:].lstrip())
        if data_lines:
            yield "\n".join(data_lines)

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

    async def _post_with_retry(
        self,
        url: str,
        *,
        retry_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
        retry_model: str = "",
        **kwargs,
    ) -> httpx.Response:
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
                        if retry_callback is not None:
                            await retry_callback({
                                "provider": "openai_codex_oauth",
                                "provider_label": "OpenAI Codex",
                                "model": retry_model,
                                "retry_attempt": attempt + 1,
                                "max_attempts": max_attempts,
                                "retry_after_seconds": delay,
                                "reason": "transient_http_error",
                                "error_summary": error_detail,
                            })
                        else:
                            await assured_retry_wait(
                                {
                                    "provider": "openai_codex_oauth",
                                    "provider_label": "OpenAI Codex",
                                    "model": retry_model,
                                    "retry_attempt": attempt + 1,
                                    "max_attempts": max_attempts,
                                    "retry_after_seconds": delay,
                                    "reason": "transient_http_error",
                                    "error_summary": error_detail,
                                },
                                role_id="openai_codex_direct",
                            )
                        continue
                    raise OpenAICodexRequestError(
                        f"OpenAI Codex connection failed after {self.MAX_RETRIES} retries: "
                        f"HTTP {response.status_code}: {error_detail}",
                        status_code=response.status_code,
                        error_code=self._error_code_from_text(response.text),
                        failure_kind="transient_http_exhausted",
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
                    if retry_callback is not None:
                        await retry_callback({
                            "provider": "openai_codex_oauth",
                            "provider_label": "OpenAI Codex",
                            "model": retry_model,
                            "retry_attempt": attempt + 1,
                            "max_attempts": max_attempts,
                            "retry_after_seconds": delay,
                            "reason": "connection_error",
                            "error_summary": f"[{error_type}] {error_detail}",
                        })
                    else:
                        await assured_retry_wait(
                            {
                                "provider": "openai_codex_oauth",
                                "provider_label": "OpenAI Codex",
                                "model": retry_model,
                                "retry_attempt": attempt + 1,
                                "max_attempts": max_attempts,
                                "retry_after_seconds": delay,
                                "reason": "connection_error",
                                "error_summary": f"[{error_type}] {error_detail}",
                            },
                            role_id="openai_codex_direct",
                        )
                    continue
                raise OpenAICodexRequestError(
                    f"OpenAI Codex connection failed after {self.MAX_RETRIES} retries: "
                    f"[{error_type}] {error_detail}",
                    failure_kind="transport_exhausted",
                )

    @classmethod
    def _websocket_url(cls) -> str:
        base = cls.CODEX_BASE_URL
        for http_scheme, ws_scheme in (("https://", "wss://"), ("http://", "ws://")):
            if base.startswith(http_scheme):
                base = ws_scheme + base[len(http_scheme):]
                break
        return f"{base}/responses"

    @classmethod
    def _websocket_headers(cls, headers: Dict[str, str]) -> Dict[str, str]:
        """Handshake headers: the auth/account headers plus the websocket beta marker."""
        ws_headers = {
            name: value
            for name, value in headers.items()
            if name.lower() not in {"accept", "content-type"}
        }
        ws_headers["OpenAI-Beta"] = cls.CODEX_WEBSOCKET_BETA_HEADER
        ws_headers["originator"] = cls.DEFAULT_ORIGINATOR
        return ws_headers

    @staticmethod
    def _websocket_request_text(payload: Dict[str, Any]) -> str:
        # The backend accepts the HTTP request body plus the event type, including
        # stream=true, which is how the Codex CLI serializes websocket turns.
        return json.dumps({"type": "response.create", **payload}, ensure_ascii=False)

    def _websocket_transport_allowed(self) -> bool:
        return bool(self.websocket_transport_enabled) and self._websocket_disabled_reason is None

    def _open_websocket(self, url: str, headers: Dict[str, str]):
        """Return the websocket connection context manager; tests inject a fake here."""
        return websocket_connect(
            url,
            additional_headers=headers,
            open_timeout=self.WEBSOCKET_OPEN_TIMEOUT_SECONDS,
            ping_interval=self.WEBSOCKET_PING_INTERVAL_SECONDS,
            ping_timeout=self.WEBSOCKET_PING_TIMEOUT_SECONDS,
            max_size=None,
        )

    def _note_websocket_unavailable(self, exc: OpenAICodexWebsocketUnavailableError) -> None:
        detail = sanitize_provider_error_text(str(exc), max_chars=self.LIVE_ERROR_DETAIL_LIMIT)
        if exc.permanent:
            if self._websocket_disabled_reason is None:
                self._websocket_disabled_reason = detail
                logger.warning(
                    "OpenAI Codex websocket transport is unavailable for this process (%s); "
                    "using the SSE transport for later requests.",
                    detail,
                )
            return
        logger.warning(
            "OpenAI Codex websocket transport is unavailable for this request (%s); "
            "using the SSE transport.",
            detail,
        )

    @staticmethod
    def _websocket_close_code(exc: ConnectionClosed) -> Optional[int]:
        for frame in (getattr(exc, "rcvd", None), getattr(exc, "sent", None)):
            code = getattr(frame, "code", None)
            if code is not None:
                return int(code)
        return None

    @staticmethod
    def _transport_drop_guidance(transport: str, elapsed: float) -> str:
        """Plain-language cause and repair hint for a stream cut before completion."""
        minutes = max(1, round(elapsed / 60.0))
        duration = "under a minute" if elapsed < 45 else f"about {minutes} minute{'s' if minutes != 1 else ''}"
        if transport == "websocket":
            cause = (
                "What this means: OpenAI's Codex service closed this connection on its side after "
                f"{duration} while the model was still working. This is a server-side connection refresh "
                "at OpenAI, not a problem with your computer, your network, or MOTO's connection, which was healthy."
            )
        else:
            cause = (
                "What this means: OpenAI's Codex service closed this HTTP stream on its side after "
                f"{duration} while the model was still working. OpenAI ends these streams at about 15 minutes; "
                "this is not a problem with your computer or network."
            )
        return (
            f"{cause} MOTO will send the request again, but a request that needs more thinking time than "
            "the connection stays open will keep getting cut off. How to fix: lower this role's reasoning "
            "effort in Settings so each call finishes sooner, or route this role through a different provider."
        )

    @staticmethod
    async def _monitor_event_loop_stalls(stall: Dict[str, float], interval: float = 1.0) -> None:
        """Record the longest event-loop stall while a websocket turn is open.

        Keepalive pongs are sent from this process's event loop, so a stall longer
        than the server's pong deadline looks to the server like a dead client.
        """
        loop = asyncio.get_running_loop()
        while True:
            started = loop.time()
            await asyncio.sleep(interval)
            lag = loop.time() - started - interval
            if lag > stall["max"]:
                stall["max"] = lag

    async def _websocket_completion_attempt(
        self,
        *,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run one response.create turn over the Responses websocket.

        Raises OpenAICodexWebsocketUnavailableError when the handshake fails (the
        caller falls back to SSE for this attempt), OpenAICodexWebsocketTransportError
        when the socket ends before response.completed (the caller retries), and
        OpenAICodexRequestError for provider-reported terminal failures, exactly like
        the SSE path.
        """
        request_text = self._websocket_request_text(payload)
        handshake_complete = False
        active_connection = None
        last_event_at = time.monotonic()
        stall: Dict[str, float] = {"max": 0.0}
        stall_monitor = asyncio.create_task(self._monitor_event_loop_stalls(stall))
        try:
            return await self._run_websocket_turn(
                request_text=request_text,
                headers=headers,
                state=state,
                stall=stall,
            )
        finally:
            stall_monitor.cancel()
            try:
                await stall_monitor
            except (asyncio.CancelledError, Exception):
                pass

    async def _run_websocket_turn(
        self,
        *,
        request_text: str,
        headers: Dict[str, str],
        state: Dict[str, Any],
        stall: Dict[str, float],
    ) -> Dict[str, Any]:
        handshake_complete = False
        active_connection = None
        last_event_at = time.monotonic()
        try:
            async with self._open_websocket(
                self._websocket_url(),
                self._websocket_headers(headers),
            ) as connection:
                handshake_complete = True
                active_connection = connection
                await connection.send(request_text)
                async for message in connection:
                    last_event_at = time.monotonic()
                    if isinstance(message, (bytes, bytearray)):
                        raise OpenAICodexWebsocketTransportError(
                            "OpenAI Codex websocket sent an unexpected binary frame before response.completed."
                        )
                    self._consume_websocket_event_text(message, state)
                    if state["terminal_seen"]:
                        break
        except asyncio.CancelledError:
            raise
        except ConnectionClosed as exc:
            if state["terminal_seen"]:
                logger.warning(
                    "OpenAI Codex websocket closed after response.completed; "
                    "using the confirmed terminal response (close_code=%s)",
                    self._websocket_close_code(exc),
                )
                return self._finalize_stream_state(state)
            raise OpenAICodexWebsocketTransportError(
                f"[{type(exc).__name__}] websocket closed (close_code={self._websocket_close_code(exc)}) "
                f"before response.completed (events={state['valid_events']}, "
                f"idle_seconds={time.monotonic() - last_event_at:.1f}, "
                f"max_loop_stall_seconds={stall['max']:.1f})"
            ) from exc
        except InvalidStatus as exc:
            status = int(exc.response.status_code)
            raise OpenAICodexWebsocketUnavailableError(
                f"websocket handshake rejected with HTTP {status}",
                status_code=status,
                permanent=status in self.WEBSOCKET_UNSUPPORTED_STATUS_CODES,
            ) from exc
        except (WebSocketException, OSError, TimeoutError, asyncio.TimeoutError) as exc:
            detail = f"[{type(exc).__name__}] {exc}"
            if not handshake_complete:
                raise OpenAICodexWebsocketUnavailableError(
                    f"websocket connection failed: {detail}"
                ) from exc
            raise OpenAICodexWebsocketTransportError(detail) from exc
        if not state["terminal_seen"]:
            close_code = getattr(active_connection, "close_code", None)
            close_reason = str(getattr(active_connection, "close_reason", None) or "")
            idle_seconds = time.monotonic() - last_event_at
            event_count = state["valid_events"]
            raise OpenAICodexWebsocketTransportError(
                "websocket closed by server before response.completed "
                f"(close_code={close_code}, close_reason={close_reason!r}, "
                f"events={event_count}, idle_seconds={idle_seconds:.1f}, "
                f"max_loop_stall_seconds={stall['max']:.1f})"
            )
        return self._finalize_stream_state(state)

    async def _stream_completion_with_retry(
        self,
        url: str,
        *,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        retry_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]],
        retry_model: str,
        request_attempt_start: int = 0,
    ) -> tuple[Optional[Dict[str, Any]], int, str]:
        """Consume a Codex turn incrementally, websocket first, retrying transport/HTTP failures.

        A rejected websocket handshake falls back to the HTTP/SSE stream for that
        attempt; a websocket or SSE stream that ends before response.completed is
        retried sequentially with the transport named in the activity detail.
        """
        # Compatibility for narrow injected test transports. The production
        # httpx.AsyncClient always exposes stream(), so runtime Codex traffic
        # cannot enter this buffered adapter.
        if not hasattr(self.client, "stream"):
            response = await self._post_with_retry(
                url,
                retry_callback=retry_callback,
                retry_model=retry_model,
                json=payload,
                headers=headers,
            )
            if hasattr(response, "aread"):
                raw_body = (await response.aread()).decode("utf-8", errors="replace")
            else:
                raw_body = str(getattr(response, "text", "") or "")
            if response.status_code >= 400:
                return None, response.status_code, raw_body
            return self._decode_response_body(raw_body), response.status_code, ""

        max_attempts = self._max_attempts()
        for attempt in range(request_attempt_start, max_attempts):
            state = self._new_stream_state()
            started_at = time.monotonic()
            transport = "websocket" if self._websocket_transport_allowed() else "SSE"
            try:
                if transport == "websocket":
                    try:
                        data = await self._websocket_completion_attempt(
                            payload=payload,
                            headers=headers,
                            state=state,
                        )
                        return data, 200, ""
                    except OpenAICodexWebsocketUnavailableError as exc:
                        self._note_websocket_unavailable(exc)
                        transport = "SSE"
                        state = self._new_stream_state()
                        started_at = time.monotonic()
                async with self.client.stream(
                    "POST",
                    url,
                    json=payload,
                    headers=headers,
                ) as response:
                    if response.status_code >= 400:
                        raw_body = (await response.aread()).decode("utf-8", errors="replace")
                        if self._is_transient_completion_response_text(response.status_code, raw_body):
                            detail = sanitize_provider_error_text(
                                raw_body,
                                max_chars=self.LIVE_ERROR_DETAIL_LIMIT,
                            )
                            if attempt < max_attempts - 1:
                                delay = self._retry_delay(attempt)
                                await self._notify_stream_retry(
                                    retry_callback,
                                    retry_model=retry_model,
                                    attempt=attempt,
                                    max_attempts=max_attempts,
                                    delay=delay,
                                    reason="transient_http_error",
                                    detail=(
                                        f"HTTP {response.status_code} transient response. Detail: {detail}"
                                    ),
                                )
                                if retry_callback is None:
                                    await assured_retry_wait(
                                        {
                                            "provider": "openai_codex_oauth",
                                            "provider_label": "OpenAI Codex",
                                            "model": retry_model,
                                            "retry_attempt": attempt + 1,
                                            "max_attempts": max_attempts,
                                            "retry_after_seconds": delay,
                                            "reason": "transient_http_error",
                                            "error_summary": detail,
                                        },
                                        role_id="openai_codex_direct",
                                    )
                                continue
                            raise OpenAICodexRequestError(
                                f"OpenAI Codex connection failed after {self.MAX_RETRIES} retries: "
                                f"HTTP {response.status_code}: {detail}",
                                status_code=response.status_code,
                                error_code=self._error_code_from_text(raw_body),
                                failure_kind="transient_http_exhausted",
                            )
                        return None, response.status_code, raw_body

                    content_type = str(response.headers.get("content-type") or "").lower()
                    if "application/json" in content_type:
                        raw_body = (await response.aread()).decode("utf-8", errors="replace")
                        return self._decode_response_body(raw_body), response.status_code, ""

                    try:
                        async for event_data in self._iter_stream_data(response):
                            self._consume_stream_event_data(event_data, state)
                    except httpx.TransportError as exc:
                        if state["terminal_seen"]:
                            logger.warning(
                                "OpenAI Codex transport ended after response.completed; "
                                "using the confirmed terminal response (error=%s)",
                                sanitize_provider_error_text(
                                    f"[{type(exc).__name__}] {exc}",
                                    max_chars=self.LIVE_ERROR_DETAIL_LIMIT,
                                ),
                            )
                            return self._finalize_stream_state(state), response.status_code, ""
                        raise
                    return self._finalize_stream_state(state), response.status_code, ""
            except asyncio.CancelledError:
                raise
            except (httpx.TransportError, OpenAICodexWebsocketTransportError) as exc:
                elapsed = time.monotonic() - started_at
                detail = sanitize_provider_error_text(
                    f"[{type(exc).__name__}] {exc}",
                    max_chars=self.LIVE_ERROR_DETAIL_LIMIT,
                )
                terminal_seen = bool(state["terminal_seen"])
                outcome = (
                    "A terminal completion was received."
                    if terminal_seen
                    else "No terminal completion was received; the remote outcome is unknown."
                )
                safe_detail = (
                    f"A remote peer or network intermediary closed the Codex {transport} stream after "
                    f"{elapsed:.1f}s. {outcome} Detail: {detail}"
                )
                if terminal_seen:
                    return self._finalize_stream_state(state), 200, ""
                safe_detail = f"{safe_detail} {self._transport_drop_guidance(transport, elapsed)}"
                if attempt < max_attempts - 1:
                    delay = self._retry_delay(attempt)
                    await self._notify_stream_retry(
                        retry_callback,
                        retry_model=retry_model,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        delay=delay,
                        reason="connection_error",
                        detail=safe_detail,
                    )
                    if retry_callback is None:
                        await assured_retry_wait(
                            {
                                "provider": "openai_codex_oauth",
                                "provider_label": "OpenAI Codex",
                                "model": retry_model,
                                "retry_attempt": attempt + 1,
                                "max_attempts": max_attempts,
                                "retry_after_seconds": delay,
                                "reason": "connection_error",
                                "error_summary": safe_detail,
                            },
                            role_id="openai_codex_direct",
                        )
                    continue
                raise OpenAICodexRequestError(
                    f"OpenAI Codex connection failed after {self.MAX_RETRIES} retries: {safe_detail}",
                    failure_kind="transport_exhausted",
                ) from exc
        raise AssertionError("unreachable Codex stream retry state")

    @classmethod
    def _is_transient_completion_response_text(cls, status_code: int, body: str) -> bool:
        return status_code in cls.TRANSIENT_COMPLETION_STATUS_CODES

    async def _notify_stream_retry(
        self,
        retry_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]],
        *,
        retry_model: str,
        attempt: int,
        max_attempts: int,
        delay: float,
        reason: str,
        detail: str,
    ) -> None:
        safe_detail = sanitize_provider_error_text(
            detail,
            max_chars=self.LIVE_ERROR_DETAIL_LIMIT,
        )
        logger.warning(
            "OpenAI Codex request attempt %s/%s failed; retrying in %.1fs: %s",
            attempt + 1,
            max_attempts,
            delay,
            safe_detail,
        )
        if retry_callback is not None:
            await retry_callback({
                "provider": "openai_codex_oauth",
                "provider_label": "OpenAI Codex",
                "model": retry_model,
                "retry_attempt": attempt + 1,
                "max_attempts": max_attempts,
                "retry_after_seconds": delay,
                "reason": reason,
                "error_summary": safe_detail,
                "message": (
                    f"OpenAI Codex request attempt {attempt + 1}/{max_attempts} failed. "
                    f"{safe_detail} Retrying in {delay:g} seconds."
                ),
            })

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
        retry_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
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
        reasoning_metadata = None
        if str(reasoning_effort or "").strip().lower() in {"auto", "maximum", "highest"}:
            reasoning_metadata = await self._model_reasoning_metadata(requested_model)
        reasoning = self._reasoning_config(reasoning_effort, reasoning_metadata)
        if reasoning:
            # Reasoning summaries keep events flowing during long thinking phases so
            # the websocket (or SSE) stream is never idle for minutes at a time.
            payload["reasoning"] = {**reasoning, "summary": "auto"}
        text_format = self._response_format(response_format)
        if text_format:
            payload["text"] = text_format
        converted_tools = self._convert_tools(tools)
        if converted_tools:
            payload["tools"] = converted_tools
            if tool_choice is not None:
                payload["tool_choice"] = "auto" if tool_choice == "auto" else tool_choice

        auth_retry_used = False
        request_attempt_start = 0
        while True:
            try:
                data, status_code, response_text = await self._stream_completion_with_retry(
                    f"{self.CODEX_BASE_URL}/responses",
                    payload=payload,
                    headers=self._headers(tokens, accept_stream=True),
                    retry_callback=retry_callback,
                    retry_model=requested_model,
                    request_attempt_start=request_attempt_start,
                )
                if status_code >= 400:
                    usage_error = self._usage_limit_error_from_text(response_text)
                    if usage_error is not None:
                        raise usage_error
                    message = (
                        "OpenAI Codex completion failed: "
                        f"{sanitize_provider_error_text(response_text, max_chars=self.LIVE_ERROR_DETAIL_LIMIT)}"
                    )
                    if status_code in {401, 403}:
                        if not auth_retry_used:
                            tokens = await self._recover_tokens_after_auth_failure(tokens, context="completion")
                            auth_retry_used = True
                            continue
                        raise OpenAICodexAuthError(message)
                    raise OpenAICodexRequestError(
                        message,
                        status_code=status_code,
                        error_code=self._error_code_from_text(response_text),
                        failure_kind="http_rejected",
                    )
                if data is None:
                    raise OpenAICodexRequestError(
                        "OpenAI Codex stream returned no response data.",
                        failure_kind="empty_stream",
                    )
                break
            except OpenAICodexRequestError as exc:
                if self._is_auth_failure_text(str(exc)) and not auth_retry_used:
                    tokens = await self._recover_tokens_after_auth_failure(tokens, context="completion")
                    auth_retry_used = True
                    continue
                if self._is_auth_failure_text(str(exc)):
                    raise OpenAICodexAuthError(str(exc)) from exc
                if exc.failure_kind in {
                    "response_incomplete",
                    "transient_http_exhausted",
                    "transport_exhausted",
                }:
                    raise
                if self._is_transient_completion_text(str(exc)):
                    if request_attempt_start < self.MAX_RETRIES:
                        failed_attempt = request_attempt_start
                        delay = self._retry_delay(failed_attempt)
                        request_attempt_start += 1
                        logger.warning(
                            "OpenAI Codex transient streamed response (request attempt %s/%s): %s; "
                            "retrying in %.1fs",
                            failed_attempt + 1,
                            self._max_attempts(),
                            sanitize_provider_error_text(str(exc)),
                            delay,
                        )
                        if retry_callback is not None:
                            safe_detail = sanitize_provider_error_text(
                                str(exc),
                                max_chars=self.LIVE_ERROR_DETAIL_LIMIT,
                            )
                            await retry_callback({
                                "provider": "openai_codex_oauth",
                                "provider_label": "OpenAI Codex",
                                "model": requested_model,
                                "retry_attempt": failed_attempt + 1,
                                "max_attempts": self._max_attempts(),
                                "retry_after_seconds": delay,
                                "reason": "stream_error",
                                "error_summary": safe_detail,
                                "message": (
                                    f"OpenAI Codex returned a transient stream error. "
                                    f"Detail: {safe_detail} Retrying in {delay:g} seconds "
                                    f"(request attempt {failed_attempt + 1}/{self._max_attempts()})."
                                ),
                            })
                        else:
                            await assured_retry_wait(
                                {
                                    "provider": "openai_codex_oauth",
                                    "provider_label": "OpenAI Codex",
                                    "model": requested_model,
                                    "retry_attempt": failed_attempt + 1,
                                    "max_attempts": self._max_attempts(),
                                    "retry_after_seconds": delay,
                                    "reason": "stream_error",
                                    "error_summary": sanitize_provider_error_text(
                                        str(exc),
                                        max_chars=self.LIVE_ERROR_DETAIL_LIMIT,
                                    ),
                                },
                                role_id="openai_codex_direct",
                            )
                        continue
                    raise OpenAICodexRequestError(
                        f"OpenAI Codex connection failed after {self.MAX_RETRIES} retries "
                        f"while reading streamed response: {exc}",
                        error_code=exc.error_code,
                        failure_kind="transient_stream_exhausted",
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
