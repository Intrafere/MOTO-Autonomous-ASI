"""
xAI Grok/SuperGrok subscription OAuth client.

This adapter is intentionally separate from the pay-as-you-go xAI API-key path.
It uses locally stored OAuth tokens and exposes an OpenAI-compatible
Chat Completions response shape to the rest of MOTO.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import secrets
import time
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from backend.shared.log_redaction import redact_log_text
from backend.shared.openrouter_client import sanitize_provider_error_text
from backend.shared.secret_store import (
    SecretStoreError,
    clear_xai_grok_oauth_tokens,
    load_xai_grok_oauth_tokens,
    store_xai_grok_oauth_tokens,
)

logger = logging.getLogger(__name__)


class XAIGrokError(RuntimeError):
    """Base error for xAI Grok OAuth-backed requests."""


class XAIGrokAuthError(XAIGrokError):
    """Raised when xAI Grok OAuth credentials are missing or unusable."""


class XAIGrokRequestError(XAIGrokError):
    """Raised when xAI rejects a request after authentication."""


class XAIGrokClient:
    """Client for xAI Grok OAuth and the xAI OpenAI-compatible API surface."""

    CLIENT_ID = os.getenv("MOTO_XAI_GROK_OAUTH_CLIENT_ID", "b1a00492-073a-47ea-816f-4c329264a828")
    AUTH_URL = os.getenv("MOTO_XAI_GROK_AUTH_URL", "https://auth.x.ai/oauth2/authorize")
    TOKEN_URL = os.getenv("MOTO_XAI_GROK_TOKEN_URL", "https://auth.x.ai/oauth2/token")
    REVOKE_URL = os.getenv("MOTO_XAI_GROK_REVOKE_URL", "https://auth.x.ai/oauth2/revoke")
    API_BASE_URL = os.getenv("MOTO_XAI_GROK_BASE_URL", "https://api.x.ai/v1").rstrip("/")
    DEFAULT_REDIRECT_URI = os.getenv("MOTO_XAI_GROK_REDIRECT_URI", "http://127.0.0.1:56121/callback")
    DEFAULT_MODEL = os.getenv("MOTO_XAI_GROK_DEFAULT_MODEL", "grok-4.3")
    DEFAULT_SCOPE = os.getenv(
        "MOTO_XAI_GROK_OAUTH_SCOPE",
        "openid profile email offline_access grok-cli:access api:access",
    )
    DEFAULT_PLAN = os.getenv("MOTO_XAI_GROK_OAUTH_PLAN", "generic")
    DEFAULT_REFERRER = os.getenv("MOTO_XAI_GROK_OAUTH_REFERRER", "moto")
    REFRESH_SKEW_SECONDS = 120
    MAX_RETRIES = 3
    RETRY_DELAY = 2.0
    TRANSIENT_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524}
    TRANSIENT_MARKERS = (
        "bad gateway",
        "connection timeout",
        "gateway timeout",
        "peer closed connection",
        "service unavailable",
        "temporarily unavailable",
        "upstream provider timeout",
    )
    CHAT_UNSUPPORTED_MODEL_MARKERS = (
        "multi-agent",
        "multi_agent",
    )
    KNOWN_MODEL_LIMITS = {
        # Public Grok OAuth integrations currently expose Grok 4.3 as a 1M
        # context subscription model. Provider metadata remains authoritative.
        "grok-4": {
            "context_length": 1000000,
            "max_output_tokens": 131072,
        },
        "grok-4.2": {
            "context_length": 1000000,
            "max_output_tokens": 131072,
        },
        "grok-4.3": {
            "context_length": 1000000,
            "max_output_tokens": 131072,
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
        nonce: str = "",
        redirect_uri: str = DEFAULT_REDIRECT_URI,
    ) -> str:
        """Build the xAI OAuth authorization URL for Grok subscription login."""
        params = {
            "response_type": "code",
            "client_id": cls.CLIENT_ID,
            "redirect_uri": redirect_uri,
            "scope": cls.DEFAULT_SCOPE,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "state": state,
        }
        if nonce:
            params["nonce"] = nonce
        if cls.DEFAULT_PLAN:
            params["plan"] = cls.DEFAULT_PLAN
        if cls.DEFAULT_REFERRER:
            params["referrer"] = cls.DEFAULT_REFERRER
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
        jwt_payload = cls._jwt_payload(id_token or access_token)
        expires_at = payload.get("expires_at") or payload.get("expires")
        if not expires_at and jwt_payload.get("exp"):
            expires_at = int(jwt_payload["exp"])
        elif payload.get("expires_in"):
            expires_at = int(time.time()) + int(payload["expires_in"])

        account_id = (
            payload.get("account_id")
            or payload.get("accountId")
            or jwt_payload.get("sub")
            or jwt_payload.get("account_id")
        )
        email = payload.get("email") or jwt_payload.get("email")

        normalized = {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "id_token": id_token,
            "expires_at": int(expires_at or 0),
            "account_id": str(account_id or ""),
            "email": str(email or ""),
            "provider": "xai_grok_oauth",
            "updated_at": int(time.time()),
        }
        return {key: value for key, value in normalized.items() if value not in ("", None)}

    @staticmethod
    def _access_token(tokens: Optional[Dict[str, Any]]) -> str:
        return str((tokens or {}).get("access_token") or "")

    @classmethod
    def _is_auth_failure_text(cls, text: str) -> bool:
        lowered = (text or "").lower()
        return (
            "invalid token" in lowered
            or "invalid_grant" in lowered
            or "expired token" in lowered
            or "token expired" in lowered
            or "unauthorized" in lowered
        )

    @classmethod
    def _is_entitlement_failure(cls, text: str) -> bool:
        lowered = (text or "").lower()
        return (
            "forbidden" in lowered
            or "entitlement" in lowered
            or "subscription" in lowered
            or "premium" in lowered
            or "supergrok" in lowered
            or "quota" in lowered
        )

    @classmethod
    def _is_transient_text(cls, text: str) -> bool:
        lowered = (text or "").lower()
        return any(marker in lowered for marker in cls.TRANSIENT_MARKERS)

    async def exchange_code(
        self,
        *,
        code: str,
        code_verifier: str,
        code_challenge: str = "",
        redirect_uri: str = DEFAULT_REDIRECT_URI,
    ) -> Dict[str, Any]:
        """Exchange an OAuth authorization code for persisted xAI Grok tokens."""
        if not code or not code_verifier:
            raise XAIGrokAuthError("Authorization code and PKCE verifier are required.")

        data = {
            "client_id": self.CLIENT_ID,
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "code_verifier": code_verifier,
        }
        # xAI's OAuth surface has required the original challenge in some
        # integrations; include it when available while keeping the standard
        # verifier exchange intact.
        if code_challenge:
            data["code_challenge"] = code_challenge
            data["code_challenge_method"] = "S256"

        response = await self.client.post(
            self.TOKEN_URL,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if response.status_code >= 400:
            raise XAIGrokAuthError(
                f"xAI Grok OAuth exchange failed: {sanitize_provider_error_text(response.text)}"
            )
        tokens = self._normalize_token_payload(response.json())
        if not tokens.get("access_token") or not tokens.get("refresh_token"):
            raise XAIGrokAuthError("xAI Grok OAuth exchange did not return usable tokens.")
        store_xai_grok_oauth_tokens(tokens)
        return self.safe_status(tokens)

    async def refresh_tokens(self, tokens: Dict[str, Any]) -> Dict[str, Any]:
        """Refresh stored xAI OAuth tokens and persist the replacement bundle."""
        refresh_token = str(tokens.get("refresh_token") or "")
        if not refresh_token:
            raise XAIGrokAuthError("xAI Grok refresh token is missing.")

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
            raise XAIGrokAuthError(
                f"xAI Grok token refresh failed: {sanitize_provider_error_text(response.text)}"
            )
        refreshed = self._normalize_token_payload({**tokens, **response.json()})
        store_xai_grok_oauth_tokens(refreshed)
        return refreshed

    def load_tokens(self) -> Optional[Dict[str, Any]]:
        """Load persisted xAI Grok OAuth tokens."""
        try:
            return load_xai_grok_oauth_tokens()
        except SecretStoreError:
            raise
        except Exception as exc:
            raise XAIGrokAuthError("Failed to load xAI Grok OAuth tokens.") from exc

    async def get_valid_tokens(self) -> Dict[str, Any]:
        """Load and refresh tokens if they are close to expiry."""
        tokens = self.load_tokens()
        if not tokens or not tokens.get("access_token"):
            raise XAIGrokAuthError("xAI Grok OAuth is not configured.")

        expires_at = int(tokens.get("expires_at") or 0)
        if expires_at and time.time() < expires_at - self.REFRESH_SKEW_SECONDS:
            return tokens

        async with self._refresh_lock:
            tokens = self.load_tokens()
            if not tokens or not tokens.get("access_token"):
                raise XAIGrokAuthError("xAI Grok OAuth is not configured.")
            expires_at = int(tokens.get("expires_at") or 0)
            if expires_at and time.time() < expires_at - self.REFRESH_SKEW_SECONDS:
                return tokens
            return await self.refresh_tokens(tokens)

    async def _recover_tokens_after_auth_failure(
        self,
        used_tokens: Dict[str, Any],
        *,
        context: str,
    ) -> Dict[str, Any]:
        """Reload or refresh xAI OAuth tokens after an auth rejection."""
        async with self._refresh_lock:
            current_tokens = self.load_tokens()
            if not current_tokens or not current_tokens.get("access_token"):
                raise XAIGrokAuthError(
                    f"xAI Grok {context} failed because OAuth is no longer configured."
                )
            if self._access_token(current_tokens) != self._access_token(used_tokens):
                logger.info(
                    "xAI Grok %s auth failed with an older token; retrying with newer stored OAuth token.",
                    context,
                )
                return current_tokens
            logger.info("xAI Grok %s auth failed; refreshing OAuth token once before retrying.", context)
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
        """Return current xAI Grok OAuth status."""
        return self.safe_status(self.load_tokens())

    @staticmethod
    def _positive_int(*values: Any) -> Optional[int]:
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
    def _known_model_limits(cls, model_id: str, model_name: str = "") -> Dict[str, int]:
        normalized_candidates = [
            str(value or "").strip().lower().replace("_", "-")
            for value in (model_id, model_name)
        ]
        for candidate in normalized_candidates:
            exact = cls.KNOWN_MODEL_LIMITS.get(candidate)
            if exact:
                return exact
        return {}

    @classmethod
    def _is_chat_completion_supported_model(cls, model_id: str, model_name: str = "") -> bool:
        normalized = " ".join(
            str(value or "").strip().lower().replace("_", "-")
            for value in (model_id, model_name)
        )
        return not any(marker in normalized for marker in cls.CHAT_UNSUPPORTED_MODEL_MARKERS)

    @classmethod
    def _normalize_model_metadata(cls, model: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        model_id = model.get("id") or model.get("slug")
        if not model_id:
            return None
        model_id = str(model_id)
        model_name = str(model.get("name") or model.get("title") or model_id)
        if not cls._is_chat_completion_supported_model(model_id, model_name):
            logger.info("Skipping xAI Grok OAuth model not supported by chat completions: %s", model_id)
            return None
        known = cls._known_model_limits(model_id, model_name)
        context_length = cls._positive_int(
            model.get("context_length"),
            model.get("context_window"),
            model.get("contextTokens"),
            known.get("context_length"),
        )
        max_output_tokens = cls._positive_int(
            model.get("max_output_tokens"),
            model.get("max_completion_tokens"),
            model.get("output_tokens"),
            known.get("max_output_tokens"),
        )
        normalized: Dict[str, Any] = {
            "id": model_id,
            "name": model_name,
            "pricing": {"prompt": "subscription", "completion": "subscription"},
            "provider_metadata": {
                "source": "xai_grok_oauth",
                "raw_context_length": model.get("context_length") or model.get("contextTokens"),
                "raw_context_window": model.get("context_window"),
                "raw_max_output_tokens": model.get("max_output_tokens") or model.get("max_completion_tokens"),
            },
        }
        if context_length:
            normalized["context_length"] = context_length
        if max_output_tokens:
            normalized["max_output_tokens"] = max_output_tokens
        return normalized

    def _headers(self, tokens: Dict[str, Any]) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {tokens['access_token']}",
            "Content-Type": "application/json",
        }

    async def list_models(self) -> List[Dict[str, Any]]:
        """List Grok models available to the signed-in xAI account."""
        tokens = await self.get_valid_tokens()
        response = await self.client.get(
            f"{self.API_BASE_URL}/models",
            headers=self._headers(tokens),
        )
        if response.status_code in {401, 403}:
            tokens = await self._recover_tokens_after_auth_failure(tokens, context="model listing")
            response = await self.client.get(
                f"{self.API_BASE_URL}/models",
                headers=self._headers(tokens),
            )
        if response.status_code >= 400:
            message = sanitize_provider_error_text(response.text)
            if response.status_code == 403 or self._is_entitlement_failure(message):
                raise XAIGrokAuthError(
                    "xAI Grok model listing failed. Check that this account has eligible "
                    f"SuperGrok/X Premium access and remaining quota: {message}"
                )
            raise XAIGrokAuthError(f"xAI Grok model listing failed: {message}")
        data = response.json()
        raw_models = data.get("data") if isinstance(data, dict) else None
        if raw_models is None and isinstance(data, dict):
            raw_models = data.get("models")
        models = []
        for model in raw_models or []:
            if not isinstance(model, dict):
                continue
            normalized = self._normalize_model_metadata(model)
            if normalized:
                models.append(normalized)
        return models

    async def clear_tokens(self) -> None:
        """Revoke best-effort and clear persisted xAI Grok OAuth tokens."""
        tokens = self.load_tokens()
        token = str((tokens or {}).get("refresh_token") or "")
        if token:
            try:
                await self.client.post(
                    self.REVOKE_URL,
                    data={"client_id": self.CLIENT_ID, "token": token},
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
            except Exception:
                logger.debug("xAI Grok token revoke failed; clearing local credential anyway.")
        clear_xai_grok_oauth_tokens()

    async def _post_with_retry(self, url: str, **kwargs) -> httpx.Response:
        """POST with retry on transient transport/provider errors."""
        for attempt in range(self.MAX_RETRIES):
            try:
                response = await self.client.post(url, **kwargs)
                if response.status_code >= 400 and (
                    response.status_code in self.TRANSIENT_STATUS_CODES
                    or self._is_transient_text(response.text)
                ):
                    error_detail = sanitize_provider_error_text(response.text)
                    logger.warning(
                        "xAI Grok transient completion response (attempt %s/%s): status=%s error=%s",
                        attempt + 1,
                        self.MAX_RETRIES,
                        response.status_code,
                        error_detail,
                    )
                    if attempt < self.MAX_RETRIES - 1:
                        await asyncio.sleep(self.RETRY_DELAY * (attempt + 1))
                        continue
                    raise ValueError(
                        f"xAI Grok connection failed after {self.MAX_RETRIES} attempts: "
                        f"HTTP {response.status_code}: {error_detail}"
                    )
                return response
            except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError) as exc:
                error_type = type(exc).__name__
                error_detail = sanitize_provider_error_text(str(exc) or repr(exc))
                logger.warning(
                    "xAI Grok connection error (attempt %s/%s): [%s] %s",
                    attempt + 1,
                    self.MAX_RETRIES,
                    error_type,
                    error_detail,
                )
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.RETRY_DELAY * (attempt + 1))
                    continue
                raise ValueError(
                    f"xAI Grok connection failed after {self.MAX_RETRIES} attempts: "
                    f"[{error_type}] {error_detail}"
                )

    @staticmethod
    def _response_format(response_format: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
        if response_format and response_format.get("type") == "json_object":
            return {"type": "json_object"}
        return None

    @staticmethod
    def _reasoning_effort(reasoning_effort: Optional[str]) -> Optional[str]:
        if not reasoning_effort:
            return None
        effort = str(reasoning_effort).strip().lower()
        if effort in {"auto", "max", "maximum", "highest", "xhigh"}:
            return "high"
        if effort == "minimal":
            return "low"
        if effort in {"high", "medium", "low"}:
            return effort
        return None

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
        selected_model = model or self.DEFAULT_MODEL
        if not self._is_chat_completion_supported_model(selected_model):
            raise XAIGrokRequestError(
                f"xAI Grok model '{selected_model}' is not supported by the OAuth chat-completions "
                "route. Choose a regular Grok chat model in Settings, such as grok-4.3."
            )
        tokens = await self.get_valid_tokens()
        payload: Dict[str, Any] = {
            "model": selected_model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = int(max_tokens)
        json_format = self._response_format(response_format)
        if json_format:
            payload["response_format"] = json_format
        effort = self._reasoning_effort(reasoning_effort)
        if effort:
            payload["reasoning_effort"] = effort
        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice

        auth_retry_used = False
        while True:
            response = await self._post_with_retry(
                f"{self.API_BASE_URL}/chat/completions",
                json=payload,
                headers=self._headers(tokens),
            )
            if response.status_code >= 400:
                message = sanitize_provider_error_text(response.text)
                if response.status_code == 401:
                    if not auth_retry_used:
                        tokens = await self._recover_tokens_after_auth_failure(tokens, context="completion")
                        auth_retry_used = True
                        continue
                    raise XAIGrokAuthError(f"xAI Grok completion failed: {message}")
                if response.status_code == 403 or self._is_entitlement_failure(message):
                    raise XAIGrokAuthError(
                        "xAI Grok completion failed. Check eligible SuperGrok/X Premium access "
                        f"and subscription quota: {message}"
                    )
                raise XAIGrokRequestError(f"xAI Grok completion failed: {message}")
            data = response.json()
            if not isinstance(data, dict):
                raise XAIGrokRequestError("xAI Grok completion returned an invalid response shape.")
            if not data.get("choices"):
                raise XAIGrokRequestError("xAI Grok completion returned no choices.")
            return data

    async def close(self) -> None:
        await self.client.aclose()


xai_grok_client = XAIGrokClient()
