"""
Cloud provider credential and account-login routes.
"""
from __future__ import annotations

import asyncio
import html
import logging
import secrets
import time
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.shared.config import rag_config, system_config
from backend.shared.openai_codex_client import OpenAICodexAuthError, openai_codex_client
from backend.shared.provider_notification_store import list_provider_notifications
from backend.shared.sakana_fugu_client import SakanaFuguAuthError, SakanaFuguError, sakana_fugu_client
from backend.shared.xai_grok_client import XAIGrokAuthError, xai_grok_client

router = APIRouter(prefix="/api/cloud-access", tags=["cloud-access"])
logger = logging.getLogger(__name__)

_PENDING_CODEX_OAUTH: Dict[str, Dict[str, Any]] = {}
_PENDING_XAI_GROK_OAUTH: Dict[str, Dict[str, Any]] = {}
_PENDING_TTL_SECONDS = 15 * 60


async def _safe_oauth_status(provider_label: str, status_coro) -> Dict[str, Any]:
    """Return provider OAuth status without letting one bad token break all cloud status."""
    try:
        return await status_coro
    except Exception as exc:
        logger.warning("%s OAuth status check failed: %s", provider_label, exc)
        return {
            "configured": False,
            "status_error": "OAuth status unavailable. Reconnect this provider if the issue persists.",
        }


class _CodexCallbackServerState:
    def __init__(self) -> None:
        self.server: Optional[asyncio.AbstractServer] = None
        self.lock = asyncio.Lock()


_CODEX_CALLBACK_SERVER_STATE = _CodexCallbackServerState()


class _XAIGrokCallbackServerState:
    def __init__(self) -> None:
        self.server: Optional[asyncio.AbstractServer] = None
        self.lock = asyncio.Lock()


_XAI_GROK_CALLBACK_SERVER_STATE = _XAIGrokCallbackServerState()


class CodexOAuthStartRequest(BaseModel):
    redirect_uri: Optional[str] = None


class CodexOAuthExchangeRequest(BaseModel):
    code: str = ""
    state: str = ""
    redirect_url: str = ""
    redirect_uri: Optional[str] = None


class XAIGrokOAuthStartRequest(BaseModel):
    redirect_uri: Optional[str] = None


class XAIGrokOAuthExchangeRequest(BaseModel):
    code: str = ""
    state: str = ""
    redirect_url: str = ""
    redirect_uri: Optional[str] = None


class SakanaFuguApiKeyRequest(BaseModel):
    api_key: str


def _ensure_desktop_codex_allowed() -> None:
    if system_config.generic_mode:
        raise HTTPException(
            status_code=501,
            detail=(
                "OpenAI Codex account login is currently desktop-only. "
                "Hosted mode should use OpenRouter keys until callback/proxy login is designed."
            ),
        )


def _ensure_desktop_xai_grok_allowed() -> None:
    if system_config.generic_mode:
        raise HTTPException(
            status_code=501,
            detail=(
                "xAI Grok account login is currently desktop-only. "
                "Hosted mode should use OpenRouter keys until callback/proxy login is designed."
            ),
        )


def _ensure_desktop_sakana_fugu_allowed() -> None:
    if system_config.generic_mode:
        raise HTTPException(
            status_code=501,
            detail=(
                "Sakana Fugu direct subscription API access is currently desktop-only. "
                "Hosted mode should use OpenRouter keys until direct provider support is designed."
            ),
        )


def _resolve_codex_redirect_uri(requested_redirect_uri: Optional[str]) -> str:
    """Keep the Codex OAuth redirect pinned to the local loopback callback."""
    default_redirect_uri = openai_codex_client.DEFAULT_REDIRECT_URI
    if requested_redirect_uri and requested_redirect_uri != default_redirect_uri:
        raise HTTPException(
            status_code=400,
            detail="OpenAI Codex OAuth only supports the fixed local loopback redirect URI.",
        )
    return default_redirect_uri


def _resolve_xai_grok_redirect_uri(requested_redirect_uri: Optional[str]) -> str:
    """Keep the xAI Grok OAuth redirect pinned to the local loopback callback."""
    default_redirect_uri = xai_grok_client.DEFAULT_REDIRECT_URI
    try:
        _xai_grok_callback_binding(default_redirect_uri)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if requested_redirect_uri and requested_redirect_uri != default_redirect_uri:
        raise HTTPException(
            status_code=400,
            detail="xAI Grok OAuth only supports the fixed local loopback redirect URI.",
        )
    return default_redirect_uri


def _xai_grok_callback_binding(redirect_uri: Optional[str] = None) -> tuple[str, int, str]:
    """Return the loopback host/port/path for the configured xAI callback URI."""
    parsed = urlparse(redirect_uri or xai_grok_client.DEFAULT_REDIRECT_URI)
    if parsed.scheme != "http":
        raise ValueError("xAI Grok OAuth callback must use a local http:// loopback redirect URI.")
    host = parsed.hostname or ""
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("xAI Grok OAuth callback must use localhost or 127.0.0.1.")
    if parsed.port is None:
        raise ValueError("xAI Grok OAuth callback redirect URI must include an explicit port.")
    return host, int(parsed.port), parsed.path or "/callback"


async def _stop_codex_callback_server_if_idle() -> None:
    """Release the conventional Codex OAuth callback port when no login is pending."""
    async with _CODEX_CALLBACK_SERVER_STATE.lock:
        server = _CODEX_CALLBACK_SERVER_STATE.server
        if _PENDING_CODEX_OAUTH or server is None:
            return
        server.close()
        await server.wait_closed()
        _CODEX_CALLBACK_SERVER_STATE.server = None


async def _stop_xai_grok_callback_server_if_idle() -> None:
    """Release the xAI OAuth callback port when no login is pending."""
    async with _XAI_GROK_CALLBACK_SERVER_STATE.lock:
        server = _XAI_GROK_CALLBACK_SERVER_STATE.server
        if _PENDING_XAI_GROK_OAUTH or server is None:
            return
        server.close()
        await server.wait_closed()
        _XAI_GROK_CALLBACK_SERVER_STATE.server = None


async def _prune_pending() -> None:
    now = time.time()
    expired = [state for state, payload in _PENDING_CODEX_OAUTH.items() if now > payload["expires_at"]]
    for state in expired:
        _PENDING_CODEX_OAUTH.pop(state, None)
    expired_xai = [state for state, payload in _PENDING_XAI_GROK_OAUTH.items() if now > payload["expires_at"]]
    for state in expired_xai:
        _PENDING_XAI_GROK_OAUTH.pop(state, None)
    await _stop_codex_callback_server_if_idle()
    await _stop_xai_grok_callback_server_if_idle()


async def _write_http_response(writer: asyncio.StreamWriter, status: str, body: str) -> None:
    payload = body.encode("utf-8", errors="replace")
    writer.write(
        (
            f"HTTP/1.1 {status}\r\n"
            "Content-Type: text/html; charset=utf-8\r\n"
            f"Content-Length: {len(payload)}\r\n"
            "Connection: close\r\n"
            "\r\n"
        ).encode("ascii") + payload
    )
    await writer.drain()
    writer.close()
    await writer.wait_closed()


async def _handle_codex_callback(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        request_line = await reader.readline()
        parts = request_line.decode("ascii", errors="ignore").split()
        path = parts[1] if len(parts) >= 2 else ""
        parsed = urlparse(path)
        query = parse_qs(parsed.query)
        state = query.get("state", [""])[0]
        code = query.get("code", [""])[0]
        error = query.get("error", [""])[0]

        if parsed.path != "/auth/callback":
            await _write_http_response(writer, "404 Not Found", "<h1>Not Found</h1>")
            return
        if error:
            if state:
                _PENDING_CODEX_OAUTH.pop(state, None)
            await _write_http_response(writer, "400 Bad Request", f"<h1>OpenAI login failed</h1><p>{html.escape(error)}</p>")
            await _stop_codex_callback_server_if_idle()
            return

        pending = _PENDING_CODEX_OAUTH.pop(state, None)
        if not state or not pending or not code:
            await _write_http_response(
                writer,
                "400 Bad Request",
                "<h1>OpenAI Codex login could not be completed</h1><p>The login state expired or the code was missing. Return to MOTO and start login again.</p>",
            )
            await _stop_codex_callback_server_if_idle()
            return

        await openai_codex_client.exchange_code(
            code=code,
            code_verifier=pending["code_verifier"],
            redirect_uri=pending["redirect_uri"],
        )
        await _write_http_response(
            writer,
            "200 OK",
            "<h1>OpenAI Codex login complete</h1><p>You can close this tab and return to MOTO.</p>",
        )
        await _stop_codex_callback_server_if_idle()
    except Exception as exc:
        logger.warning("OpenAI Codex OAuth callback failed: %s", exc)
        try:
            await _write_http_response(
                writer,
                "500 Internal Server Error",
                "<h1>OpenAI Codex login failed</h1><p>Return to MOTO and paste the callback URL manually, or start login again.</p>",
            )
        finally:
            await _stop_codex_callback_server_if_idle()


async def _handle_xai_grok_callback(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        request_line = await reader.readline()
        parts = request_line.decode("ascii", errors="ignore").split()
        path = parts[1] if len(parts) >= 2 else ""
        parsed = urlparse(path)
        query = parse_qs(parsed.query)
        state = query.get("state", [""])[0]
        code = query.get("code", [""])[0]
        error = query.get("error", [""])[0]

        _host, _port, callback_path = _xai_grok_callback_binding()
        if parsed.path != callback_path:
            await _write_http_response(writer, "404 Not Found", "<h1>Not Found</h1>")
            return
        if error:
            if state:
                _PENDING_XAI_GROK_OAUTH.pop(state, None)
            await _write_http_response(writer, "400 Bad Request", f"<h1>xAI Grok login failed</h1><p>{html.escape(error)}</p>")
            await _stop_xai_grok_callback_server_if_idle()
            return

        pending = _PENDING_XAI_GROK_OAUTH.pop(state, None)
        if not state or not pending or not code:
            await _write_http_response(
                writer,
                "400 Bad Request",
                "<h1>xAI Grok login could not be completed</h1><p>The login state expired or the code was missing. Return to MOTO and start login again.</p>",
            )
            await _stop_xai_grok_callback_server_if_idle()
            return

        await xai_grok_client.exchange_code(
            code=code,
            code_verifier=pending["code_verifier"],
            code_challenge=pending["code_challenge"],
            redirect_uri=pending["redirect_uri"],
        )
        await _write_http_response(
            writer,
            "200 OK",
            "<h1>xAI Grok login complete</h1><p>You can close this tab and return to MOTO.</p>",
        )
        await _stop_xai_grok_callback_server_if_idle()
    except Exception as exc:
        logger.warning("xAI Grok OAuth callback failed: %s", exc)
        try:
            await _write_http_response(
                writer,
                "500 Internal Server Error",
                "<h1>xAI Grok login failed</h1><p>Return to MOTO and paste the callback URL manually, or start login again.</p>",
            )
        finally:
            await _stop_xai_grok_callback_server_if_idle()


async def _ensure_codex_callback_server() -> bool:
    """Start the temporary loopback callback server if the port is available."""
    async with _CODEX_CALLBACK_SERVER_STATE.lock:
        server = _CODEX_CALLBACK_SERVER_STATE.server
        if server and server.is_serving():
            return True
        try:
            _CODEX_CALLBACK_SERVER_STATE.server = await asyncio.start_server(
                _handle_codex_callback,
            host="localhost",
                port=1455,
            )
        except OSError:
            _CODEX_CALLBACK_SERVER_STATE.server = None
            return False
        return True


async def _ensure_xai_grok_callback_server() -> bool:
    """Start the temporary xAI loopback callback server if the port is available."""
    async with _XAI_GROK_CALLBACK_SERVER_STATE.lock:
        server = _XAI_GROK_CALLBACK_SERVER_STATE.server
        if server and server.is_serving():
            return True
        host, port, _path = _xai_grok_callback_binding()
        try:
            _XAI_GROK_CALLBACK_SERVER_STATE.server = await asyncio.start_server(
                _handle_xai_grok_callback,
                host=host,
                port=port,
            )
        except OSError:
            _XAI_GROK_CALLBACK_SERVER_STATE.server = None
            return False
        return True


@router.get("/status")
async def get_cloud_access_status() -> Dict[str, Any]:
    """Return non-secret cloud credential status for the header overlay."""
    await _prune_pending()
    codex_status = {"configured": False} if system_config.generic_mode else await _safe_oauth_status(
        "OpenAI Codex",
        openai_codex_client.status(),
    )
    xai_grok_status = {"configured": False} if system_config.generic_mode else await _safe_oauth_status(
        "xAI Grok",
        xai_grok_client.status(),
    )
    sakana_status = {"configured": False} if system_config.generic_mode else await _safe_oauth_status(
        "Sakana Fugu",
        sakana_fugu_client.status(),
    )
    return {
        "success": True,
        "generic_mode": system_config.generic_mode,
        "providers": {
            "openrouter": {
                "configured": bool(rag_config.openrouter_api_key),
                "available": True,
            },
            "openai_codex_oauth": {
                **codex_status,
                "available": not system_config.generic_mode,
                "desktop_only": True,
            },
            "xai_grok_oauth": {
                **xai_grok_status,
                "available": not system_config.generic_mode,
                "desktop_only": True,
            },
            "sakana_fugu": {
                **sakana_status,
                "available": not system_config.generic_mode,
                "desktop_only": True,
            },
        },
    }


@router.get("/provider-notifications")
async def get_provider_notifications() -> Dict[str, Any]:
    """Return recent non-secret provider/OAuth notifications missed by live WebSocket clients."""
    return {
        "success": True,
        "notifications": await asyncio.to_thread(list_provider_notifications),
    }


@router.post("/openai-codex/oauth/start")
async def start_openai_codex_oauth(request: CodexOAuthStartRequest) -> Dict[str, Any]:
    """Start the OpenAI Codex OAuth PKCE login flow."""
    _ensure_desktop_codex_allowed()
    await _prune_pending()
    callback_available = await _ensure_codex_callback_server()
    code_verifier, code_challenge = openai_codex_client.generate_pkce_pair()
    state = secrets.token_urlsafe(32)
    redirect_uri = _resolve_codex_redirect_uri(request.redirect_uri)
    _PENDING_CODEX_OAUTH[state] = {
        "code_verifier": code_verifier,
        "redirect_uri": redirect_uri,
        "expires_at": time.time() + _PENDING_TTL_SECONDS,
    }
    return {
        "success": True,
        "authorization_url": openai_codex_client.build_authorization_url(
            code_challenge=code_challenge,
            state=state,
            redirect_uri=redirect_uri,
        ),
        "state": state,
        "redirect_uri": redirect_uri,
        "expires_in": _PENDING_TTL_SECONDS,
        "callback_available": callback_available,
    }


@router.post("/openai-codex/oauth/exchange")
async def exchange_openai_codex_oauth(request: CodexOAuthExchangeRequest) -> Dict[str, Any]:
    """Exchange a callback code or pasted callback URL for Codex tokens."""
    _ensure_desktop_codex_allowed()
    await _prune_pending()
    requested_redirect_uri = _resolve_codex_redirect_uri(request.redirect_uri) if request.redirect_uri else None
    code, parsed_state = openai_codex_client.extract_code_and_state(
        code=request.code,
        redirect_url=request.redirect_url,
    )
    state = request.state or parsed_state
    pending = _PENDING_CODEX_OAUTH.pop(state, None)
    if not state or not pending:
        raise HTTPException(status_code=400, detail="OAuth state is missing or expired. Please start login again.")
    if not code:
        await _stop_codex_callback_server_if_idle()
        raise HTTPException(status_code=400, detail="OAuth authorization code is required.")
    redirect_uri = requested_redirect_uri or _resolve_codex_redirect_uri(pending["redirect_uri"])
    try:
        status = await openai_codex_client.exchange_code(
            code=code,
            code_verifier=pending["code_verifier"],
            redirect_uri=redirect_uri,
        )
    except OpenAICodexAuthError as exc:
        await _stop_codex_callback_server_if_idle()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await _stop_codex_callback_server_if_idle()
    return {"success": True, "provider": "openai_codex_oauth", "status": status}


@router.get("/openai-codex/status")
async def get_openai_codex_status() -> Dict[str, Any]:
    """Return OpenAI Codex OAuth status."""
    if system_config.generic_mode:
        return {"success": True, "status": {"configured": False, "available": False, "desktop_only": True}}
    return {"success": True, "status": await openai_codex_client.status()}


@router.get("/openai-codex/models")
async def get_openai_codex_models() -> Dict[str, Any]:
    """Return available Codex-backed models for the signed-in account."""
    _ensure_desktop_codex_allowed()
    try:
        models = await openai_codex_client.list_models()
    except OpenAICodexAuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"success": True, "models": models}


@router.delete("/openai-codex")
async def clear_openai_codex_oauth() -> Dict[str, Any]:
    """Clear the stored OpenAI Codex OAuth credential."""
    _ensure_desktop_codex_allowed()
    await openai_codex_client.clear_tokens()
    return {"success": True, "message": "OpenAI Codex login cleared"}


@router.post("/xai-grok/oauth/start")
async def start_xai_grok_oauth(request: XAIGrokOAuthStartRequest) -> Dict[str, Any]:
    """Start the xAI Grok OAuth PKCE login flow."""
    _ensure_desktop_xai_grok_allowed()
    await _prune_pending()
    callback_available = await _ensure_xai_grok_callback_server()
    code_verifier, code_challenge = xai_grok_client.generate_pkce_pair()
    state = secrets.token_urlsafe(32)
    nonce = secrets.token_urlsafe(32)
    redirect_uri = _resolve_xai_grok_redirect_uri(request.redirect_uri)
    _PENDING_XAI_GROK_OAUTH[state] = {
        "code_verifier": code_verifier,
        "code_challenge": code_challenge,
        "redirect_uri": redirect_uri,
        "nonce": nonce,
        "expires_at": time.time() + _PENDING_TTL_SECONDS,
    }
    return {
        "success": True,
        "authorization_url": xai_grok_client.build_authorization_url(
            code_challenge=code_challenge,
            state=state,
            nonce=nonce,
            redirect_uri=redirect_uri,
        ),
        "state": state,
        "redirect_uri": redirect_uri,
        "expires_in": _PENDING_TTL_SECONDS,
        "callback_available": callback_available,
    }


@router.post("/xai-grok/oauth/exchange")
async def exchange_xai_grok_oauth(request: XAIGrokOAuthExchangeRequest) -> Dict[str, Any]:
    """Exchange a callback code or pasted callback URL for xAI Grok tokens."""
    _ensure_desktop_xai_grok_allowed()
    await _prune_pending()
    requested_redirect_uri = _resolve_xai_grok_redirect_uri(request.redirect_uri) if request.redirect_uri else None
    code, parsed_state = xai_grok_client.extract_code_and_state(
        code=request.code,
        redirect_url=request.redirect_url,
    )
    state = request.state or parsed_state
    pending = _PENDING_XAI_GROK_OAUTH.pop(state, None)
    if not state or not pending:
        raise HTTPException(status_code=400, detail="OAuth state is missing or expired. Please start login again.")
    if not code:
        await _stop_xai_grok_callback_server_if_idle()
        raise HTTPException(status_code=400, detail="OAuth authorization code is required.")
    redirect_uri = requested_redirect_uri or _resolve_xai_grok_redirect_uri(pending["redirect_uri"])
    try:
        status = await xai_grok_client.exchange_code(
            code=code,
            code_verifier=pending["code_verifier"],
            code_challenge=pending["code_challenge"],
            redirect_uri=redirect_uri,
        )
    except XAIGrokAuthError as exc:
        await _stop_xai_grok_callback_server_if_idle()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await _stop_xai_grok_callback_server_if_idle()
    return {"success": True, "provider": "xai_grok_oauth", "status": status}


@router.get("/xai-grok/status")
async def get_xai_grok_status() -> Dict[str, Any]:
    """Return xAI Grok OAuth status."""
    if system_config.generic_mode:
        return {"success": True, "status": {"configured": False, "available": False, "desktop_only": True}}
    return {"success": True, "status": await xai_grok_client.status()}


@router.get("/xai-grok/models")
async def get_xai_grok_models() -> Dict[str, Any]:
    """Return available Grok models for the signed-in account."""
    _ensure_desktop_xai_grok_allowed()
    try:
        models = await xai_grok_client.list_models()
    except XAIGrokAuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"success": True, "models": models}


@router.delete("/xai-grok")
async def clear_xai_grok_oauth() -> Dict[str, Any]:
    """Clear the stored xAI Grok OAuth credential."""
    _ensure_desktop_xai_grok_allowed()
    await xai_grok_client.clear_tokens()
    return {"success": True, "message": "xAI Grok login cleared"}


@router.get("/sakana-fugu/status")
async def get_sakana_fugu_status() -> Dict[str, Any]:
    """Return Sakana Fugu API-key status."""
    if system_config.generic_mode:
        return {"success": True, "status": {"configured": False, "available": False, "desktop_only": True}}
    return {"success": True, "status": await sakana_fugu_client.status()}


@router.post("/sakana-fugu/api-key")
async def set_sakana_fugu_api_key(request: SakanaFuguApiKeyRequest) -> Dict[str, Any]:
    """Store a Sakana Fugu API key in the desktop keyring."""
    _ensure_desktop_sakana_fugu_allowed()
    try:
        key = request.api_key.strip()
        models = await sakana_fugu_client.list_models(api_key=key)
        sakana_fugu_client.set_api_key(key)
    except (ValueError, SakanaFuguAuthError, SakanaFuguError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "success": True,
        "provider": "sakana_fugu",
        "status": await sakana_fugu_client.status(),
        "models": models,
    }


@router.get("/sakana-fugu/models")
async def get_sakana_fugu_models() -> Dict[str, Any]:
    """Return available Sakana Fugu models for the configured API key."""
    _ensure_desktop_sakana_fugu_allowed()
    try:
        models = await sakana_fugu_client.list_models()
    except SakanaFuguAuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"success": True, "models": models}


@router.delete("/sakana-fugu")
async def clear_sakana_fugu_api_key() -> Dict[str, Any]:
    """Clear the stored Sakana Fugu API key."""
    _ensure_desktop_sakana_fugu_allowed()
    await sakana_fugu_client.clear_api_key()
    return {"success": True, "message": "Sakana Fugu API key cleared"}
