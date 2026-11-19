"""
Middleware for CORS and error handling.
"""
import hmac
import os
import re
import time
from urllib.parse import urlparse
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette import status
import logging

from backend.api.proxy_auth import (
    EMPTY_BODY_SHA256,
    PROXY_BODY_SHA256_HEADER,
    ProxyAuthError,
    hash_proxy_body,
    is_proxy_auth_allowlisted,
    routed_path,
    routed_query,
    validate_proxy_headers,
)
from backend.shared.config import system_config

logger = logging.getLogger(__name__)

# Default allowed origins for local development
DEFAULT_ORIGINS = [
    f"http://localhost:{system_config.frontend_port}",
    f"http://127.0.0.1:{system_config.frontend_port}",
    f"http://localhost:{system_config.backend_port}",
    f"http://127.0.0.1:{system_config.backend_port}",
]
DESKTOP_API_TOKEN_HEADER = "X-Moto-Desktop-Token"
UNSAFE_HTTP_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
DESKTOP_PUBLIC_PROOF_EXPORT_RE = re.compile(r"^/api/proofs/[^/]+/certificate(?:\.lean)?$")
STALE_DESKTOP_TAB_LOG_INTERVAL_SECONDS = 300
_stale_desktop_tab_log_state: dict[tuple[str, str], tuple[float, int]] = {}


def _is_desktop_public_export(method: str, path: str) -> bool:
    """Allow direct local browser downloads for read-only generated artifacts."""
    normalized_method = (method or "").upper()
    normalized_path = path or ""
    if normalized_method not in {"GET", "HEAD"}:
        return False
    return bool(DESKTOP_PUBLIC_PROOF_EXPORT_RE.fullmatch(normalized_path))


def _origin_from_url(value: str) -> str:
    """Return scheme://host[:port] for an Origin/Referer-like value."""
    parsed = urlparse(value or "")
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}"


def _log_desktop_auth_rejection(request: Request, exc: ProxyAuthError) -> None:
    """Log stale-tab token misses without flooding desktop logs."""
    if exc.status_code != status.HTTP_401_UNAUTHORIZED:
        logger.warning("Rejected desktop request %s %s: %s", request.method, routed_path(request.scope), exc.detail)
        return

    key = (request.method, routed_path(request.scope))
    now = time.monotonic()
    last_log_at, suppressed = _stale_desktop_tab_log_state.get(key, (0.0, 0))

    if now - last_log_at < STALE_DESKTOP_TAB_LOG_INTERVAL_SECONDS:
        _stale_desktop_tab_log_state[key] = (last_log_at, suppressed + 1)
        return

    suffix = f" Suppressed {suppressed} repeat stale-tab probe(s) for this route." if suppressed else ""
    logger.info(
        "Ignored desktop request without the current MOTO tab token: %s %s. "
        "This is usually a harmless stale browser tab from an older launch; returning 401.%s",
        request.method,
        routed_path(request.scope),
        suffix,
    )
    _stale_desktop_tab_log_state[key] = (now, 0)


def _validate_desktop_token(request: Request, allowed_origins: list[str]) -> None:
    """Require the launcher-provided desktop API token outside public routes."""
    if is_proxy_auth_allowlisted(request.method, routed_path(request.scope)):
        return
    if _is_desktop_public_export(request.method, routed_path(request.scope)):
        return

    expected = (system_config.desktop_api_token or "").strip()
    if not expected:
        raise ProxyAuthError(
            "Desktop API token is not configured for this runtime.",
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    provided = (request.headers.get(DESKTOP_API_TOKEN_HEADER) or "").strip()
    if not provided or not hmac.compare_digest(provided, expected):
        raise ProxyAuthError(
            "Missing or invalid desktop API token.",
            status.HTTP_401_UNAUTHORIZED,
        )

    if request.method.upper() in UNSAFE_HTTP_METHODS:
        origin = (request.headers.get("origin") or "").strip()
        referer = _origin_from_url(request.headers.get("referer") or "")
        candidate = origin or referer
        if candidate and candidate not in allowed_origins:
            raise ProxyAuthError(
                "Unsafe request origin is not allowed for this desktop runtime.",
                status.HTTP_403_FORBIDDEN,
            )


def _validate_generic_content_length(request: Request) -> None:
    """Reject oversized hosted requests before route handlers parse the body."""
    raw_content_length = (request.headers.get("content-length") or "").strip()
    if not raw_content_length:
        return

    try:
        content_length = int(raw_content_length)
    except ValueError as exc:
        raise ProxyAuthError(
            "Invalid Content-Length header.",
            status.HTTP_400_BAD_REQUEST,
        ) from exc

    max_bytes = _generic_max_request_bytes()
    if content_length > max_bytes:
        raise ProxyAuthError(
            f"Request body exceeds hosted limit of {max_bytes} bytes.",
            status.HTTP_413_CONTENT_TOO_LARGE,
        )


def _generic_max_request_bytes() -> int:
    return max(int(system_config.generic_max_request_bytes or 0), 1)


async def _read_generic_body(request: Request) -> bytes:
    """Read the request body, stopping once it exceeds the hosted size limit.

    A body sent without ``Content-Length`` (chunked) is not covered by the
    header check, so the limit is enforced on the bytes actually received.
    """
    max_bytes = _generic_max_request_bytes()
    chunks: list[bytes] = []
    received = 0
    async for chunk in request.stream():
        received += len(chunk)
        if received > max_bytes:
            raise ProxyAuthError(
                f"Request body exceeds hosted limit of {max_bytes} bytes.",
                status.HTTP_413_CONTENT_TOO_LARGE,
            )
        chunks.append(chunk)
    body = b"".join(chunks)
    # Cache it the way ``Request.body()`` does so the route still receives it.
    request._body = body
    return body


async def _validate_generic_body_hash(request: Request, expected_hash: str) -> None:
    """Verify the signed body hash against the actual request body.

    Call this only after the proxy signature over ``expected_hash`` is verified,
    so an unsigned request never has its body read.
    """
    body = await _read_generic_body(request)
    actual_hash = hash_proxy_body(body)
    if not hmac.compare_digest(expected_hash, actual_hash):
        raise ProxyAuthError(
            "X-Moto body hash does not match the received request body.",
            status.HTTP_403_FORBIDDEN,
        )

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request._receive = receive


def setup_middleware(app: FastAPI) -> None:
    """Setup middleware for the FastAPI app."""
    
    # Allow custom origins via environment variable (comma-separated)
    # Example: CORS_ORIGINS=http://localhost:3000,http://example.com
    custom_origins = os.environ.get("MOTO_CORS_ORIGINS", "") or os.environ.get("CORS_ORIGINS", "")
    if custom_origins:
        origins = [o.strip() for o in custom_origins.split(",") if o.strip()]
        logger.info(f"Using custom CORS origins: {origins}")
    else:
        origins = DEFAULT_ORIGINS
        logger.info(f"Using default CORS origins: {origins}")
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def moto_request_auth(request: Request, call_next):
        """Require hosted proxy auth or desktop instance tokens for protected routes."""
        if system_config.generic_mode:
            try:
                if not is_proxy_auth_allowlisted(request.method, routed_path(request.scope)):
                    _validate_generic_content_length(request)

                body_hash = request.headers.get(PROXY_BODY_SHA256_HEADER)
                has_signed_body = (
                    not is_proxy_auth_allowlisted(request.method, routed_path(request.scope))
                    and request.method.upper() not in {"GET", "HEAD"}
                )
                if has_signed_body and not body_hash:
                    raise ProxyAuthError(
                        "Missing required X-Moto body hash header.",
                        status.HTTP_401_UNAUTHORIZED,
                    )
                # Verify the signature over the claimed body hash first, then read
                # the body and check it against that hash. An unsigned request is
                # rejected before any of its body is read.
                validate_proxy_headers(
                    request.headers,
                    method=request.method,
                    path=routed_path(request.scope),
                    query_string=routed_query(request.scope),
                    body_hash=(body_hash or "") if has_signed_body else EMPTY_BODY_SHA256,
                    expected_instance_id=system_config.instance_id,
                    shared_secret=system_config.internal_proxy_secret or "",
                )
                if has_signed_body:
                    await _validate_generic_body_hash(request, body_hash or "")
            except ProxyAuthError as exc:
                logger.warning("Rejected generic-mode request %s %s: %s", request.method, routed_path(request.scope), exc.detail)
                return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        else:
            try:
                _validate_desktop_token(request, origins)
            except ProxyAuthError as exc:
                _log_desktop_auth_rejection(request, exc)
                return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

        return await call_next(request)

    @app.exception_handler(Exception)
    async def uncaught_error_response(request: Request, exc: Exception):
        """Return a generic 500; never echo exception text to the client.

        The server-error middleware still logs the failure through the
        process-wide trade-data log boundary.
        """
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
        )
    
    logger.info("Middleware configured")
