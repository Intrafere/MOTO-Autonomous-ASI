"""
Middleware for CORS and error handling.
"""
import hmac
import os
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


def _origin_from_url(value: str) -> str:
    """Return scheme://host[:port] for an Origin/Referer-like value."""
    parsed = urlparse(value or "")
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}"


def _validate_desktop_token(request: Request, allowed_origins: list[str]) -> None:
    """Require the launcher-provided desktop API token outside public routes."""
    if is_proxy_auth_allowlisted(request.method, request.url.path):
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

    max_bytes = max(int(system_config.generic_max_request_bytes or 0), 1)
    if content_length > max_bytes:
        raise ProxyAuthError(
            f"Request body exceeds hosted limit of {max_bytes} bytes.",
            status.HTTP_413_CONTENT_TOO_LARGE,
        )


async def _validate_generic_body_hash(request: Request, expected_hash: str) -> str:
    """Verify the signed body hash against the actual request body."""
    body = await request.body()
    actual_hash = hash_proxy_body(body)
    if not hmac.compare_digest(expected_hash, actual_hash):
        raise ProxyAuthError(
            "X-Moto body hash does not match the received request body.",
            status.HTTP_403_FORBIDDEN,
        )

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request._receive = receive
    return actual_hash


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
                if not is_proxy_auth_allowlisted(request.method, request.url.path):
                    _validate_generic_content_length(request)

                body_hash = request.headers.get(PROXY_BODY_SHA256_HEADER)
                verified_body_hash = EMPTY_BODY_SHA256
                if (
                    not is_proxy_auth_allowlisted(request.method, request.url.path)
                    and request.method.upper() not in {"GET", "HEAD"}
                    and not body_hash
                ):
                    raise ProxyAuthError(
                        "Missing required X-Moto body hash header.",
                        status.HTTP_401_UNAUTHORIZED,
                    )
                if (
                    not is_proxy_auth_allowlisted(request.method, request.url.path)
                    and request.method.upper() not in {"GET", "HEAD"}
                ):
                    verified_body_hash = await _validate_generic_body_hash(request, body_hash or "")
                validate_proxy_headers(
                    request.headers,
                    method=request.method,
                    path=request.url.path,
                    query_string=request.url.query,
                    body_hash=verified_body_hash,
                    expected_instance_id=system_config.instance_id,
                    shared_secret=system_config.internal_proxy_secret or "",
                )
            except ProxyAuthError as exc:
                logger.warning("Rejected generic-mode request %s %s: %s", request.method, request.url.path, exc.detail)
                return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        else:
            try:
                _validate_desktop_token(request, origins)
            except ProxyAuthError as exc:
                logger.warning("Rejected desktop request %s %s: %s", request.method, request.url.path, exc.detail)
                return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

        return await call_next(request)
    
    logger.info("Middleware configured")
