"""
Middleware for CORS and error handling.
"""
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from backend.api.proxy_auth import ProxyAuthError, validate_proxy_headers
from backend.shared.config import system_config

logger = logging.getLogger(__name__)

# Default allowed origins for local development
DEFAULT_ORIGINS = [
    f"http://localhost:{system_config.frontend_port}",
    f"http://127.0.0.1:{system_config.frontend_port}",
    f"http://localhost:{system_config.backend_port}",
    f"http://127.0.0.1:{system_config.backend_port}",
]


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
    async def generic_mode_proxy_auth(request: Request, call_next):
        """Require signed internal proxy headers for protected hosted routes."""
        if system_config.generic_mode:
            try:
                validate_proxy_headers(
                    request.headers,
                    method=request.method,
                    path=request.url.path,
                    expected_instance_id=system_config.instance_id,
                    shared_secret=system_config.internal_proxy_secret or "",
                )
            except ProxyAuthError as exc:
                logger.warning("Rejected generic-mode request %s %s: %s", request.method, request.url.path, exc.detail)
                return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

        return await call_next(request)
    
    logger.info("Middleware configured")
