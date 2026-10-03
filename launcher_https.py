"""Verified HTTPS helpers shared by the MOTO launcher and updater."""
from __future__ import annotations

import os
import ssl
import sys
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def create_launcher_ssl_context() -> ssl.SSLContext:
    """Build a verified context using native Windows trust when appropriate."""
    explicit_ca_config = bool(
        os.environ.get("SSL_CERT_FILE") or os.environ.get("SSL_CERT_DIR")
    )
    if sys.platform == "win32" and not explicit_ca_config:
        try:
            import truststore

            return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        except ImportError:
            # The one-click launcher bootstraps truststore before its first
            # request. Direct script callers still receive a verified stdlib
            # context if that optional bootstrap has not run.
            pass
    return ssl.create_default_context()


def verified_urlopen(
    request: str | Request,
    *,
    timeout: float | None = None,
) -> Any:
    """Open an HTTPS URL with hostname and certificate verification enabled."""
    url = request.full_url if isinstance(request, Request) else str(request)
    if urlparse(url).scheme.lower() != "https":
        raise ValueError("verified_urlopen only accepts HTTPS URLs")

    context = create_launcher_ssl_context()
    if context.verify_mode != ssl.CERT_REQUIRED or not context.check_hostname:
        raise RuntimeError("Launcher HTTPS verification is not enabled")
    return urlopen(request, timeout=timeout, context=context)
