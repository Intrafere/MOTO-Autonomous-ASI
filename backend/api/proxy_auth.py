"""
Generic-mode internal proxy authentication helpers.
"""
from __future__ import annotations

import hashlib
import hmac
import time
from typing import Mapping

from fastapi import status

PROXY_INSTANCE_HEADER = "X-Moto-Instance-Id"
PROXY_TIMESTAMP_HEADER = "X-Moto-Proxy-Timestamp"
PROXY_SIGNATURE_HEADER = "X-Moto-Proxy-Signature"
PROXY_AUTH_MAX_SKEW_SECONDS = 60
PROXY_AUTH_ALLOWLIST = {
    ("GET", "/health"),
    ("GET", "/api/health"),
    ("GET", "/api/features"),
}


class ProxyAuthError(RuntimeError):
    """Raised when generic-mode proxy authentication fails."""

    def __init__(self, detail: str, status_code: int):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


def normalize_proxy_path(path: str) -> str:
    """Normalize request paths before signing or validating them."""
    normalized = (path or "").strip()
    return normalized or "/"


def is_proxy_auth_allowlisted(method: str, path: str) -> bool:
    """Return True when a route is intentionally public in generic mode."""
    normalized_method = (method or "").upper()
    normalized_path = normalize_proxy_path(path)
    if normalized_method == "OPTIONS":
        return True
    return (normalized_method, normalized_path) in PROXY_AUTH_ALLOWLIST


def build_proxy_signature(secret: str, instance_id: str, timestamp: str, method: str, path: str) -> str:
    """Build the expected HMAC signature for a proxied request."""
    payload = f"{instance_id}:{timestamp}:{(method or '').upper()}:{normalize_proxy_path(path)}"
    return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def validate_proxy_headers(
    headers: Mapping[str, str],
    *,
    method: str,
    path: str,
    expected_instance_id: str,
    shared_secret: str,
    now: int | None = None,
) -> None:
    """Validate the signed generic-mode proxy headers for one request."""
    if is_proxy_auth_allowlisted(method, path):
        return

    if not shared_secret:
        raise ProxyAuthError(
            "Generic-mode proxy authentication is not configured for this runtime.",
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    instance_id = (headers.get(PROXY_INSTANCE_HEADER) or "").strip()
    timestamp_raw = (headers.get(PROXY_TIMESTAMP_HEADER) or "").strip()
    signature = (headers.get(PROXY_SIGNATURE_HEADER) or "").strip()

    if not instance_id or not timestamp_raw or not signature:
        raise ProxyAuthError(
            "Missing required X-Moto proxy authentication headers.",
            status.HTTP_401_UNAUTHORIZED,
        )

    if instance_id != expected_instance_id:
        raise ProxyAuthError(
            "X-Moto-Instance-Id does not match the active runtime instance.",
            status.HTTP_403_FORBIDDEN,
        )

    try:
        timestamp_value = int(timestamp_raw)
    except ValueError as exc:
        raise ProxyAuthError(
            "Invalid X-Moto-Proxy-Timestamp header.",
            status.HTTP_401_UNAUTHORIZED,
        ) from exc

    current_time = int(time.time() if now is None else now)
    if abs(current_time - timestamp_value) > PROXY_AUTH_MAX_SKEW_SECONDS:
        raise ProxyAuthError(
            "X-Moto-Proxy-Timestamp is outside the allowed clock-skew window.",
            status.HTTP_401_UNAUTHORIZED,
        )

    expected_signature = build_proxy_signature(
        secret=shared_secret,
        instance_id=expected_instance_id,
        timestamp=timestamp_raw,
        method=method,
        path=path,
    )
    if not hmac.compare_digest(signature, expected_signature):
        raise ProxyAuthError(
            "Invalid X-Moto-Proxy-Signature for the requested path.",
            status.HTTP_403_FORBIDDEN,
        )
