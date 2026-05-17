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
PROXY_BODY_SHA256_HEADER = "X-Moto-Body-SHA256"
PROXY_AUTH_MAX_SKEW_SECONDS = 60
PROXY_REPLAY_CACHE_MAX_ENTRIES = 4096
EMPTY_BODY_SHA256 = hashlib.sha256(b"").hexdigest()
PROXY_AUTH_ALLOWLIST = {
    ("GET", "/health"),
    ("GET", "/api/health"),
    ("GET", "/api/features"),
}
_SEEN_PROXY_SIGNATURES: dict[str, int] = {}


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


def normalize_proxy_query(query_string: str | bytes | None) -> str:
    """Normalize the raw query string used for proxy signatures."""
    if isinstance(query_string, bytes):
        query_string = query_string.decode("utf-8", errors="surrogatepass")
    normalized = (query_string or "").strip()
    return normalized[1:] if normalized.startswith("?") else normalized


def hash_proxy_body(body: bytes | str | None) -> str:
    """Return the SHA-256 hex digest for the request body."""
    if body is None:
        raw_body = b""
    elif isinstance(body, bytes):
        raw_body = body
    else:
        raw_body = body.encode("utf-8")
    return hashlib.sha256(raw_body).hexdigest()


def _remember_proxy_signature(signature: str, timestamp_value: int, current_time: int) -> None:
    """Reject replayed signatures within the accepted timestamp skew window."""
    stale_cutoff = current_time - PROXY_AUTH_MAX_SKEW_SECONDS
    stale_signatures = [
        seen_signature
        for seen_signature, seen_timestamp in _SEEN_PROXY_SIGNATURES.items()
        if seen_timestamp < stale_cutoff
    ]
    for seen_signature in stale_signatures:
        _SEEN_PROXY_SIGNATURES.pop(seen_signature, None)

    if signature in _SEEN_PROXY_SIGNATURES:
        raise ProxyAuthError(
            "Replayed X-Moto-Proxy-Signature was rejected.",
            status.HTTP_401_UNAUTHORIZED,
        )

    _SEEN_PROXY_SIGNATURES[signature] = timestamp_value
    if len(_SEEN_PROXY_SIGNATURES) > PROXY_REPLAY_CACHE_MAX_ENTRIES:
        for seen_signature, _ in sorted(
            _SEEN_PROXY_SIGNATURES.items(),
            key=lambda item: item[1],
        )[: len(_SEEN_PROXY_SIGNATURES) - PROXY_REPLAY_CACHE_MAX_ENTRIES]:
            _SEEN_PROXY_SIGNATURES.pop(seen_signature, None)


def is_proxy_auth_allowlisted(method: str, path: str) -> bool:
    """Return True when a route is intentionally public in generic mode."""
    normalized_method = (method or "").upper()
    normalized_path = normalize_proxy_path(path)
    if normalized_method == "OPTIONS":
        return True
    return (normalized_method, normalized_path) in PROXY_AUTH_ALLOWLIST


def build_proxy_signature(
    secret: str,
    instance_id: str,
    timestamp: str,
    method: str,
    path: str,
    query_string: str | bytes | None = "",
    body_hash: str | None = EMPTY_BODY_SHA256,
) -> str:
    """Build the expected HMAC signature for a proxied request."""
    payload = "\n".join(
        (
            instance_id,
            timestamp,
            (method or "").upper(),
            normalize_proxy_path(path),
            normalize_proxy_query(query_string),
            body_hash or EMPTY_BODY_SHA256,
        )
    )
    return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def validate_proxy_headers(
    headers: Mapping[str, str],
    *,
    method: str,
    path: str,
    query_string: str | bytes | None = "",
    body: bytes | str | None = b"",
    body_hash: str | None = None,
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
        query_string=query_string,
        body_hash=body_hash or hash_proxy_body(body),
    )
    if not hmac.compare_digest(signature, expected_signature):
        raise ProxyAuthError(
            "Invalid X-Moto-Proxy-Signature for the requested path.",
            status.HTTP_403_FORBIDDEN,
        )

    _remember_proxy_signature(signature, timestamp_value, current_time)
