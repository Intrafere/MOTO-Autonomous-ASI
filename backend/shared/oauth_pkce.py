"""Shared OAuth PKCE (RFC 7636, S256 only) and state primitives.

Desktop provider logins and the future proof-trade native client use these
helpers. ``plain`` PKCE is never supported. Verifiers, challenges, and state
values are secrets for the lifetime of a pending login and must never be
logged or persisted.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

PKCE_METHOD = "S256"
PKCE_VERIFIER_BYTES = 64
OAUTH_STATE_BYTES = 32


class UnsupportedPkceMethodError(ValueError):
    """Raised when anything other than S256 PKCE is requested."""


def require_s256(method: str) -> str:
    """Return ``S256`` or reject the method (including ``plain``)."""
    if method != PKCE_METHOD:
        raise UnsupportedPkceMethodError("Only S256 PKCE is supported.")
    return PKCE_METHOD


def pkce_s256_challenge(verifier: str) -> str:
    """Return the unpadded base64url SHA-256 challenge for ``verifier``."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def generate_pkce_pair() -> tuple[str, str]:
    """Return a PKCE verifier and its S256 challenge."""
    verifier = secrets.token_urlsafe(PKCE_VERIFIER_BYTES)
    return verifier, pkce_s256_challenge(verifier)


def generate_oauth_state() -> str:
    """Return an unguessable single-login OAuth state value."""
    return secrets.token_urlsafe(OAUTH_STATE_BYTES)


def oauth_state_matches(expected: str, received: str) -> bool:
    """Compare state values in constant time; non-strings never match."""
    if not isinstance(expected, str) or not isinstance(received, str):
        return False
    if not expected or not received:
        return False
    return hmac.compare_digest(expected.encode("utf-8"), received.encode("utf-8"))
