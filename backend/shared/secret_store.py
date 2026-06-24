"""
Secure secret persistence for API keys.

Stores user-provided credentials in the OS-backed keyring instead of browser
storage so keys survive restarts without being written to frontend localStorage.
"""
from typing import Optional
import json
import logging

import keyring
from keyring.errors import KeyringError, PasswordDeleteError

from backend.shared.config import system_config

logger = logging.getLogger(__name__)

_DEFAULT_SERVICE_NAME = "MOTO-Autonomous-ASI"
_OPENROUTER_KEY = "openrouter_api_key"
_OPENAI_CODEX_OAUTH = "openai_codex_oauth"
_OPENAI_CODEX_OAUTH_CHUNK_PREFIX = "openai_codex_oauth_chunk"
_OPENAI_CODEX_OAUTH_CHUNK_COUNT = "openai_codex_oauth_chunk_count"
_XAI_GROK_OAUTH = "xai_grok_oauth"
_XAI_GROK_OAUTH_CHUNK_PREFIX = "xai_grok_oauth_chunk"
_XAI_GROK_OAUTH_CHUNK_COUNT = "xai_grok_oauth_chunk_count"
_SAKANA_FUGU_API_KEY = "sakana_fugu_api_key"
_SYNTHETICLIB4_API_KEY = "syntheticlib4_api_key"
# Windows Credential Manager limits blobs to 2560 bytes, which is about
# 1280 UTF-16 characters through keyring/win32cred. Keep chunks below that.
_SECRET_CHUNK_SIZE = 1000
_WOLFRAM_KEY = "wolfram_alpha_api_key"


class SecretStoreError(RuntimeError):
    """Raised when the secure secret store is unavailable or fails."""


def _get_service_name() -> str:
    """Return the OS-keyring service name for the active instance."""
    namespace = system_config.secret_namespace
    if namespace:
        return f"{_DEFAULT_SERVICE_NAME}::{namespace}"
    return _DEFAULT_SERVICE_NAME


def _normalize_secret(value: Optional[str]) -> Optional[str]:
    """Trim whitespace and collapse empty values to None."""
    if value is None:
        return None

    stripped = value.strip()
    return stripped or None


def _get_secret(secret_name: str) -> Optional[str]:
    """Load a secret from the OS-backed keyring."""
    try:
        return _normalize_secret(keyring.get_password(_get_service_name(), secret_name))
    except KeyringError as exc:
        raise SecretStoreError(
            "Secure credential storage is unavailable. Please ensure the OS keyring is accessible."
        ) from exc


def _set_secret(secret_name: str, secret_value: str) -> None:
    """Persist a secret to the OS-backed keyring."""
    normalized = _normalize_secret(secret_value)
    if not normalized:
        raise ValueError("Secret value is required")

    try:
        keyring.set_password(_get_service_name(), secret_name, normalized)
    except KeyringError as exc:
        raise SecretStoreError(
            "Failed to persist the credential in the OS keyring."
        ) from exc


def _delete_secret(secret_name: str) -> None:
    """Delete a persisted secret if one exists."""
    try:
        keyring.delete_password(_get_service_name(), secret_name)
    except PasswordDeleteError:
        return
    except KeyringError as exc:
        raise SecretStoreError(
            "Failed to delete the credential from the OS keyring."
        ) from exc


def get_active_service_name() -> str:
    """Return the OS-keyring service name this process is currently using.

    Exposed for startup diagnostics so operators can verify the keyring
    namespace has not drifted between launches (which would make saved API
    keys look like they "disappeared").
    """
    return _get_service_name()


def load_openrouter_api_key() -> Optional[str]:
    """Load the persisted global OpenRouter API key."""
    return _get_secret(_OPENROUTER_KEY)


def store_openrouter_api_key(api_key: str) -> None:
    """Persist the global OpenRouter API key securely."""
    _set_secret(_OPENROUTER_KEY, api_key)


def clear_openrouter_api_key() -> None:
    """Delete the persisted global OpenRouter API key."""
    _delete_secret(_OPENROUTER_KEY)


def _load_chunked_secret(prefix: str, count_name: str) -> Optional[str]:
    """Load a large secret split across several keyring entries."""
    raw_count = _get_secret(count_name)
    if not raw_count:
        return None
    try:
        count = int(raw_count)
    except ValueError as exc:
        raise SecretStoreError("Stored chunked credential metadata is invalid.") from exc
    if count < 1 or count > 100:
        raise SecretStoreError("Stored chunked credential metadata is out of range.")
    chunks = []
    for index in range(count):
        chunk = _get_secret(f"{prefix}_{index}")
        if chunk is None:
            raise SecretStoreError("Stored chunked credential is incomplete.")
        chunks.append(chunk)
    return "".join(chunks)


def _store_chunked_secret(prefix: str, count_name: str, secret_value: str) -> None:
    """Store a large secret across several keyring entries."""
    _delete_chunked_secret(prefix, count_name)
    chunks = [
        secret_value[index:index + _SECRET_CHUNK_SIZE]
        for index in range(0, len(secret_value), _SECRET_CHUNK_SIZE)
    ]
    if not chunks:
        raise ValueError("Secret value is required")
    for index, chunk in enumerate(chunks):
        _set_secret(f"{prefix}_{index}", chunk)
    _set_secret(count_name, str(len(chunks)))


def _delete_chunked_secret(prefix: str, count_name: str) -> None:
    """Delete a chunked secret, tolerating missing chunks."""
    raw_count = None
    try:
        raw_count = _get_secret(count_name)
    except SecretStoreError:
        raw_count = None
    max_count = 100
    if raw_count:
        try:
            max_count = max(100, int(raw_count) + 5)
        except ValueError:
            max_count = 100
    for index in range(max_count):
        try:
            _delete_secret(f"{prefix}_{index}")
        except SecretStoreError:
            continue
    try:
        _delete_secret(count_name)
    except SecretStoreError:
        return


def load_openai_codex_oauth_tokens() -> Optional[dict]:
    """Load persisted OpenAI Codex OAuth tokens."""
    raw_value = _load_chunked_secret(_OPENAI_CODEX_OAUTH_CHUNK_PREFIX, _OPENAI_CODEX_OAUTH_CHUNK_COUNT)
    if not raw_value:
        raw_value = _get_secret(_OPENAI_CODEX_OAUTH)
    if not raw_value:
        return None
    try:
        payload = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise SecretStoreError("Stored OpenAI Codex OAuth token payload is invalid.") from exc
    return payload if isinstance(payload, dict) else None


def store_openai_codex_oauth_tokens(tokens: dict) -> None:
    """Persist OpenAI Codex OAuth tokens securely."""
    payload = json.dumps(tokens, separators=(",", ":"))
    _store_chunked_secret(_OPENAI_CODEX_OAUTH_CHUNK_PREFIX, _OPENAI_CODEX_OAUTH_CHUNK_COUNT, payload)
    # Remove the pre-chunking storage entry if it exists.
    _delete_secret(_OPENAI_CODEX_OAUTH)


def clear_openai_codex_oauth_tokens() -> None:
    """Delete persisted OpenAI Codex OAuth tokens."""
    _delete_secret(_OPENAI_CODEX_OAUTH)
    _delete_chunked_secret(_OPENAI_CODEX_OAUTH_CHUNK_PREFIX, _OPENAI_CODEX_OAUTH_CHUNK_COUNT)


def load_xai_grok_oauth_tokens() -> Optional[dict]:
    """Load persisted xAI Grok OAuth tokens."""
    raw_value = _load_chunked_secret(_XAI_GROK_OAUTH_CHUNK_PREFIX, _XAI_GROK_OAUTH_CHUNK_COUNT)
    if not raw_value:
        raw_value = _get_secret(_XAI_GROK_OAUTH)
    if not raw_value:
        return None
    try:
        payload = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise SecretStoreError("Stored xAI Grok OAuth token payload is invalid.") from exc
    return payload if isinstance(payload, dict) else None


def store_xai_grok_oauth_tokens(tokens: dict) -> None:
    """Persist xAI Grok OAuth tokens securely."""
    payload = json.dumps(tokens, separators=(",", ":"))
    _store_chunked_secret(_XAI_GROK_OAUTH_CHUNK_PREFIX, _XAI_GROK_OAUTH_CHUNK_COUNT, payload)
    _delete_secret(_XAI_GROK_OAUTH)


def clear_xai_grok_oauth_tokens() -> None:
    """Delete persisted xAI Grok OAuth tokens."""
    _delete_secret(_XAI_GROK_OAUTH)
    _delete_chunked_secret(_XAI_GROK_OAUTH_CHUNK_PREFIX, _XAI_GROK_OAUTH_CHUNK_COUNT)


def load_sakana_fugu_api_key() -> Optional[str]:
    """Load the persisted Sakana Fugu API key."""
    return _get_secret(_SAKANA_FUGU_API_KEY)


def store_sakana_fugu_api_key(api_key: str) -> None:
    """Persist the Sakana Fugu API key securely."""
    _set_secret(_SAKANA_FUGU_API_KEY, api_key)


def clear_sakana_fugu_api_key() -> None:
    """Delete the persisted Sakana Fugu API key."""
    _delete_secret(_SAKANA_FUGU_API_KEY)


def load_syntheticlib4_api_key() -> Optional[str]:
    """Load the persisted SyntheticLib4 corpus API key."""
    return _get_secret(_SYNTHETICLIB4_API_KEY)


def store_syntheticlib4_api_key(api_key: str) -> None:
    """Persist the SyntheticLib4 corpus API key securely."""
    _set_secret(_SYNTHETICLIB4_API_KEY, api_key)


def clear_syntheticlib4_api_key() -> None:
    """Delete the persisted SyntheticLib4 corpus API key."""
    _delete_secret(_SYNTHETICLIB4_API_KEY)


def load_wolfram_api_key() -> Optional[str]:
    """Load the persisted Wolfram Alpha API key."""
    return _get_secret(_WOLFRAM_KEY)


def store_wolfram_api_key(api_key: str) -> None:
    """Persist the Wolfram Alpha API key securely."""
    _set_secret(_WOLFRAM_KEY, api_key)


def clear_wolfram_api_key() -> None:
    """Delete the persisted Wolfram Alpha API key."""
    _delete_secret(_WOLFRAM_KEY)
