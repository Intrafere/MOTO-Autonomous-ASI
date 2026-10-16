"""
Secure secret persistence for API keys.

Stores user-provided credentials in the OS-backed keyring instead of browser
storage so keys survive restarts without being written to frontend localStorage.
"""
from typing import Optional
import json
import logging
import re

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
_SYNTHETICLIB4_API_KEY = "syntheticlib4_api_key"
_MOTO_INSTALLATION_IDENTITY_STATE = "moto_installation_identity_state"
# Windows Credential Manager limits blobs to 2560 bytes, which is about
# 1280 UTF-16 characters through keyring/win32cred. Keep chunks below that.
_SECRET_CHUNK_SIZE = 1000
_WOLFRAM_KEY = "wolfram_alpha_api_key"
# Subscription providers store one single-value key each (e.g. "sakana_fugu_api_key").
_SUBSCRIPTION_SECRET_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,62}_api_key")
_RESERVED_SECRET_NAMES = frozenset({_OPENROUTER_KEY, _SYNTHETICLIB4_API_KEY, _WOLFRAM_KEY})


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


def _get_global_secret(secret_name: str) -> Optional[str]:
    """Load installation-global state outside configurable instance namespaces."""
    try:
        return _normalize_secret(keyring.get_password(_DEFAULT_SERVICE_NAME, secret_name))
    except KeyringError as exc:
        raise SecretStoreError(
            "Secure installation identity storage is unavailable."
        ) from exc


def _set_global_secret(secret_name: str, secret_value: str) -> None:
    normalized = _normalize_secret(secret_value)
    if not normalized:
        raise ValueError("Installation identity state is required")
    try:
        keyring.set_password(_DEFAULT_SERVICE_NAME, secret_name, normalized)
    except KeyringError as exc:
        raise SecretStoreError(
            "Failed to persist the installation identity securely."
        ) from exc


def _delete_global_secret(secret_name: str) -> None:
    try:
        keyring.delete_password(_DEFAULT_SERVICE_NAME, secret_name)
    except PasswordDeleteError:
        return
    except KeyringError as exc:
        raise SecretStoreError(
            "Failed to clear the installation identity securely."
        ) from exc


def _load_atomic_secret(secret_name: str) -> Optional[str]:
    """Load a single-value secret from its committed slot or legacy entry."""
    active_slot = _get_secret(f"{secret_name}_active")
    if active_slot in {"a", "b"}:
        value = _get_secret(f"{secret_name}_{active_slot}")
        if value is None:
            raise SecretStoreError("Stored credential commit points to a missing value.")
        return value
    return _get_secret(secret_name)


def _store_atomic_secret(secret_name: str, secret_value: str) -> None:
    """Replace a secret without overwriting the previously committed value."""
    normalized = _normalize_secret(secret_value)
    if not normalized:
        raise ValueError("Secret value is required")
    current_slot = _get_secret(f"{secret_name}_active")
    next_slot = "b" if current_slot == "a" else "a"
    next_name = f"{secret_name}_{next_slot}"
    _set_secret(next_name, normalized)
    if _get_secret(next_name) != normalized:
        try:
            _delete_secret(next_name)
        except SecretStoreError:
            logger.warning("Failed to clean an uncommitted credential slot", exc_info=True)
        raise SecretStoreError("Stored credential could not be verified.")
    try:
        _set_secret(f"{secret_name}_active", next_slot)
    except Exception:
        try:
            _delete_secret(next_name)
        except SecretStoreError:
            logger.warning("Failed to clean an uncommitted credential slot", exc_info=True)
        raise

    obsolete_names = [secret_name]
    if current_slot in {"a", "b"}:
        obsolete_names.append(f"{secret_name}_{current_slot}")
    for obsolete_name in obsolete_names:
        try:
            _delete_secret(obsolete_name)
        except SecretStoreError:
            logger.warning("Committed credential; obsolete keyring entry cleanup failed", exc_info=True)


def _clear_atomic_secret(secret_name: str) -> None:
    """Remove committed, inactive, and legacy copies of a single-value secret."""
    previous_value = _load_atomic_secret(secret_name)
    try:
        for name in (
            secret_name,
            f"{secret_name}_a",
            f"{secret_name}_b",
            f"{secret_name}_active",
        ):
            _delete_secret(name)
    except SecretStoreError:
        if previous_value:
            try:
                _store_atomic_secret(secret_name, previous_value)
            except Exception:
                logger.error("Failed to restore credential after an incomplete clear", exc_info=True)
        raise


def keyring_backend_available() -> bool:
    """Report, without reading or writing entries, whether a real OS keyring is usable."""
    if system_config.generic_mode:
        return False
    try:
        backend = keyring.get_keyring()
        priority = backend.priority
    except Exception:
        return False
    try:
        from keyring.backends import fail as fail_backend
    except ImportError:  # pragma: no cover - keyring always ships the fail backend
        fail_backend = None
    if fail_backend is not None and isinstance(backend, fail_backend.Keyring):
        return False
    try:
        return float(priority) > 0
    except (TypeError, ValueError):
        return False


def get_active_service_name() -> str:
    """Return the OS-keyring service name this process is currently using.

    Exposed for startup diagnostics so operators can verify the keyring
    namespace has not drifted between launches (which would make saved API
    keys look like they "disappeared").
    """
    return _get_service_name()


def load_openrouter_api_key() -> Optional[str]:
    """Load the persisted global OpenRouter API key."""
    return _load_atomic_secret(_OPENROUTER_KEY)


def store_openrouter_api_key(api_key: str) -> None:
    """Persist the global OpenRouter API key securely."""
    _store_atomic_secret(_OPENROUTER_KEY, api_key)


def clear_openrouter_api_key() -> None:
    """Delete the persisted global OpenRouter API key."""
    _clear_atomic_secret(_OPENROUTER_KEY)


def _load_chunked_secret(prefix: str, count_name: str) -> Optional[str]:
    """Load a large secret split across several keyring entries."""
    active_slot = _get_secret(f"{count_name}_active")
    effective_prefix = f"{prefix}_{active_slot}" if active_slot in {"a", "b"} else prefix
    effective_count_name = f"{count_name}_{active_slot}" if active_slot in {"a", "b"} else count_name
    raw_count = _get_secret(effective_count_name)
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
        chunk = _get_secret(f"{effective_prefix}_{index}")
        if chunk is None:
            raise SecretStoreError("Stored chunked credential is incomplete.")
        chunks.append(chunk)
    return "".join(chunks)


def _store_chunked_secret(prefix: str, count_name: str, secret_value: str) -> None:
    """Atomically replace a chunked secret using alternating committed slots."""
    chunks = [
        secret_value[index:index + _SECRET_CHUNK_SIZE]
        for index in range(0, len(secret_value), _SECRET_CHUNK_SIZE)
    ]
    if not chunks:
        raise ValueError("Secret value is required")
    current_slot = _get_secret(f"{count_name}_active")
    next_slot = "b" if current_slot == "a" else "a"
    next_prefix = f"{prefix}_{next_slot}"
    next_count_name = f"{count_name}_{next_slot}"
    _delete_chunked_slot(next_prefix, next_count_name)
    for index, chunk in enumerate(chunks):
        _set_secret(f"{next_prefix}_{index}", chunk)
    _set_secret(next_count_name, str(len(chunks)))
    if _load_chunked_slot(next_prefix, next_count_name) != secret_value:
        _delete_chunked_slot(next_prefix, next_count_name)
        raise SecretStoreError("Stored chunked credential could not be verified.")
    try:
        _set_secret(f"{count_name}_active", next_slot)
    except Exception:
        try:
            _delete_chunked_slot(next_prefix, next_count_name)
        except SecretStoreError:
            logger.warning("Failed to clean an uncommitted OAuth credential slot", exc_info=True)
        raise
    if current_slot in {"a", "b"}:
        try:
            _delete_chunked_slot(f"{prefix}_{current_slot}", f"{count_name}_{current_slot}")
        except SecretStoreError:
            logger.warning("Committed OAuth credential; obsolete slot cleanup failed", exc_info=True)
    else:
        try:
            _delete_chunked_slot(prefix, count_name)
        except SecretStoreError:
            logger.warning("Committed OAuth credential; legacy entry cleanup failed", exc_info=True)


def _load_chunked_slot(prefix: str, count_name: str) -> Optional[str]:
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


def _delete_chunked_slot(prefix: str, count_name: str) -> None:
    raw_count = _get_secret(count_name)
    count = 100
    if raw_count:
        try:
            count = max(count, int(raw_count) + 5)
        except ValueError:
            pass
    for index in range(count):
        _delete_secret(f"{prefix}_{index}")
    _delete_secret(count_name)


def _delete_chunked_secret(prefix: str, count_name: str) -> None:
    """Delete active, inactive, and legacy chunked credential storage."""
    previous_value = _load_chunked_secret(prefix, count_name)
    try:
        _delete_chunked_slot(prefix, count_name)
        _delete_chunked_slot(f"{prefix}_a", f"{count_name}_a")
        _delete_chunked_slot(f"{prefix}_b", f"{count_name}_b")
        _delete_secret(f"{count_name}_active")
    except SecretStoreError:
        if previous_value:
            try:
                _store_chunked_secret(prefix, count_name, previous_value)
            except Exception:
                logger.error("Failed to restore OAuth credential after an incomplete clear", exc_info=True)
        raise


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


def _subscription_secret_name(secret_name: str) -> str:
    name = str(secret_name or "").strip()
    if not _SUBSCRIPTION_SECRET_NAME_PATTERN.fullmatch(name) or name in _RESERVED_SECRET_NAMES:
        raise ValueError("Subscription API-key secret names must look like '<provider>_api_key'.")
    return name


def load_subscription_api_key(secret_name: str) -> Optional[str]:
    """Load a persisted subscription-provider API key (for example Sakana Fugu or Kimi Code)."""
    return _load_atomic_secret(_subscription_secret_name(secret_name))


def store_subscription_api_key(secret_name: str, api_key: str) -> None:
    """Persist a subscription-provider API key securely."""
    _store_atomic_secret(_subscription_secret_name(secret_name), api_key)


def clear_subscription_api_key(secret_name: str) -> None:
    """Delete a persisted subscription-provider API key."""
    _clear_atomic_secret(_subscription_secret_name(secret_name))


def load_syntheticlib4_api_key() -> Optional[str]:
    """Load the persisted SyntheticLib4 corpus API key."""
    return _load_atomic_secret(_SYNTHETICLIB4_API_KEY)


def store_syntheticlib4_api_key(api_key: str) -> None:
    """Persist the SyntheticLib4 corpus API key securely."""
    _store_atomic_secret(_SYNTHETICLIB4_API_KEY, api_key)


def clear_syntheticlib4_api_key() -> None:
    """Delete the persisted SyntheticLib4 corpus API key."""
    _clear_atomic_secret(_SYNTHETICLIB4_API_KEY)


def load_moto_installation_identity_state() -> Optional[str]:
    """Load the installation-global identity/root-binding registry."""
    return _get_global_secret(_MOTO_INSTALLATION_IDENTITY_STATE)


def store_moto_installation_identity_state(state: str) -> None:
    """Persist the installation-global identity/root-binding registry."""
    _set_global_secret(_MOTO_INSTALLATION_IDENTITY_STATE, state)


def clear_moto_installation_identity_state() -> None:
    """Explicitly clear the installation-global identity/root-binding registry."""
    _delete_global_secret(_MOTO_INSTALLATION_IDENTITY_STATE)


def load_wolfram_api_key() -> Optional[str]:
    """Load the persisted Wolfram Alpha API key."""
    return _load_atomic_secret(_WOLFRAM_KEY)


def store_wolfram_api_key(api_key: str) -> None:
    """Persist the Wolfram Alpha API key securely."""
    _store_atomic_secret(_WOLFRAM_KEY, api_key)


def clear_wolfram_api_key() -> None:
    """Delete the persisted Wolfram Alpha API key."""
    _clear_atomic_secret(_WOLFRAM_KEY)
