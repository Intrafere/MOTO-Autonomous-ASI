"""
Secure secret persistence for API keys.

Stores user-provided credentials in the OS-backed keyring instead of browser
storage so keys survive restarts without being written to frontend localStorage.
"""
from typing import Optional
import logging

import keyring
from keyring.errors import KeyringError, PasswordDeleteError

from backend.shared.config import system_config

logger = logging.getLogger(__name__)

_DEFAULT_SERVICE_NAME = "MOTO-Autonomous-ASI"
_OPENROUTER_KEY = "openrouter_api_key"
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


def load_wolfram_api_key() -> Optional[str]:
    """Load the persisted Wolfram Alpha API key."""
    return _get_secret(_WOLFRAM_KEY)


def store_wolfram_api_key(api_key: str) -> None:
    """Persist the Wolfram Alpha API key securely."""
    _set_secret(_WOLFRAM_KEY, api_key)


def clear_wolfram_api_key() -> None:
    """Delete the persisted Wolfram Alpha API key."""
    _delete_secret(_WOLFRAM_KEY)
