"""
Durable non-secret runtime settings.

This stores user-controlled process settings that are safe to persist under the
active data root. Provider keys and other secrets remain in secret_store.py or
runtime memory according to deployment mode.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from backend.shared.config import system_config
from backend.shared.free_model_manager import free_model_manager
from backend.shared.log_redaction import redact_log_text

logger = logging.getLogger(__name__)

RUNTIME_SETTINGS_FILENAME = "runtime_settings.json"


class RuntimeSettingsError(RuntimeError):
    """Raised when non-secret runtime settings cannot be persisted."""

_PROOF_BOOL_FIELDS = {
    "lean4_lsp_enabled",
    "smt_enabled",
}

_PROOF_INT_FIELDS = {
    "lean4_proof_timeout": (10, 3600),
    "lean4_lsp_idle_timeout": (60, 7200),
    "proof_max_parallel_candidates": (0, 1000),
    "smt_timeout": (1, 600),
}

# Every save stores all proof fields, including defaults the user never changed,
# so a stored value equal to a default shipped under that file's settings version
# follows the current default; any other stored value is applied as saved. When a
# default changes, bump the version and record the outgoing default under the
# previous version. Version 1 covers files saved before versioning existed.
_PROOF_SETTINGS_VERSION = 2
_SHIPPED_PROOF_DEFAULTS_BY_VERSION: Dict[int, Dict[str, frozenset[str]]] = {
    1: {
        "lean4_proof_timeout": frozenset({"600", "900"}),
        "smt_timeout": frozenset({"30"}),
    },
}

_CONNECTIVITY_BOOL_FIELDS = {
    "syntheticlib4_enabled",
    "agent_conversation_memory_enabled",
    "wolfram_alpha_enabled",
}


def _settings_path() -> Path:
    return Path(system_config.data_dir) / RUNTIME_SETTINGS_FILENAME


def _read_settings() -> Dict[str, Any]:
    path = _settings_path()
    try:
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning(
            "Ignoring corrupt runtime settings file %s: %s",
            redact_log_text(path, 240),
            redact_log_text(exc, 240),
        )
        return {}
    except OSError as exc:
        logger.warning(
            "Failed to read runtime settings file %s: %s",
            redact_log_text(path, 240),
            redact_log_text(exc, 240),
        )
        return {}

    return payload if isinstance(payload, dict) else {}


def _write_settings(payload: Dict[str, Any]) -> None:
    path = _settings_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.tmp")
        temp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(path)
    except OSError as exc:
        logger.warning(
            "Failed to persist runtime settings file %s: %s",
            redact_log_text(path, 240),
            redact_log_text(exc, 240),
        )
        raise RuntimeSettingsError("Failed to persist runtime settings") from exc


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _coerce_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _proof_settings_from_config() -> Dict[str, Any]:
    return {
        "settings_version": _PROOF_SETTINGS_VERSION,
        "lean4_user_enabled": bool(system_config.lean4_user_enabled),
        "lean4_lsp_enabled": bool(system_config.lean4_lsp_enabled),
        "lean4_proof_timeout": int(system_config.lean4_proof_timeout),
        "lean4_lsp_idle_timeout": int(system_config.lean4_lsp_idle_timeout),
        "proof_max_parallel_candidates": int(system_config.proof_max_parallel_candidates),
        "smt_enabled": bool(system_config.smt_enabled),
        "smt_timeout": int(system_config.smt_timeout),
    }


def reconcile_lean4_runtime_state() -> None:
    """Apply immutable process capability to the durable Lean user preference."""
    if system_config.lean4_runtime_available is None:
        system_config.lean4_runtime_available = bool(system_config.lean4_enabled)
    if system_config.lean4_user_enabled is None:
        system_config.lean4_user_enabled = bool(system_config.lean4_enabled)
    system_config.lean4_enabled = bool(
        system_config.lean4_runtime_available
        and system_config.lean4_user_enabled
        and not system_config.generic_mode
    )


def _free_model_settings_from_manager() -> Dict[str, Any]:
    status = free_model_manager.get_status()
    return {
        "looping_enabled": bool(status.get("looping_enabled", False)),
        "auto_selector_enabled": bool(status.get("auto_selector_enabled", False)),
    }


def _connectivity_toggles_from_config() -> Dict[str, Any]:
    return {
        "syntheticlib4_enabled": bool(system_config.syntheticlib4_enabled),
        "agent_conversation_memory_enabled": bool(system_config.agent_conversation_memory_enabled),
        "wolfram_alpha_enabled": bool(system_config.wolfram_alpha_enabled),
    }


def save_proof_runtime_settings() -> None:
    """Persist current non-secret Lean/SMT proof runtime settings."""
    payload = _read_settings()
    payload["proof_settings"] = _proof_settings_from_config()
    _write_settings(payload)


def save_free_model_runtime_settings() -> None:
    """Persist current non-secret free-model rotation settings."""
    payload = _read_settings()
    payload["free_model_settings"] = _free_model_settings_from_manager()
    _write_settings(payload)


def save_connectivity_runtime_settings() -> None:
    """Persist current non-secret connectivity feature toggles."""
    payload = _read_settings()
    payload["connectivity_toggles"] = _connectivity_toggles_from_config()
    _write_settings(payload)


_SUBSCRIPTION_REGIONS_KEY = "subscription_endpoint_regions"


def get_subscription_endpoint_region(provider_id: str) -> str:
    """Return the persisted endpoint region id for a subscription provider, or ``""``."""
    regions = _read_settings().get(_SUBSCRIPTION_REGIONS_KEY)
    if not isinstance(regions, dict):
        return ""
    value = regions.get(provider_id)
    return value.strip() if isinstance(value, str) else ""


def set_subscription_endpoint_region(provider_id: str, region_id: str) -> None:
    """Persist (or with an empty region, remove) a subscription provider's endpoint region."""
    payload = _read_settings()
    regions = payload.get(_SUBSCRIPTION_REGIONS_KEY)
    regions = dict(regions) if isinstance(regions, dict) else {}
    if region_id:
        regions[provider_id] = region_id
    else:
        regions.pop(provider_id, None)
    if regions:
        payload[_SUBSCRIPTION_REGIONS_KEY] = regions
    else:
        payload.pop(_SUBSCRIPTION_REGIONS_KEY, None)
    _write_settings(payload)


def get_persisted_connectivity_toggles() -> Dict[str, bool]:
    """Return persisted connectivity toggles, omitting fields not yet saved."""
    toggles = _read_settings().get("connectivity_toggles")
    if not isinstance(toggles, dict):
        return {}
    result: Dict[str, bool] = {}
    for field in _CONNECTIVITY_BOOL_FIELDS:
        if field in toggles:
            result[field] = _coerce_bool(toggles[field], bool(getattr(system_config, field)))
    return result


def apply_persisted_runtime_settings() -> None:
    """Apply persisted non-secret runtime settings to process globals."""
    payload = _read_settings()

    if not system_config.generic_mode:
        proof_settings = payload.get("proof_settings")
        if isinstance(proof_settings, dict):
            preference_value = proof_settings.get(
                "lean4_user_enabled",
                proof_settings.get("lean4_enabled"),
            )
            if preference_value is not None:
                system_config.lean4_user_enabled = _coerce_bool(
                    preference_value,
                    bool(system_config.lean4_enabled),
                )
            for field in _PROOF_BOOL_FIELDS:
                if field in proof_settings:
                    setattr(
                        system_config,
                        field,
                        _coerce_bool(proof_settings[field], bool(getattr(system_config, field))),
                    )
            stored_version = _coerce_int(
                proof_settings.get("settings_version"),
                1,
                1,
                _PROOF_SETTINGS_VERSION,
            )
            shipped_defaults = _SHIPPED_PROOF_DEFAULTS_BY_VERSION.get(stored_version, {})
            for field, (minimum, maximum) in _PROOF_INT_FIELDS.items():
                if field in proof_settings:
                    default = int(getattr(system_config, field))
                    if str(proof_settings[field]).strip() in shipped_defaults.get(field, ()):
                        proof_settings[field] = default
                    setattr(
                        system_config,
                        field,
                        _coerce_int(
                            proof_settings[field],
                            default,
                            minimum,
                            maximum,
                        ),
                    )

    reconcile_lean4_runtime_state()

    free_model_settings = payload.get("free_model_settings")
    if isinstance(free_model_settings, dict):
        looping_enabled = _coerce_bool(
            free_model_settings.get("looping_enabled"),
            free_model_manager.looping_enabled,
        )
        auto_selector_enabled = _coerce_bool(
            free_model_settings.get("auto_selector_enabled"),
            free_model_manager.auto_selector_enabled,
        )
        free_model_manager.configure(
            looping=looping_enabled,
            auto_selector=auto_selector_enabled,
        )

    connectivity_toggles = payload.get("connectivity_toggles")
    if isinstance(connectivity_toggles, dict):
        for field in _CONNECTIVITY_BOOL_FIELDS:
            if field in connectivity_toggles:
                setattr(
                    system_config,
                    field,
                    _coerce_bool(connectivity_toggles[field], bool(getattr(system_config, field))),
                )
