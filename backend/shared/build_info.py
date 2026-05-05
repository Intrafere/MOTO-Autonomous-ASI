"""
Build identity helpers for the shared update contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_MANIFEST_PATH = REPO_ROOT / "moto-update-manifest.json"
PACKAGE_JSON_PATH = REPO_ROOT / "package.json"

_DEFAULT_BUILD_INFO = {
    "manifest_version": 1,
    "version": "0.0.0-dev",
    "build_commit": "dev",
    "update_channel": "main",
    "api_contract_version": "build5-v1",
}

_ENV_OVERRIDES = {
    "MOTO_VERSION": "version",
    "MOTO_BUILD_COMMIT": "build_commit",
    "MOTO_UPDATE_CHANNEL": "update_channel",
    "MOTO_API_CONTRACT_VERSION": "api_contract_version",
}


@dataclass(frozen=True)
class BuildInfo:
    """Normalized build identity shared by runtime APIs and updater metadata."""

    version: str
    build_commit: str
    update_channel: str
    api_contract_version: str
    manifest_version: int = 1

    def as_features_payload(self, capability_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return the public `/api/features` payload with optional capability flags."""
        payload: dict[str, Any] = {
            "version": self.version,
            "build_commit": self.build_commit,
            "update_channel": self.update_channel,
            "api_contract_version": self.api_contract_version,
        }
        if capability_overrides:
            payload.update(capability_overrides)
        return payload


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        logger.warning("Ignoring invalid build metadata file at %s: %s", path, exc)
        return None
    except OSError as exc:
        logger.warning("Failed to read build metadata file at %s: %s", path, exc)
        return None


def _load_default_version() -> str:
    package_json = _load_json(PACKAGE_JSON_PATH)
    if isinstance(package_json, dict):
        version = str(package_json.get("version", "")).strip()
        if version:
            return version
    return str(_DEFAULT_BUILD_INFO["version"])


def _coerce_manifest_version(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(_DEFAULT_BUILD_INFO["manifest_version"])


@lru_cache(maxsize=1)
def get_build_info() -> BuildInfo:
    """Resolve build identity from the committed manifest with env overrides."""
    payload: dict[str, Any] = dict(_DEFAULT_BUILD_INFO)
    payload["version"] = _load_default_version()

    manifest = _load_json(BUILD_MANIFEST_PATH)
    if isinstance(manifest, dict):
        for field in ("version", "build_commit", "update_channel", "api_contract_version"):
            value = str(manifest.get(field, "")).strip()
            if value:
                payload[field] = value
        payload["manifest_version"] = _coerce_manifest_version(
            manifest.get("manifest_version", payload["manifest_version"])
        )
    else:
        logger.warning(
            "Build manifest not found at %s; falling back to package metadata defaults.",
            BUILD_MANIFEST_PATH,
        )

    for env_name, field_name in _ENV_OVERRIDES.items():
        override = os.environ.get(env_name, "").strip()
        if override:
            payload[field_name] = override

    return BuildInfo(
        version=str(payload["version"]),
        build_commit=str(payload["build_commit"]),
        update_channel=str(payload["update_channel"]),
        api_contract_version=str(payload["api_contract_version"]),
        manifest_version=_coerce_manifest_version(payload["manifest_version"]),
    )
