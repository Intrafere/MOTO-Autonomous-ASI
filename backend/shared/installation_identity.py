"""Stable desktop installation identity and data-root binding.

The installation identity is intentionally independent of runtime instance IDs,
paths, and configurable credential namespaces.  A non-secret marker travels
with a data root, while the authoritative identity and root-binding registry
remain in the OS keyring.  Copying a marked root to another path therefore
fails closed until an explicit relocation approval is performed.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from backend.shared.atomic_json import atomic_write_json
from backend.shared.config import system_config
from backend.shared.secret_store import (
    clear_moto_installation_identity_state,
    load_moto_installation_identity_state,
    store_moto_installation_identity_state,
)

INSTALLATION_IDENTITY_VERSION = "moto-installation-identity-v1"
ROOT_BINDING_VERSION = "moto-data-root-binding-v1"
ROOT_BINDING_MARKER = ".moto_installation_binding.json"


class InstallationIdentityError(RuntimeError):
    """Raised when installation/root identity cannot be established safely."""


class InstallationBindingMismatchError(InstallationIdentityError):
    """A copied or moved root requires explicit operator approval."""


@dataclass(frozen=True)
class InstallationBinding:
    installation_id: str
    root_binding_id: str
    identity_version: str = INSTALLATION_IDENTITY_VERSION
    binding_version: str = ROOT_BINDING_VERSION


def _new_opaque_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _normalized_root(data_root: Path) -> str:
    return os.path.normcase(str(data_root.resolve(strict=False)))


def _root_fingerprint(data_root: Path) -> str:
    return hashlib.sha256(_normalized_root(data_root).encode("utf-8")).hexdigest()


def _marker_path(data_root: Path) -> Path:
    return data_root / ROOT_BINDING_MARKER


def _load_marker(data_root: Path) -> Optional[Dict[str, str]]:
    path = _marker_path(data_root)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise InstallationIdentityError("Installation binding marker is invalid.") from exc
    if not isinstance(payload, dict):
        raise InstallationIdentityError("Installation binding marker must be an object.")
    required = ("binding_version", "installation_id", "root_binding_id")
    if any(not str(payload.get(field) or "").strip() for field in required):
        raise InstallationIdentityError("Installation binding marker is incomplete.")
    if payload["binding_version"] != ROOT_BINDING_VERSION:
        raise InstallationIdentityError("Installation binding marker version is unsupported.")
    return {field: str(payload[field]).strip() for field in required}


def _load_state() -> Optional[Dict[str, Any]]:
    raw = load_moto_installation_identity_state()
    if not raw:
        return None
    try:
        state = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InstallationIdentityError("Stored installation identity is invalid.") from exc
    if not isinstance(state, dict) or state.get("identity_version") != INSTALLATION_IDENTITY_VERSION:
        raise InstallationIdentityError("Stored installation identity version is unsupported.")
    if not str(state.get("installation_id") or "").strip():
        raise InstallationIdentityError("Stored installation identity is incomplete.")
    bindings = state.get("root_bindings")
    if not isinstance(bindings, dict):
        raise InstallationIdentityError("Stored installation root bindings are invalid.")
    return state


def _store_state(state: Dict[str, Any]) -> None:
    store_moto_installation_identity_state(
        json.dumps(state, sort_keys=True, separators=(",", ":"))
    )


def ensure_installation_binding(data_root: Optional[Path] = None) -> InstallationBinding:
    """Load or create the stable desktop identity for one active data root.

    Unknown/mismatched marked roots fail closed. A new unmarked empty root can
    join the installation, receiving its own binding ID. A populated unmarked
    root requires an explicit adoption decision so copied proof stores cannot
    silently inherit authorization.
    """
    if system_config.generic_mode:
        raise InstallationIdentityError(
            "Stable proof-trade installation identity is unavailable in generic mode."
        )
    root = Path(data_root or system_config.data_dir).resolve(strict=False)
    marker = _load_marker(root)
    state = _load_state()

    if state is None:
        if marker is not None:
            raise InstallationBindingMismatchError(
                "This data root is bound to an unavailable installation identity."
            )
        state = {
            "identity_version": INSTALLATION_IDENTITY_VERSION,
            "installation_id": _new_opaque_id("moto_installation"),
            "root_bindings": {},
        }

    installation_id = str(state["installation_id"])
    bindings: Dict[str, Any] = state["root_bindings"]
    fingerprint = _root_fingerprint(root)

    if marker is None:
        existing_entries = [
            path
            for path in root.iterdir()
            if path.name != ROOT_BINDING_MARKER
        ] if root.exists() else []
        if state["root_bindings"] and existing_entries:
            raise InstallationBindingMismatchError(
                "This populated data root is unbound; explicitly adopt or reset it before use."
            )
        root_binding_id = _new_opaque_id("moto_root")
        bindings[root_binding_id] = {"root_fingerprint": fingerprint}
        _store_state(state)
        atomic_write_json(
            _marker_path(root),
            {
                "binding_version": ROOT_BINDING_VERSION,
                "installation_id": installation_id,
                "root_binding_id": root_binding_id,
            },
        )
    else:
        return _verified_binding(marker, state, root)

    return InstallationBinding(
        installation_id=installation_id,
        root_binding_id=root_binding_id,
    )


def _verified_binding(
    marker: Dict[str, str],
    state: Dict[str, Any],
    root: Path,
) -> InstallationBinding:
    installation_id = str(state["installation_id"])
    if marker["installation_id"] != installation_id:
        raise InstallationBindingMismatchError(
            "This data root belongs to a different MOTO installation."
        )
    root_binding_id = marker["root_binding_id"]
    registered = state["root_bindings"].get(root_binding_id)
    if not isinstance(registered, dict):
        raise InstallationBindingMismatchError(
            "This data root binding is not registered to this installation."
        )
    if registered.get("root_fingerprint") != _root_fingerprint(root):
        raise InstallationBindingMismatchError(
            "This data root was moved or copied; approve relocation before use."
        )
    return InstallationBinding(
        installation_id=installation_id,
        root_binding_id=root_binding_id,
    )


def read_installation_binding(data_root: Optional[Path] = None) -> Optional[InstallationBinding]:
    """Return the verified existing binding without creating any identity.

    Returns ``None`` when the root has no marker or no stored identity exists;
    copied, moved, or foreign roots raise exactly as ``ensure_installation_binding``.
    """
    if system_config.generic_mode:
        raise InstallationIdentityError(
            "Stable proof-trade installation identity is unavailable in generic mode."
        )
    root = Path(data_root or system_config.data_dir).resolve(strict=False)
    marker = _load_marker(root)
    state = _load_state()
    if marker is None or state is None:
        return None
    return _verified_binding(marker, state, root)


def adopt_unmarked_runtime_root(data_root: Optional[Path] = None) -> InstallationBinding:
    """Explicitly bind a populated unmarked root to the current installation."""
    if system_config.generic_mode:
        raise InstallationIdentityError("Root adoption is unavailable in generic mode.")
    root = Path(data_root or system_config.data_dir).resolve(strict=False)
    if _load_marker(root) is not None:
        return ensure_installation_binding(root)
    state = _load_state()
    if state is None:
        return ensure_installation_binding(root)
    binding_id = _new_opaque_id("moto_root")
    state["root_bindings"][binding_id] = {
        "root_fingerprint": _root_fingerprint(root)
    }
    _store_state(state)
    atomic_write_json(
        _marker_path(root),
        {
            "binding_version": ROOT_BINDING_VERSION,
            "installation_id": state["installation_id"],
            "root_binding_id": binding_id,
        },
    )
    return InstallationBinding(
        installation_id=str(state["installation_id"]),
        root_binding_id=binding_id,
    )


def approve_runtime_root_relocation(data_root: Optional[Path] = None) -> InstallationBinding:
    """Explicitly approve a marked root's relocation for this installation."""
    if system_config.generic_mode:
        raise InstallationIdentityError(
            "Installation relocation is unavailable in generic mode."
        )
    root = Path(data_root or system_config.data_dir).resolve(strict=False)
    marker = _load_marker(root)
    state = _load_state()
    if marker is None or state is None:
        raise InstallationIdentityError("No existing installation binding can be relocated.")
    if marker["installation_id"] != state["installation_id"]:
        raise InstallationBindingMismatchError(
            "The root marker belongs to a different installation."
        )
    binding_id = marker["root_binding_id"]
    if binding_id not in state["root_bindings"]:
        raise InstallationBindingMismatchError(
            "The root binding is not registered to this installation."
        )
    state["root_bindings"][binding_id] = {
        "root_fingerprint": _root_fingerprint(root)
    }
    _store_state(state)
    return InstallationBinding(
        installation_id=str(state["installation_id"]),
        root_binding_id=binding_id,
    )


def reset_installation_identity(data_root: Optional[Path] = None) -> InstallationBinding:
    """Explicitly reset local installation identity without deleting proofs.

    Proofs carrying the previous locator remain historical and unmodified.
    Reset is refused while the proof-trade ledger holds linked or queued state,
    or cannot be read safely; archive the local ledger (or unlink) first.
    """
    if system_config.generic_mode:
        raise InstallationIdentityError("Installation reset is unavailable in generic mode.")
    root = Path(data_root or system_config.data_dir).resolve(strict=False)
    from backend.shared.proof_trade_ledger import ledger_blocks_installation_reset

    if ledger_blocks_installation_reset(root):
        raise InstallationIdentityError(
            "Installation reset is blocked while proof-trade ledger state is linked, "
            "queued, or unreadable."
        )
    clear_moto_installation_identity_state()
    marker = _marker_path(root)
    if marker.exists():
        marker.unlink()
    return ensure_installation_binding(root)
