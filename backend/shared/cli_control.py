"""Desktop command-line control surface: instance discovery file and start templates.

The ``moto`` command (``moto_cli.py``) controls a running desktop MOTO backend
over its normal REST/WebSocket routes. Two small artifacts under the active data
root make that possible without the browser:

* ``.moto_control.json`` - written once the backend owns its data-root lease and
  removed at shutdown. It names the loopback backend URL, the per-launch desktop
  API token, and the owning process, the way a Jupyter runtime file does. The
  temp-file writer creates it owner-only on POSIX; on Windows it inherits the
  data root's user-profile ACL.
* ``cli_start_templates/<mode>.json`` - the most recent successful start request
  for each workflow mode, so ``moto <mode> start --prompt ...`` can reuse the role
  configuration chosen in the app. Start requests carry no provider credentials;
  any credential-shaped key is still dropped defensively.

Both are desktop-only. Hosted/generic mode authenticates through the proxy and
never writes either file. Failures are logged and never affect a workflow.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from backend.shared.atomic_json import atomic_write_json
from backend.shared.config import system_config

logger = logging.getLogger(__name__)

CONTROL_FILE_NAME = ".moto_control.json"
CONTROL_FILE_SCHEMA = "moto-cli-control-v1"
START_TEMPLATE_DIR_NAME = "cli_start_templates"
START_TEMPLATE_SCHEMA = "moto-cli-start-template-v1"
# "think" holds direct-answer Autonomous Research starts (`moto think`) apart from
# the settings a plain Autonomous Research start reuses.
START_TEMPLATE_MODES = ("autonomous", "aggregator", "compiler", "leanoj", "think")

_WILDCARD_HOSTS = {"", "0.0.0.0", "::", "[::]"}
_CREDENTIAL_KEY_MARKERS = ("api_key", "secret", "password", "access_token", "refresh_token")


def _data_root() -> Path:
    return Path(system_config.data_dir)


def control_file_path(data_root: Optional[Path] = None) -> Path:
    return Path(data_root or _data_root()) / CONTROL_FILE_NAME


def _loopback_host(host: str) -> str:
    normalized = (host or "").strip()
    if normalized in _WILDCARD_HOSTS:
        return "127.0.0.1"
    if ":" in normalized and not normalized.startswith("["):
        return f"[{normalized}]"
    return normalized


def write_control_file(*, version: str) -> None:
    """Publish how the local ``moto`` command reaches this backend."""
    if system_config.generic_mode:
        return
    token = (system_config.desktop_api_token or "").strip()
    if not token:
        return
    host = _loopback_host(system_config.backend_host)
    payload = {
        "schema": CONTROL_FILE_SCHEMA,
        "instance_id": system_config.instance_id,
        "pid": os.getpid(),
        "backend_url": f"http://{host}:{int(system_config.backend_port)}",
        "frontend_url": f"http://localhost:{int(system_config.frontend_port)}",
        "desktop_api_token": token,
        "data_root": str(Path(system_config.data_dir).resolve()),
        "version": version,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        atomic_write_json(control_file_path(), payload)
    except OSError as exc:
        logger.warning("Could not write the MOTO command-line control file: %s", type(exc).__name__)


def remove_control_file() -> None:
    """Remove the control file when this process still owns it."""
    path = control_file_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(payload, dict) or payload.get("pid") != os.getpid():
        return
    try:
        path.unlink()
    except OSError:
        logger.debug("Could not remove the MOTO command-line control file", exc_info=True)


def _strip_credentials(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_credentials(item)
            for key, item in value.items()
            if not any(marker in str(key).lower() for marker in _CREDENTIAL_KEY_MARKERS)
        }
    if isinstance(value, list):
        return [_strip_credentials(item) for item in value]
    return value


def remember_start_request(mode: str, request: Any) -> None:
    """Save the latest successful start request for ``moto <mode> start`` reuse."""
    if system_config.generic_mode or mode not in START_TEMPLATE_MODES:
        return
    try:
        body = request.model_dump(mode="json") if hasattr(request, "model_dump") else dict(request)
        payload: Mapping[str, Any] = {
            "schema": START_TEMPLATE_SCHEMA,
            "mode": mode,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "request": _strip_credentials(body),
        }
        atomic_write_json(_data_root() / START_TEMPLATE_DIR_NAME / f"{mode}.json", payload)
    except Exception as exc:  # A template is a convenience; never fail a start.
        logger.warning("Could not save the %s command-line start template: %s", mode, type(exc).__name__)
