"""Crash-safe JSON file replacement for small durable state files."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


def _fsync_directory(directory: Path) -> None:
    """Best-effort durability for the rename itself (unsupported on Windows)."""
    if os.name == "nt":
        return
    try:
        descriptor = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write sorted JSON to a unique sibling temp file, fsync, then replace.

    Readers observe either the previous complete file or the new complete file;
    a failure before the replace leaves the previous file untouched and removes
    the temporary sibling.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    _fsync_directory(path.parent)
