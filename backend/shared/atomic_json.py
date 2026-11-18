"""Crash-safe JSON file replacement for small durable state files."""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

# Windows refuses to replace a file that another handle holds open without
# delete sharing (a MOTO reader, antivirus, or the search indexer). Those holds
# last milliseconds, so the rename is retried briefly before the error stands.
_REPLACE_RETRIES_ON_PERMISSION_ERROR = os.name == "nt"
_REPLACE_RETRY_DELAYS_SECONDS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.4)


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


def replace_file(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
    """``os.replace`` that waits out a brief Windows sharing lock on either path.

    Only ``PermissionError`` on Windows is retried, for under a second in total;
    any other error, or a lock that outlasts the retries, is raised unchanged.
    """
    if _REPLACE_RETRIES_ON_PERMISSION_ERROR:
        for delay in _REPLACE_RETRY_DELAYS_SECONDS:
            try:
                os.replace(source, destination)
                return
            except PermissionError:
                logger.debug(
                    "%s is briefly locked by another handle; retrying the replace in %.2f s",
                    Path(destination).name,
                    delay,
                )
                time.sleep(delay)
    os.replace(source, destination)


def atomic_write_text(path: Path, text: str) -> None:
    """Replace a UTF-8 text file through a unique sibling temp file.

    Used for durable working documents (papers, outlines, accepted-submission
    databases, session metadata) so a crash mid-write never leaves a truncated
    or empty file behind: readers see the previous complete file or the new one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        replace_file(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    _fsync_directory(path.parent)


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
        replace_file(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    _fsync_directory(path.parent)
