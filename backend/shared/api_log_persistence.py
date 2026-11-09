"""Shared persistence helpers for metadata-only API logs."""

import os
import tempfile
from pathlib import Path
from typing import Iterable


def ensure_private_log_file(path: Path) -> None:
    """Create a log file with owner-only POSIX permissions when supported."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        if os.name == "posix":
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
        else:
            path.touch()
    if os.name == "posix":
        path.chmod(0o600)


def atomic_write_log_lines(path: Path, lines: Iterable[str]) -> None:
    """Replace a JSONL log atomically without widening POSIX permissions."""
    ensure_private_log_file(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        if os.name == "posix":
            os.chmod(temporary_path, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            # fdopen owns the descriptor from here; closing it again below
            # could close an unrelated file another thread has since opened.
            descriptor = None
            handle.writelines(lines)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
        if os.name == "posix":
            path.chmod(0o600)
    except Exception:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        temporary_path.unlink(missing_ok=True)
        raise
