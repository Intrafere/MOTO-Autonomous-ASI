"""Best-effort snapshot of free system memory for diagnostic messages.

Windows refuses allocations once its commit limit (RAM plus page file) is
used up, even while Task Manager still shows free RAM. Lean loading Mathlib
needs several GB, so its failures name this figure. Standard library only.
"""
from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

_GIB = 1024 ** 3


@dataclass(frozen=True)
class MemorySnapshot:
    """Free and total memory of the kind that limits new allocations."""

    available_bytes: int
    limit_bytes: int
    kind: Literal["commit", "physical"]


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _windows_commit_snapshot() -> Optional[MemorySnapshot]:
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    # "PageFile" fields are the system commit limit and the commit still free.
    return MemorySnapshot(
        available_bytes=int(status.ullAvailPageFile),
        limit_bytes=int(status.ullTotalPageFile),
        kind="commit",
    )


def _proc_meminfo_snapshot(meminfo_path: Path) -> Optional[MemorySnapshot]:
    values: dict[str, int] = {}
    for line in meminfo_path.read_text(encoding="utf-8").splitlines():
        name, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            values[name.strip()] = int(parts[0]) * 1024
    if "MemAvailable" not in values or "MemTotal" not in values:
        return None
    return MemorySnapshot(
        available_bytes=values["MemAvailable"],
        limit_bytes=values["MemTotal"],
        kind="physical",
    )


def _is_windows() -> bool:
    return os.name == "nt"


def system_memory_snapshot(*, meminfo_path: Path = Path("/proc/meminfo")) -> Optional[MemorySnapshot]:
    """Return free memory now, or None when the platform does not report it."""
    try:
        if _is_windows():
            return _windows_commit_snapshot()
        if meminfo_path.is_file():
            return _proc_meminfo_snapshot(meminfo_path)
    except (OSError, ValueError, AttributeError):
        return None
    return None


def describe_memory_snapshot(snapshot: Optional[MemorySnapshot]) -> str:
    """One sentence stating how much memory was free, or "" when unknown."""
    if snapshot is None or snapshot.limit_bytes <= 0:
        return ""
    available = snapshot.available_bytes / _GIB
    limit = snapshot.limit_bytes / _GIB
    if snapshot.kind == "commit":
        return (
            f"At the time, Windows had {available:.1f} GB of its {limit:.1f} GB memory "
            "commit limit free (RAM plus page file; it can run out while Task Manager still "
            "shows free RAM)."
        )
    return f"At the time, this computer had {available:.1f} GB of {limit:.1f} GB memory available."
