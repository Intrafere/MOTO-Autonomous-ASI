"""
Safe maintenance for ChromaDB's persistent on-disk cache.

Chroma collections are rebuildable indexes over MOTO's durable text/JSON
sources. Some Chroma versions can leave orphaned UUID segment directories after
collection deletion; this module detects that specific buildup and resets only
the Chroma cache directory.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import shutil
import sqlite3
from typing import Iterable

logger = logging.getLogger(__name__)

_UUID_DIR_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_MIN_ORPHAN_UUID_DIRS_FOR_RESET = 64
_MIN_ORPHAN_RATIO_FOR_RESET = 0.75
_REBUILD_MARKER_SUFFIX = ".rebuild.json"
_QUARANTINE_TOKEN = ".quarantine-"


@dataclass(frozen=True)
class ChromaCacheMaintenanceResult:
    """Outcome of a Chroma cache maintenance pass."""

    checked: bool
    reset_performed: bool
    reason: str = ""
    uuid_dir_count: int = 0
    unreferenced_uuid_dir_count: int = 0


def _safe_resolve_chroma_cache_dir(chroma_dir: str | Path, data_dir: str | Path) -> Path:
    """Resolve and validate the managed Chroma cache directory."""
    data_root = Path(data_dir).resolve()
    cache_dir = Path(chroma_dir).resolve()

    data_root_real = os.path.realpath(os.path.normpath(str(data_root)))
    cache_dir_real = os.path.realpath(os.path.normpath(str(cache_dir)))
    data_prefix = data_root_real if data_root_real.endswith(os.sep) else data_root_real + os.sep

    if cache_dir_real == data_root_real or not cache_dir_real.startswith(data_prefix):
        raise ValueError("Chroma cache directory must be a child of the configured data root")

    if cache_dir.parent == cache_dir:
        raise ValueError("Refusing to manage filesystem root as Chroma cache directory")

    return Path(cache_dir_real)


def _uuid_directories(cache_dir: Path) -> set[str]:
    if not cache_dir.exists():
        return set()
    return {child.name for child in cache_dir.iterdir() if child.is_dir() and _UUID_DIR_RE.match(child.name)}


def _referenced_chroma_ids(sqlite_path: Path) -> set[str]:
    """Read live Chroma collection/segment IDs from SQLite metadata."""
    referenced: set[str] = set()
    con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        cur = con.cursor()
        for table in ("collections", "segments"):
            try:
                rows: Iterable[tuple] = cur.execute(f"select * from {table}").fetchall()
            except sqlite3.Error:
                continue
            for row in rows:
                for value in row:
                    if isinstance(value, str) and _UUID_DIR_RE.match(value):
                        referenced.add(value)
    finally:
        con.close()
    return referenced


def _rebuild_marker(cache_dir: Path) -> Path:
    return cache_dir.parent / f"{cache_dir.name}{_REBUILD_MARKER_SUFFIX}"


def _validated_quarantine_path(cache_dir: Path, candidate: Path) -> Path:
    resolved = candidate.resolve()
    if resolved.parent != cache_dir.parent or not resolved.name.startswith(
        f"{cache_dir.name}{_QUARANTINE_TOKEN}"
    ):
        raise ValueError("Invalid Chroma quarantine path")
    return resolved


def recover_interrupted_chroma_rebuild(
    chroma_dir: str | Path,
    data_dir: str | Path,
) -> None:
    """Recover a prior atomic cache quarantine without touching durable sources."""
    cache_dir = _safe_resolve_chroma_cache_dir(chroma_dir, data_dir)
    marker = _rebuild_marker(cache_dir)
    if not marker.exists():
        return
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        quarantine = _validated_quarantine_path(cache_dir, Path(payload["quarantine"]))
    except Exception:
        logger.exception("Invalid Chroma rebuild marker at %s; leaving it for operator review", marker)
        return

    # A fresh cache directory means the atomic switch completed. The old cache
    # is derived data and can be removed. If no fresh directory exists, restore
    # the quarantined cache so startup never silently loses the last usable index.
    try:
        if cache_dir.exists():
            if quarantine.exists():
                shutil.rmtree(quarantine)
        elif quarantine.exists():
            quarantine.replace(cache_dir)
        marker.unlink(missing_ok=True)
    except OSError:
        logger.warning("Chroma rebuild recovery deferred because cache files are locked", exc_info=True)


def quarantine_chroma_cache(
    chroma_dir: str | Path,
    data_dir: str | Path,
) -> tuple[Path, Path | None]:
    """Atomically move the inactive cache aside and create a rebuild marker."""
    cache_dir = _safe_resolve_chroma_cache_dir(chroma_dir, data_dir)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    marker = _rebuild_marker(cache_dir)
    quarantine: Path | None = None
    if cache_dir.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        quarantine = _validated_quarantine_path(
            cache_dir,
            cache_dir.parent / f"{cache_dir.name}{_QUARANTINE_TOKEN}{stamp}",
        )
        marker.write_text(
            json.dumps({"cache": str(cache_dir), "quarantine": str(quarantine)}),
            encoding="utf-8",
        )
        cache_dir.replace(quarantine)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir, quarantine


def complete_chroma_cache_rebuild(
    chroma_dir: str | Path,
    data_dir: str | Path,
    quarantine: Path | None,
) -> None:
    """Commit a cache rebuild and remove the inactive quarantine best-effort."""
    cache_dir = _safe_resolve_chroma_cache_dir(chroma_dir, data_dir)
    if quarantine is not None:
        quarantine = _validated_quarantine_path(cache_dir, quarantine)
        try:
            if quarantine.exists():
                shutil.rmtree(quarantine)
        except OSError:
            logger.warning("Could not remove old Chroma quarantine %s", quarantine, exc_info=True)
            return
    _rebuild_marker(cache_dir).unlink(missing_ok=True)


def abort_chroma_cache_rebuild(
    chroma_dir: str | Path,
    data_dir: str | Path,
    quarantine: Path | None,
) -> None:
    """Restore the prior inactive cache when fresh native initialization fails."""
    cache_dir = _safe_resolve_chroma_cache_dir(chroma_dir, data_dir)
    if quarantine is not None:
        quarantine = _validated_quarantine_path(cache_dir, quarantine)
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        if quarantine.exists():
            quarantine.replace(cache_dir)
    _rebuild_marker(cache_dir).unlink(missing_ok=True)


def maintain_chroma_cache_directory(
    chroma_dir: str | Path,
    data_dir: str | Path,
    *,
    orphan_threshold: int = _MIN_ORPHAN_UUID_DIRS_FOR_RESET,
    orphan_ratio_threshold: float = _MIN_ORPHAN_RATIO_FOR_RESET,
) -> ChromaCacheMaintenanceResult:
    """
    Reset Chroma's cache directory only when a clear orphan buildup is detected.

    Durable user/session files live outside this directory. If metadata cannot be
    read, this function logs and skips cleanup rather than guessing.
    """
    cache_dir = _safe_resolve_chroma_cache_dir(chroma_dir, data_dir)
    recover_interrupted_chroma_rebuild(cache_dir, data_dir)
    if not cache_dir.exists():
        return ChromaCacheMaintenanceResult(checked=True, reset_performed=False, reason="missing")

    uuid_dirs = _uuid_directories(cache_dir)
    if len(uuid_dirs) < orphan_threshold:
        return ChromaCacheMaintenanceResult(
            checked=True,
            reset_performed=False,
            reason="below_threshold",
            uuid_dir_count=len(uuid_dirs),
        )

    sqlite_path = cache_dir / "chroma.sqlite3"
    if not sqlite_path.exists():
        logger.warning(
            "Chroma cache has %d UUID directories but no SQLite metadata; skipping automatic reset.",
            len(uuid_dirs),
        )
        return ChromaCacheMaintenanceResult(
            checked=True,
            reset_performed=False,
            reason="missing_sqlite_metadata",
            uuid_dir_count=len(uuid_dirs),
        )

    try:
        referenced_ids = _referenced_chroma_ids(sqlite_path)
    except sqlite3.Error as exc:
        logger.warning("Could not inspect Chroma SQLite metadata; skipping cache reset: %s", exc)
        return ChromaCacheMaintenanceResult(
            checked=True,
            reset_performed=False,
            reason="metadata_unreadable",
            uuid_dir_count=len(uuid_dirs),
        )

    unreferenced_dirs = uuid_dirs - referenced_ids
    orphan_ratio = len(unreferenced_dirs) / max(len(uuid_dirs), 1)
    should_reset = (
        len(unreferenced_dirs) >= orphan_threshold
        and orphan_ratio >= orphan_ratio_threshold
    )
    if not should_reset:
        return ChromaCacheMaintenanceResult(
            checked=True,
            reset_performed=False,
            reason="referenced_or_below_ratio",
            uuid_dir_count=len(uuid_dirs),
            unreferenced_uuid_dir_count=len(unreferenced_dirs),
        )

    logger.warning(
        "Resetting Chroma cache at %s after detecting %d unreferenced UUID directories "
        "out of %d total UUID directories.",
        cache_dir,
        len(unreferenced_dirs),
        len(uuid_dirs),
    )
    _, quarantine = quarantine_chroma_cache(cache_dir, data_dir)
    complete_chroma_cache_rebuild(cache_dir, data_dir, quarantine)
    return ChromaCacheMaintenanceResult(
        checked=True,
        reset_performed=True,
        reason="orphan_uuid_directories",
        uuid_dir_count=len(uuid_dirs),
        unreferenced_uuid_dir_count=len(unreferenced_dirs),
    )
