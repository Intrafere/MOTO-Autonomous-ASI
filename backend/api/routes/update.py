"""
Self-update routes — allows the frontend to trigger an in-place update
(git pull for git clones, ZIP overlay for downloaded installs) and poll
progress in real time.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

from fastapi import APIRouter, HTTPException

from backend.shared.config import system_config

router = APIRouter(tags=["update"])
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]

_pull_state: Dict[str, Any] = {
    "status": "idle",
    "output_lines": [],
    "returncode": None,
    "install_kind": None,
}


def _parse_semver(version_str: str) -> Tuple[int, ...]:
    """Extract numeric version tuple from a semver string (e.g. '1.0.8' -> (1,0,8))."""
    parts = re.findall(r"\d+", version_str or "")
    return tuple(int(p) for p in parts) if parts else (0,)


def _is_downgrade(local_version: str, remote_version: str) -> bool:
    """Return True if the remote version is strictly older than local."""
    return _parse_semver(remote_version) < _parse_semver(local_version)


def _detect_install_kind() -> str:
    """Classify install as 'git' or 'zip' based on .git presence."""
    if (_REPO_ROOT / ".git").exists():
        return "git"
    return "zip"


async def _run_git_command(*args: str) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        "git",
        *args,
        cwd=str(_REPO_ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await proc.communicate()
    return proc.returncode, stdout.decode("utf-8", errors="replace").strip()


async def _run_git_pull() -> None:
    """Fast-forward a clean git checkout to the configured update channel."""
    import sys
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from moto_updater import load_local_manifest, fetch_remote_manifest, fetch_branch_head_fallback

    global _pull_state
    _pull_state = {"status": "running", "output_lines": [], "returncode": None, "install_kind": "git"}

    try:
        local_manifest = load_local_manifest()
        channel = local_manifest.update_channel or "main"

        try:
            remote_manifest = fetch_remote_manifest(local_manifest)
        except Exception:
            remote_manifest = fetch_branch_head_fallback(local_manifest)

        if remote_manifest and _is_downgrade(local_manifest.version, remote_manifest.version):
            _pull_state["output_lines"].append(
                f"Refused: remote {remote_manifest.version} is older than local {local_manifest.version}. "
                f"Downgrades are not supported via the updater."
            )
            _pull_state["returncode"] = 1
            _pull_state["status"] = "error"
            return

        if remote_manifest and remote_manifest.build_commit == local_manifest.build_commit:
            _pull_state["output_lines"].append("Already up to date.")
            _pull_state["returncode"] = 0
            _pull_state["status"] = "done"
            return

        status_code, status_output = await _run_git_command("status", "--porcelain", "--untracked-files=no")
        if status_code != 0:
            _pull_state["output_lines"].append(status_output or "Failed to inspect git status.")
            _pull_state["returncode"] = status_code
            _pull_state["status"] = "error"
            return
        if status_output.strip():
            _pull_state["output_lines"].append(
                "Refused: tracked files have local modifications. Clean the checkout before updating."
            )
            _pull_state["returncode"] = 1
            _pull_state["status"] = "error"
            return

        fetch_code, fetch_output = await _run_git_command("fetch", "origin", channel, "--quiet")
        if fetch_code != 0:
            _pull_state["output_lines"].append(fetch_output or f"Failed to fetch origin/{channel}.")
            _pull_state["returncode"] = fetch_code
            _pull_state["status"] = "error"
            return

        divergence_code, divergence_output = await _run_git_command(
            "rev-list",
            "--left-right",
            "--count",
            f"HEAD...origin/{channel}",
        )
        if divergence_code != 0:
            _pull_state["output_lines"].append(divergence_output or f"Failed to compare HEAD with origin/{channel}.")
            _pull_state["returncode"] = divergence_code
            _pull_state["status"] = "error"
            return

        try:
            ahead_str, behind_str = divergence_output.split()
            ahead = int(ahead_str)
            behind = int(behind_str)
        except (ValueError, TypeError):
            _pull_state["output_lines"].append("Failed to parse git divergence counts.")
            _pull_state["returncode"] = 1
            _pull_state["status"] = "error"
            return

        if ahead:
            _pull_state["output_lines"].append(
                f"Refused: this checkout is ahead of origin/{channel} and cannot be updated automatically."
            )
            _pull_state["returncode"] = 1
            _pull_state["status"] = "error"
            return
        if not behind:
            _pull_state["output_lines"].append("Already up to date.")
            _pull_state["returncode"] = 0
            _pull_state["status"] = "done"
            return

        merge_code, merge_output = await _run_git_command("merge", "--ff-only", f"origin/{channel}")
        if merge_output:
            _pull_state["output_lines"].extend(merge_output.splitlines())
        _pull_state["returncode"] = merge_code
        _pull_state["status"] = "done" if merge_code == 0 else "error"
    except Exception as exc:
        logger.exception("git pull failed with exception")
        _pull_state["output_lines"].append(f"Exception: {exc}")
        _pull_state["returncode"] = -1
        _pull_state["status"] = "error"


def _run_zip_update_sync(state_lines: list) -> None:
    """Blocking ZIP update logic — meant to be run via asyncio.to_thread."""
    import sys
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from moto_updater import (
        cleanup_launcher_state,
        collect_preserved_relatives,
        sync_snapshot_into_install,
        restore_snapshot_from_backup,
        load_local_manifest,
        fetch_remote_manifest,
        fetch_branch_head_fallback,
        archive_url_for_manifest,
        _download_archive,
        _extract_archive,
        cleanup_path,
        _write_installed_manifest,
    )

    state_lines.append("Detecting update target...")

    local_manifest = load_local_manifest()
    try:
        remote_manifest = fetch_remote_manifest(local_manifest)
    except Exception:
        state_lines.append("Manifest not found, falling back to branch HEAD...")
        remote_manifest = fetch_branch_head_fallback(local_manifest)

    if remote_manifest is None:
        raise RuntimeError("Could not determine remote update target.")

    if remote_manifest.build_commit == local_manifest.build_commit:
        state_lines.append("Already up to date.")
        return

    if _is_downgrade(local_manifest.version, remote_manifest.version):
        raise RuntimeError(
            f"Refused: remote {remote_manifest.version} is older than local "
            f"{local_manifest.version}. Downgrades are not supported via the updater."
        )

    archive_url = archive_url_for_manifest(remote_manifest)
    state_lines.append(f"Downloading update from {archive_url}...")

    work_root = Path(tempfile.mkdtemp(prefix="moto-update-"))
    extract_root = work_root / "extract"
    archive_path = work_root / "update.zip"
    backup_root = Path(tempfile.mkdtemp(prefix="moto-update-backup-"))

    journal = None
    try:
        _download_archive(remote_manifest, archive_path)
        state_lines.append("Download complete. Extracting...")

        extracted_source = _extract_archive(archive_path, extract_root)

        state_lines.append("Applying update (preserving data/config)...")

        active_instances = cleanup_launcher_state()
        preserved_relatives = collect_preserved_relatives(os.environ, active_instances)
        journal = sync_snapshot_into_install(extracted_source, _REPO_ROOT, preserved_relatives, backup_root)
        _write_installed_manifest(remote_manifest)

        state_lines.append(
            f"Update applied: {local_manifest.version} ({local_manifest.short_commit}) "
            f"-> {remote_manifest.version} ({remote_manifest.short_commit})"
        )
        state_lines.append("Restart the application to complete the update.")

    except Exception:
        state_lines.append("Update failed mid-apply, restoring previous state...")
        try:
            if journal is not None:
                restore_snapshot_from_backup(_REPO_ROOT, backup_root, journal)
                state_lines.append("Rollback complete — previous install restored.")
            else:
                state_lines.append("Failure occurred before file overlay — no rollback needed.")
        except Exception as rb_exc:
            state_lines.append(f"Rollback also failed: {rb_exc}")
        raise
    finally:
        cleanup_path(work_root)
        cleanup_path(backup_root)


async def _run_zip_update() -> None:
    """Download and overlay a ZIP update for non-git installs (runs blocking I/O in a thread)."""
    global _pull_state
    _pull_state = {"status": "running", "output_lines": [], "returncode": None, "install_kind": "zip"}

    try:
        await asyncio.to_thread(_run_zip_update_sync, _pull_state["output_lines"])
        _pull_state["returncode"] = 0
        _pull_state["status"] = "done"
    except Exception as exc:
        logger.exception("ZIP update failed with exception")
        _pull_state["output_lines"].append(f"Exception: {exc}")
        _pull_state["returncode"] = -1
        _pull_state["status"] = "error"


@router.post("/api/update/pull")
async def start_pull() -> Dict[str, Any]:
    """Kick off an update. Routes to git pull or ZIP overlay depending on install type."""
    if system_config.generic_mode:
        raise HTTPException(
            status_code=501,
            detail="Self-update is unavailable in hosted generic mode.",
        )

    if _pull_state["status"] == "running":
        return {"started": False, "reason": "An update is already in progress."}

    install_kind = _detect_install_kind()

    if install_kind == "git":
        asyncio.create_task(_run_git_pull())
    else:
        asyncio.create_task(_run_zip_update())

    return {"started": True, "install_kind": install_kind}


@router.get("/api/update/pull-status")
async def get_pull_status() -> Dict[str, Any]:
    """Return current update state including streamed output lines."""
    return _pull_state
