"""Reusable filesystem assertions for real-adapter workflow tests."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable


def assert_paths_within(root: Path, paths: Iterable[Path]) -> None:
    """Assert that every supplied path is contained by the temporary runtime root."""
    resolved_root = root.resolve()
    for path in paths:
        assert path.resolve().is_relative_to(resolved_root), (
            f"{path} escaped temporary runtime root {resolved_root}"
        )


def assert_files_exist(root: Path, *relative_paths: str) -> None:
    """Assert that the named files exist below the supplied runtime root."""
    paths = [root / relative_path for relative_path in relative_paths]
    assert_paths_within(root, paths)
    for path in paths:
        assert path.is_file(), f"Expected workflow file does not exist: {path}"


def assert_paths_absent(root: Path, *relative_paths: str) -> None:
    """Assert that the named files or directories are absent below the runtime root."""
    paths = [root / relative_path for relative_path in relative_paths]
    assert_paths_within(root, paths)
    for path in paths:
        assert not path.exists(), f"Workflow path should have been removed: {path}"
