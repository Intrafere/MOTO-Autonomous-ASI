"""
Windows path-length limits for the Mathlib-backed Lean 4 workspace.

Mathlib nests its compiled build files deep below the workspace root. Lean's
tools are not long-path aware, so on Windows any file path longer than MAX_PATH
is unreadable even when the system long-path setting is on. A data root inside a
deeply nested install folder therefore keeps its Lean workspace in a short
per-user folder instead.

Stdlib-only: the desktop launcher imports this before backend dependencies are
installed.
"""
from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

# MAX_PATH (260) minus the terminating NUL.
WINDOWS_MAX_FILE_PATH = 259

# Longest Mathlib build output path relative to the workspace root, measured for
# the v4.35 Mathlib toolchain:
#   .lake/packages/mathlib/.lake/build/lib/lean/Mathlib/Analysis/SpecialFunctions/
#   ContinuousFunctionalCalculus/Rpow/IntegralRepresentation.olean.private.hash
MATHLIB_LONGEST_BUILD_RELATIVE_PATH = 153

# Headroom for Mathlib modules that grow longer than the measured maximum.
WORKSPACE_PATH_MARGIN = 5

MAX_SAFE_WORKSPACE_PATH = (
    WINDOWS_MAX_FILE_PATH - 1 - MATHLIB_LONGEST_BUILD_RELATIVE_PATH - WORKSPACE_PATH_MARGIN
)

# Build output for module `Mathlib/A/B` lands at
#   <workspace>/.lake/packages/mathlib/.lake/build/lib/lean/Mathlib/A/B<suffix>
_MATHLIB_BUILD_LIB_PREFIX_LENGTH = len(
    os.path.join(".lake", "packages", "mathlib", ".lake", "build", "lib", "lean")
)
_MATHLIB_LONGEST_BUILD_SUFFIX_LENGTH = len(".olean.private.hash")

PathLike = Union[str, "os.PathLike[str]"]


@dataclass(frozen=True)
class WorkspacePathProblem:
    """A Lean workspace path that is too long for Mathlib on Windows."""

    workspace_path_length: int
    max_supported_length: int


def is_windows_platform() -> bool:
    """Return True on Windows; patched in tests to exercise Windows behavior."""
    return sys.platform == "win32"


def _workspace_path_length(workspace_dir: PathLike) -> int:
    return len(os.fspath(workspace_dir).rstrip("\\/"))


def windows_workspace_path_problem(
    workspace_dir: PathLike,
    *,
    is_windows: Optional[bool] = None,
) -> Optional[WorkspacePathProblem]:
    """Return a problem when ``workspace_dir`` is too long for current Mathlib.

    ``workspace_dir`` must be absolute; only its string length is inspected.
    """
    if not (is_windows_platform() if is_windows is None else is_windows):
        return None
    length = _workspace_path_length(workspace_dir)
    if length <= MAX_SAFE_WORKSPACE_PATH:
        return None
    return WorkspacePathProblem(
        workspace_path_length=length,
        max_supported_length=MAX_SAFE_WORKSPACE_PATH,
    )


IN_ROOT_WORKSPACE_NAME = "lean4_workspace"
WORKSPACE_ROOT_FILE = "MOTOProofWorkspace.lean"


def data_root_workspace_key(data_root: PathLike) -> str:
    """Stable per-data-root folder name, so installs never share a workspace."""
    normalized = os.path.normcase(os.path.abspath(os.fspath(data_root)))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


def lean_workspace_dir_for(
    data_root: PathLike,
    *,
    is_windows: Optional[bool] = None,
    local_app_data: Optional[str] = None,
) -> Path:
    """Return the Lean workspace for ``data_root``.

    Normally ``<data_root>/lean4_workspace``. When that path is too long for
    Mathlib on Windows, a per-data-root folder under ``%LOCALAPPDATA%\\MOTO\\lean``
    instead. If that is unavailable or also too long, the in-root path is kept so
    the path-too-long repair message explains the problem.
    """
    windows = is_windows_platform() if is_windows is None else is_windows
    in_root = Path(data_root) / IN_ROOT_WORKSPACE_NAME
    if windows_workspace_path_problem(
        os.path.abspath(os.fspath(in_root)),
        is_windows=windows,
    ) is None:
        return in_root
    base = local_app_data if local_app_data is not None else os.environ.get("LOCALAPPDATA", "")
    if not base:
        return in_root
    relocated = Path(base) / "MOTO" / "lean" / data_root_workspace_key(data_root)
    if windows_workspace_path_problem(relocated, is_windows=windows) is not None:
        return in_root
    return relocated


def is_moto_lean_workspace(path: PathLike) -> bool:
    """True only for a folder MOTO created as its Lean workspace."""
    return (Path(path) / WORKSPACE_ROOT_FILE).is_file()


def measured_workspace_path_problem(
    workspace_dir: PathLike,
    *,
    is_windows: Optional[bool] = None,
) -> Optional[WorkspacePathProblem]:
    """Measure the cloned Mathlib sources and report an over-long workspace.

    Catches Mathlib releases whose module paths outgrow the static margin. Walks
    the filesystem, so async callers must run it off the event loop.
    """
    if not (is_windows_platform() if is_windows is None else is_windows):
        return None
    package_dir = Path(workspace_dir) / ".lake" / "packages" / "mathlib"
    source_root = package_dir / "Mathlib"
    if not source_root.is_dir():
        return None
    longest_module = 0
    for current_dir, _dirnames, filenames in os.walk(source_root):
        for filename in filenames:
            if not filename.endswith(".lean"):
                continue
            module_path = os.path.relpath(os.path.join(current_dir, filename), package_dir)
            longest_module = max(longest_module, len(module_path) - len(".lean"))
    if not longest_module:
        return None
    length = _workspace_path_length(workspace_dir)
    longest_relative = (
        _MATHLIB_BUILD_LIB_PREFIX_LENGTH + 1 + longest_module + _MATHLIB_LONGEST_BUILD_SUFFIX_LENGTH
    )
    if length + 1 + longest_relative <= WINDOWS_MAX_FILE_PATH:
        return None
    return WorkspacePathProblem(
        workspace_path_length=length,
        max_supported_length=WINDOWS_MAX_FILE_PATH - 1 - longest_relative,
    )


def describe_workspace_path_problem(problem: WorkspacePathProblem) -> str:
    """Explain the problem without exposing the full local path."""
    return (
        "MOTO's folder path is too long for Lean on Windows. The Lean workspace path is "
        f"{problem.workspace_path_length} characters, but Mathlib needs it to be at most "
        f"{problem.max_supported_length} characters: Mathlib's compiled files are nested "
        "deep inside the workspace, and Windows rejects file paths longer than 260 characters."
    )


WORKSPACE_PATH_FIX = (
    "Move the MOTO folder somewhere with a short path, such as C:\\MOTO (avoid extracting "
    "the ZIP into a folder with the same name)"
)
WORKSPACE_PATH_GUIDANCE = f"{WORKSPACE_PATH_FIX}, then relaunch MOTO."
