"""
Helpers for resolving user-influenced paths within trusted storage roots.
"""
import os
from pathlib import Path


def validate_single_path_component(value: str, label: str = "path component") -> str:
    """Allow only one non-empty path component with no traversal separators."""
    normalized = (value or "").strip()
    if not normalized:
        raise ValueError(f"{label} is required")

    if normalized in {".", ".."}:
        raise ValueError(f"Invalid {label}: {value}")

    separators = {os.path.sep}
    if os.path.altsep:
        separators.add(os.path.altsep)

    if any(separator in normalized for separator in separators):
        raise ValueError(f"Invalid {label}: {value}")

    return normalized


def resolve_path_within_root(root: Path, *unsafe_parts: str) -> Path:
    """
    Resolve an untrusted relative path within a trusted root.

    Uses normpath/realpath containment checks so the resolved result cannot
    escape the configured storage root.
    """
    root_real = os.path.realpath(os.path.normpath(str(root)))
    candidate_real = os.path.realpath(
        os.path.normpath(os.path.join(root_real, *unsafe_parts))
    )

    root_prefix = root_real if root_real.endswith(os.sep) else root_real + os.sep
    if candidate_real != root_real and not candidate_real.startswith(root_prefix):
        raise ValueError("Resolved path escapes trusted root")

    return Path(candidate_real)
