"""Central SyntheticLib product lifecycle and private-runtime policy."""
from __future__ import annotations

from enum import Enum
from typing import Any

from fastapi import HTTPException

from backend.shared.config import system_config


class SyntheticLibProductState(str, Enum):
    COMING_SOON = "coming_soon"
    DEVELOPER_PREVIEW = "developer_preview"
    LAUNCHED = "launched"


def normalize_syntheticlib_product_state(value: Any) -> SyntheticLibProductState:
    """Normalize deployment input without allowing unknown states to enable access."""
    if isinstance(value, SyntheticLibProductState):
        return value
    try:
        return SyntheticLibProductState(str(value or "").strip())
    except ValueError:
        return SyntheticLibProductState.COMING_SOON


def get_syntheticlib_product_state() -> SyntheticLibProductState:
    return normalize_syntheticlib_product_state(
        system_config.syntheticlib_product_state
    )


def _known_product_state() -> SyntheticLibProductState | None:
    raw_state = system_config.syntheticlib_product_state
    if isinstance(raw_state, SyntheticLibProductState):
        return raw_state
    try:
        return SyntheticLibProductState(str(raw_state or "").strip())
    except ValueError:
        return None


def syntheticlib_runtime_allowed() -> bool:
    """Return backend execution authority; browser developer mode is irrelevant."""
    state = _known_product_state()
    if state is None:
        return False
    if state is SyntheticLibProductState.LAUNCHED:
        return not system_config.generic_mode
    return bool(
        system_config.syntheticlib_developer_preview_authorized
        and not system_config.generic_mode
        and state
        in {
            SyntheticLibProductState.COMING_SOON,
            SyntheticLibProductState.DEVELOPER_PREVIEW,
        }
    )


def syntheticlib_developer_tooling_allowed() -> bool:
    """Developer diagnostics/reset authority: local preview authority even after launch."""
    return bool(
        _known_product_state() is not None
        and system_config.syntheticlib_developer_preview_authorized
        and not system_config.generic_mode
    )


def syntheticlib_corpus_access_allowed() -> bool:
    """Whether the SyntheticLib4 proof-search corpus may be searched or injected.

    Launched corpus search is a product capability in either deployment mode;
    before launch it needs the same local preview authority as private runtime.
    """
    if _known_product_state() is SyntheticLibProductState.LAUNCHED:
        return True
    return syntheticlib_runtime_allowed()


def require_syntheticlib_runtime() -> None:
    if not syntheticlib_runtime_allowed():
        raise HTTPException(
            status_code=403,
            detail=(
                "SyntheticLib integration runtime is unavailable. "
                "Prelaunch execution requires explicit local developer-preview authority."
            ),
        )


def require_syntheticlib_developer_tooling() -> None:
    if not syntheticlib_developer_tooling_allowed():
        raise HTTPException(
            status_code=403,
            detail=(
                "SyntheticLib developer tooling is unavailable. "
                "It requires explicit local developer-preview authority on a desktop backend."
            ),
        )
