"""Shared SyntheticLib lifecycle/authority cases for gate tests."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import pytest

from backend.shared.config import system_config

# (product_state, local_preview_authorized, generic_mode) tuples with no
# backend SyntheticLib runtime authority.
RUNTIME_DENIED_CONFIGS = [
    ("coming_soon", False, False),
    ("developer_preview", False, False),
    ("future_unknown_state", True, False),
    ("developer_preview", True, True),
    ("launched", True, True),
]

# Developer tooling additionally stays denied after launch without local
# preview authority (the post-launch authority split).
DEVELOPER_TOOLING_DENIED_CONFIGS = RUNTIME_DENIED_CONFIGS + [
    ("launched", False, False),
]

_LIFECYCLE_FIELDS = (
    "syntheticlib_product_state",
    "syntheticlib_developer_preview_authorized",
    "generic_mode",
    "data_dir",
    "auto_sessions_base_dir",
)


@pytest.fixture
def lifecycle_config():
    previous = tuple(getattr(system_config, field) for field in _LIFECYCLE_FIELDS)
    yield system_config
    for field, value in zip(_LIFECYCLE_FIELDS, previous):
        setattr(system_config, field, value)


def apply_lifecycle(config, product_state: str, authorized: bool, generic_mode: bool) -> None:
    config.syntheticlib_product_state = product_state
    config.syntheticlib_developer_preview_authorized = authorized
    config.generic_mode = generic_mode


@contextmanager
def syntheticlib_lifecycle(
    product_state: str = "developer_preview",
    *,
    authorized: bool = True,
    generic_mode: bool = False,
) -> Iterator[None]:
    """Temporarily set the lifecycle; defaults grant local preview authority."""
    previous = tuple(getattr(system_config, field) for field in _LIFECYCLE_FIELDS[:3])
    apply_lifecycle(system_config, product_state, authorized, generic_mode)
    try:
        yield
    finally:
        for field, value in zip(_LIFECYCLE_FIELDS[:3], previous):
            setattr(system_config, field, value)
