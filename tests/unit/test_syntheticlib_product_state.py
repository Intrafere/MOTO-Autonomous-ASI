from fastapi import HTTPException
import pytest

from backend.shared.config import system_config
from backend.shared.syntheticlib_product import (
    SyntheticLibProductState,
    normalize_syntheticlib_product_state,
    require_syntheticlib_developer_tooling,
    require_syntheticlib_runtime,
    syntheticlib_corpus_access_allowed,
    syntheticlib_developer_tooling_allowed,
    syntheticlib_runtime_allowed,
)


@pytest.fixture(autouse=True)
def restore_product_config():
    previous = (
        system_config.syntheticlib_product_state,
        system_config.syntheticlib_developer_preview_authorized,
        system_config.generic_mode,
    )
    yield
    (
        system_config.syntheticlib_product_state,
        system_config.syntheticlib_developer_preview_authorized,
        system_config.generic_mode,
    ) = previous


@pytest.mark.parametrize("value", [None, "", "future", "LAUNCHED", object()])
def test_unknown_product_state_fails_closed(value) -> None:
    assert normalize_syntheticlib_product_state(value) is SyntheticLibProductState.COMING_SOON


def test_prelaunch_runtime_requires_explicit_local_authority() -> None:
    system_config.syntheticlib_product_state = "developer_preview"
    system_config.syntheticlib_developer_preview_authorized = False
    system_config.generic_mode = False
    assert syntheticlib_runtime_allowed() is False
    with pytest.raises(HTTPException) as exc_info:
        require_syntheticlib_runtime()
    assert exc_info.value.status_code == 403

    system_config.syntheticlib_developer_preview_authorized = True
    assert syntheticlib_runtime_allowed() is True


def test_generic_mode_denies_preview_and_launched_runtime() -> None:
    system_config.syntheticlib_developer_preview_authorized = True
    system_config.generic_mode = True
    for state in ("developer_preview", "launched"):
        system_config.syntheticlib_product_state = state
        assert syntheticlib_runtime_allowed() is False


# (state, authorized, generic) -> (runtime, developer tooling, corpus access)
_AUTHORITY_MATRIX = [
    ("coming_soon", False, False, False, False, False),
    ("coming_soon", True, False, True, True, True),
    ("developer_preview", True, False, True, True, True),
    ("developer_preview", True, True, False, False, False),
    ("future_unknown_state", True, False, False, False, False),
    ("launched", False, False, True, False, True),
    ("launched", True, False, True, True, True),
    ("launched", False, True, False, False, True),
    ("launched", True, True, False, False, True),
]


@pytest.mark.parametrize(
    ("state", "authorized", "generic", "runtime", "tooling", "corpus"),
    _AUTHORITY_MATRIX,
)
def test_runtime_developer_tooling_and_corpus_authority_are_independent(
    state, authorized, generic, runtime, tooling, corpus
) -> None:
    system_config.syntheticlib_product_state = state
    system_config.syntheticlib_developer_preview_authorized = authorized
    system_config.generic_mode = generic

    assert syntheticlib_runtime_allowed() is runtime
    assert syntheticlib_developer_tooling_allowed() is tooling
    assert syntheticlib_corpus_access_allowed() is corpus
    if not tooling:
        with pytest.raises(HTTPException) as exc_info:
            require_syntheticlib_developer_tooling()
        assert exc_info.value.status_code == 403
