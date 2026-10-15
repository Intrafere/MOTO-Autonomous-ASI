from unittest import mock

import pytest

from backend.api.routes import connectivity
from backend.api.routes.connectivity import (
    ConnectivityToggleRequest,
    _syntheticlib4_is_outdated,
)
from backend.shared.config import system_config


def test_syntheticlib4_outdated_when_cached_snapshot_has_expired_membership() -> None:
    assert _syntheticlib4_is_outdated(
        {
            "credential_configured": True,
            "auth_mode": "api_key",
            "authenticated": True,
            "membership_active": False,
        },
        ready=True,
    )


def test_syntheticlib4_inactive_runtime_is_not_outdated_without_a_ready_snapshot() -> None:
    assert not _syntheticlib4_is_outdated(
        {
            "credential_configured": False,
            "auth_mode": "inactive",
            "authenticated": False,
            "membership_active": False,
        },
        ready=False,
    )


def test_syntheticlib4_local_snapshot_with_explicit_membership_failure_is_outdated() -> None:
    assert _syntheticlib4_is_outdated(
        {
            "credential_configured": False,
            "auth_mode": "local_snapshot",
            "authenticated": True,
            "membership_active": False,
        },
        ready=True,
    )


def test_syntheticlib4_expired_access_timestamp_is_outdated() -> None:
    assert _syntheticlib4_is_outdated(
        {
            "credential_configured": True,
            "auth_mode": "api_key",
            "authenticated": True,
            "membership_active": True,
            "access_expires_at": "2000-01-01T00:00:00Z",
        },
        ready=True,
    )


def test_syntheticlib4_future_access_timestamp_is_not_outdated() -> None:
    assert not _syntheticlib4_is_outdated(
        {
            "credential_configured": True,
            "auth_mode": "api_key",
            "authenticated": True,
            "membership_active": True,
            "access_expires_at": "2999-01-01T00:00:00Z",
        },
        ready=True,
    )


@pytest.mark.asyncio
async def test_prelaunch_connectivity_status_does_not_read_credentials() -> None:
    previous = (
        system_config.syntheticlib_product_state,
        system_config.syntheticlib_developer_preview_authorized,
        system_config.generic_mode,
    )
    system_config.syntheticlib_product_state = "coming_soon"
    system_config.syntheticlib_developer_preview_authorized = False
    system_config.generic_mode = False
    try:
        with mock.patch.object(
            connectivity.syntheticlib4_client,
            "get_status",
            side_effect=AssertionError("credential status must not be read"),
        ) as get_status:
            result = await connectivity._syntheticlib4_status()
    finally:
        (
            system_config.syntheticlib_product_state,
            system_config.syntheticlib_developer_preview_authorized,
            system_config.generic_mode,
        ) = previous

    assert result["status"] == "coming soon"
    assert result["runtime_available"] is False
    get_status.assert_not_called()


@pytest.mark.asyncio
async def test_disabling_session_history_clears_live_and_durable_assistant_state() -> None:
    old_enabled = system_config.agent_conversation_memory_enabled
    try:
        with (
            mock.patch.object(connectivity, "_workflow_is_active", return_value=False),
            mock.patch.object(connectivity, "save_connectivity_runtime_settings"),
            mock.patch.object(
                connectivity.assistant_proof_search_coordinator,
                "stop_all",
                new=mock.AsyncMock(),
            ) as stop_all,
            mock.patch.object(
                connectivity.assistant_proof_search_coordinator,
                "clear_cooldown_state",
                new=mock.AsyncMock(),
            ) as clear_cooldown_state,
            mock.patch.object(
                connectivity,
                "get_connectivity_status",
                new=mock.AsyncMock(return_value={"success": True}),
            ),
        ):
            result = await connectivity.update_connectivity_toggles(
                ConnectivityToggleRequest(agent_conversation_memory_enabled=False)
            )
    finally:
        system_config.agent_conversation_memory_enabled = old_enabled

    assert result == {"success": True}
    stop_all.assert_awaited_once_with(
        clear_packs=True,
        broadcast=True,
        reason="agent_conversation_memory_disabled",
    )
    clear_cooldown_state.assert_awaited_once_with()

