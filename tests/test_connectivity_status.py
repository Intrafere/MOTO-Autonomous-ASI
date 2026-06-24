from backend.api.routes.connectivity import _syntheticlib4_is_outdated


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


def test_syntheticlib4_offline_ready_snapshot_is_not_outdated_without_live_access() -> None:
    assert not _syntheticlib4_is_outdated(
        {
            "credential_configured": False,
            "auth_mode": "built_in_offline_fixture",
            "authenticated": True,
            "membership_active": True,
        },
        ready=True,
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

