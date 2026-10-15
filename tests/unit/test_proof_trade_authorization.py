from unittest import mock

import pytest

from backend.shared import proof_trade_authorization as authorization
from backend.shared import secret_store
from backend.shared import syntheticlib_trade_compatibility as compatibility
from backend.shared.config import system_config
from backend.shared.installation_identity import InstallationBinding


@pytest.fixture
def desktop_config(tmp_path):
    previous = (system_config.generic_mode, system_config.data_dir)
    system_config.generic_mode = False
    system_config.data_dir = str(tmp_path)
    yield tmp_path
    system_config.generic_mode, system_config.data_dir = previous


def test_exact_scope_set_is_accepted_in_any_order() -> None:
    assert authorization.validate_exact_proof_trade_scopes(
        ["proof_trade:read_own", "proof_trade:create"]
    ) == ()


@pytest.mark.parametrize(
    ("granted", "expected"),
    [
        (["proof_trade:create"], ("scope_missing",)),
        ([], ("scope_missing",)),
        (["submissions:write", "proof_trade:read_own"], ("scope_missing", "scope_unexpected")),
        (
            ["proof_trade:create", "proof_trade:read_own", "moto:retrieve"],
            ("scope_unexpected",),
        ),
        (
            ["proof_trade:create", "proof_trade:read_own", "moto:attest"],
            ("scope_unexpected",),
        ),
        (
            ["proof_trade:create ", "proof_trade:read_own"],
            ("scope_malformed", "scope_missing"),
        ),
        (["PROOF_TRADE:CREATE", "proof_trade:read_own"], ("scope_missing", "scope_unexpected")),
        ([None, "proof_trade:create", "proof_trade:read_own"], ("scope_malformed",)),
    ],
)
def test_scope_aliases_extras_and_omissions_are_rejected(granted, expected) -> None:
    assert authorization.validate_exact_proof_trade_scopes(granted) == expected


def test_credential_record_is_distinct_and_bound_to_client_and_installation() -> None:
    first = authorization.proof_trade_credential_record_name("client-a", "moto_install_1")
    assert first.startswith("proof_trade_credential::")
    assert first.endswith("::moto_install_1")
    assert "client-a" not in first
    assert first != authorization.proof_trade_credential_record_name("client-b", "moto_install_1")
    assert first != authorization.proof_trade_credential_record_name("client-a", "moto_install_2")
    reserved = {
        secret_store._SYNTHETICLIB4_API_KEY,
        secret_store._OPENAI_CODEX_OAUTH,
        secret_store._XAI_GROK_OAUTH,
        secret_store._OPENROUTER_KEY,
        secret_store._MOTO_INSTALLATION_IDENTITY_STATE,
    }
    assert not any(first.startswith(name) for name in reserved)
    with pytest.raises(ValueError):
        authorization.proof_trade_credential_record_name("", "moto_install_1")
    with pytest.raises(ValueError):
        authorization.proof_trade_credential_record_name("client-a", " ")


def test_generic_mode_readiness_fails_closed_without_probing(desktop_config) -> None:
    system_config.generic_mode = True
    with mock.patch.object(
        secret_store, "keyring_backend_available", side_effect=AssertionError("no probe")
    ):
        readiness = authorization.get_authorization_readiness()
    assert readiness.authorization_enabled is False
    assert readiness.reason_codes == ("generic_mode_unavailable",)


def test_unpublished_profile_and_unbound_installation_are_reason_coded(desktop_config) -> None:
    with mock.patch.object(secret_store, "keyring_backend_available", return_value=True), mock.patch(
        "backend.shared.installation_identity.read_installation_binding", return_value=None
    ):
        readiness = authorization.get_authorization_readiness()
    assert readiness.authorization_enabled is False
    assert readiness.pkce_method == "S256"
    assert readiness.required_scopes == ("proof_trade:create", "proof_trade:read_own")
    assert set(readiness.reason_codes) >= {
        "native_client_profile_unpublished",
        "frozen_syntheticlib_contract_unpublished",
        "installation_identity_unbound",
    }
    assert list(desktop_config.iterdir()) == []


def test_keyring_unavailable_skips_installation_lookup(desktop_config) -> None:
    with mock.patch.object(secret_store, "keyring_backend_available", return_value=False), mock.patch(
        "backend.shared.installation_identity.read_installation_binding",
        side_effect=AssertionError("must not read identity without a keyring"),
    ):
        readiness = authorization.get_authorization_readiness()
    assert readiness.keyring_available is False
    assert {"keyring_unavailable", "installation_identity_unavailable"} <= set(
        readiness.reason_codes
    )


def test_moved_root_reports_unavailable_installation(desktop_config) -> None:
    from backend.shared.installation_identity import InstallationBindingMismatchError

    with mock.patch.object(secret_store, "keyring_backend_available", return_value=True), mock.patch(
        "backend.shared.installation_identity.read_installation_binding",
        side_effect=InstallationBindingMismatchError(r"C:\Users\alice moved"),
    ):
        readiness = authorization.get_authorization_readiness()
    assert "installation_identity_unavailable" in readiness.reason_codes
    assert "alice" not in repr(readiness)


def test_bound_installation_and_published_assets_still_never_enable(desktop_config, tmp_path) -> None:
    plan_root = tmp_path / "plan"
    for _, relative_path, kind in compatibility.FROZEN_ASSET_REQUIREMENTS:
        target = plan_root / relative_path
        if kind == "file":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("{}", encoding="utf-8")
        else:
            target.mkdir(parents=True, exist_ok=True)
            (target / "member.json").write_text("{}", encoding="utf-8")
    binding = InstallationBinding(installation_id="moto_install_x", root_binding_id="moto_root_x")
    original_inventory = compatibility.inventory_frozen_assets
    with mock.patch.object(secret_store, "keyring_backend_available", return_value=True), mock.patch(
        "backend.shared.installation_identity.read_installation_binding", return_value=binding
    ), mock.patch.object(
        compatibility,
        "inventory_frozen_assets",
        side_effect=lambda: original_inventory(plan_root),
    ):
        readiness = authorization.get_authorization_readiness()
    assert readiness.profile_published is True
    assert readiness.installation_bound is True
    assert readiness.authorization_enabled is False
    assert readiness.reason_codes == ("frozen_contract_not_registered",)


def test_invalid_asset_inventory_fails_closed(desktop_config) -> None:
    with mock.patch.object(secret_store, "keyring_backend_available", return_value=True), mock.patch(
        "backend.shared.installation_identity.read_installation_binding", return_value=None
    ), mock.patch.object(
        compatibility,
        "inventory_frozen_assets",
        side_effect=compatibility.CompatibilityContractError("case collision"),
    ):
        readiness = authorization.get_authorization_readiness()
    assert readiness.profile_published is False
    assert "frozen_asset_inventory_invalid" in readiness.reason_codes


def test_keyring_probe_rejects_fail_backend_and_generic_mode(desktop_config) -> None:
    from keyring.backends import fail

    with mock.patch.object(secret_store.keyring, "get_keyring", return_value=fail.Keyring()):
        assert secret_store.keyring_backend_available() is False

    class _Usable:
        priority = 5

    with mock.patch.object(secret_store.keyring, "get_keyring", return_value=_Usable()):
        assert secret_store.keyring_backend_available() is True
        system_config.generic_mode = True
        assert secret_store.keyring_backend_available() is False

    class _Negative:
        priority = -1

    system_config.generic_mode = False
    with mock.patch.object(secret_store.keyring, "get_keyring", return_value=_Negative()):
        assert secret_store.keyring_backend_available() is False
