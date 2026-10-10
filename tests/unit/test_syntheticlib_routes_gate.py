from unittest import mock

import pytest

from backend.api.routes import syntheticlib4
from backend.shared.config import system_config
from tests.unit.syntheticlib_gate_cases import (
    DEVELOPER_TOOLING_DENIED_CONFIGS,
    RUNTIME_DENIED_CONFIGS,
    lifecycle_config,  # noqa: F401 - pytest fixture
)
from tests.unit.test_proof_trade_ledger import identity_store  # noqa: F401 - pytest fixture
from tests.unit.test_syntheticlib_trade_eligibility import (
    _populate_installation,
    _write_valid_store,
)


@pytest.mark.asyncio
async def test_prelaunch_status_is_safe_and_does_not_read_credentials() -> None:
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
            syntheticlib4.syntheticlib4_client,
            "get_status",
            side_effect=AssertionError("credential status must not be read"),
        ) as get_status:
            result = await syntheticlib4.get_syntheticlib4_status()
    finally:
        (
            system_config.syntheticlib_product_state,
            system_config.syntheticlib_developer_preview_authorized,
            system_config.generic_mode,
        ) = previous

    assert result["runtime_available"] is False
    assert result["status"]["credential_configured"] is False
    get_status.assert_not_called()


@pytest.mark.asyncio
async def test_prelaunch_api_key_route_is_denied_before_secret_store() -> None:
    previous = (
        system_config.syntheticlib_product_state,
        system_config.syntheticlib_developer_preview_authorized,
        system_config.generic_mode,
    )
    system_config.syntheticlib_product_state = "developer_preview"
    system_config.syntheticlib_developer_preview_authorized = False
    system_config.generic_mode = False
    try:
        with mock.patch.object(syntheticlib4.syntheticlib4_client, "set_api_key") as set_key:
            with pytest.raises(Exception) as exc_info:
                await syntheticlib4.set_syntheticlib4_api_key(
                    syntheticlib4.SyntheticLib4ApiKeyRequest(api_key="not-used")
                )
    finally:
        (
            system_config.syntheticlib_product_state,
            system_config.syntheticlib_developer_preview_authorized,
            system_config.generic_mode,
        ) = previous

    assert getattr(exc_info.value, "status_code", None) == 403
    set_key.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("product_state", "authorized", "generic_mode"),
    RUNTIME_DENIED_CONFIGS,
)
async def test_trade_evidence_diagnostics_denied_without_runtime_authority(
    lifecycle_config, product_state, authorized, generic_mode
) -> None:
    lifecycle_config.syntheticlib_product_state = product_state
    lifecycle_config.syntheticlib_developer_preview_authorized = authorized
    lifecycle_config.generic_mode = generic_mode
    with mock.patch.object(
        syntheticlib4,
        "project_installation_proof_stores",
        side_effect=AssertionError("canonical stores must not be read"),
    ) as projector:
        with pytest.raises(Exception) as exc_info:
            await syntheticlib4.get_syntheticlib_trade_evidence_diagnostics()

    assert getattr(exc_info.value, "status_code", None) == 403
    projector.assert_not_called()


@pytest.mark.asyncio
async def test_trade_evidence_diagnostics_are_metadata_only_and_ineligible(
    lifecycle_config, tmp_path
) -> None:
    _write_valid_store(tmp_path / "manual_proofs")
    _write_valid_store(
        tmp_path / "auto_sessions" / "SECRET_PROMPT_folder" / "proofs",
        store_kind="autonomous",
        stable_store_id="store_session",
        source_type="leanoj_final",
        lean_code="import Mathlib\n\ntheorem other : 2 + 2 = 4 := by norm_num\n",
    )
    lifecycle_config.syntheticlib_product_state = "developer_preview"
    lifecycle_config.syntheticlib_developer_preview_authorized = True
    lifecycle_config.generic_mode = False
    lifecycle_config.data_dir = str(tmp_path)
    lifecycle_config.auto_sessions_base_dir = str(tmp_path / "auto_sessions")

    response = await syntheticlib4.get_syntheticlib_trade_evidence_diagnostics()
    serialized = response.model_dump_json()

    assert response.eligible_to_offer is False
    assert response.product_state == "developer_preview"
    assert response.source_selection_policy_version == "moto-trade-source-selection-v2"
    assert {store.store_role for store in response.stores} == {
        "active_manual",
        "autonomous_session",
    }
    assert len(response.diagnostics) == 2
    assert all(item.eligible_to_offer is False for item in response.diagnostics)
    assert all(item.source_selected for item in response.diagnostics)
    for private_value in (
        "SECRET_PROMPT",
        str(tmp_path),
        "private user prompt",
        "private source title",
        "private-source-id",
        "theorem other",
    ):
        assert private_value not in serialized


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("product_state", "authorized", "generic_mode"),
    DEVELOPER_TOOLING_DENIED_CONFIGS,
)
async def test_trade_ledger_status_denied_without_runtime_authority(
    lifecycle_config, product_state, authorized, generic_mode
) -> None:
    lifecycle_config.syntheticlib_product_state = product_state
    lifecycle_config.syntheticlib_developer_preview_authorized = authorized
    lifecycle_config.generic_mode = generic_mode
    with mock.patch.object(
        syntheticlib4,
        "get_ledger_status",
        side_effect=AssertionError("ledger must not be read"),
    ) as reader:
        with pytest.raises(Exception) as exc_info:
            await syntheticlib4.get_syntheticlib_trade_ledger_status()

    assert getattr(exc_info.value, "status_code", None) == 403
    reader.assert_not_called()


@pytest.mark.asyncio
async def test_trade_ledger_status_is_read_only_and_metadata_only(
    lifecycle_config, tmp_path
) -> None:
    lifecycle_config.syntheticlib_product_state = "developer_preview"
    lifecycle_config.syntheticlib_developer_preview_authorized = True
    lifecycle_config.generic_mode = False
    lifecycle_config.data_dir = str(tmp_path)

    with mock.patch(
        "backend.shared.installation_identity.load_moto_installation_identity_state",
        return_value=None,
    ):
        response = await syntheticlib4.get_syntheticlib_trade_ledger_status()

    assert response.available is True
    assert response.exists is False
    assert response.upload_enabled is False
    assert response.schema_version == "moto-proof-trade-ledger-v1"
    assert str(tmp_path) not in response.model_dump_json()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_trade_ledger_status_reports_unreadable_ledger_by_reason_code(
    lifecycle_config, tmp_path
) -> None:
    lifecycle_config.syntheticlib_product_state = "developer_preview"
    lifecycle_config.syntheticlib_developer_preview_authorized = True
    lifecycle_config.generic_mode = False
    lifecycle_config.data_dir = str(tmp_path)
    ledger = tmp_path / "syntheticlib_trade" / "proof_trade_ledger.json"
    ledger.parent.mkdir()
    ledger.write_text("{corrupt", encoding="utf-8")

    response = await syntheticlib4.get_syntheticlib_trade_ledger_status()

    assert response.available is False
    assert response.exists is True
    assert response.reason_code == "ledger_corrupt"
    assert ledger.read_text(encoding="utf-8") == "{corrupt"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("product_state", "authorized", "generic_mode"),
    DEVELOPER_TOOLING_DENIED_CONFIGS,
)
async def test_trade_authorization_readiness_denied_without_runtime_authority(
    lifecycle_config, product_state, authorized, generic_mode
) -> None:
    lifecycle_config.syntheticlib_product_state = product_state
    lifecycle_config.syntheticlib_developer_preview_authorized = authorized
    lifecycle_config.generic_mode = generic_mode
    with mock.patch.object(
        syntheticlib4,
        "get_authorization_readiness",
        side_effect=AssertionError("readiness must not be evaluated"),
    ) as reader:
        with pytest.raises(Exception) as exc_info:
            await syntheticlib4.get_syntheticlib_trade_authorization_readiness()

    assert getattr(exc_info.value, "status_code", None) == 403
    reader.assert_not_called()


@pytest.mark.asyncio
async def test_trade_authorization_readiness_is_disabled_and_metadata_only(
    lifecycle_config, tmp_path
) -> None:
    lifecycle_config.syntheticlib_product_state = "developer_preview"
    lifecycle_config.syntheticlib_developer_preview_authorized = True
    lifecycle_config.generic_mode = False
    lifecycle_config.data_dir = str(tmp_path)

    with mock.patch(
        "backend.shared.secret_store.keyring_backend_available", return_value=True
    ), mock.patch(
        "backend.shared.installation_identity.load_moto_installation_identity_state",
        return_value=None,
    ):
        response = await syntheticlib4.get_syntheticlib_trade_authorization_readiness()

    assert response.authorization_enabled is False
    assert response.upload_enabled is False
    assert response.pkce_method == "S256"
    assert response.redirect_policy == "profile_defined"
    assert response.required_scopes == ["proof_trade:create", "proof_trade:read_own"]
    assert response.installation_bound is False
    assert "native_client_profile_unpublished" in response.reason_codes
    assert "frozen_syntheticlib_contract_unpublished" in response.reason_codes
    assert "installation_identity_unbound" in response.reason_codes
    assert str(tmp_path) not in response.model_dump_json()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_trade_authorization_readiness_errors_are_type_only(lifecycle_config) -> None:
    lifecycle_config.syntheticlib_product_state = "developer_preview"
    lifecycle_config.syntheticlib_developer_preview_authorized = True
    lifecycle_config.generic_mode = False
    with mock.patch.object(
        syntheticlib4,
        "get_authorization_readiness",
        side_effect=RuntimeError(r"C:\Users\alice token=abc"),
    ):
        with pytest.raises(Exception) as exc_info:
            await syntheticlib4.get_syntheticlib_trade_authorization_readiness()

    assert getattr(exc_info.value, "status_code", None) == 500
    assert "alice" not in exc_info.value.detail
    assert "token" not in exc_info.value.detail


@pytest.mark.asyncio
async def test_syntheticlib_route_errors_never_echo_exception_text(lifecycle_config) -> None:
    lifecycle_config.syntheticlib_product_state = "developer_preview"
    lifecycle_config.syntheticlib_developer_preview_authorized = True
    lifecycle_config.generic_mode = False
    with mock.patch.object(
        syntheticlib4.syntheticlib4_client,
        "list_releases",
        side_effect=RuntimeError(r"C:\Users\alice\syntheticlib4 token=abc"),
    ):
        with pytest.raises(Exception) as exc_info:
            await syntheticlib4.list_syntheticlib4_releases()

    detail = getattr(exc_info.value, "detail", "")
    assert getattr(exc_info.value, "status_code", None) == 500
    assert detail == "SyntheticLib4 release listing failed (RuntimeError)"


_SNAPSHOT_ID_FIXTURE = "moto_selection_" + "a" * 32
_BATCH_ID_FIXTURE = "moto_batch_" + "a" * 32

_LEDGER_ROUTE_CASES = [
    ("list_syntheticlib_trade_ledger_entries", None, "list_ledger_entries"),
    (
        "create_syntheticlib_trade_selection_snapshot",
        lambda: syntheticlib4.TradeLedgerSelectionRequest(client_proof_ids=["moto_proof_" + "a" * 32]),
        "create_selection_snapshot",
    ),
    (
        "reconcile_syntheticlib_trade_selection_snapshot",
        lambda: syntheticlib4.TradeLedgerSnapshotRequest(snapshot_id=_SNAPSHOT_ID_FIXTURE),
        "reconcile_selection_snapshot",
    ),
    (
        "cancel_syntheticlib_trade_selection_snapshot",
        lambda: syntheticlib4.TradeLedgerSnapshotRequest(snapshot_id=_SNAPSHOT_ID_FIXTURE),
        "cancel_selection_snapshot",
    ),
    (
        "prepare_syntheticlib_trade_batch",
        lambda: syntheticlib4.TradeLedgerSnapshotRequest(snapshot_id=_SNAPSHOT_ID_FIXTURE),
        "prepare_batch",
    ),
    (
        "cancel_syntheticlib_trade_batch",
        lambda: syntheticlib4.TradeLedgerBatchRequest(client_batch_id=_BATCH_ID_FIXTURE),
        "cancel_batch",
    ),
    (
        "archive_and_reset_syntheticlib_trade_ledger",
        lambda: syntheticlib4.TradeLedgerArchiveResetRequest(confirm=True),
        "archive_and_reset_local_ledger",
    ),
    (
        "delete_archived_syntheticlib_trade_ledgers",
        lambda: syntheticlib4.TradeLedgerArchiveResetRequest(confirm=True),
        "delete_archived_local_ledgers",
    ),
]

# The offer panel's calls stay on product authority after launch; every other
# ledger control is developer tooling.
_PRODUCT_AUTHORITY_LEDGER_ROUTES = {
    "create_syntheticlib_trade_selection_snapshot",
    "prepare_syntheticlib_trade_batch",
}


async def _call_ledger_route(route_name, request_factory):
    route = getattr(syntheticlib4, route_name)
    if request_factory is None:
        return await route()
    return await route(request_factory())


@pytest.mark.asyncio
@pytest.mark.parametrize(("route_name", "request_factory", "ledger_function"), _LEDGER_ROUTE_CASES)
@pytest.mark.parametrize(
    ("product_state", "authorized", "generic_mode"),
    DEVELOPER_TOOLING_DENIED_CONFIGS,
)
async def test_trade_ledger_controls_denied_without_runtime_authority(
    lifecycle_config,
    tmp_path,
    route_name,
    request_factory,
    ledger_function,
    product_state,
    authorized,
    generic_mode,
) -> None:
    if (product_state, authorized, generic_mode) not in RUNTIME_DENIED_CONFIGS and (
        route_name in _PRODUCT_AUTHORITY_LEDGER_ROUTES
    ):
        pytest.skip("product-authority route is allowed after launch (covered separately)")
    lifecycle_config.syntheticlib_product_state = product_state
    lifecycle_config.syntheticlib_developer_preview_authorized = authorized
    lifecycle_config.generic_mode = generic_mode
    lifecycle_config.data_dir = str(tmp_path)
    with mock.patch.object(
        syntheticlib4,
        ledger_function,
        side_effect=AssertionError("ledger must not be touched"),
    ) as ledger_call:
        with pytest.raises(Exception) as exc_info:
            await _call_ledger_route(route_name, request_factory)

    assert getattr(exc_info.value, "status_code", None) == 403
    ledger_call.assert_not_called()
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def authorized_installation(lifecycle_config, tmp_path, identity_store):  # noqa: F811
    records = _populate_installation(tmp_path)
    lifecycle_config.syntheticlib_product_state = "developer_preview"
    lifecycle_config.syntheticlib_developer_preview_authorized = True
    lifecycle_config.generic_mode = False
    lifecycle_config.data_dir = str(tmp_path)
    lifecycle_config.auto_sessions_base_dir = str(tmp_path / "auto_sessions")
    return tmp_path, records


@pytest.mark.asyncio
async def test_trade_ledger_controls_walk_local_lifecycle_without_upload(
    authorized_installation,
) -> None:
    root, records = authorized_installation
    client_proof_id = records["active"].client_proof_id

    created = await syntheticlib4.create_syntheticlib_trade_selection_snapshot(
        syntheticlib4.TradeLedgerSelectionRequest(client_proof_ids=[client_proof_id])
    )
    snapshot_id = created.snapshot.snapshot_id
    assert created.snapshot.state == "selected"
    assert created.upload_enabled is False

    entries = await syntheticlib4.list_syntheticlib_trade_ledger_entries()
    assert [item.snapshot_id for item in entries.snapshots] == [snapshot_id]
    assert entries.upload_enabled is False
    serialized = entries.model_dump_json()
    for private in (str(root), client_proof_id, "PRIVATE_PROMPT"):
        assert private not in serialized

    reconciled = await syntheticlib4.reconcile_syntheticlib_trade_selection_snapshot(
        syntheticlib4.TradeLedgerSnapshotRequest(snapshot_id=snapshot_id)
    )
    assert reconciled.snapshot.reconciliation_outcome == "ok"

    prepared = await syntheticlib4.prepare_syntheticlib_trade_batch(
        syntheticlib4.TradeLedgerSnapshotRequest(snapshot_id=snapshot_id)
    )
    assert prepared.outcome == "blocked"
    assert prepared.client_batch_id is None
    assert prepared.reason_codes == ["batch_blocked_ineligible"]
    assert prepared.upload_enabled is False

    cancelled = await syntheticlib4.cancel_syntheticlib_trade_selection_snapshot(
        syntheticlib4.TradeLedgerSnapshotRequest(snapshot_id=snapshot_id)
    )
    assert cancelled.snapshot.state == "cancelled"

    reset = await syntheticlib4.archive_and_reset_syntheticlib_trade_ledger(
        syntheticlib4.TradeLedgerArchiveResetRequest(confirm=True)
    )
    assert reset.archived is True
    assert reset.server_side_deletion_claimed is False
    after = await syntheticlib4.list_syntheticlib_trade_ledger_entries()
    assert after.exists is False
    assert after.snapshots == []


@pytest.mark.asyncio
async def test_trade_ledger_rejections_report_reason_codes_without_writing(
    authorized_installation,
) -> None:
    root, _records = authorized_installation

    with pytest.raises(Exception) as exc_info:
        await syntheticlib4.create_syntheticlib_trade_selection_snapshot(
            syntheticlib4.TradeLedgerSelectionRequest(
                client_proof_ids=["moto_proof_" + "f" * 32]
            )
        )
    assert getattr(exc_info.value, "status_code", None) == 409
    assert exc_info.value.detail == (
        "Proof-trade ledger request rejected: selection_proof_not_found"
    )
    assert not (root / "syntheticlib_trade").exists()

    with pytest.raises(Exception) as exc_info:
        await syntheticlib4.cancel_syntheticlib_trade_batch(
            syntheticlib4.TradeLedgerBatchRequest(client_batch_id=_BATCH_ID_FIXTURE)
        )
    assert getattr(exc_info.value, "status_code", None) == 409
    assert "batch_unknown" in exc_info.value.detail


@pytest.mark.asyncio
async def test_trade_ledger_unreadable_state_is_reason_coded_and_preserved(
    authorized_installation,
) -> None:
    root, records = authorized_installation
    ledger = root / "syntheticlib_trade" / "proof_trade_ledger.json"
    ledger.parent.mkdir()
    ledger.write_text("{corrupt", encoding="utf-8")

    with pytest.raises(Exception) as exc_info:
        await syntheticlib4.create_syntheticlib_trade_selection_snapshot(
            syntheticlib4.TradeLedgerSelectionRequest(
                client_proof_ids=[records["active"].client_proof_id]
            )
        )

    assert getattr(exc_info.value, "status_code", None) == 409
    assert exc_info.value.detail == "Proof-trade ledger unavailable: ledger_corrupt"
    assert ledger.read_text(encoding="utf-8") == "{corrupt"


@pytest.mark.asyncio
async def test_trade_ledger_control_errors_never_echo_exception_text(
    authorized_installation,
) -> None:
    with mock.patch.object(
        syntheticlib4,
        "list_ledger_entries",
        side_effect=RuntimeError(r"C:\Users\alice\ledger token=abc"),
    ):
        with pytest.raises(Exception) as exc_info:
            await syntheticlib4.list_syntheticlib_trade_ledger_entries()

    assert getattr(exc_info.value, "status_code", None) == 500
    assert exc_info.value.detail == "SyntheticLib trade-ledger listing failed (RuntimeError)"


def test_trade_ledger_control_requests_reject_malformed_bodies() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        syntheticlib4.TradeLedgerSelectionRequest(client_proof_ids=[])
    with pytest.raises(ValidationError):
        syntheticlib4.TradeLedgerSelectionRequest(
            client_proof_ids=["moto_proof_" + "a" * 32] * 501
        )
    with pytest.raises(ValidationError):
        syntheticlib4.TradeLedgerArchiveResetRequest(confirm=False)
    with pytest.raises(ValidationError):
        syntheticlib4.TradeLedgerSnapshotRequest(snapshot_id="")


@pytest.mark.asyncio
@pytest.mark.parametrize(("route_name", "request_factory", "ledger_function"), _LEDGER_ROUTE_CASES)
async def test_launched_without_preview_splits_product_and_developer_authority(
    lifecycle_config, tmp_path, route_name, request_factory, ledger_function
) -> None:
    lifecycle_config.syntheticlib_product_state = "launched"
    lifecycle_config.syntheticlib_developer_preview_authorized = False
    lifecycle_config.generic_mode = False
    lifecycle_config.data_dir = str(tmp_path)
    sentinel = AssertionError("ledger reached")
    with mock.patch.object(syntheticlib4, ledger_function, side_effect=sentinel) as ledger_call:
        with pytest.raises(Exception) as exc_info:
            await _call_ledger_route(route_name, request_factory)

    if route_name in _PRODUCT_AUTHORITY_LEDGER_ROUTES:
        # Product authority admits the call; the patched ledger then fails as a
        # type-only 500, proving the gate itself allowed it.
        ledger_call.assert_called_once()
        assert getattr(exc_info.value, "status_code", None) == 500
    else:
        ledger_call.assert_not_called()
        assert getattr(exc_info.value, "status_code", None) == 403
