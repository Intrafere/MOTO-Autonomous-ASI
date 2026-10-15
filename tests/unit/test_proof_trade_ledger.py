from __future__ import annotations

import ast
import asyncio
import json
import os
import shutil
from pathlib import Path

import pytest

import backend.shared.proof_trade_ledger as ledger_module
from backend.shared import installation_identity as identity
from backend.shared.proof_trade_ledger import (
    ProofTradeLedgerRejected,
    ProofTradeLedgerUnavailable,
    archive_and_reset_local_ledger,
    cancel_batch,
    cancel_selection_snapshot,
    create_selection_snapshot,
    delete_archived_local_ledgers,
    get_ledger_status,
    ledger_blocks_installation_reset,
    ledger_path,
    list_ledger_entries,
    prepare_batch,
    reconcile_selection_snapshot,
)
from tests.unit.test_syntheticlib_trade_eligibility import (
    LEAN_CODE,
    _populate_installation,
    _rewrite,
    _write_valid_store,
)


@pytest.fixture
def identity_store(monkeypatch):
    stored = {"value": None}
    monkeypatch.setattr(identity, "load_moto_installation_identity_state", lambda: stored["value"])
    monkeypatch.setattr(
        identity,
        "store_moto_installation_identity_state",
        lambda value: stored.__setitem__("value", value),
    )
    monkeypatch.setattr(
        identity,
        "clear_moto_installation_identity_state",
        lambda: stored.__setitem__("value", None),
    )
    monkeypatch.setattr(identity.system_config, "generic_mode", False)
    return stored


@pytest.fixture
def installation(tmp_path, identity_store):
    records = _populate_installation(tmp_path)
    return tmp_path, records


def _ledger_json(root: Path) -> dict:
    return json.loads(ledger_path(root).read_text(encoding="utf-8"))


def _allow_eligibility(monkeypatch) -> None:
    original = ledger_module.project_installation_proof_stores

    def eligible(*args, **kwargs):
        projection = original(*args, **kwargs)
        diagnostics = tuple(
            type(item)(**{**item.__dict__, "eligible_to_offer": True})
            for item in projection.diagnostics
        )
        return type(projection)(**{**projection.__dict__, "diagnostics": diagnostics})

    monkeypatch.setattr(ledger_module, "project_installation_proof_stores", eligible)


def test_status_of_absent_ledger_creates_nothing(tmp_path, identity_store):
    status = get_ledger_status(tmp_path)

    assert status["available"] is True
    assert status["exists"] is False
    assert status["upload_enabled"] is False
    assert status["ledger_generation"] is None
    assert not ledger_path(tmp_path).exists()
    assert identity_store["value"] is None
    assert not (tmp_path / identity.ROOT_BINDING_MARKER).exists()


def test_snapshot_is_metadata_only_and_bound_to_canonical_identity(installation):
    root, records = installation
    active = records["active"]

    summary = create_selection_snapshot([active.client_proof_id], data_root=root)

    assert summary["state"] == "selected"
    assert summary["entry_count"] == 1
    payload = _ledger_json(root)
    serialized = json.dumps(payload)
    for private in (
        LEAN_CODE,
        "1 + 1 = 2",
        "PRIVATE_PROMPT",
        str(root),
        active.proof_id,
        active.run_id,
    ):
        assert private not in serialized
    snapshot = payload["selection_snapshots"][summary["snapshot_id"]]
    entry = snapshot["entries"][0]
    assert entry["client_proof_id"] == active.client_proof_id
    assert entry["canonical_locator_digest"] == active.canonical_locator_digest
    assert entry["stable_store_id"] == "store_active"
    assert entry["proof_set_revision"] == 4
    assert payload["account_binding"] is None
    assert payload["contract_version"] is None
    binding = identity.read_installation_binding(root)
    assert payload["installation_id"] == binding.installation_id
    assert payload["root_binding_id"] == binding.root_binding_id


def test_reselecting_same_proofs_is_idempotent(installation):
    root, records = installation
    ids = [records["active"].client_proof_id, records["legacy"].client_proof_id]

    first = create_selection_snapshot(ids, data_root=root)
    generation = _ledger_json(root)["ledger_generation"]
    second = create_selection_snapshot(list(reversed(ids)), data_root=root)

    assert first["snapshot_id"] == second["snapshot_id"]
    assert _ledger_json(root)["ledger_generation"] == generation
    assert len(_ledger_json(root)["selection_snapshots"]) == 1


@pytest.mark.parametrize(
    ("ids", "reason"),
    [
        ([], "selection_empty"),
        (["not-an-id"], "selection_malformed_id"),
        (["moto_proof_" + "a" * 32, "moto_proof_" + "a" * 32], "selection_duplicate_id"),
        (["moto_proof_" + "f" * 32], "selection_proof_not_found"),
    ],
)
def test_invalid_selection_is_rejected_without_writing(installation, ids, reason):
    root, _records = installation

    with pytest.raises(ProofTradeLedgerRejected) as caught:
        create_selection_snapshot(ids, data_root=root)

    assert reason in caught.value.reason_codes
    assert not ledger_path(root).exists()


def test_incomplete_or_deselected_proof_cannot_be_selected(tmp_path, identity_store):
    record = _write_valid_store(tmp_path / "manual_proofs", local_rank="not_novel")

    with pytest.raises(ProofTradeLedgerRejected) as caught:
        create_selection_snapshot([record.client_proof_id], data_root=tmp_path)

    assert "selection_source_not_selected" in caught.value.reason_codes
    assert not ledger_path(tmp_path).exists()


def test_reconcile_ok_and_advanced_revision_keeps_binding(installation):
    root, records = installation
    snapshot = create_selection_snapshot([records["active"].client_proof_id], data_root=root)

    assert reconcile_selection_snapshot(snapshot["snapshot_id"], data_root=root)[
        "reconciliation_outcome"
    ] == "ok"

    (root / "manual_proofs" / "proof_set_revision.json").write_text(
        json.dumps({"proof_set_revision": 9}), encoding="utf-8"
    )
    result = reconcile_selection_snapshot(snapshot["snapshot_id"], data_root=root)

    assert result["state"] == "selected"
    stored = _ledger_json(root)["selection_snapshots"][snapshot["snapshot_id"]]
    assert stored["last_reconciliation"]["revision_advanced"] is True
    assert stored["entries"][0]["proof_set_revision"] == 4


def _corrupt_deleted(root: Path, records) -> None:
    record = records["active"]
    for name in (
        f"proof_{record.proof_id}.json",
        f"proof_{record.proof_id}_lean.lean",
        f"proof_{record.proof_id}_checked.lean",
    ):
        path = root / "manual_proofs" / name
        if path.exists():
            path.unlink()


def _corrupt_moved(root: Path, records) -> None:
    source = root / "manual_proofs"
    target = root / "manual_proof_runs" / "manual-run-2" / "proofs"
    target.parent.mkdir(parents=True)
    shutil.move(str(source), str(target))


def _corrupt_replaced(root: Path, records) -> None:
    record = records["active"]
    _rewrite(root / "manual_proofs", record, canonical_locator_digest="0" * 64)


def _corrupt_evidence(root: Path, records) -> None:
    record = records["active"]
    _rewrite(root / "manual_proofs", record, compile_receipt=None)


@pytest.mark.parametrize(
    ("corrupt", "reason"),
    [
        (_corrupt_deleted, "reconcile_proof_missing"),
        (_corrupt_moved, "reconcile_proof_moved"),
        (_corrupt_replaced, "reconcile_hash_changed"),
        (_corrupt_evidence, "reconcile_evidence_incomplete"),
    ],
)
def test_reconcile_failure_is_terminal_and_invalidates_batches(
    installation, monkeypatch, corrupt, reason
):
    root, records = installation
    _allow_eligibility(monkeypatch)
    snapshot = create_selection_snapshot([records["active"].client_proof_id], data_root=root)
    batch = prepare_batch(snapshot["snapshot_id"], data_root=root)
    assert batch["outcome"] == "prepared"

    corrupt(root, records)
    result = reconcile_selection_snapshot(snapshot["snapshot_id"], data_root=root)

    assert result["state"] == "invalidated"
    assert reason in result["reason_codes"]
    payload = _ledger_json(root)
    assert payload["batches"][batch["client_batch_id"]]["state"] == "invalidated"
    with pytest.raises(ProofTradeLedgerRejected) as caught:
        reconcile_selection_snapshot(snapshot["snapshot_id"], data_root=root)
    assert caught.value.reason_codes == ("snapshot_not_active",)


def test_prepare_batch_is_blocked_while_nothing_is_eligible(installation):
    root, records = installation
    snapshot = create_selection_snapshot([records["active"].client_proof_id], data_root=root)

    result = prepare_batch(snapshot["snapshot_id"], data_root=root)

    assert result == {
        "outcome": "blocked",
        "client_batch_id": None,
        "reason_codes": ["batch_blocked_ineligible"],
    }
    assert _ledger_json(root)["batches"] == {}
    assert ledger_blocks_installation_reset(root) is False


def test_prepare_batch_is_idempotent_and_cancellable(installation, monkeypatch):
    root, records = installation
    _allow_eligibility(monkeypatch)
    snapshot = create_selection_snapshot([records["active"].client_proof_id], data_root=root)

    first = prepare_batch(snapshot["snapshot_id"], data_root=root)
    second = prepare_batch(snapshot["snapshot_id"], data_root=root)

    assert first["outcome"] == "prepared"
    assert second == {**first, "outcome": "existing"}
    batches = _ledger_json(root)["batches"]
    assert list(batches) == [first["client_batch_id"]]
    entry = batches[first["client_batch_id"]]
    for reserved in (
        "idempotency_key",
        "request_digest",
        "contract_version",
        "server_receipt",
        "status_cursor",
        "retry_state",
    ):
        assert entry[reserved] is None
    assert ledger_blocks_installation_reset(root) is True

    cancel_batch(first["client_batch_id"], data_root=root)

    assert _ledger_json(root)["batches"][first["client_batch_id"]]["state"] == "cancelled"
    assert ledger_blocks_installation_reset(root) is False
    with pytest.raises(ProofTradeLedgerRejected):
        cancel_batch(first["client_batch_id"], data_root=root)


def test_cancel_snapshot_cancels_prepared_batches(installation, monkeypatch):
    root, records = installation
    _allow_eligibility(monkeypatch)
    snapshot = create_selection_snapshot([records["active"].client_proof_id], data_root=root)
    batch = prepare_batch(snapshot["snapshot_id"], data_root=root)

    result = cancel_selection_snapshot(snapshot["snapshot_id"], data_root=root)

    assert result["state"] == "cancelled"
    assert _ledger_json(root)["batches"][batch["client_batch_id"]]["state"] == "cancelled"
    status = get_ledger_status(root)
    assert status["snapshot_counts"] == {"selected": 0, "invalidated": 0, "cancelled": 1}
    assert status["batch_counts"] == {"prepared": 0, "cancelled": 1, "invalidated": 0}


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda payload: "{not json", "ledger_corrupt"),
        (lambda payload: json.dumps({**payload, "schema_version": "moto-proof-trade-ledger-v9"}), "ledger_schema_unsupported"),
        (lambda payload: json.dumps({**payload, "account_binding": {"account": "x"}}), "ledger_corrupt"),
        (lambda payload: json.dumps({**payload, "unexpected": True}), "ledger_corrupt"),
        (lambda payload: json.dumps([payload]), "ledger_corrupt"),
    ],
)
def test_unreadable_ledger_fails_closed_without_overwrite(installation, mutation, reason):
    root, records = installation
    create_selection_snapshot([records["active"].client_proof_id], data_root=root)
    path = ledger_path(root)
    damaged = mutation(_ledger_json(root))
    path.write_text(damaged, encoding="utf-8")

    with pytest.raises(ProofTradeLedgerUnavailable) as caught:
        create_selection_snapshot([records["legacy"].client_proof_id], data_root=root)

    assert caught.value.reason_code == reason
    assert path.read_text(encoding="utf-8") == damaged
    assert get_ledger_status(root)["reason_code"] == reason
    assert get_ledger_status(root)["available"] is False
    assert ledger_blocks_installation_reset(root) is True


def test_tampered_selection_digest_fails_closed(installation):
    root, records = installation
    snapshot = create_selection_snapshot([records["active"].client_proof_id], data_root=root)
    payload = _ledger_json(root)
    payload["selection_snapshots"][snapshot["snapshot_id"]]["entries"][0][
        "proof_set_revision"
    ] = 99
    ledger_path(root).write_text(json.dumps(payload), encoding="utf-8")

    assert get_ledger_status(root)["reason_code"] == "ledger_corrupt"


def test_symlinked_ledger_fails_closed(installation, tmp_path_factory):
    root, records = installation
    outside = tmp_path_factory.mktemp("outside") / "ledger.json"
    outside.write_text("{}", encoding="utf-8")
    ledger_path(root).parent.mkdir(parents=True)
    try:
        os.symlink(outside, ledger_path(root))
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation requires privilege")

    with pytest.raises(ProofTradeLedgerUnavailable) as caught:
        create_selection_snapshot([records["active"].client_proof_id], data_root=root)

    assert caught.value.reason_code == "ledger_not_regular"
    assert outside.read_text(encoding="utf-8") == "{}"


def test_copied_root_ledger_fails_closed_until_archived(installation, tmp_path_factory):
    root, records = installation
    create_selection_snapshot([records["active"].client_proof_id], data_root=root)
    copy = tmp_path_factory.mktemp("copy") / "data"
    shutil.copytree(root, copy)

    assert get_ledger_status(copy)["reason_code"] == "ledger_installation_binding_mismatch"
    with pytest.raises(ProofTradeLedgerUnavailable) as caught:
        create_selection_snapshot([records["legacy"].client_proof_id], data_root=copy)
    assert caught.value.reason_code == "ledger_installation_binding_mismatch"

    identity.approve_runtime_root_relocation(copy)
    assert get_ledger_status(copy)["available"] is True


def test_adopted_root_does_not_reuse_ledger_binding(installation, tmp_path_factory):
    root, records = installation
    create_selection_snapshot([records["active"].client_proof_id], data_root=root)
    copy = tmp_path_factory.mktemp("adopt") / "data"
    shutil.copytree(root, copy)
    (copy / identity.ROOT_BINDING_MARKER).unlink()

    identity.adopt_unmarked_runtime_root(copy)

    assert get_ledger_status(copy)["reason_code"] == "ledger_installation_binding_mismatch"


def test_generic_mode_is_unavailable(installation, monkeypatch):
    root, records = installation
    monkeypatch.setattr(ledger_module.system_config, "generic_mode", True)

    with pytest.raises(ProofTradeLedgerUnavailable) as caught:
        create_selection_snapshot([records["active"].client_proof_id], data_root=root)

    assert caught.value.reason_code == "ledger_generic_mode_unavailable"
    assert get_ledger_status(root)["reason_code"] == "ledger_generic_mode_unavailable"
    with pytest.raises(ProofTradeLedgerUnavailable):
        archive_and_reset_local_ledger(root)


def test_failed_replace_keeps_previous_ledger(installation, monkeypatch):
    root, records = installation
    create_selection_snapshot([records["active"].client_proof_id], data_root=root)
    before = ledger_path(root).read_text(encoding="utf-8")

    def fail_replace(*_args, **_kwargs):
        raise OSError("simulated crash before replace")

    monkeypatch.setattr("backend.shared.atomic_json.os.replace", fail_replace)
    with pytest.raises(OSError):
        create_selection_snapshot([records["legacy"].client_proof_id], data_root=root)

    assert ledger_path(root).read_text(encoding="utf-8") == before
    assert [path.name for path in ledger_path(root).parent.iterdir()] == [
        ledger_path(root).name
    ]


def test_leftover_temp_files_are_ignored(installation):
    root, records = installation
    create_selection_snapshot([records["active"].client_proof_id], data_root=root)
    (ledger_path(root).parent / f".{ledger_path(root).name}.abc.tmp").write_text(
        "{partial", encoding="utf-8"
    )

    status = get_ledger_status(root)

    assert status["available"] is True
    assert status["snapshot_counts"]["selected"] == 1


def test_cancelled_awaiting_task_still_commits_consistent_write(installation):
    root, records = installation

    async def run() -> None:
        task = asyncio.ensure_future(
            asyncio.to_thread(
                create_selection_snapshot,
                [records["active"].client_proof_id],
                data_root=root,
            )
        )
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())

    status = get_ledger_status(root)
    assert status["available"] is True
    assert status["snapshot_counts"]["selected"] == 1


def test_archive_and_reset_preserves_old_ledger_without_server_claim(installation):
    root, records = installation
    create_selection_snapshot([records["active"].client_proof_id], data_root=root)
    ledger_path(root).write_text("{corrupt", encoding="utf-8")

    result = archive_and_reset_local_ledger(root)

    assert result == {"archived": True, "server_side_deletion_claimed": False}
    assert not ledger_path(root).exists()
    archived = list((ledger_path(root).parent / "archive").iterdir())
    assert len(archived) == 1
    assert archived[0].read_text(encoding="utf-8") == "{corrupt"
    assert get_ledger_status(root)["available"] is True
    assert archive_and_reset_local_ledger(root)["archived"] is False


def test_delete_archived_ledgers_removes_only_regular_archives(installation):
    root, records = installation
    create_selection_snapshot([records["active"].client_proof_id], data_root=root)
    archive_and_reset_local_ledger(root)
    create_selection_snapshot([records["active"].client_proof_id], data_root=root)
    archive_dir = ledger_path(root).parent / "archive"
    unrelated = archive_dir / "notes.txt"
    unrelated.write_text("keep", encoding="utf-8")
    assert get_ledger_status(root)["archived_ledger_count"] == 1

    result = delete_archived_local_ledgers(root)

    assert result == {
        "deleted_count": 1,
        "skipped_count": 0,
        "server_side_deletion_claimed": False,
    }
    assert sorted(path.name for path in archive_dir.iterdir()) == ["notes.txt"]
    # The current ledger is never touched.
    assert get_ledger_status(root)["snapshot_counts"]["selected"] == 1
    assert get_ledger_status(root)["archived_ledger_count"] == 0
    assert delete_archived_local_ledgers(root)["deleted_count"] == 0


def test_delete_archived_ledgers_never_follows_symlinks(installation, tmp_path_factory):
    root, _ = installation
    archive_dir = ledger_path(root).parent / "archive"
    archive_dir.mkdir(parents=True)
    outside = tmp_path_factory.mktemp("outside") / "target.json"
    outside.write_text("{}", encoding="utf-8")
    try:
        (archive_dir / "proof_trade_ledger.20260101T000000000000Z.json").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink privilege unavailable")

    result = delete_archived_local_ledgers(root)

    assert result["deleted_count"] == 0
    assert result["skipped_count"] == 1
    assert outside.exists()


def test_delete_archived_ledgers_unavailable_in_generic_mode(installation, monkeypatch):
    root, _ = installation
    monkeypatch.setattr(ledger_module.system_config, "generic_mode", True)
    with pytest.raises(ProofTradeLedgerUnavailable) as exc_info:
        delete_archived_local_ledgers(root)
    assert exc_info.value.reason_code == "ledger_generic_mode_unavailable"


def test_installation_reset_is_blocked_by_queued_or_unreadable_ledger(
    installation, monkeypatch
):
    root, records = installation
    _allow_eligibility(monkeypatch)
    snapshot = create_selection_snapshot([records["active"].client_proof_id], data_root=root)
    prepare_batch(snapshot["snapshot_id"], data_root=root)

    with pytest.raises(identity.InstallationIdentityError):
        identity.reset_installation_identity(root)

    ledger_path(root).write_text("{corrupt", encoding="utf-8")
    with pytest.raises(identity.InstallationIdentityError):
        identity.reset_installation_identity(root)

    archive_and_reset_local_ledger(root)
    assert identity.reset_installation_identity(root).installation_id


def test_ledger_module_is_separate_from_workflow_and_search_state():
    source = Path(ledger_module.__file__).read_text(encoding="utf-8")
    imported = {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module
    }
    forbidden = (
        "backend.autonomous",
        "backend.leanoj",
        "backend.shared.proof_search",
        "backend.api",
    )
    assert not [
        module for module in imported if module.startswith(forbidden)
    ]


def test_entry_listing_of_absent_ledger_creates_nothing(tmp_path, identity_store):
    listing = list_ledger_entries(tmp_path)

    assert listing["available"] is True
    assert listing["exists"] is False
    assert listing["snapshots"] == []
    assert listing["batches"] == []
    assert listing["upload_enabled"] is False
    assert list(tmp_path.iterdir()) == []
    assert identity_store["value"] is None


def test_entry_listing_is_metadata_only_and_newest_first(installation, monkeypatch):
    root, records = installation
    _allow_eligibility(monkeypatch)
    first = create_selection_snapshot([records["active"].client_proof_id], data_root=root)
    second = create_selection_snapshot([records["legacy"].client_proof_id], data_root=root)
    batch = prepare_batch(second["snapshot_id"], data_root=root)
    reconcile_selection_snapshot(first["snapshot_id"], data_root=root)

    listing = list_ledger_entries(root)

    assert [item["snapshot_id"] for item in listing["snapshots"]] == [
        second["snapshot_id"],
        first["snapshot_id"],
    ]
    reconciled = listing["snapshots"][1]
    assert reconciled["reconciliation_outcome"] == "ok"
    assert reconciled["reconciled_at"] is not None
    assert reconciled["entry_count"] == 1
    assert listing["batches"] == [
        {
            "client_batch_id": batch["client_batch_id"],
            "snapshot_id": second["snapshot_id"],
            "state": "prepared",
            "created_at": listing["batches"][0]["created_at"],
            "updated_at": listing["batches"][0]["updated_at"],
        }
    ]
    serialized = json.dumps(listing)
    for private in (
        str(root),
        records["active"].client_proof_id,
        records["active"].canonical_locator_digest,
        records["active"].proof_id,
        LEAN_CODE,
        "PRIVATE_PROMPT",
    ):
        assert private not in serialized


def test_entry_listing_reports_unreadable_ledger_without_repair(installation):
    root, records = installation
    create_selection_snapshot([records["active"].client_proof_id], data_root=root)
    path = ledger_path(root)
    path.write_text("{not json", encoding="utf-8")

    listing = list_ledger_entries(root)

    assert listing["available"] is False
    assert listing["reason_code"] == "ledger_corrupt"
    assert listing["snapshots"] == []
    assert path.read_text(encoding="utf-8") == "{not json"
