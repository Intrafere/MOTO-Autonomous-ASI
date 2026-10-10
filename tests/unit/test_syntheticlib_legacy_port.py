"""Explicit legacy-proof migration and the legacy_port evidence class."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import backend.shared.syntheticlib_legacy_port as legacy_module
from backend.autonomous.memory.proof_database import ProofDatabase
from backend.shared.config import system_config
from backend.shared.installation_identity import InstallationIdentityError
from backend.shared.legacy_port_receipt import legacy_port_receipt_digest
from backend.shared.models import ProofRecord
from backend.shared.proof_trade_ledger import create_selection_snapshot
from backend.shared.syntheticlib_legacy_port import (
    LegacyPortUnavailable,
    migrate_legacy_proofs,
)
from backend.shared.syntheticlib_trade_eligibility import project_installation_proof_stores
from backend.shared.workflow_start_guard import workflow_start_guard
from tests.unit.test_proof_trade_ledger import (  # noqa: F401 - pytest fixture
    _allow_eligibility,
    identity_store,
)

LEAN_CODE = "import Mathlib\n\ntheorem legacy_fixture : 1 + 1 = 2 := by norm_num\n"


def _legacy_payload(proof_id: str, **overrides) -> dict:
    """A record as released builds stored it: no identity or trade evidence."""
    payload = {
        "proof_id": proof_id,
        "theorem_statement": f"1 + 1 = 2 ({proof_id})",
        "theorem_name": "legacy_fixture",
        "source_type": "paper",
        "source_id": "paper_legacy",
        "run_id": "",
        "lean_code": LEAN_CODE.replace("legacy_fixture", f"legacy_{proof_id}"),
        "novel": True,
        "novelty_tier": "novel_variant",
        "created_at": "2026-01-01T00:00:00",
        "dependency_extraction_status": "not_attempted",
    }
    payload.update(overrides)
    return payload


def _write_legacy_store(store: Path, payloads: list[dict], *, index_only: bool = False) -> None:
    store.mkdir(parents=True, exist_ok=True)
    (store / "proofs_index.json").write_text(
        json.dumps({"next_proof_id": len(payloads) + 1, "proofs": payloads}),
        encoding="utf-8",
    )
    if index_only:
        return
    for payload in payloads:
        (store / f"proof_{payload['proof_id']}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        (store / f"proof_{payload['proof_id']}_lean.lean").write_text(
            payload["lean_code"], encoding="utf-8"
        )


@pytest.fixture
def data_root(tmp_path, identity_store, monkeypatch):  # noqa: F811
    monkeypatch.setattr(system_config, "data_dir", str(tmp_path))
    monkeypatch.setattr(system_config, "auto_sessions_base_dir", str(tmp_path / "auto_sessions"))
    monkeypatch.setattr(system_config, "generic_mode", False)
    return tmp_path


def _migrate(dry_run: bool = False) -> dict:
    return asyncio.run(migrate_legacy_proofs(dry_run=dry_run))


def _diagnostics(root: Path) -> list:
    return list(project_installation_proof_stores(root).diagnostics)


def test_dry_run_counts_without_writing(data_root):
    store = data_root / "manual_proofs"
    _write_legacy_store(store, [_legacy_payload("proof_001")])
    before = {path.name: path.read_bytes() for path in store.iterdir()}

    result = _migrate(dry_run=True)

    assert result["eligible_count"] == 1
    assert result["migrated_count"] == 0
    assert result["upload_enabled"] is False and result["server_side_claim"] is False
    assert {path.name: path.read_bytes() for path in store.iterdir()} == before


def test_migration_makes_legacy_proof_selectable_without_receipts(data_root):
    store = data_root / "manual_proofs"
    _write_legacy_store(store, [_legacy_payload("proof_001")])

    result = _migrate()

    assert result["migrated_count"] == 1
    record = ProofRecord(**json.loads((store / "proof_proof_001.json").read_text()))
    assert record.client_proof_id and record.canonical_locator is not None
    assert record.canonical_locator.store_kind == "manual"
    assert record.run_id == "legacy:paper:paper_legacy"
    assert record.compile_receipt is None
    assert record.verification_evidence_status == "legacy_unrecorded"
    assert record.legacy_port_receipt.source_local_rank == "novel_variant"
    assert (store / "proof_proof_001_lean.lean").read_text() == record.lean_code

    [diagnostic] = _diagnostics(data_root)
    assert diagnostic.evidence_class == "legacy_port"
    assert diagnostic.evidence_complete is True
    assert diagnostic.source_selected is True
    assert diagnostic.eligible_to_offer is False
    assert "compile_receipt_missing" not in diagnostic.reason_codes
    assert "novelty_receipt_missing" not in diagnostic.reason_codes


def test_migration_is_idempotent(data_root):
    _write_legacy_store(data_root / "manual_proofs", [_legacy_payload("proof_001")])
    _migrate()

    again = _migrate()

    assert again["migrated_count"] == 0
    assert again["already_qualified_count"] == 1


def test_skips_and_exclusions_are_reason_coded(data_root):
    _write_legacy_store(
        data_root / "manual_proofs",
        [
            _legacy_payload("proof_001", novelty_tier="not_novel", novel=False),
            _legacy_payload("proof_002", novelty_tier="legacy_unknown_tier"),
            _legacy_payload("proof_003", verification_evidence_status="incomplete"),
        ],
    )

    result = _migrate()

    assert result["migrated_count"] == 1
    assert result["skipped"] == {
        "legacy_port_not_legacy_record": 1,
        "legacy_port_rank_unknown": 1,
    }
    migrated = [item for item in _diagnostics(data_root) if item.evidence_class == "legacy_port"]
    assert len(migrated) == 1
    assert "source_local_rank_not_novel" in migrated[0].reason_codes
    assert migrated[0].source_selected is False


def test_tampered_or_conflicting_receipt_fails_closed(data_root):
    store = data_root / "manual_proofs"
    _write_legacy_store(store, [_legacy_payload("proof_001"), _legacy_payload("proof_002")])
    _migrate()
    first = json.loads((store / "proof_proof_001.json").read_text())
    first["legacy_port_receipt"]["source_local_rank"] = "major_mathematical_discovery"
    first["legacy_port_receipt"]["receipt_digest"] = legacy_port_receipt_digest(
        first["legacy_port_receipt"]
    )
    (store / "proof_proof_001.json").write_text(json.dumps(first))
    second = json.loads((store / "proof_proof_002.json").read_text())
    second["verification_evidence_status"] = "incomplete"
    (store / "proof_proof_002.json").write_text(json.dumps(second))

    by_reason = {
        reason
        for item in _diagnostics(data_root)
        for reason in item.reason_codes
        if reason.startswith("legacy_port")
    }

    assert by_reason == {"legacy_port_receipt_invalid", "legacy_port_receipt_conflict"}


def test_index_only_store_keeps_skipped_records(data_root):
    store = data_root / "manual_proofs"
    _write_legacy_store(
        store,
        [
            _legacy_payload("proof_001"),
            _legacy_payload("proof_002", novelty_tier="legacy_unknown_tier"),
        ],
        index_only=True,
    )

    _migrate()

    assert (store / "proof_proof_001.json").exists()
    assert (store / "proof_proof_002.json").exists()

    async def reload_ids():
        database = ProofDatabase()
        database.set_base_dir(store)
        await database.initialize()
        return sorted(proof.proof_id for proof in await database.get_all_proofs())

    assert asyncio.run(reload_ids()) == ["proof_001", "proof_002"]


def test_session_and_archive_scope_follow_new_occurrence_rules(data_root):
    _write_legacy_store(
        data_root / "auto_sessions" / "session_alpha" / "proofs",
        [_legacy_payload("proof_001")],
    )
    _write_legacy_store(
        data_root / "manual_proof_runs" / "manual-run-1" / "proofs",
        [_legacy_payload("proof_002")],
    )

    _migrate()

    session_record = ProofRecord(
        **json.loads(
            (data_root / "auto_sessions" / "session_alpha" / "proofs" / "proof_proof_001.json").read_text()
        )
    )
    archive_record = ProofRecord(
        **json.loads(
            (data_root / "manual_proof_runs" / "manual-run-1" / "proofs" / "proof_proof_002.json").read_text()
        )
    )
    assert session_record.canonical_locator.store_kind == "autonomous"
    assert session_record.canonical_locator.session_id == "session_alpha"
    assert archive_record.canonical_locator.archive_id == "manual-run-1"
    assert all(item.evidence_class == "legacy_port" for item in _diagnostics(data_root))


def test_refused_while_a_workflow_is_active(data_root):
    _write_legacy_store(data_root / "manual_proofs", [_legacy_payload("proof_001")])
    lease = workflow_start_guard.commit("autonomous")
    try:
        with pytest.raises(LegacyPortUnavailable) as exc_info:
            _migrate()
    finally:
        workflow_start_guard.release(lease)
    assert exc_info.value.reason_code == "legacy_port_workflow_active"


def test_refused_in_generic_mode(data_root, monkeypatch):
    monkeypatch.setattr(system_config, "generic_mode", True)
    with pytest.raises(LegacyPortUnavailable) as exc_info:
        _migrate(dry_run=True)
    assert exc_info.value.reason_code == "legacy_port_generic_mode_unavailable"


def test_binding_failure_writes_nothing(data_root, monkeypatch):
    store = data_root / "manual_proofs"
    _write_legacy_store(store, [_legacy_payload("proof_001")])
    before = (store / "proof_proof_001.json").read_bytes()

    def refuse(*_args, **_kwargs):
        raise InstallationIdentityError("copied root")

    monkeypatch.setattr(
        "backend.autonomous.memory.proof_database.ensure_installation_binding", refuse
    )
    with pytest.raises(LegacyPortUnavailable) as exc_info:
        _migrate()

    assert exc_info.value.reason_code == "legacy_port_installation_identity_unavailable"
    assert (store / "proof_proof_001.json").read_bytes() == before


def test_migrated_proof_can_be_selected_once_eligible(data_root, monkeypatch):
    _write_legacy_store(data_root / "manual_proofs", [_legacy_payload("proof_001")])
    _migrate()
    _allow_eligibility(monkeypatch)
    [diagnostic] = _diagnostics(data_root)

    summary = create_selection_snapshot([diagnostic.client_proof_id], data_root=data_root)

    assert summary["state"] == "selected"
    assert summary["entry_count"] == 1


def test_migration_result_is_path_free(data_root):
    _write_legacy_store(
        data_root / "auto_sessions" / "secret_prompt_words" / "proofs",
        [_legacy_payload("proof_001")],
    )
    serialized = json.dumps(_migrate())
    assert "secret_prompt_words" not in serialized
    assert str(data_root) not in serialized
    assert legacy_module.LEGACY_PORT_MIGRATION_VERSION in serialized


def test_route_reports_counts_and_reason_coded_refusal(data_root):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from backend.api.routes import syntheticlib4
    from tests.unit.syntheticlib_gate_cases import syntheticlib_lifecycle

    _write_legacy_store(data_root / "manual_proofs", [_legacy_payload("proof_001")])
    app = FastAPI()
    app.include_router(syntheticlib4.router)
    client = TestClient(app)
    url = "/api/syntheticlib4/developer/legacy-port/migrate"
    with syntheticlib_lifecycle():
        preview = client.post(url, json={"confirm": True, "dry_run": True})
        lease = workflow_start_guard.commit("aggregator")
        try:
            refused = client.post(url, json={"confirm": True, "dry_run": False})
        finally:
            workflow_start_guard.release(lease)
        migrated = client.post(url, json={"confirm": True, "dry_run": False})
        malformed = client.post(url, json={"dry_run": False})

    assert preview.status_code == 200
    assert preview.json()["eligible_count"] == 1
    assert refused.status_code == 409
    assert refused.json()["detail"] == "Legacy proof migration refused: legacy_port_workflow_active"
    assert migrated.status_code == 200
    assert migrated.json()["migrated_count"] == 1
    assert migrated.json()["stores"][0]["store_role"] == "active_manual"
    assert malformed.status_code == 422
