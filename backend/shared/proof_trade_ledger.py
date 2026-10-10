"""Crash-safe, metadata-only local proof-trade ledger.

The ledger records local proof-trade intent under the active data root. It
references canonical proof occurrences only through opaque client proof IDs,
privacy-safe store/record keys, and canonical hashes from the installation
evidence projection; it never stores proof bodies, prompts, or paths.

Guarantees:

* Each mutation is a serialized read-validate-mutate-replace cycle written with
  ``atomic_write_json``. Callers on the event loop must use ``asyncio.to_thread``
  so cancellation of the awaiting task can never tear a write.
* Corrupt, unsupported, non-regular, or root-mismatched ledgers fail closed and
  are never overwritten or repaired; only an explicit archive/reset recovers.
* Selection snapshots bind to canonical hashes, store/record identity, and the
  proof-set revision. Reconciliation fails closed (terminal ``invalidated``)
  when a selected proof is deleted, moved, replaced, hash-changed, loses
  evidence, or loses source selection.
* Contract-owned fields (account binding, contract version, idempotency key,
  request digest, server receipt, status cursor, retry state) are reserved and
  must remain ``null`` until the frozen SyntheticLib contract is ingested.
* Nothing here uploads, authorizes, or contacts SyntheticLib, and batches can
  only be prepared from proofs the canonical projection marks eligible, which
  is impossible until the rank mapping and frozen contract exist.

This ledger is deliberately separate from proof-run managers, LeanOJ resume
state, proof search, and usage attestations.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from backend.shared.atomic_json import atomic_write_json
from backend.shared.config import system_config
from backend.shared.installation_identity import (
    InstallationBinding,
    InstallationBindingMismatchError,
    InstallationIdentityError,
    ensure_installation_binding,
    read_installation_binding,
)
from backend.shared.proof_locator import (
    is_valid_client_batch_id,
    is_valid_client_proof_id,
    is_valid_selection_snapshot_id,
    new_client_batch_id,
    new_selection_snapshot_id,
)
from backend.shared.syntheticlib_trade_compatibility import canonical_json_bytes
from backend.shared.syntheticlib_trade_eligibility import (
    InstallationTradeEvidenceProjection,
    TradeEvidenceDiagnostic,
    project_installation_proof_stores,
)

LEDGER_SCHEMA_VERSION = "moto-proof-trade-ledger-v1"
LEDGER_DIRNAME = "syntheticlib_trade"
LEDGER_FILENAME = "proof_trade_ledger.json"
LEDGER_ARCHIVE_DIRNAME = "archive"
_ARCHIVED_LEDGER_PREFIX = "proof_trade_ledger."
MAX_LEDGER_EVENTS = 200

_HEX64 = r"^[0-9a-f]{64}$"
_KEY20 = r"^[0-9a-f]{20}$"
_REVISION_RACE_REASON = "proof_set_revision_changed_during_projection"
_LEDGER_LOCK = threading.Lock()

SnapshotState = Literal["selected", "invalidated", "cancelled"]
BatchState = Literal["prepared", "cancelled", "invalidated"]
LedgerEventKind = Literal[
    "snapshot_created",
    "snapshot_reconciled",
    "snapshot_invalidated",
    "snapshot_cancelled",
    "batch_prepared",
    "batch_cancelled",
    "batch_invalidated",
]

_T = TypeVar("_T")


class ProofTradeLedgerUnavailable(RuntimeError):
    """The ledger cannot be used safely; ``reason_code`` explains why."""

    def __init__(self, reason_code: str):
        self.reason_code = reason_code
        super().__init__(f"Proof-trade ledger unavailable: {reason_code}")


class ProofTradeLedgerRejected(ValueError):
    """A requested ledger operation was refused; nothing was written."""

    def __init__(self, reason_codes: Sequence[str]):
        self.reason_codes = tuple(sorted(set(reason_codes)))
        super().__init__("Proof-trade ledger request rejected: " + ", ".join(self.reason_codes))


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SnapshotEntry(_Strict):
    client_proof_id: str
    canonical_locator_digest: str = Field(pattern=_HEX64)
    canonical_theorem_statement_hash: str = Field(pattern=_HEX64)
    canonical_lean_code_hash: str = Field(pattern=_HEX64)
    store_key: str = Field(pattern=_KEY20)
    record_key: str = Field(pattern=_KEY20)
    stable_store_id: str = Field(min_length=1, max_length=128)
    proof_set_revision: int = Field(ge=0)

    @model_validator(mode="after")
    def _valid_client_proof_id(self) -> "SnapshotEntry":
        if not is_valid_client_proof_id(self.client_proof_id):
            raise ValueError("invalid client proof id")
        return self


class ReconciliationRecord(_Strict):
    at: datetime
    outcome: Literal["ok", "failed"]
    revision_advanced: bool = False
    reason_codes: list[str] = Field(default_factory=list)


class SelectionSnapshot(_Strict):
    snapshot_id: str
    state: SnapshotState
    created_at: datetime
    updated_at: datetime
    projection_schema_version: str = Field(min_length=1)
    source_selection_policy_version: str = Field(min_length=1)
    installation_projection_fingerprint: str = Field(pattern=_HEX64)
    selection_digest: str = Field(pattern=_HEX64)
    entries: list[SnapshotEntry] = Field(min_length=1)
    last_reconciliation: Optional[ReconciliationRecord] = None

    @model_validator(mode="after")
    def _consistent(self) -> "SelectionSnapshot":
        if not is_valid_selection_snapshot_id(self.snapshot_id):
            raise ValueError("invalid selection snapshot id")
        client_ids = [entry.client_proof_id for entry in self.entries]
        if len(client_ids) != len(set(client_ids)):
            raise ValueError("duplicate selection entry")
        if self.selection_digest != selection_digest(self.entries):
            raise ValueError("selection digest mismatch")
        return self


class BatchEntry(_Strict):
    client_batch_id: str
    snapshot_id: str
    state: BatchState
    created_at: datetime
    updated_at: datetime
    # Reserved for the frozen SyntheticLib contract (Builds 8-10); null in v1.
    idempotency_key: None = None
    request_digest: None = None
    contract_version: None = None
    server_receipt: None = None
    status_cursor: None = None
    retry_state: None = None

    @model_validator(mode="after")
    def _valid_ids(self) -> "BatchEntry":
        if not is_valid_client_batch_id(self.client_batch_id):
            raise ValueError("invalid client batch id")
        if not is_valid_selection_snapshot_id(self.snapshot_id):
            raise ValueError("invalid selection snapshot id")
        return self


class LedgerEvent(_Strict):
    at: datetime
    kind: LedgerEventKind
    subject_id: str
    reason_codes: list[str] = Field(default_factory=list)


class ProofTradeLedger(_Strict):
    schema_version: Literal["moto-proof-trade-ledger-v1"]
    installation_id: str = Field(min_length=1)
    root_binding_id: str = Field(min_length=1)
    ledger_generation: int = Field(ge=0)
    created_at: datetime
    updated_at: datetime
    # Reserved for desktop authorization and the frozen contract; null in v1.
    account_binding: None = None
    contract_version: None = None
    selection_snapshots: dict[str, SelectionSnapshot] = Field(default_factory=dict)
    batches: dict[str, BatchEntry] = Field(default_factory=dict)
    events: list[LedgerEvent] = Field(default_factory=list, max_length=MAX_LEDGER_EVENTS)

    @model_validator(mode="after")
    def _consistent(self) -> "ProofTradeLedger":
        for key, snapshot in self.selection_snapshots.items():
            if key != snapshot.snapshot_id:
                raise ValueError("snapshot key mismatch")
        for key, batch in self.batches.items():
            if key != batch.client_batch_id:
                raise ValueError("batch key mismatch")
            if batch.snapshot_id not in self.selection_snapshots:
                raise ValueError("batch references unknown snapshot")
        return self


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _data_root(data_root: Optional[Path | str]) -> Path:
    return Path(data_root or system_config.data_dir).resolve(strict=False)


def ledger_path(data_root: Optional[Path | str] = None) -> Path:
    """Return the ledger location under the active (or given) data root."""
    return _data_root(data_root) / LEDGER_DIRNAME / LEDGER_FILENAME


def _read_payload(path: Path) -> Optional[dict[str, Any]]:
    directory = path.parent
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        raise ProofTradeLedgerUnavailable("ledger_not_regular")
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ProofTradeLedgerUnavailable("ledger_not_regular")
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise ProofTradeLedgerUnavailable("ledger_corrupt") from None
    if not isinstance(payload, dict):
        raise ProofTradeLedgerUnavailable("ledger_corrupt")
    if payload.get("schema_version") != LEDGER_SCHEMA_VERSION:
        raise ProofTradeLedgerUnavailable("ledger_schema_unsupported")
    return payload


def _parse(payload: dict[str, Any]) -> ProofTradeLedger:
    try:
        return ProofTradeLedger.model_validate(payload)
    except ValidationError:
        raise ProofTradeLedgerUnavailable("ledger_corrupt") from None


def _require_binding_match(ledger: ProofTradeLedger, binding: Optional[InstallationBinding]) -> None:
    if (
        binding is None
        or ledger.installation_id != binding.installation_id
        or ledger.root_binding_id != binding.root_binding_id
    ):
        raise ProofTradeLedgerUnavailable("ledger_installation_binding_mismatch")


def _binding_for_mutation(root: Path) -> InstallationBinding:
    if system_config.generic_mode:
        raise ProofTradeLedgerUnavailable("ledger_generic_mode_unavailable")
    try:
        return ensure_installation_binding(root)
    except InstallationBindingMismatchError:
        raise ProofTradeLedgerUnavailable("ledger_installation_binding_mismatch") from None
    except InstallationIdentityError:
        raise ProofTradeLedgerUnavailable("ledger_installation_identity_unavailable") from None


def _append_event(
    ledger: ProofTradeLedger,
    kind: LedgerEventKind,
    subject_id: str,
    reason_codes: Sequence[str] = (),
) -> None:
    ledger.events.append(
        LedgerEvent(at=_now(), kind=kind, subject_id=subject_id, reason_codes=list(reason_codes))
    )
    del ledger.events[:-MAX_LEDGER_EVENTS]


def _mutate(
    data_root: Optional[Path | str],
    mutator: Callable[[ProofTradeLedger, Path], tuple[bool, _T]],
) -> _T:
    """Serialize one read-validate-mutate-replace cycle; write only on change."""
    with _LEDGER_LOCK:
        root = _data_root(data_root)
        binding = _binding_for_mutation(root)
        path = ledger_path(root)
        payload = _read_payload(path)
        if payload is None:
            timestamp = _now()
            ledger = ProofTradeLedger(
                schema_version=LEDGER_SCHEMA_VERSION,
                installation_id=binding.installation_id,
                root_binding_id=binding.root_binding_id,
                ledger_generation=0,
                created_at=timestamp,
                updated_at=timestamp,
            )
        else:
            ledger = _parse(payload)
            _require_binding_match(ledger, binding)
        working = ledger.model_copy(deep=True)
        changed, result = mutator(working, root)
        if changed:
            working.ledger_generation = ledger.ledger_generation + 1
            working.updated_at = _now()
            validated = ProofTradeLedger.model_validate(working.model_dump())
            atomic_write_json(path, validated.model_dump(mode="json"))
        return result


def _project(root: Path) -> InstallationTradeEvidenceProjection:
    sessions_root = system_config.auto_sessions_base_dir
    use_configured_sessions = (
        sessions_root
        and Path(system_config.data_dir).resolve(strict=False) == root
    )
    return project_installation_proof_stores(
        root,
        auto_sessions_root=Path(sessions_root) if use_configured_sessions else None,
    )


def _index_diagnostics(
    projection: InstallationTradeEvidenceProjection,
) -> dict[str, list[TradeEvidenceDiagnostic]]:
    indexed: dict[str, list[TradeEvidenceDiagnostic]] = {}
    for diagnostic in projection.diagnostics:
        if diagnostic.client_proof_id:
            indexed.setdefault(diagnostic.client_proof_id, []).append(diagnostic)
    return indexed


def selection_digest(entries: Sequence[SnapshotEntry]) -> str:
    """Canonical digest over the ordered metadata-only selection entries."""
    payload = [entry.model_dump(mode="json") for entry in entries]
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _resolve_selection(
    projection: InstallationTradeEvidenceProjection,
    client_proof_ids: Sequence[str],
) -> list[SnapshotEntry]:
    indexed = _index_diagnostics(projection)
    stable_ids = {store.store_key: store.stable_store_id for store in projection.stores}
    reasons: set[str] = set()
    entries: list[SnapshotEntry] = []
    for client_proof_id in sorted(client_proof_ids):
        matches = indexed.get(client_proof_id, [])
        if not matches:
            reasons.add("selection_proof_not_found")
            continue
        if len(matches) > 1:
            reasons.add("selection_proof_ambiguous")
            continue
        diagnostic = matches[0]
        if not diagnostic.evidence_complete:
            reasons.add("selection_evidence_incomplete")
        if not diagnostic.source_selected:
            reasons.add("selection_source_not_selected")
        stable_store_id = stable_ids.get(diagnostic.store_key, "")
        if not stable_store_id or diagnostic.proof_set_revision is None:
            reasons.add("selection_store_identity_incomplete")
            continue
        try:
            entries.append(
                SnapshotEntry(
                    client_proof_id=client_proof_id,
                    canonical_locator_digest=diagnostic.canonical_locator_digest,
                    canonical_theorem_statement_hash=diagnostic.canonical_theorem_statement_hash,
                    canonical_lean_code_hash=diagnostic.canonical_lean_code_hash,
                    store_key=diagnostic.store_key,
                    record_key=diagnostic.record_key,
                    stable_store_id=stable_store_id,
                    proof_set_revision=diagnostic.proof_set_revision,
                )
            )
        except ValidationError:
            reasons.add("selection_evidence_incomplete")
    if reasons:
        raise ProofTradeLedgerRejected(reasons)
    return entries


def _reconcile_entries(
    projection: InstallationTradeEvidenceProjection,
    entries: Sequence[SnapshotEntry],
) -> tuple[list[str], list[TradeEvidenceDiagnostic]]:
    """Return (failure reason codes, current diagnostics in entry order)."""
    indexed = _index_diagnostics(projection)
    stable_ids = {store.store_key: store.stable_store_id for store in projection.stores}
    reasons: set[str] = set()
    current: list[TradeEvidenceDiagnostic] = []
    for entry in entries:
        matches = indexed.get(entry.client_proof_id, [])
        if not matches:
            reasons.add("reconcile_proof_missing")
            continue
        if len(matches) > 1:
            reasons.add("reconcile_proof_ambiguous")
            continue
        diagnostic = matches[0]
        current.append(diagnostic)
        if (
            diagnostic.store_key != entry.store_key
            or diagnostic.record_key != entry.record_key
            or stable_ids.get(diagnostic.store_key, "") != entry.stable_store_id
        ):
            reasons.add("reconcile_proof_moved")
        if (
            diagnostic.canonical_locator_digest != entry.canonical_locator_digest
            or diagnostic.canonical_theorem_statement_hash != entry.canonical_theorem_statement_hash
            or diagnostic.canonical_lean_code_hash != entry.canonical_lean_code_hash
        ):
            reasons.add("reconcile_hash_changed")
        if _REVISION_RACE_REASON in diagnostic.reason_codes:
            reasons.add("reconcile_projection_unstable")
        elif not diagnostic.evidence_complete:
            reasons.add("reconcile_evidence_incomplete")
        if not diagnostic.source_selected:
            reasons.add("reconcile_source_deselected")
        if diagnostic.proof_set_revision is None:
            reasons.add("reconcile_projection_unstable")
    return sorted(reasons), current


def _active_snapshot(ledger: ProofTradeLedger, snapshot_id: str) -> SelectionSnapshot:
    snapshot = ledger.selection_snapshots.get(snapshot_id)
    if snapshot is None:
        raise ProofTradeLedgerRejected(["snapshot_unknown"])
    if snapshot.state != "selected":
        raise ProofTradeLedgerRejected(["snapshot_not_active"])
    return snapshot


def _reconcile_in_place(
    ledger: ProofTradeLedger,
    snapshot: SelectionSnapshot,
    root: Path,
) -> tuple[list[str], list[TradeEvidenceDiagnostic]]:
    """Reconcile and record the outcome; failures are terminal for the snapshot."""
    reasons, current = _reconcile_entries(_project(root), snapshot.entries)
    timestamp = _now()
    snapshot.updated_at = timestamp
    if reasons:
        snapshot.state = "invalidated"
        snapshot.last_reconciliation = ReconciliationRecord(
            at=timestamp, outcome="failed", reason_codes=reasons
        )
        _append_event(ledger, "snapshot_invalidated", snapshot.snapshot_id, reasons)
        for batch in ledger.batches.values():
            if batch.snapshot_id == snapshot.snapshot_id and batch.state == "prepared":
                batch.state = "invalidated"
                batch.updated_at = timestamp
                _append_event(ledger, "batch_invalidated", batch.client_batch_id, reasons)
        return reasons, current
    # Entries stay bound to the selection-time revision (the digest covers them);
    # an advanced revision is acceptable only because every hash still matches.
    snapshot.last_reconciliation = ReconciliationRecord(
        at=timestamp,
        outcome="ok",
        revision_advanced=any(
            diagnostic.proof_set_revision != entry.proof_set_revision
            for entry, diagnostic in zip(snapshot.entries, current)
        ),
    )
    _append_event(ledger, "snapshot_reconciled", snapshot.snapshot_id)
    return [], current


def _snapshot_summary(snapshot: SelectionSnapshot) -> dict[str, Any]:
    reconciliation = snapshot.last_reconciliation
    return {
        "snapshot_id": snapshot.snapshot_id,
        "state": snapshot.state,
        "entry_count": len(snapshot.entries),
        "selection_digest": snapshot.selection_digest,
        "reconciliation_outcome": reconciliation.outcome if reconciliation else None,
        "reason_codes": list(reconciliation.reason_codes) if reconciliation else [],
    }


def create_selection_snapshot(
    client_proof_ids: Sequence[str],
    *,
    data_root: Optional[Path | str] = None,
) -> dict[str, Any]:
    """Bind explicitly selected canonical proofs to a metadata-only snapshot.

    Re-selecting the same proof set while a selected snapshot exists returns it
    unchanged. Rejected selections write nothing.
    """
    requested = [str(value or "") for value in client_proof_ids]
    if not requested:
        raise ProofTradeLedgerRejected(["selection_empty"])
    if any(not is_valid_client_proof_id(value) for value in requested):
        raise ProofTradeLedgerRejected(["selection_malformed_id"])
    if len(set(requested)) != len(requested):
        raise ProofTradeLedgerRejected(["selection_duplicate_id"])

    def mutate(ledger: ProofTradeLedger, root: Path) -> tuple[bool, dict[str, Any]]:
        wanted = set(requested)
        for snapshot in ledger.selection_snapshots.values():
            if snapshot.state == "selected" and {
                entry.client_proof_id for entry in snapshot.entries
            } == wanted:
                return False, _snapshot_summary(snapshot)
        projection = _project(root)
        entries = _resolve_selection(projection, requested)
        timestamp = _now()
        snapshot = SelectionSnapshot(
            snapshot_id=new_selection_snapshot_id(),
            state="selected",
            created_at=timestamp,
            updated_at=timestamp,
            projection_schema_version=projection.schema_version,
            source_selection_policy_version=projection.source_selection_policy_version,
            installation_projection_fingerprint=projection.projection_fingerprint,
            selection_digest=selection_digest(entries),
            entries=entries,
        )
        ledger.selection_snapshots[snapshot.snapshot_id] = snapshot
        _append_event(ledger, "snapshot_created", snapshot.snapshot_id)
        return True, _snapshot_summary(snapshot)

    return _mutate(data_root, mutate)


def reconcile_selection_snapshot(
    snapshot_id: str,
    *,
    data_root: Optional[Path | str] = None,
) -> dict[str, Any]:
    """Re-check a selected snapshot against canonical stores; failures are terminal."""

    def mutate(ledger: ProofTradeLedger, root: Path) -> tuple[bool, dict[str, Any]]:
        snapshot = _active_snapshot(ledger, snapshot_id)
        _reconcile_in_place(ledger, snapshot, root)
        return True, _snapshot_summary(snapshot)

    return _mutate(data_root, mutate)


def prepare_batch(
    snapshot_id: str,
    *,
    data_root: Optional[Path | str] = None,
) -> dict[str, Any]:
    """Reconcile a snapshot and record a local prepared batch if every proof is eligible.

    Returns ``outcome`` of ``existing``, ``prepared``, ``invalidated``, or
    ``blocked``. A prepared batch holds no request payload; contract-owned
    fields stay null until the frozen contract and serializer exist.
    """

    def mutate(ledger: ProofTradeLedger, root: Path) -> tuple[bool, dict[str, Any]]:
        snapshot = _active_snapshot(ledger, snapshot_id)
        reasons, current = _reconcile_in_place(ledger, snapshot, root)
        if reasons:
            return True, {"outcome": "invalidated", "client_batch_id": None, "reason_codes": reasons}
        for batch in ledger.batches.values():
            if batch.snapshot_id == snapshot_id and batch.state == "prepared":
                return True, {
                    "outcome": "existing",
                    "client_batch_id": batch.client_batch_id,
                    "reason_codes": [],
                }
        if not all(diagnostic.eligible_to_offer for diagnostic in current):
            return True, {
                "outcome": "blocked",
                "client_batch_id": None,
                "reason_codes": ["batch_blocked_ineligible"],
            }
        timestamp = _now()
        batch = BatchEntry(
            client_batch_id=new_client_batch_id(),
            snapshot_id=snapshot_id,
            state="prepared",
            created_at=timestamp,
            updated_at=timestamp,
        )
        ledger.batches[batch.client_batch_id] = batch
        _append_event(ledger, "batch_prepared", batch.client_batch_id)
        return True, {"outcome": "prepared", "client_batch_id": batch.client_batch_id, "reason_codes": []}

    return _mutate(data_root, mutate)


def cancel_selection_snapshot(
    snapshot_id: str,
    *,
    data_root: Optional[Path | str] = None,
) -> dict[str, Any]:
    """Cancel a selected snapshot and its prepared batches before any upload."""

    def mutate(ledger: ProofTradeLedger, root: Path) -> tuple[bool, dict[str, Any]]:
        snapshot = _active_snapshot(ledger, snapshot_id)
        timestamp = _now()
        snapshot.state = "cancelled"
        snapshot.updated_at = timestamp
        _append_event(ledger, "snapshot_cancelled", snapshot_id)
        for batch in ledger.batches.values():
            if batch.snapshot_id == snapshot_id and batch.state == "prepared":
                batch.state = "cancelled"
                batch.updated_at = timestamp
                _append_event(ledger, "batch_cancelled", batch.client_batch_id)
        return True, _snapshot_summary(snapshot)

    return _mutate(data_root, mutate)


def cancel_batch(
    client_batch_id: str,
    *,
    data_root: Optional[Path | str] = None,
) -> dict[str, Any]:
    """Cancel one prepared batch before upload; nothing remains queued."""

    def mutate(ledger: ProofTradeLedger, root: Path) -> tuple[bool, dict[str, Any]]:
        batch = ledger.batches.get(client_batch_id)
        if batch is None:
            raise ProofTradeLedgerRejected(["batch_unknown"])
        if batch.state != "prepared":
            raise ProofTradeLedgerRejected(["batch_not_active"])
        batch.state = "cancelled"
        batch.updated_at = _now()
        _append_event(ledger, "batch_cancelled", client_batch_id)
        return True, {"client_batch_id": client_batch_id, "state": batch.state}

    return _mutate(data_root, mutate)


def archive_and_reset_local_ledger(
    data_root: Optional[Path | str] = None,
) -> dict[str, Any]:
    """Move the current ledger (even if corrupt) into the archive and start empty.

    This only clears local pending material; it never claims server-side deletion.
    """
    if system_config.generic_mode:
        raise ProofTradeLedgerUnavailable("ledger_generic_mode_unavailable")
    with _LEDGER_LOCK:
        path = ledger_path(data_root)
        if path.parent.is_symlink():
            raise ProofTradeLedgerUnavailable("ledger_not_regular")
        archived = False
        if path.exists() or path.is_symlink():
            archive_dir = path.parent / LEDGER_ARCHIVE_DIRNAME
            archive_dir.mkdir(parents=True, exist_ok=True)
            stamp = _now().strftime("%Y%m%dT%H%M%S%fZ")
            os.replace(path, archive_dir / f"{_ARCHIVED_LEDGER_PREFIX}{stamp}.json")
            archived = True
        return {"archived": archived, "server_side_deletion_claimed": False}


def _archived_ledger_entries(archive_dir: Path) -> list[os.DirEntry]:
    if not archive_dir.exists() or archive_dir.is_symlink() or not archive_dir.is_dir():
        return []
    with os.scandir(archive_dir) as entries:
        return [
            entry
            for entry in entries
            if entry.name.startswith(_ARCHIVED_LEDGER_PREFIX) and entry.name.endswith(".json")
        ]


def _count_archived_ledgers(root: Path) -> int:
    archive_dir = ledger_path(root).parent / LEDGER_ARCHIVE_DIRNAME
    if ledger_path(root).parent.is_symlink():
        return 0
    return sum(
        1
        for entry in _archived_ledger_entries(archive_dir)
        if entry.is_file(follow_symlinks=False)
    )


def delete_archived_local_ledgers(
    data_root: Optional[Path | str] = None,
) -> dict[str, Any]:
    """Permanently delete archived local ledgers after an explicit reset.

    Only regular archived ledger files are unlinked; symlinks and other entries
    are skipped and never followed. The current ledger is untouched, and no
    server-side deletion is claimed.
    """
    if system_config.generic_mode:
        raise ProofTradeLedgerUnavailable("ledger_generic_mode_unavailable")
    with _LEDGER_LOCK:
        ledger_dir = ledger_path(data_root).parent
        archive_dir = ledger_dir / LEDGER_ARCHIVE_DIRNAME
        if ledger_dir.is_symlink() or archive_dir.is_symlink():
            raise ProofTradeLedgerUnavailable("ledger_not_regular")
        deleted = 0
        skipped = 0
        for entry in _archived_ledger_entries(archive_dir):
            if not entry.is_file(follow_symlinks=False):
                skipped += 1
                continue
            os.unlink(entry.path)
            deleted += 1
        return {
            "deleted_count": deleted,
            "skipped_count": skipped,
            "server_side_deletion_claimed": False,
        }


def _read_for_status(root: Path) -> tuple[Optional[ProofTradeLedger], str]:
    if system_config.generic_mode:
        return None, "ledger_generic_mode_unavailable"
    try:
        payload = _read_payload(ledger_path(root))
        if payload is None:
            return None, ""
        ledger = _parse(payload)
        try:
            binding = read_installation_binding(root)
        except InstallationIdentityError:
            binding = None
        _require_binding_match(ledger, binding)
        return ledger, ""
    except ProofTradeLedgerUnavailable as exc:
        return None, exc.reason_code


def get_ledger_status(data_root: Optional[Path | str] = None) -> dict[str, Any]:
    """Return metadata-only ledger status without creating any identity or file."""
    root = _data_root(data_root)
    ledger, reason_code = _read_for_status(root)
    snapshot_counts = {state: 0 for state in ("selected", "invalidated", "cancelled")}
    batch_counts = {state: 0 for state in ("prepared", "cancelled", "invalidated")}
    if ledger is not None:
        for snapshot in ledger.selection_snapshots.values():
            snapshot_counts[snapshot.state] += 1
        for batch in ledger.batches.values():
            batch_counts[batch.state] += 1
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "available": not reason_code,
        "reason_code": reason_code,
        "exists": not system_config.generic_mode
        and (ledger_path(root).exists() or ledger_path(root).is_symlink()),
        "ledger_generation": ledger.ledger_generation if ledger is not None else None,
        "snapshot_counts": snapshot_counts,
        "batch_counts": batch_counts,
        "account_binding_present": bool(ledger is not None and ledger.account_binding is not None),
        "contract_version_present": bool(ledger is not None and ledger.contract_version is not None),
        "archived_ledger_count": 0 if system_config.generic_mode else _count_archived_ledgers(root),
        "upload_enabled": False,
    }


def list_ledger_entries(data_root: Optional[Path | str] = None) -> dict[str, Any]:
    """Return metadata-only snapshot/batch summaries without creating any identity or file.

    Snapshot entries, hashes, and paths are never included; callers get counts,
    digests, states, and reconciliation reason codes only.
    """
    root = _data_root(data_root)
    ledger, reason_code = _read_for_status(root)
    snapshots: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    if ledger is not None:
        for snapshot in sorted(
            ledger.selection_snapshots.values(),
            key=lambda item: (item.created_at, item.snapshot_id),
            reverse=True,
        ):
            reconciliation = snapshot.last_reconciliation
            snapshots.append(
                {
                    **_snapshot_summary(snapshot),
                    "created_at": snapshot.created_at.isoformat(),
                    "updated_at": snapshot.updated_at.isoformat(),
                    "reconciled_at": reconciliation.at.isoformat() if reconciliation else None,
                    "revision_advanced": bool(reconciliation and reconciliation.revision_advanced),
                }
            )
        for batch in sorted(
            ledger.batches.values(),
            key=lambda item: (item.created_at, item.client_batch_id),
            reverse=True,
        ):
            batches.append(
                {
                    "client_batch_id": batch.client_batch_id,
                    "snapshot_id": batch.snapshot_id,
                    "state": batch.state,
                    "created_at": batch.created_at.isoformat(),
                    "updated_at": batch.updated_at.isoformat(),
                }
            )
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "available": not reason_code,
        "reason_code": reason_code,
        "exists": not system_config.generic_mode
        and (ledger_path(root).exists() or ledger_path(root).is_symlink()),
        "ledger_generation": ledger.ledger_generation if ledger is not None else None,
        "snapshots": snapshots,
        "batches": batches,
        "upload_enabled": False,
    }


def ledger_blocks_installation_reset(data_root: Optional[Path | str] = None) -> bool:
    """Fail closed: linked, queued, or unreadable ledger state blocks identity reset."""
    root = _data_root(data_root)
    try:
        payload = _read_payload(ledger_path(root))
        if payload is None:
            return False
        ledger = _parse(payload)
    except ProofTradeLedgerUnavailable:
        return True
    return ledger.account_binding is not None or any(
        batch.state == "prepared" for batch in ledger.batches.values()
    )
