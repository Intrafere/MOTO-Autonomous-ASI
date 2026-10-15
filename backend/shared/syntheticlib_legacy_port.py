"""Explicit one-time legacy-proof migration for SyntheticLib porting.

Proofs stored by released MOTO builds predate client identity, compile
receipts, and local novelty receipts. SyntheticLib ports those proofs without
a compile receipt, so this migration mints client proof IDs and locators and
marks each with a digest-bound legacy-port receipt. New proofs are unaffected
and still require full proof-time evidence.

The migration is explicit, desktop-only, refused while any workflow or manual
proof run is active, and metadata-only in what it reports: stores are named by
privacy-safe keys, never paths, and nothing is uploaded or claimed server-side.
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from backend.shared.config import system_config
from backend.shared.installation_identity import InstallationIdentityError
from backend.shared.syntheticlib_source_selection import CanonicalStoreRole
from backend.shared.syntheticlib_trade_eligibility import (
    canonical_store_key,
    enumerate_canonical_store_roots,
)
from backend.shared.trade_data_boundary import assert_trade_safe

LEGACY_PORT_MIGRATION_VERSION = "moto-legacy-port-migration-v1"

# Locator store kind for each canonical store role.
_ROLE_STORE_KINDS: dict[str, str] = {
    "active_manual": "manual",
    "archived_manual": "manual",
    "autonomous_session": "autonomous",
    "autonomous_legacy": "autonomous_legacy",
}


class LegacyPortUnavailable(RuntimeError):
    """Migration refused; carries a reason code only."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


async def _workflow_active() -> bool:
    from backend.autonomous.core.proof_run_manager import proof_run_manager
    from backend.shared.workflow_start_guard import workflow_start_guard

    return bool(workflow_start_guard.active_owner) or await proof_run_manager.has_active_runs()


def _locator_scope(role: CanonicalStoreRole, path: Path) -> tuple[str, str]:
    """Return (session_id, archive_id) the same way new occurrences record them."""
    if role == "autonomous_session":
        return path.parent.name, ""
    if role == "archived_manual":
        return "", path.parent.name
    return "", ""


_RECORD_FILE_RE = re.compile(r"^proof_(.+)\.json$")


def _dry_run_store_sync(path: Path) -> dict[str, Any]:
    """Count migration candidates from canonical files without loading the store.

    Loading through ProofDatabase may persist index metadata, so a dry run reads
    record files (or, for index-only legacy stores, the embedded index) as-is.
    """
    from backend.autonomous.memory.proof_database import ProofDatabase
    from backend.shared.models import ProofRecord

    payloads: list[Any] = []
    record_files = [
        child
        for child in sorted(path.iterdir())
        if _RECORD_FILE_RE.match(child.name) and child.is_file() and not child.is_symlink()
    ]
    if record_files:
        for child in record_files:
            try:
                payloads.append(json.loads(child.read_text(encoding="utf-8")))
            except (OSError, UnicodeError, json.JSONDecodeError):
                payloads.append(None)
    else:
        index = path / "proofs_index.json"
        try:
            payloads = list(json.loads(index.read_text(encoding="utf-8")).get("proofs", []))
        except (OSError, UnicodeError, json.JSONDecodeError, AttributeError):
            payloads = []
    result: dict[str, Any] = {
        "eligible_count": 0,
        "migrated_count": 0,
        "already_qualified_count": 0,
        "skipped": {},
    }
    for payload in payloads:
        try:
            record = ProofRecord(**payload)
        except Exception:
            reason = "legacy_port_record_invalid"
        else:
            reason = ProofDatabase._legacy_port_skip_reason(record)  # noqa: SLF001
        if not reason:
            result["eligible_count"] += 1
        elif reason in {"legacy_port_already_migrated", "legacy_port_already_qualified"}:
            result["already_qualified_count"] += 1
        else:
            result["skipped"][reason] = result["skipped"].get(reason, 0) + 1
    return result


def _database_for(path: Path):
    """Use the live singleton for its own store so its lock and cache stay coherent."""
    from backend.autonomous.memory.proof_database import (
        ProofDatabase,
        manual_proof_database,
        proof_database,
    )

    resolved = path.resolve(strict=False)
    for singleton in (proof_database, manual_proof_database):
        if singleton._base_dir.resolve(strict=False) == resolved:  # noqa: SLF001
            return singleton, True
    database = ProofDatabase()
    database.set_base_dir(path)
    return database, False


async def migrate_legacy_proofs(*, dry_run: bool) -> dict[str, Any]:
    """Mint port identity for released-build proofs across every canonical store."""
    if system_config.generic_mode:
        raise LegacyPortUnavailable("legacy_port_generic_mode_unavailable")
    if await _workflow_active():
        raise LegacyPortUnavailable("legacy_port_workflow_active")

    data_root = Path(system_config.data_dir)
    sessions_root = (
        Path(system_config.auto_sessions_base_dir)
        if system_config.auto_sessions_base_dir
        else data_root / "auto_sessions"
    )
    enumerated, container_reasons = enumerate_canonical_store_roots(data_root, sessions_root)

    stores: list[dict[str, Any]] = []
    totals = {"eligible_count": 0, "migrated_count": 0, "already_qualified_count": 0}
    skipped_totals: dict[str, int] = {}
    for role, label, path, unsafe_parent in enumerated:
        store_key = canonical_store_key(role, label)
        if unsafe_parent or path.is_symlink() or not path.is_dir():
            stores.append(
                {
                    "store_key": store_key,
                    "store_role": role,
                    "eligible_count": 0,
                    "migrated_count": 0,
                    "already_qualified_count": 0,
                    "skipped": {"legacy_port_store_not_regular": 1},
                }
            )
            skipped_totals["legacy_port_store_not_regular"] = (
                skipped_totals.get("legacy_port_store_not_regular", 0) + 1
            )
            continue
        if dry_run:
            result = await asyncio.to_thread(_dry_run_store_sync, path)
            stores.append({"store_key": store_key, "store_role": role, **result})
            for key in totals:
                totals[key] += int(result.get(key, 0))
            for reason, count in result["skipped"].items():
                skipped_totals[reason] = skipped_totals.get(reason, 0) + int(count)
            continue
        session_id, archive_id = _locator_scope(role, path)
        database, is_singleton = _database_for(path)
        try:
            result = await database.migrate_legacy_identity(
                store_kind=_ROLE_STORE_KINDS[role],
                session_id=session_id,
                archive_id=archive_id,
            )
        except InstallationIdentityError:
            raise LegacyPortUnavailable("legacy_port_installation_identity_unavailable") from None
        except RuntimeError:
            # Corrupt or conflicting store identity: report it and continue.
            result = {
                "eligible_count": 0,
                "migrated_count": 0,
                "already_qualified_count": 0,
                "skipped": {"legacy_port_store_unreadable": 1},
            }
        if not is_singleton:
            database._index_data = None  # noqa: SLF001 - drop the temporary cache
        stores.append({"store_key": store_key, "store_role": role, **result})
        for key in totals:
            totals[key] += int(result.get(key, 0))
        for reason, count in result.get("skipped", {}).items():
            skipped_totals[reason] = skipped_totals.get(reason, 0) + int(count)

    if not dry_run:
        # Any other cached view of these stores must reload from disk.
        from backend.autonomous.memory.proof_database import (
            manual_proof_database,
            proof_database,
        )

        for singleton in (proof_database, manual_proof_database):
            async with singleton._lock:  # noqa: SLF001
                singleton._index_data = None  # noqa: SLF001

    payload = {
        "migration_version": LEGACY_PORT_MIGRATION_VERSION,
        "dry_run": dry_run,
        **totals,
        "skipped": dict(sorted(skipped_totals.items())),
        "reason_codes": sorted(set(container_reasons)),
        "stores": stores,
        "upload_enabled": False,
        "server_side_claim": False,
    }
    assert_trade_safe(payload)
    return payload
